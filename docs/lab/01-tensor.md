# Lab 01 算子层：同一数学的三种实现

!!! info "本章对应"
    - 生产组件：FlashMLA / DeepGEMM / DeepEP / TileLang
    - miniflow 代码：`miniflow/tensor.py`、`csrc/kernel.cpp`、`setup.py`
    - 前置阅读：[04 内核层](../guide/04-kernels.md)

guide 04 讲了算子层决定性能上限。本章动手验证三个最根本的算子层事实：**循环序决定缓存局部性**、**融合省带宽**、**后端可插拔**。三个实验都围绕同一组代码——同一数学、不同工程实现的差距从哪来。

lab 章的读法和 guide 不同：guide 是「读进去」，lab 是「跑起来」。建议全程开着 `miniflow/tensor.py` 和 `csrc/kernel.cpp` 两份源码对照——**每一段文字都对应你能亲眼看到的代码行**。本实验全部在 CPU 上完成（不需要 GPU），所有跑分数字在小笔记本上 30 秒内出结果。

## 0. 环境准备与第一次运行

五分钟热身（确保后面实验顺畅）：

```bash
# 1. 安装本仓（可编辑模式）
pip install -e .

# 2. 验证基础后端
python -c "from miniflow.tensor import backend; print(backend())"
# 预期输出：numpy （还没编译 C++ 扩展）

# 3. 跑算子层测试
python -m pytest tests/test_tensor.py -q
# 预期：全绿 —— NumPy 路径的 17 个断言

# 4.（可选）编译 C++ 后端
pip install pybind11 && python setup.py build_ext --inplace
python -c "from miniflow.tensor import backend; print(backend())"
# 预期输出：cpp
```

第 3 步全绿是硬前提——**先正确后快速**是本仓的宪法（guide/04 的「先写对的、再写快的、用前者当裁判」）。第 4 步失败不影响任何后续实验（回落机制会兜底），但推荐做——`ikj` 跑分需要它。

## 1. 结构：参考实现 + 加速实现 + 派发

```text
miniflow/tensor.py     ← NumPy 参考实现（清晰、可单测）
csrc/kernel.cpp        ← C++ 后端（pybind11 编译为 miniflow._C）
                          ↳ BACKEND 探测：可用则派发，否则回落 NumPy
```

`tensor.py` 的探测逻辑只有 5 行，但它复刻的是生产框架「多算子后端按可用性自动选择」的机制：

```python
try:
    import miniflow._C as _C
    BACKEND = "cpp"
except ImportError:
    _C = None
    BACKEND = "numpy"
```

**可选性是设计的一部分**：没编译 C++ 扩展时一切照常——功能与正确性由 NumPy 路径保证（17 个测试在两种后端下都通过）。本仓开发时沙箱无网络装不了 pybind11，这条降级路径本身就是被真实使用过的：引擎全程跑 NumPy 后端完成开发与测试，C++ 核是「锦上添花」而非「必需品」。

值得看一眼派发的完整形态（`tensor.py` 里每个算子函数的结尾）：

```python
def matmul(a, b):
    if _C is not None:
        return _C.matmul(a, b)     # 快路径
    return _numpy_matmul(a, b)     # 参考路径
```

**每个算子一个 if**——朴素但透明。生产的派发器更复杂（按输入 dtype/形状/设备路由到不同 kernel 的注册表模式），但语义就是这个 if：**同一数学的多级实现，运行时挑一个**。guide/04 的 5.4 节说过这是「韧性」的来源——部署环境缺什么就回落什么，功能不损。

## 2. 实验：循环序与缓存局部性

`csrc/kernel.cpp` 提供两个矩阵乘——**同样的 O(M·K·N) 计算量**，只差循环顺序：

```cpp
// ijk 内积视角：c[i][j] = <a 的第 i 行, b 的第 j 列>
for (i...) for (j...) for (k...)
    acc += pa[i*K+k] * pb[k*N+j];
// ↑ pb[k*N+j]：k 在最内层变，b 按列跳着访问 —— 每步换一个缓存行

// ikj 外积视角：c 的第 i 行 += a[i][k] * b 的第 k 行（整行广播）
for (i...) for (k...) for (j...)
    ci[j] += aik * bk[j];
// ↑ bk[j]：j 在最内层变，b 沿一行连续访问 —— 一个缓存行能用满才换
```

### 2.1 差距的机理：缓存行经济学

差距的来源值得说透。CPU 从内存取数据不是按字节，是按**缓存行**（通常 64 字节 = 16 个 float）。

`ijk` 版本里最内循环沿 b 的**列**走——每前进一个 k 就跳到下一行，取来 16 个 float 只用 1 个，下一个 k 又要新的一行；`ikj` 版本沿 b 的**行**走——取来一行用满 16 个再换。

把这笔账算精确（256×256 的矩阵，float32）：

```text
ijk 的 b 访问：
  总访问次数 = 256×256×256 = 1670 万次乘法，每次读 b 一个元素
  b 的列内相邻元素跨 256×4B = 1KB —— 每次访问都是新缓存行！
  缓存行加载次数 ≈ 1670 万（L1 miss 率 ~100%）
  
ikj 的 b 访问：
  同样 1670 万次乘法，但沿行连续 —— 每 16 次访问共享一个缓存行
  缓存行加载次数 ≈ 104 万（L1 miss 率 ~6%）
  
差距：16 倍的内存流量 —— 这就是跑分里一个数量级的来源
```

注意一个微妙点：**总计算量（乘加次数）一个都没变**——变的是「每个字节的搬运被摊给了几次计算」。这正是 [guide/04](../guide/04-kernels.md) 里 DeepGEMM 分块思想的 CPU 版：**分块的本质就是让每个缓存行在被逐出前被用尽可能多次**。GPU 的 SRAM tile（guide/04 0.4 节）和 CPU 的缓存行是同一思想在不同硬件上的粒度——**「摊销搬运」不关心你摊的是 HBM 还是 DRAM**。

### 2.2 跑分验证

跑分脚本（把下面存成 `bench_matmul.py` 再运行——比交互式片段更可复现）：

```python
import numpy as np, timeit
from miniflow import _C

a = np.random.rand(256, 256).astype(np.float32)
b = np.random.rand(256, 256).astype(np.float32)

for name, fn in [("naive ijk", _C.matmul_naive),
                 ("cache ikj", _C.matmul),
                 ("numpy BLAS", lambda x, y: x @ y)]:
    t = timeit.timeit(lambda: fn(a, b), number=10)
    print(f"{name:12s} {t*100:7.1f} ms / 10 次")
```

```bash
pip install pybind11 && python setup.py build_ext --inplace
python bench_matmul.py
```

典型结果（i7 笔记本，量级示意）：

```text
naive ijk    ~850 ms / 10 次
cache ikj    ~120 ms / 10 次     ← 7 倍：纯循环序的收益
numpy BLAS   ~4 ms / 10 次       ← 30 倍：分块+SIMD+多线程的全家桶
```

三行结果的读法：

- **第 1→2 行的差距**是「循环序」单独的贡献——不改算法不改数学，**7 倍**；
- **第 2→3 行的差距**是「分块 + SIMD 向量化 + 多线程」的叠加——NumPy 背后是几十年的 BLAS 工程。**教学核和工业核的差距是「同一方向的更多步」**，不是另一个方向；
- 思考：ikj 版本自己加个 64×64 的分块（三级循环嵌进 tile 循环），能吃到第 2→3 行差距的几成？（动手试试——这是思考题 3 的实验版。）

??? question "思考题"

    1. 为什么 `ikj` 里 `if (aik == 0.0f) continue;` 是个合法优化，
       而在 `ijk` 版里加同样判断收益小得多？（提示：外层判断每次
       跳过的是一整行的工作，内层判断每次只跳过一次乘加。）
    2. 阈值大约在什么矩阵尺寸上，NumPy 的调度开销开始被 BLAS 的
       速度优势盖过？（提示：很小的矩阵上 `_C.matmul` 可能反超——
       pybind11 直达 C++，没有任何解释层。）
    3. 给 `ikj` 加 64×64 分块：把 2.1 节的缓存行账重算一遍——分块后
       a 和 b 的行加载次数各多少？预测加速比，实测验证。再想想：
       为什么 64 是常见选择（L1 通常是 32~48 KB，三个 64×64 的
       float tile 占多少）？
    4. 把 `a` 换成 4096×4096（放不进 L2）重跑——`ikj` 和 `ijk` 的
       差距变大还是变小？为什么（提示：缓存层级越深，miss 的代价
       越贵——locality 的价值随存储层级放大）。

### 2.3 附加实验：亲手测量缓存行大小

一个 10 分钟的附加实验，把「64 字节」从书上的数字变成你的实测（了解底层之后回头看跑分，感觉完全不同）：

```python
# stride_test.py：不同步长遍历数组，测时间跳变
import numpy as np, time

arr = np.zeros(64 * 1024 * 1024, dtype=np.float32)   # 256 MB，远超缓存
N = len(arr)

for stride in [1, 2, 4, 8, 16, 32, 64]:
    n_access = 4_000_000                            # 固定总访问次数
    idx = np.arange(0, n_access * stride, stride) % N
    t0 = time.perf_counter()
    s = arr[idx].sum()                                # 触摸内存
    dt = time.perf_counter() - t0
    per = dt / n_access * 1e9
    print(f"stride={stride:3d} 元素 ({stride*4:3d} B): {per:6.1f} ns/次")
```

预期曲线：stride 1~16（4~64 字节）时每次访问时间几乎不变（一个缓存行装下）——**stride 过 16（64 字节）后每次访问时间跳一档**（每个访问都 miss）。那个跳变的临界步长 × 4 字节 = 你机器的缓存行大小。**这个实验教你的不是某个数字，而是「怎么用程序问硬件问题」**——profiling 的原语思维。

## 3. 实验：算子融合省带宽

`fused_silu_swiglu` 把 SwiGLU 的 `silu(gate) * up` 融合成单趟核：

```cpp
for (i...)
    po[i] = pg[i] / (1 + exp(-pg[i])) * pu[i];
```

对比朴素路径：先算 `silu(gate)` **写**一个临时张量，再读回来和 `up` 相乘**再写**——中间张量在内存里往返两趟。融合后数据待在寄存器里一遍过。

把这个「两趟 vs 一趟」的账算开（假设 4096×4096 的 gate/up，float32）：

```text
朴素路径的内存流量：
  读 gate（64 MB）→ 写 silu 结果（64 MB）→ 读 silu（64 MB）
  → 读 up（64 MB）→ 写输出（64 MB）= 320 MB

融合路径：
  读 gate（64 MB）+ 读 up（64 MB）→ 写输出（64 MB）= 192 MB

省 40% —— 中间张量的一次往返（128 MB）被消灭
```

对齐生产逻辑：decode 阶段**访存受限**（[guide/01](../guide/01-overview.md) 的账本）——每省一次中间张量的读写，省下的都是实打实的 HBM 带宽。GPU 上融合还有第二重收益（kernel 启动开销：3 个小 kernel 变 1 个，省 2 次启动的微秒级固定成本）——**CPU 实验里你看到的是「省带宽」这一重，第二重在 GPU 的 profile 里才可见**。

TileLang 生成的融合核是这个思想的全自动版本：你写算法描述，编译器决定融合与布局。

### 3.1 融合的正确性陷阱（进阶）

融合不是无脑合并——它有**数值上的陷阱**值得知道（这也是为什么生产融合核要过严格的精度回归）：

朴素路径的 `silu` 结果**先舍入存储**（float32 写回内存）再参与乘法；融合路径里它**以寄存器精度**（可能被编译器保存在扩展精度或重排运算）直接乘——**两者的最后一位可能不同**。单独看无害（1 ULP），但下游若做归约（softmax 分母），ULP 差会传播（guide/09 的放大链）。

生产的解法：融合核的精度回归以「端到端输出对齐到若干位」为标准（不是逐位）——**融合的收益和数值的严格性要在契约里平衡**。miniflow 的 `tests/test_tensor.py` 对融合核的断言是 `np.allclose`（容差比较）而非 `array_equal`（逐位）——就是这个平衡的体现。

## 4. 参考实现里值得读的三处

`miniflow/tensor.py`（NumPy 侧）有三个值得停下看的函数：

1. **数值稳定 softmax**：先减 max 再 exp。直接 `exp(x)` 在大 logits 下会溢出（inf/inf = nan）；减去行最大值后最大项是 $e^0=1$，永远安全。它对应 FlashMLA 在线 softmax 的「分块重整化」数学——running max 就是把「减 max」搬进分块循环里做（[guide/04 第 1.2 节](../guide/04-kernels.md)）；

   ```python
   # tensor.py 的 softmax（值得逐行读）
   def softmax(x, axis=-1):
       m = np.max(x, axis=axis, keepdims=True)   # 行最大
       e = np.exp(x - m)                          # 平移后永远安全
       return e / np.sum(e, axis=axis, keepdims=True)
   ```

   做一个 30 秒实验验证「不稳定版」真的会炸：`np.exp([1000, 1001])` 直接 overflow 出 inf/inf=nan；平移后 `[e^-1, e^0]` 完美。**「为什么教科书都这么写」的原因亲手看到一次就忘不了**。

2. **RMSNorm 少一次中心化**：`x / sqrt(mean(x²) + eps) * w`。LayerNorm 要先减均值再除标准差，RMSNorm 只除均方根——少一趟归约少一趟访存，DeepSeek 全系用它。这个「少一次归约」在 guide/09 的确定性语境里还有一层意义：**归约越少，浮点顺序的暴露面越小**——RMSNorm 天然比 LayerNorm 更 batch-invariant 友好（少一个求和就少一个顺序敏感点）。

3. **RoPE 奇偶配对**：`x[..., 0::2]` 与 `x[..., 1::2]` 旋转配对。生产实现在 MLA 核内融合，教学版独立成函数，先看清数学再谈融合。

## 5. 验收清单

- [ ] `python -c "from miniflow.tensor import backend; print(backend())"`——无 C++ 扩展时输出 `numpy`，编译后输出 `cpp`
- [ ] 编译后全量测试通过：`python -m pytest tests/ -q`（两种后端的正确性等价）
- [ ] 跑分观察到 `naive` 与 `ikj` 的量级差
- [ ] （附加）`stride_test.py` 测出你机器的缓存行大小
- [ ] （附加）给 `ikj` 加分块，实测吃到 BLAS 差距的几成

!!! note "为什么教学核「跑不过 BLAS」是预期行为"

    本实验的目标是**看清差距从哪来**（循环序 / 融合 / 派发），不是替代
    BLAS。NumPy 底层是工业级的 MKL/OpenBLAS，一个教学 C++ 核跑不过是
    正常且诚实的。生产框架同样只在通用库达不到要求时才手写核
    （[guide/04 第 4 节](../guide/04-kernels.md)）——「不写」也是
    工程决策。

    更深一层：**这个 note 本身是教学的一部分**——「教学核跑分被 BLAS
    碾压」的诚实披露，对应生产世界「什么时候该自研、什么时候该用
    现成库」的判断力。flash 系算子的存在理由是「通用库的形状假设
    不匹配」（guide/04 2.4 节的连续布局），不是「自研一定更快」——
    看清这条边界，你自己的项目就不会掉进「什么都手写」的坑。

## 6. 本章收获的自检

离开本章前，确认你能不看书回答：

1. `ijk` 和 `ikj` 差 7 倍的**全部**原因是什么（缓存行 16 float 的摊销）？计算量变了吗（没有）？
2. 融合省的是哪一段内存流量（中间张量的一次往返）？占朴素路径的几成（40%）？
3. `BACKEND` 探测复刻了生产的什么机制（能力探测+优雅回落）？它为什么重要（部署韧性）？
4. softmax 减 max 对应 FlashMLA 的什么（在线重整化的 running max）？

四题都顺的话，guide/04 的核心（「摊销搬运」的四种化身）就有了肌肉记忆。

| 下一章 |
|---|
| [Lab 02 模型层与手写反向传播](02-model.md) |