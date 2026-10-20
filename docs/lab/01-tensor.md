# Lab 01 算子层：同一数学的三种实现

!!! info "本章对应"
    - 生产组件：FlashMLA / DeepGEMM / DeepEP / TileLang
    - miniflow 代码：`miniflow/tensor.py`、`csrc/kernel.cpp`、`setup.py`
    - 前置阅读：[04 内核层](../guide/04-kernels.md)

guide 04 讲了算子层决定性能上限。本章动手验证三个最根本的算子层事实：**循环序决定缓存局部性**、**融合省带宽**、**后端可插拔**。三个实验都围绕同一组代码——同一数学、不同工程实现的差距从哪来。

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

差距的来源值得说透。CPU 从内存取数据不是按字节，是按**缓存行**（通常 64 字节 = 16 个 float）。`ijk` 版本里最内循环沿 b 的**列**走——每前进一个 k 就跳到下一行，取来 16 个 float 只用 1 个，下一个 k 又要新的一行；`ikj` 版本沿 b 的**行**走——取来一行用满 16 个再换。同样的总乘法次数，内存搬运量差了一个数量级。这正是 [guide/04](../guide/04-kernels.md) 里 DeepGEMM 分块思想的 CPU 版：**分块的本质就是让每个缓存行在被逐出前被用尽可能多次**。

跑分验证（把下面存成 `bench_matmul.py` 再运行——比交互式片段更可复现）：

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

??? question "思考题"

    1. 为什么 `ikj` 里 `if (aik == 0.0f) continue;` 是个合法优化，
       而在 `ijk` 版里加同样判断收益小得多？（提示：外层判断每次
       跳过的是一整行的工作，内层判断每次只跳过一次乘加。）
    2. 阈值大约在什么矩阵尺寸上，NumPy 的调度开销开始被 BLAS 的
       速度优势盖过？（提示：很小的矩阵上 `_C.matmul` 可能反超——
       pybind11 直达 C++，没有任何解释层。）

## 3. 实验：算子融合省带宽

`fused_silu_swiglu` 把 SwiGLU 的 `silu(gate) * up` 融合成单趟核：

```cpp
for (i...)
    po[i] = pg[i] / (1 + exp(-pg[i])) * pu[i];
```

对比朴素路径：先算 `silu(gate)` **写**一个临时张量，再读回来和 `up` 相乘**再写**——中间张量在内存里往返两趟。融合后数据待在寄存器里一遍过。

对齐生产逻辑：decode 阶段**访存受限**（[guide/01](../guide/01-overview.md) 的账本）——每省一次中间张量的读写，省下的都是实打实的 HBM 带宽。TileLang 生成的融合核是这个思想的全自动版本：你写算法描述，编译器决定融合与布局。

## 4. 参考实现里值得读的三处

`miniflow/tensor.py`（NumPy 侧）有三个值得停下看的函数：

1. **数值稳定 softmax**：先减 max 再 exp。直接 `exp(x)` 在大 logits 下会溢出（inf/inf = nan）；减去行最大值后最大项是 $e^0=1$，永远安全。它对应 FlashMLA 在线 softmax 的「分块重整化」数学——running max 就是把「减 max」搬进分块循环里做（[guide/04 第 1.2 节](../guide/04-kernels.md)）；
2. **RMSNorm 少一次中心化**：`x / sqrt(mean(x²) + eps) * w`。LayerNorm 要先减均值再除标准差，RMSNorm 只除均方根——少一趟归约少一趟访存，DeepSeek 全系用它；
3. **RoPE 奇偶配对**：`x[..., 0::2]` 与 `x[..., 1::2]` 旋转配对。生产实现在 MLA 核内融合，教学版独立成函数，先看清数学再谈融合。

## 5. 验收清单

- [ ] `python -c "from miniflow.tensor import backend; print(backend())"`——无 C++ 扩展时输出 `numpy`，编译后输出 `cpp`
- [ ] 编译后全量测试通过：`python -m pytest tests/ -q`（两种后端的正确性等价）
- [ ] 跑分观察到 `naive` 与 `ikj` 的量级差

!!! note "为什么教学核「跑不过 BLAS」是预期行为"

    本实验的目标是**看清差距从哪来**（循环序 / 融合 / 派发），不是替代
    BLAS。NumPy 底层是工业级的 MKL/OpenBLAS，一个教学 C++ 核跑不过是
    正常且诚实的。生产框架同样只在通用库达不到要求时才手写核
    （[guide/04 第 4 节](../guide/04-kernels.md)）——「不写」也是
    工程决策。

| 下一章 |
|---|
| [Lab 02 模型层与手写反向传播](02-model.md) |