# Lab 01 算子层：同一数学的三种实现

!!! info "本章对应"
    - 生产组件：FlashMLA / DeepGEMM / DeepEP / TileLang
    - miniflow 代码：`miniflow/tensor.py`、`csrc/kernel.cpp`、`setup.py`
    - 前置阅读：[04 内核层](../guide/04-kernels.md)

---

## 1. 结构：参考实现 + 加速实现 + 派发

```text
miniflow/tensor.py     ← NumPy 参考实现（清晰、可单测）
csrc/kernel.cpp        ← C++ 后端（pybind11 编译为 miniflow._C）
                          ↳ BACKEND 探测：可用则派发，否则回落 NumPy
```

`tensor.py` 的探测逻辑只有 5 行，但它复刻的是生产框架「多算子后端按
硬件/负载自动选择」的机制：

```python
try:
    import miniflow._C as _C
    BACKEND = "cpp"
except ImportError:
    _C = None
    BACKEND = "numpy"
```

**可选性是设计的一部分**：没编译 C++ 扩展时一切照常 —— 功能与
正确性由 NumPy 路径保证（17 个测试两种后端都过）。

## 2. 实验：循环序与缓存局部性

`csrc/kernel.cpp` 提供两个矩阵乘 —— **同样的 O(M·K·N)**：

```cpp
// ijk 内积视角：c[i][j] = <a 的第 i 行, b 的第 j 列>
// b 按列访问 => 每步缓存 miss —— 教科书慢的根源
for (i...) for (j...) for (k...)
    acc += pa[i*K+k] * pb[k*N+j];

// ikj 外积视角：c 的第 i 行 += a[i][k] * b 的第 k 行（整行广播）
// b 的行在内存里连续 —— 缓存友好的最小改造
for (i...) for (k...) for (j...)
    ci[j] += aik * bk[j];
```

编译后跑分（教学小矩阵即可观察）：

```bash
pip install pybind11 && python setup.py build_ext --inplace
python - <<'PY'
import numpy as np, timeit
from miniflow import _C
a = np.random.rand(256, 256).astype(np.float32)
b = np.random.rand(256, 256).astype(np.float32)
for name, fn in [("naive ijk", _C.matmul_naive),
                 ("cache ikj", _C.matmul),
                 ("numpy BLAS", lambda x, y: x @ y)]:
    t = timeit.timeit(lambda: fn(a, b), number=10)
    print(f"{name:12s} {t*100:7.1f} ms / 10 次")
PY
```

??? question "思考题"
    1. 为什么 `ikj` 里 `if (aik == 0.0f) continue;` 是合法的优化，
       而 `ijk` 版里加同样判断收益小得多？
    2. 阈值大约在什么矩阵尺寸上，NumPy 的调用开销开始小于 BLAS
       的速度优势？（提示：小矩阵上 `_C.matmul` 可能反超 ——
       pybind11 直达 C++ 没有任何解释层。）

## 3. 实验：算子融合省带宽

`fused_silu_swiglu` 把 `silu(gate) * up` 融合成单趟核：

```cpp
for (i...)
    po[i] = pg[i] / (1 + exp(-pg[i])) * pu[i];
// 朴素路径：silu(gate) 写临时张量 → mul 读回再写 → 两度额外访存
```

对齐生产逻辑：decode 阶段**访存受限** —— 每省一次中间张量的读写，
省下的都是 HBM 带宽。TileLang 生成的融合核是这个思想的全自动版本
（写算法描述，编译器决定融合布局）。

## 4. 参考实现里值得读的三处

`miniflow/tensor.py`（NumPy 侧）：

1. **数值稳定 softmax**：先减 max 再 exp —— 对应 FlashMLA 在线
   softmax 的「分块重整化」数学（[guide/04](../guide/04-kernels.md)）；
2. **RMSNorm 少一次中心化**：`x / sqrt(mean(x²) + eps) * w` ——
   DeepSeek 全系用它，kernel 比 LayerNorm 简单一截；
3. **RoPE 奇偶配对**：`x[..., 0::2]` 与 `x[..., 1::2]` 旋转配对 ——
   生产实现在 MLA 核内融合，教学版独立成函数先看清数学。

## 5. 验收清单

- [ ] `python -c "from miniflow.tensor import backend; print(backend())"`
      无 C++ 扩展时输出 `numpy`，编译后输出 `cpp`
- [ ] 编译后全量测试通过：`python -m pytest tests/ -q`
      （两种后端的正确性等价性）
- [ ] 跑分观察到 `naive` 与 `ikj` 的量级差

!!! note "为什么教学核「跑不过 BLAS"是预期行为"
    本实验的目标是**看清差距从哪来**（循环序 / 融合 / 派发），
    不是替代 BLAS。生产框架同样只在通用库达不到要求时才手写核
    （[guide/04 第 4 节](../guide/04-kernels.md)）—— 「不写」
    也是工程决策。

| 下一章 |
|---|
| [Lab 02 模型层与手写反向传播](02-model.md) |