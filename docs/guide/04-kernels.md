# 04 内核层：FlashMLA / DeepGEMM / DeepEP / TileLang

!!! question "本章问题"
    - 为什么推理框架的性能上限由**算子**决定？
    - MLA 的数学结构如何被 FlashMLA 兑现成「KV 缩容 + 快注意力」？
    - DeepGEMM 为什么需要**两种模式**？这和 E/P/D 分离有什么呼应？
    - MoE 的瓶颈为什么在**通信**（DeepEP）而不在计算？

---

## 0. 一个先决认知：算子决定上限

引擎层的调度、组批、流水线做得再好，也只是把「好形状的输入」递给算子。
**乘法发生在算子里** —— 算子慢 20%，全系统慢 20%。DeepInfer 把最硬的
功夫全砸在这一层：

| 算子 | 服务对象 | 优化维度 |
|---|---|---|
| **FlashMLA** | MLA 注意力（prefill + decode） | 显存带宽、在线 softmax |
| **DeepGEMM** | 全部矩阵乘 | 分块满血 vs 连续布局两模式 |
| **DeepEP** | MoE all-to-all | 通信量、低精度传输 |
| **TileLang** | 算子生成 | 用 DSL 写高性能核的工程效率 |

## 1. FlashMLA：为 MLA 量身定制

### 1.1 MLA 先把 KV「压扁」

DeepSeek 系列的 MLA（Multi-head Latent Attention）核心是把 KV 投影到
低秩 latent 空间：

```text
标准 MHA：  KV cache 每 token 存 [num_heads × head_dim]   （大）
MLA：       KV cache 每 token 存 [latent_dim]             （小得多）
            读出时经 up-projection 矩阵恢复各头的 K/V
```

关键技巧（**absorption**）：up-projection 可以**吸收进 Q 的权重** ——
注意力计算等价变形后，**latent 向量直接参与点积**，无需物化展开的 K/V。
于是 KV cache 的体积与注意力计算量同时下降。

### 1.2 FlashMLA 把这个数学变成核

FlashMLA 在一块 GPU 核里完成「读 latent → 融合吸收矩阵 → 在线
softmax → 输出」，两个要害：

1. **在线 softmax**（分块重整化）：注意力分数矩阵 O(seq²) 从不物化，
   分块计算、边算边维护 running max/sum —— 显存 O(seq) 而非 O(seq²)；
2. **decode 优化布局**：decode 时 seq 维极小、batch 维（请求数）极大，
   FlashMLA 的分块按「请求 × latent」排布，把访存合并成大连续读 ——
   正好治 [第 01 章](01-overview.md) 说的「decode 喂不饱芯片」。

!!! note "GQA 是 MLA 的「穷人版」"
    MLA：所有头共享一个 latent 向量（经吸收恢复差异）；
    GQA：若干 Q 头共享一组 KV 头。两者都在压 KV，MLA 压得更狠但数学
    更复杂。**miniflow 用 GQA 复刻「KV 头数 < Q 头数」的结构收益**
    （`miniflow/config.py` 的 `num_kv_heads=2 < num_attention_heads=4`），
    复杂度换成了可读性 —— 教学取舍。

## 2. DeepGEMM：两种模式对应两种负载

矩阵乘是 prefill 与 decode 共同的主体，但两边的「最优实现」长得完全
不一样。DeepGEMM 干脆提供两种模式：

| 模式 | 形状特征 | 分块策略 | 主要场景 |
|---|---|---|---|
| **分块满血**（TMA 满血 GEMM） | 大 M、大 N（批量 token × 大专家 batch） | 大 tile + 异步流水，算力打满 | **prefill / P 节点** |
| **连续布局**（decode 组） | M 小（每请求 1 token）但**组数多** | 把「请求 × 专家」拼成连续大 M | **decode / D 节点** |

decode 模式的巧劲：单条请求只算 1 个 token 的新 K/Q —— 矩阵乘的 M=1，
怎样都喂不饱 GPU。但一个 batch 里有 N 条请求 × 各自路由到 E 个专家，
把「(请求, 专家) 对」的行**拼接成连续布局**，M 从 1 变成 N×平均专家数
—— 小矩阵乘拼成大矩阵乘，DeepGEMM 的 decode 模式就是为这个形状定制的。

!!! tip "与 E/P/D 的呼应"
    DeepGEMM 的两模式 = P/D 分离在算子层的投影。**架构分离不是拍脑袋，
    是算子形状差异大到值得物理分离** —— 反过来说：算子库的形状感知
    （shape-aware）越强，分离的收益越大。

## 3. DeepEP：MoE 的瓶颈是通信

MoE 每层要做两件事：

```text
dispatch：  各卡的 token 按路由表发往「专家所在卡」   （all-to-all）
compute：   专家算自己的矩阵乘
combine：   结果发回原卡，加权求和                     （all-to-all）
```

FLOPs 不是问题 —— 问题是 **dispatch/combine 的通信量随 EP 规模线性
增长**，且 decode 时每卡发出的 token 数极少（消息小、延迟占比高）。
DeepEP 的三板斧：

1. **分模式**：prefill 用高吞吐常规 all-to-all；decode 用
   **低延迟内核**（RDMA 直达、小消息优化）；
2. **低精度传输**：dispatch 的 token 以 FP8/BF16 压缩传输 ——
   通信量直接减半，精度损失可接受；
3. **钩子式重叠**：通信与计算分阶段流水（dispatch 先行、算子跟进），
   通信延迟被计算掩盖。

## 4. TileLang：手写与生成的分界线

FlashMLA/DeepEP 这类核是**手写**的（CUDA/汇编级），工程成本极高。
TileLang 是 DeepSeek 的算子 DSL：用「分块 + 循环 + stage」的抽象描述
算法，编译器生成高性能核。分界线画在：

- **通用热核**（MLA、GEMM、EP 通信）：手写，榨干每一滴；
- **长尾/新结构核**（新算子原型、融合变体）：TileLang 生成，
  快速验证再决定要不要手写。

## 5. miniflow 的算子层教学

miniflow 用三层结构复刻「算子层值得单独一讲」这个事实：

```text
miniflow/tensor.py     NumPy 参考实现（清晰、可单测、可迁移）
csrc/kernel.cpp        C++ 后端（pybind11 编译为 miniflow._C）
                       ↳ BACKEND 探测：可用则派发，否则回落 NumPy
```

三个教学点直接写在代码里（`csrc/kernel.cpp`）：

1. **`matmul_naive` vs `matmul`**：同样的 O(M·K·N)，ijk 内积视角与
   ikj 外积视角差一个量级 —— **循环序决定缓存局部性**，
   DeepGEMM 的分块是同一思想的工程极致；
2. **`fused_silu_swiglu`**：`silu(gate) * up` 三个算子融合成一个核，
   省两度中间张量访存 —— decode 访存受限，**省带宽就是省时间**；
3. **`softmax` / `rms_norm`**：数值稳定的标准实现，对应 FlashMLA 的
   在线 softmax（分块重整化）。

```bash
pip install pybind11 && python setup.py build_ext --inplace
python -c "from miniflow.tensor import backend; print(backend())"
# 编译成功输出 cpp，否则 numpy —— 功能完全等价，只是性能不同
```

!!! warning "教学取舍"
    miniflow 的 C++ 核在 CPU 上**未必比 NumPy（BLAS）快** —— 教学价值
    在于看清「同一数学的三种实现工程差距从哪来」，不在跑分。

---

| 上一章 | 下一章 |
|---|---|
| [03 总体架构](03-architecture.md) | [05 并行策略：DP/EP/EPLB](05-parallelism.md) |