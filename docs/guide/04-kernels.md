# 04 内核层：FlashMLA / DeepGEMM / DeepEP / TileLang

引擎层的调度、组批、流水线做得再好，也只是把「形状合适的输入」递给算子——**乘法本身发生在算子里**。一个占全程 60% 时间的矩阵乘算子慢 20%，整个服务就慢 12%。所以推理框架的性能上限由算子层决定，DeepInfer 把最硬的功夫都砸在这里。本章讲四个组件：FlashMLA（注意力算子）、DeepGEMM（矩阵乘算子）、DeepEP（MoE 通信算子）、TileLang（算子的生产工具）。

## 1. FlashMLA：为 MLA 注意力量身定制

### 1.1 先讲 MLA 解决什么问题

第 01 章说过，decode 的显存开销大头是 KV Cache。标准多头注意力（MHA）下，**每个 token、每一层都要为每个注意力头存一份 K 和 V**。以 128 头、每头 128 维计，一个 token 一层的 KV 是 2×128×128×2 字节（FP16）≈ 128 KB；128 层的模型，一个 token 的 KV 就是 16 MB，一条 4000 token 的请求要 64 GB——单卡根本放不下几条请求，D 节点「KV 驻留密度」这个核心指标直接崩盘。

MLA（Multi-head Latent Attention，多头潜在注意力）的做法：**不存完整的 K/V，存一个低维的压缩向量**。把 KV 投影到一个很小的 latent 空间（DeepSeek-V3 用 512 维，对比上面的 32768 维份量），缓存里只存这个 latent 向量；要用的时候，通过一个 up-projection（升维）矩阵把各头的 K/V「恢复」出来。KV Cache 直接缩到几十分之一——这是模型侧给系统侧的礼物。

代价是计算结构变复杂了：恢复出的 K/V 要和 Q 做点积。这里有一个关键技巧叫**吸收**（absorption）：注意到「恢复再点积」在数学上可以换序——

$$
q^\top (W_{up} \, c) = (q^\top W_{up}) \, c
$$

其中 $c$ 是 latent 向量，$W_{up}$ 是升维矩阵。与其给每个 token 恢复 K 再算点积，不如把 $W_{up}$ 预先乘进 Q 那侧（Q 是当前 token 现算的，顺手就做了）。变形之后，**latent 向量直接参与点积，展开的 K/V 从未真正存在过**。KV Cache 的体积和注意力的计算量同时下降。

### 1.2 FlashMLA 把这个数学做成核

有了 MLA 的数学结构，还要有一个把它跑满带宽的 GPU 核。FlashMLA 的两个要害：

**要害一：在线 softmax（分块重整化）。** 注意力需要对整行分数做 softmax。朴素实现要先把整个分数矩阵算出来存进显存（O(seq²) 空间），再读回来归一化——写一遍读一遍，来回搬运。Flash 系列算子的核心思想是**根本不物化这个矩阵**：把 K/V 切成小块逐块处理，每处理一块就维护两个运行量——当前见过的最大分数 $m$ 和指数和 $l$。新来一块时，先算这块的局部最大值 $m_{new} = \max(m, m_{blk})$，再把已累积的输出按 $\exp(m - m_{new})$ 缩放修正，然后累加本块贡献。数学上严格等价于先算完整 softmax 再加权求和，但中间结果全部待在寄存器/共享内存里，显存占用从 O(seq²) 降到 O(seq)，每个数据只搬一次。

**要害二：decode 专用布局。** decode 时每条请求只有 1 个新 token，但批内可能有 200 条请求——形状是「序列维度极小、请求维度极大」。通用注意力核按「头 × 序列」分块，在这个形状下访存全是一个 token 一个 token 的碎读。FlashMLA 按「请求 × latent 维」排布分块，把碎片访问合并成大段连续读——正好治第 01 章诊断的「decode 喂不饱芯片」。

!!! note "MHA → GQA → MLA 的递进"

    三代结构都在压 KV Cache：MHA 每个头独立存 KV；GQA 让若干个 Q 头共享一组 KV 头（头数维度直接砍）；MLA 更进一步，所有头共享一个 latent 向量，靠吸收技巧恢复各头差异。miniflow 用 GQA 复刻「KV 头数 < Q 头数」的结构收益（`miniflow/config.py`：`num_kv_heads=2 < num_attention_heads=4`）——数学复杂度换成可读性，是教学取舍。

## 2. DeepGEMM：两种模式对应两种负载

矩阵乘占整个前向计算的绝大部分（ attention 之外全是他）。DeepGEMM 是 DeepSeek 的矩阵乘算子库，它的特殊之处是**明确分成两种模式**，而分界线正是第 03 章的 P/D 分离：

**满血模式（prefill 侧）。** prefill 的矩阵乘形状天然肥硕：几百条请求 × 各几千 token 的 M 维，乘上大 N 维。这种形状用大 tile 分块 + 异步流水线（TMA 异步搬运，计算与下一次搬运重叠）把算力打满——经典的高算术强度优化，没什么特别。

**连续布局模式（decode 侧）才是巧劲。** decode 时单条请求的矩阵乘 M=1——一个 token 乘权重矩阵，怎么摆都是「瘦条」，GPU 一次能并行算的 tile 填不满。但换个视角：批里有 200 条请求，每条路由到 8 个专家，就有 200×8 个「(请求, 专家) 对」，每对都是一个 M=1 的小矩阵乘。DeepGEMM 的 decode 模式把所有这些对的输入行**拼接成一段连续内存**，M 从 1 变成 1600——小矩阵乘拼成了大矩阵乘。这不是简单的 concat 技巧：它要求路由、dispatch、算子的内存布局从头设计成一体，也正是它必须和 DeepEP（下一节）配合的原因。

一个数字感受两模式的差距：M=1 的 GEMM 在 H800 上可能只跑到峰值的 2%；拼成 M=1600 的连续布局，能到 60% 以上。**P/D 分离在算子层的投影就是这两套 GEMM**——反过来说，两类负载的算子形状差异大到值得物理分离，架构不是拍脑袋拍的。

## 3. DeepEP：MoE 的瓶颈是通信

DeepSeek 的 MoE 层每前向一步要做三件事：

```text
dispatch：  每张卡把自己负责的 token 按路由表发往「专家所在的卡」  （all-to-all）
compute：   各卡上的专家对自己的 token 做矩阵乘
combine：   算完的结果发回原卡，按路由权重加权求和              （all-to-all）
```

计算本身不重（MoE 每 token 只激活 37B/671B 的参数），**瓶颈在两次 all-to-all 通信**。而且两头受气：

- prefill 侧：token 多，通信量 = token 数 × top-k 路由 × hidden 维度，随规模线性涨，拼的是**带宽**；
- decode 侧：每卡每步只发一两个 token，消息极小——网络传输的固定开销（延迟）占比反超数据本身的传输时间，拼的是**延迟**。

DeepEP 的三板斧对着这两个痛点：

1. **双模式内核**：prefill 用高吞吐常规 all-toall（NVLink/RDMA 满带宽）；decode 用**低延迟内核**——RDMA 直达、绕过常规协议栈、按小消息特征微调。一套库装两副牙齿；
2. **低精度传输**：dispatch 的 token 以 FP8/BF16 传输，通信字节数直接减半。通信传输的是中间激活值，FP8 的精度损失在最终输出里几乎不可见——用一点精度换一倍带宽，划算；
3. **通信-计算重叠**：dispatch 按「先到先算」钩子式推进——哪批 token 先到位就先算哪个专家，通信的尾巴被计算盖住，不干等。

## 4. TileLang：手写与生成的分界线

FlashMLA 和 DeepEP 这类核是手写的（CUDA、甚至汇编级微调），一个核的工程成本以人月计。每个新算子都手写不现实，全靠通用编译又拿不到极致性能。TileLang 是 DeepSeek 的算子 DSL（领域专用语言）：你用「分块 + 循环 + stage」的抽象描述算法结构（比如「K/V 切 64 列一块，循环内累加，寄存器里维护 running max」），编译器生成接近手写水平的 GPU 核。分界线画在：

- **通用热核**（MLA、GEMM、EP 通信）：手写，榨干每一滴带宽；
- **长尾与新结构核**（新算子原型、融合变体）：TileLang 写，快速验证性能上限，跑赢了再考虑要不要手工精修。

这套「手写保底 + DSL 提效」的组合，是算子团队能跟上模型结构快速迭代的方法论——第 01 章说「每一代模型的变化兑现成新算子」，TileLang 就是兑现的流水线。

## 5. miniflow 的算子层

miniflow 用一个三层结构把「算子层值得单独一章」这件事复刻出来：

```text
miniflow/tensor.py     NumPy 参考实现（清晰、可单测、行为基准）
csrc/kernel.cpp        C++ 后端（pybind11 编译为 miniflow._C）
                       ↳ BACKEND 探测：编译成功则派发 C++，否则回落 NumPy
```

三个教学点直接写在代码里：

1. **`matmul_naive` vs `matmul`**：同样的 O(M·K·N)，`ijk`（内积视角）与 `ikj`（外积视角）两种循环序在 CPU 上差一个量级——**循环序决定缓存局部性**。DeepGEMM 的分块是同一思想的 GPU 极致版：让每个数据块在被缓存期间被用尽可能多次；
2. **`fused_silu_swiglu`**：SwiGLU 的 `silu(gate) * up` 原本是三个算子（两次逐元素算子加一次乘法），中间张量要在内存里写读两遍；融合成一个核后一遍过——decode 访存受限，**省一次搬运就是纯赚**；
3. **`softmax` / `rms_norm`**：数值稳定的标准实现（减 max 再指数化）。FlashMLA 的在线 softmax 是它的分块推广——`tests/` 里可以验证两者数学等价。

```bash
pip install pybind11 && python setup.py build_ext --inplace
python -c "from miniflow.tensor import backend; print(backend())"
# 编译成功输出 cpp，否则 numpy —— 功能完全等价，只是性能不同
```

!!! warning "一个诚实的提醒"

    miniflow 的 C++ 核在 CPU 上**未必比 NumPy 快**——NumPy 底层是工业级 BLAS。这里的数值在于看清「同一数学的三种实现，工程差距从哪来」（循环序、融合、布局），不在跑分。GPU 上的 FlashMLA/DeepGEMM 与 CPU 上的 naive 实现，差距来源是同一组思想，只是幅度不同。

---

| 上一章 | 下一章 |
|---|---|
| [03 总体架构](03-architecture.md) | [05 并行策略：DP/EP/EPLB](05-parallelism.md) |