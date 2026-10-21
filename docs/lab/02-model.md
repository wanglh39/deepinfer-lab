# Lab 02 模型层与手写反向传播

!!! info "本章对应"
    - 生产组件：模型结构（MLA/MoE/SWA 的教学形态）
    - miniflow 代码：`miniflow/model/`、`tools/train.py`、`tools/gradcheck.py`
    - 前置阅读：[04 内核层](../guide/04-kernels.md)、[05 并行策略](../guide/05-parallelism.md)

本章两件事：看清教学模型的结构（一个「缩微版 DeepSeek」），以及**手写反向传播**——不用 autograd，逐层推梯度。后者附赠本 lab 最有价值的内容：四个真实的反向传播 bug 全实录，每一个都是经典陷阱的范本，每一个都由数值校验抓住。

## 1. 一个「缩微版 DeepSeek」的形态

`miniflow/config.py` 的 `ModelConfig` 声明了教学模型的结构异构性：

| 结构 | 配置 | DeepSeek 对应 |
|---|---|---|
| GQA | `num_kv_heads=2 < num_attention_heads=4` | MLA 的「KV 头数少于 Q 头数」精神 |
| SWA 层 | `swa_layers=(2, 3)`，窗口 16 | V4.1 的稀疏注意力层 |
| MoE 层 | `moe_layers=(1, 3)`，4 专家 top-2 | V4.1 的专家层 |
| dense 层 | 其余层用普通 FFN | deepseek 系的 dense/MoE 混排 |
| RMSNorm | eps 1e-5 | 全系同款 |

**结构异构是并行策略按层混搭的物质基础**（[guide/05](../guide/05-parallelism.md)）。落到代码上，就是 transformer 层循环里按 config 分派：

```python
# miniflow/model/transformer.py 的层循环（骨架）
for layer_idx in range(cfg.num_layers):
    x = rms_norm(x, ...)
    x = attention(x, ..., window=cfg.swa_window if cfg.uses_swa(layer_idx) else None)
    x = rms_norm(x, ...)
    x = moe_ffn(x, ...) if cfg.uses_moe(layer_idx) else dense_ffn(x, ...)
```

## 2. 增量前向契约

引擎侧的模型接口是**增量**的——它是 paged KV 的直接用户（[Lab 03](03-paged-kv.md)）：

```python
logits = model.forward(seq.token_ids, seq.seq_id, kv)
# 只计算 [kv.computed[seq][0], len(token_ids)) 的新 token：
#   读旧 KV → 算新 token 的注意力 → 写新 KV → 返回最后位置的 logits
```

为什么必须增量：decode 每步只多 1 个 token，重算整个前缀是 $O(t^2)$ 的浪费；chunked prefill 每步只推进一小段；推测解码验证时一批推进几个。所有这些模式共享同一个契约——「已算过的部分直接读 KV，只算新的」。

`transformer.py` 开头的断言是这条契约的守卫：

```python
assert n > 0, "forward 需要至少一个新 token（增量契约）"
```

!!! tip "这个 assert 后来真抓到了 bug"

    [Lab 05](05-speculative.md) 的推测解码回滚量算错——bonus token 被
    标记「已算」但从未 forward 过。下一次 draft 调用 n=0，被这条断言
    当场拦下。**契约断言是最便宜的防御**——比起静默产出错值，fail-fast
    把定位时间从小时级压到分钟级。

## 3. 手写反向传播：四个真实 bug

`tools/train.py` 不用 autograd——注意力、RMSNorm、RoPE、SwiGLU、GQA、MoE 的反向全部手推。训练 400 步 loss 4.23 → 0.07（125 秒，152256 参数，纯 CPU）。

先立方法论：**没有数值校验，手写反向几乎必然错，而且错了看不出来**——loss 照样下降（梯度方向大致对、幅度错），只是收敛慢或收敛到坏点。`tools/gradcheck.py` 用中心差分做数值梯度（float64），与解析梯度逐参数对照。以下四个 bug 全部由它抓住。

### 3.0 热身：一次完整的链式法则推导

在进入 bug 之前，值得完整推一遍一个组件的反向——体会「手推」到底在推什么（以 RMSNorm 为例，它最短但要素齐全）。

前向（$x \in \mathbb{R}^H$，对一个 token）：

$$
r = \frac{1}{\sqrt{\frac{1}{H}\sum_i x_i^2 + \epsilon}}, \quad y_i = x_i \cdot r \cdot g_i
$$

（$g$ 是可学的缩放，$\epsilon$ 防零。）已知 $dy$（上游梯度），求 $dx$ 和 $dg$。

**对 $g_i$**（最简单，$y_i$ 线性依赖于 $g_i$）：

$$
\frac{\partial y_i}{\partial g_i} = x_i r \quad\Rightarrow\quad dg_i = dy_i \cdot x_i r
$$

**对 $x_j$**（$x_j$ 出现在两个地方：分子 $x_j r$ 和 $r$ 的定义里）：

$$
\frac{\partial y_i}{\partial x_j} = \underbrace{\delta_{ij} \cdot r \cdot g_i}_{\text{直接路径}} + x_i g_i \cdot \frac{\partial r}{\partial x_j}
$$

算 $\partial r/\partial x_j$（$r = s^{-1/2}$，$s = \frac{1}{H}\sum x^2 + \epsilon$）：

$$
\frac{\partial r}{\partial x_j} = -\frac{1}{2} s^{-3/2} \cdot \frac{2 x_j}{H} = -\frac{x_j r^3}{H}
$$

代回去，两侧同乘 $dy_i$ 并对 $i$ 求和：

$$
dx_j = dy_j \cdot r \cdot g_j - \frac{x_j r^3}{H} \sum_i dy_i \, x_i g_i
$$

翻译成 NumPy（`tensor.py` / `train.py` 里的 `_rms_bwd`）：

```python
def rms_norm_bwd(x, g, dy, eps):
    r = 1.0 / np.sqrt(np.mean(x*x, -1, keepdims=True) + eps)
    dg = dy * x * r                       # 对 g：直接路径
    inner = np.sum(dy * x * g, -1, keepdims=True)
    dx = dy * r * g - x * r**3 / x.shape[-1] * inner
    return dx, dg
```

五行代码，每行对应推导的一步——**这就是「手写反向」的全部含义：把链式法则从纸上搬到代码，每一步都可核对**。生产的 autograd 做的就是这件事的自动化（记录前向的计算图、反向时自动套链式法则）——**手推一遍的意义在于：出 bug 时你知道梯度从哪来，而不是对着 autograd 的报错发呆**。

（注意力/RoPE/MoE 的反向推导更长但方法相同——注意力要处理 softmax 的雅可比（$dp_i = p_i(dy_i - \sum_j p_j dy_j)$，这个形式值得单独记）、GQA 要处理「KV 头被多个 Q 头共享」的梯度累加。`train.py` 里都有，建议至少读懂 RMSNorm 和 softmax 这两段。）

### 3.1 gradcheck 的方法论：为什么中心差分 + float64

数值梯度的原理是中心差分：

$$
\frac{\partial f}{\partial x} \approx \frac{f(x+h) - f(x-h)}{2h}
$$

两个工程参数的选择都有讲究：

**为什么中心差分而不是单侧**（$\frac{f(x+h)-f(x)}{h}$）？误差项分析：单侧的截断误差是 $O(h)$（一阶），中心是 $O(h^2)$（二阶）——h 取 1e-6 时单侧误差 ~1e-6、中心 ~1e-12。**免费的精度提升一百万倍**，代价只是多算一次 f。

**为什么 h 取 1e-6**？两个反向力量的平衡点：h 太大 → 截断误差（$O(h^2)$）主导；h 太小 → **舍入误差**主导（$f(x+h)$ 和 $f(x-h)$ 在浮点里可能相等——差被量化吞掉，分子变噪声）。float64 下相邻数的间隔约 1e-16（相对），f 的量级 ~1 时 h=1e-6 的量化噪声是 1e-16/1e-6 = 1e-10——远小于截断误差 1e-12？不对，1e-10 > 1e-12——**h=1e-6 时舍入其实略占主导**，最优 h 在 $\sqrt[3]{1\text{e-}16} \approx 1\text{e-}5$ 附近（浮点梯度校验的经典结果 $h^* \sim \sqrt[3]{\epsilon_{machine}}$）。1e-6 和 1e-5 都在可用区间——这个数量级推理值得会：**任何数值方法的参数都应该能推出来，而不是背常数**。

**为什么全程 float64**——Bug 2 的完整故事在下面，但先给结论：校验工具的精度必须显著高于被校验对象的精度（校验 float32 的梯度要 float64 的基准——**裁判的分辨率要高于选手**，否则看到的全是裁判自己的噪声）。

### Bug 1：tied embedding 的梯度方向搞反

前向：`logits = xf @ embed.T`（embed 形状 [V, H]，输出层与输入层共享权重，即 tied embedding）。对 x 反向：

```python
# 错：dx[:, :-1] = dlogits @ w["embed"].T      # 形状对，语义错！
# 对：dx[:, :-1] = dlogits @ w["embed"]
```

为什么直觉会骗人，值得把链式法则写开。前向是 $\text{logits} = x \cdot W^\top$（W 即 embed，[V, H]）。对 $x$ 求导：$\frac{\partial\,\text{logits}}{\partial x} = W$，所以 $dx = d\text{logits} \cdot W$——**不带转置**。直觉的「反向就该转置」来自「对权重求导」的情形（$dW = dx^\top \cdot d\text{logits}$ 这种），而这里是对输入求导。更糟的是形状检查失效：`dlogits @ embed.T` 和 `dlogits @ embed` 在 tied 权重下形状**恰好都能对上**（H 维和 V 维互换了位置但赋值目标切片放过了它），NumPy 不报任何错。

!!! note "教训"
    **形状正确 ≠ 语义正确**。tied 权重让两个本该不同的矩阵变成一个，
    「转置与否」的判断不能再靠形状兜底——只能靠把链式法则写出来。

### Bug 2：float32 量化抹掉梯度校验的分母

```python
logits = (...).astype(np.float32)   # ← 把 float64 的前向结果压回 float32
```

数值梯度的原理是中心差分 $\frac{f(x+h) - f(x-h)}{2h}$，h 取 ~1e-6。若 f 本身被量化到 float32（相邻可表示数的间隔 ~1e-7 量级），那么 $f(x+h)$ 与 $f(x-h)$ 的差**可能整个落在量化噪声里**——分子是噪声，除出来的「梯度」是垃圾。表现极具迷惑性：有时全对（噪声恰好小）、有时全错（噪声恰好大），稳定性极差。修复：gradcheck 的前向全程保持 float64，权重手动转 float64 再校验。

!!! note "教训"
    混合精度训练的正确性校验必须**全程 float64 基准**——校验工具
    自己先要不受量化污染，否则它校验的是噪声不是梯度。

### Bug 3：CE 损失漏除 batch 总数（18.27 倍的指纹）

数值梯度与解析梯度在 **target 位置**的比值恰好是 **18.27 ≈ 1 + 1/18**，而 18 正是 batch 里的总 token 数。这个数字直接暴露了病灶：交叉熵的 one-hot 项（`np.put_along_axis` 减去的那个 1.0）没有除以 B·(T-1)——target 位置的梯度比正确值大了一个「未归一化的 1/N」项，非 target 位置正确，于是整体比值恰为 $1 + 1/N$。

!!! note "教训"
    **比值是结构性的指纹**：全错 → 推导错了；只有特定位置错、且比值
    恰为 (1 + 1/N) → 归一化遗漏。先看错误的模式再动手改代码——
    数字会告诉你它想说什么。

### Bug 4：SwiGLU 乘积反向漏因子（w1 全错、w2/w3 全对）

SwiGLU 前向：`act = silu(gate) * up`，其中 gate/up 是两个不同权重矩阵的输出。对 gate 侧反向：

```python
# 错：dgate = dact * silu'(gate)              # 漏了 up！
# 对：dgate = dact * silu'(gate) * up         # 乘积路径互为对方的因子
#     dup   = dact * silu(gate)
```

法则本身很简单：$f = g(u) \cdot v$ 对 $u$ 求导**必须带上 $v$**（$\partial f/\partial u = g'(u)\, v$）。漏掉的原因是手推时注意力全被 `silu'`（自己这条路径的导数）吸走了，忘了另一条路径是**乘法因子**。诊断特征极具辨识度：`w1`（gate 侧权重）的梯度全错，`w2`/`w3` 全对——「只有一侧错」直接指向乘积结构里漏了另一侧的因子。

!!! note "教训"
    定位路径：按层二分（0 层隔离头部 / 1 层隔离单层）→ 组件级数值
    校验（`_rms_bwd`/`_rope`/`_silu_grad` 单独测）→ **符合模式的局部
    错误**直接指认结构性遗漏。三个手段层层收窄，没有一个需要「盯着
    代码找」。

## 4. 训练与验证流程

```bash
# 训练（约 2 分钟）
python tools/train.py --out checkpoints/tiny --steps 400

# 数值梯度校验（中心差分，float64 —— Bug 2 的教训写进了实现）
python tools/gradcheck.py
#   max relative error: 7.88e-05   ← 通过阈值

# 快速冒烟：CLI 端到端
python -m miniflow.entrypoints.cli \
    --checkpoint checkpoints/tiny.npz \
    --tokenizer checkpoints/tiny.tok.json --prompt "the cat"
```

### 4.1 训练曲线的读法

`train.py` 每步打印 loss——一条健康的曲线长什么样、各阶段意味着什么（值得对照你的实际输出看）：

```text
step    loss     阶段含义
0       4.23     ln(66)≈4.19 —— 均匀分布的理论值（初始 loss ≈ ln(词表)）
                 ↑ 这条基线值得记住：初始 loss 离 ln(V) 很远 = 初始化有偏
50      2.8      快降段：学到了高频词的边际分布（"the/cat" 类）
150     1.2      慢降段：开始学到 bigram 结构（"the cat" 后接什么）
300     0.3      记忆段：过拟合开始（152K 参数 vs 小语料）
400     0.07     深度过拟合 —— 对教学目标是"成功"（风格续写要的就是记忆）
```

**教学模型的目标是过拟合**（风格续写是记忆的展示）——这和生产的训练目标（泛化）相反。所以「loss 0.07」在教学语境是勋章、在生产语境是事故——**指标的意义由目标定义**，这句话对读任何训练报告都适用。

### 4.2 一个观察实验：错误梯度的「隐性危害」

Bug 3（CE 漏除 batch 数）会让梯度整体放大 ~18 倍——但训练**照样收敛**（学习率除以 18 就等效了）。这值得做个实验亲眼看：

```bash
# 备份正确版，故意重引入 Bug 3（去掉归一化）：
#   在 train.py 的 CE 反向里注释掉 / (B*(T-1))
python tools/train.py --out checkpoints/bug3 --steps 400
# 观察两条曲线：bug 版 loss 更早发散或更慢收敛（取决于学习率）
```

预期看到三种结局之一：学习率不适配 → 前几十步 loss 爆炸（梯度太大）；碰巧适配 → 收敛但路径不同；完全等效（学习率自补偿）。**无论哪种，「loss 还在降」都不能证明梯度正确**——这就是 gradcheck 存在的理由：**收敛性是梯度的弱检验，数值一致性才是强检验**。

## 5. 验收清单

- [ ] `gradcheck.py` 相对误差 < 1e-4
- [ ] 训练 loss 收敛到 0.1 以下
- [ ] CLI 对训练语料的句子能风格续写
- [ ] （选做）改坏一处反向（比如重引入 Bug 4 的漏因子），确认 gradcheck 变红——亲眼看一次指纹，胜过读十遍教训
- [ ] （选做）3.0 节的 RMSNorm 反向：遮住 `train.py` 的 `_rms_bwd`，自己从推导写出五 NumPy 行，gradcheck 验证

## 6. 本章收获的自检

1. 增量前向契约是什么（只算 `[computed, len)` 的新 token）？哪条断言守护它？
2. 手写反向为什么必须配数值校验（loss 下降是弱检验——方向大致对的错误梯度照样收敛）？
3. 四个 bug 的「指纹」各是什么（形状对语义错/量化噪声/18.27 倍比值/单侧全错）？
4. 中心差分的 h 为什么是 1e-6（截断 vs 舍入的平衡）？为什么必须 float64（裁判分辨率高于选手）？

## 7. 给进阶者：从教学模型到 DeepSeek 的距离

诚实标注教学模型和生产模型的差距（免得「手写反向 + 125 秒训练」给人错误的规模感）：

| 维度 | miniflow | DeepSeek-V3 | 差距含义 |
|---|---|---|---|
| 参数量 | 152K | 671B | 9 个数量级——教学模型连玩具都算小 |
| 层数 | 4 | 61 | 结构异构（dense/MoE/SWA 混排）同款但层数差 15 倍 |
| 注意力 | GQA 2:4 | MLA 128 头 | guide/04 的吸收数学未实现（GQA 是简化近似） |
| 训练数据 | ~KB 语料 | 14.8T token | 教学是「记忆」生产是「泛化」 |
| 训练算力 | 笔记本 2 分钟 | 千卡月级 | —— |

但**机制的覆盖度**是另一张表：增量前向、KV 契约、结构异构、手推反向、数值校验——全部同构。**教学的价值在机制的完整性，不在规模的相似性**——就像解剖课用青蛙不用人，学的是神经系统不是体型。

| 上一章 | 下一章 |
|---|---|
| [Lab 01 算子层](01-tensor.md) | [Lab 03 分页 KV 与前缀缓存](03-paged-kv.md) |