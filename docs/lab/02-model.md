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

## 5. 验收清单

- [ ] `gradcheck.py` 相对误差 < 1e-4
- [ ] 训练 loss 收敛到 0.1 以下
- [ ] CLI 对训练语料的句子能风格续写
- [ ] （选做）改坏一处反向（比如重引入 Bug 4 的漏因子），确认 gradcheck 变红——亲眼看一次指纹，胜过读十遍教训

| 上一章 | 下一章 |
|---|---|
| [Lab 01 算子层](01-tensor.md) | [Lab 03 分页 KV 与前缀缓存](03-paged-kv.md) |