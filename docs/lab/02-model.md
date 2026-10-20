# Lab 02 模型层与手写反向传播

!!! info "本章对应"
    - 生产组件：模型结构（MLA/MoE/SWA 的教学形态）
    - miniflow 代码：`miniflow/model/`、`tools/train.py`、`tools/gradcheck.py`
    - 前置阅读：[04 内核层](../guide/04-kernels.md)、[05 并行策略](../guide/05-parallelism.md)

---

## 1. 一个「缩微版 DeepSeek」的形态

`miniflow/config.py` 的 `ModelConfig` 声明了教学模型的结构异构性：

| 结构 | 配置 | DeepSeek 对应 |
|---|---|---|
| GQA | `num_kv_heads=2 < num_attention_heads=4` | MLA 的「KV 头数少于 Q 头数」精神 |
| SWA 层 | `swa_layers=(2, 3)`，窗口 16 | V4.1 的稀疏注意力层 |
| MoE 层 | `moe_layers=(1, 3)`，4 专家 top-2 | V4.1 的专家层 |
| dense 层 | 其余层用普通 FFN | deepseek 系的 dense/MoE 混排 |
| RMSNorm | eps 1e-5 | 全系同款 |

**结构异构是并行策略按层混搭的物质基础**（[guide/05](../guide/05-parallelism.md)）
—— 一个 config 字段（`uses_swa`/`uses_moe` 的层判断）在 transformer
的层循环里分派到不同的 forward 实现：

```python
# miniflow/model/transformer.py 的层循环（骨架）
for layer_idx in range(cfg.num_layers):
    x = rms_norm(x, ...)                        # 前置 norm
    x = attention(x, ..., window=cfg.swa_window if cfg.uses_swa(layer_idx) else None)
    x = rms_norm(x, ...)
    x = moe_ffn(x, ...) if cfg.uses_moe(layer_idx) else dense_ffn(x, ...)
```

## 2. 增量前向契约

引擎侧的模型接口是**增量**的（paged KV 的用户，见
[Lab 03](03-paged-kv.md)）：

```python
logits = model.forward(seq.token_ids, seq.seq_id, kv)
# 只计算 [kv.computed[seq][0], len(token_ids)) 的新 token：
#   读旧 KV → 算新 token 的注意力 → 写新 KV → 返回最后位置的 logits
```

`transformer.py` 开头的断言是这条契约的守卫：

```python
assert n > 0, "forward 需要至少一个新 token（增量契约）"
```

!!! tip "这个 assert 后来真抓到了 bug"
    [Lab 05](05-speculative.md) 的推测解码回滚量算错 —— bonus token
    被标记「已算」但从未 forward 过。下一次 draft 调用 n=0，被这条
    断言当场拦下。**契约断言是最便宜的防御** —— 比起静默产出错值，
    fail-fast 把定位时间从小时级压到分钟级。

## 3. 手写反向传播：四个真实 bug

`tools/train.py` 不用 autograd —— 逐层手写梯度（教学核心：注意力、
RMSNorm、RoPE、SwiGLU、GQA、MoE 的反向都要自己推）。训练 400 步
loss 4.23 → 0.07（125 秒，152256 参数）。

**没有数值校验，手写反向几乎必然错** —— `tools/gradcheck.py`
（中心差分，float64）是第二个核心工具。以下是它抓到的四个 bug，
每一个都是「反向传播的经典陷阱」范本：

### Bug 1：tied embedding 的梯度方向搞反

```python
# 错：dx[:, :-1] = dlogits @ w["embed"].T      # 形状对，语义错！
# 对：dx[:, :-1] = dlogits @ w["embed"]
```

前向是 `logits = xf @ embed.T`（embed 形状 [V, H]，tied lm_head）。
反向对 **x** 求导时链式法则带的是 `W`（不是 `W.T`）—— 直觉的
「转置配对」在这里骗人：形状恰好能对上（V 维吞掉了），NumPy 不报错。

!!! note "教训"
    **形状正确 ≠ 语义正确**。tied 权重让两个本该不同的矩阵变成一个，
    「转置与否」的判断不能再靠形状。

### Bug 2：float32 量化抹掉梯度校验的分母

```python
logits = (...).astype(np.float32)   # ← 把 float64 的前向结果压回 float32
```

数值梯度用中心差分 `(f(x+h) - f(x-h)) / 2h`，h ~ 1e-6 —— 若 f 本身
被量化到 float32（精度 ~1e-7），**差分分子全是量化噪声**，校验的
相对误差分母被抹掉，全部「看起来对」或全部「看起来错」。
修复：前向保持 float64，gradcheck 里手动把权重转 float64。

!!! note "教训"
    混合精度训练的正确性校验必须**全程 float64 基准** —— 校验工具
    自己先要不受量化污染。

### Bug 3：CE 损失漏除 batch 总数（18.27 倍的指纹）

数值梯度与解析梯度在 **target 位置** 的比值恰好是 **18.27 ≈ 1 + 1/18**
（18 = batch 里的总 token 数）。这不是巧合 —— one-hot 项 `np.put_along_axis`
减去的 1.0 没除以 B·(T-1)，导致 target 位置梯度大了一个「没归一化的
1/BT」项。

!!! note "教训"
    **比值是结构性的指纹**：全错 → 推导错误；只有特定位置错且比值
    恰为 (1 + 1/N) → 归一化遗漏。先看比值模式再改代码。

### Bug 4：SwiGLU 乘积反向漏因子（w1 全错、w2/w3 全对）

`act = silu(gate) * up` 的反向：

```python
# 错：dgate = dact * silu'(gate)              # 漏了 up！
# 对：dgate = dact * silu'(gate) * up         # 乘积路径互为对方的因子
#     dup   = dact * silu(gate)
```

**乘积路径的链式法则必须互为对方因子** —— `f = g(u) * v` 对 u 求导
必带 v。诊断特征极具辨识度：`w1`（gate 侧）全错，`w2`/`w3` 全对 ——
「只有一侧错」直接指向乘积结构里漏了另一侧。

!!! note "教训"
    按层二分（0 层隔离头部 / 1 层隔离单层）+ 组件级数值校验
    （`_rms_bwd`/`_rope`/`_silu_grad` 单独测）+ **符合模式的局部错误**
    → 直接定位结构性遗漏。

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
- [ ] （选做）改坏一处反向（比如重引入 Bug 4），确认 gradcheck 变红

| 上一章 | 下一章 |
|---|---|
| [Lab 01 算子层](01-tensor.md) | [Lab 03 分页 KV 与前缀缓存](03-paged-kv.md) |