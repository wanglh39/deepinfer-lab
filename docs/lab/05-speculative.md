# Lab 05 推测解码

!!! info "本章对应"
    - 生产组件：DSpark（变长验证 + 置信度调度）
    - miniflow 代码：`miniflow/speculative.py`、`miniflow/engine.py`
    - 前置阅读：[08 推测解码](../guide/08-speculative.md)

---

## 1. 教学映射（保真机制，简化角色）

| DSpark 生产 | miniflow 教学 | 保真度 |
|---|---|---|
| 独立 draft 小模型 / MTP 头 | 主模型温度采样（draft_temperature=1.2） | 机制同、角色简 |
| Confidence Head 预测接受率 | draft token 在 softmax 分布中的概率 | 信号同源 |
| 变长验证（置信前缀） | `_verify_len`：阈值 0.35 的最长达标前缀 | 一比一 |
| 残差采样（分布不变） | **argmax 验证 + argmax 修正** | 可测试性优先 |
| KV 回滚 + 覆盖写 | `computed` 按层回退 | 一比一 |

`spec.stats` 输出 DSpark 论文的同款指标（demo 里直接对照生产数据）：

```text
steps=5  avg_verified_len=5.0  avg_accepted_len=4.8
accept_rate=0.96  tokens_per_step=5.0
```

## 2. draft 循环的不变量（本章核心）

整个 `SpeculativeDecoder.step()` 建立在一条不变量上：

> **进入每次 forward 时，恰有 1 个未算 token。**

因为「算最后一个 token」才能得到**下一位置**的分布 —— draft 的每一步
都是「forward 最后 token → 采样新 draft → append」。这条不变量贯穿
prefill 交接（prefill 采样 1 个 token 后 `computed = num_tokens - 1`）
与提交回滚（`kv_keep = base + len(keep) - 1`）。

!!! danger "踩坑实录：回滚量算错，不变量破坏"
    早期版本的回滚按分支推导：

    ```python
    if accepted < k_verify:
        kv_keep = base_len + accepted + 1   # bonus token 标记「已算」
    else:
        kv_keep = base_len + accepted       # 全接受：computed == num_tokens
    ```

    两个分支都错：bonus token **从未进过 forward**（它的 logits 来自
    `logits_list[accepted]`，不是自己算的）；全接受分支把 computed 推到
    num_tokens。下一步 draft 的第一次 forward `n = 0`，
    `transformer.py` 的增量契约断言当场拦截 —— **契约断言把一个
    「静默产出错值」的 bug 变成「fail-fast」**（[Lab 02 第 2 节](02-model.md)）。

    统一修复（一个公式覆盖全部分支）：

    ```python
    kv_keep = base_len + len(keep_tokens) - 1
    # bonus 分支：bonus token 无 KV ✓（留给下步 draft 第一次 forward 算）
    # 全接受分支：最后 1 个 keep token 的 KV 已写但标记未算 ——
    #            下步重算，确定性 forward 数值一致，无害
    ```

!!! danger "踩坑实录：一步提交多 token 越过 max_tokens"
    spec 一步净提交「接受长度 + bonus」个 token，可能**越过**
    max_tokens（基线 24 个，spec 输出了 25 个 —— demo 的对比断言
    前缀对齐所以没炸，但长度语义已错）。修复：提交前按剩余额度截断：

    ```python
    room = seq.sampling_params.max_tokens - hist_out_len
    if len(keep_tokens) > room:
        keep_tokens = keep_tokens[:room]
    ```

## 3. 验证与提交的完整逻辑

```python
# miniflow/speculative.py 的 step()（骨架）
logits_list = self._draft_logits(seq)          # draft：5 次「算1采1」
draft_tokens = seq.token_ids[base_len:]        # 假设的未来
confidences = [softmax(l)[t] for l, t in zip(logits_list, draft_tokens)]

k_verify = self._verify_len(confidences)       # 置信度达标的最长前缀
accepted = 0                                    # 命中 argmax 则接受
for i in range(k_verify):
    if int(np.argmax(logits_list[i])) == draft_tokens[i]:
        accepted += 1
    else:
        break

keep_tokens = draft_tokens[:accepted]
if accepted < k_verify:                         # 被拒 → argmax 修正
    keep_tokens.append(int(np.argmax(logits_list[accepted])))
# 截断到 max_tokens → 回滚 KV 进度（-1 不变量）→ 提交 output
```

**输出与纯 greedy 严格一致**的契约由「argmax 验证 + argmax 修正」
保证 —— `tests/test_speculative.py::test_spec_matches_greedy` 把它
钉死为回归测试。

## 4. KV 交互：预写未来再回滚

draft 阶段 k 个 token 的 KV **先写进池**；被拒槽位不擦除 —— 下次
forward 覆盖写（`paged_kv.append` 的 `k_pools[layer][slots] = k` 直接
赋值）。这依赖 [Lab 03](03-paged-kv.md) 的「按层进度 + 覆盖写」语义 ——
**KV 管理是推测解码的地基**。

## 5. 动手实验

```bash
python examples/demo_speculative.py
#   [基线]     steps = 24
#   [推测解码] steps = 6    ← 同样 24 个 token
#   输出与纯 greedy 完全一致: True
#   等效加速: 4.00x
#   accept_rate = 0.96（draft 与 argmax 的吻合度）

python -m pytest tests/test_speculative.py -q
#   spec == greedy（token 级）/ 步数更少 / stats 合理性
```

??? question "思考题"
    1. 把 `confidence_threshold` 从 0.35 提到 0.9：`avg_verified_len`
       与 `accept_rate` 各怎么动？吞吐呢？（变长验证的收益边界）
    2. 把 `draft_temperature` 降到 0.0（draft 变 greedy）：accept_rate
       会到多少？为什么此时 spec 与 greedy 的关系反而更「平凡」？
       （提示：draft == argmax 时每步全接受 —— 退化成「批量化 greedy」）
    3. 为什么 `bonus_tokens` 恰好说明「被拒也有产出」？没有 bonus 的
       speculative sampling（纯拒绝采样）单步最差产出几个 token？

| 上一章 | 下一章 |
|---|---|
| [Lab 04 调度器](04-scheduler.md) | [Lab 06 结构化输出](06-structured.md) |