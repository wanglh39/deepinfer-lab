# Lab 05 推测解码

!!! info "本章对应"
    - 生产组件：DSpark（变长验证 + 置信度调度）
    - miniflow 代码：`miniflow/speculative.py`、`miniflow/engine.py`
    - 前置阅读：[08 推测解码](../guide/08-speculative.md)

guide 08 讲了「先猜后验」的原理与分布不变性的数学。本章进代码：教学版怎么用一条**不变量**把 draft 循环、提交回滚、prefill 交接全部串起来——以及这条不变量被破坏时是什么样（真实踩坑）。

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

先推导它为什么成立。forward 的产出语义是：算掉序列的第 $n$ 个 token，得到**第 $n+1$ 个位置**的输出分布。反过来说：想采样第 $n+1$ 个 token，就必须 forward 到第 $n$ 个——**一个不多、一个不少**。0 个未算 token，forward 无活可干、无分布可产；2 个以上，等于把还没定的 token 当成已定去算了（它们此刻根本不存在）。

所以 draft 的每一步都是「forward 最后那个 token → 得到下一位置分布 → 采样新 draft → append」。这条不变量贯穿三个交接点：prefill 交接（prefill 算完 prompt 后采样第 1 个输出 token，此时 `computed = num_tokens - 1`——最后那个 token 的 KV 留给 spec 的第一次 forward）、draft 循环（每次恰好推进 1）、提交回滚（下节）。

!!! danger "踩坑实录：回滚量算错，不变量破坏"

    早期版本的回滚按分支各自推导：

    ```python
    if accepted < k_verify:
        kv_keep = base_len + accepted + 1   # bonus token 标记「已算」
    else:
        kv_keep = base_len + accepted       # 全接受：computed == num_tokens
    ```

    两个分支都错。bonus 分支：被拒位置的修正 token（bonus）**从未
    进过 forward**——它的 logits 来自验证批里 `logits_list[accepted]`
    （是被拒位置自己的分布），不是算它自己算出来的；把它标记「已算」
    违反不变量。全接受分支：`computed` 被推到 `num_tokens`，下步
    draft 的第一次 forward 发现 0 个未算 token。

    幸运的是这个 bug 没有静默发作——`transformer.py` 的增量契约断言
    （`assert n > 0`，[Lab 02 第 2 节](02-model.md)）当场把它拦下。
    **契约断言把「静默产出错值」变成「fail-fast」**。

    统一修复（一个公式覆盖全部分支，不需要再按分支推导）：

    ```python
    kv_keep = base_len + len(keep_tokens) - 1
    # bonus 分支：bonus token 无 KV ✓（留给下步 draft 第一次 forward 算）
    # 全接受分支：最后 1 个 keep token 的 KV 已写但标记未算 ——
    #            下步重算，确定性 forward 数值一致，无害
    ```

    注意「公式统一」本身就是修复的一部分：两个分支各自推导，就永远
    有推漏一个的风险；一条公式对所有分支成立，验证一次就够。

!!! danger "踩坑实录：一步提交多 token 越过 max_tokens"

    spec 一步净提交「接受长度 + bonus」个 token，可能**越过**
    max_tokens（基线 24 个 token 上限，spec 输出了 25 个——demo 的
    对比断言只查前缀对齐所以没炸，但长度语义已经错了）。修复：提交
    前按剩余额度截断：

    ```python
    room = seq.sampling_params.max_tokens - hist_out_len
    if len(keep_tokens) > room:
        keep_tokens = keep_tokens[:room]
    ```

    「批量推进」的每个环节都要重审「逐 token 时代」的边界检查——
    不变量换了，边界条件的执行时机也要跟着换。

## 3. 验证与提交的完整逻辑

被拒为什么还有 bonus 可拿，值得单独说：验证是**一次 forward 算整个 draft 批**，每个 draft 位置都有自己的 logits——第 $i$ 个被拒位置的 logits 描述的正是「第 $i+1$ 个位置该出什么」。它的 argmax 是一个白拿的修正 token（比重新走一步 forward 省一整步）。这是「被拒也有产出」的含义。

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
if accepted < k_verify:                         # 被拒 → argmax 修正（bonus）
    keep_tokens.append(int(np.argmax(logits_list[accepted])))
# 截断到 max_tokens → 回滚 KV 进度（-1 不变量）→ 提交 output
```

**输出与纯 greedy 严格一致**的契约由「argmax 验证 + argmax 修正」保证：接受的 token 等于目标的 argmax，修正 token 也是目标的 argmax——每一步产出都等于 greedy 会选的。`tests/test_speculative.py::test_spec_matches_greedy` 把它钉死为回归测试。

## 4. KV 交互：预写未来再回滚

draft 阶段 k 个 token 的 KV **先写进池**（它们当过「已定 token」参与过 forward）；被拒槽位不擦除——下次 forward 覆盖写（`paged_kv.append` 的 `k_pools[layer][slots] = k` 是直接赋值）。这依赖 [Lab 03](03-paged-kv.md) 的「按层进度 + 覆盖写」语义——**KV 管理是推测解码的地基**，回滚是改记账（`computed`），不是擦数据。

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
       与 `accept_rate` 各怎么动？吞吐呢？（变长验证的收益边界——
       对照 DSpark 生产数据：验证长度 5.0 → 3.4，接受长度只掉 0.09。）
    2. 把 `draft_temperature` 降到 0.0（draft 变 greedy）：accept_rate
       会到多少？为什么此时 spec 与 greedy 的关系反而更「平凡」？
       （提示：draft == argmax 时每步全接受——退化成「批量化 greedy」。）
    3. 没有 bonus 的 speculative sampling（纯拒绝采样）单步最差产出
       几个 token？有 bonus 呢？（提示：全拒时，前者 0、后者 1——
       验证 forward 的 logits 不能白花。）

| 上一章 | 下一章 |
|---|---|
| [Lab 04 调度器](04-scheduler.md) | [Lab 06 结构化输出](06-structured.md) |