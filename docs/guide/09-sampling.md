# 09 采样：确定性与批次不变

!!! question "本章问题"
    - 同一请求换个「批内邻居」输出就变 —— 这在生产上意味着什么？
    - batch-invariant 采样要拆掉哪三处「批维泄漏」？
    - `seq_seed` 的随机流设计为什么能扛住重调度？

---

## 1. 问题：批组成是一个隐藏变量

常规实现的采样把整批 logits 拼矩阵统一处理：top-p 的截断顺序、
浮点求和顺序、批维归约 —— 全都让「同一条请求」的结果依赖批内
**其他请求**。换一批邻居、换个批大小、换个到达顺序 → 输出漂移。

生产上的真实代价：

| 场景 | 后果 |
|---|---|
| 回归测试 | 同输入不同输出 → 测试不可复现，红绿灯失灵 |
| A/B 实验 | 采样噪声混入策略对比 → 结论失真 |
| 分片重放 | 按请求分片重跑与在线结果对不齐 → 审计困难 |
| 故障定位 | 「昨天好好的今天变了」→ 排查方向被带偏 |

DeepInfer 把它当**正确性**问题处理（QA 里明确列为部署要点）。

## 2. 拆除三处批维泄漏

miniflow 的 `miniflow/sampler.py` 逐条拆解（注释即文档）：

```python
# 泄漏一：随机流
#   坏：批级共享 RNG —— 批大小变化 => 每条请求消费的随机数错位
#   好：每序列独立 RNG，种子 = [seq_seed, seq_id]
rng = np.random.default_rng([int(seed), int(seq.seq_id)])

# 泄漏二：归约顺序
#   坏：整批 logits 矩阵统一 softmax —— 浮点求和顺序随批形状变
#   好：逐请求独立归约（softmax/top-p 只在自己那行 logits 上做）

# 泄漏三：过滤截断
#   坏：top-p 按批排序截断 —— 邻居的分布影响我的候选集
#   好：top-k/top-p 截断只看本请求的 logits
```

还有一个精妙细节 —— **RNG 消费次数恒定**：每步采样恰好消费 1 个
随机数。即使某请求被抢占重调度、批被打散重组，它的随机流也**不错位**
—— 这是「重调度不变性」的最后一环。

## 3. 确定性组合拳

DeepInfer QA 的表述（miniflow 复刻为 README 的第一公式）：

```text
batch-invariant 推理 + 固定 seq_seed + 关闭推测解码
    => 相同输入 token 必得相同输出 token
```

三个开关各管一段：

- `batch-invariant`：批组成不进输出函数的变量集；
- `seq_seed`：随机流显式化（用户可指定，默认 0 —— 不留隐式状态）；
- 关闭推测解码：spec 的 draft 带额外随机性（教学版 argmax 验证
  例外，见 [第 08 章](08-speculative.md)）。

!!! note "greedy 是 batch-invariant 的免费特例"
    temperature=0 时输出 = argmax(logits)，纯函数无随机 —— 但只有
    **逐请求独立归约**做到位，浮点结果才真正与批形状无关。
    miniflow 的 `test_batch_composition_invariant` 同时测了 greedy
    与带温度两条路径。

## 4. 序列种子：为什么混入 seq_id

```python
np.random.default_rng([seed, seq_id])
```

同 seed 的两条请求（比如对照组实验）各自的流**互相独立** ——
不混入 seq_id 的话同 seed 请求会产出相同随机序列，反而引入
「内容相同」的伪相关。

`MF_BATCH_INVARIANT=0` 环境变量可关闭整套机制 —— 保留一个
「坏的对照组」用于演示批组成如何影响输出（教学项目的刻意设计，
对应 DeepInfer 的 `DEEPINFER_*` 环境变量开关体系）。

## 5. 与其他子系统的协议

| 交互方 | 协议 |
|---|---|
| 推测解码 | `sample_one(seq, logits)`：单序列路径，RNG 流与批量路径同源（不错位） |
| 结构化输出 | mask 在**采样前**施加（`-inf` 屏蔽非法 token）—— softmax 在掩码后归一化，合法路径的分布形状不被扭曲 |
| 抢占/重调度 | RNG 每步恒定消费 1 次 → 重调度后随机流位置不变 |

第三条常被忽视：请求被抢占 → 回 waiting → 重新 prefill → 继续
decode —— 中间经历任意次调度变化，它的随机流**依然从上次的位置继续**。
「逐请求随机流 + 恒定消费」两件小事合起来才是完整的重调度不变性。

## miniflow 对应实验

```bash
python examples/offline_inference.py
#   [seed=42]  两次运行 token 一致: True
#   [seed= 7]  换种子输出不同  : True

python -m pytest tests/test_batch_invariance.py -q
#   同 seed 可复现 / greedy 批组成不变 / 换 seed 输出变
```

!!! danger "测试中的一个真实发现"
    `test_seed_changes_output` 初版在 temperature=1.0 下断言「换 seed
    输出不同」—— 失败：400 步过拟合的小模型分布接近 one-hot，
    任何种子的采样都落在 argmax 上，seed 的作用**不可观测**。
    教训：确定性测试要确保被测变量真的进入了输出函数 —— 最终
    temperature 提到 2.5 摊平分布才测出差异。低温下「看起来确定」
    和「真的确定」是两回事。

| 上一章 | 下一章 |
|---|---|
| [08 推测解码](08-speculative.md) | [10 结构化输出](10-structured.md) |