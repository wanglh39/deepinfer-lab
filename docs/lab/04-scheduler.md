# Lab 04 调度器与三重死锁

!!! info "本章对应"
    - 生产组件：continuous batching / chunked prefill / 抢占 / 零开销批调度
    - miniflow 代码：`miniflow/scheduler.py`、`miniflow/engine.py`
    - 前置阅读：[07 调度器](../guide/07-scheduler.md)

---

这是全项目**最贵的一章** —— miniflow 的调度器在开发中依次踩过三重
死锁，每一重都是「正确性边界」的实物教材。修复后的 `schedule()`
只有 ~60 行，但每行注释都在讲一个踩过的坑。

## 0. 复现环境

```bash
python examples/demo_scheduler.py preempt
# 3 条请求 × (prompt+24) token，只给 10 块 × 4 slot 的 KV 池
# → preemptions = 4（被抢占比块回收重算，请求仍全部完成）
```

修复前这个演示的表现是**沙箱执行超时** —— 不是慢，是死循环。

## 1. 死锁一：容量检查的语义错误（自我抢占）

**症状**：单条序列独跑也永不完成 —— 诊断脚本显示它周期性地
`num_computed` 归零（被抢）、重新 prefill、长到同一点、又被抢。

**根源**：decode 检查误用了「整序列所需块 ≤ 空闲块」：

```python
# 错误语义：序列自己占的块不在空闲池里！
if not self.kv.can_allocate(seq.num_tokens + 1):   # 整序列需求
    preempt()
# 序列占 5 块、空闲 5 块、总需求 6 块 → 判「不够」→ 抢占自己 →
# 释放全部 → 重新 prefill → 长到 6 块需求 → 又判「不够」→ 无限循环
```

**修复**：检查用**增量语义** —— 只看「还要补几块」：

```python
need_new = blocks_needed(num_tokens + 1) - len(table)   # 增量
if need_new > free: preempt()
# 占 5 块、空闲 5 块、只需再补 1 块 → 通过 ✓
```

**普适教训**：对「已持有部分资源的主体」做容量检查，分母里**绝不能
包含它自己已占的部分** —— 否则它会被自己挡住。

## 2. 死锁二：检查通过但分配滞后（同批双重花销）

修完死锁一后出现新症状：`MemoryError: 物理块耗尽` 从 engine 的
`ensure_capacity` 里炸出来 —— 调度器明明检查过。

**根源**：同批 N 条序列的检查**共享同一瞬时空闲数**：

```text
空闲 1 块。B1 检查：需 1 块 ≤ 1 ✓。B2 检查：需 1 块 ≤ 1 ✓（还是那个 1！）
执行：B1 分配（空闲 0）→ B2 分配 → 炸
```

检查是「记账」，分配发生在之后 —— **窗口期里别人把富余花掉了**。
即使单线程，循环内多条目共享同一读数也是竞态。

**修复**：检查与分配**原子化** —— 调度期就把预留实体化为物理块
（`ensure_append_capacity`：先算够不够，够就当场分配）：

```python
# miniflow/cache/paged_kv.py
def ensure_append_capacity(self, seq, extra=1) -> bool:
    needed = self.blocks_needed(seq.num_tokens + extra)
    if needed - len(table) > self.num_free_blocks:
        return False          # 失败不动任何状态
    while len(table) < needed:
        table.append(self._allocate_block())   # 当场实体化
    return True
```

## 3. 死锁三：组批顺序颠倒（接纳-抢占震荡）

修完死锁二后，诊断脚本显示**新的零进展循环**：

```text
step 16: run=1（seq 0，占 5 块），wait=[1, 2]，free=5
step 17: preempt_total+1，free=5，seq 0 的 num_tokens 不再增长
step 18..59: 每步 preempt+1，状态完全不变
```

**根源推演**（每步内部）：

```text
waiting 段：seq 1（需 5 块）can_allocate(5≤5) ✓ → 接纳（吃光 5 块）
decode 段：seq 0 需追加 1 块，空闲 0 → 抢占「最晚进入 running 的」
           = 刚接纳的 seq 1 → 释放 5 块
下一步：   又接纳 seq 1 → 又抢占 seq 1 → 每步零进展
```

调度顺序（先 waiting 后 decode）让**新请求挤占了在跑请求** ——
而抢占的 victim 选择（最晚 running）恰好每次都选中这个新来者。
两个「各自合理」的策略组合成死循环。

**修复**：**decode 优先组批，剩余容量再接纳 prefill**（vLLM 同款顺序）：

```python
# miniflow/scheduler.py（修复后的顺序）
# 1) RUNNING decode：逐条「检查-分配」原子推进
for seq in list(self.running):
    if not self.kv.ensure_append_capacity(seq, 1):
        out.preempted.extend(self._preempt())
        break
    out.items.append(ScheduledSeq(seq, is_prefill=False))
# 2) WAITING 接纳：用 decode 预留后的真实剩余容量
...
```

**普适教训**：当两个策略（接纳策略 + 抢占策略）共享一个资源池，
**顺序本身就是正确性问题** —— 不是性能偏好。

## 4. 修复后的完整调度序

```text
1. decode 优先：逐条「增量检查 → 实体化分配」原子推进
   └ 容量不足 → 抢占最晚 running → 本步停止扩批（保守）
2. waiting 接纳（剩余容量）：
   ├ 前缀 match（命中块 = 已覆盖，只加引用不耗空闲）
   ├ 容量：blocks_needed(全序列) - 命中块 ≤ 空闲
   ├ 预算：need ≤ budget（无活可干时强制放行 ← 预算死锁的兜底）
   └ 实体化分配到 num_tokens + 1（为下一 token 留位）
3. liveness 防御：无可执行项 + 有等待者 → RuntimeError（配置错误
   要报错，不要空转 —— 单序列需求 > 整池是唯一的真死锁）
```

注意三个「防御性设计」都来自真实事故：预算放行（被抢序列重算超预算
出不了队）、抢占后过滤执行名单（victim 已入列的窗口期）、
liveness raise（池子绝对不够）。

## 5. 异步流水线（AsyncLLMEngine）

调度线程与执行线程的双线程复刻（`miniflow/engine.py`）：

```python
# 调度线程：schedule(N+1) → 投递队列（maxsize=2 背压）
# 执行线程：get → forward(N) → 对「调度时活着、执行时已 finish」
#           的序列防御性跳过（名单滞后一步 —— DeepInfer 同款约束）
```

两个实现细节值得注意：

1. **时间基准必须统一**：两个线程各自 `time.perf_counter()` 会让
   事件时间线**坐标系错位**（线程启动延迟被算进所有事件）——
   共享同一个 `t0`（`demo_scheduler.py async` 踩过，重叠率算出 0%）；
2. **重叠的可观测性**：CPU/NumPy 的 forward 受 GIL 约束，靠 NumPy
   C 层释放 GIL 的窗口并行 —— 结构与生产（GPU 独立设备）相同，
   并行「强度」不同（demo 的诚实说明）。

## 6. 动手实验

```bash
python examples/demo_scheduler.py        # cb / preempt / async 三幕
python -m pytest tests/test_scheduler.py tests/test_async.py -q
#   抢占 liveness（4 次抢占全部完成，重算不丢 token）
#   插队请求与在跑请求共享解码步
#   死锁防御（池子绝对不够时 RuntimeError）
#   异步输出 == 同步输出（滞后名单的语义等价）
```

??? question "思考题"
    1. 把 `num_blocks` 从 10 调到 8：`preemptions` 怎么变？调到 30 呢？
       （先推理再验证 —— 「恰好挤占」的区间在哪？）
    2. 把 `_preempt` 的 victim 改成「最早到达的」会怎样？
       （提示：老请求可能永远做不完 —— 饥饿；但也会减少重算总量
       —— 这就是生产上 swap vs recompute 之外的第三个旋钮）
    3. 为什么抢占后要 `break` 而不是继续检查下一条？
       （提示：释放的块要留给「刚才失败的那条」—— 继续检查会把
       块发给别人，刚才失败的序列下一步又触发抢占……）

| 上一章 | 下一章 |
|---|---|
| [Lab 03 分页 KV](03-paged-kv.md) | [Lab 05 推测解码](05-speculative.md) |