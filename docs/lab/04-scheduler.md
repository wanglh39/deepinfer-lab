# Lab 04 调度器与三重死锁

!!! info "本章对应"
    - 生产组件：continuous batching / chunked prefill / 抢占 / 零开销批调度
    - miniflow 代码：`miniflow/scheduler.py`、`miniflow/engine.py`
    - 前置阅读：[07 调度器](../guide/07-scheduler.md)

这是全项目**最贵的一章**——miniflow 的调度器在开发中依次踩过三重死锁，每一重都是「正确性边界」的实物教材。修复后的 `schedule()` 只有约 60 行，但每一行都在讲一个踩过的坑。建议精读：三个死锁的根源各不相同（语义错误 / 时序窗口 / 顺序耦合），恰好覆盖并发资源系统 bug 的三大谱系。

## 0. 复现环境

```bash
python examples/demo_scheduler.py preempt
# 3 条请求 × (prompt+24) token，只给 10 块 × 4 slot 的 KV 池
# → preemptions = 4（被抢占比块回收重算，请求仍全部完成）
```

修复前这个演示的表现是**执行超时**——不是慢，是死循环。

## 1. 死锁一：容量检查的语义错误（自我抢占）

**症状**：单条序列独跑也永不完成——诊断脚本显示它周期性地 `num_computed` 归零（被抢）、重新 prefill、长到同一点、又被抢，无限循环。

**根源**：decode 的容量检查误用了「整序列所需块 ≤ 空闲块」。走一遍数字：

```python
# 错误语义：序列自己占的块不在空闲池里！
if not self.kv.can_allocate(seq.num_tokens + 1):   # 检查整序列需求
    preempt()
```

设总池 10 块、序列已长到占 5 块、空闲 5 块、下一 token 后全序列共需 6 块：检查算的是 `6 ≤ 5`？**否**——但那 5 个「空闲」块里并不包含它自己占着的 5 块，而它需要的 6 块里有 5 块**就是自己脚下踩着的**。判定「不够」→ 抢占自己 → 释放全部 5 块 → 重新 prefill → 长到同样的位置 → 同样的判定 → 循环。序列被自己挡住了。

**修复**：检查改用**增量语义**——只看「还要补几块」：

```python
need_new = blocks_needed(num_tokens + 1) - len(table)   # 增量：6 - 5 = 1
if need_new > free: preempt()
# 占 5 块、空闲 5 块、只需再补 1 块 → 1 ≤ 5 通过 ✓
```

**普适教训**：对「已持有部分资源的主体」做容量检查，分母里**绝不能包含它自己已占的部分**——数据库的事务锁管理、内存分配器的增长判断，全是同一条原则。

## 2. 死锁二：检查通过但分配滞后（同批双重花销）

修完死锁一，新症状出现：`MemoryError: 物理块耗尽` 从 engine 的 `ensure_capacity` 里炸出来——调度器明明检查过。

**根源**：组批循环里，同批 N 条序列的检查**读的是同一瞬间的空闲数**：

```text
空闲 1 块。B1 检查：需 1 块 ≤ 1 ✓。B2 检查：需 1 块 ≤ 1 ✓（还是那个 1！）
执行：B1 分配（空闲 0）→ B2 分配 → 炸
```

检查是「记账」，物理分配发生在其后的执行阶段——**窗口期里同批的邻居把富余花掉了**。注意这甚至不是多线程：单线程的组批循环里，多条目共享同一份过期读数，同样构成竞态。竞态的本质是「读到的是旧世界」，不是「有多个线程」。

**修复**：检查与分配**原子化**——调度期就把预留实体化为物理块分配（`ensure_append_capacity`：先算够不够，够就当场分配）：

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

修完死锁二，诊断脚本显示出**第三种零进展循环**，形态与前两个都不同：

```text
step 16: run=1（seq 0，占 5 块），wait=[1, 2]，free=5
step 17: preempt_total+1，free=5，seq 0 的 num_tokens 不再增长
step 18..59: 每步 preempt+1，状态完全不变
```

**根源推演**（逐条看一个 step 内部发生了什么）：

```text
waiting 段：seq 1（需 5 块）can_allocate(5≤5) ✓ → 接纳（吃光 5 块）
decode 段：seq 0 需追加 1 块，空闲 0 → 触发抢占
            victim = 「最晚进入 running 的」= 刚接纳的 seq 1 → 释放 5 块
下一步：   又接纳 seq 1 → 又抢占 seq 1 → 每步零进展
```

诡异之处在于每个策略**单独看都合理**：waiting 有富余就接纳（提高吞吐）、decode 缺块就抢占（保护在跑）、victim 选最晚到达（让老请求先走）。三个合理策略组合成死循环——调度顺序（先 waiting 后 decode）让新请求挤占了在跑请求，而 victim 规则恰好每次都选中这个新来者，系统就在「接纳-抢占」之间永远震荡。

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

**普适教训**：当多个策略共享一个资源池时，**它们的作用顺序本身就是正确性问题**——不是性能偏好。「先到先服务」和「保护在跑」都对，但谁先谁后决定了系统是收敛还是震荡。

## 4. 修复后的完整调度序

```text
1. decode 优先：逐条「增量检查 → 实体化分配」原子推进
   └ 容量不足 → 抢占最晚 running → 本步停止扩批（保守）
2. waiting 接纳（剩余容量）：
   ├ 前缀 match（命中块 = 已覆盖，只加引用不耗空闲）
   ├ 容量：blocks_needed(全序列) - 命中块 ≤ 空闲
   ├ 预算：need ≤ budget（无活可干时强制放行 ← 预算死锁的兜底）
   └ 实体化分配到 num_tokens + 1（为下一 token 留位）
3. liveness 防御：无可执行项 + 有等待者 → RuntimeError
   （单序列需求 > 整池是唯一的真死锁 —— 配置错误要报错，不要空转）
```

注意三个防御性设计各自来自真实事故：**预算强制放行**（被抢占序列的重算长度超过预算、永远出不了队——预算死锁）；**抢占后 break**（释放的块留给刚才失败的那条，若继续检查下一条会把块发给别人，刚才的失败者下一步又触发抢占——死锁三的变体）；**liveness raise**（池子绝对不够时明确报错而非无限重试）。

## 5. 异步流水线（AsyncLLMEngine）

「零开销批调度」的双线程复刻（`miniflow/engine.py`）：

```python
# 调度线程：schedule(N+1) → 投递队列（maxsize=2 背压）
# 执行线程：get → forward(N) → 对「调度时活着、执行时已 finish」
#           的序列防御性跳过（名单滞后一步 —— DeepInfer 同款约束，
#           见 guide/07 第 3 节「容忍过时信息 + 执行侧自治修正」）
```

两个实现细节值得单独说：

1. **时间基准必须统一**：两个线程各自调 `time.perf_counter()` 作为 t0，事件时间线的**坐标系就错位了**（线程启动延迟被算进所有事件）——`demo_scheduler.py async` 踩过这个坑，流水线重叠率算出来是 0%，修好 t0 后 12/15 个 schedule 完全落在 forward 区间内（前 3 个是流水线填充期的正常现象）；
2. **重叠强度的诚实说明**：CPU/NumPy 的 forward 受 GIL 约束，靠 NumPy 在 C 层释放 GIL 的窗口获得并行——**结构**与生产（GPU 是独立设备、天然并行）相同，**强度**不同。demo 的统计口径因此定义为「多少个 schedule 完全落在某个 forward 区间内」，而不是含糊的「快了多少」。

## 6. 动手实验

```bash
python examples/demo_scheduler.py        # cb / preempt / async 三幕
python -m pytest tests/test_scheduler.py tests/test_async.py -q
#   抢占 liveness（多次抢占全部完成，重算不丢 token）
#   插队请求与在跑请求共享解码步
#   死锁防御（池子绝对不够时 RuntimeError）
#   异步输出 == 同步输出（滞后名单的语义等价）
```

??? question "思考题"

    1. 把 `num_blocks` 从 10 调到 8：`preemptions` 怎么变？调到 30 呢？
       （先推理再验证——「恰好挤占」的区间在哪？）
    2. 把 `_preempt` 的 victim 改成「最早到达的」会怎样？
       （提示：老请求可能永远做不完——饥饿；但重算总量会减少——
       这是生产上 swap vs recompute 之外的第三个旋钮。）
    3. 为什么抢占后要 `break` 而不是继续检查下一条 running？
       （提示：释放的块要留给「刚才失败的那条」——继续检查会把块
       发给别人，刚才失败的序列下一步又触发抢占……）

| 上一章 | 下一章 |
|---|---|
| [Lab 03 分页 KV](03-paged-kv.md) | [Lab 05 推测解码](05-speculative.md) |