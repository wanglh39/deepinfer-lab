# Lab 03 分页 KV 与前缀缓存

!!! info "本章对应"
    - 生产组件：PagedAttention / 3FS 块级 KV / 前缀缓存
    - miniflow 代码：`miniflow/cache/paged_kv.py`、`miniflow/cache/prefix_cache.py`
    - 前置阅读：[06 KV Cache](../guide/06-kvcache.md)

guide 06 讲了分页、共享、按层推进的原理。本章进代码：PagedKVCache 的物理结构长什么样、引用计数的三条路径怎么走、以及两个真实踩坑——「缓存条目依赖源序列的块表」和「先挂载后检查」，后者是引用泄漏的范本。

lab 章的方法：**每个机制先看数据结构（长什么样），再走数字例子（怎么转），最后看事故现场（怎么坏）**。KV 管理是全书状态最重的组件——它的代码量不大（两个文件 ~400 行），但每行都在管「谁拥有什么」——这正是 guide/12 原则三说的「最难的是生命周期」的具体化。

## 1. PagedKVCache 的物理结构

```python
# 物理池（生产上是 HBM，这里是一组 NumPy 数组，层间独立）
k_pools = [np.zeros((num_slots, kv_heads, head_dim)) for _ in range(L)]
#          num_slots = num_blocks * block_size

free_blocks: List[int]                  # 空闲块号栈
ref_counts: Dict[int, int]              # 物理块 → 引用数（共享的基石）
block_tables: Dict[int, List[int]]      # seq_id → 块表（逻辑→物理）
computed: Dict[int, List[int]]          # seq_id → 每层各自的已算 token 数
```

四份数据各管一件事——值得逐个说清「为什么是这四份、缺一份会怎样」：

| 数据 | 管什么 | 缺了会怎样 |
|---|---|---|
| `k_pools`（物理池） | 字节本身 | 没有存储（显然） |
| `free_blocks`（空闲栈） | 哪些块可分配 | 分配器不知道找谁（每次 O(N) 扫描） |
| `ref_counts`（引用计数） | 一个块被几方持有 | 共享块被第一方释放后Second方读到悬空数据 |
| `block_tables`（块表） | 每序列的逻辑→物理映射 | 序列不知道自己的 KV 在哪 |
| `computed`（按层进度） | 每序列每层算到哪 | 断点续算/回滚全部失灵（增量契约无锚点） |

**栈式空闲表**（`free_blocks` 用 List 的 pop/append）值得一提：后释放的块先被复用（LIFO）——刚释放的块大概率还在 CPU 缓存里（生产上对应「刚释放的页还在 TLB/缓存里」），**分配策略本身是一层微观的局部性优化**。对比空闲链表（FIFO）：实现同样简单但没有这个缓存亲和。小选择背后有理由——读生产代码时值得对每个小选择都问一句「为什么」。

「逻辑 token 位置 → 物理池偏移」的翻译一行讲清：

```text
slot = table[logical // block_size] * block_size + logical % block_size
```

用具体数字走一遍（block_size=4，某序列的块表 = [7, 3, 11]）：逻辑位置 9 → 第 `9 // 4 = 2` 个块（即块 11）内偏移 `9 % 4 = 1` → 物理槽位 `11*4 + 1 = 45`。

再走两个边界位置加固这个翻译（自己先算再对答案）：

```text
逻辑位置 0  → 块表[0]=块 7，偏移 0 → 槽位 28
逻辑位置 4  → 块表[1]=块 3，偏移 0 → 槽位 12   ← 跨块边界的第一个位置
逻辑位置 11 → 块表[2]=块 11，偏移 3 → 槽位 47   ← 当前序列的最后位置
```

序列在显存里完全不连续（块 7、3、11 散落各处），但通过块表，模型代码只用逻辑位置——这就是「虚拟地址」的思路：**使用者看到连续，物理层随便散**。操作系统页表 → vLLM 块表 → miniflow 的 `block_tables`，同一个翻译公式跨越五十年。

**尾浪费有上界**：每条序列至多浪费一个未填满的块（`stats()` 的 `tail_waste_slots` 直接度量它）。生产上报这个数是为了回答「块大小选多大」——块太大尾浪费高，块太小块表长、翻译开销高。

### 1.1 一个完整的分配-回收生命周期（数字版）

用 10 个块的池子（block_size=4）走一条序列的一生——**每个数字建议手推**：

```text
初始：free = [9,8,7,6,5,4,3,2,1,0]（栈顶是 9）
      ref_counts = {}，block_tables = {}

序列 0 到达，prompt 5 个 token（需要 blocks_needed(5)=2 块，
外加「+1 位置」预留 → 实际算 2 块）：
  分配块 9（ref=1）、块 8（ref=1）
  block_tables[0] = [9, 8]
  computed[0] = [0,0,0,0]（4 层，各算 0 个）
  
序列 0 prefill：5 token 逐层推进 → computed[0] = [5,5,5,5]
  （块 9 装位置 0-3，块 8 装位置 4-7——只用 1 槽，浪费 3）

序列 0 decode：每步 +1 token。第 4 步后位置 8 出现 → 需要第 3 块：
  分配块 7（ref=1），block_tables[0] = [9, 8, 7]

序列 0 结束（生成了 10 个 token，全长 15）：
  free_seq(0)：块 9、8、7 各 ref-1 → 全部归零 → 回收
  free = [7, 8, 9, ...]（注意顺序：栈式回收）
  block_tables.pop(0)，computed.pop(0)
```

这个手推覆盖了分配、增长、回收的完整循环——`demo_paged_kv.py` 打印的就是这条轨迹的自动化版。**对账的感觉建立后，后面两个踩坑（引用泄漏、悬空映射）的病灶就一目了然了**——它们都是这条轨迹上「某一跳没走干净」。

## 2. 按层进度：断点续算的地基

`computed[seq_id][layer]` 是**每层独立**的推进计数。为什么必须按层：第 L 层刚写的 KV 对第 L+1 层不可见（第 L+1 层的 KV 要等第 L+1 层的输入出现才算得出）。

这个语义撑起三个高级特性（断点续算 / 推测解码回滚 / chunked prefill，见 [guide/06 第 6 节](../guide/06-kvcache.md)），`set_computed`（前缀命中直接设进度）与 `reset_computed`（spec 回滚退回保留长度）是它的两个操纵面。

值得看的操纵面实现（各三行，但语义承重）：

```python
def set_computed(self, seq_id, n):
    # 前缀命中：命中段的 KV 是"全层现成"的——所有层一起设
    self.computed[seq_id] = [n] * self.mcfg.num_layers

def reset_computed(self, seq_id):
    # spec 回滚：全部层对齐到第 0 层的进度（层间依赖的保守下界）
    keep = self.computed[seq_id][0]
    self.computed[seq_id] = [keep] * self.mcfg.num_layers
```

两个「×全层」的赋值语义相反（一个前进到 n、一个退回到第 0 层）但**形式对称**——「全层齐设」是「该段 KV 完整存在」这个事实的表达：命中段的 KV 从缓存来（全层都有），回滚到保留位置（那个位置之前的全层有效）。**进度表的状态永远编码着「哪些层的哪些位置是有效的」**——想通这一点，Lab 05 的回滚 bug（`reset_computed` 用错基准）的病灶就不用死记了。

## 3. 引用计数：三个引用路径

| 路径 | 代码 | 语义 |
|---|---|---|
| 序列自有 | `_allocate_block` → ref=1 | 普通分配 |
| 前缀挂载 | `acquire_blocks` → ref+=1 | 块表尾部追加共享块 |
| 缓存持有 | `acquire_reference` → ref+=1 | 缓存条目在 TTL 语义下持有 |

回收统一走 `release_reference` / `free_seq`：**减到零才真正归还空闲池**。

三条路径的「数字版对账」接着 1.1 节的生命周期走（前缀缓存登场）：

```text
序列 1 到达，prompt 前 4 个 token 和序列 0 的开头相同
  → 前缀缓存命中：块 9 覆盖位置 0-3（如果序列 0 的块还在缓存里）
  
  缓存登记（序列 0 结束时）：块 9 的 ref 曾归零——但缓存条目
  持有它（TTL 语义）→ 实际上序列 0 结束时：
    ref(块9) = 1（序列0 的释放被缓存持有抵消）
    
  序列 1 命中挂载：acquire_blocks → ref(块9) = 2
    （缓存 1 份 + 序列 1 的块表 1 份）
    
  序列 1 结束：free_seq(1) → ref(块9) = 1（缓存的还在）
  
  TTL 过期（教学版：缓存逐出策略触发）：release_reference
    → ref(块9) = 0 → 真正回收 ✓
```

**一个块的一生里 ref 从 0 → 1 → 2 → 1 → 0**——三个 +1 路径（分配/挂载/登记）和它们对应的 -1 时刻（释放/释放/过期）配平。泄漏就是配平被打破（下面两个坑都是）。

!!! danger "踩坑实录：条目不能依赖源序列的块表"

    PrefixCache 早期版本把「块号列表」存在缓存条目里，但**块表仍在
    源序列名下**——源序列 free_seq 时块表被 pop，缓存条目指向的物理
    块引用还在、**「逻辑位置 → 物理块」的映射却没了**。后续命中路径
    读到悬空映射，行为未定义。重写为「条目自带块列表 + 持有全部前缀
    块引用」——对应 3FS 的哲学：**KV 数据与 KV 到 slot 的映射都归
    存储层所有**，计算侧（源序列）只是临时挂载者
    （[guide/03 第 3 节](../guide/03-architecture.md))。

    这个坑的解剖值得做深一层：它是一个**所有权语义错位**——引用计数
    （数据层）说「块活着」，块表（映射层）却说「不知道块在哪」。
    **两层所有权不同步**：数据拥有权给了缓存（ref 持有），映射拥有权
    留在了源序列（块表）。修复的本质是把两层所有权一起移交——
    「条目自带块列表」= 映射也归缓存。**多层次的资源要同时移交所有权，
    半移交就是悬空**——这个模式在你自己的代码里搜「谁持有谁」，
    凡是两层持有不一致的地方都是候选 bug。

!!! danger "踩坑实录：先挂载后检查 = 引用泄漏"

    调度器对 waiting 序列的前缀命中路径，早期顺序是
    `acquire_blocks → can_allocate 检查（失败 break）`——检查失败时
    已挂的引用不回滚；下一轮重试又 acquire 一遍，**引用只增不减**，
    块永远无法回收（池子慢性死亡）。修复：先检查（命中块算「已覆盖、
    不需新块」），通过后再挂载：

    ```python
    # miniflow/scheduler.py（修复后）
    need_new = self.kv.blocks_needed(seq.num_tokens + 1) - len(hit_blocks)
    if need_new > self.kv.num_free_blocks:
        break                      # ← 先检查，未挂载，无泄漏
    ...
    self.kv.acquire_blocks(hit_blocks, seq.seq_id)   # ← 后挂载
    ```

    和 [Lab 04](04-scheduler.md) 的死锁二（检查与分配原子化）同一个
    病根：**凡是「先动状态、后检查、失败不回滚」的顺序都是泄漏源**。

    数字版病灶演化（把泄漏的慢性死亡量化）：

    ```text
    设每轮调度有 30% 概率触发"检查失败"（块池紧张时）
    每次失败泄漏 len(hit_blocks) ≈ 3 个引用
    
    第 1 轮：泄漏 3 ref → 3 个块永远少 1 才能回收
    第 10 轮：累计 30 ref → ~10 个块死亡
    第 100 轮：~100 个块死亡 —— 池子（几百块）过半报废
    
    表现：跑得越久 free_blocks 越少、OOM 越频繁——
    "跑久了变慢/挂掉"的病几乎都在这条曲线上
    ```

    **慢性泄漏的监测签名就是这条单调递减曲线**——`demo_paged_kv.py`
    的 stats 输出里盯着 free_blocks 的长期趋势，任何单调漂移都是
    泄漏在敲门（生产的 Prometheus 上同款告警）。

## 4. 前缀命中的收缩语义

命中长度必须收缩到**至少留 1 个 token** 给 forward 产出采样 logits：

```python
max_blocks = (seq.num_tokens - 1) // self.kv.block_size
if len(hit_blocks) > max_blocks:
    hit_blocks = hit_blocks[:max_blocks]
```

vLLM 同款约束。为什么必须是 `-1`：prefill 算完第 n 个 token 得到的是第 n+1 位置的输出分布——如果命中覆盖了全部 n 个 token，这次 forward 没有任何新 token 可算，增量契约（[Lab 02 第 2 节](02-model.md)）直接违反，第一个输出 token 无从采起。

值得做一个「故意打破」的实验（10 分钟，对这个约束的记忆刻骨铭心）：

```bash
# 改 prefix_cache.py 的收缩逻辑：把 -1 去掉（hit_blocks 不收缩）
# 跑 demo_prefix_cache.py：
#   预期报错：transformer.py 的 assert n > 0 触发
#   "forward 需要至少一个新 token（增量契约）"
```

**约束被打破时的报错信息是约束本身的最好讲解**——`assert n > 0` 的失败让你亲眼看到「全命中 = 无 logits 可采」的因果链。看完改回去（或者干脆保留这个实验脚本，给下一个读代码的人）。

## 5. 动手实验

```bash
python examples/demo_paged_kv.py
# 观测点：
#   块的分配顺序（栈式 pop）与回收（归零才回收）
#   free_seq(0) 回收块: [255, 254, 253, 250]   ← 引用计数生效
#   stats: used/free/tail_waste_slots           ← 尾浪费有界

python examples/demo_prefix_cache.py
# 观测点：
#   A 完成后条目登记（块引用 +1）
#   B 的 prompt 前缀命中 → acquire_blocks 挂载 + set_computed 跳过
#   hit_rate = 0.5，B 的 prefill 少算 20 token
#   生产对应：3FS 的 TTL 内同前缀流量整段跳过 prefill

python -m pytest tests/test_prefix_cache.py -q
#   契约：命中省算力（saved_prefill_tokens > 0）
#         复用不改语义（B 的输出 token 级等于无缓存引擎）
```

demo 的进阶玩法（值得做的三个变体）：

**变体一（泄漏复现）**：在 `demo_paged_kv.py` 里加一段「挂载后故意 break」的循环（模仿踩坑二），循环 50 轮——看 stats 里 free_blocks 单调下降。**亲手让慢性病发生，以后在生产的监控曲线上认得出它**。

**变体二（块大小扫描）**：`block_size` 从 2 到 32 扫一遍（改 `CacheConfig`），记录 `tail_waste_slots` 和块表平均长度——画出 guide/06 2.3 节「四维权衡」的两维。**教学池子小、最优块也小（4），但曲线形状和生产（128）同构**。

**变体三（命中率-流量耦合）**：改 `demo_prefix_cache.py` 的共享前缀长度（0 到全长扫描），记录 saved_prefill_tokens——验证「命中率 × 前缀长度 = 省算量」的线性关系，找出非线性偏差的位置（提示：留 1 收缩在全长附近制造拐点）。

??? question "思考题"

    1. 把 `block_size` 从 4 改成 16，重跑 `demo_paged_kv.py`：
       `tail_waste_slots` 怎么变？什么时候块变大反而亏？
    2. 命中收缩语义里为什么是 `num_tokens - 1` 而不是
       `num_tokens`？（提示：增量契约 + 采样 logits 的来源。）
    3. `set_computed` 为什么必须**所有层一起**设？（对比：正常
       forward 是逐层推进的——前缀块里的 KV 是「全层现成」的，
       命中段没有「算到一半」的层。）
    4. 1.1 节的生命周期里，序列 0 结束时块 9 回收了。但如果序列 1
       此刻正在共享块 9（前缀命中），回收会怎么被引用计数拦下？
       写出 ref 的变化序列。然后想：如果踩坑二存在（泄漏），这个
       序列怎么被污染？
    5. 变体一（泄漏复现）里，泄漏的是「引用」还是「块」？（提示：
       块的物理内存在池子里始终存在——泄漏的是"它可以被重新分配"
       的资格。这个区分对修复策略有什么暗示——修复要回滚什么？）

## 6. 本章收获的自检

1. 逻辑位置→物理槽位的翻译公式，用块表 [5, 2, 9] 和 block_size=4 算逻辑位置 6 的槽位（答：`9*4 + 2 = 38`）。
2. 三条 +1 路径和三个 -1 时刻的配平关系（分配↔释放、挂载↔释放、登记↔过期）。
3. 两个踩坑的病根各是什么（所有权两层错位 / 先动状态后检查）？它们的监测签名（悬空读 / 单调递减曲线）？
4. 留 1 收缩保护的是什么契约（增量前向——全命中则无 logits）？

| 上一章 | 下一章 |
|---|---|
| [Lab 02 模型层](02-model.md) | [Lab 04 调度器与三重死锁](04-scheduler.md) |