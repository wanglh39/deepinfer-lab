# Lab 03 分页 KV 与前缀缓存

!!! info "本章对应"
    - 生产组件：PagedAttention / 3FS 块级 KV / 前缀缓存
    - miniflow 代码：`miniflow/cache/paged_kv.py`、`miniflow/cache/prefix_cache.py`
    - 前置阅读：[06 KV Cache](../guide/06-kvcache.md)

---

## 1. PagedKVCache 的物理结构

```python
# 物理池（生产上是 HBM，这里是一组 NumPy 数组，层间独立）
k_pools = [np.zeros((num_slots, kv_heads, head_dim)) for _ in range(L)]
#          num_slots = num_blocks * block_size

free_blocks: List[int]        # 空闲块号栈
ref_counts: Dict[int, int]    # 物理块 → 引用数（共享的基石）
block_tables: Dict[int, List[int]]   # seq_id → 块表（逻辑→物理）
computed: Dict[int, List[int]]       # seq_id → 每层各自的已算 token 数
```

slot 定位一行讲清：`slot = table[logical // block_size] * block_size +
logical % block_size` —— 逻辑位置经块表翻译到物理池偏移。

**尾浪费上限**：每条序列至多一个块的部分浪费（`stats()` 的
`tail_waste_slots` 直接度量 —— [guide/06 第 2 节](../guide/06-kvcache.md)）。

## 2. 按层进度：断点续算的地基

`computed[seq_id][layer]` 是**每层独立**的推进计数 —— 第 L 层刚写的
KV 对第 L+1 层不可见。这个语义撑起三个高级特性（断点续算 / 推测解码
回滚 / chunked prefill），`set_computed`（前缀命中直接设进度）与
`reset_computed`（spec 回滚退回保留长度）是它的两个操纵面。

## 3. 引用计数：三个引用路径

| 路径 | 代码 | 语义 |
|---|---|---|
| 序列自有 | `_allocate_block` → ref=1 | 普通分配 |
| 前缀挂载 | `acquire_blocks` → ref+=1 | 块表尾部追加共享块 |
| 缓存持有 | `acquire_reference` → ref+=1 | 缓存条目在 TTL 语义下持有 |

回收统一走 `release_reference` / `free_seq`：**减到零才真正归还空闲池**。

!!! danger "踩坑实录：条目不能依赖源序列的块表"
    PrefixCache 早期版本把「块号列表」存在缓存条目里，但**块表仍在
    源序列名下** —— 源序列 free_seq 时块表被 pop，缓存条目指向的
    引用还在、**映射没了**（无法从逻辑位置找到物理块）。命中路径
    读到悬空的映射。重写为「条目自带块列表 + 持全部前缀块引用」——
    对应 3FS 的哲学：**KV 与 KV 到 slot 的映射都归存储层所有**，
    计算侧（源序列）只是临时挂载者（[guide/06 第 4 节](../guide/06-kvcache.md)）。

!!! danger "踩坑实录：先挂载后检查 = 引用泄漏"
    调度器对 waiting 序列的前缀命中路径，早期顺序是
    `acquire_blocks → can_allocate 检查（失败 break）`—— 检查失败时
    已挂的引用不回滚；下一轮重试又 acquire 一遍，**引用只增不减**，
    块永远无法回收。修复：先检查（命中块算「已覆盖、不需新块」），
    通过后再挂载：

    ```python
    # miniflow/scheduler.py（修复后）
    need_new = self.kv.blocks_needed(seq.num_tokens + 1) - len(hit_blocks)
    if need_new > self.kv.num_free_blocks:
        break                      # ← 先检查，未挂载，无泄漏
    ...
    self.kv.acquire_blocks(hit_blocks, seq.seq_id)   # ← 后挂载
    ```

## 4. 前缀命中的收缩语义

命中长度必须收缩到**至少留 1 个 token** 给 forward 产出采样 logits：

```python
max_blocks = (seq.num_tokens - 1) // self.kv.block_size
if len(hit_blocks) > max_blocks:
    hit_blocks = hit_blocks[:max_blocks]
```

vLLM 同款约束（prefill 至少重算最后一个 token）—— 否则命中了整个
prompt，本步 forward 没有新 token 可算，增量契约直接违反。

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

??? question "思考题"
    1. 把 `block_size` 从 4 改成 16，重跑 `demo_paged_kv.py`：
       `tail_waste_slots` 怎么变？什么时候块变大反而亏？
    2. 命中收缩语义里为什么是 `num_tokens - 1` 而不是
       `num_tokens`？（提示：增量契约 + 采样 logits 的来源）
    3. `set_computed` 为什么必须**所有层一起**设？（对比：正常
       forward 是逐层推进的 —— 前缀块里的 KV 是「全层现成」的）

| 上一章 | 下一章 |
|---|---|
| [Lab 02 模型层](02-model.md) | [Lab 04 调度器与三重死锁](04-scheduler.md) |