"""Paged KV Cache —— 复刻 vLLM PagedAttention / DeepInfer 块状 KV 管理的核心思想。

**为什么需要分页？**
朴素 KV cache 为每条序列预留 max_len 连续内存：碎片 + 浪费。
PagedAttention 借操作系统虚拟内存的思想：
  * 物理池切成固定大小的 **块（block）**；
  * 每条序列一张 **块表（block_table）**：逻辑位置 -> 物理块号；
  * 序列增长时按块分配，删除时按块回收 —— 碎片上限是块内尾部浪费。

**DeepInfer 的取舍**（文档站《KV Cache》一章详述）：
  * 生产块大小 128 token，3FS 上以块为粒度持久化（TTL 3~7 天）；
  * 前缀命中时物理块被跨序列共享 -> 因此块有 **引用计数**；
  * SWA 层的 KV 不进块池，走定长 StateBlock。

miniflow 的物理池是 NumPy 数组（生产上是 HBM），层间独立：
    k_pool: [num_slots, kv_heads, head_dim]   （slot = block_id * block_size + offset）

**分层进度**：每层各自记录「已算 token 数」`computed[seq][layer]`。
一次 forward 里第 L 层刚写入的新 KV，第 L+1 层尚不可见 ——
这正是生产框架「按层同步推进」的语义（也是断点续算/推测解码回滚的基石）。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..config import CacheConfig, ModelConfig
from ..sequence import Sequence


class PagedKVCache:
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig):
        self.cfg = cache_config
        self.mcfg = model_config
        self.block_size: int = cache_config.block_size
        self.num_blocks: int = cache_config.num_blocks
        self.num_slots: int = self.block_size * self.num_blocks

        shape = (self.num_slots, model_config.num_kv_heads, model_config.head_dim)
        self.k_pools = [np.zeros(shape, dtype=np.float32) for _ in range(model_config.num_layers)]
        self.v_pools = [np.zeros(shape, dtype=np.float32) for _ in range(model_config.num_layers)]

        self.free_blocks: List[int] = list(range(self.num_blocks))
        self.ref_counts: Dict[int, int] = {}
        self.block_tables: Dict[int, List[int]] = {}
        # seq_id -> 每层已写入 KV 的 token 数
        self.computed: Dict[int, List[int]] = {}

    # ------------------------------------------------------------------
    # 容量与分配
    # ------------------------------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        """新序列从零分配：整序列所需块 <= 空闲块。"""
        return self.blocks_needed(num_tokens) <= self.num_free_blocks


    def _allocate_block(self) -> int:
        if not self.free_blocks:
            raise MemoryError("PagedKVCache: 物理块耗尽（对应生产上的抢占/换出路径）")
        block = self.free_blocks.pop()
        self.ref_counts[block] = 1
        return block

    def ensure_capacity(self, seq: Sequence) -> None:
        """块表扩充到覆盖 seq.num_tokens（执行期兜底，通常已被调度器满足）。"""
        self.ensure_capacity_tokens(seq.seq_id, seq.num_tokens)

    def ensure_capacity_tokens(self, seq_id: int, num_tokens: int) -> None:
        """块表扩充到覆盖 num_tokens（调度期调用即「预留实体化」）。"""
        table = self.block_tables.setdefault(seq_id, [])
        needed = self.blocks_needed(num_tokens)
        while len(table) < needed:
            table.append(self._allocate_block())
        self.computed.setdefault(seq_id, [0] * self.mcfg.num_layers)

    def ensure_append_capacity(self, seq: Sequence, extra: int = 1) -> bool:
        """为 running 序列的增长预留 extra 个 token 的容量。

        成功则**当场实体化分配**（检查与分配原子化），失败则不动任何
        状态。两个坑的解药（miniflow 开发中都真实踩过）：
          * 检查必须用增量语义（还要补几块），不是整序列需求 vs 空闲块
            —— 否则序列被自己已占的块挡住，自我抢占无限循环；
          * 分配必须即时生效，不能只记账 —— 否则同批 N 条序列都
            「看到」同一份富余，执行期物理分配炸 MemoryError。
        """
        table = self.block_tables.setdefault(seq.seq_id, [])
        needed = self.blocks_needed(seq.num_tokens + extra)
        if needed - len(table) > self.num_free_blocks:
            return False
        while len(table) < needed:
            table.append(self._allocate_block())
        self.computed.setdefault(seq.seq_id, [0] * self.mcfg.num_layers)
        return True

    # ------------------------------------------------------------------
    # 读写（每层独立进度）
    # ------------------------------------------------------------------
    def slots_for(self, seq_id: int, start: int, length: int) -> np.ndarray:
        table = self.block_tables[seq_id]
        slots = []
        for logical in range(start, start + length):
            block = table[logical // self.block_size]
            slots.append(block * self.block_size + logical % self.block_size)
        return np.asarray(slots, dtype=np.int64)

    def append(self, layer: int, seq_id: int, start: int,
               k: np.ndarray, v: np.ndarray) -> None:
        """写入 [start, start+n) 的新 K/V 并推进该层进度。k: [n, kv_heads, dim]。"""
        n = k.shape[0]
        slots = self.slots_for(seq_id, start, n)
        self.k_pools[layer][slots] = k
        self.v_pools[layer][slots] = v
        prog = self.computed[seq_id]
        prog[layer] = max(prog[layer], start + n)

    def get_kv(self, layer: int, seq_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """读出该层目前已写入的 (K, V, positions)。"""
        length = self.computed[seq_id][layer]
        if length == 0:
            empty_k = np.zeros((0, self.mcfg.num_kv_heads, self.mcfg.head_dim), np.float32)
            return empty_k, empty_k.copy(), np.zeros(0, dtype=np.int64)
        slots = self.slots_for(seq_id, 0, length)
        positions = np.arange(length, dtype=np.int64)
        return self.k_pools[layer][slots], self.v_pools[layer][slots], positions

    def set_computed(self, seq_id: int, length: int) -> None:
        """前缀缓存命中：所有层进度直接设为命中长度（块内已有现成 KV）。"""
        self.computed[seq_id] = [length] * self.mcfg.num_layers

    def reset_computed(self, seq_id: int) -> None:
        """推测解码回滚：进度退回保留长度，之后重算被拒的 draft。"""
        keep = self.computed[seq_id][0]
        self.computed[seq_id] = [keep] * self.mcfg.num_layers

    # ------------------------------------------------------------------
    # 生命周期（引用计数：前缀共享块要等所有引用结束才回收）
    # ------------------------------------------------------------------
    def acquire_blocks(self, blocks: List[int], to_seq: int) -> List[int]:
        """引用一组已有物理块（前缀缓存命中路径）：ref+1 并挂到块表。"""
        table = self.block_tables.setdefault(to_seq, [])
        self.computed.setdefault(to_seq, [0] * self.mcfg.num_layers)
        for b in blocks:
            self.ref_counts[b] = self.ref_counts.get(b, 0) + 1
            table.append(b)
        return blocks

    def acquire_reference(self, blocks: List[int]) -> None:
        """登记进前缀缓存时由缓存持有引用（逐出时 release_reference）。"""
        for b in blocks:
            self.ref_counts[b] = self.ref_counts.get(b, 0) + 1

    def release_reference(self, blocks: List[int]) -> None:
        for b in blocks:
            self.ref_counts[b] -= 1
            if self.ref_counts[b] <= 0:
                del self.ref_counts[b]
                self.free_blocks.append(b)

    def free_seq(self, seq_id: int) -> List[int]:
        """序列结束：该序列持有的引用减一，归零的块回收。返回回收块列表。"""
        table = self.block_tables.pop(seq_id, [])
        self.computed.pop(seq_id, None)
        freed = []
        for b in table:
            self.ref_counts[b] -= 1
            if self.ref_counts[b] <= 0:
                del self.ref_counts[b]
                self.free_blocks.append(b)
                freed.append(b)
        return freed

    # ------------------------------------------------------------------
    # 观测
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        used = self.num_blocks - self.num_free_blocks
        progress = [sum(p) // self.mcfg.num_layers for p in self.computed.values()]
        waste = 0
        for seq_id, prog in self.computed.items():
            table = self.block_tables.get(seq_id, [])
            length = prog[0]
            if table and length > 0:
                used_in_last = length - (len(table) - 1) * self.block_size
                waste += self.block_size - used_in_last
        return {
            "used_blocks": used, "free_blocks": self.num_free_blocks,
            "num_sequences": len(self.computed),
            "total_computed_tokens": int(sum(progress)),
            "tail_waste_slots": int(waste),
        }