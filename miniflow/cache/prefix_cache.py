"""前缀缓存 —— 复刻 DeepInfer KVCache Storage 的「哈希链前缀匹配」思想。

**生产背景**：DeepSeek 把块级 KV cache 持久化在 3FS 分布式文件系统上
（~100 PiB SSD，TTL：agent 流量 3 天 / chat 流量 7 天）。新请求到来时，
Entrypoint 对块哈希链做 **仅元数据** 的前缀匹配，命中部分无需重算 prefill。
共享系统 prompt / 长 agent 会话的流量因此大幅省算力 —— 155T tokens/日
的集群里，这是被精确计量的「免费算力」。

**块哈希链**（对应 3FS 上的 key 设计）：
    block_hash[0] = hash(seed, tokens[0:block_size])
    block_hash[i]  = hash(block_hash[i-1], tokens[i*bs:(i+1)*bs])
链式包含前缀 => 哈希相同则前缀必然一致（教学版用内置 hash 简化，
生产上用抗碰撞摘要 + 仅元数据匹配，不读 KV 本体）。

**所有权模型（关键设计）**：登记进缓存的条目 **自带块映射** 并对全部
前缀块持有引用 —— 源序列结束释放自己的引用后，块仍被缓存钉住，
条目依然有效。这对应 3FS 的语义：KV 与其逻辑映射归存储所有，
不随生成它的请求消亡；逐出（LRU 超限）时释放引用，块才真正回收。

**与 PagedKVCache 的协作**：
  * put:    序列 prefill 完成后登记块哈希链（缓存持引用）；
  * match:  新请求查链取最长整块命中，返回块列表；
  * 命中后  engine 调 kv.acquire_blocks 把块挂到新序列块表
            （等价于 3FS 命中后 swap-in 到 HBM），并 set_computed 跳过重算；
  * 逐出:   LRU 超限时 release_reference。
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

from ..config import CacheConfig
from .paged_kv import PagedKVCache

_SEED = 0x9E3779B97F4A7C15
MAX_ENTRIES = 512


def block_hashes(token_ids: List[int], block_size: int) -> List[int]:
    n_blocks = len(token_ids) // block_size
    hashes: List[int] = []
    prev = _SEED
    for i in range(n_blocks):
        chunk = tuple(token_ids[i * block_size:(i + 1) * block_size])
        prev = hash((prev, chunk)) & 0xFFFFFFFFFFFFFFFF
        hashes.append(prev)
    return hashes


class PrefixCache:
    """块哈希 -> (前缀块列表, 块数) 的登记表（LRU 逐出，条目持引用）。

    条目 value 自带块映射（blocks 列表），因此 **源序列消亡后条目仍有效**
    —— 引用计数保证物理块活着，映射不需要回源查询。
    """

    def __init__(self, cache_config: CacheConfig):
        self.enabled = cache_config.prefix_cache_enabled
        self.block_size = cache_config.block_size
        self.table: "OrderedDict[int, Tuple[List[int], int]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------
    def match(self, token_ids: List[int], kv: PagedKVCache
              ) -> Tuple[int, List[int]]:
        """返回 (可复用 token 数, 前缀块列表)。

        命中策略与 DeepInfer 同构：只认 **整块** 命中（KV 以块为粒度存取）。
        沿新请求的哈希链逐块查表，断了就停 —— 前缀匹配是最长公共块数。
        """
        if not self.enabled:
            return 0, []
        hashes = block_hashes(token_ids, self.block_size)
        hit_blocks: List[int] = []
        for h in hashes:
            entry = self.table.get(h)
            if entry is None:
                break
            hit_blocks, n_blocks = entry[0], entry[1]
            self.table.move_to_end(h)        # LRU touch
        if hit_blocks:
            self.hits += 1
            return len(hit_blocks) * self.block_size, list(hit_blocks)
        self.misses += 1
        return 0, []

    # ------------------------------------------------------------------
    def put(self, seq_id: int, token_ids: List[int], kv: PagedKVCache) -> None:
        """序列 prefill 完成后登记其块（条目持引用，供后续复用）。"""
        if not self.enabled:
            return
        hashes = block_hashes(token_ids, self.block_size)
        table = kv.block_tables.get(seq_id, [])
        for i, h in enumerate(hashes):
            if h in self.table or i >= len(table):
                continue                      # 已登记 / 块表未覆盖（防御）
            blocks = list(table[:i + 1])     # 条目自带前缀块映射
            self.table[h] = (blocks, i + 1)
            kv.acquire_reference(blocks)     # 条目持有全部前缀块的引用
        self._evict_overflow(kv)

    def _evict_overflow(self, kv: PagedKVCache) -> None:
        while len(self.table) > MAX_ENTRIES:
            _h, (blocks, _n) = self.table.popitem(last=False)   # LRU 淘汰
            kv.release_reference(blocks)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "entries": len(self.table),
            "hits": self.hits, "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }