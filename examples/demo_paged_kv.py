"""Paged KV Cache 演示 —— 复刻 vLLM PagedAttention / DeepInfer 块状 KV 管理。

用法（无需训练，随机权重演示机制）：
    python examples/demo_paged_kv.py

看点：
  1. 块表（block_table）：逻辑位置 -> 物理块号的映射，两条序列的物理块
     在池中交错分配 —— 朴素「连续预留」的碎片问题就此消失；
  2. 尾部浪费（tail_waste_slots）：碎片上限 = 每序列不足一块的尾部；
  3. 分层进度 computed[seq][layer]：一次 forward 后每层都推进到 token 数；
  4. 释放与复用：free_seq 后块回到空闲池，供新序列使用。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniflow.config import CacheConfig, ModelConfig  # noqa: E402
from miniflow.model.transformer import TransformerModel  # noqa: E402
from miniflow.sampling_params import SamplingParams  # noqa: E402
from miniflow.sequence import Sequence  # noqa: E402
from miniflow.cache.paged_kv import PagedKVCache  # noqa: E402


def main() -> None:
    cfg = ModelConfig(vocab_size=66, hidden_size=64, num_layers=4,
                      num_attention_heads=4, num_kv_heads=2,
                      moe_layers=(), swa_layers=(),
                      intermediate_size=128, max_position_embeddings=256)
    model = TransformerModel.random_init(cfg, seed=0)
    kv = PagedKVCache(CacheConfig(block_size=4, num_blocks=256), cfg)

    def new_seq(seq_id: int, n_tokens: int) -> Sequence:
        ids = list(np.random.default_rng(seq_id).integers(5, 66, size=n_tokens))
        return Sequence(seq_id, ids, SamplingParams())

    print("== 1) 两条序列交错分配物理块 ==")
    seq_a, seq_b = new_seq(0, 9), new_seq(1, 5)
    kv.ensure_capacity(seq_a)
    kv.ensure_capacity(seq_b)
    print(f"seq 0（9 token，3 块）: block_table = {kv.block_tables[0]}")
    print(f"seq 1（5 token，2 块）: block_table = {kv.block_tables[1]}")
    print("  -> 物理块号不连续：分页分配不需要连续内存（对应 GPU 显存碎片免疫）")

    print("\n== 2) 序列增长：按块追加，不动旧块 ==")
    seq_a.output_token_ids.extend([10, 11, 12, 13])   # 模拟又解码出 4 个 token
    kv.ensure_capacity(seq_a)
    print(f"seq 0（13 token，4 块）: block_table = {kv.block_tables[0]}")
    print("  -> 只有跨过块边界才新分配一块，旧块纹丝不动")

    print("\n== 3) 增量前向 + 分层进度 ==")
    logits = model.forward(seq_a.token_ids, seq_a.seq_id, kv)
    prog = kv.computed[seq_a.seq_id]
    print(f"forward 后 logits: {logits.shape}（每增量 token 一行）")
    print(f"computed[seq0] = {prog}  <- 每层独立推进到同一进度")
    hist_k, hist_v, hist_pos = kv.get_kv(0, seq_a.seq_id)
    print(f"get_kv(第0层): K {hist_k.shape}，positions[0..3] = {hist_pos[:4]}")
    print("  -> 读出的 K 已是 RoPE 之后的（位置信息烘焙在数值里）")

    print("\n== 4) 尾部浪费与释放复用 ==")
    st = kv.stats()
    print(f"stats: used={st['used_blocks']} free={st['free_blocks']} "
          f"tail_waste={st['tail_waste_slots']} slots")
    freed = kv.free_seq(seq_a.seq_id)
    print(f"free_seq(0) 回收块: {freed}")
    st = kv.stats()
    print(f"stats: used={st['used_blocks']} free={st['free_blocks']}  -> 块回到空闲池")


if __name__ == "__main__":
    main()