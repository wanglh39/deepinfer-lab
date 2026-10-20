"""前缀缓存演示 —— 复刻 DeepInfer「3FS 块哈希链前缀匹配」的省算力效应。

用法（需要训练好的 checkpoint）：
    python tools/train.py --out checkpoints/tiny --steps 400
    python examples/demo_prefix_cache.py

场景：请求 A 与请求 B 共享同一段长 prompt（生产里是共享系统 prompt /
多轮 agent 会话）。A 结束时其 KV 块被登记进前缀缓存（缓存持引用，
对应 3FS 持久化）；B 到来时按块哈希链命中，命中部分的 prefill 直接跳过。

对比指标：B 的 saved_prefill_tokens、engine steps（省掉的解码步）、
prefix_cache 命中率。注意块大小 = 4：只有「整块」对齐的前缀才可复用
（KV 以块为粒度存取 —— 与生产 128-token 块同一约束）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniflow.entrypoints.cli import load_engine  # noqa: E402
from miniflow.sampling_params import SamplingParams  # noqa: E402

CKPT = "checkpoints/tiny.npz"
TOK = "checkpoints/tiny.tok.json"

SHARED_PROMPT = "the cat sat on the mat. the dog ran in the park."


def main() -> None:
    if not os.path.exists(CKPT):
        print("未找到 checkpoint，请先运行: python tools/train.py --out checkpoints/tiny")
        sys.exit(1)

    engine = load_engine(CKPT, TOK)
    ids = engine.tokenizer.encode(SHARED_PROMPT, add_bos=True)
    print(f"共享 prompt: {SHARED_PROMPT!r}")
    print(f"token 数: {len(ids)}（块大小 4 => {len(ids) // 4} 个整块可命中）\n")

    # ---- 请求 A：冷启动，全量 prefill，结束时登记前缀缓存 ----
    engine.add_request(SHARED_PROMPT, SamplingParams(max_tokens=8, temperature=0.0))
    (out_a,) = engine.run()
    st = engine.stats()
    print(f"[A 冷启动] steps={st['steps']}  prefix_cache={st['prefix_cache']}")
    print(f"           输出: {out_a.output_text!r}\n")

    # ---- 请求 B：同 prompt，前缀命中，跳过命中部分的重算 ----
    engine.add_request(SHARED_PROMPT, SamplingParams(max_tokens=8, temperature=0.0))
    (out_b,) = engine.run()
    st = engine.stats()
    print(f"[B 前缀命中] steps={st['steps']}（含 A 的步数）  "
          f"saved_prefill_tokens={st['saved_prefill_tokens']}")
    print(f"            prefix_cache={st['prefix_cache']}")
    print(f"            输出: {out_b.output_text!r}\n")

    print("解读：B 的 prompt 前缀块直接挂到块表（acquire_blocks + set_computed），")
    print("prefill 只需算剩余后缀 —— 生产上共享系统 prompt 的流量因此大幅省算力，")
    print("DeepSeek 把这些块持久化在 3FS（agent 3 天 / chat 7 天 TTL）。")


if __name__ == "__main__":
    main()