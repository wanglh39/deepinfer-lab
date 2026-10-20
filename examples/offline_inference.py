"""离线批推理 —— 对应 DeepInfer 的 examples/offline_inference.py。

用法（先训练好 checkpoint）：
    python tools/train.py --out checkpoints/tiny --steps 400
    python examples/offline_inference.py

展示三件事：
  1. 一条 prompt 的 greedy 续写（temperature=0，完全确定）；
  2. 多请求组批推理（continuous batching：请求 A 解码时请求 B 插队 prefill）；
  3. 带温度 + seq_seed 的确定性采样 —— 相同种子重跑必得相同输出
     （batch-invariant + seq_seed 的组合拳，DeepInfer 可复现推理的核心）。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniflow.config import EngineConfig, MiniFlowConfig  # noqa: E402
from miniflow.entrypoints.cli import load_engine  # noqa: E402
from miniflow.sampling_params import SamplingParams  # noqa: E402

CKPT = "checkpoints/tiny.npz"
TOK = "checkpoints/tiny.tok.json"


def main() -> None:
    if not os.path.exists(CKPT):
        print("未找到 checkpoint，请先运行: python tools/train.py --out checkpoints/tiny")
        sys.exit(1)

    # ---- 1) greedy 单请求 ----
    engine = load_engine(CKPT, TOK)
    engine.add_request("the cat", SamplingParams(max_tokens=24, temperature=0.0))
    (out,) = engine.run()
    print(f"[greedy]   the cat -> {out.output_text!r}")
    print()

    # ---- 2) 多请求组批（continuous batching）----
    engine = load_engine(CKPT, TOK)
    prompts = ["the cat", "the dog", "the bird", "the fish"]
    for p in prompts:
        engine.add_request(p, SamplingParams(max_tokens=16, temperature=0.0))
    outputs = engine.run()
    for out in outputs:
        print(f"[batch]    {out.prompt!r} -> {out.output_text!r}")
    stats = engine.stats()
    print(f"          steps={stats['steps']}  (4 条请求共用了同一批解码步)")
    print()

    # ---- 3) 确定性采样：同 seed 重跑，输出必须一致 ----
    # 注：教学模型在微型语料上训到 loss~0.07，分布极其尖锐，需要较高的
    # 温度才能看到随机性 —— 这本身就是「置信度与采样多样性」的现场教材。
    def seeded_run(seed: int) -> list:
        eng = load_engine(CKPT, TOK)
        eng.add_request("the bird", SamplingParams(
            max_tokens=16, temperature=3.0, top_k=-1, top_p=1.0, seq_seed=seed))
        return eng.run()[0].token_ids

    a, b = seeded_run(42), seeded_run(42)
    print(f"[seed=42]  两次运行 token 一致: {a == b}")
    c = seeded_run(7)
    print(f"[seed= 7]  换种子输出不同  : {a != c}")


if __name__ == "__main__":
    main()