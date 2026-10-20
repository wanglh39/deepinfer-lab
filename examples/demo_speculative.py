"""推测解码演示 —— 复刻 DSpark「变长验证 + 置信度调度」的加速效应。

用法（需要训练好的 checkpoint）：
    python tools/train.py --out checkpoints/tiny --steps 400
    python examples/demo_speculative.py

核心账本：decode 每步只算 1 个 token（访存受限、算力闲置）；推测解码
一次 forward「审」多个候选 —— 验证 k 个 token 的算力 ≈ 生成 1 个。
draft 命中 argmax 则接受，被拒位置用 argmax 修正 => **输出与纯 greedy
严格一致**，但步数更少（本 demo 的可验证加速）。

DSSpark 生产数据（DeepInfer 部署 QA）：变长验证把平均验证长度
5.00 -> 3.2~4.4，接受长度几乎不变，吞吐 +4~7%。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniflow.config import CacheConfig, EngineConfig, MiniFlowConfig  # noqa: E402
from miniflow.entrypoints.cli import load_engine, load_model_config  # noqa: E402
from miniflow.sampling_params import SamplingParams  # noqa: E402
from miniflow.tokenizer import Tokenizer  # noqa: E402
from miniflow.model.transformer import TransformerModel  # noqa: E402
from miniflow.engine import LLMEngine  # noqa: E402

CKPT = "checkpoints/tiny.npz"
TOK = "checkpoints/tiny.tok.json"
PROMPT = "the cat"


def build_engine(spec: bool) -> LLMEngine:
    tok = Tokenizer.load(TOK)
    cfg = MiniFlowConfig(
        model=load_model_config(CKPT, tok),
        cache=CacheConfig(prefix_cache_enabled=False),
        engine=EngineConfig(speculative_decoding=spec, max_draft_tokens=5),
    )
    model = TransformerModel.load(CKPT, cfg.model)
    return LLMEngine(cfg, model, tok)


def main() -> None:
    if not os.path.exists(CKPT):
        print("未找到 checkpoint，请先运行: python tools/train.py --out checkpoints/tiny")
        sys.exit(1)

    params = SamplingParams(max_tokens=24, temperature=0.0)

    base = build_engine(spec=False)
    base.add_request(PROMPT, params)
    (out_base,) = base.run()
    steps_base = base.stats()["steps"]

    spec = build_engine(spec=True)
    spec.add_request(PROMPT, params)
    (out_spec,) = spec.run()
    steps_spec = spec.stats()["steps"]

    print(f"prompt: {PROMPT!r}  (greedy, max_tokens=24)\n")
    print(f"[基线]    steps = {steps_base}")
    print(f"         output = {out_base.output_text!r}")
    print(f"[推测解码] steps = {steps_spec}   <- 同样 24 个 token，步数更少")
    print(f"         output = {out_spec.output_text!r}")
    print(f"\n输出与纯 greedy 完全一致: "
          f"{out_spec.token_ids[:len(out_base.token_ids)] == out_base.token_ids}")
    print(f"等效加速（steps 基线/推测）: {steps_base / steps_spec:.2f}x\n")

    st = spec.stats()["speculative"]
    print("DSpark 同款指标（spec.stats）：")
    for k, v in st.items():
        print(f"  {k:18s} = {v}")
    print("\n解读：avg_verified_len < max_draft(5) 即「变长验证」生效 ——")
    print("置信度不达标的 draft 前缀直接丢弃（KV 回滚），验证算力只花在")
    print("边际收益高的 token 上；accept_rate 是 draft 与 argmax 的吻合度。")


if __name__ == "__main__":
    main()