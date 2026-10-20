"""命令行入口 —— 对应 examples/offline_inference.py 的角色。

用法：
    python -m miniflow.entrypoints.cli --checkpoint checkpoints/tiny.npz \
        --tokenizer checkpoints/tiny.tok.json --prompt "the cat" --max-tokens 40

环境开关（对应 DEEPINFER_* 体系）：
    MF_SPEC_DECODING=1   开启推测解码
    MF_ASYNC_SCHED=0     关闭异步调度（同步对照）
"""

from __future__ import annotations

import argparse
import json
import os

from ..config import CacheConfig, EngineConfig, MiniFlowConfig, ModelConfig
from ..model.transformer import TransformerModel
from ..sampling_params import SamplingParams
from ..tokenizer import Tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="miniflow", description="miniflow 离线推理 CLI")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="the cat")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--async-sched", action="store_true",
                    help="使用异步调度引擎（AsyncLLMEngine）")
    return p


def load_model_config(checkpoint: str, tokenizer: Tokenizer,
                      model_config: ModelConfig | None = None) -> ModelConfig:
    """优先读 checkpoint 旁的 .cfg.json（train.py 的产物）——
    对应生产框架 model_config_loader 从 HF 权重目录加载 config.json：
    结构超参属于「权重的一部分」，凭默认值猜测必然出错。"""
    if model_config is not None:
        return model_config
    cfg_path = (checkpoint[: -len(".npz")] if checkpoint.endswith(".npz")
                else checkpoint) + ".cfg.json"
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            d = json.load(f)
        return ModelConfig(
            vocab_size=d["vocab_size"], hidden_size=d["hidden_size"],
            num_layers=d["num_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_kv_heads=d["num_kv_heads"],
            swa_layers=tuple(d.get("swa_layers", ())),
            moe_layers=tuple(d.get("moe_layers", ())),
            intermediate_size=d["intermediate_size"],
            max_position_embeddings=d["max_position_embeddings"],
        )
    return ModelConfig(vocab_size=tokenizer.vocab_size)


def load_engine(checkpoint: str, tokenizer_path: str,
                model_config: ModelConfig | None = None) -> "LLMEngine":
    from ..engine import LLMEngine
    tok = Tokenizer.load(tokenizer_path)
    cfg = MiniFlowConfig(
        model=load_model_config(checkpoint, tok, model_config),
        cache=CacheConfig(prefix_cache_enabled=True),
        engine=EngineConfig(),
    )
    model = TransformerModel.load(checkpoint, cfg.model)
    return LLMEngine(cfg, model, tok)


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    engine = load_engine(args.checkpoint, args.tokenizer)
    params = SamplingParams(max_tokens=args.max_tokens,
                            temperature=args.temperature, seq_seed=args.seed)
    engine.add_request(args.prompt, params)

    if args.async_sched:
        from ..engine import AsyncLLMEngine
        async_engine = AsyncLLMEngine(engine)
        outputs = async_engine.run()
    else:
        outputs = engine.run()

    for out in outputs:
        print(f"[seq {out.seq_id}] prompt={out.prompt!r}")
        print(f"  output ({out.finish_reason}): {out.output_text!r}")
    print("\nengine stats:")
    for k, v in engine.stats().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()