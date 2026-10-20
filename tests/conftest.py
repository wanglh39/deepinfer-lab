"""测试公共设施：checkpoint 探测 + 引擎工厂。

所有引擎级测试共用同一工厂 —— 各子系统（前缀缓存/推测解码/批不变性）
按测试需要开关，对应生产框架「同一份代码，环境变量切换行为」的思想。
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CKPT = os.path.join(ROOT, "checkpoints", "tiny.npz")
TOK = os.path.join(ROOT, "checkpoints", "tiny.tok.json")

from miniflow.config import CacheConfig, EngineConfig, MiniFlowConfig  # noqa: E402
from miniflow.engine import LLMEngine  # noqa: E402
from miniflow.entrypoints.cli import load_model_config  # noqa: E402
from miniflow.model.transformer import TransformerModel  # noqa: E402
from miniflow.tokenizer import Tokenizer  # noqa: E402


@pytest.fixture(scope="session")
def paths():
    if not os.path.exists(CKPT):
        pytest.skip("未找到 checkpoint，请先运行: "
                    "python tools/train.py --out checkpoints/tiny")
    return CKPT, TOK


@pytest.fixture(scope="session")
def engine_factory(paths):
    """返回构建函数：build(prefix_cache=..., spec=..., num_blocks=...)。"""

    def build(*, prefix_cache: bool = True, spec: bool = False,
              num_blocks: int = 256, block_size: int = 4,
              max_draft: int = 5, batch_invariant: bool = True) -> LLMEngine:
        tokenizer = Tokenizer.load(TOK)
        cfg = MiniFlowConfig(
            model=load_model_config(CKPT, tokenizer),
            cache=CacheConfig(block_size=block_size, num_blocks=num_blocks,
                              prefix_cache_enabled=prefix_cache),
            engine=EngineConfig(speculative_decoding=spec,
                                max_draft_tokens=max_draft,
                                batch_invariant=batch_invariant),
        )
        model = TransformerModel.load(CKPT, cfg.model)
        return LLMEngine(cfg, model, tokenizer)

    return build