"""采样参数 —— 对应 deepinfer/sampling_params.py。

关键教学点：`seq_seed`。DeepInfer 用它配合 batch-invariant 实现确定性采样——
相同输入 token 序列 + 相同采样参数 => 输出完全一致，且与批内其他请求无关。
这在生产上的价值是可复现（回归测试、灰度对比、审计）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1                  # -1 表示不启用
    top_p: float = 1.0               # 1.0 表示不启用
    max_tokens: int = 32
    stop: Sequence[str] = field(default_factory=tuple)
    ignore_eos: bool = False
    seq_seed: Optional[int] = None   # 逐请求随机流种子（确定性采样的核心）