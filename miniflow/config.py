"""引擎配置 —— 对应 deepinfer/config.py 与 environ.py。

DeepInfer 的做法值得注意：
  * 配置分为 ModelConfig / CacheConfig / EngineConfig 等正交的部分；
  * 大量行为开关通过 `DEEPINFER_*` 环境变量暴露（例如
    DEEPINFER_ENABLE_MTP_VARLEN、DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE），
    这让同一份代码可以在不改动的条件下切换生产/调试行为。

miniflow 用 `MF_` 前缀环境变量复刻这一机制。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def env_flag(name: str, default: bool) -> bool:
    """读取 MF_* 布尔环境变量，缺省回落到 default。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ModelConfig:
    """模型结构超参 —— 对应 HuggingFace config.json 的加载 (model_config_loader.py)。

    教学模型是一个「缩微版 DeepSeek 风格 Transformer」：
      * attention 使用 GQA（MLA 的精神：用少量 KV 头压缓存）；
      * 若干层启用 SWA（滑动窗口注意力，对应 V4.1 的稀疏注意力层）；
      * FFN 使用 MoE（对应 V4.1 的专家层）。
    """

    vocab_size: int = 256
    hidden_size: int = 64
    num_layers: int = 4
    num_attention_heads: int = 4
    num_kv_heads: int = 2          # GQA：KV 头数 < Q 头数，缓存更小
    swa_layers: tuple = (2, 3)     # 哪些层使用滑动窗口注意力
    swa_window: int = 16           # SWA 窗口大小（token 数）
    num_experts: int = 4           # MoE 专家总数
    num_experts_per_tok: int = 2   # top-k 路由
    moe_layers: tuple = (1, 3)     # 哪些层使用 MoE（其余用 dense FFN）
    intermediate_size: int = 128
    max_position_embeddings: int = 512
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def uses_swa(self, layer_idx: int) -> bool:
        return layer_idx in self.swa_layers

    def uses_moe(self, layer_idx: int) -> bool:
        return layer_idx in self.moe_layers


@dataclass
class CacheConfig:
    """分页 KV cache 配置 —— 对应 vLLM/DeepInfer 的 block_size 与显存预算。

    DeepInfer 生产配置中块大小为 128 token；miniflow 用小得多的块以便观察行为。
    """

    block_size: int = 4
    num_blocks: int = 256          # KV 内存预算：num_blocks * block_size 个 token 槽位
    prefix_cache_enabled: bool = True


@dataclass
class EngineConfig:
    """引擎行为开关 —— 对应 DEEPINFER_* 环境变量体系。"""

    async_scheduling: bool = field(default_factory=lambda: env_flag("MF_ASYNC_SCHED", True))
    speculative_decoding: bool = field(default_factory=lambda: env_flag("MF_SPEC_DECODING", False))
    max_draft_tokens: int = int(os.environ.get("MF_MAX_DRAFT_TOKENS", "5"))
    batch_invariant: bool = field(default_factory=lambda: env_flag("MF_BATCH_INVARIANT", True))

    # 调度器每步最多接受的 token 数（预填充切块预算的雏形，
    # 对应 DeepInfer 用二次成本模型求解 chunk 大小的思想）
    max_tokens_per_step: int = 64


@dataclass
class MiniFlowConfig:
    """聚合配置。"""

    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)