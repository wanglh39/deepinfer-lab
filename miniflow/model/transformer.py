"""模型层（三）：Transformer 整合 —— 对应 model_executor/models/ 的模型实现。

把前面的积木拼成完整前向：

    token_ids
      -> embedding
      -> N x [ RMSNorm -> GQA/SWA 注意力(读写 PagedKVCache) -> 残差
               RMSNorm -> dense FFN 或 MoE -> 残差 ]
      -> RMSNorm -> lm_head -> logits [n, vocab]

引擎层要的两个接口约定：
  * `forward(token_ids, seq_id, kv)` 只前向 **增量部分**：
    起点 = 该序列已算进度（kv.computed），prompt 未算的后缀 / decode 的
    新 token —— 断点续算、推测解码回滚都建立在「进度 + 覆盖写」上；
  * 权重用扁平 dict 保存/加载（npz），键形如 "l2.attn.q"。

权重绑定：lm_head 复用 embedding（减少参数量，训练更快）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..cache.paged_kv import PagedKVCache
from ..config import ModelConfig
from ..tensor import rms_norm
from .layers import attention_forward, dense_ffn
from .moe import MoELayer


def _rand(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return (rng.standard_normal(shape) * 0.02).astype(np.float32)


class TransformerModel:
    def __init__(self, config: ModelConfig, weights: Dict[str, np.ndarray]):
        self.cfg = config
        self.w = weights

    # ------------------------------------------------------------------
    @classmethod
    def random_init(cls, config: ModelConfig, seed: int = 0) -> "TransformerModel":
        rng = np.random.default_rng(seed)
        c = config
        w: Dict[str, np.ndarray] = {}
        w["embed"] = _rand(rng, c.vocab_size, c.hidden_size)
        for i in range(c.num_layers):
            w[f"l{i}.attn.q"] = _rand(rng, c.hidden_size, c.num_attention_heads * c.head_dim)
            w[f"l{i}.attn.k"] = _rand(rng, c.hidden_size, c.num_kv_heads * c.head_dim)
            w[f"l{i}.attn.v"] = _rand(rng, c.hidden_size, c.num_kv_heads * c.head_dim)
            w[f"l{i}.attn.o"] = _rand(rng, c.num_attention_heads * c.head_dim, c.hidden_size)
            w[f"l{i}.norm1"] = np.ones(c.hidden_size, dtype=np.float32)
            w[f"l{i}.norm2"] = np.ones(c.hidden_size, dtype=np.float32)
            if c.uses_moe(i):
                w[f"l{i}.moe.gate"] = _rand(rng, c.hidden_size, c.num_experts)
                for e in range(c.num_experts):
                    w[f"l{i}.moe.e{e}.w1"] = _rand(rng, c.hidden_size, c.intermediate_size)
                    w[f"l{i}.moe.e{e}.w3"] = _rand(rng, c.hidden_size, c.intermediate_size)
                    w[f"l{i}.moe.e{e}.w2"] = _rand(rng, c.intermediate_size, c.hidden_size)
            else:
                w[f"l{i}.ffn.w1"] = _rand(rng, c.hidden_size, c.intermediate_size)
                w[f"l{i}.ffn.w3"] = _rand(rng, c.hidden_size, c.intermediate_size)
                w[f"l{i}.ffn.w2"] = _rand(rng, c.intermediate_size, c.hidden_size)
        w["final_norm"] = np.ones(c.hidden_size, dtype=np.float32)
        return cls(config, w)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez_compressed(path, **self.w)

    @classmethod
    def load(cls, path: str, config: ModelConfig) -> "TransformerModel":
        with np.load(path) as z:
            w = {k: z[k].astype(np.float32) for k in z.files}
        return cls(config, w)

    # ------------------------------------------------------------------
    def forward(self, token_ids: List[int], seq_id: int,
                kv: PagedKVCache, moe_observer=None) -> np.ndarray:
        """前向该序列的增量 token，返回 **每个增量 token** 的 logits [n, vocab]。

        增量区间 = [kv.computed[seq][0], len(token_ids))。
        KV cache 需已 ensure_capacity 到 len(token_ids)。
        """
        c = self.cfg
        start = kv.computed[seq_id][0]
        n = len(token_ids) - start
        assert n > 0, "forward 需要至少一个新 token（增量契约）"
        new_ids = np.asarray(token_ids[start:], dtype=np.int64)
        positions = np.arange(start, len(token_ids), dtype=np.int64)

        x = self.w["embed"][new_ids]

        for layer in range(c.num_layers):
            h = rms_norm(x, self.w[f"l{layer}.norm1"], c.rms_norm_eps)

            q = (h @ self.w[f"l{layer}.attn.q"]
                 ).reshape(n, c.num_attention_heads, c.head_dim)
            k = (h @ self.w[f"l{layer}.attn.k"]
                 ).reshape(n, c.num_kv_heads, c.head_dim)
            v = (h @ self.w[f"l{layer}.attn.v"]
                 ).reshape(n, c.num_kv_heads, c.head_dim)

            hist_k, hist_v, hist_pos = kv.get_kv(layer, seq_id)
            window = c.swa_window if c.uses_swa(layer) else None
            attn_out, k_rot = attention_forward(
                q, hist_k, hist_v, hist_pos, k, v, positions,
                num_kv_heads=c.num_kv_heads, window=window,
            )
            x = x + attn_out @ self.w[f"l{layer}.attn.o"]

            h = rms_norm(x, self.w[f"l{layer}.norm2"], c.rms_norm_eps)
            if c.uses_moe(layer):
                experts = [(self.w[f"l{layer}.moe.e{e}.w1"],
                            self.w[f"l{layer}.moe.e{e}.w3"],
                            self.w[f"l{layer}.moe.e{e}.w2"])
                           for e in range(c.num_experts)]
                moe = MoELayer(self.w[f"l{layer}.moe.gate"], experts,
                               c.num_experts_per_tok)
                ff, route = moe.forward(h)
                if moe_observer is not None:
                    moe_observer(route, c.num_experts_per_tok)
            else:
                ff = dense_ffn(h, self.w[f"l{layer}.ffn.w1"],
                               self.w[f"l{layer}.ffn.w3"],
                               self.w[f"l{layer}.ffn.w2"])
            x = x + ff

            # 注意：写入的是 RoPE 之后的 K（数值已烘焙位置信息）
            kv.append(layer, seq_id, start, k_rot, v)

        x = rms_norm(x, self.w["final_norm"], c.rms_norm_eps)
        logits = x @ self.w["embed"].T
        return logits

    # ------------------------------------------------------------------
    def num_parameters(self) -> int:
        return int(sum(v.size for v in self.w.values()))