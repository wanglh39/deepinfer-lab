"""模型层（一）：注意力与 FFN —— 对应 model_executor/layers/ 的注意力实现。

三个教学重点：

1. **GQA**（DeepSeek MLA 的精神内核之一）：
   Q 头数 > KV 头数。KV cache 只存 kv_heads 份 K/V —— 缓存体积按比例缩小。
   MLA 走得更远：把 K/V 联合压缩成一个低秩 latent 向量（缓存再缩一个量级，
   并支持「吸收」推理），miniflow 用 GQA 讲同一件事的经济账。

2. **SWA 滑动窗口注意力**（V4.1 Flash 的稀疏注意力层）：
   只允许 query 看最近 `window` 个 token —— 每层注意力代价 O(seq * window)。
   它是「SWA KV cache 只存窗口」与「有界重放」的前提（见 cache/）。

3. **读 KV cache 的方式**：
   prefill 整段并行（算力受限）；decode 每步 query=1 读全部历史（访存受限）
   —— 这就是 E/P/D 分离部署的负载根源（文档站《调度器》一章的出发点）。
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..tensor import apply_rope, softmax


def attention_forward(
    q: np.ndarray,
    hist_k: np.ndarray, hist_v: np.ndarray, hist_pos: np.ndarray,
    new_k: np.ndarray, new_v: np.ndarray, new_pos: np.ndarray,
    num_kv_heads: int,
    window: Optional[int] = None,
) -> np.ndarray:
    """单层注意力前向（GQA + 因果 + 可选 SWA 掩码）。

    参数
    ----
    q: [nq, heads, dim]                 本步需要前向的 token 的 query
    hist_k/v: [T, kv_heads, dim]        该序列已缓存的 K/V（PagedKVCache 读出）
    hist_pos: [T]                       缓存 token 的绝对位置
    new_k/v: [nq, kv_heads, dim]        本步 token 的新 K/V（将由调用方写入 cache）
    new_pos: [nq]                       新 token 的绝对位置（RoPE 与因果掩码共用）

    返回 (attention 输出 [nq, heads * dim], 旋转后的 K [nq, kv_heads, dim])。

    第二个返回值是给调用方写入 KV cache 用的：**cache 必须存 RoPE 之后的 K**
    （位置信息烘焙进数值，读出即可直接参与点积 —— 生产核如 FlashMLA 在核内
    完成同样的融合，避免额外的位置查表访存）。
    """
    q_rot, k_rot = apply_rope(q, new_k, new_pos)

    # 历史 + 新 K/V 拼成完整 key/value 序列
    keys = np.concatenate([hist_k, k_rot], axis=0)     # [T+nq, kv_heads, dim]
    values = np.concatenate([hist_v, new_v], axis=0)
    key_pos = np.concatenate([hist_pos, new_pos])      # [T+nq]

    # GQA 广播：kv_heads -> heads（MLA「吸收」的低配版：共享 KV，多头读）
    heads = q.shape[1]
    rep = heads // num_kv_heads
    keys = np.repeat(keys, rep, axis=1)                # [T+nq, heads, dim]
    values = np.repeat(values, rep, axis=1)

    # scores: [nq, heads, T+nq]
    scores = np.einsum("qhd,thd->qht", q_rot, keys) / np.sqrt(q.shape[-1])

    # 因果 + SWA 掩码：位置 p 的 query 只看 <= p（且 >= p-window+1）的 key
    q_pos = new_pos[:, None, None]
    k_pos_all = key_pos[None, None, :]
    mask = k_pos_all <= q_pos
    if window is not None:
        mask &= k_pos_all > (q_pos - window)
    scores = np.where(mask, scores, -np.inf)

    probs = softmax(scores, axis=-1)
    out = np.einsum("qht,thd->qhd", probs, values)
    return out.reshape(q.shape[0], -1), k_rot


def dense_ffn(x: np.ndarray, w1: np.ndarray, w3: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """SwiGLU FFN：down(silu(gate(x)) * up(x)) —— DeepSeek 系列的 FFN 形态。"""
    gate = x @ w1
    up = x @ w3
    return (silu(gate) * up) @ w2


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))