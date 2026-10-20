"""微型训练器 —— 为 miniflow 生成可推理的 checkpoint。

教学定位：这里刻意展示「训练前向」与「推理前向」的差异 ——
  * 训练：batch 维稠密注意力（整段矩阵乘，反向传播），无 KV cache；
  * 推理：增量 token + KV cache（单 token 逐步），见 model/transformer.py。
这正是「推理框架」作为独立学科存在的理由：训练引擎的负载形态
（大 batch、稠密、无状态）与服务负载（变长、增量、长寿命状态）截然不同。

配置说明：训练用 dense FFN + 全注意力（moe_layers/swa_layers 为空）——
MoE/SWA 的运行机制由 examples 里的 demo 用随机权重演示（机制与语义无关）。

手动反向传播覆盖：RMSNorm / RoPE（正交变换，反向即转置旋转）/
GQA 注意力 / SwiGLU / tied lm_head。优化器：手写 Adam。

用法：
    python tools/train.py --out checkpoints/tiny --steps 400
产物：
    checkpoints/tiny.npz        模型权重
    checkpoints/tiny.tok.json   分词器
    checkpoints/tiny.cfg.json   模型配置（推理引擎自动读取）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniflow.config import ModelConfig  # noqa: E402
from miniflow.model.transformer import TransformerModel  # noqa: E402
from miniflow.tokenizer import Tokenizer  # noqa: E402

CORPUS = (
    "the cat sat on the mat. the dog ran in the park. "
    "the bird flew over the tree. the fish swam in the pond. "
    "the cat ran in the park. the dog sat on the mat. "
    "the bird swam in the pond. the fish flew over the tree. "
) * 8


# ----------------------------------------------------------------------
# 基础算子（训练版，含反向）
# ----------------------------------------------------------------------

def _rope(x: np.ndarray, theta: float, inverse: bool = False) -> np.ndarray:
    """x: [B, T, H, D]。RoPE 的反向 = 角度取负（正交变换）。"""
    b, t, h, d = x.shape
    inv_freq = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
    ang = np.arange(t, dtype=np.float32)[:, None] * inv_freq[None, :]   # [T, D/2]
    sign = -1.0 if inverse else 1.0
    cos, sin = np.cos(ang), np.sin(ang) * sign
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x_even * cos[None, :, None, :] - x_odd * sin[None, :, None, :]
    out[..., 1::2] = x_even * sin[None, :, None, :] + x_odd * cos[None, :, None, :]
    return out


def _rms_fwd(x, w, eps):
    s = np.mean(x ** 2, axis=-1, keepdims=True) + eps
    inv = 1.0 / np.sqrt(s)
    return x * w * inv, inv


def _rms_bwd(x, w, inv, dy):
    """y = x*w*inv, inv = (mean(x^2)+eps)^-1/2 的反向。

    dx_j = dy_j w_j inv - (x_j inv^3 / N) * sum_i(dy_i w_i x_i)
    """
    n = x.shape[-1]
    g = dy * w * inv
    a = np.sum(dy * w * x, axis=-1, keepdims=True)
    dx = g - x * (a * inv ** 3 / n)
    dw = np.sum(dy * x * inv, axis=(0, 1))
    return dx, dw


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _silu_grad(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


# ----------------------------------------------------------------------
# 前向 + 反向
# ----------------------------------------------------------------------

def forward_backward(model: TransformerModel, ids: np.ndarray):
    """一次训练前向+反向。ids: [B, T]。返回 (loss, grads dict)。"""
    c = model.cfg
    w = model.w
    B, T = ids.shape
    eps = c.rms_norm_eps
    grads = {k: np.zeros_like(v) for k, v in w.items()}
    causal = np.tril(np.ones((T, T), dtype=bool))
    scale = 1.0 / np.sqrt(c.head_dim)

    x = w["embed"][ids]                                     # [B, T, H]
    cache = []

    for layer in range(c.num_layers):
        x_in = x
        h1, inv1 = _rms_fwd(x_in, w[f"l{layer}.norm1"], eps)
        q = (h1 @ w[f"l{layer}.attn.q"]).reshape(B, T, c.num_attention_heads, c.head_dim)
        k = (h1 @ w[f"l{layer}.attn.k"]).reshape(B, T, c.num_kv_heads, c.head_dim)
        v = (h1 @ w[f"l{layer}.attn.v"]).reshape(B, T, c.num_kv_heads, c.head_dim)

        q_r, k_r = _rope(q, c.rope_theta), _rope(k, c.rope_theta)
        rep = c.num_attention_heads // c.num_kv_heads
        k_e = np.repeat(k_r, rep, axis=2)                   # [B, T, H, D]
        v_e = np.repeat(v, rep, axis=2)

        att = np.einsum("bqhd,bkhd->bhqk", q_r, k_e) * scale
        att = np.where(causal[None, None], att, -np.inf)
        m = att.max(axis=-1, keepdims=True)
        e = np.exp(att - m)
        P = e / e.sum(axis=-1, keepdims=True)               # [B, H, T, T]
        O = np.einsum("bhqk,bkhd->bqhd", P, v_e)            # [B, T, H, D]
        O_flat = O.reshape(B, T, -1)
        attn_out = O_flat @ w[f"l{layer}.attn.o"]
        x_mid = x_in + attn_out

        h2, inv2 = _rms_fwd(x_mid, w[f"l{layer}.norm2"], eps)
        gate = h2 @ w[f"l{layer}.ffn.w1"]
        up = h2 @ w[f"l{layer}.ffn.w3"]
        act = _silu(gate) * up
        ff = act @ w[f"l{layer}.ffn.w2"]
        x = x_mid + ff

        cache.append(dict(x_in=x_in, h1=h1, inv1=inv1, q_r=q_r, k_e=k_e, v_e=v_e,
                          P=P, O_flat=O_flat, x_mid=x_mid, h2=h2, inv2=inv2,
                          gate=gate, up=up, act=act))

    xf, invf = _rms_fwd(x, w["final_norm"], eps)
    logits = xf @ w["embed"].T

    tgt = ids[:, 1:]
    lg = logits[:, :-1]                                     # [B, T-1, V]
    m = lg.max(axis=-1, keepdims=True)
    lse = (m + np.log(np.exp(lg - m).sum(axis=-1, keepdims=True))).squeeze(-1)
    loss = float(np.mean(lse - np.take_along_axis(
        lg, tgt[..., None], axis=-1).squeeze(-1)))

    # ---- 反向 ----
    # dL/dlogits = (softmax - onehot) / (B*(T-1))：mean loss 对每个元素除以总数，
    # one-hot 项同样要除 —— 漏掉它会让 target 位置的梯度大 B*(T-1) 倍。
    dlogits = np.exp(lg - lse[..., None]) / (B * (T - 1))
    np.put_along_axis(
        dlogits, tgt[..., None],
        np.take_along_axis(dlogits, tgt[..., None], axis=-1)
        - 1.0 / (B * (T - 1)), axis=-1)

    dx = np.zeros_like(xf)
    dx[:, :-1] = dlogits @ w["embed"]                         # embed 是 [V,H]：dxf = dlogits @ embed
    grads["embed"] += np.einsum("btv,bth->vh", dlogits, xf[:, :-1])

    dx, dwf = _rms_bwd(x, w["final_norm"], invf, dx)
    grads["final_norm"] += dwf

    for layer in reversed(range(c.num_layers)):
        cc = cache[layer]
        Wo = w[f"l{layer}.attn.o"]

        # ---- FFN 块（输入 x_mid，输出 x_out = x_mid + ff）----
        dff = dx                                            # 残差直通
        grads[f"l{layer}.ffn.w2"] += np.einsum("bti,bto->io", cc["act"], dff)
        dact = dff @ w[f"l{layer}.ffn.w2"].T
        # act = silu(gate) * up 是乘积：过 gate 的链式法则要带上 up 因子，
        # 过 up 的链式法则带上 silu(gate) 因子（两项互为对方的「另一个乘数」）
        dgate = dact * _silu_grad(cc["gate"]) * cc["up"]
        dup = dact * _silu(cc["gate"])
        grads[f"l{layer}.ffn.w1"] += np.einsum("bth,bti->hi", cc["h2"], dgate)
        grads[f"l{layer}.ffn.w3"] += np.einsum("bth,bti->hi", cc["h2"], dup)
        dh2 = dgate @ w[f"l{layer}.ffn.w1"].T + dup @ w[f"l{layer}.ffn.w3"].T
        dx_mid, dnorm2 = _rms_bwd(cc["x_mid"], w[f"l{layer}.norm2"],
                                  cc["inv2"], dh2)
        grads[f"l{layer}.norm2"] += dnorm2
        dx = dx + dx_mid                                    # x_mid 总梯度

        # ---- 注意力块（输入 x_in，输出 x_mid = x_in + attn_out）----
        grads[f"l{layer}.attn.o"] += np.einsum("btd,bth->dh", cc["O_flat"], dx)
        dO = (dx @ Wo.T).reshape(B, T, c.num_attention_heads, c.head_dim)
        dP = np.einsum("bqhd,bkhd->bhqk", dO, cc["v_e"])
        dV_e = np.einsum("bhqk,bqhd->bkhd", cc["P"], dO)
        dS = cc["P"] * (dP - (dP * cc["P"]).sum(axis=-1, keepdims=True))
        dS = np.where(causal[None, None], dS, 0.0)
        dQ = np.einsum("bhqk,bkhd->bqhd", dS, cc["k_e"]) * scale
        dK_e = np.einsum("bhqk,bqhd->bkhd", dS, cc["q_r"]) * scale
        # GQA 折叠：heads -> kv_heads（rep 维求和）
        dK = dK_e.reshape(B, T, c.num_kv_heads, rep, c.head_dim).sum(axis=3)
        dV = dV_e.reshape(B, T, c.num_kv_heads, rep, c.head_dim).sum(axis=3)
        # RoPE 反向（正交变换：角度取负）
        dq = _rope(dQ, c.rope_theta, inverse=True)
        dk = _rope(dK, c.rope_theta, inverse=True)
        dq_flat, dk_flat, dv_flat = (dq.reshape(B, T, -1), dk.reshape(B, T, -1),
                                     dV.reshape(B, T, -1))
        grads[f"l{layer}.attn.q"] += np.einsum("bth,btd->hd", cc["h1"], dq_flat)
        grads[f"l{layer}.attn.k"] += np.einsum("bth,btd->hd", cc["h1"], dk_flat)
        grads[f"l{layer}.attn.v"] += np.einsum("bth,btd->hd", cc["h1"], dv_flat)
        dh1 = (dq_flat @ w[f"l{layer}.attn.q"].T
               + dk_flat @ w[f"l{layer}.attn.k"].T
               + dv_flat @ w[f"l{layer}.attn.v"].T)
        dx_in, dnorm1 = _rms_bwd(cc["x_in"], w[f"l{layer}.norm1"],
                                 cc["inv1"], dh1)
        grads[f"l{layer}.norm1"] += dnorm1
        dx = dx + dx_in                                     # x_in 总梯度

    np.add.at(grads["embed"], ids, dx)                      # embedding 查表反向
    return loss, grads


# ----------------------------------------------------------------------
# Adam 与主循环
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params: dict, lr=3e-3, betas=(0.9, 0.95), eps=1e-8):
        self.p, self.lr, self.b, self.eps = params, lr, betas, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict) -> None:
        self.t += 1
        b1, b2 = self.b
        for k, g in grads.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/tiny")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    tok = Tokenizer.train(CORPUS, target_vocab=160)
    cfg = ModelConfig(vocab_size=tok.vocab_size, hidden_size=64, num_layers=4,
                      num_attention_heads=4, num_kv_heads=2,
                      moe_layers=(), swa_layers=(),
                      intermediate_size=128, max_position_embeddings=256)
    model = TransformerModel.random_init(cfg, seed=args.seed)

    data = np.asarray(tok.encode(CORPUS), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    opt = Adam(model.w, lr=args.lr)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        starts = rng.integers(0, len(data) - args.seq_len - 1, size=args.batch)
        ids = np.stack([data[s:s + args.seq_len] for s in starts])
        loss, grads = forward_backward(model, ids)
        if not all(np.all(np.isfinite(g)) for g in grads.values()):
            print(f"step {step}: 梯度异常，跳过本步")
            continue
        opt.step(grads)
        if step % 50 == 0 or step == 1:
            print(f"step {step:4d}  loss {loss:.4f}  ({time.time() - t0:.1f}s)")

    model.save(args.out + ".npz")
    tok.save(args.out + ".tok.json")
    with open(args.out + ".cfg.json", "w", encoding="utf-8") as f:
        json.dump({"vocab_size": cfg.vocab_size, "hidden_size": cfg.hidden_size,
                   "num_layers": cfg.num_layers,
                   "num_attention_heads": cfg.num_attention_heads,
                   "num_kv_heads": cfg.num_kv_heads,
                   "moe_layers": list(cfg.moe_layers),
                   "swa_layers": list(cfg.swa_layers),
                   "intermediate_size": cfg.intermediate_size,
                   "max_position_embeddings": cfg.max_position_embeddings},
                  f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {args.out}.npz / .tok.json / .cfg.json")
    print(f"vocab size: {tok.vocab_size}, params: {model.num_parameters()}")


if __name__ == "__main__":
    main()