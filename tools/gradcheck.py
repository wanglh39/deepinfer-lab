"""数值梯度校验 —— 验证 tools/train.py 的手写反向传播。

教学定位：DeepInfer 这类推理框架不需要反向，但「从零复刻」的实验室必须
证明自己前向/反向一致，否则训练出的 checkpoint 毫无意义。方法是经典的
中心差分：对每个参数张量随机抽取若干坐标，比较
    (L(w+eps) - L(w-eps)) / 2eps   vs   解析梯度
相对误差 < 1e-4 视为通过（float32 下用 1e-3 更稳妥）。

用法：
    python tools/gradcheck.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniflow.config import ModelConfig  # noqa: E402
from miniflow.model.transformer import TransformerModel  # noqa: E402
from tools.train import forward_backward  # noqa: E402


def numerical_grad(model, ids, key, idx, eps=1e-3):
    saved = model.w[key][idx]
    model.w[key][idx] = saved + eps
    lp, _ = forward_backward(model, ids)
    model.w[key][idx] = saved - eps
    lm, _ = forward_backward(model, ids)
    model.w[key][idx] = saved
    return (lp - lm) / (2 * eps)


def main() -> None:
    rng = np.random.default_rng(7)
    cfg = ModelConfig(vocab_size=17, hidden_size=32, num_layers=2,
                      num_attention_heads=4, num_kv_heads=2,
                      moe_layers=(), swa_layers=(),
                      intermediate_size=48, max_position_embeddings=64)
    model = TransformerModel.random_init(cfg, seed=7)
    for k in model.w:  # 提升到 float64：中心差分需要比 float32 更高的分辨率
        model.w[k] = model.w[k].astype(np.float64)
    ids = rng.integers(0, cfg.vocab_size, size=(2, 10))

    _, grads = forward_backward(model, ids)

    worst, worst_key = 0.0, ""
    for key in sorted(model.w.keys()):
        flat = model.w[key].reshape(-1)
        picks = rng.choice(flat.size, size=min(4, flat.size), replace=False)
        for i in picks:
            idx = np.unravel_index(i, model.w[key].shape)
            num = numerical_grad(model, ids, key, idx)
            ana = float(grads[key][idx])
            rel = abs(num - ana) / max(1e-6, abs(num) + abs(ana))
            if rel > worst:
                worst, worst_key = rel, f"{key}{idx}"
    print(f"最大相对误差: {worst:.2e}  ({worst_key})")
    assert worst < 1e-3, "梯度校验失败"
    print("梯度校验通过 ✓")


if __name__ == "__main__":
    main()