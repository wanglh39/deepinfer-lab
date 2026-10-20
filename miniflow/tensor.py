"""算子层 —— 对应 DeepInfer 的 FlashMLA / DeepGEMM / DeepEP / TileLang 算子库。

DeepInfer 的关键认知：**推理框架的性能上限由算子决定，引擎层的调度与并行
都是为了让算子「吃饱」**。FlashMLA（MLA 注意力核）、DeepGEMM（矩阵乘）、
DeepEP（MoE all-to-all 通信）每一块都是为特定负载手写/TileLang 生成的核。

miniflow 的教学结构：
  * 参考实现：NumPy（本文件），逻辑清晰、可单测；
  * 加速实现：csrc/kernel.cpp 的 C++ 版本经 pybind11 编译为 miniflow._C；
  * 派发：`_backend()` 探测可用后端 —— 这对应生产框架里「多算子后端
    按硬件/负载自动选择」的机制（NVIDIA/昇腾双平台支持的基础）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:                                    # 对应 deepinfer/_C：可选的 C++ 扩展
    import miniflow._C as _C            # type: ignore
    BACKEND = "cpp"
except ImportError:
    _C = None
    BACKEND = "numpy"


def backend() -> str:
    """当前算子后端（'cpp' 或 'numpy'）—— 用于教学演示与基准对比。"""
    return BACKEND


# ----------------------------------------------------------------------
# 矩阵乘：一切的开端（DeepGEMM 的领地）
# ----------------------------------------------------------------------

def naive_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """教科书式三重循环实现 —— 看清 matmul 的本质：O(M*K*N) 次乘加。

    生产框架不会这样写；这里仅作为教学基准，用于展示「算子实现差 100 倍」
    这一事实（对比 numpy 的 BLAS 与本函数的耗时即可体会）。
    """
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"shape mismatch: {a.shape} @ {b.shape}"
    out = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for t in range(k):
                acc += a[i, t] * b[t, j]
            out[i, j] = acc
    return out


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """矩阵乘派发：C++ 后端可用则用之，否则 NumPy(BLAS)。"""
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    if _C is not None:
        return _C.matmul(a, b)
    return a @ b


# ----------------------------------------------------------------------
# Softmax：注意力的心脏（FlashMLA 里的在线 softmax）
# ----------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax。

    FlashMLA 在 GPU 上用「在线 softmax」分块完成同样的数学，
    避免 O(seq^2) 显存的注意力矩阵 —— 见文档站《内核层》一章。
    """
    x = x.astype(np.float32, copy=False)
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm（DeepSeek 系列全部用它，代替 LayerNorm）。

    y = x / sqrt(mean(x^2) + eps) * weight
    相比 LayerNorm 少一次均值中心化，kernel 更简单、更快。
    """
    ms = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


# ----------------------------------------------------------------------
# RoPE：位置编码（DeepInfer 在 MLA/注意力的核内融合）
# ----------------------------------------------------------------------

def apply_rope(q: np.ndarray, k: np.ndarray, positions: np.ndarray,
               theta: float = 10000.0) -> tuple:
    """旋转位置编码。q: [seq, heads, dim], k: [seq, kv_heads, dim]。

    只对奇偶维做旋转配对；返回旋转后的 (q, k)。
    """
    def rot(x: np.ndarray) -> np.ndarray:
        seq, heads, dim = x.shape
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        ang = positions[:, None].astype(np.float32) * inv_freq[None, :]  # [seq, dim/2]
        cos, sin = np.cos(ang), np.sin(ang)
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2] = x_even * cos[:, None, :] - x_odd * sin[:, None, :]
        out[..., 1::2] = x_even * sin[:, None, :] + x_odd * cos[:, None, :]
        return out

    return rot(q), rot(k)


# ----------------------------------------------------------------------
# 检索工具：给采样器/推测解码用
# ----------------------------------------------------------------------

def top_k_top_p_filter(logits: np.ndarray, k: int, p: float) -> np.ndarray:
    """top-k / top-p 截断：返回过滤后的 logits（被截断位置 = -inf）。"""
    logits = logits.astype(np.float32, copy=True)
    if k > 0:
        kth = np.sort(logits)[-min(k, logits.shape[-1])]
        logits[logits < kth] = -np.inf
    if p < 1.0:
        order = np.argsort(logits)[::-1]
        probs = softmax(logits[order])
        cum = np.cumsum(probs)
        cut = int(np.searchsorted(cum, p)) + 1
        keep = order[:cut]
        mask = np.full_like(logits, -np.inf)
        mask[keep] = logits[keep]
        logits = mask
    return logits


def argmax(logits: np.ndarray) -> int:
    return int(np.argmax(logits))


def greedy_verify_token(logits: np.ndarray, token: int) -> bool:
    return argmax(logits) == token