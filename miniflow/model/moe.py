"""模型层（二）：MoE 与 EPLB —— 对应 V4.1 的专家层与 DeepInfer 的专家负载均衡。

**MoE**：token 只路由到 top-k 个专家（V4.1：细粒度专家 + 共享专家），
算力随 token 数恒定、参数容量随专家数扩展 —— 解耦「容量」与「计算」。

**EPLB（Expert-Parallel Load Balancer）**：专家热度天然倾斜（幂律），
EP 部署下最热的卡拖慢整个 step（木桶效应）。DeepInfer 的解法：
  * 每卡在「本职专家」外挂少量 **冗余专家** 槽位，存放热门专家副本；
  * 统计最近 N 步各专家负载，每 M 步做一次全局重分配
    （prefill：4 冗余/卡、100 步统计窗口、1000 步重分配；
     decode：1 冗余/卡、1000 步窗口、10000 步重分配）；
  * 实测 moe_lb（最慢卡/均值）压到 1.045~1.09。

本文件内置一个可观测的 EPLB 演示：`ExpertBalancer` 统计负载窗口，
按「最少负载优先」重排专家->卡的映射，并输出 moe_lb 前后对比。
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple

import numpy as np

from ..tensor import matmul, softmax
from .layers import silu


class MoELayer:
    def __init__(self, gate_w: np.ndarray,
                 experts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 top_k: int):
        self.gate_w = gate_w                # [hidden, num_experts]
        self.experts = experts              # 每个专家 (w1, w3, w2) SwiGLU
        self.top_k = top_k

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (输出, 路由矩阵 [n, num_experts]，0/稀疏，含权重)。

        稀疏执行的教学形态：逐 token 只算被选中的专家 —— 生产上这由
        DeepEP 的 all-to-all 完成：token 按 expert 归属发给对应卡，
        算完再发回来（通信与计算重叠是 Wide-EP 的命门）。
        """
        n = x.shape[0]
        route = softmax(x @ self.gate_w, axis=-1)          # [n, E]
        topk_idx = np.argsort(route, axis=-1)[:, -self.top_k:]   # [n, k]
        out = np.zeros((n, x.shape[1]), dtype=np.float32)
        for i in range(n):
            for e in topk_idx[i]:
                w = route[i, e]
                w1, w3, w2 = self.experts[int(e)]
                gate = x[i] @ w1
                up = x[i] @ w3
                out[i] += w * ((silu(gate) * up) @ w2)
        return out, route


class ExpertBalancer:
    """EPLB 教学演示：负载统计窗口 + 冗余专家重分配。

    对应 DeepInfer 的两个关键参数：统计窗口 N 步、重分配周期 M 步。
    """

    def __init__(self, num_experts: int, world_size: int,
                 redundant_per_rank: int = 1, window: int = 100):
        self.num_experts = num_experts
        self.world_size = world_size
        self.redundant = redundant_per_rank
        self.window = window
        self.load_history: deque = deque(maxlen=window)
        # expert_id -> 槽位（槽位 = rank * slots_per_rank + slot）；
        # 热专家可占多个槽（副本），每个槽由一个 rank 持有
        self.placement: List[int] = list(range(num_experts))
        self.replica_of: Dict[int, int] = {}      # 槽位 -> 原专家 id

    def observe(self, route: np.ndarray, top_k: int) -> None:
        """收集一步的专家负载（token 数）。"""
        topk = np.argsort(route, axis=-1)[:, -top_k:]
        load = np.bincount(topk.reshape(-1), minlength=self.num_experts)
        self.load_history.append(load)

    def current_load(self) -> np.ndarray:
        if not self.load_history:
            return np.zeros(self.num_experts, dtype=np.int64)
        return np.sum(np.stack(list(self.load_history)), axis=0)

    def moe_lb(self) -> float:
        """DeepInfer 的均衡度指标：最慢 rank 负载 / 平均负载（越接近 1 越好）。"""
        load = self._rank_load(self.current_load(), self.placement)
        return float(np.max(load) / max(np.mean(load), 1e-9))

    def _rank_load(self, expert_load: np.ndarray, placement: List[int]) -> np.ndarray:
        per_rank = np.zeros(self.world_size, dtype=np.float64)
        for slot, expert in enumerate(placement):
            rank = slot % self.world_size
            per_rank[rank] += expert_load[expert] if expert < len(expert_load) else 0
        return per_rank

    def rebalance(self) -> List[int]:
        """重分配：热门专家复制到冗余槽位，按「最少负载优先」摆放。

        真实 EPLB 的约束：专家权重在重分配时要经 NVLink symm_mem 搬运，
        因此重分配周期 M 不能太频繁（成本收益权衡，见文档站）。
        """
        load = self.current_load().astype(np.float64)
        total_slots = self.num_experts + self.redundant * self.world_size
        # 1) 排序：热度降序；热门专家优先分配副本
        order = np.argsort(load)[::-1]
        placement: List[int] = [-1] * total_slots
        assigned: Dict[int, int] = {int(e): 0 for e in range(self.num_experts)}
        # 2) 贪心：按热度给专家分配槽位（副本数 = 冗余上限内尽量均衡热度/负载）
        rank_load = np.zeros(self.world_size, dtype=np.float64)
        for expert in order:
            expert = int(expert)
            copies = 1
            if load[expert] > 0:
                share = load[expert] / (np.sum(load) / total_slots)
                copies = max(1, min(1 + self.redundant, int(round(share))))
            for _ in range(copies):
                rank = int(np.argmin(rank_load))
                slot = next(s for s in range(total_slots) if placement[s] == -1)
                placement[slot] = expert
                rank_load[rank] += load[expert] / copies
        self.placement = placement
        return placement

    def stats(self) -> dict:
        return {"moe_lb": round(self.moe_lb(), 4),
                "window": len(self.load_history),
                "placement": list(self.placement)}