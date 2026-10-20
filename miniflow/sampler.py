"""采样器 —— 对应 DeepInfer 的「确定性采样 + 批次不变推理」。

**为什么难？** 常规做法把整批 logits 拼成矩阵后统一采样：top-p 的截断、
浮点求和顺序、批维归约都会让「同一条请求」的结果依赖批内其他请求 ——
换一批邻居，输出就变。这在生产上意味着：回归测试不可复现、A/B 对比失真、
分片重放对不齐。

**DeepInfer 的答案（miniflow 复刻）**：
  1. **逐请求独立随机流**：每条序列持有自己的 RNG，种子 = seq_seed
     （用户可显式传入 SamplingParams）。采样只消费自己流里的随机数 ——
     与批组合、批顺序都无关；
  2. **逐请求独立归约**：softmax/top-p 截断只在该请求自己的 logits 上做，
     任何批维的向量化归约都被禁止（教学版因此逐条循环，性能不重要）；
  3. `MF_BATCH_INVARIANT=0` 可关闭以对比「非不变」版的行为差异
     （演示批组成如何影响输出）。

确定性组合拳（README 同款）：MF_BATCH_INVARIANT=1 + 固定 seq_seed
+ 关闭推测解码 => 相同输入 token 必得相同输出。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .sequence import Sequence
from .tensor import softmax, top_k_top_p_filter


class Sampler:
    def __init__(self, batch_invariant: bool = True):
        self.batch_invariant = batch_invariant
        # 非不变模式的「批级」RNG：批组成变化 -> 随机流错位（教学对照）
        self.shared_rng = np.random.default_rng(1234)

    # ------------------------------------------------------------------
    def _rng_for(self, seq: Sequence) -> np.random.Generator:
        seed = seq.sampling_params.seq_seed
        if seed is None:
            seed = 0
        # 种子混合 seq_id：同 seed 下不同请求的流也互相独立
        return np.random.default_rng([int(seed), int(seq.seq_id)])

    # ------------------------------------------------------------------
    def sample(self, batch: List[Tuple[Sequence, np.ndarray]]
               ) -> List[Tuple[Sequence, int, float]]:
        """对一个 batch 逐条采样。返回 [(seq, token_id, logprob)]。

        batch: [(seq, logits)]，logits 是该序列最后一步的 [vocab]。
        """
        results: List[Tuple[Sequence, int, float]] = []
        if not self.batch_invariant:
            # 对照组：批级 RNG（批大小/顺序变化 => 输出漂移）
            for seq, logits in batch:
                params = seq.sampling_params
                filt = top_k_top_p_filter(logits.copy(), params.top_k, params.top_p)
                probs = softmax(filt / max(params.temperature, 1e-5))
                tok = int(self.shared_rng.choice(len(probs), p=probs))
                results.append((seq, tok, float(np.log(probs[tok] + 1e-12))))
            return results

        for seq, logits in batch:
            results.append(self._sample_one(seq, logits, self._rng_for(seq)))
        return results

    def sample_one(self, seq: Sequence, logits: np.ndarray) -> Tuple[int, float]:
        """单序列采样（推测解码/流式路径共用）。"""
        rng = self._rng_for(seq) if self.batch_invariant else self.shared_rng
        return self._sample_one(seq, logits, rng)

    def _sample_one(self, seq: Sequence, logits: np.ndarray,
                    rng: np.random.Generator) -> Tuple[int, float]:
        params = seq.sampling_params
        t = params.temperature
        if t <= 1e-5:                       # greedy：完全确定，与 RNG 无关
            tok = int(np.argmax(logits))
            probs = softmax(logits)
            return tok, float(np.log(probs[tok] + 1e-12))
        filt = top_k_top_p_filter(logits.copy(), params.top_k, params.top_p)
        probs = softmax(filt / t)
        # RNG 消费次数固定为 1：即使某请求被重调度，随机流也不会错位
        tok = int(rng.choice(len(probs), p=probs))
        return tok, float(np.log(probs[tok] + 1e-12))