"""推测解码 —— 复刻 DSpark 的「draft + 置信度调度验证」思想。

**为什么推测解码有效？** decode 每步只算 1 个 token，GPU 喂不饱；
若一次 forward 能「审」完 k 个候选 token，接受 m 个，等效单步产出 m 个。
关键洞察：**验证 k 个 token 的算力 ≈ 生成 1 个 token**（一次矩阵乘摊开），
而 draft 很便宜 —— 拿便宜的猜测换昂贵的串行，吞吐就上去了。

**DSpark 的两板斧**（DeepInfer 生产实现，miniflow 教学复刻）：
1. **变长验证**：draft token 的接受率随请求/位置差异巨大，固定验证 5 个
   会把算力浪费在低概率 token 上。生产数据：平均验证长度 5.00 -> 3.2~4.4，
   接受长度几乎不变（3.29 vs 3.20），生产点吞吐 +4~7%；
2. **置信度调度**：Draft 侧用 Confidence Head 预测每个 draft token 的
   接受率，调度时把验证算力分给边际收益最高的 token 前缀。

**miniflow 的教学映射**（保证逻辑严谨可测）：
* draft  = 对主模型分布做 **温度采样**（生产中是独立小模型/MTP 头）；
* 置信度 = 采样出的 token 在分布中的概率（对应 Confidence Head）；
* 验证   = 主模型同输入的前向是确定性的：logits 的 argmax 即「标准答案」，
  draft token 命中 argmax 则接受 —— 因此 **输出与纯 greedy 完全一致**
  （可用测试验证正确性），但步数更少（体现加速）；
* 变长验证 = 只验证「置信度 >= 阈值的最长前缀」，其余 draft 直接丢弃
  （KV 回滚）—— 复刻「把验证算力花在高边际收益 token 上」。

KV 回滚依赖 PagedKVCache 的「进度 + 覆盖写」语义：被拒的 draft 槽位
在下一次 forward 时被自然覆盖，无需物理擦除。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .cache.paged_kv import PagedKVCache
from .model.transformer import TransformerModel
from .sequence import Sequence, SequenceStatus
from .tensor import softmax


@dataclass
class SpecStats:
    """推测解码观测 —— 用于 demo 展示 DSpark 论文里的同款指标。"""
    steps: int = 0
    draft_tokens: int = 0
    verified_tokens: int = 0
    accepted_tokens: int = 0
    bonus_tokens: int = 0

    @property
    def accept_rate(self) -> float:
        return self.accepted_tokens / max(self.verified_tokens, 1)

    @property
    def avg_draft_len(self) -> float:
        return self.verified_tokens / max(self.steps, 1)

    @property
    def tokens_per_step(self) -> float:
        return (self.accepted_tokens + self.bonus_tokens) / max(self.steps, 1)

    def as_dict(self) -> dict:
        return {"steps": self.steps,
                "avg_verified_len": round(self.avg_draft_len, 2),
                "avg_accepted_len": round(self.accepted_tokens / max(self.steps, 1), 2),
                "accept_rate": round(self.accept_rate, 3),
                "tokens_per_step": round(self.tokens_per_step, 2)}


class SpeculativeDecoder:
    def __init__(self, model: TransformerModel, kv: PagedKVCache,
                 sampler, max_draft: int = 5,
                 confidence_threshold: float = 0.35,
                 draft_temperature: float = 1.2):
        self.model = model
        self.kv = kv
        self.sampler = sampler            # 复用引擎的 sampler（保持确定性语义）
        self.max_draft = max_draft
        self.threshold = confidence_threshold
        self.draft_temperature = draft_temperature
        self.stats = SpecStats()

    # ------------------------------------------------------------------
    def _draft_logits(self, seq: Sequence) -> List[np.ndarray]:
        """draft 阶段：逐步温度采样，返回每步的 logits（供验证/修正复用）。

        每步 forward 增量 1 token 并写 KV —— draft 写入的 KV 与验证后的
        重算结果数值一致（同模型同输入确定性），因此无需显式擦除。
        """
        logits_list: List[np.ndarray] = []
        for _ in range(self.max_draft):
            if seq.num_tokens >= self.kv.num_slots - 1:
                break
            self.kv.ensure_capacity(seq)
            logits = self.model.forward(seq.token_ids, seq.seq_id, self.kv)
            logits_list.append(logits[-1])
            # 温度采样起草（生产中这是 draft 小模型的分布）
            filt = logits[-1] / self.draft_temperature
            probs = softmax(filt)
            rng = self.sampler._rng_for(seq) if self.sampler.batch_invariant \
                else np.random.default_rng()
            tok = int(rng.choice(len(probs), p=probs))
            seq.append_token(tok)
        return logits_list

    def _verify_len(self, confidences: List[float]) -> int:
        """置信度调度：验证「置信度达标的最长前缀」（变长验证的核心）。"""
        n = 0
        for c in confidences:
            if c >= self.threshold:
                n += 1
            else:
                break
        return max(n, 1) if confidences else 0

    # ------------------------------------------------------------------
    def step(self, seq: Sequence, eos_id: int) -> int:
        """对一条 RUNNING 序列执行「draft -> 置信度调度 -> 验证 -> 提交」。

        返回本步净产出的 token 数（>=1）。
        """
        self.stats.steps += 1
        base_len = seq.num_tokens

        logits_list = self._draft_logits(seq)
        draft_tokens = seq.token_ids[base_len:]
        confidences = [float(softmax(l)[t]) for l, t in
                       zip(logits_list, draft_tokens)]

        k_verify = self._verify_len(confidences)
        self.stats.draft_tokens += len(draft_tokens)
        self.stats.verified_tokens += k_verify

        # 验证：draft token 命中 argmax（标准答案）则接受
        accepted = 0
        for i in range(k_verify):
            if int(np.argmax(logits_list[i])) == draft_tokens[i]:
                accepted += 1
            else:
                break
        self.stats.accepted_tokens += accepted

        # 提交：截断到接受长度；被拒位置用该步 logits 的 argmax 修正（bonus）。
        # 标准 speculative sampling 在被拒位置从「残差分布」采样以保持目标分布；
        # 教学版采用 argmax 验证 => 修正同样取 argmax，输出与纯 greedy 严格一致
        # （tests/ 里「推测解码 == 贪心」断言正是建立在这一契约上）。
        keep_tokens = list(draft_tokens[:accepted])
        if accepted < k_verify:                    # 被拒 -> argmax 修正（bonus）
            keep_tokens.append(int(np.argmax(logits_list[accepted])))
            self.stats.bonus_tokens += 1
        # spec 一步净提交多个 token，不得越过 max_tokens 上限
        hist_out_len = base_len - len(seq.prompt_token_ids)
        room = seq.sampling_params.max_tokens - hist_out_len
        if len(keep_tokens) > room:
            keep_tokens = keep_tokens[:room]

        # 序列状态与 KV 进度回滚（覆盖写语义：作废槽位下轮自然覆盖）。
        # 回滚量必须留最后 1 个 token 不算：draft 循环的不变量是
        # 「进入每次 forward 时恰好 1 个未算 token」—— 算最后一个 token
        # 才能拿到下一位置的分布。把 computed 推到 num_tokens 会让下一
        # step 的第一次 forward 无 token 可算（增量契约直接断言失败）。
        hist_out_len = base_len - len(seq.prompt_token_ids)
        seq.output_token_ids = seq.output_token_ids[:hist_out_len] + keep_tokens
        kv_keep = base_len + len(keep_tokens) - 1
        for layer in range(len(self.kv.computed[seq.seq_id])):
            self.kv.computed[seq.seq_id][layer] = kv_keep

        if eos_id in keep_tokens:
            seq.finish(SequenceStatus.FINISHED_STOPPED)
        return len(keep_tokens)