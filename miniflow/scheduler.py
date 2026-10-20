"""调度器 —— 对应 deepinfer/engine 的批调度（Continuous Batching + 异步调度）。

**Continuous Batching（迭代级调度）**：不像静态 batch（凑齐一批一起进、
一起出），而是 **每个 decode step 重新组批**：完成的序列立刻让位，
新到的 prompt 立即插队 —— GPU 时间被切给「正好有活干」的请求。
这是 vLLM/DeepInfer 高吞吐的地基。

**Prefill 预算**：新请求的 prompt 太长时切块计算（chunked prefill）。
DeepInfer 用拟合的二次成本模型 cost(q, c) 在时间预算内反解 chunk 大小 q；
miniflow 用固定 token 预算 max_tokens_per_step 讲同一思想：
prefill（算力受限）与 decode（访存受限）混跑会互相拖累，预算控制混跑比例。

**抢占（preemption）**：KV 块耗尽时，回收最不「划算」的序列
（教学版：最晚到达的 running 序列）已占的块，状态回退等待重算 ——
对应 vLLM 的 recompute 型抢占与 DeepInfer 的调度让路。

**异步调度（零开销批调度器）**：schedule/post-process 是纯 CPU 逻辑，
朴素流水里 GPU 会等 CPU。DeepInfer 让 CPU 在 GPU 跑
forward(N)->sample(N) 的同时执行 schedule(N+1)/launch_forward(N+1)
/post-process(N-1) —— GPU 无空转。miniflow 在 engine.py 用双线程
流水线复刻（演示时间线见 examples/demo_scheduler.py）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from .cache.paged_kv import PagedKVCache
from .cache.prefix_cache import PrefixCache
from .config import MiniFlowConfig
from .sequence import Sequence, SequenceStatus


@dataclass
class ScheduledSeq:
    """一条序列本步的执行计划：区间 [num_computed, num_tokens)。"""
    seq: Sequence
    is_prefill: bool

    @property
    def num_new_tokens(self) -> int:
        return self.seq.num_tokens - self.seq.num_computed


@dataclass
class StepOutput:
    """一次调度的产出（engine 据此驱动模型与采样）。"""
    items: List[ScheduledSeq] = field(default_factory=list)
    preempted: List[int] = field(default_factory=list)
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.items


class Scheduler:
    def __init__(self, config: MiniFlowConfig,
                 kv: PagedKVCache, prefix_cache: PrefixCache):
        self.config = config
        self.kv = kv
        self.prefix_cache = prefix_cache
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.finished: List[Sequence] = []
        self._next_seq_id = 0
        # 观测
        self.num_steps = 0
        self.num_preemptions = 0
        self.num_prefix_hits = 0
        self.saved_prefill_tokens = 0

    # ------------------------------------------------------------------
    def add_request(self, prompt_token_ids: List[int], sampling_params) -> Sequence:
        seq = Sequence(self._next_seq_id, prompt_token_ids, sampling_params)
        self._next_seq_id += 1
        self.waiting.append(seq)
        return seq

    def has_unfinished(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    # ------------------------------------------------------------------
    def schedule(self) -> StepOutput:
        """组一个 step：decode 优先，剩余容量再接纳新 prefill。

        **顺序是正确性的关键**（vLLM 同款）：先保证在跑请求吃饱，再让
        新请求进来。若反过来，会出现「接纳-抢占」震荡 —— 新 prefill
        恰好用光剩余块，在跑序列追加 1 块失败，调度器抢掉刚接纳的
        序列，下一步又接纳它……每步零进展（miniflow 踩过的死循环）。
        """
        self.num_steps += 1
        out = StepOutput()

        # 1) RUNNING decode：逐条「检查-分配」原子化推进（预留即时
        #    实体化为物理块，块耗尽不可能绕过调度器到达 engine）。
        for seq in list(self.running):
            if not self.kv.ensure_append_capacity(seq, 1):
                out.preempted.extend(self._preempt())
                break   # 抢占后本步停止扩批，下一步重新评估（保守，vLLM 同款）
            out.items.append(ScheduledSeq(seq, is_prefill=False))
            out.num_decode_tokens += 1

        # 2) WAITING：前缀匹配 + 尽力接纳（受剩余容量与 prefill 预算约束）
        budget = self.config.engine.max_tokens_per_step
        while self.waiting:
            seq = self.waiting[0]
            # 前缀缓存命中：块级复用，跳过命中部分的重算
            hit_tokens, hit_blocks = (0, [])
            if self.prefix_cache.enabled and seq.num_computed == 0:
                hit_tokens, hit_blocks = self.prefix_cache.match(
                    seq.prompt_token_ids, self.kv)
                # 至少保留 1 个 token 给 forward 产出采样 logits：
                # 命中长度必须整块收缩到 num_tokens - 1 以内（生产实现同款约束，
                # vLLM 里 prefill 至少重算最后一个 token）
                max_blocks = (seq.num_tokens - 1) // self.kv.block_size
                if len(hit_blocks) > max_blocks:
                    hit_blocks = hit_blocks[:max_blocks]
                    hit_tokens = max_blocks * self.kv.block_size

            need = seq.num_tokens - seq.num_computed
            # 容量检查：命中块是共享的（只加引用、不耗空闲块），
            # 只为未被覆盖的部分要新块；先检查后 acquire —— 若检查
            # 失败就 break，已挂上的共享块引用会泄漏且下轮重复挂载
            need_new = (self.kv.blocks_needed(seq.num_tokens + 1)
                        - len(hit_blocks))
            if need_new > self.kv.num_free_blocks:
                break
            if need > budget and (self.running or out.items):
                # 预算保护的是同批 decode 的延迟；无活可干时强制放行，
                # 否则被抢占序列的重算长度一旦超预算就永远出不了队。
                # 生产框架用 chunked prefill 把长 prefill 切多步消化，
                # 这里整段放行是教学版的 liveness 兜底。
                break
            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            if hit_tokens > 0:
                self.kv.acquire_blocks(hit_blocks, seq.seq_id)
                self.kv.set_computed(seq.seq_id, hit_tokens)
                seq.num_computed = hit_tokens
                self.num_prefix_hits += 1
                self.saved_prefill_tokens += hit_tokens
            # 实体化分配到 num_tokens + 1：为即将生成的 token 留位
            self.kv.ensure_capacity_tokens(seq.seq_id, seq.num_tokens + 1)
            out.items.append(ScheduledSeq(seq, is_prefill=True))
            out.num_prefill_tokens += need
            budget -= need

        self.running = [s for s in self.running if not s.is_finished()]

        # liveness 防御：没有任何可执行项却还有等待者 —— 说明单序列需求
        # 已超过整个块池（配置错误），明确报错好过无限空转。
        if not out.items and self.waiting and not self.running:
            head = self.waiting[0]
            need_blocks = self.kv.blocks_needed(head.num_tokens + 1)
            raise RuntimeError(
                f"调度死锁：序列 {head.seq_id} 需要 {need_blocks} 块，"
                f"KV 池仅 {self.kv.num_blocks} 块（请增大 CacheConfig.num_blocks）")
        return out

    def _preempt(self) -> List[int]:
        """块池吃紧：回收最晚进入 running 的序列（先让老请求走完）。"""
        victim = None
        for seq in reversed(self.running):
            if seq.status == SequenceStatus.RUNNING:
                victim = seq
                break
        if victim is None:
            return []
        self.kv.free_seq(victim.seq_id)
        victim.num_computed = 0
        victim.status = SequenceStatus.WAITING
        victim.output_token_ids = victim.output_token_ids  # 保留已生成，重算 prefill 后继续
        self.running.remove(victim)
        self.waiting.appendleft(victim)
        self.num_preemptions += 1
        return [victim.seq_id]

    # ------------------------------------------------------------------
    def finish_seq(self, seq: Sequence) -> None:
        """序列完成：登记前缀缓存（供后来者复用）并释放其 KV 引用。"""
        seq.status = seq.status  # 由 engine 设置
        self.prefix_cache.put(seq.seq_id, seq.token_ids, self.kv)
        self.kv.free_seq(seq.seq_id)
        self.running = [s for s in self.running if s.seq_id != seq.seq_id]
        self.finished.append(seq)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "steps": self.num_steps,
            "waiting": len(self.waiting), "running": len(self.running),
            "finished": len(self.finished),
            "preemptions": self.num_preemptions,
            "prefix_hits": self.num_prefix_hits,
            "saved_prefill_tokens": self.saved_prefill_tokens,
            "kv": self.kv.stats(),
            "prefix_cache": self.prefix_cache.stats(),
        }