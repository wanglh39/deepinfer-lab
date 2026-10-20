"""引擎 —— 对应 deepinfer/engine 的 LLMEngine：把所有子系统组装成推理服务。

一条请求的完整旅程（同步路径）：

    add_request            分词、入 WAITING 队列
      -> scheduler.schedule   前缀缓存匹配 / 组批（continuous batching）
      -> model.forward        增量前向（读 KV、算新 KV）
      -> structured_output    语法自动机 mask（若请求带约束）
      -> sampler.sample       逐请求独立采样（batch-invariant）
      -> finish 判定          EOS / max_tokens / stop 串 / 语法完成
      -> scheduler.finish_seq 前缀缓存登记、KV 引用释放

**异步调度（AsyncLLMEngine）**：复刻 DeepInfer「零开销批调度器」的流水线：
CPU 调度线程在执行线程跑 forward(N) 的同时构建 schedule(N+1)。
由于 sample(N) 的结果尚未返回，调度基于 **滞后一步** 的序列状态 —— 这正是
DeepInfer QA 里「schedule(N+1) 时 confidence 未返回，CPU 只固定总预算，
每请求长度在 GPU 侧重算」的同款约束；执行线程对 finish 序列做防御性跳过，
语义等价于 GPU 侧的自治微调。时间线见 examples/demo_scheduler.py。
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from .cache.paged_kv import PagedKVCache
from .cache.prefix_cache import PrefixCache
from .config import MiniFlowConfig
from .model.transformer import TransformerModel
from .sampler import Sampler
from .scheduler import Scheduler, StepOutput
from .sequence import RequestOutput, Sequence, SequenceStatus
from .speculative import SpeculativeDecoder
from .structured_output import GrammarAutomaton
from .tokenizer import EOS_ID, Tokenizer

_N_SPECIAL = 5   # 词表前 5 位是特殊 token（bos/eos/pad + 2 个控制位），SO 约束下一律屏蔽


class LLMEngine:
    def __init__(self, config: MiniFlowConfig, model: TransformerModel,
                 tokenizer: Tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.kv = PagedKVCache(config.cache, config.model)
        self.prefix_cache = PrefixCache(config.cache)
        self.scheduler = Scheduler(config, self.kv, self.prefix_cache)
        self.sampler = Sampler(config.engine.batch_invariant)
        self.spec: Optional[SpeculativeDecoder] = None
        if config.engine.speculative_decoding:
            self.spec = SpeculativeDecoder(
                model, self.kv, self.sampler,
                max_draft=config.engine.max_draft_tokens)
        self.so_registry: Dict[int, GrammarAutomaton] = {}
        self._requests: Dict[int, dict] = {}

    # ------------------------------------------------------------------
    def add_request(self, prompt: str, sampling_params=None,
                    so_pattern: Optional[str] = None) -> Sequence:
        """so_pattern: 结构化输出约束（正则子集），对应 so_args 机制。"""
        from .sampling_params import SamplingParams
        params = sampling_params or SamplingParams()
        ids = self.tokenizer.encode(prompt, add_bos=True)
        seq = self.scheduler.add_request(ids, params)
        self._requests[seq.seq_id] = {"prompt": prompt, "params": params}
        if so_pattern is not None:
            self.so_registry[seq.seq_id] = GrammarAutomaton.compile(so_pattern)
        return seq

    # ------------------------------------------------------------------
    def step(self) -> List[RequestOutput]:
        """推进一步（组批 -> 前向 -> 采样 -> 完成处理）。"""
        so = self.scheduler.schedule()
        if so.is_empty:
            return []
        outputs: List[RequestOutput] = []
        for it in so.items:
            outputs += self.step_for_seq(it.seq, it.is_prefill)
        return outputs

    def step_for_seq(self, seq: Sequence, is_prefill: bool) -> List[RequestOutput]:
        """单序列执行（同步 step 与异步执行线程共用）。"""
        if seq.is_finished():
            return []
        if self.spec is not None and not is_prefill:
            # 推测解码路径：整引擎级开关（生产是逐请求/逐负载切换）
            self.kv.ensure_capacity(seq)
            self.spec.step(seq, EOS_ID)
            if len(seq.output_token_ids) >= seq.sampling_params.max_tokens:
                seq.finish(SequenceStatus.FINISHED_LENGTH_CAPPED)
            if seq.is_finished():
                return self._finalize(seq)
            return []
        return self._ordinary_step(seq, is_prefill)

    def _ordinary_step(self, seq: Sequence, is_prefill: bool) -> List[RequestOutput]:
        self.kv.ensure_capacity(seq)
        logits = self.model.forward(seq.token_ids, seq.seq_id, self.kv)
        last = logits[-1]

        automaton = self.so_registry.get(seq.seq_id)
        if automaton is not None:
            if automaton.in_accept() and not automaton.has_feasible_token(
                    self.tokenizer.vocab[_N_SPECIAL:]):
                # 语法完整匹配且无合法后继：正常收笔（SO 的 completion 路径）
                seq.finish(SequenceStatus.FINISHED_STOPPED)
                return self._finalize(seq)
            try:
                last = self._so_mask(automaton, last)
            except ValueError:
                seq.finish(SequenceStatus.FINISHED_ABORTED)   # 语法死局
                return self._finalize(seq)

        tok, _lp = self.sampler.sample_one(seq, last)
        seq.append_token(tok)
        if automaton is not None and not automaton.advance(self.tokenizer.vocab[tok]):
            seq.finish(SequenceStatus.FINISHED_ABORTED)   # 语法死局

        if is_prefill:
            seq.num_computed = seq.num_tokens
        if self._check_finish(seq, tok):
            return self._finalize(seq)
        return []

    def _so_mask(self, automaton: GrammarAutomaton, logits: np.ndarray) -> np.ndarray:
        masked = logits.astype(np.float32, copy=True)
        masked[:_N_SPECIAL] = -np.inf      # 约束生成时屏蔽全部特殊 token
        return automaton.mask(masked, self.tokenizer.vocab)

    def _check_finish(self, seq: Sequence, tok: int) -> bool:
        if seq.is_finished():
            return True
        params = seq.sampling_params
        if tok == EOS_ID and not params.ignore_eos:
            seq.finish(SequenceStatus.FINISHED_STOPPED)
        elif len(seq.output_token_ids) >= params.max_tokens:
            seq.finish(SequenceStatus.FINISHED_LENGTH_CAPPED)
        else:
            text = self.tokenizer.decode(seq.output_token_ids)
            for s in params.stop:
                if s and s in text:
                    seq.finish(SequenceStatus.FINISHED_STOPPED)
                    break
        return seq.is_finished()

    def _finalize(self, seq: Sequence) -> List[RequestOutput]:
        req = self._requests.get(seq.seq_id, {})
        out = RequestOutput(
            seq_id=seq.seq_id, prompt=req.get("prompt", ""),
            output_text=self.tokenizer.decode(seq.output_token_ids),
            token_ids=list(seq.output_token_ids),
            finish_reason=seq.status.value,
            metrics={"cached_prefill": seq.num_cached_prefill},
        )
        self.scheduler.finish_seq(seq)
        return [out]

    # ------------------------------------------------------------------
    def run(self) -> List[RequestOutput]:
        outputs: List[RequestOutput] = []
        while self.scheduler.has_unfinished():
            outputs += self.step()
        return outputs

    def stats(self) -> dict:
        d = self.scheduler.stats()
        if self.spec is not None:
            d["speculative"] = self.spec.stats.as_dict()
        return d


class AsyncLLMEngine:
    """异步调度引擎：调度线程与执行线程流水线重叠（零开销批调度器复刻）。

    events: [(tag, start, end, step)]，供 demo 绘制重叠时间线。
    """

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.events: List[tuple] = []
        self._q_sched: "queue.Queue[Optional[StepOutput]]" = queue.Queue(maxsize=2)
        self._q_out: "queue.Queue[List[RequestOutput]]" = queue.Queue()
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    def run(self) -> List[RequestOutput]:
        t0 = time.perf_counter()
        self._t0 = t0   # 两个线程共用同一时间基准（否则时间线坐标系错位）
        sched_thread = threading.Thread(target=self._sched_loop, daemon=True)
        sched_thread.start()
        outputs: List[RequestOutput] = []
        step_idx = 0
        while True:
            so = self._q_sched.get()
            if so is None:
                break
            t_f0 = time.perf_counter() - t0
            step_outputs = self._execute(so, step_idx)
            t_f1 = time.perf_counter() - t0
            self.events.append(("forward", t_f0, t_f1, step_idx))
            outputs += step_outputs
            step_idx += 1
        sched_thread.join()
        return outputs

    def _sched_loop(self) -> None:
        t0 = self._t0
        step_idx = 0
        while not self._stop.is_set():
            if not self.engine.scheduler.has_unfinished():
                self._q_sched.put(None)
                return
            t_s0 = time.perf_counter() - t0
            so = self.engine.scheduler.schedule()
            t_s1 = time.perf_counter() - t0
            self.events.append(("schedule", t_s0, t_s1, step_idx))
            if so.is_empty:
                continue
            self._q_sched.put(so)
            step_idx += 1

    def _execute(self, so: StepOutput, step_idx: int) -> List[RequestOutput]:
        """执行线程：防御性跳过「调度时还活着、执行时已 finish」的序列。

        调度（N+1）与采样（N）并发 => 名单滞后一步 —— DeepInfer 的
        「CPU 固定总预算 + GPU 侧修正」在 miniflow 里的对应物。
        """
        outputs: List[RequestOutput] = []
        for it in so.items:
            seq = it.seq
            if seq.is_finished():
                continue
            outputs += self.engine.step_for_seq(seq, it.is_prefill)
        return outputs