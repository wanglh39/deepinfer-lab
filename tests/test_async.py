"""异步调度流水线：输出与同步引擎一致，schedule 与 forward 时间线重叠。

对应 DeepInfer 零开销批调度器：调度线程在执行线程跑 forward(N) 的
同时构建 schedule(N+1)。执行线程对「调度时活着、执行时已 finish」的
序列防御性跳过（名单滞后一步的同款约束），语义仍与同步路径等价。
"""

from __future__ import annotations

from miniflow.engine import AsyncLLMEngine
from miniflow.sampling_params import SamplingParams

PROMPTS = ["the cat", "the dog"]
P = SamplingParams(max_tokens=12, temperature=0.0)


def test_async_matches_sync(engine_factory):
    sync = engine_factory(prefix_cache=False)
    for p in PROMPTS:
        sync.add_request(p, P)
    out_sync = sync.run()

    async_eng = engine_factory(prefix_cache=False)
    for p in PROMPTS:
        async_eng.add_request(p, P)
    out_async = AsyncLLMEngine(async_eng).run()

    by_id_sync = {o.seq_id: o.token_ids for o in out_sync}
    by_id_async = {o.seq_id: o.token_ids for o in out_async}
    assert by_id_sync == by_id_async


def test_pipeline_overlap(engine_factory):
    """至少一个 schedule 完全落在某个 forward 区间内（调度不阻塞执行）。"""
    eng = engine_factory(prefix_cache=False)
    for p in PROMPTS:
        eng.add_request(p, P)
    wrapper = AsyncLLMEngine(eng)
    wrapper.run()

    events = wrapper.events
    assert events, "时间线为空"
    fwd_iv = [(s, e) for tag, s, e, _ in events if tag == "forward"]
    assert fwd_iv, "没有 forward 事件"
    overlapped = any(
        any(fs <= s and e <= fe for fs, fe in fwd_iv)
        for tag, s, e, _ in events if tag == "schedule")
    assert overlapped, "schedule 与 forward 完全串行 —— 流水线未生效"