"""推测解码：输出与纯 greedy 严格一致，步数更少。

argmax 验证契约：draft token 命中主模型 argmax 则接受，被拒位置用
argmax 修正 —— 因此 spec 输出 == greedy 输出（token 级），但一步
提交多个 token，步数严格减少。这是 DSpark 变长验证的正确性地基。
"""

from __future__ import annotations

from miniflow.sampling_params import SamplingParams

PROMPT = "the cat"
P = SamplingParams(max_tokens=24, temperature=0.0)


def test_spec_matches_greedy(engine_factory):
    base = engine_factory(prefix_cache=False, spec=False)
    base.add_request(PROMPT, P)
    (out_base,) = base.run()

    spec = engine_factory(prefix_cache=False, spec=True, max_draft=5)
    spec.add_request(PROMPT, P)
    (out_spec,) = spec.run()

    assert out_spec.token_ids == out_base.token_ids


def test_spec_fewer_steps(engine_factory):
    spec = engine_factory(prefix_cache=False, spec=True, max_draft=5)
    spec.add_request(PROMPT, P)
    spec.run()

    base = engine_factory(prefix_cache=False, spec=False)
    base.add_request(PROMPT, P)
    base.run()

    assert spec.stats()["steps"] < base.stats()["steps"]


def test_spec_stats_sane(engine_factory):
    spec = engine_factory(prefix_cache=False, spec=True, max_draft=5)
    spec.add_request(PROMPT, P)
    spec.run()

    d = spec.stats()["speculative"]
    assert 0.0 < d["accept_rate"] <= 1.0
    assert d["tokens_per_step"] >= 1.0
    assert d["avg_verified_len"] <= 5