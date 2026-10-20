"""批次不变性：输出只由（输入, 种子）决定，与批组合/顺序/邻居无关。

对应 DeepInfer 的 batch-invariant 采样：逐请求独立随机流 + 独立归约。
测试从三个角度钉死这一契约（回归测试可复现的地基）：
  1. 同 seed 两次独立运行 -> token 级一致；
  2. greedy 下批组成（1 条 vs 3 条）不影响任何一条的输出；
  3. 换 seed -> 采样路径改变（seed 真的在起作用）。
"""

from __future__ import annotations

from miniflow.sampling_params import SamplingParams

PROMPTS = ["the cat", "the dog", "the bird"]


def _tokens_of(outputs, seq_id):
    return next(o.token_ids for o in outputs if o.seq_id == seq_id)


def test_same_seed_reproducible(engine_factory):
    params = SamplingParams(max_tokens=16, temperature=1.0, seq_seed=42)

    eng1 = engine_factory(prefix_cache=False)
    eng1.add_request(PROMPTS[0], params)
    out1 = eng1.run()

    eng2 = engine_factory(prefix_cache=False)
    eng2.add_request(PROMPTS[0], params)
    out2 = eng2.run()

    assert _tokens_of(out1, 0) == _tokens_of(out2, 0)


def test_batch_composition_invariant(engine_factory):
    """A 单独跑 vs A 混在 3 条批里跑（greedy）：token 级一致。"""
    p = SamplingParams(max_tokens=16, temperature=0.0)

    solo = engine_factory(prefix_cache=False)
    solo.add_request(PROMPTS[0], p)
    out_solo = solo.run()

    mixed = engine_factory(prefix_cache=False)
    for prompt in PROMPTS:
        mixed.add_request(prompt, p)
    out_mixed = mixed.run()

    assert _tokens_of(out_solo, 0) == _tokens_of(out_mixed, 0)


def test_seed_changes_output(engine_factory):
    """不同 seq_seed 的采样流不同 -> 输出（大概率）不同。

    temperature 要足够高：400 步过拟合的小模型分布接近 one-hot，
    低温下任何种子都采样出 greedy 路径，seed 的作用不可观测。
    """
    hot = dict(max_tokens=16, temperature=2.5)

    eng42 = engine_factory(prefix_cache=False)
    eng42.add_request(PROMPTS[0], SamplingParams(seq_seed=42, **hot))
    out42 = eng42.run()

    eng7 = engine_factory(prefix_cache=False)
    eng7.add_request(PROMPTS[0], SamplingParams(seq_seed=7, **hot))
    out7 = eng7.run()

    assert _tokens_of(out42, 0) != _tokens_of(out7, 0)