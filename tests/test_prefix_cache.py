"""前缀缓存：命中省算力，且复用不改变输出语义。

对应 DeepInfer 的共享前缀 KV（生产上持久化在 3FS，TTL 3~7 天）：
  1. 后来者命中先前者的前缀块 -> 跳过命中部分的重算（省 prefill token）；
  2. 块级复用 + set_computed 不改变 forward 的数值路径 -> 输出 token 级一致。
"""

from __future__ import annotations

from miniflow.sampling_params import SamplingParams

SHARED_PREFIX = "the cat sat on the"
PROMPT_A = SHARED_PREFIX + " mat"
PROMPT_B = SHARED_PREFIX + " mat. the dog"
P = SamplingParams(max_tokens=8, temperature=0.0)


def test_prefix_hit_saves_prefill(engine_factory):
    """A 完成并登记缓存后 B 才到达：B 的前缀块直接复用。

    必须串行提交 —— A、B 同批时 B 被调度那一刻 A 尚未 finish 登记，
    命中不会发生（这也是生产上「前缀缓存放行时机」的语义）。
    """
    eng = engine_factory(prefix_cache=True)
    eng.add_request(PROMPT_A, P)
    eng.run()                              # A 完成 -> 前缀块登记进缓存
    eng.add_request(PROMPT_B, P)
    eng.run()

    st = eng.stats()
    assert st["prefix_hits"] >= 1
    assert st["saved_prefill_tokens"] >= 4  # 至少复用一个块（block_size=4）

    no_cache = engine_factory(prefix_cache=False)
    no_cache.add_request(PROMPT_A, P)
    no_cache.run()
    no_cache.add_request(PROMPT_B, P)
    no_cache.run()
    assert no_cache.stats()["prefix_hits"] == 0


def test_prefix_reuse_output_identical(engine_factory):
    """B 在「有前缀缓存」与「无前缀缓存」引擎下输出一致。"""
    with_cache = engine_factory(prefix_cache=True)
    with_cache.add_request(PROMPT_A, P)
    with_cache.run()                        # A 完成并登记前缀
    with_cache.add_request(PROMPT_B, P)
    out_b = with_cache.run()

    no_cache = engine_factory(prefix_cache=False)
    no_cache.add_request(PROMPT_B, P)
    out_b_ref = no_cache.run()

    assert out_b[0].token_ids == out_b_ref[0].token_ids