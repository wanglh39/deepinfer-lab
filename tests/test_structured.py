"""结构化输出：语法 mask 强制输出匹配模式，且不破坏收笔语义。

对应 DeepInfer structure_output 的编译-运行两段式：
  1. 约束生成的文本严格 fullmatch 模式（约束的力量：改写 argmax）；
  2. 语法完整匹配且无合法后继 -> 正常收笔 finished_stopped；
  3. 词表与语法无交集 -> 死局 abort，而不是喂全 -inf 的 mask 给采样器。
注意：BPE 词表里空格是独立 token '▁'，模式必须按字符级自动机写。
"""

from __future__ import annotations

import re

from miniflow.sampling_params import SamplingParams

PROMPT = "the cat"
PATTERN = "(▁ran|▁flew)(▁in|▁over)▁the"
TEXT_EQUIV = r"( ran| flew)( in| over) the"


def test_so_output_matches_pattern(engine_factory):
    eng = engine_factory(prefix_cache=False)
    eng.add_request(PROMPT, SamplingParams(max_tokens=12, temperature=0.0),
                    so_pattern=PATTERN)
    (out,) = eng.run()
    assert re.fullmatch(TEXT_EQUIV, out.output_text) is not None


def test_so_finishes_cleanly(engine_factory):
    """语法走完自动收笔（completion 路径），不是被 max_tokens 截断。"""
    eng = engine_factory(prefix_cache=False)
    eng.add_request(PROMPT, SamplingParams(max_tokens=12, temperature=0.0),
                    so_pattern=PATTERN)
    (out,) = eng.run()
    assert out.finish_reason == "finished_stopped"


def test_constrained_differs_from_free(engine_factory):
    """同一 prompt：约束输出 != 自由 greedy 输出（约束真实生效）。"""
    free = engine_factory(prefix_cache=False)
    free.add_request(PROMPT, SamplingParams(max_tokens=8, temperature=0.0))
    (out_free,) = free.run()

    so = engine_factory(prefix_cache=False)
    so.add_request(PROMPT, SamplingParams(max_tokens=8, temperature=0.0),
                   so_pattern=PATTERN)
    (out_so,) = so.run()

    assert out_so.token_ids != out_free.token_ids


def test_dead_pattern_aborts(engine_factory):
    """'z' 不在词表字符集里：初始即无合法 token -> 死局 abort。"""
    eng = engine_factory(prefix_cache=False)
    eng.add_request(PROMPT, SamplingParams(max_tokens=8, temperature=0.0),
                    so_pattern="z")
    (out,) = eng.run()
    assert out.finish_reason == "finished_aborted"
    assert out.output_text == ""