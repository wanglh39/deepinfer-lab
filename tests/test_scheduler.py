"""调度器正确性：抢占 liveness / continuous batching / 死锁防御。

重点覆盖 miniflow 开发中真实踩过的三个调度坑：
  1. 容量检查误用整序列语义 => 自我抢占无限循环（liveness 回归）；
  2. 先接纳 prefill 后组 decode => 「接纳-抢占」震荡零进展；
  3. 检查通过但分配滞后 => 同批多序列双重花销同一份富余。
"""

from __future__ import annotations

import pytest

from miniflow.sampling_params import SamplingParams


def test_preemption_liveness(engine_factory):
    """块池远不够三条请求并行：抢占-重算后必须全部完成（不卡死）。"""
    eng = engine_factory(num_blocks=10, prefix_cache=False)
    for p in ("the cat", "the dog", "the bird"):
        eng.add_request(p, SamplingParams(max_tokens=24, temperature=0.0))
    outputs = eng.run()

    st = eng.stats()
    assert len(outputs) == 3
    assert st["finished"] == 3
    assert st["preemptions"] >= 1          # 确实走过抢占路径
    for out in outputs:
        assert len(out.token_ids) == 24    # 重算不丢 token


def test_continuous_batching_insert(engine_factory):
    """中途插队的新请求与在跑请求共享解码步，同样完成。"""
    from miniflow.sampling_params import SamplingParams as SP
    eng = engine_factory(prefix_cache=False)
    eng.add_request("the cat", SP(max_tokens=12, temperature=0.0))
    eng.add_request("the dog", SP(max_tokens=12, temperature=0.0))

    step = 0
    while eng.scheduler.has_unfinished():
        if step == 3:                       # 第 3 步后新请求到达
            eng.add_request("the bird", SP(max_tokens=12, temperature=0.0))
        eng.step()
        step += 1
        assert step < 200, "调度未收敛（疑似死循环）"

    assert len(eng.scheduler.finished) == 3
    late = [s for s in eng.scheduler.finished
            if len(s.prompt_token_ids) > 0 and s.seq_id == 2]
    assert late and len(late[0].output_token_ids) == 12


def test_deadlock_guard(engine_factory):
    """单序列需求超过整个块池：明确报错好过无限空转。"""
    eng = engine_factory(num_blocks=2, prefix_cache=False)
    eng.add_request("the cat", SamplingParams(max_tokens=24, temperature=0.0))
    with pytest.raises(RuntimeError, match="调度死锁"):
        eng.run()