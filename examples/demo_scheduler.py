"""调度器演示 —— Continuous Batching、抢占、异步流水线时间线。

用法（需要训练好的 checkpoint）：
    python tools/train.py --out checkpoints/tiny --steps 400
    python examples/demo_scheduler.py

三个演示：
  1. **Continuous batching**：请求中途插队 —— 新 prompt 的 prefill 与
     在跑请求的 decode 同 step 组批（对应 vLLM/DeepInfer 迭代级调度）；
  2. **KV 压力下的抢占**：块池吃紧时回收最晚到达的序列，重算后继续
     （对应 vLLM recompute 型抢占 / DeepInfer 调度让路）；
  3. **异步调度时间线**：调度线程与执行线程重叠 —— schedule(N+1) 在
     forward(N) 期间完成，GPU 侧无空转（DeepInfer 零开销批调度器的本质）。
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniflow.config import CacheConfig, EngineConfig, MiniFlowConfig  # noqa: E402
from miniflow.entrypoints.cli import load_engine, load_model_config  # noqa: E402
from miniflow.sampling_params import SamplingParams  # noqa: E402
from miniflow.tokenizer import Tokenizer  # noqa: E402
from miniflow.model.transformer import TransformerModel  # noqa: E402
from miniflow.engine import LLMEngine, AsyncLLMEngine  # noqa: E402

CKPT = "checkpoints/tiny.npz"
TOK = "checkpoints/tiny.tok.json"


def demo_continuous_batching() -> None:
    print("== 1) Continuous Batching：请求中途插队 ==")
    engine = load_engine(CKPT, TOK)
    engine.add_request("the cat", SamplingParams(max_tokens=12, temperature=0.0))
    engine.add_request("the dog", SamplingParams(max_tokens=12, temperature=0.0))
    print("  step | running | waiting | 动作")
    step = 0
    while engine.scheduler.has_unfinished():
        if step == 3:                        # 第 3 步后新请求到达
            engine.add_request("the bird",
                               SamplingParams(max_tokens=12, temperature=0.0))
            print("  ----  ^^^ 新请求 'the bird' 到达，插队 ----")
        n_run = len(engine.scheduler.running)
        n_wait = len(engine.scheduler.waiting)
        outputs = engine.step()
        step += 1
        note = []
        if outputs:
            note.append(f"完成 {len(outputs)} 条")
        print(f"  {step:4d} | {n_run:7d} | {n_wait:7d} | "
              + ("; ".join(note) if note else "组批推进"))
    print("  -> 解码步被「正好有活干」的请求共享，完成即让位。\n")


def demo_preemption() -> None:
    print("== 2) KV 块池压力下的抢占 ==")
    tok = Tokenizer.load(TOK)
    cfg = MiniFlowConfig(
        model=load_model_config(CKPT, tok),
        cache=CacheConfig(block_size=4, num_blocks=10, prefix_cache_enabled=False),
        engine=EngineConfig(),
    )
    model = TransformerModel.load(CKPT, cfg.model)
    engine = LLMEngine(cfg, model, tok)
    for p in ["the cat", "the dog", "the bird"]:
        engine.add_request(p, SamplingParams(max_tokens=24, temperature=0.0))
    outputs = engine.run()
    st = engine.stats()
    print(f"  3 条请求 x (prompt+24) token，只给 10 块 x 4 slot 的 KV 池")
    print(f"  -> preemptions = {st['preemptions']}（被抢占比块回收重算，请求仍全部完成）")
    print(f"  -> finished = {st['finished']}，outputs = {len(outputs)}\n")


def demo_async_timeline() -> None:
    print("== 3) 异步调度时间线（schedule 与 forward 重叠）==")
    engine = load_engine(CKPT, TOK)
    engine.add_request("the cat", SamplingParams(max_tokens=12, temperature=0.0))
    engine.add_request("the dog", SamplingParams(max_tokens=12, temperature=0.0))
    async_engine = AsyncLLMEngine(engine)
    t0 = time.perf_counter()
    async_engine.run()
    wall = time.perf_counter() - t0

    events = async_engine.events
    span = max(e[2] for e in events) - min(e[1] for e in events) + 1e-9
    width = 46
    fwd = [e for e in events if e[0] == "forward"]
    print(f"  共 {len(fwd)} 个 forward 步，墙钟 {wall * 1000:.1f} ms")
    print("  （P=forward 执行段，S=schedule 调度段；S 落在 P 区间内 => 重叠）\n")
    for tag, s, e, step in events[:14]:
        bar = "#" * max(1, int((e - s) / span * width))
        print(f"  step {step:2d} {tag:8s} [{(e - s) * 1000:6.2f} ms] {bar}")
    fwd_iv = sorted((s, e) for tag, s, e, _ in events if tag == "forward")
    n_sched = sum(1 for e in events if e[0] == "schedule")
    n_overlap = sum(
        1 for tag, s, e, _ in events
        if tag == "schedule" and any(fs <= s and e <= fe for fs, fe in fwd_iv))
    print(f"\n  流水线生效：{n_overlap}/{n_sched} 个 schedule 完全落在某个"
          f" forward 区间内 —— 调度不再阻塞执行")
    print("  （生产上 forward 在 GPU 这个独立设备上跑，重叠是物理必然；")
    print("   miniflow 的 forward 是 CPU/NumPy，靠 GIL 释放窗口并行 ——")
    print("   结构相同，并行「强度」取决于执行体是否独立于 CPU）")


def main() -> None:
    if not os.path.exists(CKPT):
        print("未找到 checkpoint，请先运行: python tools/train.py --out checkpoints/tiny")
        sys.exit(1)
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("cb", "all"):
        demo_continuous_batching()
    if part in ("preempt", "all"):
        demo_preemption()
    if part in ("async", "all"):
        demo_async_timeline()


if __name__ == "__main__":
    main()