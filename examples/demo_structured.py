"""结构化输出演示 —— 复刻 DeepInfer structure_output 的「编译-运行」两段式。

用法（需要训练好的 checkpoint）：
    python tools/train.py --out checkpoints/tiny --steps 400
    python examples/demo_structured.py

机制：compile(pattern) 把正则子集编译成 Thompson NFA（对应生产上
异步跑在专用 SO 机器上的「编译期」）；每个 decode step 用自动机算出
合法 token 集合，生成 -inf mask 施加到采样 logits 上（「运行期」）。

注意：BPE 词表里空格是 '▁' 字符（GPT-2 同款），所以模式里写 '▁in'
而不是 ' in'。演示的模式强制模型在 (ran|flew) 与 (in|over) 中选择 ——
即使自由 greedy 想输出别的（约束的力量）。语法完整匹配且无合法后继时
引擎正常收笔（finished_stopped）。
"""

from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniflow.entrypoints.cli import load_engine  # noqa: E402
from miniflow.sampling_params import SamplingParams  # noqa: E402
from miniflow.structured_output import GrammarAutomaton  # noqa: E402

CKPT = "checkpoints/tiny.npz"
TOK = "checkpoints/tiny.tok.json"

PATTERN = "(▁ran|▁flew)(▁in|▁over)▁the"
TEXT_EQUIV = r"( ran| flew)( in| over) the"


def main() -> None:
    if not os.path.exists(CKPT):
        print("未找到 checkpoint，请先运行: python tools/train.py --out checkpoints/tiny")
        sys.exit(1)

    print(f"prompt  : 'the cat'")
    print(f"pattern : {PATTERN!r}   (▁ = BPE 词表里的空格)\n")

    # ---- 对照：自由 greedy（无约束）----
    free = load_engine(CKPT, TOK)
    free.add_request("the cat", SamplingParams(max_tokens=8, temperature=0.0))
    (out_free,) = free.run()
    print(f"[自由 greedy]   output = {out_free.output_text!r} "
          f"({out_free.finish_reason})")

    # ---- 约束生成：SO mask 每步强制合法 token ----
    eng = load_engine(CKPT, TOK)
    eng.add_request("the cat", SamplingParams(max_tokens=12, temperature=0.0),
                    so_pattern=PATTERN)
    (out,) = eng.run()
    print(f"[SO 约束]       output = {out.output_text!r} "
          f"({out.finish_reason})")

    ok = re.fullmatch(TEXT_EQUIV, out.output_text) is not None
    print(f"\n输出严格匹配模式 {TEXT_EQUIV!r}: {ok}")

    # ---- 附加：mask 的直接观测（编译期产物 -> 运行期 mask）----
    auto = GrammarAutomaton.compile(PATTERN)
    tok = eng.tokenizer
    logits = [0.0] * tok.vocab_size
    logits[5] = 10.0                      # 假装模型最想要 id=5 的 token
    masked = auto.mask(__import__("numpy").array(logits), tok.vocab)
    legal = [tok.vocab[i] for i, v in enumerate(masked) if v > -1e8]
    print(f"初始状态下合法 token: {legal[:8]}{'...' if len(legal) > 8 else ''}")
    print("-> 模型的 argmax 被语法改写：mask 在采样前生效，分布形状不被扭曲")


if __name__ == "__main__":
    main()