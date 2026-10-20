"""结构化输出 —— 复刻 DeepInfer structure_output 的「编译-运行」两段式。

**DeepInfer 的做法**：
  * 编译期（专用 SO 机器，异步）：把 regex / JSON Schema / EBNF 编译成
    **语法自动机**；
  * 运行期：每个 decode step 推进自动机，得到「合法 token 集合」，
    生成 mask 施加到采样 logits 上（-inf 屏蔽非法 token），
    输出因此被强制约束在目标结构内。

关键工程点：编译与推理解耦（约束重的 schema 不阻塞首 token）、
mask 在采样前生效（不扭曲已合法路径的分布形状）。

**miniflow 教学版**：实现一个正则子集（字面量 / [a-z] 字符类 / . / * / | / ()）
的 Thompson NFA：
  * `compile(pattern)`   -> 自动机（对应 SO 编译机器的产物）；
  * `advance(token_str)` -> 消费一个 token，状态集转移（对应每步推进）；
  * `mask(logits, vocab)` -> 合法 token 的 -inf 掩码（对应采样 mask）。

token 级合法性 = 该 token 的整个字符串能从当前状态集走通且不落入死态。
这是「词表与语法联合约束」的最小可运行演示。
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Set, Tuple

import numpy as np

_EPS = "\0"        # Thompson 构造的空转移符


class _Node:
    """NFA 片段：start 状态集与（单）接受状态。"""

    __slots__ = ("start", "accept")

    def __init__(self, start: int, accept: int):
        self.start = start
        self.accept = accept


class GrammarAutomaton:
    def __init__(self, transitions: Dict[Tuple[int, str], Set[int]],
                 accept_states: Set[int], start_state: int = 0):
        self.transitions = transitions      # (state, char) -> {states}
        self.accept_states = accept_states
        self.start_state = start_state      # Thompson 构造的编号从 1 起，
        self.states: Set[int] = set()       # 不得硬编码 0（会得到空状态集）
        self._reset()

    def _reset(self) -> None:
        self.states = self._eps_closure({self.start_state})

    # ------------------------------------------------------------------
    # Thompson 构造：pattern -> NFA
    # ------------------------------------------------------------------
    @classmethod
    def compile(cls, pattern: str) -> "GrammarAutomaton":
        trans: Dict[Tuple[int, str], Set[int]] = {}
        accept: Set[int] = set()
        counter = [0]

        def new_state() -> int:
            counter[0] += 1
            return counter[0]

        def add(s: int, ch: str, t: int) -> None:
            trans.setdefault((s, ch), set()).add(t)

        def parse_alt(pos: int) -> Tuple[_Node, int]:
            node, pos = parse_seq(pos)
            while pos < len(pattern) and pattern[pos] == "|":
                rhs, pos = parse_seq(pos + 1)
                s, t = new_state(), new_state()
                add(s, _EPS, node.start)
                add(s, _EPS, rhs.start)
                trans.setdefault((node.accept, _EPS), set()).add(t)
                trans.setdefault((rhs.accept, _EPS), set()).add(t)
                node = _Node(s, t)
            return node, pos

        def parse_seq(pos: int) -> Tuple[_Node, int]:
            first, pos = parse_rep(pos)
            while pos < len(pattern) and pattern[pos] not in "|)":
                nxt, pos = parse_rep(pos)
                trans.setdefault((first.accept, _EPS), set()).add(nxt.start)
                first = _Node(first.start, nxt.accept)
            return first, pos

        def parse_rep(pos: int) -> Tuple[_Node, int]:
            node, pos = parse_atom(pos)
            if pos < len(pattern) and pattern[pos] == "*":
                s, t = new_state(), new_state()
                add(s, _EPS, node.start)
                trans.setdefault((node.accept, _EPS), set()).add(node.start)
                trans.setdefault((node.accept, _EPS), set()).add(t)
                add(s, _EPS, t)
                node = _Node(s, t)
                pos += 1
            return node, pos

        def parse_atom(pos: int) -> Tuple[_Node, int]:
            ch = pattern[pos]
            if ch == "(":
                node, pos = parse_alt(pos + 1)
                assert pattern[pos] == ")", "括号不匹配"
                return node, pos + 1
            if ch == "[":
                end = pattern.index("]", pos)
                inner = pattern[pos + 1:end]
                s, t = new_state(), new_state()
                for c in _expand_class(inner):
                    add(s, c, t)
                return _Node(s, t), end + 1
            if ch == ".":
                s, t = new_state(), new_state()
                for c in "abcdefghijklmnopqrstuvwxyz0123456789 ":
                    add(s, c, t)
                return _Node(s, t), pos + 1
            s, t = new_state(), new_state()
            add(s, ch, t)
            return _Node(s, t), pos + 1

        def _expand_class(inner: str) -> str:
            out = []
            i = 0
            while i < len(inner):
                if i + 2 < len(inner) and inner[i + 1] == "-":
                    out.extend(chr(c) for c in range(ord(inner[i]), ord(inner[i + 2]) + 1))
                    i += 3
                else:
                    out.append(inner[i])
                    i += 1
            return "".join(out)

        node, final = parse_alt(0)
        assert final == len(pattern), f"无法完整解析: 停在 {final}"
        accept.add(node.accept)
        return cls(trans, accept, start_state=node.start)

    # ------------------------------------------------------------------
    # 运行时
    # ------------------------------------------------------------------
    def _eps_closure(self, states: Set[int]) -> Set[int]:
        stack, seen = list(states), set(states)
        while stack:
            s = stack.pop()
            for t in self.transitions.get((s, _EPS), ()):  # 空转移闭包
                if t not in seen:
                    seen.add(t)
                    stack.append(t)
        return seen

    def _step_chars(self, states: Set[int], text: str) -> Set[int]:
        """吃掉 text 的全部字符后的状态集（中途为空 = 违约）。"""
        for ch in text:
            nxt: Set[int] = set()
            for s in states:
                nxt |= self.transitions.get((s, ch), set())
            states = self._eps_closure(nxt)
            if not states:
                break
        return states

    def token_feasible(self, token_str: str) -> bool:
        """token 是否可作为下一输出（走通且状态集非空）。"""
        return bool(self._step_chars(set(self.states), token_str))

    def advance(self, token_str: str) -> bool:
        """消费一个 token（每 decode step 调用一次）。"""
        nxt = self._step_chars(set(self.states), token_str)
        if not nxt:
            return False
        self.states = nxt
        return True

    def in_accept(self) -> bool:
        return bool(self.states & self.accept_states)

    def has_feasible_token(self, vocab: List[str]) -> bool:
        """当前状态下是否存在合法后继 token。

        配合 in_accept() 判断「语法已完整匹配且无合法后继」——
        对应生产 SO 的 completion：此时应正常收笔，而不是把
        全 -inf 的 mask 喂给采样器。
        """
        return any(self.token_feasible(t) for t in vocab)

    # ------------------------------------------------------------------
    def mask(self, logits: np.ndarray, vocab: List[str]) -> np.ndarray:
        """合法 token 掩码（对应采样前施加的 SO mask）。"""
        out = logits.astype(np.float32, copy=True)
        for tid, tok in enumerate(vocab):
            if not self.token_feasible(tok):
                out[tid] = -np.inf
        if np.all(np.isinf(out)):
            raise ValueError("语法死局：当前状态下无合法 token（schema 与模型词表冲突）")
        return out