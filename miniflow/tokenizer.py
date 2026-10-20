"""分词器 —— 复刻 deepseek-recipe 的核心思想（教学简化版）。

deepseek-recipe（Rust）在编解码层做了三件关键的事，miniflow 各复刻其精神：

1. **特殊 token 防注入**：用户输入里字面出现 "young " 这类控制标记文本时，
   不能被当成控制 token —— `escape_special()` 把用户文本消毒后，
   字面特殊 token 只会编码成普通 token 序列，无法改变模型的行为分界。

2. **分段编码**：prompt 模板由「特殊 token + 普通文本段」拼接而成，
   段与段独立编码再拼接（BPE 的正确性保证：不会跨段合并出歧义 token）。

3. **流式解码**：decode 过滤特殊 token，只返回可见文本。

实现为「字符级起步的可训练 BPE」：初始词表 = 特殊 token + 字符集，
`train()` 用经典最高频对合并算法扩充词表（GPT 系列同源）。
空格映射为 '▁' 字符参与词表（GPT-2 的做法）。
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Tuple

SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|pad|>", "young ", "old "]
CONTROL_TOKENS = ["young ", "old "]          # 会被用户输入伪造的敏感 token
SPECIAL_TOKEN_IDS: Dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
ID_TO_SPECIAL: Dict[int, str] = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}
BOS_ID = SPECIAL_TOKEN_IDS["<|bos|>"]
EOS_ID = SPECIAL_TOKEN_IDS["<|eos|>"]
ESCAPE = "\\"
SPACE = "▁"


def escape_special(text: str) -> str:
    """用户输入消毒：给字面控制 token 加转义前缀（防注入）。"""
    out = text.replace(ESCAPE, ESCAPE + ESCAPE)
    for tok in CONTROL_TOKENS:
        out = out.replace(tok, ESCAPE + tok)
    return out


class Tokenizer:
    """字符级 + BPE 合并的可训练分词器。

    vocab 布局：[0, len(SPECIAL)) 特殊 token；之后先放单字符，再放 merge 出的子词。
    """

    def __init__(self, vocab: List[str], merges: List[Tuple[str, str]]):
        assert len(vocab) >= len(SPECIAL_TOKENS)
        self.vocab = vocab
        self.merges = merges
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    # ------------------------------------------------------------------
    @classmethod
    def train(cls, corpus: str, target_vocab: int = 200) -> "Tokenizer":
        """经典 BPE 训练：初始词表 = 特殊 token + 语料字符，迭代合并最高频对。"""
        corpus = corpus.replace(" ", SPACE)
        chars = sorted(set(corpus))
        vocab: List[str] = SPECIAL_TOKENS + chars
        merges: List[Tuple[str, str]] = []
        words = [[c for c in w] for w in corpus.split(SPACE) if w]
        while len(vocab) < target_vocab and words:
            pair_counts = Counter(
                tuple(w[i:i + 2]) for w in words for i in range(len(w) - 1))
            if not pair_counts:
                break
            best, _ = pair_counts.most_common(1)[0]
            merges.append(best)
            new_token = best[0] + best[1]
            vocab.append(new_token)
            words = [cls._merge_word(w, best, new_token) for w in words]
        return cls(vocab, merges)

    @staticmethod
    def _merge_word(word: List[str], pair: Tuple[str, str], new_token: str) -> List[str]:
        out, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                out.append(new_token)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """编码。段 = 特殊 token（模板层合法使用）或普通文本段（独立 BPE）。"""
        ids: List[int] = [BOS_ID] if add_bos else []
        for segment in self._split_on_special(text):
            if segment in SPECIAL_TOKEN_IDS:
                ids.append(SPECIAL_TOKEN_IDS[segment])
            else:
                ids.extend(self._encode_plain(segment))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    def _split_on_special(self, text: str) -> List[str]:
        """按转义符与字面特殊 token 切分。

        转义序列 ``\\X``：X 按字面进入普通段（含特殊 token 的首字符）。
        未转义的特殊 token 字符串：视为模板层合法的控制 token，单独成段。
        """
        segments: List[str] = []
        buf: List[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == ESCAPE and i + 1 < len(text):
                buf.append(text[i + 1])
                i += 2
                continue
            matched = next((tok for tok in CONTROL_TOKENS
                            if text.startswith(tok, i)), None)
            if matched is not None:
                if buf:
                    segments.append("".join(buf))
                    buf = []
                segments.append(matched)
                i += len(matched)
            else:
                buf.append(ch)
                i += 1
        if buf:
            segments.append("".join(buf))
        return segments

    def _encode_plain(self, segment: str) -> List[int]:
        """BPE 编码一段普通文本（空格 -> '▁'；按词独立编码）。"""
        ids: List[int] = []
        for word in segment.replace(" ", SPACE).split(SPACE):
            word = SPACE + word if ids else word
            pieces = list(word)
            while True:
                pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
                known = [(p, self.merge_ranks[p]) for p in pairs if p in self.merge_ranks]
                if not known:
                    break
                best = min(known, key=lambda pr: pr[1])[0]
                pieces = self._merge_word(pieces, best, best[0] + best[1])
            ids.extend(self.token_to_id[p] for p in pieces)
        return ids

    # ------------------------------------------------------------------
    def decode(self, token_ids: List[int]) -> str:
        """解码为可见文本（特殊 token 不进入输出）。

        不做 strip：前导 '▁' 承载「这是续写的新单词」语义，
        结构化输出的模式匹配依赖它（去掉会吞掉首个空格）。
        """
        out: List[str] = []
        for tid in token_ids:
            if tid in ID_TO_SPECIAL:
                continue
            out.append(self.vocab[tid])
        return "".join(out).replace(SPACE, " ")

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])