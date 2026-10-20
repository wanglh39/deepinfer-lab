"""请求与序列抽象 —— 对应 deepinfer/sequence.py。

推理框架里「一条请求」的生命周期状态机：

    WAITING -> RUNNING -> FINISHED_STOPPED / FINISHED_LENGTH_CAPPED

调度器（scheduler.py）维护 WAITING/RUNNING 两个队列，
KV cache（cache/paged_kv.py）按 block_table 为每个序列分配物理块。
output_token_ids 是解码阶段逐步 append 的结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "finished_stopped"
    FINISHED_LENGTH_CAPPED = "finished_length_capped"
    FINISHED_ABORTED = "finished_aborted"


class Sequence:
    """一条推理请求（prompt 已分词）。

    block_table: 该序列在 PagedKVCache 中的物理块编号列表。
    num_cached_prefill: 前缀缓存命中后无需重算的 token 数（见 prefix_cache.py）。
    """

    def __init__(self, seq_id: int, prompt_token_ids: List[int],
                 sampling_params: SamplingParams):
        self.seq_id = seq_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.sampling_params = sampling_params
        self.status = SequenceStatus.WAITING
        self.block_table: List[int] = []
        self.num_cached_prefill: int = 0   # 前缀缓存命中的长度
        self.num_computed: int = 0          # 已经完成前向的 token 数（断点续算）

    # ---- token 视图 ----
    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def last_token_id(self) -> int:
        return self.token_ids[-1]

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    # ---- 生命周期 ----
    def is_finished(self) -> bool:
        return self.status not in (SequenceStatus.WAITING, SequenceStatus.RUNNING)

    def finish(self, status: SequenceStatus) -> None:
        assert status.value.startswith("finished")
        self.status = status

    # ---- 供调度器使用 ----
    def num_new_tokens(self) -> int:
        """本次调度还剩多少 token 需要前向（预填充未算完的部分）。"""
        return self.num_tokens - self.num_computed

    def __repr__(self) -> str:
        return (f"Sequence(id={self.seq_id}, status={self.status.value}, "
                f"prompt={len(self.prompt_token_ids)}, out={len(self.output_token_ids)})")


@dataclass
class RequestOutput:
    """一次推理调用的对外结果（对应 entrypoints 层返回给用户的数据）。"""

    seq_id: int
    prompt: str
    output_text: str
    token_ids: List[int]
    finish_reason: Optional[str] = None
    metrics: dict = field(default_factory=dict)