import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

from serving_engine.block import BlockId


class RequestPhase(Enum):
    """Decision 2: phase is explicit, externally-visible state on the request
    itself, not an internal scheduler variable -- this is what lets Phase 4/5
    move a request's prefill and decode phases to separate processes later
    without restructuring the request object."""

    NEEDS_PREFILL = auto()
    NEEDS_DECODE = auto()


class RequestStatus(Enum):
    WAITING = auto()      # arrived, not yet admitted by the scheduler
    RUNNING = auto()       # admitted, has blocks allocated
    PREEMPTED = auto()     # was running, evicted back to WAITING to free blocks
    FINISHED = auto()


@dataclass
class Request:
    request_id: str
    prompt_token_ids: List[int]
    max_new_tokens: int = 256
    eos_token_id: int = None

    output_token_ids: List[int] = field(default_factory=list)
    phase: RequestPhase = RequestPhase.NEEDS_PREFILL
    status: RequestStatus = RequestStatus.WAITING

    # logical block index -> physical block id. Decision 2: block ids, not
    # raw pointers/tensor references, so a block table remains meaningful
    # even if a block later lives in another process's memory.
    block_table: List[BlockId] = field(default_factory=list)

    # how many of this request's tokens already have KV cache entries computed.
    num_computed_tokens: int = 0

    arrival_time: float = field(default_factory=time.monotonic)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.output_len

    @property
    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED

    def all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids
