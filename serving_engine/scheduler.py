from dataclasses import dataclass, field
from typing import List

from serving_engine.block_manager import BlockManager
from serving_engine.request import Request


@dataclass
class SchedulerOutput:
    scheduled_requests: List[Request] = field(default_factory=list)
    preempted_requests: List[Request] = field(default_factory=list)


class Scheduler:
    """Per-iteration admission loop: prefill/decode mix, gated by
    BlockManager capacity. Yours."""

    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
        self.waiting: List[Request] = []
        self.running: List[Request] = []

    def add_request(self, request: Request) -> None:
        raise NotImplementedError  # TODO

    def schedule(self) -> SchedulerOutput:
        raise NotImplementedError  # TODO

    def update_after_step(self, scheduler_output: SchedulerOutput) -> None:
        raise NotImplementedError  # TODO

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)
