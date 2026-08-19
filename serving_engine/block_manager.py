from typing import Optional

from serving_engine.block import BLOCK_SIZE
from serving_engine.request import Request


class BlockManager:
    """Free-list allocation, block tables, ref counting, eviction. Yours."""

    def __init__(self, num_gpu_blocks: int, block_size: int = BLOCK_SIZE):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size

    def can_allocate(self, request: Request) -> bool:
        raise NotImplementedError  # TODO

    def allocate(self, request: Request) -> None:
        raise NotImplementedError  # TODO

    def can_append_slot(self, request: Request) -> bool:
        raise NotImplementedError  # TODO

    def append_slot(self, request: Request) -> Optional[int]:
        raise NotImplementedError  # TODO

    def free(self, request: Request) -> None:
        raise NotImplementedError  # TODO

    def fork(self, parent: Request, child: Request) -> None:
        raise NotImplementedError  # TODO

    def get_num_free_blocks(self) -> int:
        raise NotImplementedError  # TODO

    def preempt(self, request: Request) -> None:
        raise NotImplementedError  # TODO
