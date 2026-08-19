from typing import Optional

from serving_engine.block import BLOCK_SIZE, Block
from serving_engine.request import Request


class BlockManager:
    """Free-list allocation, block tables, ref counting, eviction. Yours."""

    def __init__(self, num_gpu_blocks: int, block_size: int = BLOCK_SIZE):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.blocks = [Block(i, 0) for i in range(num_gpu_blocks)]
        self.free_blocks = list(range(num_gpu_blocks))

    def can_allocate(self, request: Request) -> bool:
        num_blocks_needed = (request.prompt_len + self.block_size - 1) // self.block_size
        return num_blocks_needed <= len(self.free_blocks)

    def allocate(self, request: Request) -> None:
        assert self.can_allocate(request)
        num_blocks_needed = (request.prompt_len + self.block_size - 1) // self.block_size

        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            request.block_table.append(block_id)
            self.blocks[block_id].ref_count += 1

    def can_append_slot(self, request: Request) -> bool:
        return len(self.free_blocks) >= 1

    # TODO: Handle CoW case
    def append_slot(self, request: Request) -> Optional[int]:
        if request.total_len % self.block_size != 0:
            return None

        assert self.can_append_slot(request)
        block_id = self.free_blocks.pop()
        request.block_table.append(block_id)
        self.blocks[block_id].ref_count += 1
        return block_id

    def free(self, request: Request) -> None:
        for block_id in request.block_table:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self.free_blocks.append(block_id)
        request.block_table = []

    def fork(self, parent: Request, child: Request) -> None:
        child.block_table = parent.block_table.copy()
        for block_id in child.block_table:
            self.blocks[block_id].ref_count += 1

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def preempt(self, request: Request) -> None:
        raise NotImplementedError  # TODO
