from typing import Optional

from serving_engine.block import BLOCK_SIZE, Block
from serving_engine.request import Request


class BlockManager:
    """Free-list allocation, block tables, ref counting, eviction. Yours."""

    def __init__(self, num_gpu_blocks: int, block_size: int = BLOCK_SIZE):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.blocks = [Block(i, 0) for i in range(num_gpu_blocks)]
        self.free_ls = [True for _ in range(num_gpu_blocks)]
        self.num_free_blocks = num_gpu_blocks

    def can_allocate(self, request: Request) -> bool:
        num_blocks_needed = (request.prompt_len + self.block_size - 1) // self.block_size
        return num_blocks_needed <= self.num_free_blocks

    def allocate(self, request: Request) -> None:
        num_blocks_needed = (request.prompt_len + self.block_size - 1) // self.block_size
        self.num_free_blocks -= num_blocks_needed

        for i in range(self.num_gpu_blocks):
            if self.free_ls[i] == True:
                request.block_table.append(i)
                self.free_ls[i] = False
                num_blocks_needed -= 1
                self.blocks[i].ref_count += 1

                if num_blocks_needed == 0: break

    def can_append_slot(self, request: Request) -> bool:
        return self.num_free_blocks >= 1

    # TODO: Handle CoW case
    def append_slot(self, request: Request) -> Optional[int]:
        for i in range(self.num_gpu_blocks):
            if self.free_ls[i] == True:
                request.block_table.append(i)
                self.blocks[i].ref_count += 1
                self.free_ls[i] = False
                self.num_free_blocks -= 1
                break

    def free(self, request: Request) -> None:
        for id in request.block_table:
            self.blocks[id].ref_count -= 1
            if self.blocks[id].ref_count == 0:
                self.free_ls[id] = True
                self.num_free_blocks += 1
        request.block_table = []

    def fork(self, parent: Request, child: Request) -> None:
        child.block_table = parent.block_table.copy()
        for id in child.block_table:
            self.blocks[id].ref_count += 1

    def get_num_free_blocks(self) -> int:
        return self.num_free_blocks

    def preempt(self, request: Request) -> None:
        raise NotImplementedError  # TODO
