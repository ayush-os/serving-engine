from typing import Optional

from serving_engine.block import BLOCK_SIZE, Block
from serving_engine.request import Request, RequestPhase, RequestStatus


class BlockManager:
    """Free-list allocation, block tables, ref counting, eviction. Yours."""

    def __init__(self, num_gpu_blocks: int, block_size: int = BLOCK_SIZE):
        self.num_gpu_blocks = num_gpu_blocks
        self.block_size = block_size
        self.blocks = [Block(i, 0) for i in range(num_gpu_blocks)]
        self.free_blocks = list(range(num_gpu_blocks))

    def can_allocate(self, request: Request) -> bool:
        num_blocks_needed = (request.total_len + self.block_size - 1) // self.block_size
        return num_blocks_needed <= self.get_num_free_blocks()

    def allocate(self, request: Request) -> None:
        assert self.can_allocate(request)
        num_blocks_needed = (request.total_len + self.block_size - 1) // self.block_size

        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            request.block_table.append(block_id)
            self.blocks[block_id].ref_count += 1

    def _has_capacity_for(self, request: Request) -> bool:
        """Whether the request's current block_table already covers
        total_len -- checked directly against actual capacity rather than
        derived from total_len % block_size, which silently assumes the
        table grew via a specific incremental history (one block at a
        time, starting from allocate()'s own ceil(total_len/block_size)).
        That assumption breaks whenever total_len is an exact multiple of
        block_size at admission time -- true after any preemption
        recompute, but also just a fresh prompt whose length happens to
        land on a block boundary -- silently leaving the table one block
        short for the very next decode step (IndexError in
        ModelRunner._flat_slot)."""
        return len(request.block_table) * self.block_size >= request.total_len

    def can_append_slot(self, request: Request) -> bool:
        if self._has_capacity_for(request):
            return True
        return self.get_num_free_blocks() >= 1

    # TODO: Handle CoW case
    def append_slot(self, request: Request) -> Optional[int]:
        if self._has_capacity_for(request):
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
        self.free(request)
        request.num_computed_tokens = 0
        request.phase = RequestPhase.NEEDS_PREFILL
        request.status = RequestStatus.PREEMPTED
