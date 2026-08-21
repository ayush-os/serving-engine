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
        self.hash_to_block: dict[int, int] = {}

    def match_prefix(self, request: Request) -> None:
        """Walk request.prompt_token_ids in full block_size chunks against a
        chained hash (each block's hash folds in the previous block's hash,
        not just its own content) -- a block's real KV values depend on
        everything before it via causal self-attention, so content-only
        hashing could conflate two different histories that happen to share
        later-block content (see design discussion). Stops at the first
        block not in the cache: prefix matching is inherently contiguous
        from the start, there's no matching a later block without its exact
        predecessor chain matching too."""
        parent_hash = None
        num_full_blocks = request.prompt_len // self.block_size

        for i in range(num_full_blocks):
            chunk = tuple(request.prompt_token_ids[i * self.block_size:(i + 1) * self.block_size])
            h = hash((parent_hash, chunk))
            block_id = self.hash_to_block.get(h)
            if block_id is None:
                break
            request.block_table.append(block_id)
            self.blocks[block_id].ref_count += 1
            parent_hash = h

        request.num_computed_tokens = len(request.block_table) * self.block_size

    def can_allocate(self, request: Request) -> bool:
        num_blocks_needed = (request.total_len + self.block_size - 1) // self.block_size - len(request.block_table)
        return num_blocks_needed <= self.get_num_free_blocks()

    def allocate(self, request: Request) -> None:
        assert self.can_allocate(request)
        num_blocks_needed = (request.total_len + self.block_size - 1) // self.block_size - len(request.block_table)

        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop()
            request.block_table.append(block_id)
            self.blocks[block_id].ref_count += 1

    def register_computed_blocks(self, request: Request) -> None:
        """Hash and register any of request's prompt blocks that just became
        fully computed (bounded by num_computed_tokens, already updated by
        the caller before this runs) and aren't registered yet, so future
        match_prefix calls can find them. Skips/resumes via each block's
        existing content_hash instead of rehashing from block 0 every call --
        a block already registered (by this request earlier, or matched/
        forked from another request's cache) just hands back its hash as the
        next parent_hash."""
        limit = min(request.num_computed_tokens, request.prompt_len)
        num_registerable_blocks = limit // self.block_size

        parent_hash = None
        for i in range(num_registerable_blocks):
            block = self.blocks[request.block_table[i]]
            if block.content_hash is not None:
                parent_hash = block.content_hash
                continue
            chunk = tuple(request.prompt_token_ids[i * self.block_size:(i + 1) * self.block_size])
            h = hash((parent_hash, chunk))
            # First-writer-wins: two requests can independently miss the
            # cache for the same content (both arrive before either
            # registers) and each get distinct physical blocks. Only the
            # first to register becomes the canonical entry -- if the
            # second overwrote it, both blocks' content_hash would think
            # they own the same hash_to_block key, and whichever gets
            # freed second would KeyError on an already-deleted entry (or,
            # freed in the other order, silently orphan the still-live
            # block from the lookup table).
            if h not in self.hash_to_block:
                self.hash_to_block[h] = block.block_id
                block.content_hash = h
            parent_hash = h

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
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_blocks.append(block_id)
                if block.content_hash is not None:
                    del self.hash_to_block[block.content_hash]
                    block.content_hash = None
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
