from dataclasses import dataclass
from typing import Optional

# Llama-3-8B-Instruct: 2 (K,V) x 32 layers x 8 kv_heads x 128 head_dim x 2 bytes (bf16)
# = 128 KiB/token -> 16 tokens/block = 2 MiB/block. Small enough to keep internal
# fragmentation negligible, large enough to keep block-table/free-list bookkeeping cheap.
BLOCK_SIZE = 16

BlockId = int


@dataclass
class Block:
    block_id: BlockId
    ref_count: int = 1
    # Chained hash of this block's token content once fully computed, for
    # prefix-cache lookup. None until registered (allocated-but-uncomputed
    # or content-changing blocks are never matchable) and cleared again once
    # ref_count drops to 0 (a freed block's physical slot gets reused, so a
    # stale hash must never keep pointing at it).
    content_hash: Optional[int] = None
