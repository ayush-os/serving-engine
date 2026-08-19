from dataclasses import dataclass

# Llama-3-8B-Instruct: 2 (K,V) x 32 layers x 8 kv_heads x 128 head_dim x 2 bytes (bf16)
# = 128 KiB/token -> 16 tokens/block = 2 MiB/block. Small enough to keep internal
# fragmentation negligible, large enough to keep block-table/free-list bookkeeping cheap.
BLOCK_SIZE = 16

BlockId = int


@dataclass
class Block:
    block_id: BlockId
    ref_count: int = 1
