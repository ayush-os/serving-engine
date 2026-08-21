from serving_engine.block_manager import BlockManager
from serving_engine.request import Request, RequestPhase, RequestStatus


def make_request(request_id: str, prompt_len: int) -> Request:
    return Request(request_id=request_id, prompt_token_ids=list(range(prompt_len)))


def test_allocate_uses_ceil_div_blocks():
    bm = BlockManager(num_gpu_blocks=10)
    req = make_request("a", prompt_len=20)  # ceil(20/16) = 2 blocks
    assert bm.can_allocate(req)
    bm.allocate(req)
    assert len(req.block_table) == 2
    assert bm.get_num_free_blocks() == 8


def test_can_allocate_respects_capacity():
    bm = BlockManager(num_gpu_blocks=1)
    req = make_request("a", prompt_len=20)  # needs 2 blocks, pool has 1
    assert not bm.can_allocate(req)


def test_append_slot_consumes_one_block():
    bm = BlockManager(num_gpu_blocks=10)
    req = make_request("a", prompt_len=16)
    bm.allocate(req)  # 1 block, capacity=16
    req.output_token_ids = [1]  # total_len=17, exceeds current capacity -- a real decode step needs a new block
    free_before = bm.get_num_free_blocks()
    bm.append_slot(req)
    assert bm.get_num_free_blocks() == free_before - 1
    assert len(req.block_table) == 2


def test_free_restores_num_free_blocks():
    bm = BlockManager(num_gpu_blocks=10)
    req = make_request("a", prompt_len=32)  # 2 blocks
    bm.allocate(req)
    assert bm.get_num_free_blocks() == 8
    bm.free(req)
    assert bm.get_num_free_blocks() == 10


def test_free_then_reallocate_does_not_exhaust_pool():
    bm = BlockManager(num_gpu_blocks=2)
    req_a = make_request("a", prompt_len=32)  # 2 blocks -- fills the pool
    bm.allocate(req_a)
    bm.free(req_a)

    req_b = make_request("b", prompt_len=32)
    assert bm.can_allocate(req_b)
    bm.allocate(req_b)


def test_free_clears_block_table():
    bm = BlockManager(num_gpu_blocks=10)
    req = make_request("a", prompt_len=16)
    bm.allocate(req)
    bm.free(req)
    assert req.block_table == []


def test_fork_shares_parent_blocks_and_bumps_ref_count():
    bm = BlockManager(num_gpu_blocks=10)
    parent = make_request("p", prompt_len=16)
    bm.allocate(parent)
    child = make_request("c", prompt_len=16)

    bm.fork(parent, child)

    assert child.block_table == parent.block_table
    assert bm.blocks[parent.block_table[0]].ref_count == 2


def test_fork_child_block_table_is_independent_list():
    bm = BlockManager(num_gpu_blocks=10)
    parent = make_request("p", prompt_len=16)
    bm.allocate(parent)
    child = make_request("c", prompt_len=16)
    bm.fork(parent, child)
    child.output_token_ids = [1]  # total_len=17, exceeds the forked capacity -- needs a new block

    bm.append_slot(child)

    assert len(parent.block_table) == 1, "child append_slot must not mutate parent's block_table"
    assert len(child.block_table) == 2


def test_allocate_sizes_by_total_len_not_prompt_len():
    """A recompute after preemption needs blocks for prompt + everything
    already generated, not just the prompt -- allocate() must size off
    total_len, or a recompute-prefill would run out of blocks mid-forward."""
    bm = BlockManager(num_gpu_blocks=10)
    req = make_request("a", prompt_len=8)
    req.output_token_ids = list(range(10))  # total_len=18, ceil(18/16)=2 blocks
    assert bm.can_allocate(req)
    bm.allocate(req)
    assert len(req.block_table) == 2


def test_preempt_frees_blocks_and_resets_request_for_recompute():
    bm = BlockManager(num_gpu_blocks=10)
    req = make_request("a", prompt_len=16)
    req.output_token_ids = [1, 2, 3]
    req.num_computed_tokens = 19
    req.phase = RequestPhase.NEEDS_DECODE
    req.status = RequestStatus.RUNNING
    bm.allocate(req)

    bm.preempt(req)

    assert req.block_table == []
    assert bm.get_num_free_blocks() == 10
    assert req.num_computed_tokens == 0
    assert req.phase == RequestPhase.NEEDS_PREFILL
    assert req.status == RequestStatus.PREEMPTED


def test_boundary_aligned_admission_gets_room_for_the_next_decode_step():
    """Regression test for a real GPU crash: allocate() sizes exactly
    ceil(total_len/block_size) blocks with no headroom. When total_len at
    admission time is an exact multiple of block_size -- true after any
    preemption recompute, or just a fresh prompt whose length happens to
    land on a block boundary -- the very next decode step needs a block
    that was never provisioned, and ModelRunner._flat_slot indexes past
    the end of block_table. can_append_slot/append_slot must check real
    capacity against total_len directly, not the total_len % block_size
    shortcut, which silently assumed a specific incremental-growth
    history that allocate() doesn't follow."""
    bm = BlockManager(num_gpu_blocks=10, block_size=16)
    req = make_request("a", prompt_len=16)  # exactly one block, no headroom
    bm.allocate(req)
    assert len(req.block_table) == 1

    req.output_token_ids = [1]  # total_len=17 -- write position 16 needs a 2nd block
    assert bm.can_append_slot(req)
    bm.append_slot(req)
    assert len(req.block_table) == 2, "must have grown -- position 16 isn't covered by 1 block"


def test_match_prefix_no_match_on_empty_cache():
    bm = BlockManager(num_gpu_blocks=10, block_size=16)
    req = make_request("a", prompt_len=32)

    bm.match_prefix(req)

    assert req.block_table == []
    assert req.num_computed_tokens == 0


def test_match_prefix_matches_after_registration():
    bm = BlockManager(num_gpu_blocks=10, block_size=16)
    req_a = make_request("a", prompt_len=32)  # 2 full blocks
    bm.allocate(req_a)
    req_a.num_computed_tokens = 32
    bm.register_computed_blocks(req_a)

    req_b = make_request("b", prompt_len=32)  # identical content (range(32))
    bm.match_prefix(req_b)

    assert req_b.block_table == req_a.block_table
    assert req_b.num_computed_tokens == 32
    assert bm.blocks[req_a.block_table[0]].ref_count == 2


def test_match_prefix_stops_at_first_diverging_block():
    bm = BlockManager(num_gpu_blocks=10, block_size=16)
    req_a = make_request("a", prompt_len=32)  # blocks: [0..15], [16..31]
    bm.allocate(req_a)
    req_a.num_computed_tokens = 32
    bm.register_computed_blocks(req_a)

    # same first block, diverges in the second
    req_b = Request(request_id="b", prompt_token_ids=list(range(16)) + [999] * 16)
    bm.match_prefix(req_b)

    assert len(req_b.block_table) == 1
    assert req_b.block_table[0] == req_a.block_table[0]
    assert req_b.num_computed_tokens == 16


def test_match_prefix_ignores_partial_trailing_block():
    bm = BlockManager(num_gpu_blocks=10, block_size=16)
    req_a = make_request("a", prompt_len=16)  # exactly 1 full block
    bm.allocate(req_a)
    req_a.num_computed_tokens = 16
    bm.register_computed_blocks(req_a)

    # shares the full first block, plus 4 extra tokens that never form a
    # full block and so are never hashed/matched
    req_b = Request(request_id="b", prompt_token_ids=list(range(16)) + [1, 2, 3, 4])
    bm.match_prefix(req_b)

    assert len(req_b.block_table) == 1
    assert req_b.num_computed_tokens == 16


def test_register_computed_blocks_matches_regardless_of_chunking():
    """Registering in two separate calls (as chunked prefill would) must
    produce the same hash chain as registering all at once -- resuming
    parent_hash from an already-registered block's content_hash has to
    reconstruct the identical value a fresh, unchunked walk would."""
    bm_chunked = BlockManager(num_gpu_blocks=10, block_size=16)
    req_chunked = make_request("a", prompt_len=32)
    bm_chunked.allocate(req_chunked)
    req_chunked.num_computed_tokens = 16
    bm_chunked.register_computed_blocks(req_chunked)
    req_chunked.num_computed_tokens = 32
    bm_chunked.register_computed_blocks(req_chunked)

    bm_oneshot = BlockManager(num_gpu_blocks=10, block_size=16)
    req_oneshot = make_request("b", prompt_len=32)
    bm_oneshot.allocate(req_oneshot)
    req_oneshot.num_computed_tokens = 32
    bm_oneshot.register_computed_blocks(req_oneshot)

    hashes_chunked = [bm_chunked.blocks[bid].content_hash for bid in req_chunked.block_table]
    hashes_oneshot = [bm_oneshot.blocks[bid].content_hash for bid in req_oneshot.block_table]
    assert hashes_chunked == hashes_oneshot


def test_register_computed_blocks_only_registers_computed_prefix():
    bm = BlockManager(num_gpu_blocks=10, block_size=16)
    req = make_request("a", prompt_len=32)
    bm.allocate(req)
    req.num_computed_tokens = 16  # only the first block computed so far

    bm.register_computed_blocks(req)

    assert bm.blocks[req.block_table[0]].content_hash is not None
    assert bm.blocks[req.block_table[1]].content_hash is None
    assert len(bm.hash_to_block) == 1


def test_register_computed_blocks_first_writer_wins_on_collision():
    """Two requests can independently miss the cache for the same content
    (both arrive before either registers) and land on distinct physical
    blocks for semantically-identical content. Registration must not let
    the second overwrite the first's hash_to_block entry, or freeing them
    in either order KeyErrors/corrupts the surviving entry."""
    bm = BlockManager(num_gpu_blocks=10, block_size=16)
    req_a = make_request("a", prompt_len=16)
    req_b = make_request("b", prompt_len=16)  # identical content, but no match_prefix --
                                               # simulates both missing the cache before either registers
    bm.allocate(req_a)
    bm.allocate(req_b)
    assert req_a.block_table != req_b.block_table  # distinct physical blocks despite identical content

    req_a.num_computed_tokens = 16
    req_b.num_computed_tokens = 16
    bm.register_computed_blocks(req_a)
    bm.register_computed_blocks(req_b)

    assert len(bm.hash_to_block) == 1  # only the first writer's entry survives
    assert bm.blocks[req_a.block_table[0]].content_hash is not None
    assert bm.blocks[req_b.block_table[0]].content_hash is None  # never became canonical

    # freeing in either order must not KeyError or corrupt the surviving entry
    bm.free(req_b)
    assert len(bm.hash_to_block) == 1
    bm.free(req_a)
    assert bm.hash_to_block == {}


def test_free_invalidates_hash_entry():
    bm = BlockManager(num_gpu_blocks=10, block_size=16)
    req = make_request("a", prompt_len=16)
    bm.allocate(req)
    req.num_computed_tokens = 16
    bm.register_computed_blocks(req)
    assert len(bm.hash_to_block) == 1

    bm.free(req)

    assert bm.hash_to_block == {}


def test_free_does_not_invalidate_still_shared_block():
    bm = BlockManager(num_gpu_blocks=10, block_size=16)
    req_a = make_request("a", prompt_len=16)
    bm.allocate(req_a)
    req_a.num_computed_tokens = 16
    bm.register_computed_blocks(req_a)

    req_b = make_request("b", prompt_len=16)  # identical content, matches and shares
    bm.match_prefix(req_b)
    assert len(req_b.block_table) == 1

    bm.free(req_a)

    assert bm.get_num_free_blocks() == 9  # still held by req_b
    assert len(bm.hash_to_block) == 1
    assert bm.blocks[req_b.block_table[0]].content_hash is not None


def test_can_allocate_accounts_for_already_matched_blocks():
    """Regression test for the delta fix: a request that matched 1 of its
    2 needed blocks via a shared prefix should only need capacity for the
    remaining 1, not the full ceil(total_len/block_size)."""
    bm = BlockManager(num_gpu_blocks=2, block_size=16)
    req_a = make_request("a", prompt_len=16)  # 1 block, fills half the pool
    bm.allocate(req_a)
    req_a.num_computed_tokens = 16
    bm.register_computed_blocks(req_a)

    req_b = Request(request_id="b", prompt_token_ids=list(range(32)))  # shares block 0, needs 1 more
    bm.match_prefix(req_b)
    assert len(req_b.block_table) == 1

    assert bm.can_allocate(req_b)  # ceil(32/16) - 1 = 1, exactly what's free
    bm.allocate(req_b)
    assert len(req_b.block_table) == 2
    assert bm.get_num_free_blocks() == 0
