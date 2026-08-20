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
    bm.allocate(req)
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
