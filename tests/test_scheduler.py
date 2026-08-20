from serving_engine.block_manager import BlockManager
from serving_engine.request import Request, RequestPhase, RequestStatus
from serving_engine.scheduler import Scheduler, SchedulerOutput


def make_request(request_id, prompt_len, max_new_tokens=256, eos_token_id=None):
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
    )


def make_scheduler(num_gpu_blocks, block_size=4, max_num_batched_tokens=None, max_num_seqs=None):
    return Scheduler(
        BlockManager(num_gpu_blocks=num_gpu_blocks, block_size=block_size),
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
    )


def test_add_request_goes_to_waiting():
    sched = make_scheduler(num_gpu_blocks=10)
    req = make_request("a", prompt_len=4)

    sched.add_request(req)

    assert req in sched.waiting
    assert req not in sched.running


def test_schedule_admits_waiting_request_when_capacity_allows():
    sched = make_scheduler(num_gpu_blocks=10, block_size=4)
    req = make_request("a", prompt_len=4)
    sched.add_request(req)

    output = sched.schedule()

    assert req in output.scheduled_requests
    assert req in sched.running
    assert req not in sched.waiting
    assert req.status == RequestStatus.RUNNING
    assert len(req.block_table) == 1


def test_schedule_does_not_admit_when_pool_too_small():
    sched = make_scheduler(num_gpu_blocks=1, block_size=4)
    req = make_request("a", prompt_len=8)  # needs 2 blocks, pool has 1
    sched.add_request(req)

    output = sched.schedule()

    assert output.scheduled_requests == []
    assert req in sched.waiting
    assert req not in sched.running


def test_schedule_continues_decode_request_with_room_in_last_block():
    sched = make_scheduler(num_gpu_blocks=10, block_size=4)
    req = make_request("a", prompt_len=3)  # ceil(3/4)=1 block, capacity=4 -- one slot of room before growth
    sched.add_request(req)
    sched.schedule()  # admits + prefills, 1 block allocated

    req.phase = RequestPhase.NEEDS_DECODE
    req.output_token_ids.append(999)  # total_len=4, still within the 1 block's capacity

    output = sched.schedule()

    assert req in output.scheduled_requests
    assert len(req.block_table) == 1  # no new block needed


def test_schedule_skips_blocked_decode_without_starving_other_candidates():
    """Regression test for the earlier break-vs-continue bug: one candidate
    that can't get a block must not prevent later candidates from being
    scheduled in the same iteration. Also covers the eviction-scoping rule:
    since `other` is still making progress, this isn't a true stall, so
    `blocked` must simply wait rather than force an eviction -- forcing one
    here would waste a recompute that isn't actually necessary, since
    `other` will eventually finish and free its block naturally."""
    sched = make_scheduler(num_gpu_blocks=1, block_size=4)

    blocked = make_request("blocked", prompt_len=3)  # consumes the whole pool
    sched.add_request(blocked)
    sched.schedule()  # ceil(3/4)=1 block, capacity=4
    blocked.phase = RequestPhase.NEEDS_DECODE
    blocked.output_token_ids = [1, 2]  # total_len=5, exceeds capacity=4 -- needs a new block, pool is empty

    other = make_request("other", prompt_len=3)
    other.phase = RequestPhase.NEEDS_DECODE
    other.status = RequestStatus.RUNNING
    other.block_table = [0]  # capacity=4 -- sufficient on its own, so this never touches the real pool
    other.output_token_ids = [1]  # total_len=4, within capacity -- no new block needed
    sched.running.append(other)

    output = sched.schedule()

    assert blocked not in output.scheduled_requests
    assert other in output.scheduled_requests
    assert output.preempted_requests == []


def test_schedule_prioritizes_running_over_waiting_when_blocks_are_scarce():
    sched = make_scheduler(num_gpu_blocks=1, block_size=4)

    running_req = make_request("r", prompt_len=4)
    running_req.phase = RequestPhase.NEEDS_DECODE
    running_req.status = RequestStatus.RUNNING
    sched.running.append(running_req)  # total_len=4, a block boundary -- needs the sole free block

    waiting_req = make_request("w", prompt_len=4)  # also needs 1 block
    sched.add_request(waiting_req)

    output = sched.schedule()

    assert running_req in output.scheduled_requests
    assert waiting_req not in output.scheduled_requests
    assert waiting_req in sched.waiting


def test_schedule_evicts_lifo_victim_on_true_stall():
    """Every running request simultaneously blocked on the pool, nothing
    scheduled at all -- a real stall, not just one unlucky candidate.
    Evicting the most recently admitted request must free exactly enough
    room for the earliest-blocked one to proceed this same step."""
    sched = make_scheduler(num_gpu_blocks=2, block_size=4)

    needs_room = make_request("needs_room", prompt_len=4)  # admitted first
    sched.block_manager.allocate(needs_room)
    needs_room.phase = RequestPhase.NEEDS_DECODE
    needs_room.status = RequestStatus.RUNNING
    needs_room.output_token_ids = [1, 2, 3, 4]  # total_len=8, a block boundary -- needs a new block
    sched.running.append(needs_room)

    victim = make_request("victim", prompt_len=4)  # admitted more recently -- the LIFO victim
    sched.block_manager.allocate(victim)
    victim.phase = RequestPhase.NEEDS_DECODE
    victim.status = RequestStatus.RUNNING
    victim.output_token_ids = [1, 2, 3, 4]  # also total_len=8, also blocked -- true stall, pool is 0/2 free
    sched.running.append(victim)

    output = sched.schedule()

    assert needs_room in output.scheduled_requests
    assert len(needs_room.block_table) == 2, "should have consumed victim's freed block"

    assert victim in output.preempted_requests
    assert victim not in output.scheduled_requests, "must not be re-admitted in the same step it was evicted"
    assert victim in sched.waiting
    assert victim not in sched.running
    assert victim.block_table == []
    assert victim.phase == RequestPhase.NEEDS_PREFILL
    assert victim.status == RequestStatus.PREEMPTED


def test_schedule_skips_decode_when_no_eviction_victim_available():
    sched = make_scheduler(num_gpu_blocks=1, block_size=4)

    only = make_request("only", prompt_len=4)
    sched.block_manager.allocate(only)
    only.phase = RequestPhase.NEEDS_DECODE
    only.status = RequestStatus.RUNNING
    only.output_token_ids = [1, 2, 3, 4]  # total_len=8, a boundary -- needs a block, pool has 0 free
    sched.running.append(only)

    output = sched.schedule()

    assert output.scheduled_requests == []
    assert output.preempted_requests == []
    assert only in sched.running, "nothing to evict -- stays put for a future step, not a crash"


def test_update_after_step_advances_phase_when_not_finished():
    sched = make_scheduler(num_gpu_blocks=10, block_size=4)
    req = make_request("a", prompt_len=4, max_new_tokens=10)
    sched.block_manager.allocate(req)
    req.output_token_ids = [1]  # this iteration's prefill step produced 1 token

    sched.update_after_step(SchedulerOutput(scheduled_requests=[req]))

    assert req.phase == RequestPhase.NEEDS_DECODE
    assert req.status != RequestStatus.FINISHED


def test_update_after_step_finishes_on_max_new_tokens():
    sched = make_scheduler(num_gpu_blocks=10, block_size=4)
    req = make_request("a", prompt_len=4, max_new_tokens=1)
    sched.block_manager.allocate(req)
    sched.running.append(req)
    req.output_token_ids = [1]  # hits max_new_tokens=1 on the very first (prefill) token

    sched.update_after_step(SchedulerOutput(scheduled_requests=[req]))

    assert req.status == RequestStatus.FINISHED
    assert req not in sched.running
    assert req.block_table == []
    assert sched.block_manager.get_num_free_blocks() == 10


def test_update_after_step_finishes_on_eos_during_prefill_iteration():
    """Regression test: finishing must not be gated on phase == NEEDS_DECODE,
    or a request that hits EOS on its very first (prefill) token gets
    scheduled for one extra, unwanted decode step."""
    sched = make_scheduler(num_gpu_blocks=10, block_size=4)
    req = make_request("a", prompt_len=4, max_new_tokens=256, eos_token_id=999)
    sched.block_manager.allocate(req)
    sched.running.append(req)
    req.output_token_ids = [999]

    sched.update_after_step(SchedulerOutput(scheduled_requests=[req]))

    assert req.status == RequestStatus.FINISHED
    assert req.phase == RequestPhase.NEEDS_PREFILL  # never advanced -- finished first
    assert req not in sched.running


def test_update_after_step_finishes_on_eos_during_decode():
    sched = make_scheduler(num_gpu_blocks=10, block_size=4)
    req = make_request("a", prompt_len=4, max_new_tokens=256, eos_token_id=999)
    sched.block_manager.allocate(req)
    req.phase = RequestPhase.NEEDS_DECODE
    req.status = RequestStatus.RUNNING
    req.output_token_ids = [1, 2, 999]
    sched.running.append(req)

    sched.update_after_step(SchedulerOutput(scheduled_requests=[req]))

    assert req.status == RequestStatus.FINISHED
    assert req not in sched.running


def test_schedule_respects_max_num_seqs_cap():
    sched = make_scheduler(num_gpu_blocks=10, block_size=4, max_num_seqs=1)
    req_a = make_request("a", prompt_len=4)
    req_b = make_request("b", prompt_len=4)
    sched.add_request(req_a)
    sched.add_request(req_b)

    output = sched.schedule()

    assert output.scheduled_requests == [req_a]
    assert req_b in sched.waiting, "ample blocks but at the seq cap -- must wait, not error"


def test_schedule_respects_max_num_batched_tokens_cap():
    sched = make_scheduler(num_gpu_blocks=10, block_size=4, max_num_batched_tokens=4)
    req_a = make_request("a", prompt_len=4)  # exactly consumes the token budget
    req_b = make_request("b", prompt_len=4)  # nothing left for this one
    sched.add_request(req_a)
    sched.add_request(req_b)

    output = sched.schedule()

    assert output.scheduled_requests == [req_a]
    assert req_b in sched.waiting


def test_schedule_admits_lone_candidate_that_exceeds_the_token_cap_alone():
    """Regression test: found via Phase 2's real load sweep, where a
    lognormal-tailed synthetic prompt exceeded max_num_batched_tokens on
    its own. Before this fix, the token-cap check applied even at
    num_batched_tokens==0, so a candidate whose own cost alone exceeds the
    cap could never be scheduled -- not now, not ever, regardless of how
    idle the rest of the system was. It's a NEEDS_PREFILL candidate, so it
    never reaches blocked_decode_candidates either, meaning nothing could
    ever unstick it: permanent starvation, silently returning an empty
    scheduled_requests forever."""
    sched = make_scheduler(num_gpu_blocks=10, block_size=4, max_num_batched_tokens=4)
    huge = make_request("huge", prompt_len=10)  # cost=10, already over the cap=4 alone
    sched.add_request(huge)

    output = sched.schedule()

    assert output.scheduled_requests == [huge]
    assert huge not in sched.waiting


def test_schedule_running_decode_still_prioritized_under_token_cap():
    """The token cap must not let a plain first-come-first-served candidate
    order starve in-flight decode work -- running is still scanned before
    waiting, so a tight budget is spent on the decode step first."""
    sched = make_scheduler(num_gpu_blocks=10, block_size=4, max_num_batched_tokens=1)

    running_req = make_request("r", prompt_len=4)
    running_req.phase = RequestPhase.NEEDS_DECODE
    running_req.status = RequestStatus.RUNNING
    sched.block_manager.allocate(running_req)
    running_req.output_token_ids = [1]
    sched.running.append(running_req)  # decode step costs 1 token -- exactly the whole budget

    waiting_req = make_request("w", prompt_len=4)  # a fresh prefill costs 4 tokens, no budget left regardless
    sched.add_request(waiting_req)

    output = sched.schedule()

    assert output.scheduled_requests == [running_req]
    assert waiting_req in sched.waiting


def test_schedule_token_cap_blocked_candidate_does_not_trigger_eviction():
    """A second candidate skipped for being over the (already partially
    spent) token budget is a scheduling choice, not a resource stall -- it
    must not be treated as eviction bait the way a genuine block-capacity
    block is. Uses two candidates, not one: a lone candidate is always
    admitted regardless of its own cost (see the "always admit the first
    candidate" fix in schedule()) specifically so a single request can
    never starve forever just because its cost alone exceeds the cap."""
    sched = make_scheduler(num_gpu_blocks=10, block_size=4, max_num_batched_tokens=1)

    first = make_request("first", prompt_len=1)
    first.phase = RequestPhase.NEEDS_DECODE
    first.status = RequestStatus.RUNNING
    sched.block_manager.allocate(first)
    first.output_token_ids = [1]
    sched.running.append(first)  # admitted -- spends the whole budget (cost=1)

    second = make_request("second", prompt_len=1)
    second.phase = RequestPhase.NEEDS_DECODE
    second.status = RequestStatus.RUNNING
    sched.block_manager.allocate(second)
    second.output_token_ids = [1]
    sched.running.append(second)  # budget already spent -- skipped by the cap, not by block capacity

    output = sched.schedule()

    assert output.scheduled_requests == [first]
    assert output.preempted_requests == []
    assert second in sched.running


def test_has_unfinished_requests_true_while_waiting():
    sched = make_scheduler(num_gpu_blocks=10, block_size=4)
    sched.add_request(make_request("a", prompt_len=4))

    assert sched.has_unfinished_requests()


def test_has_unfinished_requests_false_once_everything_finishes():
    """End-to-end regression test for the earlier missing-removal-from-
    running bug: without it, this loop never terminates."""
    sched = make_scheduler(num_gpu_blocks=10, block_size=4)
    req = make_request("a", prompt_len=4, max_new_tokens=1)
    sched.add_request(req)
    output = sched.schedule()
    req.output_token_ids = [1]

    assert sched.has_unfinished_requests()
    sched.update_after_step(output)
    assert not sched.has_unfinished_requests()
