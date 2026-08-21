from dataclasses import dataclass, field
from typing import Dict, List, Set

from serving_engine.block_manager import BlockManager
from serving_engine.request import Request, RequestPhase, RequestStatus


@dataclass
class SchedulerOutput:
    scheduled_requests: List[Request] = field(default_factory=list)
    preempted_requests: List[Request] = field(default_factory=list)
    # request_id -> tokens actually prefilled this step. Only populated for
    # NEEDS_PREFILL candidates -- decode's contribution is always exactly 1
    # token and doesn't need per-request bookkeeping here.
    prefill_chunk_sizes: Dict[str, int] = field(default_factory=dict)
    # request_ids whose chunk this step reaches total_len -- decided here,
    # at schedule time, not re-derived later from total_len itself: total_len
    # is prompt_len + output_len, and engine.step() appends this step's
    # sampled token to output_len *before* the phase transition gets
    # checked, which would silently inflate total_len out from under a
    # post-hoc re-check of the same request that just finished prefilling.
    prefill_final_chunk: Set[str] = field(default_factory=set)


class Scheduler:
    """Per-iteration admission loop: prefill/decode mix, gated by
    BlockManager capacity. Yours."""

    def __init__(
        self,
        block_manager: BlockManager,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        min_chunk_size: int | None = None,
    ):
        self.block_manager = block_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.min_chunk_size = min_chunk_size
        self.waiting: List[Request] = []
        self.running: List[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def _select_eviction_victim(self, exclude: Request) -> Request | None:
        """LIFO: evict the most recently admitted running request, excluding
        the candidate we're trying to unblock."""
        for req in reversed(self.running):
            if req is not exclude:
                return req
        return None

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        # running before waiting: in-flight decode work is deliberately
        # prioritized over admitting new requests when blocks are scarce.
        candidates = self.running.copy() + self.waiting.copy()
        blocked_decode_candidates = []
        num_batched_tokens = 0
        num_seqs = 0

        for candidate in candidates:
            if self.max_num_seqs is not None and num_seqs >= self.max_num_seqs:
                break

            # total_len, not prompt_len: after a preemption recompute,
            # num_computed_tokens is reset to 0 but output_token_ids
            # survives, so there can be already-generated tokens that also
            # need reprocessing, not just the original prompt -- matches
            # model_runner's existing all_token_ids()/total_len convention
            # for the prefill path. Unused by decode, but computed
            # unconditionally so it's always bound below.
            remaining = candidate.total_len - candidate.num_computed_tokens

            if candidate.phase == RequestPhase.NEEDS_DECODE:
                cost = 1
            else:
                # Gate on a worthwhile floor, not the full remaining amount --
                # chunked prefill only needs enough budget for one decent
                # chunk. A final chunk smaller than the floor (the request is
                # nearly done prefilling) is exempt: there's no future chunk
                # being avoided by waiting, so let it through as-is.
                cost = remaining if self.min_chunk_size is None else min(self.min_chunk_size, remaining)

            # The token-budget check only limits piling more work on top of
            # something already batched this step (num_batched_tokens > 0).
            # A lone candidate whose own cost exceeds the cap still has to
            # get in when nothing else is competing for the budget --
            # otherwise a single prompt/chunk-floor longer than
            # max_num_batched_tokens would never be schedulable, forever.
            if (
                self.max_num_batched_tokens is not None
                and num_batched_tokens > 0
                and num_batched_tokens + cost > self.max_num_batched_tokens
            ):
                continue

            if candidate.phase == RequestPhase.NEEDS_DECODE:
                if not self.block_manager.can_append_slot(candidate):
                    blocked_decode_candidates.append(candidate)
                    continue
                self.block_manager.append_slot(candidate)
                output.scheduled_requests.append(candidate)
                num_seqs += 1
                num_batched_tokens += cost
            else:  # candidate.phase == RequestPhase.NEEDS_PREFILL
                # Already RUNNING means this is a continuing chunk, not a
                # fresh admission: allocate() already reserved
                # ceil(total_len/block_size) blocks for the whole request
                # on its first admission (total_len == prompt_len then), so
                # there's nothing new to claim -- can_allocate() would just
                # be checking free-pool capacity irrelevant to this request.
                # WAITING and PREEMPTED (post-eviction readmission) both
                # need a fresh claim, same as today.
                already_running = candidate.status == RequestStatus.RUNNING
                if not already_running and not self.block_manager.can_allocate(candidate):
                    continue

                # Actual chunk fills whatever's left of the budget this
                # iteration, not just the floor used for gating above --
                # otherwise this request would only ever get floor-sized
                # chunks even when nothing else needed the rest of the budget.
                cost = (
                    remaining if self.max_num_batched_tokens is None
                    else min(remaining, self.max_num_batched_tokens - num_batched_tokens)
                )

                if not already_running:
                    self.block_manager.allocate(candidate)
                    candidate.status = RequestStatus.RUNNING
                    self.waiting = [r for r in self.waiting if r is not candidate]
                    self.running.append(candidate)

                output.scheduled_requests.append(candidate)
                output.prefill_chunk_sizes[candidate.request_id] = cost
                if candidate.num_computed_tokens + cost >= candidate.total_len:
                    output.prefill_final_chunk.add(candidate.request_id)
                num_seqs += 1
                num_batched_tokens += cost

        # Last resort: nothing at all advanced this step, so the pool is
        # genuinely exhausted with every candidate blocked on it -- without
        # intervention nothing will ever finish to free blocks naturally,
        # a true deadlock. If even one candidate above succeeded (decode
        # needing no new block, or a fresh prefill admission), forward
        # progress is still happening and blocked candidates are left to
        # wait, since something will eventually finish and free blocks on
        # its own -- no need to force a recompute.
        if not output.scheduled_requests and blocked_decode_candidates:
            retry = blocked_decode_candidates[0]
            victim = self._select_eviction_victim(exclude=retry)
            if victim is not None:
                self.block_manager.preempt(victim)
                self.running.remove(victim)
                self.waiting.append(victim)
                output.preempted_requests.append(victim)

                # Evicting one victim always frees exactly enough for one
                # blocked candidate: can_append_slot only ever needs at
                # most a single new block at a time.
                if self.block_manager.can_append_slot(retry):
                    self.block_manager.append_slot(retry)
                    output.scheduled_requests.append(retry)

        return output

    def update_after_step(self, scheduler_output: SchedulerOutput) -> None:
        for req in scheduler_output.scheduled_requests:
            # output_token_ids can be empty here now: an intermediate
            # prefill chunk (more prompt left after this one) shouldn't
            # sample a token from a partial-prompt forward pass, unlike
            # today's always-whole-prompt-in-one-step case.
            if req.output_token_ids and (req.output_len >= req.max_new_tokens or
                req.output_token_ids[-1] == req.eos_token_id):
                req.status = RequestStatus.FINISHED
                self.block_manager.free(req)
                self.running = [r for r in self.running if r is not req]
            elif req.phase == RequestPhase.NEEDS_PREFILL:
                req.num_computed_tokens += scheduler_output.prefill_chunk_sizes[req.request_id]
                if req.request_id in scheduler_output.prefill_final_chunk:
                    req.phase = RequestPhase.NEEDS_DECODE

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)
