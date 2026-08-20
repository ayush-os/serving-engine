from dataclasses import dataclass, field
from typing import List

from serving_engine.block_manager import BlockManager
from serving_engine.request import Request, RequestPhase, RequestStatus


@dataclass
class SchedulerOutput:
    scheduled_requests: List[Request] = field(default_factory=list)
    preempted_requests: List[Request] = field(default_factory=list)


class Scheduler:
    """Per-iteration admission loop: prefill/decode mix, gated by
    BlockManager capacity. Yours."""

    def __init__(self, block_manager: BlockManager):
        self.block_manager = block_manager
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

    # TODO: enforce max_num_batched_tokens/max_num_seqs
    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        # running before waiting: in-flight decode work is deliberately
        # prioritized over admitting new requests when blocks are scarce.
        candidates = self.running.copy() + self.waiting.copy()
        blocked_decode_candidates = []

        for candidate in candidates:
            if candidate.phase == RequestPhase.NEEDS_DECODE:
                if not self.block_manager.can_append_slot(candidate):
                    blocked_decode_candidates.append(candidate)
                    continue
                self.block_manager.append_slot(candidate)
                output.scheduled_requests.append(candidate)
            elif candidate.phase == RequestPhase.NEEDS_PREFILL and self.block_manager.can_allocate(candidate):
                self.block_manager.allocate(candidate)
                output.scheduled_requests.append(candidate)
                candidate.status = RequestStatus.RUNNING
                self.waiting = [r for r in self.waiting if r is not candidate]
                self.running.append(candidate)

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
            if (req.output_len >= req.max_new_tokens or
                req.output_token_ids[-1] == req.eos_token_id):
                req.status = RequestStatus.FINISHED
                self.block_manager.free(req)
                self.running = [r for r in self.running if r is not req]
            elif req.phase == RequestPhase.NEEDS_PREFILL:
                req.phase = RequestPhase.NEEDS_DECODE

    def has_unfinished_requests(self) -> bool:
        return bool(self.waiting or self.running)
