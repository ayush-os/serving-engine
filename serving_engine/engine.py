import uuid
from typing import List

from serving_engine.block import BLOCK_SIZE
from serving_engine.block_manager import BlockManager
from serving_engine.model_runner import ModelRunner
from serving_engine.request import Request, RequestPhase, RequestStatus
from serving_engine.scheduler import Scheduler


class LLMEngine:
    """Top-level wiring: request lifecycle (arrival -> tokenize -> prefill ->
    decode loop -> completion -> block release) through the scheduler and
    block manager. This file is just plumbing -- the interesting logic lives
    in scheduler.py and block_manager.py.
    """

    def __init__(
        self,
        num_gpu_blocks: int | None = None,
        model_name: str = None,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        min_chunk_size: int | None = None,
        attn_implementation: str | None = None,
    ):
        """num_gpu_blocks=None sizes the KV cache pool from actual free GPU
        memory (see ModelRunner._infer_num_gpu_blocks) instead of a fixed
        count -- BlockManager is sized to match whatever ModelRunner resolves.

        max_num_batched_tokens/max_num_seqs were built and unit-tested in
        Scheduler but never threaded through here -- nothing in Phase 1 ran
        a batch wide enough to need them. They matter now regardless of
        attn_implementation: the fixed startup activation-memory headroom
        (gpu_memory_utilization in ModelRunner) doesn't account for an
        unbounded batch, so these caps are still the real memory guard even
        with a correctly-sized KV pool. The eager path (attn_implementation=
        "paged|eager") additionally materializes one dense [total_query,
        total_read] score matrix across the WHOLE scheduled batch on top of
        that -- Phase 4's tiled Triton kernel (the default) is what avoids
        this, see handoff.md.

        min_chunk_size: Phase 2.5's chunked-prefill floor -- see Scheduler.
        None means no floor (any leftover budget, however small, still
        admits a chunk).

        attn_implementation: None uses ModelRunner's default (the Phase 4
        triton kernel); pass "paged|eager" to compare against the dense
        HF-builtin path (see tests/test_correctness.py's oracle test)."""
        kwargs = {"model_name": model_name} if model_name else {}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model_runner = ModelRunner(num_gpu_blocks, BLOCK_SIZE, **kwargs)
        self.block_manager = BlockManager(self.model_runner.num_gpu_blocks, BLOCK_SIZE)
        self.scheduler = Scheduler(
            self.block_manager,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            min_chunk_size=min_chunk_size,
        )
        self.requests = {}

    def add_request(
        self,
        prompt: str = None,
        max_new_tokens: int = 256,
        prompt_token_ids: List[int] = None,
        ignore_eos: bool = False,
    ) -> str:
        """prompt_token_ids/ignore_eos exist for scripts/benchmark_load.py:
        Phase 2's synthetic workload needs exact, pre-sampled prompt/output
        token counts (see workload.py) rather than real text -- content
        doesn't affect prefill/decode compute cost, only token count does.
        ignore_eos forces every request to run its full sampled output
        length instead of stopping early on a randomly-sampled EOS token,
        so the realized output-length distribution matches what was
        sampled, not something the simulator this is compared against
        never modeled."""
        request_id = str(uuid.uuid4())
        if prompt_token_ids is None:
            prompt_token_ids = self.model_runner.tokenizer.encode(prompt)
        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=None if ignore_eos else self.model_runner.tokenizer.eos_token_id,
        )
        self.requests[request_id] = request
        self.scheduler.add_request(request)
        return request_id

    def step(self):
        """One engine iteration: schedule -> forward -> sample -> bookkeep.
        Depends entirely on scheduler.schedule()/block_manager and
        model_runner.forward(), so this will raise NotImplementedError
        until those are filled in -- that's expected, not a bug here."""
        scheduler_output = self.scheduler.schedule()
        if not scheduler_output.scheduled_requests:
            # A genuine block-capacity stall with no eviction victim
            # available (e.g. a lone running request needs one more block
            # than the pool has left) can still return nothing to do.
            # forward() has no real batch to run in that case -- calling it
            # anyway builds an empty token tensor, which torch.tensor([])
            # defaults to float32, crashing embed_tokens on a dtype it
            # never asked for. Skip the forward pass entirely instead.
            return scheduler_output
        logits = self.model_runner.forward(scheduler_output)
        next_tokens = logits.argmax(dim=-1)
        for req, tok in zip(scheduler_output.scheduled_requests, next_tokens):
            if req.phase == RequestPhase.NEEDS_DECODE or req.request_id in scheduler_output.prefill_final_chunk:
                req.output_token_ids.append(tok.item())
            # else: NEEDS_PREFILL, not this request's last chunk -- this
            # step's last-token logits are still mid-prompt, not a genuine
            # next-token prediction, so there's nothing to sample yet.
        self.scheduler.update_after_step(scheduler_output)
        return scheduler_output

    def generate(self, prompts: List[str], max_new_tokens: int = 256) -> List[str]:
        """Run prompts to completion. Used directly by the Phase 1
        correctness checkpoint (diff token-for-token against HF
        .generate())."""
        request_ids = [self.add_request(p, max_new_tokens) for p in prompts]
        while self.scheduler.has_unfinished_requests():
            self.step()
        outputs = []
        for rid in request_ids:
            request = self.requests[rid]
            outputs.append(self.model_runner.tokenizer.decode(request.output_token_ids))
        return outputs
