import uuid
from typing import List

from serving_engine.block import BLOCK_SIZE
from serving_engine.block_manager import BlockManager
from serving_engine.model_runner import ModelRunner
from serving_engine.request import Request, RequestStatus
from serving_engine.scheduler import Scheduler


class LLMEngine:
    """Top-level wiring: request lifecycle (arrival -> tokenize -> prefill ->
    decode loop -> completion -> block release) through the scheduler and
    block manager. This file is just plumbing -- the interesting logic lives
    in scheduler.py and block_manager.py.
    """

    def __init__(self, num_gpu_blocks: int, model_name: str = None):
        self.model_runner = ModelRunner(model_name) if model_name else ModelRunner()
        self.block_manager = BlockManager(num_gpu_blocks, BLOCK_SIZE)
        self.scheduler = Scheduler(self.block_manager)
        self.requests = {}

    def add_request(self, prompt: str, max_new_tokens: int = 256) -> str:
        request_id = str(uuid.uuid4())
        prompt_token_ids = self.model_runner.tokenizer.encode(prompt)
        request = Request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.model_runner.tokenizer.eos_token_id,
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
        logits = self.model_runner.forward(scheduler_output)
        # TODO: greedy-sample next token per request (argmax -- correctness
        # oracle needs deterministic match against HF .generate()) and
        # append to output_token_ids. update_after_step() below reads that
        # to decide phase/finished transitions -- must run after this.
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
