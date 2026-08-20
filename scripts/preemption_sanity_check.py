"""GPU sanity check for BlockManager.preempt()/Scheduler's LIFO eviction --
unit-tested in pure Python (test_scheduler.py, test_block_manager.py) but
never yet run against the real model/KV cache. Run directly:

    python scripts/preemption_sanity_check.py

Setup forces a genuine eviction deterministically: 3 requests with an
identical prompt length share a 3-block pool (num_gpu_blocks=3), so admission
consumes exactly the whole pool (1 block each, 0 free). All 3 then decode in
lockstep -- same prompt length means every step advances all three requests'
total_len together -- and cross the block-16 boundary on the same step,
simultaneously needing a 2nd block with none free. That's a true stall (not
just one unlucky candidate), which is exactly the condition that triggers
Scheduler's last-resort LIFO eviction.

What to look for in the output:
  - at least one "PREEMPTED" line during the run -- confirms eviction
    actually fired, not just that the scenario avoided it.
  - the run must finish without crashing (recompute-prefill's read_index
    covering only what's actually been recomputed so far is the thing most
    likely to break for real on GPU despite passing pure-Python tests).
  - all 3 final outputs should read as coherent continuations of their
    prompt, including whichever request(s) got evicted -- confirms the
    post-eviction recompute produced a correct KV cache, not just one that
    didn't crash.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving_engine.engine import LLMEngine

PROMPT = "The capital of France is"  # same prompt for all 3 -> same prompt_len -> lockstep decode
NUM_REQUESTS = 3
MAX_NEW_TOKENS = 20  # must clear the block-16 boundary (needs output_len >= ~9 here) to force the stall


def main():
    print("Loading model with a deliberately tiny KV cache pool (num_gpu_blocks=3)...")
    engine = LLMEngine(num_gpu_blocks=3)

    request_ids = [engine.add_request(PROMPT, max_new_tokens=MAX_NEW_TOKENS) for _ in range(NUM_REQUESTS)]
    print(f"Admitted {NUM_REQUESTS} requests with identical prompt_len sharing a 3-block pool\n")

    num_preemptions = 0
    step_idx = 0
    while engine.scheduler.has_unfinished_requests():
        scheduler_output = engine.step()
        if scheduler_output.preempted_requests:
            for req in scheduler_output.preempted_requests:
                num_preemptions += 1
                print(f"[step {step_idx:3d}] PREEMPTED {req.request_id[:8]} "
                      f"(had generated {req.output_len} tokens, now back to NEEDS_PREFILL)")
        step_idx += 1

    print(f"\nFinished in {step_idx} steps, {num_preemptions} preemption(s) observed.\n")
    assert num_preemptions > 0, (
        "scenario never forced an eviction -- setup assumption broke, this run proves nothing"
    )

    print("=== Final outputs ===")
    for rid in request_ids:
        req = engine.requests[rid]
        text = engine.model_runner.tokenizer.decode(req.output_token_ids)
        print(f"{rid[:8]}: {text!r}")


if __name__ == "__main__":
    main()
