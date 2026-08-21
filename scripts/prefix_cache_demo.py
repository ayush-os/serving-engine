"""GPU demo/benchmark for prefix caching: NUM_REQUESTS requests sharing a
common, multi-block synthetic prefix should skip a real, measurable share
of their prefill after the first one runs and registers it. Run directly:

    python scripts/prefix_cache_demo.py

Uses synthetic prompt_token_ids (same convention as scripts/
benchmark_load.py/workload.py) rather than real text, for exact control
over block alignment -- SHARED_PREFIX_LEN is deliberately a multiple of
BLOCK_SIZE so the whole prefix is matchable.

Compares two scenarios on the same total workload shape: "shared" (every
request's prompt starts with the identical SHARED_PREFIX) vs "unshared"
(every request gets its own distinct prefix of the same length, so nothing
ever matches) -- same request count, same total token volume, the only
variable is whether there's anything to actually share. Requests are added
one at a time with a step() in between so each later request's match_prefix
sees an already-registered prefix, not a naive all-added-up-front pattern
that would never actually exercise a cache hit.

What to look for in the output:
  - "shared" scenario: request 0 matches 0 tokens (nothing cached yet),
    every later request matches SHARED_PREFIX_LEN tokens at admission --
    confirms the cache-hit path fires under real load, not just in
    pure-Python tests.
  - a real wall-clock delta between "shared" and "unshared" -- disagreement
    (no measurable difference) is itself a real, worth-reporting finding,
    same discipline as every other benchmark checkpoint in this project.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving_engine.block import BLOCK_SIZE
from serving_engine.engine import LLMEngine

NUM_REQUESTS = 16
SHARED_PREFIX_LEN = BLOCK_SIZE * 32  # 512 tokens, 32 full blocks -- all matchable
SUFFIX_LEN = 16                      # unique per-request tail, never shared
MAX_NEW_TOKENS = 32

SHARED_PREFIX = list(range(SHARED_PREFIX_LEN))  # arbitrary but fixed synthetic ids


def run(scenario: str) -> float:
    engine = LLMEngine(num_gpu_blocks=2048)
    request_ids = []
    start = time.monotonic()

    for i in range(NUM_REQUESTS):
        if scenario == "shared":
            prefix = SHARED_PREFIX
        else:  # "unshared" -- distinct token-id range per request, same length, never overlaps
            prefix = list(range(i * 100_000, i * 100_000 + SHARED_PREFIX_LEN))
        suffix = list(range(900_000 + i * 100, 900_000 + i * 100 + SUFFIX_LEN))

        rid = engine.add_request(
            prompt_token_ids=prefix + suffix, max_new_tokens=MAX_NEW_TOKENS, ignore_eos=True,
        )
        matched = engine.requests[rid].num_computed_tokens
        print(f"  request {i}: matched {matched}/{SHARED_PREFIX_LEN + SUFFIX_LEN} tokens at admission")
        request_ids.append(rid)
        engine.step()  # let this request's own prefill complete/register before the next one arrives

    while engine.scheduler.has_unfinished_requests():
        engine.step()

    return time.monotonic() - start


def main():
    print(f"Scenario 'shared': {NUM_REQUESTS} requests, {SHARED_PREFIX_LEN}-token common prefix\n")
    shared_time = run("shared")

    print(f"\nScenario 'unshared': {NUM_REQUESTS} requests, distinct {SHARED_PREFIX_LEN}-token prefixes\n")
    unshared_time = run("unshared")

    print(f"\n=== Results ===")
    print(f"shared:   {shared_time:.2f}s total")
    print(f"unshared: {unshared_time:.2f}s total")
    print(f"speedup:  {unshared_time / shared_time:.2f}x")


if __name__ == "__main__":
    main()
