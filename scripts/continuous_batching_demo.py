"""Phase 1's second checkpoint: demonstrate continuous batching -- a new,
staggered-arrival request gets folded into the batch mid-decode of an
already-running one, instead of waiting for it to finish. Then compares
total wall time against a naive baseline with no iteration-level batching
at all (each request runs alone, start to finish, before the next is
admitted) to show the actual speedup, not just that admission didn't stall.
Run directly (not via pytest):

    python scripts/continuous_batching_demo.py

What to look for in the output:
  - the per-step timeline: a newly-arrived request's PREFILL should show up
    in the same batch line as another request's ongoing DECODE -- that's
    the actual point of continuous batching, not just multiple requests
    existing at once.
  - the nvidia-smi dmon utilization samples should show sustained (not
    intermittently zero) SM utilization across the run.
  - the final comparison: naive-sequential wall time vs. continuous-batched
    wall time for the same 3 requests. The speedup comes from decode being
    memory-bandwidth-bound, not compute-bound -- batching several
    sequences' decode into one step costs about the same wall time as one
    sequence alone, so continuous batching needs fewer total steps to
    finish the same workload. With only 3 requests here the speedup is
    real but modest; it grows with concurrency, which is more Phase 2's
    territory than this demo's.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Run as a plain script (`python scripts/continuous_batching_demo.py`), so
# only scripts/ lands on sys.path by default -- add the repo root so
# serving_engine is importable without needing `-m` or an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving_engine.engine import LLMEngine

# (prompt, max_new_tokens, arrival_step) -- arrival_step is which
# engine.step() call this request is added before, used only by the
# continuous-batching trial. The naive trial ignores arrival_step: with no
# batching at all, when a request "arrives" doesn't change the naive
# server's total completion time, since nothing can ever overlap anyway.
REQUESTS = [
    ("Write a short story about a robot learning to paint.", 60, 0),
    ("What is 2 + 2?", 10, 5),
    ("Explain the theory of relativity in one paragraph.", 40, 12),
]


def run_continuous_batched(engine) -> float:
    """Staggered arrivals through add_request()/step() -- the scheduler
    folds each new arrival into whatever's already running."""
    pending_arrivals = sorted(REQUESTS, key=lambda r: r[2])
    seen_request_ids = set()
    step_idx = 0
    t_start = time.monotonic()

    while pending_arrivals or engine.scheduler.has_unfinished_requests():
        while pending_arrivals and pending_arrivals[0][2] == step_idx:
            prompt, max_new_tokens, _ = pending_arrivals.pop(0)
            rid = engine.add_request(prompt, max_new_tokens=max_new_tokens)
            print(f"[step {step_idx:3d}]           request {rid[:8]} ARRIVED: {prompt!r}")

        t0 = time.monotonic()
        scheduler_output = engine.step()
        dt_ms = (time.monotonic() - t0) * 1000

        # engine.step() flips req.phase NEEDS_PREFILL -> NEEDS_DECODE in
        # place before returning, on the same Request objects listed here --
        # so reading .phase now would always show DECODE. Track "seen
        # before" ourselves instead to label each request's actual first
        # scheduled step as PREFILL.
        batch = []
        for req in scheduler_output.scheduled_requests:
            label = "PREFILL" if req.request_id not in seen_request_ids else "DECODE"
            seen_request_ids.add(req.request_id)
            batch.append(f"{req.request_id[:8]}:{label}")
        print(f"[step {step_idx:3d}] {dt_ms:6.1f}ms  batch=[{', '.join(batch)}]")

        step_idx += 1

    total_wall = time.monotonic() - t_start
    print(f"\nContinuous-batched: {len(REQUESTS)} staggered requests over {step_idx} steps in {total_wall:.2f}s")
    return total_wall


def run_naive_sequential(engine) -> float:
    """No iteration-level batching: each request runs alone, start to
    finish, before the next is admitted -- the baseline continuous
    batching is meant to beat."""
    t_start = time.monotonic()
    for prompt, max_new_tokens, _ in REQUESTS:
        t0 = time.monotonic()
        engine.generate([prompt], max_new_tokens=max_new_tokens)
        print(f"  {time.monotonic() - t0:5.2f}s  {prompt!r}")

    total_wall = time.monotonic() - t_start
    print(f"\nNaive sequential: {len(REQUESTS)} requests, one at a time, in {total_wall:.2f}s")
    return total_wall


def main():
    print("Loading model and sizing KV cache pool from free GPU memory...")
    engine = LLMEngine(num_gpu_blocks=None)
    print(
        f"KV cache pool: {engine.model_runner.num_gpu_blocks} blocks "
        f"({engine.block_manager.get_num_free_blocks()} free)\n"
    )

    # Whichever trial runs first eats one-time CUDA kernel/allocator
    # warm-up cost (first-ever step is ~20x slower than steady-state) --
    # run one throwaway request now so that cost lands here, not inside
    # whichever trial happens to go first below.
    print("Warming up...")
    engine.generate(["Hello"], max_new_tokens=1)

    dmon = None
    if shutil.which("nvidia-smi"):
        dmon = subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    else:
        print("nvidia-smi not found -- skipping GPU utilization sampling.\n")

    print("=== Continuous batching (staggered arrivals) ===")
    continuous_wall = run_continuous_batched(engine)

    print("\n=== Naive sequential (no iteration-level batching) ===")
    naive_wall = run_naive_sequential(engine)

    if dmon is not None:
        dmon.terminate()
        try:
            out, _ = dmon.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            dmon.kill()
            out, _ = dmon.communicate()
        print("\nnvidia-smi dmon utilization samples across the whole run (sm% column):")
        print(out)

    print(f"\n=== Result: {naive_wall / continuous_wall:.2f}x speedup from continuous batching "
          f"({naive_wall:.2f}s naive -> {continuous_wall:.2f}s continuous) ===")


if __name__ == "__main__":
    main()
