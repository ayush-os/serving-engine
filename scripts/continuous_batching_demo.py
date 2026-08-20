"""Phase 1's second checkpoint: demonstrate continuous batching -- a new,
staggered-arrival request gets folded into the batch mid-decode of an
already-running one, instead of waiting for it to finish. Run directly
(not via pytest):

    python scripts/continuous_batching_demo.py

What to look for in the output:
  - the per-step timeline: a newly-arrived request's PREFILL should show up
    in the same batch line as another request's ongoing DECODE -- that's
    the actual point of continuous batching, not just multiple requests
    existing at once.
  - the nvidia-smi dmon utilization samples at the end should show
    sustained (not intermittently zero) SM utilization across the whole
    run, confirming the GPU wasn't sitting idle between requests.
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
from serving_engine.request import RequestPhase

# (prompt, max_new_tokens, arrival_step) -- arrival_step is which
# engine.step() call this request is added before. Staggering arrivals
# (rather than adding everything up front) is what actually exercises
# continuous batching instead of just ordinary static batching.
REQUESTS = [
    ("Write a short story about a robot learning to paint.", 60, 0),
    ("What is 2 + 2?", 10, 5),
    ("Explain the theory of relativity in one paragraph.", 40, 12),
]


def main():
    print("Loading model and sizing KV cache pool from free GPU memory...")
    engine = LLMEngine(num_gpu_blocks=None)
    print(
        f"KV cache pool: {engine.model_runner.num_gpu_blocks} blocks "
        f"({engine.block_manager.get_num_free_blocks()} free)\n"
    )

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

    pending_arrivals = sorted(REQUESTS, key=lambda r: r[2])
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

        batch = [
            f"{req.request_id[:8]}:{'PREFILL' if req.phase == RequestPhase.NEEDS_PREFILL else 'DECODE'}"
            for req in scheduler_output.scheduled_requests
        ]
        print(f"[step {step_idx:3d}] {dt_ms:6.1f}ms  batch=[{', '.join(batch)}]")

        step_idx += 1

    total_wall = time.monotonic() - t_start
    print(f"\nTotal wall time for {len(REQUESTS)} staggered requests over {step_idx} steps: {total_wall:.2f}s")

    if dmon is not None:
        dmon.terminate()
        try:
            out, _ = dmon.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            dmon.kill()
            out, _ = dmon.communicate()
        print("\nnvidia-smi dmon utilization samples during the run (sm% column):")
        print(out)


if __name__ == "__main__":
    main()
