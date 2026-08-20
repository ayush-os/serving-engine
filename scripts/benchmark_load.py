"""Phase 2's measurement harness: sweep a synthetic Poisson workload
(workload.py) across a range of offered request rates and measure real
throughput, TTFT, per-token decode latency, and GPU occupancy at each --
the raw numbers Phase 2's predicted-vs-real writeup (spec.md, 🧠 -- a
sounding-board conversation, not something this script does) puts next to
disagg_and_placement_notes.md's simulator predictions.

Run directly:

    python scripts/benchmark_load.py --rates 0.5,1,2,4,8,16 --duration 30 \
        --output results.csv

What this measures, and how:
  - Throughput: completed requests / (last completion - first arrival) for
    that rate's segment.
  - TTFT: time from a request's real admission (Request.arrival_time, set
    the instant add_request() is called) to its first sampled output
    token -- captures prefill compute AND any admission queueing delay,
    which is exactly the signal disagg_and_placement_notes.md's own
    Finding 3 (prefill_wait growing unboundedly past the real bottleneck)
    was built to detect.
  - Per-token decode latency: wall time of engine.step() calls where every
    scheduled request was already past its first token (a "pure decode"
    step) -- decode advances every scheduled request by exactly one token
    in lockstep, so that step's wall time IS the inter-token latency for
    every request in it, regardless of batch size.
  - GPU occupancy: nvidia-smi dmon sampled per rate segment, if available.

Content of the synthetic prompts is random token ids, not real text --
compute cost depends only on token count, not semantic content, and this
avoids needing a real text corpus to hit exact sampled lengths.
"""
import argparse
import csv
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving_engine.engine import LLMEngine
from serving_engine.workload import SyntheticRequest, generate_workload


def _random_token_ids(rng, vocab_size: int, special_ids: set, n: int) -> List[int]:
    ids = []
    while len(ids) < n:
        tid = rng.randrange(vocab_size)
        if tid not in special_ids:
            ids.append(tid)
    return ids


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[idx]


def _start_dmon() -> Optional[subprocess.Popen]:
    if not shutil.which("nvidia-smi"):
        return None
    return subprocess.Popen(
        ["nvidia-smi", "dmon", "-s", "u", "-d", "1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _stop_dmon_and_get_sm_avg(dmon: Optional[subprocess.Popen]) -> Optional[float]:
    if dmon is None:
        return None
    dmon.terminate()
    try:
        out, _ = dmon.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        dmon.kill()
        out, _ = dmon.communicate()

    header = None
    values = []
    for line in out.splitlines():
        if line.startswith("#"):
            if "sm" in line:
                header = line.lstrip("#").split()
            continue
        if header is None:
            continue
        parts = line.split()
        if len(parts) != len(header):
            continue
        try:
            values.append(float(parts[header.index("sm")]))
        except (ValueError, IndexError):
            continue
    return sum(values) / len(values) if values else None


def run_rate(
    engine,
    rate: float,
    duration: float,
    prompt_mean: float,
    output_mean: float,
    length_cv: float,
    seed: Optional[int],
    token_rng,
    vocab_size: int,
    special_ids: set,
    use_dmon: bool,
) -> Optional[dict]:
    workload = generate_workload(duration, rate, prompt_mean, output_mean, length_cv, seed)
    if not workload:
        print(f"  rate={rate}: no arrivals generated (duration too short for this rate) -- skipping")
        return None

    dmon = _start_dmon() if use_dmon else None

    pending: List[SyntheticRequest] = list(workload)
    seen_ids = set()
    ttft = {}
    completion_latency = {}
    decode_step_times = []
    step_count = 0
    num_completed = 0

    t_start = time.monotonic()
    first_arrival_wall = None
    last_completion_wall = None
    last_progress_print = t_start

    while pending or engine.scheduler.has_unfinished_requests():
        now = time.monotonic() - t_start
        while pending and pending[0].arrival_time <= now:
            sr = pending.pop(0)
            engine.add_request(
                prompt_token_ids=_random_token_ids(token_rng, vocab_size, special_ids, sr.prompt_len),
                max_new_tokens=sr.output_len,
                ignore_eos=True,
            )
            if first_arrival_wall is None:
                first_arrival_wall = time.monotonic()

        if not engine.scheduler.has_unfinished_requests():
            if pending:
                sleep_for = pending[0].arrival_time - (time.monotonic() - t_start)
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 0.05))
            continue

        step_t0 = time.monotonic()
        scheduler_output = engine.step()
        step_dt = time.monotonic() - step_t0
        step_count += 1

        is_pure_decode = bool(scheduler_output.scheduled_requests)
        for req in scheduler_output.scheduled_requests:
            if req.request_id not in seen_ids:
                is_pure_decode = False
                ttft[req.request_id] = time.monotonic() - req.arrival_time
            seen_ids.add(req.request_id)
            if req.is_finished:
                completion_latency[req.request_id] = time.monotonic() - req.arrival_time
                last_completion_wall = time.monotonic()
                num_completed += 1

        if is_pure_decode:
            decode_step_times.append(step_dt)

        if time.monotonic() - last_progress_print > 5:
            print(f"  rate={rate}: {num_completed}/{len(workload)} completed, {step_count} steps so far...")
            last_progress_print = time.monotonic()

    sm_avg = _stop_dmon_and_get_sm_avg(dmon)

    wall = (last_completion_wall - first_arrival_wall) if (first_arrival_wall and last_completion_wall) else None
    ttft_ms = [v * 1000 for v in ttft.values()]
    decode_ms = [v * 1000 for v in decode_step_times]

    return {
        "offered_rate_req_s": rate,
        "num_requests": len(workload),
        "num_completed": num_completed,
        "measured_throughput_req_s": (num_completed / wall) if wall else None,
        "ttft_mean_ms": (sum(ttft_ms) / len(ttft_ms)) if ttft_ms else None,
        "ttft_p50_ms": _percentile(ttft_ms, 50),
        "ttft_p99_ms": _percentile(ttft_ms, 99),
        "decode_latency_mean_ms": (sum(decode_ms) / len(decode_ms)) if decode_ms else None,
        "e2e_latency_mean_s": (sum(completion_latency.values()) / len(completion_latency))
        if completion_latency
        else None,
        "num_steps": step_count,
        "gpu_sm_util_avg_pct": sm_avg,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", type=str, default="0.5,1,2,4,8,16",
                         help="comma-separated offered request rates (req/s) to sweep")
    parser.add_argument("--duration", type=float, default=30.0,
                         help="seconds of Poisson arrivals to generate per rate")
    parser.add_argument("--prompt-mean", type=float, default=512)
    parser.add_argument("--output-mean", type=float, default=64)
    parser.add_argument("--length-cv", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpu-blocks", type=int, default=None,
                         help="default: auto-size from real free GPU memory")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048,
                         help="bounds total_query in the eager attention op's whole-batch score "
                              "matrix (see engine.py) -- the fixed startup activation headroom "
                              "doesn't scale with batch width, so this is the real OOM guard at "
                              "higher concurrency. Pass 0 to disable (Phase-1-era unbounded behavior).")
    parser.add_argument("--max-num-seqs", type=int, default=16,
                         help="bounds total_read (sum of concurrently-running requests' full "
                              "context lengths) the same way -- decode requests cost only 1 token "
                              "against --max-num-batched-tokens but still contribute their whole "
                              "context to the attention op's read side. Pass 0 to disable.")
    parser.add_argument("--no-dmon", action="store_true", help="skip nvidia-smi dmon sampling")
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    rates = [float(r) for r in args.rates.split(",")]

    print("Loading model and sizing KV cache pool from free GPU memory...")
    engine = LLMEngine(
        num_gpu_blocks=args.num_gpu_blocks,
        max_num_batched_tokens=args.max_num_batched_tokens or None,
        max_num_seqs=args.max_num_seqs or None,
    )
    print(
        f"KV cache pool: {engine.model_runner.num_gpu_blocks} blocks "
        f"({engine.block_manager.get_num_free_blocks()} free)\n"
        f"Scheduler caps: max_num_batched_tokens={engine.scheduler.max_num_batched_tokens}, "
        f"max_num_seqs={engine.scheduler.max_num_seqs}\n"
    )

    tokenizer = engine.model_runner.tokenizer
    vocab_size = tokenizer.vocab_size
    special_ids = set(tokenizer.all_special_ids)

    print("Warming up (one-time CUDA kernel/allocator cost)...")
    engine.generate(["Hello"], max_new_tokens=1)

    token_rng = random.Random(args.seed)

    rows = []
    for rate in rates:
        print(f"\n=== Sweeping rate={rate} req/s (duration={args.duration}s offered) ===")
        result = run_rate(
            engine, rate, args.duration, args.prompt_mean, args.output_mean, args.length_cv,
            args.seed, token_rng, vocab_size, special_ids, use_dmon=not args.no_dmon,
        )
        if result is None:
            continue
        rows.append(result)

        def _fmt(key, suffix=""):
            v = result[key]
            return f"{v:.3f}{suffix}" if v is not None else "n/a"

        print(
            f"  -> completed={result['num_completed']}/{result['num_requests']}  "
            f"throughput={_fmt('measured_throughput_req_s')} req/s  "
            f"TTFT mean/p50/p99={_fmt('ttft_mean_ms')}/{_fmt('ttft_p50_ms')}/{_fmt('ttft_p99_ms')} ms  "
            f"decode/token={_fmt('decode_latency_mean_ms')} ms  "
            f"SM util={_fmt('gpu_sm_util_avg_pct')}"
        )

    if rows:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")
    else:
        print("\nNo results produced -- every rate was skipped (duration too short for all offered rates?)")


if __name__ == "__main__":
    main()
