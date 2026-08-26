"""Phase 2 (spec-agent-pcie.md): run N concurrent, independent real agent
sessions against the engine and log real per-request timestamps, per-turn
prompt lengths, inter-request gaps, and prefix-cache matches at admission.

This script only generates and logs the real traffic -- it deliberately
does NOT compute the real-vs-synthetic comparison, the hit-rate writeup,
or the throughput/TTFT interpretation. Those are Phase 2's actual "first
real comparison" (spec's own 🧠), meant to be looked at and reasoned about,
not produced mechanically here. What this script gives you:

  - <prefix>_turns.csv: one row per internal ReAct turn across every
    session -- arrival time (relative to run start), prompt length, tool
    used, and matched_tokens_at_admission (the real prefix-cache hit
    signal, same definition scripts/prefix_cache_demo.py used).
  - <prefix>_summary.csv: one row of aggregate throughput/TTFT/decode-
    latency/step-kind/SM-util numbers, computed with the exact same
    methodology as scripts/benchmark_load.py's run_rate() (imported
    directly, not reimplemented), so it's a real apples-to-apples
    comparison point once you also run benchmark_load.py.

To get a comparable synthetic run: this script prints the realized
aggregate request rate (num_turns / wall_clock_s) at the end -- pass that
as a single value to `benchmark_load.py --rates <that_value>` (with the
same --max-num-batched-tokens/--max-num-seqs/--min-chunk-size/
--num-gpu-blocks flags) for the matched-load comparison Phase 2 wants.

Run directly:

    python scripts/run_phase2_agent_traffic.py --num-sessions 8
"""
import argparse
import csv
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving_engine.engine import LLMEngine
from agent.engine_driver import EngineDriver
from agent.loop import AgentSession
from agent.tasks import TASK_SEQUENCES
from scripts.benchmark_load import _percentile, _start_dmon, _stop_dmon_and_get_sm_avg


def _drive_session(session: AgentSession, task_sequence, start_after_s: float):
    time.sleep(start_after_s)
    for user_message in task_sequence:
        session.run(user_message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sessions", type=int, default=8)
    parser.add_argument("--stagger-max-s", type=float, default=3.0,
                         help="each session's start is delayed by a random "
                              "Uniform(0, this) jitter, so sessions aren't "
                              "artificially synchronized at t=0 -- a fixed "
                              "simultaneous start would itself be an unreal "
                              "traffic pattern.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpu-blocks", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--min-chunk-size", type=int, default=None)
    parser.add_argument("--no-dmon", action="store_true")
    parser.add_argument("--output-prefix", type=str, default="phase2_agent_traffic")
    args = parser.parse_args()

    print("Loading model and sizing KV cache pool from free GPU memory...")
    engine = LLMEngine(
        num_gpu_blocks=args.num_gpu_blocks,
        max_num_batched_tokens=args.max_num_batched_tokens or None,
        max_num_seqs=args.max_num_seqs or None,
        min_chunk_size=args.min_chunk_size,
    )
    print(
        f"KV cache pool: {engine.model_runner.num_gpu_blocks} blocks "
        f"({engine.block_manager.get_num_free_blocks()} free)\n"
        f"Scheduler caps: max_num_batched_tokens={engine.scheduler.max_num_batched_tokens}, "
        f"max_num_seqs={engine.scheduler.max_num_seqs}, "
        f"min_chunk_size={engine.scheduler.min_chunk_size}\n"
    )
    tokenizer = engine.model_runner.tokenizer

    print("Warming up (one-time CUDA kernel/allocator cost)...")
    engine.generate(["Hello"], max_new_tokens=1)

    driver = EngineDriver(engine)
    dmon = _start_dmon() if not args.no_dmon else None
    driver.start()

    rng = random.Random(args.seed)
    sessions = []
    threads = []
    for i in range(args.num_sessions):
        task_sequence = TASK_SEQUENCES[i % len(TASK_SEQUENCES)]
        session = AgentSession(driver, tokenizer, session_id=f"session-{i}")
        sessions.append(session)
        jitter = rng.uniform(0, args.stagger_max_s)
        threads.append(threading.Thread(target=_drive_session, args=(session, task_sequence, jitter)))

    print(f"Running {args.num_sessions} concurrent agent sessions...")
    wall_start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - wall_start

    driver.stop()
    sm_avg = _stop_dmon_and_get_sm_avg(dmon)

    turns = [row for session in sessions for row in session.turn_log]
    for row in turns:
        row["arrival_time_rel_s"] = row.pop("arrival_time") - wall_start

    turns_path = f"{args.output_prefix}_turns.csv"
    with open(turns_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(turns[0].keys()))
        writer.writeheader()
        writer.writerows(turns)
    print(f"Wrote {len(turns)} per-turn rows to {turns_path}")

    ttft_ms = [v * 1000 for v in driver.ttft.values()]
    total_step_time = sum(driver.step_kind_time.values())
    prefill_involved_time = driver.step_kind_time["pure_prefill"] + driver.step_kind_time["mixed"]

    def _mean_ms(kind):
        count = driver.step_kind_count[kind]
        return (driver.step_kind_time[kind] / count * 1000) if count else None

    total_matched = sum(row["matched_tokens_at_admission"] for row in turns)
    total_prompt_tokens = sum(row["prompt_len"] for row in turns)
    num_requests_with_any_match = sum(1 for row in turns if row["matched_tokens_at_admission"] > 0)

    summary = {
        "num_sessions": args.num_sessions,
        "num_turns": len(turns),
        "num_completed": len(driver.completion_time),
        "wall_clock_s": wall,
        "realized_aggregate_rate_req_s": len(turns) / wall if wall else None,
        "ttft_mean_ms": (sum(ttft_ms) / len(ttft_ms)) if ttft_ms else None,
        "ttft_p50_ms": _percentile(ttft_ms, 50),
        "ttft_p99_ms": _percentile(ttft_ms, 99),
        "decode_latency_mean_ms": _mean_ms("pure_decode"),
        "prefill_step_time_mean_ms": _mean_ms("pure_prefill"),
        "mixed_step_time_mean_ms": _mean_ms("mixed"),
        "num_pure_prefill_steps": driver.step_kind_count["pure_prefill"],
        "num_pure_decode_steps": driver.step_kind_count["pure_decode"],
        "num_mixed_steps": driver.step_kind_count["mixed"],
        "pct_wall_time_prefill_or_mixed": (prefill_involved_time / total_step_time * 100)
        if total_step_time else None,
        "num_steps": driver.step_count,
        "gpu_sm_util_avg_pct": sm_avg,
        "prefix_cache_token_hit_rate": (total_matched / total_prompt_tokens) if total_prompt_tokens else None,
        "prefix_cache_requests_with_any_match": f"{num_requests_with_any_match}/{len(turns)}",
    }

    summary_path = f"{args.output_prefix}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"Wrote summary to {summary_path}:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(
        f"\nFor a comparable synthetic run: python scripts/benchmark_load.py "
        f"--rates {summary['realized_aggregate_rate_req_s']:.3f} "
        f"--max-num-batched-tokens {args.max_num_batched_tokens} "
        f"--max-num-seqs {args.max_num_seqs}"
        + (f" --min-chunk-size {args.min_chunk_size}" if args.min_chunk_size else "")
    )


if __name__ == "__main__":
    main()
