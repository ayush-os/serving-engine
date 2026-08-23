# Continuous-Batching, Paged-KV LLM Serving Engine

A from-scratch inference server for Llama-3.1-8B-Instruct — hand-built continuous-batching scheduler, paged KV-cache block manager, and a block-sparse Triton paged-attention kernel, GPU-verified and benchmarked end-to-end on real A100 hardware.

**PyTorch · Triton · HuggingFace Transformers · Llama-3.1-8B-Instruct (bf16) · A100 80GB**

- **+89–94% throughput, -43 to -48% TTFT at saturation** — hand-written block-sparse Triton paged-attention kernel (GQA head-mapping, variable-length batching, block-table gather) vs. a dense eager-attention baseline, under a real synthetic-load sweep on real hardware.
- **1.53x wall-clock speedup** from hash-based prefix caching (chained block hashing, first-writer-wins registration) on a shared-system-prompt workload — no cross-request cache staleness, correctness GPU-verified.
- **Predicted the real bottleneck before writing a line of the base engine**: prefill's fixed compute ceiling, not the KV pool or decode capacity. Confirmed on real hardware — prefill-involving steps are ~8% of step count but ~33–35% of wall-clock time at saturation.

## Base engine

**Scheduler** (`scheduler.py`) — Orca-style iteration-level admission: every engine step mixes prefill and decode work for *different* requests in one batch, not a per-request or per-batch loop. Last-resort LIFO preemption under memory pressure, not eager eviction — an eager first attempt was caught by its own regression test before it ever reached GPU, since it preempted requests the pool would've freed naturally the very next step.

**Block manager** (`block_manager.py`) — paged KV cache, 16-token blocks, free-list allocator with reference counting for shared blocks. The design mirrors PagedAttention: block ids, not raw pointers, so a block table stays meaningful even across process boundaries.

**Validated against a real prediction, not just built and hoped**: before running a single load test, the prediction was that prefill's fixed compute cost — not the KV cache pool or decode's per-step cost — would dominate at saturation for a prefill-heavy workload. Real hardware confirmed it, more precisely than "prefill wins":
- Throughput plateaus hard from offered rate=4 through rate=32 (~2.0–2.17 req/s) — 8x more offered load, ~0% more completed. TTFT grows unboundedly past that point: ~150ms → 143 *seconds* at rate=32.
- Directly measured, not inferred: prefill-involving steps are only ~8% of step count at saturation but consume ~33–35% of wall-clock time, since each one costs ~7x a decode step.
- Real bugs surfaced only under real concurrent load, invisible to unit tests — e.g. eager attention's dense score matrix scales with the *whole batch's composition*, not a fixed size, so a fixed startup memory reservation had no way to bound it; a token-budget admission check applied even with nothing else batched yet, so a single oversized request could starve forever (fixed by always admitting a lone candidate, regardless of its own cost).

## Extensions

### Chunked prefill (Sarathi-Serve style)

**The bug it targets**: the scheduler's prefill path admitted a request's *entire* prompt in one iteration — real head-of-line blocking, where a big prefill stalls every decode request already in flight for that step's whole duration.

**The fix**: split a large prefill across multiple scheduling iterations, interleaved with decode the same way prefill/decode already mix per step.

**Predicted this would flatten the TTFT-vs-load curve — it didn't**, and the real effect is narrower and more interesting: at saturation, TTFT and throughput are statistically unchanged from the unchunked baseline (TTFT there is dominated by admission-queue delay, a resource chunking never touches — confirmed by `prefill_step_time_mean_ms` still dropping ~4-5x per chunk, proving the mechanism works, just isn't the bottleneck). The real win shows up pre-saturation: `ttft_p99_ms` improved 14-20% by removing a genuine all-or-nothing admission bug (a request whose arrival landed in a token-budget-exhausted iteration got skipped *entirely*, not partially served) — a tail-latency fix, not a throughput one. Decode latency got consistently, slightly worse (~3% at rate=32) — the interleaving tradeoff expected going in, not free.

### Prefix caching

Hash-based block matching (`BlockManager.match_prefix`/`register_computed_blocks`): a request's prompt is matched against already-computed blocks from other *concurrently alive* requests, sharing physical KV blocks instead of recomputing identical prefill work.

Hashing is **chained**, not content-only: `hash_i = hash(hash_{i-1}, block_i_tokens)`. A block's real KV values depend on everything before it via causal self-attention — two different histories that happen to produce identical *trailing* block content would collide under content-only hashing, silently serving wrong cached values. Chaining makes that structurally impossible.

**Result**: 16 synthetic requests sharing a 512-token system-prompt-style prefix, added one at a time so later requests see an already-registered prefix. Request 0 correctly matches nothing; requests 1–15 all match 512/528 tokens at admission — 15/15 real cache hits under actual GPU execution. Wall-clock: **2.42s (shared) vs. 3.70s (unshared control, identical request shapes) — a 1.53x speedup**, purely from skipping redundant prefill compute.

### Paged attention kernel

Extends a hand-written FlashAttention-2 Triton kernel to read directly from non-contiguous cache blocks instead of gathering into a dense buffer first. One kernel handles prefill and decode in the same launch: grid dim 1 indexes per-request rows with runtime `n_queries`/`n_keys` arrays (a decode row is just `n_queries=1`), GQA head-grouping maps query heads to the smaller KV-head cache, and every ragged edge tile (every decode row against a fixed `Q_TILE_SIZE`) is boundary-checked rather than assumed aligned.

**Predicted** that eager attention's dense `[total_query, total_read]` score matrix — materialized across the whole scheduled batch, the same mechanism behind the base engine's own real OOM bugs — forces a defensive cap on concurrency (`max_num_seqs`), and that a tiled kernel with a small, predictable memory footprint should let that cap rise safely, directly cutting TTFT at saturation.

**Real result — right direction, more specific mechanism than predicted:**

| config | throughput plateau (req/s) | TTFT @ rate=16 | TTFT @ rate=32 |
|---|---|---|---|
| eager attention (baseline) | ~2.0–2.17 | 61.9s | 143.3s |
| paged Triton kernel | ~2.79–2.82 | 32.4s | 80.1s |
| + raised `max_num_seqs` (16→32) | ~2.9–3.0 | 26.3s | 70.0s |
| + raised `max_num_batched_tokens` (1024→2048) | ~3.9–4.1 | 32.3s | 81.9s |

The kernel swap alone — at *unchanged* concurrency settings — was the dominant effect (prefill/mixed step time roughly halved: eager wastes compute on cross-request query-key pairs it immediately masks to `-inf`; the paged kernel's per-row block-table gather never computes them at all). Raising `max_num_seqs` gave a smaller secondary gain than predicted. The genuinely unpredicted finding: `max_num_batched_tokens`, not `max_num_seqs`, was the real remaining bottleneck — raising it moved the ceiling further but traded TTFT for throughput (a bigger per-step token budget means more compute per step), an ordinary batching latency/throughput tradeoff, reported as measured rather than smoothed into a clean win. **Net vs. the original eager baseline: throughput +89-94%, TTFT at saturation -43 to -48%.**

## Correctness

Every phase is checked token-for-token against a reference path on identical prompts (greedy decoding) — plain HF `.generate()` for the base engine, then each new mechanism against the engine's own prior-verified path (chunked vs. one-shot, cached vs. uncached, triton kernel vs. eager attention) to isolate what changed. Real divergences get root-caused with a raw-logit diagnostic at the first diverging token, not pattern-matched from the text diff — see `scripts/diagnose_*.py`.

## Running it

```bash
pip install -r requirements.txt
huggingface-cli login   # Llama-3.1-8B-Instruct is gated

pytest                          # unit tests, pure Python, no GPU needed
pytest -k correctness           # GPU correctness checkpoints (needs CUDA)
python scripts/benchmark_load.py --help
```

`requirements.txt` pins `torch>=2.5`/`transformers>=4.43` (Llama 3.1's `rope_scaling` format needs both). On a fresh CUDA box, check `nvidia-smi`'s CUDA-version header before `pip install torch` — a default PyPI wheel can grab a build newer than the driver supports; the `cu121` index (`--index-url https://download.pytorch.org/whl/cu121`) is the safe default for drivers under 12.4.
