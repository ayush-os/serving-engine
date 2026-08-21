# Handoff — serving-engine, Phase 1+2+2.5 complete, starting Phase 4 (paged-attention kernel)

Written so a fresh chat session (or future you) can pick up exactly where
this one left off, without re-deriving anything already settled. Read
`spec.md` first for the project's overall shape (phases, decisions,
scope) — this doc covers everything *since* Phase 0 that spec.md doesn't
capture: real implementation decisions, bugs found and fixed, real
benchmark data, and what's actually left before Phase 4 is checkpointed.

**Repo:** https://github.com/ayush-os/serving-engine (public), 60+ commits on
`main` as of this doc — check `git log` for the current count, it'll be
stale immediately.

## Status in one line

**Phase 1, Phase 2, and Phase 2.5 are all fully checkpointed.** Phase 1:
`test_correctness.py` passes (2 exact matches, 1 `xfail` for expected bf16
drift), `scripts/continuous_batching_demo.py` shows a 1.71x speedup with
sustained GPU utilization (see "First GPU session"), `preempt()`/eviction
GPU-verified (see "Second GPU session"). Phase 2: a real load sweep ran
end-to-end on real hardware (`scripts/benchmark_load.py`), found and fixed
three real bugs along the way, and produced a genuine predicted-vs-real
read against `disagg_and_placement_notes.md`'s Finding 3 — see "Third GPU
session" below for the full story, findings, and what does/doesn't
generalize from them. Phase 2.5 (chunked prefill): implemented, GPU-
correctness-verified, and re-benchmarked against Phase 2's own config —
the real finding is narrower and more interesting than the original
hypothesis ("chunking flattens the TTFT-vs-load curve" was wrong; the real
effect is a tail-latency fix pre-saturation, not a saturation-ceiling fix)
— see "Fourth GPU session" below for the full story and exactly why.
**Phase 4 (real paged-attention kernel) is next**, directly motivated by
that finding: Phase 2.5 pinned the saturation-ceiling bottleneck down to
admission-queue depth (`max_num_seqs`, a defensive cap against the eager
attention op's unbounded memory scaling), which chunking structurally
can't touch but a tiled kernel plausibly can. See "Roadmap after Phase 2"
and "Immediate next steps — Phase 4" at the bottom.

## First GPU session — model swap, real bugs found, correctness result

Rented an A100 (~40GB variant). Three real things happened, in order:

1. **Model swapped to `meta-llama/Llama-3.1-8B-Instruct`.** The
   `Meta-Llama-3-8B-Instruct` gated-repo access request was still pending
   at session start; 3.1 was already approved from prior work and is
   architecturally identical everywhere this code touches it (same
   `LlamaForCausalLM` class, same `num_hidden_layers`/`num_key_value_heads`/
   `head_dim`). spec.md's Decision 1 explicitly allows "or similar 7-8B
   dense open-weight model," so this is in-scope, not a deviation.
   `model_runner.py`'s `MODEL_NAME` now points at 3.1 — the mentions of
   "Llama-3-8B" elsewhere in this doc predate the swap and are still
   accurate in spirit (same architecture), just not the literal string.

2. **Two real bugs, both found and fixed by reading the actual installed
   `transformers==5.15.1` source directly** (not by re-guessing from this
   doc's earlier research — that research turned out to be incomplete in
   one place, see below):

   - **Wrong cache wiring.** `model_runner.forward()` was passing
     `past_key_values=cache`. That's wrong: `LlamaAttention.forward()`
     calls `past_key_values.update(key_states, value_states, layer_idx)`
     *unconditionally* whenever `past_key_values is not None` — that's the
     generic 3-arg `Cache` interface (works for `DynamicCache`), and
     `_PagedKVCache` doesn't implement it (`TypeError: missing read_index,
     write_index`). Reading `transformers/integrations/eager_paged.py`
     directly showed the real contract: the paged attention function pops
     a **separate** `cache` kwarg out of `**kwargs` and calls
     `.update(key_states=, value_states=, layer_idx=, read_index=,
     write_index=)` there — `past_key_values` is never touched on the
     paged path. Fix: pass `past_key_values=None`, `cache=cache` as a
     plain kwarg (flows through `**kwargs` to the attention function), and
     `use_cache=False` (so `LlamaModel.forward()` doesn't auto-construct
     an empty `DynamicCache` and put it back into `past_key_values` — this
     turned out to be harmless either way since `create_causal_mask`
     returns an already-4D mask as-is regardless, but `False` avoids
     relying on that being a no-op). `_PagedKVCache.update()`'s own
     signature and the flat, non-grouped `read_idxes`/`write_idxes`
     construction needed no changes — Llama has one layer group, so the
     per-group indirection the real `PagedAttentionCache.update()` does
     internally doesn't apply here.
   - **Autograd never disabled.** `forward()` had no `torch.no_grad()` /
     `inference_mode()` guard. `model.eval()` only disables
     dropout/batchnorm, not gradient tracking — with the model's weights
     requiring grad, the in-place `index_copy_` writes in
     `_PagedKVCache.update()` collided with the live autograd graph
     (`RuntimeError: a leaf Variable that requires grad is being used in
     an in-place operation`). Fixed with `@torch.inference_mode()` on
     `ModelRunner.forward()`.

   Both required extensive live source inspection (`inspect.getsource()`
   on `eager_paged_attention_forward`, `PagedAttentionCache.update()`,
   `LlamaAttention.forward()`, `LlamaModel.forward()`) to get right —
   guessing from this doc's Phase-0-era research would have missed both.

3. **Correctness result: 2/3 prompts match HF `.generate()` exactly
   token-for-token; the third (`"def fibonacci(n):"`) diverges partway
   through decode.** Diagnosed with a standalone script comparing raw
   logits for just the first generated token (bypassing the decode loop
   entirely): engine and HF agree on argmax and have near-identical top-5,
   max abs diff 0.156 on logits of magnitude ~19 — squarely bf16 rounding
   noise, not a logic error. This confirms prefill, the mask, and the
   paged cache read/write are all correct. The actual divergence happens
   several decode steps later: ordinary bf16 greedy-decoding
   non-associativity (different reduction order between this engine's
   batched/masked attention and HF's own single-sequence path) compounding
   until a near-tied token flips — the same kind of divergence you'd see
   comparing two different HF attention implementations against each
   other, not specific to this engine. Explicit decision made (not
   unilateral): treat this as the expected correctness bar rather than
   chase it further. `test_correctness.py` marks that one case `xfail`
   with the diagnostic reasoning inline; the other two remain hard
   assertions.

## Repo map (current as of this doc)

```
serving_engine/
  block.py           BLOCK_SIZE=16, Block dataclass — done, no known issues.
  request.py         Request/RequestPhase/RequestStatus — done.
                      num_computed_tokens now real and load-bearing (Phase
                      2.5): incremented per chunk in update_after_step(),
                      reset to 0 on preempt() as before.
  block_manager.py   allocate/append_slot/free/fork/preempt all done, GPU-
                      verified. append_slot's CoW branch still stubbed
                      (deliberately, see "Deliberately deferred" below).
                      Unchanged by Phase 2.5 -- allocate()'s existing
                      ceil(total_len/block_size) upfront reservation already
                      covers a request's full prefill regardless of
                      chunking, confirmed in "Fourth GPU session" below.
  scheduler.py        schedule()/update_after_step() done, GPU-verified.
                      Last-resort LIFO eviction + max_num_batched_tokens/
                      max_num_seqs cap both implemented, enforced, and
                      GPU-verified under real load (Third GPU session --
                      includes a real starvation-bug fix, see below). Phase
                      2.5: chunked prefill now lives here too -- min_
                      chunk_size floor, budget-filling chunk cost,
                      SchedulerOutput.prefill_chunk_sizes/prefill_final_
                      chunk. See "Fourth GPU session" below.
  model_runner.py     ModelRunner + _PagedKVCache + forward() — fully
                      written, GPU-verified (correctness + preemption
                      recompute both confirmed). Known real limitation,
                      Phase 4's actual target: eager (non-tiled)
                      attention's transient memory scales with batch
                      composition -- see "Third GPU session" and "Immediate
                      next steps -- Phase 4" below. Phase 2.5: prefill path
                      now slices to the scheduled chunk instead of the
                      whole prompt (still no logits_to_keep optimization --
                      see "Deliberately deferred" below, unchanged by this).
  engine.py           LLMEngine wiring — done, GPU-verified end-to-end
                      (generate(), step(), num_gpu_blocks=None auto-sizing,
                      max_num_batched_tokens/max_num_seqs/min_chunk_size
                      pass-through). step() early-returns if
                      scheduled_requests comes back empty (defensive guard
                      added Third GPU session). add_request() also accepts
                      prompt_token_ids/ignore_eos for benchmarking (bypasses
                      the tokenizer/EOS for synthetic load). Phase 2.5:
                      step() now only samples/appends a token once a
                      request's chunk actually reaches prefill_final_chunk,
                      not unconditionally every scheduled step.
  workload.py         Synthetic Poisson/lognormal request generator for
                      Phase 2 -- pure Python, unit-tested, no GPU needed.
scripts/
  continuous_batching_demo.py    Phase 1's 2nd checkpoint: staggered
                      arrivals + naive-vs-continuous speedup comparison.
                      Run directly: python scripts/continuous_batching_demo.py
  preemption_sanity_check.py     GPU sanity check for preempt()/eviction --
                      forces a real eviction deterministically, asserts it
                      fired, checks outputs are coherent post-recompute.
  benchmark_load.py   Phase 2's measurement harness: sweeps a synthetic
                      Poisson workload across offered rates, measures
                      throughput/TTFT/decode-latency/prefill-step-time/GPU
                      occupancy, writes a CSV. See "Third GPU session" for
                      how this originally ran, and "Fourth GPU session" for
                      the Phase 2.5 rerun. --min-chunk-size flag added for
                      Phase 2.5 (default None, no floor).
tests/
  test_block_manager.py   11 tests, pure Python, no torch/GPU — passing.
  test_scheduler.py        19 tests, pure Python, no torch/GPU — passing.
                      Includes a regression test for the Third-GPU-session
                      starvation bug.
  test_correctness.py       Phase 1's checkpoint (token-for-token vs HF
                      .generate()) — GPU-verified, 2 pass exactly, 1 xfail
                      (confirmed bf16 drift, not a bug -- see below). Phase
                      2.5's checkpoint also lives here now:
                      test_chunked_prefill_matches_one_shot, GPU-verified,
                      1 pass exactly + 2 xfail (confirmed same bf16-noise
                      signature as the Phase 1 xfail -- see "Fourth GPU
                      session" below for the diagnostic).
  test_workload.py          9 tests, pure Python -- Poisson/lognormal
                      distribution sanity, length-cap clamping.
benchmark_results.csv, benchmark_results_80gb.csv, benchmark_results_final.csv,
benchmark_results_chunked.csv
                      Real sweep output, committed to the repo root (not
                      results/ -- kept flat, matches how the rest of this
                      repo doesn't nest single-purpose output files).
                      _final has the prefill/decode/mixed step-time split;
                      the other two predate that instrumentation.
                      _chunked is Phase 2.5's rerun, same config as _final
                      (max_num_batched_tokens=1024/max_num_seqs=16/
                      max_prompt_len=2048/max_output_len=512) with chunking
                      active, min_chunk_size unset -- the direct before/
                      after pair. See "Third"/"Fourth GPU session" for what
                      each actually found.
.venv/                 local venv for the pure-Python test files only
                      (no torch). NOT what runs on the GPU box.
```

On the GPU box: always use an isolated `.venv-gpu` virtualenv (`python -m
venv .venv-gpu`), never the shared system Python — see "Second GPU session"
below for why (ABI conflicts with pre-loaded packages on a fresh box).

## The real story of `model_runner.forward()` — why it looks the way it does

This is the part most worth understanding before touching it again, since
it went through a real pivot mid-build. (Its real memory-scaling behavior
under load is a separate, later story — see "Third GPU session" below and
Phase 4 in spec.md; this section is about correctness/mechanism, not cost.)

**Original plan (abandoned):** gather each request's cached K/V from its
`block_table` into a dense per-request buffer, wrap it in a `DynamicCache`,
call the model normally. Turned out `DynamicCache` has no method to
pre-populate itself from external tensors (`.append()` doesn't exist),
and forcing a paged/scattered cache through an API designed for one
contiguous, incrementally-growing cache per request was fighting the tool.

**What we actually found by reading `transformers` 5.15.1 source directly**
(not from docs — docs don't cover this, which is *why* it was hard to find):
`transformers` ships a full native continuous-batching stack at
`transformers.generation.continuous_batching` (scheduler, cache, model_runner
— structurally the same architecture as this repo, independently converged
on). Its `PagedAttentionCache` is **not** a `Cache`/`DynamicCache` subclass.
Gather/scatter happens **inside the attention op**, via a registered
`attn_implementation` (`"paged|eager"`, already globally registered in
`ALL_ATTENTION_FUNCTIONS`), which reads a `cache` kwarg and calls
`cache.update(key_states, value_states, layer_idx, read_index, write_index)`.
Critically, `LlamaAttention.forward()` calls `past_key_values.update(...)`
*unconditionally whenever `past_key_values is not None`*, before the
attention interface even runs — so the cache object's `.update()` contract
is the real integration point, not a custom attention function.

**What got built instead:**
- `_PagedKVCache` (top of `model_runner.py`) — a small duck-typed class
  (not a `Cache` subclass) implementing exactly that `.update()` contract:
  reshape, `index_copy_` (write) into a flattened per-layer view of
  `self.kv_cache`, `index_select` (read) back out. Directly transcribed
  from `PagedAttentionCache`'s own full-attention branch, stripped of all
  the TP/sliding-window/attention-sink generality Llama-3-8B doesn't need.
- `self.model.set_attn_implementation("paged|eager")` — the real,
  version-checked public API for registering it (not raw config mutation).
- **One combined model call** mixing prefill and decode tokens, not two
  separate calls. The original two-call split existed to avoid padding
  decode's 1-token input up to a big prefill's length — but that padding
  problem doesn't exist under this index-based gather/scatter approach,
  so the split was dropped.

**The index math** (`write_idxes`, `read_idxes`, `position_ids`,
`attention_mask`) — this is the part that actually has real content, and
where the real bugs were:
- `write_idxes`: one flat physical slot per *new* token this step.
  Prefill contributes `prompt_len` entries (writes everything fresh);
  decode contributes exactly 1 (`total_len - 1` — already-appended token
  from the *previous* step, since sampling+append happens in `engine.step()`
  *after* `forward()` returns).
- `read_idxes`: one flat physical slot per position each token's request
  needs to attend against. Longer than `write_idxes` — decode needs its
  *entire* history (`range(total_len)`), prefill needs its own prompt's
  positions too (`range(prompt_len)`, not empty — a common wrong instinct;
  the "empty read_index" shortcut real paged implementations use only
  applies when the *whole batch* is pure prefill with nothing to read
  anywhere, which isn't this engine's case since prefill/decode are
  deliberately mixed). **This is the field chunked prefill will need to
  change**: today prefill always contributes its *entire* prompt as one
  block of write/read positions in a single call (`req.all_token_ids()`,
  `range(req.total_len)`) — see "Immediate next steps."
- `attention_mask`: `[1,1,total_query,total_read]`, built from two
  broadcast comparisons over per-token (request-group, logical-position)
  pairs — same request AND key position ≤ query position. Uses
  `torch.finfo(dtype).min`, not literal `-inf` (avoids NaN-from-fully-masked
  softmax rows, matching HF's own convention). **This is one dense matrix
  over the whole scheduled batch, not tiled** — the real mechanism behind
  the Third GPU session's OOMs, see below.
- `position_ids`: each token's position *within its own request*, not its
  index in the flattened batch — required because RoPE's relative-distance
  property only holds if positions are each token's true sequence position;
  no single global `torch.arange` works once multiple requests are mixed.

Real bugs caught and fixed along the way (all in commit history, worth
skimming if something looks subtly wrong on the GPU): decode's write-index
using only `output_token_ids` length instead of `total_len` (ignored the
prompt entirely); an off-by-one in the block-table index derived from
`logical_pos // block_size`; using a block-table index as if it *were* a
physical block id; decode's read-index off-by-one (`total_len + 1` instead
of `total_len`, reading a position that was never written).

## Second GPU session — preempt(), the scheduler cap, a real bug, GPU-verified

Both of Phase 1-era's "deferred until Phase 2" items are now done: `preempt()`
(recompute-based, LIFO eviction, last-resort trigger only — see
`scheduler.py`'s `schedule()`) and the `max_num_batched_tokens`/`max_num_seqs`
cap. Both have full pure-Python test coverage (29 tests across
`test_block_manager.py`/`test_scheduler.py`) and are now GPU-verified too via
`scripts/preemption_sanity_check.py`.

**Eviction policy, explicitly decided, not the obvious default**: last-resort,
not eager. The first attempt evicted immediately whenever any single
candidate couldn't get a block — caught by the pre-existing
`test_schedule_skips_blocked_decode_without_starving_other_candidates` before
it ever landed, since it would preempt a request even when the pool would've
freed up naturally next step. That's not just wasteful, it directly
contaminates Phase 2's actual measurement (real pool scarcity vs. manufactured
preemption noise). Fixed to: one normal scheduling pass, then evict-and-retry
only if `output.scheduled_requests` ends up completely empty — a true stall,
not just one unlucky candidate.

**A real, previously-undiscovered bug, found by the sanity script and NOT
specific to preemption**: `can_append_slot`/`append_slot` decided whether to
grow a request's block table using `total_len % block_size == 0` — a
shortcut that happens to work for normal incremental single-token decode
growth (it fires one step early, proactively), but silently assumes the
table's current capacity followed that exact incremental history.
`allocate()` doesn't follow that history — it computes
`ceil(total_len / block_size)` directly, no headroom. Whenever `total_len`
at admission time is an *exact* multiple of `block_size`, the next decode
step needs a block that was never provisioned → `IndexError` in
`ModelRunner._flat_slot`. This isn't preemption-specific — it would hit any
fresh admission whose prompt length happens to land exactly on a block
boundary too; three pre-existing tests had `prompt_len == block_size` baked
in and were asserting the under-sized table as correct, just never having
gone as far as an actual `forward()` call to surface it. Fixed by checking
the real invariant directly (`len(block_table) * block_size >= total_len`)
instead of the modular-arithmetic shortcut.

**GPU confirmation, and it's a stronger signal than "didn't crash"**: the
sanity script runs 3 requests with an *identical* prompt (so greedy decoding
makes them decode in lockstep and all three outputs should be identical).
Only one went through eviction + recompute; the other two decoded
uninterrupted. All three final outputs came back byte-for-byte identical —
strong evidence the recompute reconstructed the KV cache correctly, not just
that it avoided crashing.

**Also found, unrelated to any of the above (environment, not code)**: a
fresh Paperspace instance resolved `transformers==4.x`/`torch==2.1.1` by
default, too old for Llama 3.1's `rope_scaling` format and for each other.
`requirements.txt` now pins `torch>=2.5`/`transformers>=4.43`. Separately,
installing into that box's shared system Python (pre-loaded with
`torchaudio`/`torchvision` built against the old `torch`) broke on ABI
mismatch after upgrading `torch` in place — always use an isolated
`.venv-gpu` virtualenv, never the system Python, on a fresh box.

## Third GPU session — Phase 2 load sweep: three real bugs, then real findings

This was a long, iterative session: build the harness, hit a real bug on
the GPU, fix it, hit the *next* real bug, fix it, and so on, until a clean
sweep finally ran. Every one of these was a genuine bug caught by real
load — none of it was visible from pure-Python unit tests or from Phase 1's
light-load demos, which is exactly why Phase 2's actual load sweep was
worth building rather than declaring Phase 1's demo "good enough" evidence
of a working scheduler under real concurrency.

### Bug 1 — eager attention has no memory ceiling of its own

First sweep OOM'd at rate=1.0 req/s (fine at 0.5). `model_runner.forward()`'s
attention op (`eager_paged_attention_forward`) builds **one dense
`[total_query, total_read]` score matrix across the whole scheduled batch**
— not tiled like FlashAttention. `ModelRunner._infer_num_gpu_blocks`
reserves activation headroom as a fixed 10% of free-memory-after-weights,
computed once at startup, with no idea how wide a real batch gets. More
concurrent requests at rate=1.0 than at rate=0.5 pushed the score matrix
past that fixed headroom.

Fix: `Scheduler` already had `max_num_batched_tokens`/`max_num_seqs` caps
(built and unit-tested in the second GPU session) but `LLMEngine` never
passed them through — nothing in Phase 1 ran a batch wide enough to need
them. Wired them into `LLMEngine.__init__` (commit `d3cde62`).

### Bug 2 — a candidate over the token cap alone could starve forever

Wiring in the caps immediately exposed a real, previously-latent
`scheduler.py` bug: `schedule()`'s token-budget check
(`num_batched_tokens + cost > max_num_batched_tokens`) applied even at
`num_batched_tokens == 0`, so a single waiting prefill whose own cost
exceeded the cap could **never** be scheduled — not now, not ever, no
matter how idle the system was. It's a `NEEDS_PREFILL` candidate, so it
never reaches `blocked_decode_candidates` either, meaning nothing could
unstick it: permanent starvation, silently returning an empty
`scheduled_requests` forever. That empty batch is what actually crashed
things downstream: `torch.tensor([])` on the empty token list defaults to
float32, and `embed_tokens` wanted Long/Int.

Fix, decided explicitly (not the obvious "just raise the cap" workaround):
the token-budget check now only limits piling more work on top of
something *already* batched this step — a lone candidate always gets
admitted regardless of its own cost (commit `2ab9de5`). This is the
standard convention absent chunked prefill (which doesn't exist yet — see
"Immediate next steps"). An existing test that used a zero-budget lone
candidate as a test device for "cap-blocked, not resource-stalled" had to
be rewritten with two candidates, since a lone one is now always admitted.
Also added a defensive early-return in `engine.step()` for any other
empty-batch case (e.g. a block-capacity stall with no eviction victim
available) — `forward()` has no real batch to run in that case either.

### Bug 3 — width-bounded caps weren't depth-bounded

Caps in place, sweep OOM'd again at rate=1.0. Root cause: `max_num_seqs`
bounds how many requests run concurrently, but each one still contributes
its **entire** context length to the attention read side — nothing bounded
how long any single admitted request's context could grow. ~15 concurrent
long-tail decode contexts plus one more admitted prefill (all within
`max_num_seqs=16`/`max_num_batched_tokens=2048`) pushed the score matrix to
~5GB against a ~2.5GB headroom.

Fix: clamp the workload generator's lognormal prompt/output tail directly
(`workload.py`'s `max_prompt_len`/`max_output_len`, commit `3cf43be`) —
this turned out to be the historically-correct fix too, not just
expedient: it matches `disagg_and_placement_notes.md` §3's own "hard stop,
not compaction" admission policy. Combined with tighter scheduler defaults
(1024 tokens / 8 seqs), worst-case score matrix came out to ~650MB.

### Environment note — CUDA driver mismatch on a fresh box (fix now confirmed)

A later fresh Paperspace box (different from the ones above) hit
`RuntimeError: The NVIDIA driver on your system is too old (found version
12020)` — `pip install torch` grabbed a CUDA build newer than that box's
driver supported. Proposed fix, unconfirmed at the time: `pip uninstall -y
torch torchvision torchaudio && pip install "torch>=2.5" --index-url
https://download.pytorch.org/whl/cu121`. **Now confirmed working**
(Fourth GPU session, a different 80GB Paperspace box, driver 535.129.03 /
CUDA 12.2 per `nvidia-smi` — a default PyPI torch wheel would likely have
grabbed a cu124 build needing driver ≥550, cu121 needs only ≥530). cu121
is the safer default across arbitrary rented boxes generally (drivers are
backward-compatible with older CUDA runtime builds, never forward-
compatible) — check `nvidia-smi`'s "CUDA Version" header before any `pip
install torch` on a new box, and prefer cu121 directly over trying the
default first when that header reads anything under ~12.4.

### The OOMs were actually the 40GB A100, not a scaling flaw

Both real OOMs above happened on a 40GB A100. Switching to an 80GB card
cleared the *same* workload without needing the tight caps — and doubling
every cap back up (`max_num_seqs` 8→16, `max_prompt_len` 1024→2048,
`max_output_len` 256→512) barely moved the measured throughput ceiling at
all (~2.1 req/s either way, SM util within ~1-2pp). That's a real, useful
confirmation: the ceiling found below isn't a scheduler-cap artifact, it's
a genuine hardware/compute ceiling.

### The actual Phase 2 findings — real hardware vs. `disagg_and_placement_notes.md`'s Finding 3

Final config: `max_num_seqs=16`, `max_num_batched_tokens=1024`,
`max_prompt_len=2048`, `max_output_len=512`. Results in
`benchmark_results_final.csv` (has the prefill/decode/mixed step-time
split; `benchmark_results.csv`/`benchmark_results_80gb.csv` are earlier
sweeps without that instrumentation, kept for the cap-sensitivity
comparison above).

- **Throughput plateaus hard**: flat ~2.0–2.17 req/s from offered
  rate=4 through rate=32 — 8x more offered load, ~0% more completed.
- **TTFT grows unboundedly** past that same point: ~150ms → 143 *seconds*
  at rate=32.
- **Decode stays cheap and nearly flat**: 24→30ms, ~25% growth across a
  64x load increase — never the bottleneck, in this configuration (see
  "what generalizes" below — this is a conditional finding, not universal).
- **Directly measured, not inferred** (added specifically to nail this
  down, not just argue it from decode staying cheap): prefill-involving
  steps are only ~8% of step count at saturation but consume ~33–35% of
  wall-clock time (`pct_wall_time_prefill_or_mixed`), because each one
  costs ~7x a decode step. `prefill_step_time_mean_ms` itself plateaus
  (~215–227ms) once `max_num_batched_tokens` saturates — a literal
  fixed-cost-per-max-batch ceiling.
- **SM utilization saturates ~89–90%, never 100%** — a real ~10-11% gap,
  possibly genuine per-step CPU-side overhead: `model_runner.forward()`
  builds `write_idxes`/`read_idxes`/`position_ids`/the attention mask as
  synchronous Python-list work before every single GPU call. Not confirmed
  as the cause, just a plausible, honest candidate — the kind of thing
  spec.md's own Phase 2 checkpoint asked to watch for ("kernel launch
  overhead, scheduling overhead").
- **Net**: real hardware confirms the simulator's qualitative *shape* —
  a fixed compute ceiling, decode not the bottleneck in this configuration,
  unbounded queueing delay past the ceiling. Absolute numbers aren't
  comparable (1 A100/8B/bf16 here vs. a 29-machine TPU-8i/70B/FP4 pool in
  the simulator) — no valid unit conversion between them, only the
  mechanism transfers.

### What generalizes from this, and what doesn't — worth getting right in any future writeup

Explicitly worked through this with the user; don't flatten it back into
"decode is cheap" as a universal claim in any future report.

**Fully general (queueing theory, not LLM-specific)**: throughput
plateaus and wait time grows unboundedly once offered load exceeds
whatever the system's real service capacity is — true of any finite-
service-rate system, regardless of what's actually saturating.

**General as a mechanism, now confirmed with real hardware (not just the
simulator)**:
- Once a system is past its own compute-bound crossover, more
  concurrency/memory headroom stops buying throughput — this session's own
  cap-doubling experiment (8→16 seqs, depth 2x, ceiling barely moved) is
  real-hardware confirmation of the same asymptote
  `disagg_and_placement_notes.md` hit independently four times
  analytically.
- A hard per-iteration admission budget needs an explicit "always let the
  lone candidate through" guarantee or it can starve indefinitely (Bug 2
  above) — general scheduling-theory point, not LLM-specific.
- Untiled/eager attention has batch-composition-dependent transient
  memory, which breaks any fixed-fraction memory reservation sized at
  startup (Bugs 1 and 3) — generalizes to any engine built this way,
  explicitly does NOT hold for real production engines using
  FlashAttention-style tiling, which is exactly the gap between this
  implementation and what `disagg_and_placement_notes.md` §3.7 assumed.
- A resource can dominate wall-clock time while being a small minority by
  operation count, if its per-operation cost is disproportionate (the
  8%-of-steps/33%-of-time finding) — general cost-share-vs-frequency
  reasoning; the specific numbers are this run's, the principle isn't.

**The meta-finding (the generalizable version of "decode isn't the
bottleneck")**: there is no universal prefill-vs-decode bottleneck
ordering. Which one dominates is determined by workload shape
(input/output length ratio — this run used DistServe's prefill-heavy
512/64 average) and how much concurrency the system actually permits
(`max_num_seqs=16` here, itself an artifact of this engine's untiled
attention, not a fundamental serving constraint). Both are levers, not
constants. A decode-heavy workload (long generations, short prompts) or a
system permitting much higher decode concurrency (e.g. a tiled-attention
engine) could flip which phase dominates. *That conditional structure* is
the portable claim — "prefill dominates" is just where the levers landed
here.

## Fourth GPU session — Phase 2.5 (chunked prefill): implementation, correctness, and the real finding

Implementation itself (`scheduler.py`, `model_runner.py`, `engine.py`,
`scripts/benchmark_load.py`) was built and pure-Python-tested off-GPU
first, same testing discipline as every prior phase — see the "Immediate
next steps" section this replaces, below, for what the design forks were
and how they got resolved (floor-gated + budget-filling chunk cost;
`WAITING`/`RUNNING`-status branching so a continuing chunk doesn't
re-`allocate()` blocks it already has; decode keeps scheduling priority,
forced by the sizing policy itself rather than a separate choice). Three
real things worth remembering if this code gets touched again, none of
them GPU-discovered (caught by static review/pure-Python tests before
ever reaching the GPU box — the testing discipline held up again):

- **`total_len`, not `prompt_len`, is the correct chunking target.**
  `block_manager.preempt()` resets `num_computed_tokens` to 0 but
  `output_token_ids` survives — a post-preemption recompute has to
  reprocess already-generated output tokens too, not just the original
  prompt. `prompt_len` only coincidentally equals `total_len` in the
  never-preempted case.
- **Final-chunk detection has to be decided at schedule time, not
  re-derived later.** `total_len` is a live property (`prompt_len +
  output_len`), and `engine.step()` appends this step's sampled token to
  `output_len` *before* the phase-transition check runs (required so the
  existing EOS-finish check can see it) — a naive post-hoc `>= total_len`
  re-check would silently fail for the exact request that just finished
  prefilling, since its own append inflates the target out from under
  itself. Fixed by deciding "is this final" once in `schedule()` (where
  nothing's been sampled yet this iteration) and storing it in
  `SchedulerOutput.prefill_final_chunk`, not re-derived downstream.
- **A variable-shadowing near-miss in `model_runner.forward()`**: the
  chunk's prompt-offset was briefly named `start`, silently clobbering an
  outer `start = len(toks)` used later for `req_spans`. Harmless only by
  accident (that tuple field is never actually read), caught on a second
  read before it shipped. Renamed to `chunk_start`.

### Correctness: GPU-verified, same treatment as Phase 1's own precedent

Environment note, resolving a previously-unconfirmed flag: this session's
box (80GB A100, driver 535.129.03, CUDA 12.2 per `nvidia-smi`) needed the
cu121 wheel (`pip install "torch>=2.5" --index-url
https://download.pytorch.org/whl/cu121`) — confirmed working this time,
not just proposed. A driver capped at CUDA 12.2 can't run a default
PyPI torch wheel built for cu124 (needs driver ≥550); cu121 wheels need
driver ≥530, comfortably under 535.129.03. Worth checking `nvidia-smi`'s
"CUDA Version" header before any `pip install torch` on a new box, same
as before.

`test_matches_hf_generate` re-ran clean first (2 passed, 1 xfail,
unchanged from Phase 1 — confirms the environment itself, not yet
chunking). The real Phase 2.5 checkpoint is `test_chunked_prefill_matches_
one_shot` (new in `test_correctness.py`): same prompts through a one-shot
engine and a `max_num_batched_tokens=4`-forced-multi-chunk engine,
compared against *each other*, not HF — isolates any divergence to
chunking itself rather than conflating it with the already-understood
bf16-vs-HF drift the fibonacci case has in the other test.

Result: 1 passed (`"The capital of France is"`), 2 failed on first run.
Diagnosed with the same methodology as Phase 1's original fibonacci
diagnostic (`handoff.md`'s "First GPU session") — compare raw logits at
the first diverging output token, chunked vs. one-shot:

| | fibonacci (this session, chunked vs. one-shot) | transformer (this session, chunked vs. one-shot) | fibonacci (Phase 1, one-shot vs. HF, for comparison) |
|---|---|---|---|
| diverges at output token | 2 | 30 (deep into the 32-token window) | several tokens into decode |
| max abs logit diff | 0.156 | 0.203 | 0.156 |
| logit magnitude | ~19.5 | ~24.9 | ~19 |
| argmax | near-tied (422 `' if'` vs 4304 `' """'`) | near-tied (374 `' is'` vs 2209 `' Is'`) | near-tied |
| top-5 | identical set, reordered | identical set, reordered | near-identical |

Same signature as Phase 1's own precedent, in near-identical scale, even
though the perturbation source is genuinely different here (GPU kernel
non-determinism across matmul shapes — chunked and one-shot both call
this engine's *own* attention op, just via differently-sized `forward()`
passes — vs. Phase 1's case, which was two entirely separate attention
implementations, this engine's vs. HF's own). A real logic bug would show
a large logit gap or a non-overlapping top-5; this shows neither. Marked
`xfail` with the diagnostic inline, same treatment as Phase 1's own
precedent (`strict=False`, so a lucky exact-match rerun still passes).
Re-ran clean after: `1 passed, 2 xfailed`.

### The real benchmark rerun — and why the original hypothesis was wrong

Re-ran `scripts/benchmark_load.py` with the *exact* same config as
`benchmark_results_final.csv` (`--max-num-batched-tokens 1024
--max-num-seqs 16 --max-prompt-len 2048 --max-output-len 512`, same
`--rates 0.5,1,2,4,8,16,32`), deliberately with `--min-chunk-size` unset
(no floor) — a genuine single-variable before/after, chunking active with
nothing else changed. Output: `benchmark_results_chunked.csv`.

**The original Phase 2.5 hypothesis — "chunking flattens the TTFT-vs-load
curve" — was wrong**, and it's worth being precise about *why*, not just
noting the miss:

- **At saturation (rate≥4), nothing meaningful moved.** `ttft_mean_ms`
  and `measured_throughput_req_s` are statistically identical to the
  un-chunked baseline at every saturated rate (e.g. 142.2s vs 143.3s TTFT
  and 2.13 vs 2.10 req/s at rate=32) — well within run-to-run noise.
- **The chunking mechanism itself is confirmed working exactly as
  designed at the per-step level**: `prefill_step_time_mean_ms` dropped
  ~4-5x across every rate (221ms→45ms at rate=32, 226ms→48ms at rate=2).
  This rules out "chunking isn't actually chunking" as the explanation
  for the flat curve — it *is* chunking, it just isn't the bottleneck.
- **Why**: at saturation, `ttft_mean_ms` (tens of seconds to minutes)
  dwarfs `prefill_step_time_mean_ms` (tens of ms) by 2-3 orders of
  magnitude — meaning TTFT there is almost entirely *admission queueing
  delay* (time spent in `self.waiting` for `can_allocate()`/block
  capacity to free up), not prefill compute time once a request is
  actually running. Chunking never touches the admission gate — it only
  changes how an *already-admitted* request's compute gets scheduled. At
  saturation it's optimizing a term that's negligible next to the
  dominant one.
- **There is a real win, just not the hypothesized one**: at unsaturated
  rates (0.5, 1.0), `ttft_p99_ms` improved 20% (1355ms→1087ms) and 14%
  (2220ms→1908ms) — but `ttft_p50_ms` barely moved (41.3ms→43.2ms,
  70.0ms→70.2ms). A tail-latency fix, not a median one. Mechanism: under
  one-shot rules, a request whose arrival landed in an iteration where
  decode had already eaten most of the token budget got skipped
  *entirely* that iteration (all-or-nothing admission via the existing
  `num_batched_tokens > 0` gate) and had to wait for a future,
  lower-load iteration. Chunking removes that all-or-nothing behavior —
  it always makes at least partial progress immediately. Most requests
  never hit this unlucky case (median unaffected); the tail does.
- **`decode_latency_mean_ms` got consistently, slightly worse** — not
  better — at every single rate (30.27ms→31.11ms at rate=32, same
  direction throughout). Small, but real and monotonic, not noise.
  Matches the exact tradeoff `spec.md` flagged going in: interleaving
  means a chunk's compute now shares an iteration with decode steps it
  previously didn't.

**Net**: chunked prefill is a fairness/tail-latency fix for the
unsaturated regime, not a throughput or saturation-latency fix — a
genuinely different (and more specific/interesting) result than the
original hypothesis, same "disagreement is a more interesting finding
than agreement" pattern as Phase 2's own prefill-vs-decode result. It
also directly pins down *what the real saturation bottleneck is*:
admission-queue depth, gated by `max_num_seqs` — which exists
specifically as a defensive cap against the eager attention op's
unbounded, batch-composition-dependent memory scaling (see "Third GPU
session"'s Bugs 1 and 3 above). Chunking structurally cannot touch that
cap; a tiled kernel that never materializes the full attention matrix
plausibly *can*, by making memory usage small and predictable regardless
of batch composition — which is the direct, now-data-backed motivation
for Phase 4, not just "kernel work is the natural next thing."

## Deliberately deferred — not bugs, don't "fix" these reflexively

- **`block_manager.py`: `append_slot`'s CoW branch** (marked `# TODO`) —
  needed only once `fork()` is actually exercised by real prefix-sharing,
  which nothing currently triggers (no scheduler-level prefix detection
  exists). No longer "deferred indefinitely" with no plan — it's now a
  real (if low-priority) roadmap candidate, see "Roadmap after Phase 2"
  below. Confirmed via `grep` this session: `fork()` is only ever called
  from `tests/test_block_manager.py`, never from `scheduler.py`/`engine.py`.
- **Starvation avoidance under sustained load** — the last-resort eviction
  trigger (see "Second GPU session" above) only fires when *nothing at all*
  progressed this step. A persistently unlucky candidate could in theory
  wait a long time if something else keeps making unrelated progress every
  step, since the "nothing progressed" trigger won't refire while anything
  else is moving. This is a different mechanism from the Third-GPU-session
  starvation bug (that one was a hard, permanent block; this one is a soft
  fairness gap) — real fairness needs more machinery (e.g. tracking how
  long a candidate's been stuck) than this project's scope calls for. Not
  specifically probed during Phase 2's load sweep — still an open,
  correctly-flagged gap, not a "checked, doesn't happen" one.
- **`logits_to_keep` optimization** — `forward()` computes vocab-size
  logits for every prefill token, then discards all but the last one per
  request. Real waste, not a correctness issue. `LlamaForCausalLM.forward()`
  supports `logits_to_keep` to avoid this. Chunked prefill landed (Phase
  2.5) and did *not* end up touching this code path — chunking changed
  *which* token slice gets passed to `forward()` each step, not how many
  of that slice's logits get kept afterward, so the waste is still there,
  just spread across more, smaller calls now. Still not worth doing in
  isolation — Phase 4's kernel work is a more natural place to revisit it
  together, if it ends up touching the same output-extraction path.

## Version risk — check this first on a new GPU box

Everything about `attn_implementation="paged|eager"`, `set_attn_implementation`,
`Cache.update()`'s signature, and `PagedAttentionCache`'s internals was
verified by reading the **exact source** of `transformers==5.15.1`.
`requirements.txt` now pins `transformers>=4.43`/`torch>=2.5` (see "Second
GPU session" above), but that floor is for Llama 3.1's `rope_scaling`
format, not the paged-attention internals — the actual API surface this
project depends on isn't guaranteed stable across everything `>=4.43`
allows. **Before debugging anything else on a new box**, confirm the
resolved version, or re-verify the same source locations
(`cache_utils.py`'s `Cache`/`DynamicCache`, `integrations/eager_paged.py`,
`generation/continuous_batching/cache.py`'s `PagedAttentionCache`,
`modeling_utils.py`'s `ALL_ATTENTION_FUNCTIONS`/`set_attn_implementation`)
still match what's described above. **Also check the CUDA driver version**
before assuming a fresh box's default `pip install torch` will just work —
see the Third GPU session's "Environment note" above, unresolved.

## Roadmap after Phase 2 — ranked, with reasoning, not just a list

spec.md was a starting point, explicitly not treated as binding once real
data disagreed with its priorities (same discipline as the bf16-drift
decision in the first GPU session). Scoped out Phases 3-5 plus
alternatives across four axes: time cost, learning/time ratio, novelty vs.
skills already demonstrated elsewhere in the portfolio, and fit to target
companies (MatX #1; Etched and OpenAI tied #2 — see the `portfolio-
roadmap` memory for the full company/role context if a fresh session needs
it, not duplicated in full here).

1. ~~**Chunked prefill**~~ — **done**, see "Fourth GPU session" above.
   Real result was narrower than hoped (a tail-latency fix pre-saturation,
   not a saturation-ceiling fix), but it did real, useful work: it pinned
   down that the saturation-ceiling bottleneck is specifically admission-
   queue depth (`max_num_seqs`), not prefill-step granularity — which is
   exactly what makes Phase 4 (next) a data-motivated pick, not a
   speculative one.
2. **Phase 4 (real paged-attention kernel)** — now the top priority, and
   more directly motivated than it was before Phase 2.5's data landed:
   fixes the actual eager-attention memory-scaling problem behind Bugs 1
   and 3 (Third GPU session) *and* is now the plausible fix for the exact
   thing Phase 2.5 showed governs TTFT at saturation (a tiled kernel's
   small, predictable memory footprint could let `max_num_seqs` rise
   without the same OOM risk, directly shrinking admission-queue depth).
   Extends the existing Triton FlashAttention-2 kernel (not from scratch
   — spec.md's Decision 3 explicitly guards against re-proving
   kernel-writing ability) into a genuinely different technique
   (block-sparse gather-index vs. dense). Strong fit for the Kernel/ML
   Performance Engineer secondary target and MatX's accelerator-codesign
   focus. See "Immediate next steps — Phase 4" below.
3. **Phase 3 (real tensor parallelism)** — spec.md's own first-priority
   stretch, most direct "Anthropic/OpenAI-scale inference" story, but the
   most expensive (multi-GPU rental, distributed correctness debugging)
   and the lowest incremental learning ratio here — real prior from-
   scratch ZeRO/FSDP/DDP experience already exists, and inference-time
   TP's core mechanism (sharded matmul + collectives) is conceptually
   adjacent, even though the inference-specific KV-cache-across-ranks
   details are genuinely new. Still worth doing eventually for the
   OpenAI/Anthropic-fit value.
4. **Prefix-sharing / finishing `fork()`** — real and bounded (ref-
   counting already exists, needs scheduler-level prefix detection at
   admission time), but per the `portfolio-roadmap` memory this is
   explicitly "partially pre-empted" by an existing GRPO K-way prefix-
   sharing derivation elsewhere in the portfolio — lower marginal learning
   value, much of the intellectual content already banked. Nice-to-have,
   not a priority. Explicitly the pick for a fast "second phase after
   Phase 4" if time allows (decided directly with the user this session)
   — TP and disaggregation are each real standalone undertakings with
   meaningful failure-to-finish risk of their own, not bonus-slot
   material.
5. **Phase 5 (real disaggregation)** — highest novelty (genuine new
   mechanism, real cross-process KV handoff) and spec.md itself calls it
   "the single most direct predict-then-validate moment in this repo" —
   but also spec.md's own reach goal, and the time cost (real multi-
   process orchestration, real infra) is at least as high as Phase 3's.
   Only chase with real time to spare.

**Explicitly not recommended** (already declined elsewhere in the
portfolio, would be scope creep here too): CPU/SSD KV-cache tiering
(declined twice already per the roadmap memory), pipeline/expert
parallelism (spec.md's own "Note on scope" section excludes both).

**Decided directly with the user this session, given Phase 4 is likely
the last phase attempted**: risk of not finishing matters more now than
it did ranking these after Phase 2, since there's no next phase to fall
back to if one runs long — reinforcing spec.md's own fallback logic (a
complete phase beats a half-built one) rather than overturning the
ranking above. That's why Phase 4 (bounded, extends existing code) stays
the pick over TP or disaggregation (each real standalone undertakings
with open-ended distributed-debugging/infra risk) even under a tighter
time budget, and why prefix-sharing — not TP or disaggregation — is the
one candidate for a fast second phase if Phase 4 finishes with room to
spare.

## Immediate next steps — Phase 4 (real paged-attention kernel)

**What it is** (per spec.md's Phase 4 section, unchanged there): replace
the gather-then-dense MVP attention path with a real block-sparse kernel
that reads directly from non-contiguous cache blocks — extend the
existing Triton FlashAttention-2 kernel (**lives elsewhere in the user's
portfolio, not in this repo — locate and re-familiarize with it before
designing the integration**, spec.md's Decision 3 explicitly guards
against rebuilding kernel-writing ability from scratch) to take a block
table and gather-index *inside* the kernel, rather than gathering into a
contiguous buffer beforehand the way `_PagedKVCache.update()` does today.

**Why now, concretely, more specific than spec.md's original framing**:
Phase 2.5's real data (see "Fourth GPU session" above) pinned the
saturation bottleneck down to admission-queue depth — `max_num_seqs`
exists specifically as a defensive cap against the eager attention op's
unbounded, batch-composition-dependent memory scaling (Third GPU
session's Bugs 1 and 3: `eager_paged_attention_forward` materializes one
dense `[total_query, total_read]` score matrix in HBM across the *whole*
scheduled batch, and `ModelRunner._infer_num_gpu_blocks` reserves a fixed
activation-memory headroom once at startup with no idea how wide/deep a
real batch gets). A tiled kernel that never materializes the full matrix
has a small, predictable memory footprint regardless of batch composition
— the plausible, direct mechanism to raise `max_num_seqs` safely, which
is the thing Phase 2.5 just showed actually governs TTFT at saturation.

**Current integration point** (`model_runner.py`, read this again before
starting): `self.model.set_attn_implementation("paged|eager")` registers
`eager_paged_attention_forward` via `ALL_ATTENTION_FUNCTIONS`; it reads a
`cache` kwarg and calls `cache.update(key_states, value_states, layer_idx,
read_index, write_index)`. `_PagedKVCache.update()` (top of the file) is
the small duck-typed class actually doing the gather/scatter today —
`index_copy_`/`index_select` into/out of a flattened per-layer view of
`self.kv_cache`, then a plain dense matmul over the gathered buffer. Phase
4's kernel replaces that dense-matmul-over-a-gathered-buffer step with a
real block-sparse kernel that gathers per-tile *inside* the kernel itself.

**Real design forks to surface explicitly, not silently resolve** (same
discipline as every `🧠` decision so far):
- Scope: does the kernel need to handle this engine's full heterogeneous
  batch shape (mixed prefill chunks + decode + multiple concurrent
  requests, per-request causal masking) from the start, or is it worth
  starting narrower — e.g. decode-only tiled attention first, since
  decode-driven concurrency depth (many long-context requests) is what
  actually forced `max_num_seqs` down in Bug 3, and prefill's now
  partially mitigated by chunking — then extending to prefill/mixed
  batches once that's working? A real scope decision, not a default.
- How the kernel takes the block table: gather-index computed once in
  Python and passed in (closer to today's `read_idxes`/`write_idxes`), or
  computed inside the kernel from `block_table` directly? Affects how
  much of `_flat_slot`'s logic moves into the kernel itself.
- Whether `_PagedKVCache` stays as the integration shim (kernel called
  from inside `.update()`) or the attention-function registration
  (`set_attn_implementation`) needs to point at something new entirely.

**Correctness oracle**: token-for-token match against the current
eager/dense attention path on identical prompts, same pattern as every
correctness checkpoint in this project so far (Phase 1's HF comparison,
Phase 2.5's chunked-vs-one-shot comparison) — the kernel must change
*how* attention gets computed, not *what* gets generated.

**Real measurement, the actual checkpoint, more specific than spec.md's
original "measure the real speedup"**: with the new kernel's smaller,
predictable memory footprint, does raising `max_num_seqs` (and re-running
`scripts/benchmark_load.py` with the same `--rates 0.5,1,2,4,8,16,32`
sweep used for `benchmark_results_final.csv`/`benchmark_results_chunked.
csv`) actually reduce TTFT at saturation? That's the real, data-motivated
question Phase 2.5 set up — not just "is the kernel faster in isolation."
A genuine before/after against both existing CSVs, not a fresh,
incomparable run.

**Division of labor**: same pattern as every phase so far — design forks
above get surfaced and decided together, then implementation of the
resolved result is Claude's to do directly.

Phase 2's own report/writeup (the predicted-vs-real section spec.md's
Phase 2 originally called for) is still open — explicitly not a
Claude-authored writeup per the established division of labor (see "How
this session worked"), deferred behind Phase 2.5 at the user's own
request. Phase 2.5 is now done too, so this is still open and still not
forgotten, just now behind Phase 4 as well.

## How this session worked, for continuity

- Division of labor, refined in the second GPU session: not a strict "user
  writes all algorithmic logic" split — once a real design fork is
  explicitly resolved (see next bullet), Claude implements the mechanical
  result directly, including in `block_manager.py`/`scheduler.py` (e.g. the
  `total_len`-vs-`prompt_len` sizing fix, the last-resort eviction wiring,
  the block-growth invariant fix, all three Third-GPU-session bug fixes).
  The line that matters is: genuine *design* decisions get surfaced, not
  silently made either way.
- **Real forks get surfaced explicitly, not silently resolved.** When
  eager-vs-last-resort eviction came up, Claude didn't just pick one — it
  was framed as an actual decision with tradeoffs and asked about directly
  (see commit `96bc229`'s message for the reasoning once resolved). Same
  pattern held for: the OOM fix approach (bound the batch vs. fix the
  attention op vs. shrink the pool), the scheduler-starvation fix (always-
  admit-lone-candidate vs. reject-oversized vs. chunked prefill), and the
  post-Phase-2 roadmap ranking above. Expected to hold for chunked
  prefill's own design forks (see "Immediate next steps") and anything
  else that's a judgment call rather than a mechanical consequence of an
  already-settled decision.
- `spec.md` is a starting point, not a constraint to defer to reflexively —
  said explicitly by the user mid-session, twice now (once for the bf16
  correctness bar, once for the whole post-Phase-2 roadmap, which added a
  phase spec.md never specified). Don't treat its exact wording as unable
  to flex when a real finding warrants it — but don't silently deviate
  either; that's also a "surface it" moment, same as the bullet above.
- Review style: conceptual/logic bugs get Socratic guiding questions, not
  direct answers — syntax/typo-level mistakes get fixed directly, no
  ceremony. This produced several real caught bugs (see commit messages).
- Testing philosophy: test everything possible without the GPU first
  (pure-Python unit tests for `block_manager`/`scheduler`/`workload`, zero
  torch dependency) — GPU time is expensive/limited, so it should only be
  spent validating what genuinely can't be checked any other way. Held up
  again this session: three real bugs (all in "Third GPU session" above)
  were only findable under real load, not from unit tests or Phase 1's
  light-load demos — real load sweeps are worth their GPU cost, not a
  formality once unit tests pass.
- GPU workflow: this chat environment has no direct GPU access. The
  pattern is Claude gives exact copy-paste shell commands, the user runs
  them on the rented box and pastes output back, Claude reads/diagnoses
  from that — held up across a long iterative bug-fix loop this session
  (four real GPU round-trips before a clean sweep ran), not just one-shot
  demos.
- Commit style: small, narrated commits matching the actual build sequence
  (not squashed) — "more commits the merrier," each with a real commit
  message explaining the *why*, not just the *what*. Push after every
  commit (or small batch) rather than batching a long series unpushed.
  Benchmark result CSVs get committed too, not left only on the GPU box —
  they disappear when the instance is released.
- Comment style: terse. TODO markers stay one line, no restated context —
  established explicitly after an early draft was judged "too much hint."
- Analytical rigor: when asked whether a finding generalizes, work through
  *why* rather than reflexively agreeing or asserting — "decode isn't the
  bottleneck" was correctly challenged as workload/concurrency-specific,
  not a universal serving-systems truth, and the actual generalizable
  claim (the conditional structure itself) is worth preserving precisely,
  not flattened back into the more quotable but wrong universal version in
  any future writeup. Held up again in the Fourth GPU session: when the
  user's first instinct on the two chunked-vs-one-shot test failures was
  "probably the same known bf16 issue," that wasn't accepted at face value
  even though it turned out right — it was a genuinely different
  comparison (chunked vs. one-shot, not one-shot vs. HF), so a real
  logit-level diagnostic got run before agreeing, not just asserting the
  pattern-match. Same discipline, applied to the user's own hypothesis
  this time, not just an internal claim.
- Derivation sounding-board pattern, several real instances this session:
  when the user proposed a mechanism (chunk-size formula, the
  update_after_step() sketch), the response was to check it against the
  actual code before agreeing, not derive a fresh answer unprompted or
  rubber-stamp theirs. Caught several real things this way that would've
  been genuine bugs if implemented as first proposed: `remaining` needed
  `total_len` not `prompt_len` (preemption survives output_token_ids
  across a reset num_computed_tokens); the `==` in the user's phase-
  transition sketch needed to be `>=` (same defensive-invariant lesson as
  `block_manager._has_capacity_for`'s own precedent); a `KeyError` from
  the min-chunk-size floor's `continue`-vs-`break` non-issue nearby. Two
  more caught later by direct sanity-check re-reading (not prompted by
  the user pointing them out): a variable-shadowing near-miss in
  `model_runner.forward()` (`start` reused for two different meanings),
  and the real ordering bug where `total_len` inflating from this step's
  own token append would've silently broken final-chunk detection.
- Given Phase 4 is likely the last phase attempted, explicitly discussed
  with the user how that changes phase selection: risk of not finishing
  cleanly matters more with no next phase to fall back to, which
  reinforces (not overturns) spec.md's own fallback logic and the
  existing roadmap ranking — Phase 4 stays the pick over TP/disaggregation
  specifically because it's bounded and extends existing code, and
  prefix-sharing (not TP or disaggregation) is the one real candidate for
  a fast second phase if time remains, since TP/disaggregation are each
  standalone undertakings with their own real finish-risk, not bonus-slot
  material.
