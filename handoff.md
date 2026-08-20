# Handoff — serving-engine, Phase 1 complete, starting Phase 2

Written so a fresh chat session (or future you) can pick up exactly where
this one left off, without re-deriving anything already settled. Read
`spec.md` first for the project's overall shape (phases, decisions,
scope) — this doc covers everything *since* Phase 0 that spec.md doesn't
capture: real implementation decisions, bugs found and fixed, and what's
actually left before Phase 2 is checkpointed.

**Repo:** https://github.com/ayush-os/serving-engine (public), 22 commits on `main`.

## Status in one line

**Phase 1 is fully checkpointed.** `test_correctness.py` passes (2 exact
matches, 1 `xfail` for expected bf16 drift — see "First GPU session"
below). `scripts/continuous_batching_demo.py` demonstrates staggered
arrivals folding into an already-running batch (observed in the per-step
timeline, not assumed) and measures a 1.71x wall-time speedup over a
naive-sequential baseline on a 3-request run, with sustained non-zero
`nvidia-smi dmon` SM utilization throughout. Phase 2 is next — see
"Immediate next steps" at the bottom, now pointed at Phase 2 instead.

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

## Repo map

```
serving_engine/
  block.py           BLOCK_SIZE=16, Block dataclass — done, no known issues
  request.py         Request/RequestPhase/RequestStatus — done
  block_manager.py   allocate/append_slot/free/fork done + tested.
                      preempt() and append_slot's CoW branch deliberately
                      stubbed (see "Deliberately deferred" below)
  scheduler.py        schedule()/update_after_step() done + tested.
                      max_num_batched_tokens/max_num_seqs cap deliberately
                      not enforced yet (see below)
  model_runner.py     ModelRunner + _PagedKVCache + forward() — fully
                      written, ZERO GPU runs so far. The riskiest file.
  engine.py           LLMEngine wiring — done, untested end-to-end (needs
                      a real forward() run)
tests/
  test_block_manager.py   8 tests, pure Python, no torch/GPU — passing
  test_scheduler.py        12 tests, pure Python, no torch/GPU — passing
  test_correctness.py       Phase 1's actual checkpoint (token-for-token vs
                      HF .generate()) — needs a real GPU + real weights,
                      never run
.venv/                local venv for the two GPU-free test files. transformers
                      5.15.1 got pip-installed into it during research (see
                      "Version risk" below) — this is NOT what runs on the
                      rented GPU, just what the design was verified against.
```

## The real story of `model_runner.forward()` — why it looks the way it does

This is the part most worth understanding before touching it again, since
it went through a real pivot mid-build.

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
  deliberately mixed).
- `attention_mask`: `[1,1,total_query,total_read]`, built from two
  broadcast comparisons over per-token (request-group, logical-position)
  pairs — same request AND key position ≤ query position. Uses
  `torch.finfo(dtype).min`, not literal `-inf` (avoids NaN-from-fully-masked
  softmax rows, matching HF's own convention).
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

## Deliberately deferred — not bugs, don't "fix" these reflexively

- **`block_manager.py`: `append_slot`'s CoW branch** (marked `# TODO`) —
  needed only once `fork()` is actually exercised by real prefix-sharing,
  which nothing currently triggers (no scheduler-level prefix detection
  exists, and the spec's own scope note says not to chase that beyond what
  the block manager already gives for free). Stays deferred indefinitely.
- **Starvation avoidance under sustained load** — the last-resort eviction
  trigger (see "Second GPU session" above) only fires when *nothing at all*
  progressed this step. A persistently unlucky candidate could in theory
  wait a long time if something else keeps making unrelated progress every
  step, since the "nothing progressed" trigger won't refire while anything
  else is moving. Real fairness needs more machinery (e.g. tracking how
  long a candidate's been stuck) than this project's scope calls for right
  now — worth watching for in Phase 2's load sweep, not pre-solving here.
- **`logits_to_keep` optimization** — `forward()` computes vocab-size
  logits for every prefill token, then discards all but the last one per
  request. Real waste, not a correctness issue. `LlamaForCausalLM.forward()`
  supports `logits_to_keep` to avoid this. Not worth doing until profiling
  (Phase 2) actually shows `lm_head` compute mattering.

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
still match what's described above.

## Immediate next steps — Phase 2

Phase 1's two checkpoints (correctness, continuous-batching demo) and both
Phase-2 prerequisites (`preempt()`, the batching cap) are all done and
GPU-verified — see "Status in one line" and "Second GPU session" above.
Phase 2 itself is the benchmark report against
`disagg_and_placement_notes.md`'s simulator predictions (spec.md's Phase 2
section). Marked 🧠 in spec.md — this phase is meant to be judgment-heavy,
not just scaffolding: the actual point is a real, checked comparison
against your own prior prediction, not a demo to build and move past.

**Per spec.md's Phase 2 section:**
1. Build a synthetic request generator matching the same distributional
   assumptions the discrete-event simulator used (`disagg_and_placement_notes.md`
   §4) — Poisson arrivals, a realistic prompt/output-length distribution —
   so results are actually comparable to the simulator's predictions, not a
   different workload shape.
2. Measure real throughput (req/s), TTFT, per-token decode latency, and GPU
   occupancy under a swept load.
3. Write the predicted-vs-real comparison: the simulator found prefill's
   fixed compute ceiling (~4,138 req/s in that setup), not decode capacity
   or the KV pool, was the real bottleneck. Does real hardware confirm that
   shape, or does something the simulator's abstraction missed show up for
   real (kernel launch overhead, scheduling overhead, memory fragmentation)?
   A genuine, checked answer either way is the actual checkpoint —
   disagreement here is a more interesting finding than agreement.

`append_slot`'s CoW branch and starvation avoidance under sustained load
stay deferred indefinitely (see "Deliberately deferred" above) —
unaffected by Phase 2, though the latter is worth watching for during the
load sweep.

## How this session worked, for continuity

- Division of labor: the user writes the actual algorithmic logic
  (block manager internals, scheduler admission loop, the index math in
  `forward()`); Claude handles scaffolding, boilerplate, source-verified
  research (reading real `transformers` internals rather than guessing),
  writing test files, and reviews.
- Review style: conceptual/logic bugs get Socratic guiding questions, not
  direct answers — syntax/typo-level mistakes get fixed directly, no
  ceremony. This produced several real caught bugs (see commit messages).
- Testing philosophy: test everything possible without the GPU first
  (pure-Python unit tests for `block_manager`/`scheduler`, zero torch
  dependency) — GPU time is expensive/limited, so it should only be spent
  validating what genuinely can't be checked any other way.
- Commit style: small, narrated commits matching the actual build sequence
  (not squashed) — "more commits the merrier," each with a real commit
  message explaining the *why*, not just the *what*.
- Comment style: terse. TODO markers stay one line, no restated context —
  established explicitly after an early draft was judged "too much hint."
