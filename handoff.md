# Handoff — serving-engine, end of Phase 1 implementation

Written so a fresh chat session (or future you) can pick up exactly where
this one left off, without re-deriving anything already settled. Read
`spec.md` first for the project's overall shape (phases, decisions,
scope) — this doc covers everything *since* Phase 0 that spec.md doesn't
capture: real implementation decisions, bugs found and fixed, and what's
actually left before Phase 1 is checkpointed.

**Repo:** https://github.com/ayush-os/serving-engine (public), 22 commits on `main`.

## Status in one line

All Phase 1 code is written and internally consistent — `block_manager.py`,
`scheduler.py`, `engine.py`, `model_runner.py` — but **none of it has ever
run against the real model on a real GPU.** Everything GPU-independent has
been tested (20 passing unit tests, zero GPU). Everything GPU-dependent
(`model_runner.forward()`, `test_correctness.py`) has only been read-through
verified against `transformers` source, never executed. The first GPU
session is where this either works or reveals real bugs.

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

## Deliberately deferred — not bugs, don't "fix" these reflexively

- **`block_manager.py`: `preempt()`** — raises `NotImplementedError`. Not
  needed for Phase 1's checkpoints (neither correctness-vs-HF nor the
  continuous-batching demo forces pool exhaustion if `num_gpu_blocks` has
  headroom). **Hard blocker before Phase 2's load sweep**, though — that's
  designed to hit exactly this.
- **`block_manager.py`: `append_slot`'s CoW branch** (marked `# TODO`) —
  needed only once `fork()` is actually exercised by real prefix-sharing,
  which nothing currently triggers (no scheduler-level prefix detection
  exists, and the spec's own scope note says not to chase that beyond what
  the block manager already gives for free). Genuinely optional, unlike
  `preempt()`.
- **`scheduler.py`: `max_num_batched_tokens`/`max_num_seqs` cap** — not
  enforced. Same status as `preempt()`: fine for Phase 1, must exist before
  Phase 2's load sweep or an uncapped huge-prefill iteration could
  contaminate the exact compute-ceiling-vs-KV-pool measurement Phase 2
  exists to make.
- **`logits_to_keep` optimization** — `forward()` computes vocab-size
  logits for every prefill token, then discards all but the last one per
  request. Real waste, not a correctness issue. `LlamaForCausalLM.forward()`
  supports `logits_to_keep` to avoid this. Not worth doing until profiling
  (Phase 2) actually shows `lm_head` compute mattering.

## Version risk — check this first on the GPU box

Everything about `attn_implementation="paged|eager"`, `set_attn_implementation`,
`Cache.update()`'s signature, and `PagedAttentionCache`'s internals was
verified by reading the **exact source** of whatever `transformers` version
pip installed locally during this session (**5.15.1**). `requirements.txt`
currently pins nothing (`transformers` unpinned). If a different version
lands on the rented GPU box, any of these APIs could have shifted — this
is a fast-moving part of the library. **Before debugging anything else**,
confirm the installed version matches, or re-verify the same source
locations (`cache_utils.py`'s `Cache`/`DynamicCache`, `integrations/eager_paged.py`,
`generation/continuous_batching/cache.py`'s `PagedAttentionCache`,
`modeling_utils.py`'s `ALL_ATTENTION_FUNCTIONS`/`set_attn_implementation`)
still match what's described above. Consider pinning `transformers==5.15.1`
in `requirements.txt` to remove this risk entirely, unless there's a reason
to want newer.

## Immediate next steps to finish Phase 1

1. **Rent the A100** (single GPU — decided in Phase 0, sufficient for
   Phase 1-2; H100 was reserved for Phase 3's TP target if that's ever
   reached).
2. **Environment**: CUDA-enabled torch, `transformers`, `accelerate`;
   accept Llama-3-8B-Instruct's license/auth on Hugging Face (gated repo).
   Check the version risk above before anything else.
3. **Pick a real `num_gpu_blocks`** from actual free GPU memory
   (`torch.cuda.mem_get_info()` or `nvidia-smi` is enough — no need for a
   full automatic profiler, that's more sophistication than this reduced
   scope needs).
4. **Run `tests/test_correctness.py`**, debug until genuinely token-for-token
   against HF `.generate()`. This is where the index math, mask, and
   `_PagedKVCache` all get validated for real for the first time. Expect
   to find at least one real bug — none of this has executed yet.
5. **Write the continuous-batching demo** (not built yet at all) — the
   second Phase 1 checkpoint. Staggered-arrival, concurrent, different-length
   requests through `engine.add_request()`/`step()`, with *observed*
   evidence the GPU doesn't idle between them (`nvidia-smi dmon` or step
   timestamps — the spec's own bar is "confirmed, not assumed").

Once both checkpoints pass, Phase 1 is done. Phase 2 (the benchmark report
against `disagg_and_placement_notes.md`'s simulator predictions) is next;
`preempt()` and the token/seq cap become required before that starts.

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
