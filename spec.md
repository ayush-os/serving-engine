# Project Spec: A Real Continuous-Batching, Paged-KV Serving Engine

**Continuity note.** Four prior projects in this repo (`prefill_notes.md`,
`decode_notes.md`, `moe_routing_notes.md`, `disagg_and_placement_notes.md`)
predicted serving-system behavior — chip ratios, KV-cache placement
policy, prefill's compute ceiling as the real bottleneck — entirely
analytically or against a hand-built discrete-event simulator. None of it
has ever been checked against a real system on real GPUs. This project is
that check: build an actual serving engine, run real traffic through it,
and put real numbers next to the predictions `disagg_and_placement_notes.md`
already made. Same "predict, then validate" discipline every project in
this repo already uses — just against real hardware instead of a
simulator, for the first time.

**Why this, why now, stated plainly.** All four target companies care
about inference serving; Anthropic/OpenAI live in it directly. Your
existing work is almost entirely analytical — this is the single project
that converts "I modeled this" into "I built this and here's what
happened when reality disagreed with the model."

**Toolchain:** PyTorch, a real 7-8B open-weight dense model (Llama-3-8B-
Instruct — see Decision 1), Triton (your own existing FlashAttention-2
kernel gets reused/extended in Phase 3), 4-8x A100/H100 rented GPUs.

**Legend:** 🔧 = boilerplate/setup. 🧠 = a real decision — state what you
picked and why, don't silently default.

---

## Phase 0 — Setup (🔧 light reading, three real 🧠 decisions)

### Reading (🔧, light — you've already read most of the relevant literature)

- **Orca** (Yu et al.) — the original continuous-batching/iteration-level-
  scheduling paper. Read specifically for the actual scheduling loop
  mechanism (per-iteration admission, not per-batch).
- **vLLM/PagedAttention** (Kwon et al.) — you've already cited this from
  source in `disagg_and_placement_notes.md`; re-read specifically for the
  block manager's real data structures (block table, reference counting,
  copy-on-write for shared prefixes) rather than the high-level pitch.
- Skim your own `disagg_and_placement_notes.md` §2c/§3 again before Phase 2
  — that's the exact set of predictions this project checks.

### 🧠 Decision 1: model — a small MVP target now, a real TP stretch target decided now too

**Llama-3-8B-Instruct** (or similar 7-8B dense open-weight model) is the
**Phase 1-2 MVP target** — bf16-resident on a single GPU, so no
tensor-parallel sharding is needed just to get the scheduler/block-manager
work (the actual core gap this project exists to close) running and
correct. Real published weights + tokenizer also give a cheap correctness
oracle (Phase 1's checkpoint compares against plain HF `generate()` on
identical prompts).

**Real tensor parallelism is explicitly in scope, deliberately deferred
to Phase 3, not dropped.** You've already built real training-side
sharding from scratch (DDP, ZeRO-1, FSDP/ZeRO-3) — inference-time TP
(column/row-parallel linear layers + all-reduce/all-gather at a few fixed
points) is a genuinely different, more literal match to "parallelism,
sharding" in your own stated goal, and is mechanically simpler than
what FSDP already required, so it's a good time-for-signal trade. **Name
the TP stretch target now, so Phase 3 doesn't need fresh research
later**: a model that doesn't fit on one GPU at your rented GPU count
(e.g. Llama-3-70B-Instruct at TP=4 or TP=8) — decided here, executed in
Phase 3. Deliberately kept out of the Phase 1-2 critical path: TP
debugging is a real distraction risk against the actual identified gap
(the scheduler/block-manager layer), so it's staged as the first stretch
phase, not baked into the MVP.

MoE stays out entirely (Decision unchanged) — it would reopen expert-
parallel complexity that duplicates what `moe_routing_notes.md` already
covered analytically, for a workload this project isn't trying to
re-validate.

### 🧠 Decision 2: the pivot-to-disaggregation interface, decided now even though disagg is Phase 4

The single design choice that determines whether Phase 4 is a clean
extension or a rewrite: **KV cache blocks are a serializable, addressable
unit from day one**, and **every request object carries explicit phase
state (prefill vs. decode)** even though Phase 1 runs both roles in one
process on one GPU. Concretely: the block manager should address blocks
by an ID that doesn't assume same-process memory (not a raw pointer), and
the scheduler should treat "this request needs a prefill step" and "this
request needs a decode step" as distinct schedulable units from the start,
not a single opaque "process this request" call. This costs a small amount
of extra structure in Phase 1 and is what makes Phase 4 a real pivot
instead of a second project.

### 🧠 Decision 3: general build-vs-reuse rule (blanket reuse first, triage at the end — not decided per-component up front)

**Default to reusing existing, proven code for everything in Phase 1-2,
full stop.** Model forward-pass internals, attention op, anything else
that isn't the scheduler/block-manager — reuse it, no exceptions, so
Phase 1-2's actual deliverable (a correct engine + a real benchmark report
against your own simulator's predictions) is never put at risk by a
from-scratch component's bugs. The one thing this does *not* apply to:
**the scheduler and block manager are hand-built starting Phase 1, always**
— that's the actual target gap this project exists to close, not a
component up for the reuse-vs-build question at all.

**Then, immediately after Phase 2's benchmark report is done — one real
triage checkpoint, not an open-ended "someday":** go through the Phase 1-2
components you reused (attention op, and whatever else you touched enough
to have an opinion on) and apply a simple test to each: (a) is rewriting
it bounded (a day or two, not a week), and (b) would it teach you
something genuinely distinct from what your existing projects already
prove. You'll answer this far better now, having actually used the code,
than by reading source cold in Phase 0. This triage is what decides which
of Phase 3 (TP), Phase 4 (paged kernel), or anything else that surfaced as
interesting while building Phase 1-2, actually gets attempted, and in
what order, given whatever time is actually left — not a fixed sequence
committed to in advance.

**Checkpoint:** model MVP target and TP stretch target both chosen and
justified; block-manager/request interfaces designed with the Phase 5
pivot in mind; everything else defaults to reuse until the post-Phase-2
triage.

---

## Phase 1 — Core colocated engine: the actual from-scratch systems layer (🧠 this is the real work)

**Paged KV-cache block manager** — real design questions, not defaults:
- Fixed block size (tokens/block): the real tradeoff is bookkeeping/
  fragmentation overhead (small blocks) vs. internal fragmentation waste
  on short sequences (large blocks) — the same shape of tradeoff your MX
  project already reasoned through for a different resource. Pick one,
  state the reasoning.
- Free-list allocator, reference counting (for shared-prefix blocks — a
  real, direct callback to your own GRPO K-way prefix-sharing derivation
  in `rl_codesign_notes.md` Phase 1, now with a real mechanism to
  implement, not just a closed-form cost reduction).
- Eviction/preemption policy for when the cache is full under load — pick
  a real, stated policy (e.g., preempt lowest-priority/oldest request),
  don't leave it undefined until it breaks under benchmarking.

**Continuous batching scheduler** — the actual Orca-style mechanism: a
per-iteration admission loop that mixes prefill and decode steps for
different requests in the same engine step, checking block-manager
capacity before admitting. This is the real, novel-to-you piece — nothing
in your repo has built an iteration-level scheduler before.

**Request lifecycle**: arrival → tokenize → prefill → decode loop →
completion → block release, wired through the scheduler and block manager
above.

**Checkpoint:** a real, running single-GPU serving engine. Correctness:
outputs match plain HF `generate()` on identical prompts, token-for-token.
Real behavior check: demonstrably batches multiple concurrent, staggered-
arrival, different-length requests without leaving the GPU idle between
them — the actual point of continuous batching, confirmed, not assumed.

---

## Phase 2 — Real benchmarking against your own simulator's predictions (🧠)

- Build a synthetic request generator matching the same distributional
  assumptions your discrete-event simulator used (`disagg_and_placement_notes.md`
  §4) — Poisson arrivals, a realistic prompt/output-length distribution —
  so the numbers that come out are actually comparable to what you already
  predicted, not a different workload shape.
- Measure real throughput (req/s), TTFT, per-token decode latency, and GPU
  occupancy under a swept load.
- **The real comparison this phase exists for**: your simulator found
  prefill's fixed compute ceiling (~4,138 req/s in that setup), not decode
  capacity or the KV pool, was the real bottleneck. Does real hardware
  roughly confirm that shape (prefill saturates while decode has slack), or
  does something the simulator's abstraction missed show up for real (real
  kernel launch overhead, real scheduling overhead, real memory
  fragmentation)? A genuine, checked answer either way — disagreement here
  is a more interesting finding than agreement.

**Checkpoint:** a real benchmark report (throughput/latency vs. load) with
an explicit predicted-vs-real section against your own prior simulator
output.

**Real data landed** (see `handoff.md`'s "Third GPU session" for the full
story): throughput plateaus hard past a real saturation point, TTFT grows
unboundedly past it, and — directly measured, not inferred — a small
minority of steps by count (prefill-involving, ~8%) consume a
disproportionate share of wall-clock time (~33-35%) at saturation. Real
hardware confirms the simulator's qualitative shape; absolute numbers
aren't comparable (1 A100/8B/bf16 here vs. a 29-machine TPU-8i/70B/FP4
pool in the simulator) — only the mechanism transfers. The written report
itself is still open, deliberately not Claude-authored (see handoff.md).

---

## Phase 2.5 — Stretch: chunked prefill (🧠, motivated directly by Phase 2's own real data, not originally in this spec)

Not a phase this spec originally called for — added after Phase 2's real
benchmark data made the case directly, not from speculation. `schedule()`'s
`NEEDS_PREFILL` branch always admits a request's *entire* prompt in one
scheduling iteration; `model_runner.forward()`'s prefill path always
contributes the whole prompt as both write and read positions in one
`forward()` call. That's real head-of-line blocking — a big prefill can
stall every decode request already in flight for that step's whole
duration, which is exactly what Phase 2's unbounded TTFT growth past
saturation reflects, and what its direct prefill-time-share measurement
(~33-35% of wall time from ~8% of steps) shows mechanistically.

**The mechanism**: split a request's prefill into multiple scheduling
iterations, each processing a bounded chunk of the prompt, interleaved
with other requests' decode steps the same way prefill/decode are already
mixed today. Production term: Sarathi-Serve-style chunked prefill.
`Request.num_computed_tokens` already exists for exactly this (currently
only reset on `preempt()`, never incremented or read elsewhere).

**Real design questions, not defaults** (same discipline as every other
🧠 decision in this spec): chunk sizing policy (fixed size vs. filling
whatever's left of `max_num_batched_tokens`'s budget after decode work is
admitted — the real production answer is usually the latter); how a
partially-prefilled request's block table grows chunk-by-chunk (whether
`block_manager`'s existing decode-growth machinery transplants directly);
whether decode keeps scheduling priority within a chunk-containing
iteration.

**Correctness oracle**: token-for-token match against the current
one-shot-prefill path on identical prompts — chunking must change *how*
the compute is scheduled, not *what* gets generated.

**Real measurement**: does chunking flatten the TTFT-vs-load curve from
Phase 2's own benchmark data, at the cost of some decode throughput
(interleaving means a chunk's compute now shares an iteration with decode
steps it previously didn't)? A genuine before/after against Phase 2's own
numbers (`benchmark_results_final.csv`), same discipline as every other
real-measurement checkpoint in this spec.

Ranked ahead of Phases 3-5 for now — see `handoff.md`'s "Roadmap after
Phase 2" for the full reasoning across time cost, learning/time ratio, and
company fit. This spec was never meant to be followed past the point real
data disagrees with its own priorities.

**Real data landed** (see `handoff.md`'s "Fourth GPU session" for the full
story): the "chunking flattens the TTFT-vs-load curve" hypothesis above
was wrong, and the real result is narrower and more specific — chunking
fixed a real tail-latency pathology pre-saturation (`ttft_p99_ms` down
14-20% at unsaturated rates, a genuine all-or-nothing admission bug it
removed) but left the saturation ceiling itself untouched (`ttft_mean_ms`
and throughput statistically unchanged at rate≥4), because at saturation
TTFT is dominated by admission-queue depth (`max_num_seqs`), a resource
chunking never touches. That's a more useful result than a clean "yes it
flattened," not a less useful one — it pins the saturation bottleneck down
to something Phase 4 can plausibly fix and chunking structurally can't,
directly motivating Phase 4's priority below. Correctness: GPU-verified,
chunked-vs-one-shot match modulo the same bf16-noise signature Phase 1's
own HF-comparison xfail already established as expected, not chased
further (same discipline, see `handoff.md`'s "First GPU session").

---

## Phase 3 — Stretch: real tensor parallelism for a model that doesn't fit on one GPU (🧠, re-ranked below Phase 4 post-triage — see note)

Entry into this phase (and its priority relative to Phase 4/5) is decided
at the Decision 3 triage checkpoint, not fixed in advance — this section
describes what the work looks like *if* triage selects it. It was named
as the likely first pick before that checkpoint actually ran; the triage
(see `handoff.md`'s "Roadmap after Phase 2") re-ranked Phase 4 above this
one instead — most expensive stretch phase (multi-GPU rental, distributed
correctness debugging) with the lowest incremental learning ratio given
existing real ZeRO/FSDP/DDP experience, and real finish-risk if it's
attempted with limited time remaining. Still real, still valuable, just
not next.

Extend the engine to serve Decision 1's named TP target (e.g. Llama-3-70B
at TP=4 or TP=8) by building real column/row-parallel sharded linear
layers yourself, with all-reduce/all-gather collectives inserted at the
correct points in the forward pass — not by wrapping an existing TP
library. This is the direct "parallelism, sharding" pillar of your
original goal, applied to inference specifically rather than training,
and a natural extension of your existing DDP/FSDP work into new territory.

**Real design questions, not defaults**: which axis to shard each weight
matrix along (the standard Megatron-style split — column-parallel for the
first linear in an MLP block, row-parallel for the second, so only one
all-reduce is needed per block rather than one per matmul — derive this
rather than assuming it, the same way you derived dataflow choices for the
systolic array projects); how attention heads split across TP ranks
without breaking correctness of the softmax/output projection.

**Correctness oracle**: compare token-for-token against a reference
multi-GPU run using an existing library (HF `accelerate`/`device_map`) on
the same model and prompts — your own TP implementation should match
exactly, not approximately.

**Real measurement**: does adding TP change the throughput/latency
picture from Phase 2's benchmark in the way you'd predict (more compute
capacity, but real communication overhead per token now in the critical
path)? A genuine before/after comparison against Phase 2's own numbers.

Explicitly the first-priority stretch — do this before Phase 4/5 if time
is limited, since it's the more direct match to your stated goal and to
what Anthropic/OpenAI-scale inference actually requires.

---

## Phase 4 — Now the top-priority stretch: a real paged-attention kernel (🧠, re-ranked above Phase 3 post-triage — see note)

Replace the gather-then-dense MVP attention path with a real block-sparse
kernel that reads directly from non-contiguous cache blocks — extend your
own existing Triton FlashAttention-2 kernel to take a block table and
gather-index inside the kernel, rather than gathering into a contiguous
buffer beforehand. Measure the real speedup (or lack thereof) over the
Phase 1 MVP path — a real number, not assumed.

Originally written as optional/lower-priority; re-ranked to the top
stretch pick after Phase 2.5's real data landed (see `handoff.md`'s
"Fourth GPU session" and Phase 2.5's "Real data landed" note above), for
a concrete reason, not just general kernel-work appeal: Phase 2.5 pinned
the saturation-latency bottleneck down to admission-queue depth
(`max_num_seqs`), which exists as a defensive cap against the eager
attention op's unbounded, batch-composition-dependent memory scaling
(the same mechanism behind Phase 2's own real OOM bugs). A tiled kernel
that never materializes the full attention matrix has a small, predictable
memory footprint regardless of batch composition — the plausible, direct
fix for the exact resource Phase 2.5 showed governs TTFT at saturation.
The real measurement checkpoint is correspondingly more specific now:
does raising `max_num_seqs` post-kernel actually reduce TTFT at
saturation in a rerun of `scripts/benchmark_load.py`, not just "is the
kernel faster in isolation." See `handoff.md`'s "Immediate next steps —
Phase 4" for the full design-fork list.

Given this is likely the last phase attempted (per the user, see
`handoff.md`), its bounded scope and reuse of existing kernel work also
make it the lower-risk pick over Phase 3/5 specifically because it's more
likely to finish cleanly — spec.md's own Fallback logic already says a
complete phase beats a half-built one, which matters more with no next
phase to fall back to.

---

## Phase 5 — Stretch: disaggregated prefill/decode (🧠, the reach goal, only with time to spare)

Split prefill and decode onto separate GPU processes, using Decision 2's
interface to hand off request state (including real KV-cache bytes, over
real NVLink/network, not simulated) across the process boundary.

**The real comparison this phase exists for**: `disagg_and_placement_notes.md`
predicted a real dense handoff cost (~40 MiB/request) and concluded it's
bandwidth-dominated, clearing DistServe's own <0.1%-of-latency bar. Does a
real measured handoff, on real hardware, actually clear that bar? This is
the single most direct "predict then validate against reality" moment
available anywhere in this repo — worth doing if time allows, but a real
reach goal, not required scope.

---

## Note on scope

Resist three real temptations:

1. **Chasing full production-vLLM feature parity** — LoRA hot-swapping,
   guided/structured decoding, general cross-request prefix caching beyond
   what Phase 1's block manager already gives you for free. The mechanism
   (paged blocks + continuous batching, optionally TP and/or
   disaggregated) is the point, not feature completeness.
2. **Pipeline parallelism and expert parallelism** — real, legitimate
   future extensions, deliberately not this project's job. PP adds
   pipeline-bubble scheduling, a genuinely separate problem poorly suited
   to latency-sensitive serving in the first place; EP would require
   reopening Decision 1's MoE call for a workload `moe_routing_notes.md`
   already covers analytically. TP is the one parallelism dimension in
   scope (Phase 3) — resist the pull to keep going once it works.
3. **Re-proving kernel-writing ability from zero** — Decision 3 stages
   attention so Phase 1's correctness work never blocks on a novel kernel;
   Phase 4's kernel work extends what you've already built, it doesn't
   restart it.

## Fallback

Phase 1 alone — a correct, working, continuously-batched, paged-KV
single-GPU engine — is already a complete, real, demoable artifact: it's
the one thing in this entire repo that isn't analytical or simulated.
Phase 2 (real benchmarking vs. your own simulator's predictions) is the
natural close-the-loop addition and the strongest interview story — and
is now data-complete (see Phase 2's "Real data landed" note above).

**Stretch priority, re-ranked after Phase 2's real data came in, then
again after Phase 2.5's** (see `handoff.md`'s "Roadmap after Phase 2" for
the full reasoning across time cost, learning/time ratio, novelty vs.
skills already demonstrated elsewhere in the portfolio, and target-company
fit): ~~**Phase 2.5 (chunked prefill)** first~~ — **done**, see Phase
2.5's "Real data landed" note above and `handoff.md`'s "Fourth GPU
session." **Phase 4 (real paged-attention kernel)** is now the top
priority — more directly motivated than before, since Phase 2.5's own
data pinned the saturation-latency bottleneck down to admission-queue
depth (`max_num_seqs`), a resource a tiled kernel could plausibly free up
and chunking structurally can't touch (see Phase 4's section above).
**Phase 3 (real tensor parallelism)** still third — still valuable, the
most direct "Anthropic/OpenAI-scale inference" story, but the most
expensive and the lowest incremental learning ratio given existing real
ZeRO/FSDP/DDP experience, and (per the user, since Phase 4 is likely the
last phase attempted) the real finish-risk of a multi-GPU/distributed
debugging phase matters more now with no next phase to fall back to.
**Prefix-sharing / finishing `fork()`** is the one candidate for a fast
second phase after Phase 4 if time allows — real and bounded, though
partially pre-empted by an existing derivation elsewhere in the
portfolio (lower marginal learning value) — TP and disaggregation are
each standalone undertakings with their own finish-risk, not bonus-slot
material. **Phase 5 (real disaggregation)** last — highest novelty and
the most direct predict-then-validate story in this repo, but also the
highest time cost; only chase with real time to spare.

If time runs out partway through the stretch phases, stop after whichever
one just finished cleanly rather than leaving one half-done — a complete
Phase 4 is worth more than a half-built Phase 3 or 5, same logic as
before, just re-pointed at the new ordering.
