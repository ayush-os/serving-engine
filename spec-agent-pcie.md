# Project Spec: A Real Agent Driving the Serving Engine — Closing the PCIe/Host-Overhead Question

**Continuity note.** Two open threads, from two different points in this
body of work, converge here. First: every benchmark run against the
serving engine so far — Phase 2's load sweep, Phase 2.5's chunking rerun,
the prefix-caching demo, the paged-kernel comparison — used `workload.py`'s
synthetic Poisson/lognormal request generator. Nothing has ever driven the
engine with real, organic, data-dependent traffic. Second: Phase 2's own
handoff notes flagged a real, unexplained finding and left it open —
SM utilization saturates at ~89-90%, never higher, with `model_runner.
forward()`'s synchronous Python-side construction of `write_idxes`/
`read_idxes`/`position_ids`/the attention mask named as a "plausible,
honest candidate," explicitly **not confirmed**. This project closes both:
build a real tool-calling agent whose actual multi-turn, bursty traffic
drives the engine, and use that real load — plus targeted profiling — to
finally answer the standing hypothesis instead of leaving it as a guess.

**Why this, not another synthetic benchmark.** A real agent session has
structure no Poisson generator has: turns share a growing prefix (direct,
real exercise for the prefix-caching work, under organic traffic instead
of the synthetic shared-prefix demo), tool-call latency creates real idle
gaps between a session's requests (bursty, not memoryless), and turn count
and prompt growth are data-dependent, not drawn from a fixed distribution.
This is also the first project in this whole body of work where every
prior piece — the engine, the kernel, prefix caching, the profiling
methodology from `LLM_Architecture___Distributed_Training_Systems__From_Scratch`
— runs as one real system instead of parallel artifacts.

**Toolchain:** same model/engine (Llama-3.1-8B-Instruct, your serving
engine, unchanged) — Nsight Systems (`nsys`) for full-stack tracing,
reusing the NVTX-range methodology already used to profile DDP/attention;
`py-spy` or `torch.profiler` for host-side attribution; `nvidia-smi dmon`
for real PCIe transfer measurement; a small set of real tools (code
execution, simple retrieval) for the agent itself.

**Legend:** 🔧 = boilerplate/setup. 🧠 = a real decision.

---

## Phase 0 — Setup (🔧 light, three real 🧠 decisions)

### Reading (🔧)

- Llama 3.1's native tool-calling chat-template format — use it directly
  rather than inventing a structured-output convention; it's already
  trained into the model you're serving.
- Light ReAct-style agent-loop pattern (parse tool call → execute →
  append result → repeat) — implementation pattern, not a research
  problem, keep this reading brief.
- Nsight Systems' PCIe trace capture and CUDA-API timeline correlation —
  direct continuity with the profiling methodology already used and
  understood from the distributed-training project, not fresh territory.

### 🧠 Decision 1: tool set — bounded, just enough for real variability

Two or three real tools, not an agent framework: sandboxed code
execution (subprocess, real timeout) and a simple local retrieval/file-
search tool are enough to produce genuine multi-turn variability (turn
count, prompt growth, tool-latency gaps) without the project becoming
"build an agent product." The traffic shape is the point, not the agent's
capability.

### 🧠 Decision 2: traffic topology — concurrent sessions, not one sequential agent

Run N concurrent, independent agent sessions (each a simulated "user"
running a real multi-step task) against the engine at once — this is
what real production traffic actually looks like (many users, each
multi-turn), and it's what actually stresses the scheduler/prefix-cache
under realistic concurrency, unlike a single sequential agent which would
just be one more synthetic-shaped benchmark.

### 🧠 Decision 3: profiling scope — targeted at the standing hypothesis, not a general sweep

Phase 3 profiles `model_runner.forward()`'s index/mask construction
specifically, because there's already a real, specific hypothesis on
record to check — not an untargeted profiling fishing expedition. State
this now so Phase 3 doesn't drift into open-ended profiling.

**Checkpoint:** tool set, traffic topology, and profiling target all
decided and stated before writing the agent loop.

---

## Phase 1 — Build the agent harness (🔧 mostly, one real 🧠 check)

- Minimal tool-calling loop: system prompt with tool schema (Llama 3.1's
  native format), parse a tool call from model output, execute it, append
  the result to the conversation, continue until a final answer or a turn
  cap.
- Real tools per Decision 1, actually executing (not stubbed).
- **Real check, not assumed working**: run a handful of genuine multi-step
  tasks single-session first (e.g. "compute X using code," "look up Y
  then summarize") and confirm the loop terminates correctly and tool
  results are actually incorporated into the model's next turn — a
  sanity pass, not a deep correctness project, before this harness is
  trusted to generate load.

**Checkpoint:** a working single-session agent loop, confirmed correct on
a handful of real tasks, ready to run concurrently.

---

## Phase 2 — Real concurrent agent traffic vs. synthetic (🧠 the first real comparison)

- Run N concurrent agent sessions against the engine, logging real request
  timestamps, per-turn prompt lengths, and inter-request gaps (tool-call
  latency shows up here directly).
- **The real comparison this phase exists for**: how does this traffic's
  actual distributional shape — inter-arrival pattern, prompt growth,
  burst structure — compare to `workload.py`'s Poisson/lognormal
  assumptions? A genuine, checked answer: does the synthetic generator
  every prior benchmark relied on actually resemble real traffic, or was
  it a convenient fiction? Either answer is a real, reportable finding.
- **Real benchmark**: throughput/TTFT/decode-latency under this organic
  traffic vs. synthetic traffic at a comparable aggregate load, same
  metrics and harness as every prior serving-engine phase.
- **Prefix caching gets its first real exercise**: multi-turn sessions
  naturally share growing prefixes. Measure the real cache hit rate under
  organic concurrent traffic and compare it to the synthetic shared-
  prefix demo's clean 15/15 hit rate — real traffic is messier; does the
  mechanism still deliver a real win, and how much smaller/larger is it?

**Checkpoint:** real agent traffic's shape compared explicitly against
the synthetic generator's assumptions; a real throughput/TTFT comparison;
a real prefix-cache hit rate under organic load.

---

## Phase 3 — Closing the PCIe/host-overhead question (🧠 the payoff)

- Under real agent load (or a controlled, matched synthetic load if that
  gives cleaner signal — state which and why), capture an `nsys` trace of
  the engine's step loop.
- **Directly test the standing hypothesis**: time `model_runner.forward()`'s
  Python-side index/mask construction specifically (NVTX range or direct
  wall-clock), and compare it against total step time and against the
  ~10-11% SM-utilization gap already measured. Confirmed or refuted —
  either is a real answer to a question that's been open since Phase 2.
- **Check real PCIe transfer behavior**: `nvidia-smi dmon --pcie` (or
  `nsys`'s own transfer trace) during a load run — are these index
  tensors genuinely re-transferred host→device every single step, and
  how much of the gap does that transfer volume/latency actually explain?
- **If the hypothesis holds and a fix is tractable within scope**:
  implement one (e.g. constructing indices with GPU-resident torch ops
  instead of Python lists + a fresh H2D copy each step, or pinning host
  memory for faster transfer) and measure whether the SM-utilization gap
  actually closes — a real fix-and-validate, not a diagnosis left hanging.

**Checkpoint:** the Phase 2 hypothesis confirmed or refuted with real
profiling data; if confirmed and fixable in scope, a real measured
before/after on the utilization gap.

---

## Phase 4 — Synthesis (🧠, light)

The full closed loop, stated plainly: a real agent, generating real
traffic, run against your own serving engine, profiled with real tools,
closing a real question your own prior work left open. This is the
strongest single artifact to lead with in an interview from this entire
project sequence — not because it's the most novel individual mechanism,
but because it's the only place everything else actually runs together.

---

## Note on scope

Resist two real temptations:

1. **Building a general-purpose agent framework or product.** The agent
   exists to generate realistic traffic — two or three real tools and a
   minimal loop are enough. Don't harden retries, add more tools, or
   chase agent capability for its own sake.
2. **An untargeted profiling sweep of the whole engine.** Decision 3
   already scoped Phase 3 to the specific standing hypothesis. If
   something else interesting turns up while profiling, note it — don't
   chase it into a second investigation inside this project.

## Fallback

Phase 1-2 alone — a real agent generating real, organic, concurrent
traffic against the engine, compared explicitly against the synthetic
generator every prior benchmark relied on — is already a complete, novel
artifact: the first real end-to-end system in this entire body of work.
Phase 3 (closing the PCIe question) is the natural next step and the
strongest individual finding if it lands. Phase 4 is light synthesis, not
a required deliverable on its own.