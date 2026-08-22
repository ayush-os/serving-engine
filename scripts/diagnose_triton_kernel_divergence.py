"""Diagnostic for test_triton_kernel_matches_eager's fibonacci-prompt xfail:
compares raw logits (not just decoded text) between the eager and triton
attention paths at the first output token where they actually diverge, same
methodology as scripts/diagnose_prefix_cache_divergence.py and the existing
xfail diagnostics in tests/test_correctness.py (see handoff.md's "First GPU
session"/"Fourth GPU session"). Run directly:

    python scripts/diagnose_triton_kernel_divergence.py

Prints, at the first diverging output token: both paths' argmax token,
top-5 tokens+logits, and max abs logit diff. A near-tied argmax with a
small max-abs-diff and an overlapping top-5 (just reordered) is the same
expected bf16/kernel-shape-non-determinism signature already established
for every other xfail in this file -- different attention implementations
computing the same softmax in a different reduction order. Anything else --
a large diff, a non-overlapping top-5, a clear one-sided argmax, or
divergence at the very first token (prefill, not decode) -- would point at
a real bug in the new kernel, not noise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from serving_engine.engine import LLMEngine
from serving_engine.request import RequestPhase

PROMPT = "def fibonacci(n):"
MAX_NEW_TOKENS = 32


def run_and_capture_logits(build):
    """build() returns (engine, target_request_id) with the request already
    admitted. Steps the engine to completion -- replicating LLMEngine.step()'s
    own logic directly rather than calling it, since step() doesn't expose
    logits -- recording the target request's own per-output-token logits row
    at each step it appears in."""
    engine, target_id = build()
    target = engine.requests[target_id]
    logits_by_output_idx = []

    while engine.scheduler.has_unfinished_requests():
        scheduler_output = engine.scheduler.schedule()
        if not scheduler_output.scheduled_requests:
            continue
        logits = engine.model_runner.forward(scheduler_output)
        next_tokens = logits.argmax(dim=-1)
        for i, (req, tok) in enumerate(zip(scheduler_output.scheduled_requests, next_tokens)):
            if req.phase == RequestPhase.NEEDS_DECODE or req.request_id in scheduler_output.prefill_final_chunk:
                if req is target:
                    logits_by_output_idx.append(logits[i].detach().clone())
                req.output_token_ids.append(tok.item())
        engine.scheduler.update_after_step(scheduler_output)

    return target, logits_by_output_idx


def build_eager():
    engine = LLMEngine(num_gpu_blocks=1024, attn_implementation="paged|eager")
    rid = engine.add_request(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
    return engine, rid


def build_triton():
    engine = LLMEngine(num_gpu_blocks=1024)  # default: attn_implementation="paged|triton"
    rid = engine.add_request(PROMPT, max_new_tokens=MAX_NEW_TOKENS)
    return engine, rid


def main():
    print("Running eager path...")
    eager_req, eager_logits = run_and_capture_logits(build_eager)
    print("Running triton path...")
    triton_req, triton_logits = run_and_capture_logits(build_triton)

    eager_tokens = eager_req.output_token_ids
    triton_tokens = triton_req.output_token_ids
    print(f"\neager:  {eager_tokens}")
    print(f"triton: {triton_tokens}")

    diverge_idx = next(
        (i for i, (a, b) in enumerate(zip(eager_tokens, triton_tokens)) if a != b),
        None,
    )
    if diverge_idx is None:
        print("\nNo divergence in output_token_ids -- outputs matched exactly.")
        return

    print(f"\nFirst diverging output token index: {diverge_idx}")
    logit_e = eager_logits[diverge_idx].float()
    logit_t = triton_logits[diverge_idx].float()

    diff = (logit_e - logit_t).abs()
    print(f"max abs logit diff: {diff.max().item():.4f}")
    print(f"eager argmax:  {logit_e.argmax().item()} (logit {logit_e.max().item():.4f})")
    print(f"triton argmax: {logit_t.argmax().item()} (logit {logit_t.max().item():.4f})")

    top5_e = torch.topk(logit_e, 5)
    top5_t = torch.topk(logit_t, 5)
    print(f"eager top-5:  {list(zip(top5_e.indices.tolist(), [round(v, 4) for v in top5_e.values.tolist()]))}")
    print(f"triton top-5: {list(zip(top5_t.indices.tolist(), [round(v, 4) for v in top5_t.values.tolist()]))}")


if __name__ == "__main__":
    main()
