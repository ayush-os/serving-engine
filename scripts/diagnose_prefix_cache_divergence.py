"""Diagnostic for test_prefix_cache_matches_uncached's GPU divergence:
compares raw logits (not just decoded text) between the uncached and
cached paths at the first output token where they actually diverge, same
methodology as the existing xfail diagnostics in tests/test_correctness.py
(see handoff.md's "First GPU session"/"Fourth GPU session"). Run directly:

    python scripts/diagnose_prefix_cache_divergence.py

Prints, at the first diverging output token: both paths' argmax token,
top-5 tokens+logits, and max abs logit diff. A near-tied argmax with a
small max-abs-diff and an overlapping top-5 (just reordered) is the same
expected bf16/kernel-shape-non-determinism signature already established
for chunked prefill -- different batch composition (the cached path's real
request shares a step with the still-decoding warmup; the uncached path
never has anything else in its batch). Anything else -- a large diff, a
non-overlapping top-5, a clear one-sided argmax -- would be a real bug,
not noise.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from serving_engine.engine import LLMEngine
from serving_engine.request import RequestPhase

SHARED_PROMPT = (
    "You are a helpful, honest, and concise assistant. Always answer "
    "clearly and directly, without unnecessary padding. In machine "
    "learning, a transformer is"
)
MAX_NEW_TOKENS = 32


def run_and_capture_logits(build):
    """build() returns (engine, target_request_id) with the request(s)
    already admitted (and, for the cached case, already matched). Steps
    the engine to completion -- replicating LLMEngine.step()'s own logic
    directly rather than calling it, since step() doesn't expose logits --
    recording the target request's own per-output-token logits row at each
    step it appears in."""
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


def build_uncached():
    engine = LLMEngine(num_gpu_blocks=1024)
    rid = engine.add_request(SHARED_PROMPT, max_new_tokens=MAX_NEW_TOKENS)
    return engine, rid


def build_cached():
    engine = LLMEngine(num_gpu_blocks=1024)
    engine.add_request(SHARED_PROMPT, max_new_tokens=5)
    engine.step()
    rid = engine.add_request(SHARED_PROMPT, max_new_tokens=MAX_NEW_TOKENS)
    assert engine.requests[rid].num_computed_tokens > 0, "prefix match didn't fire"
    return engine, rid


def main():
    print("Running uncached path...")
    uncached_req, uncached_logits = run_and_capture_logits(build_uncached)
    print("Running cached path...")
    cached_req, cached_logits = run_and_capture_logits(build_cached)

    uncached_tokens = uncached_req.output_token_ids
    cached_tokens = cached_req.output_token_ids
    print(f"\nuncached: {uncached_tokens}")
    print(f"cached:   {cached_tokens}")

    diverge_idx = next(
        (i for i, (a, b) in enumerate(zip(uncached_tokens, cached_tokens)) if a != b),
        None,
    )
    if diverge_idx is None:
        print("\nNo divergence in output_token_ids -- outputs matched exactly.")
        return

    print(f"\nFirst diverging output token index: {diverge_idx}")
    logit_u = uncached_logits[diverge_idx].float()
    logit_c = cached_logits[diverge_idx].float()

    diff = (logit_u - logit_c).abs()
    print(f"max abs logit diff: {diff.max().item():.4f}")
    print(f"uncached argmax: {logit_u.argmax().item()} (logit {logit_u.max().item():.4f})")
    print(f"cached argmax:   {logit_c.argmax().item()} (logit {logit_c.max().item():.4f})")

    top5_u = torch.topk(logit_u, 5)
    top5_c = torch.topk(logit_c, 5)
    print(f"uncached top-5: {list(zip(top5_u.indices.tolist(), [round(v, 4) for v in top5_u.values.tolist()]))}")
    print(f"cached top-5:   {list(zip(top5_c.indices.tolist(), [round(v, 4) for v in top5_c.values.tolist()]))}")


if __name__ == "__main__":
    main()
