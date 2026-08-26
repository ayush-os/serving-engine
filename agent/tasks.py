"""Phase 2 task library: each entry is one session's sequence of top-level
user messages (a realistic multi-turn conversation, not just one question).
Deliberately a small fixed library, cycled across N sessions -- variety
across code-exec/retrieval/follow-ups is what's needed to produce real
turn-count and prompt-growth variability (spec-agent-pcie.md Decision 1),
not a large or clever task set.
"""

TASK_SEQUENCES = [
    [
        "What is 1837 * 429? Use code to compute it, don't do the arithmetic yourself.",
        "Now compute that same product mod 1000 using code.",
    ],
    [
        "Compute the 20th Fibonacci number (0-indexed) using code.",
        "Now do the same for the 25th Fibonacci number.",
    ],
    [
        "Search the project docs for what the paged Triton kernel's throughput "
        "improvement was over eager attention, then tell me the numbers.",
        "Now search the docs for what TTFT improvement chunked prefill found.",
    ],
    [
        "Search the docs for the KV cache block size used in this project.",
        "Using that block size, compute with code how many blocks a 10000-token "
        "sequence would need.",
    ],
    [
        "Compute 12 factorial using code.",
        "Now compute 15 factorial using code.",
        "Using code, compute the ratio of the second result to the first.",
    ],
    [
        "Search the project docs for what caused the SM utilization ceiling "
        "found in this project's load testing.",
        "Based on what you found, summarize the leading hypothesis in one sentence.",
    ],
]
