"""Synthetic load generator for Phase 2 benchmarking -- matches
disagg_and_placement_notes.md Sec 0's distributional assumptions (Poisson
arrivals, lognormal request shape anchored to DistServe's own reported
input~=512/output~=64 token averages) so throughput/latency numbers coming
out of scripts/benchmark_load.py are comparable to that simulator's
predictions, not a different workload shape.

Pure Python (random module only, no numpy/torch) so it's unit-testable
without a GPU -- see tests/test_workload.py.
"""
import math
import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SyntheticRequest:
    arrival_time: float  # seconds, relative to the start of this workload
    prompt_len: int       # tokens
    output_len: int       # tokens -- benchmark_load.py forces exactly this many via ignore_eos


def _lognormal_params(mean: float, cv: float) -> tuple:
    """mu, sigma for a lognormal with the given mean and coefficient of
    variation. The notes give only mean targets (input~=512, output~=64),
    not a shape -- cv=1.0 (below) is the free parameter picked as a
    standard heavy-tailed default for request-length traffic: mostly short
    requests, a long tail of long ones, matching real chat workloads."""
    sigma2 = math.log(1 + cv ** 2)
    mu = math.log(mean) - sigma2 / 2
    return mu, math.sqrt(sigma2)


def generate_workload(
    duration: float,
    arrival_rate: float,
    prompt_mean: float = 512,
    output_mean: float = 64,
    length_cv: float = 1.0,
    seed: Optional[int] = None,
) -> List[SyntheticRequest]:
    """Poisson arrivals (rate=arrival_rate req/s) over `duration` seconds of
    wall clock, each with an independently-sampled lognormal prompt/output
    length. Exponential inter-arrival gaps are what makes this a Poisson
    process, not the sampling loop itself."""
    rng = random.Random(seed)
    prompt_mu, prompt_sigma = _lognormal_params(prompt_mean, length_cv)
    output_mu, output_sigma = _lognormal_params(output_mean, length_cv)

    requests = []
    t = 0.0
    while True:
        t += rng.expovariate(arrival_rate)
        if t > duration:
            break
        prompt_len = max(1, round(rng.lognormvariate(prompt_mu, prompt_sigma)))
        output_len = max(1, round(rng.lognormvariate(output_mu, output_sigma)))
        requests.append(SyntheticRequest(arrival_time=t, prompt_len=prompt_len, output_len=output_len))
    return requests
