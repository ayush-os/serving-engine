from serving_engine.workload import generate_workload


def test_arrival_times_within_duration_and_sorted():
    workload = generate_workload(duration=60.0, arrival_rate=5.0, seed=0)

    assert len(workload) > 0
    times = [r.arrival_time for r in workload]
    assert times == sorted(times)
    assert all(0 < t <= 60.0 for t in times)


def test_arrival_rate_matches_poisson_rate_on_average():
    # Long enough window that count/duration should land close to the
    # target rate -- exponential inter-arrival gaps averaging 1/rate is
    # exactly what makes this a Poisson process with that rate.
    duration = 2000.0
    rate = 3.0
    workload = generate_workload(duration=duration, arrival_rate=rate, seed=1)

    observed_rate = len(workload) / duration
    assert abs(observed_rate - rate) / rate < 0.1


def test_lengths_average_near_target_means():
    workload = generate_workload(
        duration=5000.0, arrival_rate=2.0, prompt_mean=512, output_mean=64, seed=2
    )

    mean_prompt = sum(r.prompt_len for r in workload) / len(workload)
    mean_output = sum(r.output_len for r in workload) / len(workload)

    # Lognormal sampling + rounding won't hit the target mean exactly --
    # 15% tolerance over several thousand samples is loose enough to not be
    # flaky while still catching a real parameterization bug (e.g. mu/sigma
    # swapped, or the mean/median conflated).
    assert abs(mean_prompt - 512) / 512 < 0.15
    assert abs(mean_output - 64) / 64 < 0.15


def test_lengths_always_at_least_one_token():
    workload = generate_workload(duration=3000.0, arrival_rate=2.0, seed=3)

    assert all(r.prompt_len >= 1 and r.output_len >= 1 for r in workload)


def test_seed_reproducible():
    a = generate_workload(duration=100.0, arrival_rate=4.0, seed=42)
    b = generate_workload(duration=100.0, arrival_rate=4.0, seed=42)

    assert a == b


def test_different_seeds_differ():
    a = generate_workload(duration=100.0, arrival_rate=4.0, seed=1)
    b = generate_workload(duration=100.0, arrival_rate=4.0, seed=2)

    assert a != b


def test_empty_workload_when_duration_too_short_for_rate():
    # Not a crash case -- benchmark_load.py's per-rate loop must tolerate
    # an empty workload from a very low rate / short duration combination.
    workload = generate_workload(duration=0.0001, arrival_rate=0.001, seed=4)
    assert workload == []
