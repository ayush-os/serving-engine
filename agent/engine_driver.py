"""Concurrency-ready plumbing (spec-agent-pcie.md Decision 2): one
background thread drives engine.step() continuously; any number of agent
sessions submit requests and poll for completion. Phase 1 used this with a
single session; Phase 2 points N concurrent sessions at the same
EngineDriver so they land in the same scheduler/prefix-cache, which is the
entire point of the concurrent-sessions decision -- a driver-per-session
design would just be N independent sequential benchmarks again.

Phase 2 addition: per-step instrumentation (TTFT, step-kind timing) copied
from scripts/benchmark_load.py's run_rate() -- same methodology as every
prior serving-engine phase, so a real-traffic run's throughput/TTFT/
decode-latency numbers are directly comparable to a synthetic
benchmark_load.py run rather than being a different measurement entirely.
"""
import threading
import time

from serving_engine.request import Request


class EngineDriver:
    def __init__(self, engine, poll_interval_s: float = 0.002):
        self.engine = engine
        self._lock = threading.Lock()
        self._poll_interval = poll_interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run_step_loop, daemon=True)

        # Per-step instrumentation, same definitions as benchmark_load.py's
        # run_rate(): TTFT is wall time from a request's real arrival to the
        # first step that scheduled it; a step is classified by whether
        # every/some/no scheduled request was new this step.
        self._seen_ids = set()
        self.ttft = {}
        self.completion_time = {}
        self.step_kind_time = {"pure_prefill": 0.0, "pure_decode": 0.0, "mixed": 0.0}
        self.step_kind_count = {"pure_prefill": 0, "pure_decode": 0, "mixed": 0}
        self.step_count = 0

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _run_step_loop(self):
        while not self._stop.is_set():
            with self._lock:
                has_work = self.engine.scheduler.has_unfinished_requests()
                if has_work:
                    step_t0 = time.monotonic()
                    scheduler_output = self.engine.step()
                    step_dt = time.monotonic() - step_t0
                    self._record_step(scheduler_output, step_dt)
            if not has_work:
                time.sleep(self._poll_interval)

    def _record_step(self, scheduler_output, step_dt: float):
        num_new = num_old = 0
        for req in scheduler_output.scheduled_requests:
            if req.request_id not in self._seen_ids:
                num_new += 1
                self.ttft[req.request_id] = time.monotonic() - req.arrival_time
            else:
                num_old += 1
            self._seen_ids.add(req.request_id)
            if req.is_finished:
                self.completion_time[req.request_id] = time.monotonic()

        if num_new and num_old:
            kind = "mixed"
        elif num_new:
            kind = "pure_prefill"
        elif num_old:
            kind = "pure_decode"
        else:
            kind = None
        if kind is not None:
            self.step_kind_time[kind] += step_dt
            self.step_kind_count[kind] += 1
        self.step_count += 1

    def submit(self, prompt_token_ids, max_new_tokens: int):
        """Returns (request_id, matched_tokens_at_admission). The match
        count must be read inside the same locked section as add_request --
        BlockManager.match_prefix runs synchronously inside it, but the
        background step loop could start advancing num_computed_tokens
        further the instant the lock is released, so reading it afterward
        would race."""
        with self._lock:
            request_id = self.engine.add_request(
                prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens,
            )
            matched_tokens = self.engine.requests[request_id].num_computed_tokens
        return request_id, matched_tokens

    def wait_for(self, request_id: str) -> Request:
        while True:
            request = self.engine.requests[request_id]
            if request.is_finished:
                return request
            time.sleep(self._poll_interval)
