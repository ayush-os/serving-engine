"""Concurrency-ready plumbing (spec-agent-pcie.md Decision 2): one
background thread drives engine.step() continuously; any number of agent
sessions submit requests and poll for completion. Phase 1 uses this with a
single session; Phase 2 points N concurrent sessions at the same
EngineDriver so they land in the same scheduler/prefix-cache, which is the
entire point of the concurrent-sessions decision -- a driver-per-session
design would just be N independent sequential benchmarks again.
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
                    self.engine.step()
            if not has_work:
                time.sleep(self._poll_interval)

    def submit(self, prompt_token_ids, max_new_tokens: int) -> str:
        with self._lock:
            return self.engine.add_request(
                prompt_token_ids=prompt_token_ids, max_new_tokens=max_new_tokens,
            )

    def wait_for(self, request_id: str) -> Request:
        while True:
            request = self.engine.requests[request_id]
            if request.is_finished:
                return request
            time.sleep(self._poll_interval)
