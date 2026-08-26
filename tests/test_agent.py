"""Pure-Python, no-GPU checks for the Phase 1 agent harness -- mirrors
tests/test_workload.py's convention of unit-testing the parts that don't
need real model/CUDA (parsing, tool execution). The loop/engine-driving
parts need a real model and are exercised instead by
scripts/run_agent_sanity_check.py.
"""
from agent.engine_driver import EngineDriver
from agent.loop import AgentSession
from agent.prompt_format import build_prompt_ids, parse_tool_call
from agent.tools import execute_code, search_docs


class _FakeRequest:
    def __init__(self, request_id, arrival_time=0.0, is_finished=False,
                 num_computed_tokens=0, prompt_len=10, output_token_ids=None):
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.is_finished = is_finished
        self.num_computed_tokens = num_computed_tokens
        self.prompt_len = prompt_len
        self.output_token_ids = output_token_ids or []


class _FakeSchedulerOutput:
    def __init__(self, scheduled_requests):
        self.scheduled_requests = scheduled_requests


class _FakeEngine:
    """Minimal stand-in for LLMEngine's request-lookup surface -- enough for
    EngineDriver.submit()'s locking/return-shape, without a real model."""

    def __init__(self):
        self.requests = {}
        self._next_id = 0

    def add_request(self, prompt_token_ids, max_new_tokens):
        request_id = f"req-{self._next_id}"
        self._next_id += 1
        self.requests[request_id] = _FakeRequest(request_id, num_computed_tokens=17)
        return request_id


class _FakeDriver:
    """Stands in for EngineDriver from AgentSession's point of view: submit
    returns an incrementing request id with a fixed match count, wait_for
    returns a finished fake Request whose output decodes (via the fake
    tokenizer below) to a plain final answer -- enough to exercise
    AgentSession's turn_log bookkeeping without a real engine or model."""

    def __init__(self):
        self._next_id = 0

    def submit(self, prompt_token_ids, max_new_tokens):
        request_id = f"req-{self._next_id}"
        self._next_id += 1
        return request_id, 0

    def wait_for(self, request_id):
        return _FakeRequest(request_id, is_finished=True, prompt_len=5, output_token_ids=[1, 2, 3])


class _FakeSessionTokenizer:
    """apply_chat_template's actual output doesn't matter for AgentSession
    bookkeeping tests -- only that decode() returns plain text with no tool
    call, so every run() call resolves as an immediate final answer."""

    def apply_chat_template(self, messages, tools, add_generation_prompt):
        return [1, 2, 3]

    def decode(self, output_token_ids, skip_special_tokens):
        return "final answer"


class _FakeTokenizer:
    """Stubs the three response shapes apply_chat_template has been
    observed to return across transformers versions/args, so
    build_prompt_ids's normalization is testable without a real model."""

    def __init__(self, to_return):
        self._to_return = to_return

    def apply_chat_template(self, messages, tools, add_generation_prompt):
        return self._to_return


class _FakeBatchEncoding:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def test_parse_tool_call_extracts_name_args_and_end_index():
    text = '{"name": "search_docs", "parameters": {"query": "prefix caching"}} extra text'
    name, args, end = parse_tool_call(text)
    assert (name, args) == ("search_docs", {"query": "prefix caching"})
    assert text[:end] == '{"name": "search_docs", "parameters": {"query": "prefix caching"}}'


def test_parse_tool_call_ignores_rambling_after_the_call():
    """Real failure mode seen on-box: the model doesn't reliably stop after
    a tool call and rambles on, sometimes re-emitting further bogus calls.
    Only the first well-formed call should be returned."""
    text = (
        '{"name": "search_docs", "parameters": {"query": "x"}}; '
        '{"name": "execute_code", "parameters": {"code": "print(0.5)"}}'
    )
    name, args, _ = parse_tool_call(text)
    assert (name, args) == ("search_docs", {"query": "x"})


def test_parse_tool_call_returns_none_for_plain_text():
    assert parse_tool_call("The answer is 42.") is None


def test_parse_tool_call_returns_none_for_malformed_json():
    text = "{not valid json}"
    assert parse_tool_call(text) is None


def test_parse_tool_call_ignores_json_without_expected_keys():
    assert parse_tool_call('{"foo": "bar"}') is None


def test_execute_code_captures_stdout():
    output = execute_code("print(2 + 2)")
    assert output == "4"


def test_execute_code_reports_nonzero_exit():
    output = execute_code("raise ValueError('boom')")
    assert "boom" in output
    assert "exit code" in output


def test_search_docs_finds_real_content_in_repo():
    result = search_docs("prefix caching hit rate")
    assert "no matches" not in result
    first_line = result.splitlines()[0]
    assert first_line.startswith("From ") and ".md" in first_line


def test_build_prompt_ids_unwraps_plain_list():
    tokenizer = _FakeTokenizer([1, 2, 3])
    assert build_prompt_ids(tokenizer, [], []) == [1, 2, 3]


def test_build_prompt_ids_unwraps_batched_list():
    tokenizer = _FakeTokenizer([[1, 2, 3]])
    assert build_prompt_ids(tokenizer, [], []) == [1, 2, 3]


def test_build_prompt_ids_unwraps_batch_encoding():
    tokenizer = _FakeTokenizer(_FakeBatchEncoding([1, 2, 3]))
    assert build_prompt_ids(tokenizer, [], []) == [1, 2, 3]


def test_engine_driver_submit_returns_id_and_matched_tokens():
    driver = EngineDriver(_FakeEngine())
    request_id, matched = driver.submit(prompt_token_ids=[1, 2, 3], max_new_tokens=16)
    assert matched == 17
    assert driver.engine.requests[request_id].num_computed_tokens == 17


def test_engine_driver_record_step_classifies_pure_prefill_then_decode():
    driver = EngineDriver(_FakeEngine())
    req_a = _FakeRequest("a", arrival_time=100.0)
    req_b = _FakeRequest("b", arrival_time=100.0)

    driver._record_step(_FakeSchedulerOutput([req_a, req_b]), step_dt=0.05)
    assert driver.step_kind_count["pure_prefill"] == 1
    assert set(driver.ttft.keys()) == {"a", "b"}

    req_a.is_finished = True
    driver._record_step(_FakeSchedulerOutput([req_a, req_b]), step_dt=0.02)
    assert driver.step_kind_count["pure_decode"] == 1
    assert "a" in driver.completion_time
    assert "b" not in driver.completion_time
    assert driver.step_count == 2


def test_engine_driver_record_step_classifies_mixed():
    driver = EngineDriver(_FakeEngine())
    old = _FakeRequest("old", arrival_time=0.0)
    driver._record_step(_FakeSchedulerOutput([old]), step_dt=0.01)  # old is now "seen"

    new = _FakeRequest("new", arrival_time=0.0)
    driver._record_step(_FakeSchedulerOutput([old, new]), step_dt=0.03)
    assert driver.step_kind_count["mixed"] == 1


def test_agent_session_turn_log_records_expected_fields():
    session = AgentSession(_FakeDriver(), _FakeSessionTokenizer(), session_id="s0")
    answer = session.run("first question")

    assert answer == "final answer"
    assert len(session.turn_log) == 1
    record = session.turn_log[0]
    assert record["session_id"] == "s0"
    assert record["task_index"] == 0
    assert record["react_turn"] == 0
    assert record["is_final_answer"] is True
    assert record["tool_name"] is None
    assert record["prompt_len"] == 5


def test_agent_session_task_index_increments_across_run_calls():
    session = AgentSession(_FakeDriver(), _FakeSessionTokenizer(), session_id="s0")
    session.run("first question")
    session.run("follow-up question")

    assert [r["task_index"] for r in session.turn_log] == [0, 1]
    # conversation grows across calls: 2 user + 2 assistant messages
    assert len(session.messages) == 4
