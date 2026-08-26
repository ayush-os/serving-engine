"""Pure-Python, no-GPU checks for the Phase 1 agent harness -- mirrors
tests/test_workload.py's convention of unit-testing the parts that don't
need real model/CUDA (parsing, tool execution). The loop/engine-driving
parts need a real model and are exercised instead by
scripts/run_agent_sanity_check.py.
"""
from agent.prompt_format import build_prompt_ids, parse_tool_call
from agent.tools import execute_code, search_docs


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
