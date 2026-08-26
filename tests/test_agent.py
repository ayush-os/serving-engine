"""Pure-Python, no-GPU checks for the Phase 1 agent harness -- mirrors
tests/test_workload.py's convention of unit-testing the parts that don't
need real model/CUDA (parsing, tool execution). The loop/engine-driving
parts need a real model and are exercised instead by
scripts/run_agent_sanity_check.py.
"""
from agent.prompt_format import parse_tool_call
from agent.tools import execute_code, search_docs


def test_parse_tool_call_extracts_name_and_args():
    text = 'Sure, let me check.<function=search_docs>{"query": "prefix caching"}</function>'
    result = parse_tool_call(text)
    assert result == ("search_docs", {"query": "prefix caching"})


def test_parse_tool_call_returns_none_for_plain_text():
    assert parse_tool_call("The answer is 42.") is None


def test_parse_tool_call_returns_none_for_malformed_json():
    text = "<function=search_docs>{not valid json}</function>"
    assert parse_tool_call(text) is None


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
