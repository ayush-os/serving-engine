"""Llama 3.1 native tool-calling format (spec-agent-pcie.md Phase 0 reading).

Built via HF's chat template rather than hand-rolled -- transformers>=4.43
already implements Llama 3.1's exact tool-calling conventions from a plain
list of typed Python functions (it derives each tool's JSON schema from
type hints + a Google-style "Args:" docstring). Custom (non-built-in)
tools are emitted by the model as `<function=name>{json args}</function>`,
followed by `<|eom_id|>` if more turns are expected or `<|eot_id|>` if the
model considers this a final answer -- see agent/loop.py for why this
module only checks for the `<function=...>` tag rather than relying on
that token distinction.
"""
import json
import re

TOOL_CALL_RE = re.compile(r"<function=(\w+)>(\{.*?\})</function>", re.DOTALL)


def build_prompt_ids(tokenizer, messages, tools):
    """Returns a flat List[int], regardless of which shape this transformers
    version's apply_chat_template hands back -- observed in practice to be a
    BatchEncoding (not the plain List[int] the docs describe) once `tools`
    is passed, so this normalizes rather than trusting one shape."""
    encoded = tokenizer.apply_chat_template(
        messages, tools=tools, add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def parse_tool_call(text: str):
    """Returns (name, args_dict) if `text` contains a well-formed custom
    tool call, else None -- meaning the caller should treat `text` as the
    agent's final answer instead."""
    match = TOOL_CALL_RE.search(text)
    if match is None:
        return None
    name, raw_args = match.groups()
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        return None
    return name, args
