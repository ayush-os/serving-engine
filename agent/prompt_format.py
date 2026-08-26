"""Tool-calling prompt format for this environment's Llama 3.1 chat
template (spec-agent-pcie.md Phase 0 reading).

Built via HF's chat template rather than hand-rolled -- transformers
derives each tool's JSON schema from a typed Python function's signature
and Google-style "Args:" docstring. Real finding from the first GPU run,
correcting the Phase 0 assumption: Meta's own docs describe custom tool
calls as `<function=name>{json}</function>`, but the chat_template bundled
with this specific tokenizer checkout instead glues the tool schema into
the user turn and instructs a bare-JSON reply, `{"name": ..., "parameters":
{...}}`, with no distinct end-of-call token (no `<|eom_id|>` in this
template at all -- see agent/loop.py for why parsing truncates at the
JSON's own closing brace rather than a special token).
"""
import json


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
    """Returns (name, args_dict, end_index) if `text` contains a
    well-formed {"name": ..., "parameters": {...}} tool call, else None --
    meaning the caller should treat `text` as the agent's final answer
    instead. end_index is text's offset immediately after the JSON object,
    so callers can discard anything generated after it (this template
    gives no distinct stop signal for "tool call, more to come" -- the
    model sometimes keeps rambling, even hallucinating a further turn).

    Uses JSONDecoder.raw_decode rather than a regex: the JSON is
    unbounded/nested (parameters can be any shape), so a regex can't
    reliably find its true closing brace.
    """
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        if isinstance(obj, dict) and "name" in obj and "parameters" in obj:
            return obj["name"], obj["parameters"], end
        idx = text.find("{", idx + 1)
    return None
