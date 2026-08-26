"""Phase 1: minimal ReAct-style tool-calling loop (spec-agent-pcie.md Phase
1) -- parse a tool call from the model's output, execute it, append the
result, repeat until a final answer or TURN_CAP is hit.

Design note on stopping: the base engine's Request only supports a single
`eos_token_id` (see serving_engine/request.py + scheduler.py's stop check),
which resolves to the tokenizer's one default eos token -- normally
`<|eot_id|>` (final-answer turns stop cleanly). A tool-call turn instead
ends in `<|eom_id|>`, which isn't that stop id, so generation runs to
max_new_tokens and produces a degenerate tail after the function tag
closes. Rather than touching the base engine's stop logic to support a set
of stop ids (out of scope for this project -- see spec's own scope note),
this is handled agent-side: parse_tool_call finds the first well-formed
`<function=...>` tag and everything after it is discarded, so the wasted
tail costs some decode compute but never reaches conversation history.
"""
from agent.prompt_format import build_prompt_ids, parse_tool_call
from agent.tools import TOOLS, TOOLS_BY_NAME

TURN_CAP = 6
MAX_NEW_TOKENS_PER_TURN = 256


class AgentSession:
    def __init__(self, engine_driver, tokenizer, system_prompt: str = None):
        self.driver = engine_driver
        self.tokenizer = tokenizer
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def run(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(TURN_CAP):
            prompt_ids = build_prompt_ids(self.tokenizer, self.messages, TOOLS)
            request_id = self.driver.submit(prompt_ids, MAX_NEW_TOKENS_PER_TURN)
            request = self.driver.wait_for(request_id)
            raw_text = self.tokenizer.decode(request.output_token_ids, skip_special_tokens=False)

            tool_call = parse_tool_call(raw_text)
            if tool_call is None:
                final_text = self.tokenizer.decode(request.output_token_ids, skip_special_tokens=True)
                self.messages.append({"role": "assistant", "content": final_text})
                return final_text

            name, args = tool_call
            call_end = raw_text.index("</function>") + len("</function>")
            self.messages.append({"role": "assistant", "content": raw_text[:call_end]})

            tool_fn = TOOLS_BY_NAME.get(name)
            result = tool_fn(**args) if tool_fn else f"[unknown tool: {name}]"
            self.messages.append({"role": "ipython", "content": result})

        return "[turn cap reached without a final answer]"
