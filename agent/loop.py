"""Phase 1: minimal ReAct-style tool-calling loop (spec-agent-pcie.md Phase
1) -- parse a tool call from the model's output, execute it, append the
result, repeat until a final answer or TURN_CAP is hit.

Design note on stopping: this template's `eos_token_id` correctly resolves
to `<|eot_id|>` (confirmed on-box), but the model doesn't reliably emit it
right after a tool call -- observed rambling into a hallucinated second
turn instead of stopping (see prompt_format.py's docstring: this template
has no distinct "tool call, more to come" token to rely on either). Rather
than touching the base engine's single-`eos_token_id` stop logic (out of
scope -- see spec's own scope note), this is handled agent-side:
parse_tool_call finds the first well-formed JSON call and everything
generated after it is discarded, so a runaway tail costs some wasted
decode compute but never reaches conversation history.

Phase 2 addition: session_id + turn_log. A session's run() can be called
multiple times with different user_messages (self.messages persists across
calls, so each call is a real follow-up in the same growing conversation,
not a fresh one) -- exactly the "turns share a growing prefix" traffic
shape Phase 2 wants. Every request this session submits (each internal
ReAct turn, across every top-level run() call) appends one record to
turn_log with everything Phase 2's checkpoint needs: real arrival time,
prompt length, and the prefix-cache match at admission -- so the
orchestrator can compute real throughput/TTFT/hit-rate numbers without
touching AgentSession internals itself.
"""
from agent.prompt_format import build_prompt_ids, parse_tool_call
from agent.tools import TOOLS, TOOLS_BY_NAME

TURN_CAP = 6
MAX_NEW_TOKENS_PER_TURN = 256


class AgentSession:
    def __init__(self, engine_driver, tokenizer, session_id: str, system_prompt: str = None):
        self.driver = engine_driver
        self.tokenizer = tokenizer
        self.session_id = session_id
        self.messages = []
        self.turn_log = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def run(self, user_message: str) -> str:
        task_index = sum(1 for m in self.messages if m["role"] == "user")
        self.messages.append({"role": "user", "content": user_message})

        for react_turn in range(TURN_CAP):
            prompt_ids = build_prompt_ids(self.tokenizer, self.messages, TOOLS)
            request_id, matched_tokens = self.driver.submit(prompt_ids, MAX_NEW_TOKENS_PER_TURN)
            request = self.driver.wait_for(request_id)
            raw_text = self.tokenizer.decode(request.output_token_ids, skip_special_tokens=False)

            tool_call = parse_tool_call(raw_text)
            is_final = tool_call is None
            self.turn_log.append({
                "session_id": self.session_id,
                "task_index": task_index,
                "react_turn": react_turn,
                "request_id": request_id,
                "arrival_time": request.arrival_time,
                "prompt_len": request.prompt_len,
                "matched_tokens_at_admission": matched_tokens,
                "tool_name": None if is_final else tool_call[0],
                "is_final_answer": is_final,
            })

            if is_final:
                final_text = self.tokenizer.decode(request.output_token_ids, skip_special_tokens=True)
                self.messages.append({"role": "assistant", "content": final_text})
                return final_text

            name, args, call_end = tool_call
            self.messages.append({"role": "assistant", "content": raw_text[:call_end]})

            tool_fn = TOOLS_BY_NAME.get(name)
            result = tool_fn(**args) if tool_fn else f"[unknown tool: {name}]"
            self.messages.append({"role": "ipython", "content": result})

        return "[turn cap reached without a final answer]"
