"""Phase 1 checkpoint (spec-agent-pcie.md): a handful of genuine
multi-step, single-session tasks -- confirming the loop terminates
correctly and tool results are actually incorporated into the model's next
turn, before this harness is trusted to generate Phase 2 concurrent load.

Run directly:

    python scripts/run_agent_sanity_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serving_engine.engine import LLMEngine
from agent.engine_driver import EngineDriver
from agent.loop import AgentSession

TASKS = [
    "What is 1837 * 429? Use code to compute it, don't do the arithmetic yourself.",
    "Search the project docs for the prefix-caching demo's hit rate, then tell me the number.",
    "Compute the 20th Fibonacci number (0-indexed) using code.",
]


def main():
    print("Loading model and sizing KV cache pool from free GPU memory...")
    engine = LLMEngine()
    driver = EngineDriver(engine)
    driver.start()

    try:
        for i, task in enumerate(TASKS):
            print(f"\n=== TASK: {task} ===")
            session = AgentSession(driver, engine.model_runner.tokenizer, session_id=f"sanity-{i}")
            answer = session.run(task)
            for msg in session.messages:
                content = msg["content"]
                preview = content[:300] + ("..." if len(content) > 300 else "")
                print(f"[{msg['role']}] {preview}")
            print(f"--> FINAL ANSWER: {answer}")
    finally:
        driver.stop()


if __name__ == "__main__":
    main()
