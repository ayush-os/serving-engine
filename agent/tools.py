"""Real tools for the traffic-generating agent (spec-agent-pcie.md Decision
1): sandboxed code execution and local doc retrieval. Deliberately minimal
-- generating realistic multi-turn traffic is the point, not tool
capability, so this stays at two tools and no framework.
"""
import re
import subprocess
import sys
from pathlib import Path

CODE_EXEC_TIMEOUT_S = 5
MAX_TOOL_OUTPUT_CHARS = 2000
REPO_ROOT = Path(__file__).resolve().parent.parent


def execute_code(code: str) -> str:
    """Execute Python code in a subprocess and return its output.

    Args:
        code: The Python source code to execute.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=CODE_EXEC_TIMEOUT_S,
        )
        output = result.stdout or ""
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
    except subprocess.TimeoutExpired:
        output = f"[timed out after {CODE_EXEC_TIMEOUT_S}s]"
    output = output.strip()
    return output[:MAX_TOOL_OUTPUT_CHARS] if output else "[no output]"


def search_docs(query: str) -> str:
    """Search this project's local markdown documentation for relevant text.

    Args:
        query: The search query describing what to look for.
    """
    terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    if not terms:
        return "[no usable search terms in query]"

    best = None  # (score, relative_path, snippet)
    for path in REPO_ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        text = path.read_text(errors="ignore")
        lower = text.lower()
        score = sum(lower.count(t) for t in terms)
        if score == 0:
            continue
        idx = lower.find(terms[0])
        start = max(0, idx - 150)
        snippet = text[start:idx + 350].strip()
        if best is None or score > best[0]:
            best = (score, str(path.relative_to(REPO_ROOT)), snippet)

    if best is None:
        return f"[no matches for: {query}]"
    _, rel_path, snippet = best
    return f"From {rel_path}:\n{snippet}"


TOOLS = [execute_code, search_docs]
TOOLS_BY_NAME = {fn.__name__: fn for fn in TOOLS}
