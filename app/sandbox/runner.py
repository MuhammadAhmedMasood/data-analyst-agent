"""Run untrusted Python in a separate process with a timeout.
Phase 5 will lock this down further (no network, memory limits)."""
import subprocess
import sys
from pathlib import Path

CHILD_SCRIPT = Path(__file__).parent / "child.py"
MAX_OUTPUT_CHARS = 4000


def run_code(code: str, data_dir: Path, timeout: float = 30) -> str:
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(CHILD_SCRIPT)],
            input=code,
            cwd=data_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: code took longer than {timeout} seconds and was stopped."

    if result.returncode != 0:
        return "Error:\n" + result.stderr[-MAX_OUTPUT_CHARS:]

    output = result.stdout.strip()
    if not output:
        return "Code ran but printed nothing. Use print() to show results."
    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n... (output truncated)"
    return output