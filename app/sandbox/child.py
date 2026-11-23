"""Runs inside a separate, locked-down Python process.
Loads every table as a DataFrame, applies the sandbox rules, then executes
the agent's code (read from stdin)."""
import os
import resource
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CPU_SECONDS = 20
MEMORY_BYTES = 2 * 1024**3  # 2 GB

DATA_DIR = Path.cwd().resolve()
READ_ROOTS = [DATA_DIR] + [Path(p).resolve() for p in {sys.prefix, sys.base_prefix}]
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
BLOCKED_EVENTS = (
    "socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn",
    "os.spawn", "os.fork", "os.kill", "os.remove", "os.rename", "os.rmdir",
    "os.mkdir", "os.chmod", "os.chown", "os.link", "os.symlink",
    "os.truncate", "os.putenv", "os.unsetenv", "shutil.", "ctypes.",
    "urllib.", "webbrowser.",
)


def is_readable(path) -> bool:
    if isinstance(path, int):  # an already-open file descriptor
        return True
    resolved = Path(os.fsdecode(path)).resolve()
    return any(resolved.is_relative_to(root) for root in READ_ROOTS)


def audit_hook(event: str, args: tuple) -> None:
    if event == "open":
        path, mode, flags = args
        writing = (mode is not None and any(c in mode for c in "wax+")) or bool(flags & WRITE_FLAGS)
        if writing or path is None or not is_readable(path):
            raise PermissionError(f"Sandbox: access to {path!r} is not allowed")
    elif event in ("os.listdir", "os.scandir"):
        if not is_readable(args[0] or "."):
            raise PermissionError(f"Sandbox: listing {args[0]!r} is not allowed")
    elif event.startswith(BLOCKED_EVENTS):
        raise PermissionError(f"Sandbox: '{event}' is not allowed")


def limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    except (ValueError, OSError):
        pass  # macOS can't limit memory this way; Docker will in Phase 8


# 1. Load data while we are still allowed to read files.
namespace = {"pd": pd, "np": np}
for parquet_file in sorted(DATA_DIR.glob("*.parquet")):
    namespace[parquet_file.stem] = pq.read_table(parquet_file).to_pandas(date_as_object=False)
code = sys.stdin.read()

# 2. Lock down. From here on the rules apply, and they cannot be removed.
limit_resources()
sys.addaudithook(audit_hook)

# 3. Run the agent's code.
exec(compile(code, "<agent_code>", "exec"), namespace)