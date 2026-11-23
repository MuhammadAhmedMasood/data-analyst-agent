"""Attack the sandbox: every one of these must be blocked."""
import sys

import pytest

from app.data.store import DataStore
from app.sandbox.runner import run_code


@pytest.fixture(scope="module")
def data_dir():
    store = DataStore()
    store.load_csv("data/samples/Superstore.csv")
    return store.data_dir


ATTACKS = {
    "network_socket": "import socket\nsocket.create_connection(('example.com', 80))",
    "network_urllib": "import urllib.request\nurllib.request.urlopen('http://example.com')",
    "shell_command": "import os\nos.system('echo hacked')",
    "subprocess": "import subprocess\nsubprocess.run(['ls', '/'])",
    "read_system_file": "print(open('/etc/passwd').read())",
    "write_file": "open('evil.txt', 'w').write('x')",
    "delete_file": "import os\nos.remove('superstore.parquet')",
    "list_root": "import os\nprint(os.listdir('/'))",
    "ctypes": "import ctypes\nctypes.CDLL(None)",
    "add_own_hook": "import sys\nsys.addaudithook(lambda *a: None)\nopen('/etc/passwd')",
}


@pytest.mark.parametrize("name", ATTACKS)
def test_attack_is_blocked(data_dir, name):
    out = run_code(ATTACKS[name], data_dir, timeout=10)
    assert out.startswith("Error"), f"{name} was not blocked: {out}"
    assert "Sandbox" in out or "PermissionError" in out


def test_cannot_read_files_outside_data_dir(data_dir, tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("OPENAI_API_KEY=sk-secret")
    out = run_code(f"print(open({str(secret)!r}).read())", data_dir, timeout=10)
    assert "sk-secret" not in out and out.startswith("Error")


def test_api_key_not_in_environment(data_dir, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    out = run_code("import os\nprint(os.environ.get('OPENAI_API_KEY'))", data_dir, timeout=10)
    assert out == "None"


def test_normal_analysis_still_works(data_dir):
    code = (
        "import statistics\n"
        "totals = superstore.groupby('Region')['Sales'].sum()\n"
        "print(round(statistics.mean(totals), 2))"
    )
    out = run_code(code, data_dir, timeout=20)
    assert not out.startswith("Error"), out


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS can't limit memory with rlimit")
def test_memory_bomb_is_stopped(data_dir):
    out = run_code("x = bytearray(10 * 1024**3)", data_dir, timeout=20)
    assert out.startswith("Error") and "MemoryError" in out


@pytest.mark.xfail(
    strict=True,
    reason="Native libraries (pyarrow, duckdb) do file I/O in C++, which Python "
    "audit hooks can't see. Closed by OS-level isolation in Phase 8.",
)
def test_native_library_cannot_read_outside(data_dir, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("key\nsk-secret\n")
    code = f"import pyarrow.csv as pc\nprint(pc.read_csv({str(secret)!r}))"
    out = run_code(code, data_dir, timeout=10)
    assert "sk-secret" not in out