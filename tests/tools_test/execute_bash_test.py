from pathlib import Path

from tools import build_default_tool_manager


def ensure_cache_dir():
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_execute_bash_echo():
    tm = build_default_tool_manager()
    res = tm.execute("execute_bash", {"cmd": "echo hello"})
    assert res["exit_code"] == "0"
    assert "hello" in res["output"].strip()

    cache = ensure_cache_dir()
    out_file = cache / "tools_execute_bash_test_log.txt"
    with out_file.open("a", encoding="utf-8") as f:
        f.write("Case: echo\n")
        f.write(res["output"] + "\n")


def test_execute_bash_nonzero_exit():
    tm = build_default_tool_manager()
    res = tm.execute("execute_bash", {"cmd": "bash -c 'exit 3'"})
    assert res["exit_code"].startswith("Error: Exit code ")

    cache = ensure_cache_dir()
    out_file = cache / "tools_execute_bash_test_log.txt"
    with out_file.open("a", encoding="utf-8") as f:
        f.write("Case: nonzero-exit\n")
        f.write(res["exit_code"] + "\n")


def test_execute_bash_empty_command():
    tm = build_default_tool_manager()
    res = tm.execute("execute_bash", {"cmd": "   "})
    assert res["exit_code"] == "Error: Exit code 2"
    assert "Empty command." in res["output"]


def test_execute_bash_stderr_but_success():
    tm = build_default_tool_manager()
    # Writes to stderr but exits with 0; output should include stderr and exit_code should be "0"
    res = tm.execute("execute_bash", {"cmd": "echo warn 1>&2; true"})
    assert res["exit_code"] == "0"
    assert "warn" in res["output"]


def test_execute_bash_timeout():
    tm = build_default_tool_manager()
    # Force a very small timeout to trigger timeout path
    res = tm.execute("execute_bash", {"cmd": "sleep 1", "timeout": 0})
    assert res["exit_code"] == "-1"
    assert "too long" in res["output"]

