from pathlib import Path

from tools import build_default_tool_manager


def ensure_cache_dir():
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_finish_submit():
    tm = build_default_tool_manager()
    res = tm.execute("finish", {"command": "submit", "result": "ok"})
    assert res["exit_code"] == "0"
    assert res.get("done") is True
    assert res.get("result") == "ok"
    cache = ensure_cache_dir()
    out_file = cache / "tools_finish_test_log.txt"
    with out_file.open("a", encoding="utf-8") as f:
        f.write("Case: submit\n")
        f.write(str(res) + "\n")

