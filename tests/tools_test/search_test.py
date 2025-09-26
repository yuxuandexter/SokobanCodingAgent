from pathlib import Path

from tools import build_default_tool_manager


def ensure_cache_dir():
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_search_in_file(tmp_path: Path):
    p = tmp_path / "foo.py"
    p.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
    tm = build_default_tool_manager()
    res = tm.execute("search", {"search_term": "alpha", "path": str(p)})
    assert res["exit_code"] == "0"
    assert "Matches for \"alpha\"" in res["output"]
    assert ":alpha" in res["output"]

    cache = ensure_cache_dir()
    out_file = cache / "tools_search_test_log.txt"
    with out_file.open("a", encoding="utf-8") as f:
        f.write("Case: file\n")
        f.write(res["output"] + "\n")


def test_search_in_directory(tmp_path: Path):
    (tmp_path / "a.py").write_text("x=1\nalpha\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("beta\n", encoding="utf-8")
    tm = build_default_tool_manager()
    res = tm.execute("search", {"search_term": "alpha", "path": str(tmp_path)})
    assert res["exit_code"] == "0"
    assert "Found" in res["output"]

    cache = ensure_cache_dir()
    out_file = cache / "tools_search_test_log.txt"
    with out_file.open("a", encoding="utf-8") as f:
        f.write("Case: directory\n")
        f.write(res["output"] + "\n")

