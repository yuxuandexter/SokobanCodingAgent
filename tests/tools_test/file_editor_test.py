from pathlib import Path

from tools import build_default_tool_manager


def ensure_cache_dir():
    root = Path(__file__).resolve().parents[2]
    cache_dir = root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def test_file_editor_create_view_str_replace_insert_undo(tmp_path: Path):
    tm = build_default_tool_manager()
    f = tmp_path / "file.py"

    # create
    res = tm.execute("file_editor", {"command": "create", "path": str(f), "file_text": "a=1\n"})
    assert res["exit_code"] == "0"
    assert f.exists()
    cache = ensure_cache_dir()
    out_file = cache / "tools_file_editor_test_log.txt"
    with out_file.open("a", encoding="utf-8") as fo:
        fo.write("Case: create\n")
        fo.write(res["output"] + "\n")

    # view
    res = tm.execute("file_editor", {"command": "view", "path": str(f)})
    assert res["exit_code"] == "0"
    assert "a=1" in res["output"]
    with out_file.open("a", encoding="utf-8") as fo:
        fo.write("Case: view\n")
        fo.write(res["output"] + "\n")

    # str_replace
    res = tm.execute("file_editor", {"command": "str_replace", "path": str(f), "old_str": "a=1\n", "new_str": "a=2\n"})
    assert res["exit_code"] == "0"
    assert f.read_text(encoding="utf-8") == "a=2\n"
    with out_file.open("a", encoding="utf-8") as fo:
        fo.write("Case: str_replace\n")
        fo.write(res["output"] + "\n")

    # insert at line 1 (after first 0-based)
    res = tm.execute("file_editor", {"command": "insert", "path": str(f), "insert_line": 1, "new_str": "b=3\n"})
    assert res["exit_code"] == "0"
    assert f.read_text(encoding="utf-8").startswith("a=2\n")
    assert "b=3" in f.read_text(encoding="utf-8")
    with out_file.open("a", encoding="utf-8") as fo:
        fo.write("Case: insert\n")
        fo.write(res["output"] + "\n")

    # undo
    res = tm.execute("file_editor", {"command": "undo_edit", "path": str(f)})
    assert res["exit_code"] == "0"
    with out_file.open("a", encoding="utf-8") as fo:
        fo.write("Case: undo_edit\n")
        fo.write(res["output"] + "\n")

