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


def test_file_editor_view_directory_lists_python_only(tmp_path: Path):
    tm = build_default_tool_manager()
    d = tmp_path
    (d / "a.py").write_text("x=1\n", encoding="utf-8")
    (d / "b.txt").write_text("ignore\n", encoding="utf-8")
    res = tm.execute("file_editor", {"command": "view", "path": str(d)})
    assert res["exit_code"] == "0"
    assert "a.py" in res["output"]
    assert "b.txt" not in res["output"]


def test_file_editor_view_range_and_validation(tmp_path: Path):
    tm = build_default_tool_manager()
    f = tmp_path / "r.py"
    f.write_text("L1\nL2\nL3\nL4\n", encoding="utf-8")
    ok = tm.execute("file_editor", {"command": "view", "path": str(f), "view_range": [2, 3]})
    assert ok["exit_code"] == "0"
    assert "L2" in ok["output"] and "L3" in ok["output"]
    bad = tm.execute("file_editor", {"command": "view", "path": str(f), "view_range": [4, 2]})
    assert bad["exit_code"] == "-1"
    assert "Invalid view_range" in bad["output"]


def test_file_editor_str_replace_errors(tmp_path: Path):
    tm = build_default_tool_manager()
    f = tmp_path / "s.py"
    f.write_text("a\na\n", encoding="utf-8")
    # Multiple occurrences -> error
    multi = tm.execute("file_editor", {"command": "str_replace", "path": str(f), "old_str": "a", "new_str": "b"})
    assert multi["exit_code"] == "-1"
    assert "Multiple occurrences" in multi["output"]
    # No occurrences -> error
    none = tm.execute("file_editor", {"command": "str_replace", "path": str(f), "old_str": "zzz", "new_str": "b"})
    assert none["exit_code"] == "-1"
    assert "No occurrences" in none["output"]


def test_file_editor_insert_boundaries(tmp_path: Path):
    tm = build_default_tool_manager()
    f = tmp_path / "i.py"
    f.write_text("first\nsecond", encoding="utf-8")
    # insert at start
    res0 = tm.execute("file_editor", {"command": "insert", "path": str(f), "insert_line": 0, "new_str": "HEAD\n"})
    assert res0["exit_code"] == "0"
    # insert at end (len(lines)=2 after split by \n when .expandtabs keeps two lines)
    resn = tm.execute("file_editor", {"command": "insert", "path": str(f), "insert_line": 2, "new_str": "\nTAIL"})
    assert resn["exit_code"] == "0"
    # out of range
    bad = tm.execute("file_editor", {"command": "insert", "path": str(f), "insert_line": 99, "new_str": "x"})
    assert bad["exit_code"] == "-1"
    assert "out of range" in bad["output"]


def test_file_editor_undo_without_history(tmp_path: Path):
    tm = build_default_tool_manager()
    f = tmp_path / "u.py"
    f.write_text("data", encoding="utf-8")
    res = tm.execute("file_editor", {"command": "undo_edit", "path": str(f)})
    assert res["exit_code"] == "-1"
    assert "No previous edits" in res["output"]

