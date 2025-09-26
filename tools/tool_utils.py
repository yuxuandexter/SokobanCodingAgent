from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List, Tuple


def safe_run_shell(cmd: str, timeout: int = 120) -> Tuple[str, str]:
    """
    Execute a shell command safely with timeout.
    Returns (output, exit_code_str). On non-zero exit, output contains stdout+stderr, exit_code_str='Error: Exit code N'.
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return "Empty command.", "Error: Exit code 2"

    try:
        proc = subprocess.run(
            ["/bin/bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            return output, f"Error: Exit code {proc.returncode}"
        return output, str(proc.returncode)
    except subprocess.TimeoutExpired:
        return f"The command took too long to execute (>{timeout}s)", "-1"
    except Exception as e:
        return f"Error: {repr(e)}", "-1"


def list_non_hidden_files(directory: Path, max_depth: int = 2, python_only: bool = True) -> List[Path]:
    results: List[Path] = []
    base_depth = len(directory.resolve().parts)
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        depth = len(Path(root).resolve().parts) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        for f in files:
            if f.startswith("."):
                continue
            p = Path(root) / f
            if not python_only or p.suffix == ".py":
                results.append(p)
    return results


