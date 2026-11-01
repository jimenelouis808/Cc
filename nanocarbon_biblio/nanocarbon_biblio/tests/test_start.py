"""Tests for the zero-dependency launcher.

``start.py`` is the one file that must work before anything is installed, so it
is tested by running it as a subprocess with the *system* interpreter rather
than by importing it: that is how a user actually invokes it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = ROOT / "start.py"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(START), *args],
        cwd=str(ROOT), capture_output=True, text=True, check=False, timeout=180,
    )


def test_launcher_exists_and_is_executable() -> None:
    assert START.exists()
    for wrapper in ("run_gui.sh", "run_gui.bat"):
        assert (ROOT / wrapper).exists(), wrapper


def test_launcher_imports_only_the_standard_library() -> None:
    """It has to run before pip has installed anything at all."""
    source = START.read_text(encoding="utf-8")
    third_party = ("import pandas", "import streamlit", "import numpy",
                   "import rapidfuzz", "import altair")
    for name in third_party:
        assert name not in source, f"start.py must not {name}"


def test_check_reports_a_healthy_environment() -> None:
    result = _run("--check")
    assert result.returncode == 0, result.stdout + result.stderr
    out = result.stdout
    assert "Python" in out
    assert "Proyecto completo" in out
    # The project is set up in CI by the time tests run, so dependencies resolve.
    assert "dependencias" in out.lower() or "entorno virtual" in out.lower()


def test_help_lists_every_documented_flag() -> None:
    result = _run("--help")
    assert result.returncode == 0
    for flag in ("--check", "--demo", "--tests", "--reinstall"):
        assert flag in result.stdout, flag
