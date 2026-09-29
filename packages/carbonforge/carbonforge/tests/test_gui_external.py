"""«Abrir en tbkit»: the structure goes by file to a separate process, never an import."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from ase.build import graphene_nanoribbon
from ase.io import read

from carbonforge.gui import external


def test_structure_written_with_its_cell(tmp_path):
    ribbon = graphene_nanoribbon(2, 1, type="zigzag", saturated=True, vacuum=5.0)
    path = external.write_for_tbkit(ribbon, tmp_path)
    again = read(path)
    assert len(again) == len(ribbon) and list(again.get_pbc()) == list(ribbon.get_pbc())


def test_launches_a_separate_process(tmp_path, monkeypatch):
    monkeypatch.setattr(external, "tbkit_command", lambda: ["tbkit-gui"])
    launched = []
    path, command = external.open_in_tbkit(
        graphene_nanoribbon(2, 1, type="armchair", saturated=True, vacuum=5.0), tmp_path,
        launch=launched.append)
    assert launched == [["tbkit-gui", str(path)]] and path.exists()


def test_missing_tbkit_says_how_to_install(monkeypatch):
    monkeypatch.setattr(external, "tbkit_command", lambda: None)
    with pytest.raises(RuntimeError, match="tbkit"):
        external.open_in_tbkit(graphene_nanoribbon(2, 1, type="zigzag", vacuum=5.0),
                               launch=lambda c: None)


def test_carbonforge_never_imports_tbkit():
    root = Path(external.__file__).resolve().parents[1]
    for source in root.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(n == "tbkit" or n.startswith("tbkit.") for n in names), source
