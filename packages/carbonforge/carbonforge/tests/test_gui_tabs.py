"""The window is split into one module per tab; keep it that way.

Each tab is a mixin of :class:`~carbonforge.gui.app.CarbonForgeApp`. A tab
that imported another (or the app) would re-tangle what the split undid and
block moving tabs into a shared window with vibspec (PLAN_SUITE, phase B).
These tests need no display.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TABS = Path(__file__).resolve().parents[1] / "gui" / "tabs"
MODULES = sorted(p for p in TABS.glob("*.py") if p.name != "__init__.py")


def _imports(path: Path) -> set[str]:
    """Every module a file imports, as written (``..params``, ``.edlc``...)."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            found.add("." * node.level + (node.module or ""))
        elif isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
    return found


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_tab_imports_no_other_tab_and_not_the_app(path):
    siblings = {"." + p.stem for p in MODULES} | {"..tabs", "..app"}
    assert not (_imports(path) & siblings)


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.stem)
def test_tab_imports_tk_only_lazily(path):
    """Importing a tab must work without Tk (Windows without tcl, CI)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {getattr(n, "module", None) or n.names[0].name for n in top}
    assert not {"tkinter", "matplotlib.backends.backend_tkagg"} & names


def test_app_is_the_sum_of_its_tabs():
    from carbonforge.gui.app import CarbonForgeApp
    from carbonforge.gui.tabs import AnalysisTab, BuilderTab, EdlcTab, ImportTab, PreviewPanel

    for tab in (AnalysisTab, BuilderTab, EdlcTab, ImportTab, PreviewPanel):
        assert issubclass(CarbonForgeApp, tab)
    # No handler is defined twice: two tabs must not fight over a name.
    seen: dict[str, str] = {}
    for tab in (AnalysisTab, BuilderTab, EdlcTab, ImportTab, PreviewPanel):
        for name in vars(tab):
            if name.startswith("_") and not name.startswith("__"):
                assert name not in seen, f"{name} en {tab.__name__} y {seen[name]}"
                seen[name] = tab.__name__
