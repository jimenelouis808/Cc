"""Checks on the widget module that do not need a display.

``test_gui.py`` exercises the real widget tree, and skips itself wherever
Tkinter or an X display is missing -- which is most machines that are not
CI. That leaves the largest single module in the package with no coverage
at all in an ordinary checkout, and the two bugs it hides best are exactly
the two a linter cannot see: a ``command=`` naming a method that does not
exist, and a ``self.x`` read but never assigned. Both are attribute
lookups, resolved at click time, in a window nobody opened.

So these read the source rather than importing it, and check it against
itself.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "gui" / "app.py"

#: Attributes the widget class gets from Tk or from its own construction
#: helpers rather than from a plain assignment.
INHERITED = {"root", "tk", "ttk"}


def source() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_the_widget_module_parses():
    ast.parse(source())


def test_every_self_attribute_used_is_assigned_somewhere():
    tree = ast.parse(source())
    for node in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        assigned: set[str] = set()
        read: set[str] = set()
        for child in ast.walk(node):
            if not (isinstance(child, ast.Attribute)
                    and isinstance(child.value, ast.Name)
                    and child.value.id == "self"):
                continue
            if isinstance(child.ctx, (ast.Store, ast.Del)):
                assigned.add(child.attr)
            else:
                read.add(child.attr)
        defined = {n.name for n in ast.walk(node)
                   if isinstance(n, ast.FunctionDef)}
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(
                    statement.target, ast.Name):
                defined.add(statement.target.id)
            elif isinstance(statement, ast.Assign):
                defined |= {t.id for t in statement.targets
                            if isinstance(t, ast.Name)}
        missing = read - assigned - defined - INHERITED
        assert not missing, f"{node.name}: read but never assigned: {sorted(missing)}"


def test_every_button_callback_exists():
    text = source()
    tree = ast.parse(text)
    methods = {n.name for c in ast.walk(tree) if isinstance(c, ast.ClassDef)
               for n in ast.walk(c) if isinstance(n, ast.FunctionDef)}
    # ``command=self._canvas.yview`` and friends hand off to a widget, not
    # to a method of ours.
    referenced = {name for name in re.findall(r"command=self\.([\w.]+)", text)
                  if "." not in name}
    missing = referenced - methods
    assert not missing, f"commands with no method: {sorted(missing)}"


def test_the_preview_draws_bonds_through_the_shift_helper():
    """The regression this guards is visual and therefore silent.

    A bond list built under the minimum image convention names the pair
    and not the cell the far atom is in. Drawing it from raw coordinates
    puts a line straight across the structure for every bond that leaves
    through a face -- 131 of them on a gyroid cell, up to 46 Å long. It
    looks like a rendering artefact and it is a missing subtraction.
    """
    text = source()
    assert "_bond_shifts" in text
    assert "var_show_wrapped" in text
    # The old form, which is the bug.
    assert "(pos[a] + shift, pos[b] + shift)" not in text


def test_the_supercell_goes_through_the_helper_that_carries_the_bonds():
    """``Atoms.repeat`` copies ``info`` verbatim, so a supercell built with
    it keeps one cell's bond list, ring list and ring census while having
    eight times the atoms. Nothing about the atom count shows it."""
    text = source()
    assert "supercell(self.atoms, counts)" in text
    assert "self.atoms.repeat(counts)" not in text
