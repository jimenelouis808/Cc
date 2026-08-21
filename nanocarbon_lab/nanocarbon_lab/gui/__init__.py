"""Graphical user interface for nanocarbon_lab.

The GUI is split in two layers so the interesting part stays testable:

* :mod:`nanocarbon_lab.gui.params` — declarative parameter specs and the
  build / modify / export logic. Pure Python, no Tk, fully unit-tested.
* :mod:`nanocarbon_lab.gui.app` — the Tkinter widgets. Imported lazily so
  that a machine without Tk can still use everything else in the package.

Launch with ``nanocarbon-gui`` (installed entry point) or
``python -m nanocarbon_lab.gui``.
"""

from .params import (
    STRUCTURES,
    MODIFIER_PARAMS,
    ParamSpec,
    StructureSpec,
    build_structure,
    apply_modifiers,
    export_structure,
    describe_structure,
)

__all__ = [
    "STRUCTURES",
    "MODIFIER_PARAMS",
    "ParamSpec",
    "StructureSpec",
    "build_structure",
    "apply_modifiers",
    "export_structure",
    "describe_structure",
    "main",
]


def main() -> int:
    """Launch the Tkinter GUI. Entry point for ``nanocarbon-gui``."""
    from .app import main as _main

    return _main()
