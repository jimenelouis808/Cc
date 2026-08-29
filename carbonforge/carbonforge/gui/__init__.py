"""Graphical user interface for carbonforge.

The GUI is split in two layers so the interesting part stays testable:

* :mod:`carbonforge.gui.params` — declarative parameter specs and the
  build / modify / export logic. Pure Python, no Tk, fully unit-tested.
* :mod:`carbonforge.gui.app` — the Tkinter widgets. Imported lazily so
  that a machine without Tk can still use everything else in the package.

Launch with ``carbonforge-gui`` (installed entry point) or
``python -m carbonforge.gui``.
"""

from .params import (
    CALCULATION_PARAMS,
    STRUCTURES,
    MODIFIER_PARAMS,
    ParamSpec,
    StructureSpec,
    build_structure,
    apply_modifiers,
    export_structure,
    describe_structure,
    build_calculation_specs,
    validate_calculation,
)

__all__ = [
    "STRUCTURES",
    "MODIFIER_PARAMS",
    "CALCULATION_PARAMS",
    "build_calculation_specs",
    "validate_calculation",
    "ParamSpec",
    "StructureSpec",
    "build_structure",
    "apply_modifiers",
    "export_structure",
    "describe_structure",
    "main",
]


def main() -> int:
    """Launch the Tkinter GUI. Entry point for ``carbonforge-gui``."""
    from .app import main as _main

    return _main()
