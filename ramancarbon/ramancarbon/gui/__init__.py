"""Tkinter desktop application.

``main`` is exposed here so the console script can be
``ramancarbon.gui:main``. The import of :mod:`ramancarbon.gui.app` is
deferred into the function: importing this package on a machine without
Tkinter must not raise, because the CLI imports nothing from here but a
user may still run ``python -m ramancarbon.gui`` and deserves the
installation instructions rather than a traceback.
"""

from __future__ import annotations


def main() -> int:
    """Launch the desktop application. Returns a process exit code."""
    from .app import main as _main

    return _main()


__all__ = ["main"]
