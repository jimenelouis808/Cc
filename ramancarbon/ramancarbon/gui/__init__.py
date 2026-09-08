"""Tkinter desktop application: the four-instrument suite.

``main`` is exposed here so the console script can be
``ramancarbon.gui:main``. The import of :mod:`ramancarbon.gui.suite` is
deferred into the function: importing this package on a machine without
Tkinter must not raise, because a user may run ``python -m
ramancarbon.gui`` and deserves the installation instructions rather than a
traceback.
"""

from __future__ import annotations


def main() -> int:
    """Launch the desktop suite. Returns a process exit code."""
    from .suite import main as _main

    return _main()


__all__ = ["main"]
