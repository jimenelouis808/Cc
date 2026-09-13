"""Tkinter desktop GUI for building and exporting nanocarbon structures.

Importing this package must **not** require tkinter. :mod:`.worker` runs
builds in a subprocess and contains no widgets at all, so
``from nanocarbon_lab.gui.worker import BuildWorker`` has to succeed on a
headless machine with no Tk installed. It used to fail, because this file
imported :mod:`.app` eagerly and that module raised ``SystemExit`` when
tkinter was missing -- which pytest cannot catch during collection, so a
box without ``python3-tk`` could not run *any* test in the suite, not
merely the GUI ones.

So :mod:`.app` is imported lazily, and the advice about installing
tkinter is raised from :func:`main`, where a person is there to read it.
"""

from __future__ import annotations

TKINTER_HELP = (
    "The nanocarbon_lab GUI needs tkinter, which is not available in this "
    "Python installation.\n"
    "  Debian/Ubuntu : sudo apt install python3-tk\n"
    "  Fedora/RHEL   : sudo dnf install python3-tkinter\n"
    "  macOS/Windows : use the python.org installer (tkinter is included)\n"
    "You can still build structures without a GUI via the 'nanocarbon' "
    "command."
)

__all__ = ["TKINTER_HELP", "main"]


def main() -> int:
    """Entry point for ``nanocarbon-gui``.

    Checks for tkinter explicitly before importing :mod:`.app`, so a
    missing toolkit produces the installation advice above while any
    *other* import failure inside the app still surfaces as itself.
    """
    try:
        import tkinter  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(TKINTER_HELP) from exc

    from .app import main as _main

    return _main()
