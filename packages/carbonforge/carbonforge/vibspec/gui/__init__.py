"""The vibspec window: a thin Tkinter layer over :mod:`carbonforge.vibspec.core`.

* :mod:`~carbonforge.vibspec.gui.logic` -- forms, model building, the
  subprocess job queue and normal-mode animation data. Pure Python, tested
  headless.
* :mod:`~carbonforge.vibspec.gui.app` -- the widgets. Tk is imported only
  when the window opens, so nothing else in carbonforge needs it.

Open with ``carbonforge vibspec gui`` or ``python -m carbonforge.vibspec.gui``.
"""


def main(workdir: str | None = None) -> int:
    """Open the vibspec window."""
    from .app import main as _main

    return _main(workdir)
