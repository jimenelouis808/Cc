"""The vibspec pages: a thin Tkinter layer over :mod:`carbonforge.vibspec.core`.

* :mod:`~carbonforge.vibspec.gui.logic` -- forms, model building, the
  subprocess job queue and normal-mode animation data. Pure Python, tested
  headless.
* :mod:`~carbonforge.vibspec.gui.app` -- the widgets. Tk is imported only
  when the window opens, so nothing else in carbonforge needs it.

The pages live in carbonforge's main window. ``carbonforge vibspec gui`` or
``python -m carbonforge.vibspec.gui`` opens it at the vibspec pages.
"""


def main(workdir: str | None = None) -> int:
    """Open the main window at the vibspec pages."""
    from .app import main as _main

    return _main(workdir)
