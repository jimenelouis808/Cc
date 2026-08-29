"""Allow ``python -m nanocarbon_lab.gui``.

The ``__name__`` guard is not decoration. Builds run in a subprocess
started with the ``spawn`` method (see :mod:`nanocarbon_lab.gui.worker`),
and spawn re-imports the parent's ``__main__`` module in the child. With
``main()`` called at import time, every build would open a second window
-- and that window would open a third. The guard makes the child import
this file harmlessly.
"""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
