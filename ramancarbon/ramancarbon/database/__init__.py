"""The literature database and the typed API that reads it.

Every number this package uses that came out of a paper lives in a JSON file
under ``database/data/`` with a ``source`` and a ``confidence`` field beside
it. Nothing is hardcoded in the analysis modules. That is a deliberate
design decision with two consequences worth stating:

* A specialist can correct or extend the database — add a house
  parameterisation of the RBM relation, tighten a band window to match their
  own instrument — without touching a line of Python, and the change
  propagates to the GUI, the CLI and the reports at once.
* Every reported quantity can name the reference it came from, which is what
  makes the output usable in a methods section.

Load with :func:`load_database`; it caches, so calling it in a loop is free.
"""

from __future__ import annotations

from .loader import (
    Band,
    Database,
    DopantSignature,
    Material,
    PerturbationEffect,
    RBMParameterisation,
    clear_cache,
    load_database,
)

__all__ = [
    "Band",
    "Database",
    "DopantSignature",
    "Material",
    "PerturbationEffect",
    "RBMParameterisation",
    "clear_cache",
    "load_database",
]
