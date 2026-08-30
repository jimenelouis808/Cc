"""Bibliometric pipeline for a review on defects and doping in 1D nanocarbons.

Scopus + Web of Science exports in, a deduplicated and classified corpus out,
handed to ``bibliometrix``/``biblioshiny`` in R without losing cited references.

See ``docs/WORKFLOW.md`` for the end-to-end run and ``docs/PROTOCOL.md`` for the
scope decisions the pipeline encodes.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "classify",
    "dedupe",
    "demo",
    "exporters",
    "indicators",
    "lexicons",
    "loaders",
    "records",
    "thesaurus",
]
