"""Analysing structures this package did not build.

The builders record what they did; a file from elsewhere records nothing,
so everything has to be recovered from coordinates. The point of this
package is to do that **and keep the provenance visible**: what the file
recorded, what was measured from the atoms, and what was inferred by a
rule are reported separately, because "pentagons: 12" reads the same
whether a builder placed them or a cutoff guessed them.
"""

from __future__ import annotations

from .report import (
    analyse,
    bond_statistics,
    composition,
    coordination_census,
    format_report,
    read_structure,
)
from .rings import (
    MAX_RING_SIZE,
    is_surface_net,
    perceive_rings,
    ring_report,
    trace_faces,
)
from .shape import describe_shape, periodic_axes, vacuum_axes

__all__ = [
    "MAX_RING_SIZE",
    "analyse",
    "bond_statistics",
    "composition",
    "coordination_census",
    "describe_shape",
    "format_report",
    "is_surface_net",
    "perceive_rings",
    "periodic_axes",
    "read_structure",
    "ring_report",
    "trace_faces",
    "vacuum_axes",
]
