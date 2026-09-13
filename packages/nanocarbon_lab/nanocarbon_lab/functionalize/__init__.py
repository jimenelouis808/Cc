"""Surface functionalisation: composable groups grafted onto a structure.

Two halves, deliberately separate:

* :mod:`~nanocarbon_lab.functionalize.groups` is the *grammar* -- groups
  written as internal coordinates so an element swap rebuilds the bond
  lengths instead of carrying the old ones over.
* :mod:`~nanocarbon_lab.functionalize.attach` is the *placement* -- where
  a surface points outward, which sites are reachable, and what the
  structure records about what was grafted.
"""

from __future__ import annotations

from .attach import (
    candidate_sites,
    describe_functionalization,
    functionalize,
    inner_face_blocked,
    is_enclosing,
    sublattice_parity,
    surface_normals,
)
from .groups import (
    GROUPS,
    VALENCE,
    FunctionalGroup,
    GroupAtom,
    as_dict,
    bond_length,
    build_bridging_positions,
    build_positions,
    describe,
    get_group,
    substitute,
    viable_swaps,
)

__all__ = [
    "GROUPS",
    "VALENCE",
    "FunctionalGroup",
    "GroupAtom",
    "as_dict",
    "bond_length",
    "build_bridging_positions",
    "build_positions",
    "candidate_sites",
    "describe",
    "describe_functionalization",
    "functionalize",
    "get_group",
    "inner_face_blocked",
    "is_enclosing",
    "sublattice_parity",
    "substitute",
    "surface_normals",
    "viable_swaps",
]
