"""Substitutional dopants for nanocarbons.

The host is always carbon. These functions edit a finished structure;
nothing here builds a different material, and every builder returns pure
carbon unless a dopant is asked for.

:mod:`.chemistry` says which heteroatoms are supported and how much of
each the sp2 lattice really tolerates; :mod:`.substitutional` places them
at random or by coordination; :mod:`.rings` places them by ring size, so
a dopant can be put on the pentagons that carry a curved structure's
reactivity.
"""

from .chemistry import (
    DOPANT_CHEMISTRY,
    DOPANT_ELEMENTS,
    PLANAR_DOPANTS,
    DopantChemistry,
    describe,
    get_chemistry,
)
from .rings import dope_rings, ring_sites, ring_size_census
from .substitutional import (
    codope,
    dope_directed,
    dope_random,
    substitute_atoms,
)

__all__ = [
    "DOPANT_CHEMISTRY",
    "DOPANT_ELEMENTS",
    "PLANAR_DOPANTS",
    "DopantChemistry",
    "codope",
    "describe",
    "dope_directed",
    "dope_random",
    "dope_rings",
    "get_chemistry",
    "ring_sites",
    "ring_size_census",
    "substitute_atoms",
]
