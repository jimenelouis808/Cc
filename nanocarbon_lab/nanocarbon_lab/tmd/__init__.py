"""Transition-metal dichalcogenide (MX2) structures in 1D, 2D and 3D.

Separate from :mod:`nanocarbon_lab.builders` because a TMD is a
three-plane X-M-X sandwich rather than a one-atom-thick surface, so
almost none of the carbon machinery transfers: the metal is
six-coordinate, the bond is 2.4 Å rather than 1.42, and rolling a layer
strains it in a way rolling graphene does not.
"""

from .coil import build_tmd_coil
from .materials import MATERIALS, TMDMaterial, get_material
from .nanotube import build_tmd_nanotube, tube_radius
from .quality import geometry_report, tmd_quality
from .ribbon import build_tmd_ribbon
from .slab import build_tmd_bulk, build_tmd_layers, build_tmd_monolayer

__all__ = [
    "MATERIALS",
    "TMDMaterial",
    "build_tmd_bulk",
    "build_tmd_coil",
    "build_tmd_layers",
    "build_tmd_monolayer",
    "build_tmd_nanotube",
    "build_tmd_ribbon",
    "geometry_report",
    "get_material",
    "tmd_quality",
    "tube_radius",
]
