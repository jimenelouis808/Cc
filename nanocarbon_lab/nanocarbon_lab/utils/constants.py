"""Physical constants and tolerances used throughout nanocarbon_lab.

All distances are in Angstroms. Values chosen to match sp2 carbon chemistry
and standard nanocarbon literature (graphene C-C = 1.42 Å).
"""

from __future__ import annotations

# Equilibrium sp2 C-C bond length (graphene, CNT, graphite basal plane).
CC_BOND: float = 1.42

# Soft bond-detection window used by validation/topology.
# Below MIN_CC_DISTANCE ⇒ overlap; above MAX_CC_DISTANCE ⇒ no bond.
MIN_CC_DISTANCE: float = 1.20
MAX_CC_DISTANCE: float = 1.80

# Hard minimum: any pair closer than this is treated as a structural error.
HARD_MIN_DISTANCE: float = 0.90

# Default vacuum padding for reduced-dimensionality systems.
DEFAULT_VACUUM_2D: float = 15.0  # Å around 2D sheets
DEFAULT_VACUUM_1D: float = 12.0  # Å around 1D tubes/wires

# Supported substitutional dopants (sp2 compatible on top of the list).
DOPANT_ELEMENTS: tuple[str, ...] = ("N", "B", "S", "P")

# Covalent radii (Pyykkö, single-bond values, Å) for soft bond inference when
# non-carbon species are present.
COVALENT_RADII: dict[str, float] = {
    "C": 0.76,
    "N": 0.71,
    "B": 0.84,
    "S": 1.05,
    "P": 1.07,
    "H": 0.31,
    "O": 0.66,
}
