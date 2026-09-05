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

# Substitution fraction the interfaces offer, as the user-facing range.
# The lower bound is where one dopant in a hundred carbons starts to be a
# concentration rather than a single defect; the upper is where ordered
# stoichiometric phases (BC3, C3N4) take over from doping and a random
# placement stops describing anything real. Per-element ceilings live in
# `dopants/chemistry.py` and are tighter for most species.
MIN_DOPING_FRACTION: float = 0.01
MAX_DOPING_FRACTION: float = 0.15

# Covalent radii (Cordero et al. 2008, Å) for soft bond inference when
# non-carbon species are present.
#
# The transition metals and heavy chalcogens are not optional garnish: an
# element missing here falls back to MAX_CC_DISTANCE, which is 1.80 Å, and
# a 2.404 Å Mo-S bond then reads as *no bond at all*. Every MoS2 structure
# validated as "isolated atoms" and was refused by both exporters until
# these were added.
COVALENT_RADII: dict[str, float] = {
    # sp2 carbon and its substitutional dopants
    "C": 0.76,
    "N": 0.71,
    "B": 0.84,
    "S": 1.05,
    "P": 1.07,
    "H": 0.31,
    "O": 0.66,
    # dichalcogenide metals
    "Mo": 1.54,
    "W": 1.62,
    "Nb": 1.64,
    "Ta": 1.70,
    "V": 1.53,
    "Ti": 1.60,
    "Zr": 1.75,
    "Hf": 1.75,
    "Pt": 1.36,
    "Sn": 1.39,
    # heavier chalcogens
    "Se": 1.20,
    "Te": 1.38,
    # common terminations and substituents
    "F": 0.57,
    "Cl": 1.02,
    "Si": 1.11,
    # heavier main-group substitutional dopants for sp2 carbon
    "Al": 1.21,
    "Ge": 1.20,
    # 3d metals, as single atoms anchored in a vacancy. Cordero quotes two
    # values for the open-shell ones; the low-spin figure is used, since a
    # metal held in a carbon or M-N4 pocket is the constrained case.
    "Mn": 1.39,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
}

# Coordination a fully bonded atom of each element is expected to reach.
# sp2 carbon is 3; a dichalcogenide metal is 6 and its chalcogen 3. Judging
# a MoS2 metal against carbon's "5 or more is unphysical" would reject
# every correct structure in the tmd package.
MAX_COORDINATION: dict[str, int] = {
    "C": 4, "N": 4, "B": 4, "P": 4, "H": 1, "O": 3, "F": 1, "Cl": 1,
    "Si": 4, "Al": 4, "Ge": 4,
    # A 3d metal substituted into carbon sits in a vacancy pocket, so it
    # reaches the 3 or 4 neighbours the pocket offers rather than a bulk
    # metal's twelve.
    "Mn": 6, "Fe": 6, "Co": 6, "Ni": 6, "Cu": 6, "Zn": 6,
    "S": 6, "Se": 6, "Te": 6,          # 2 in a thiol, 6 in a MX2 sandwich
    # Six chalcogen ligands, plus at most one metal-metal partner: that
    # seventh bond is the 2.8 Å dimer that defines the 1T' phase, so a
    # ceiling of 6 rejects a correct 1T' cell.
    "Mo": 7, "W": 7, "Nb": 7, "Ta": 7, "Ti": 7, "Zr": 7, "Hf": 7,
    "Pt": 7, "Sn": 7, "V": 7,
}

# Fallback when an element is not in MAX_COORDINATION.
DEFAULT_MAX_COORDINATION: int = 6

# Homoelemental bond lengths (Å): twice the metallic radius for a metal,
# a covalent single bond for a chalcogen. Used both to detect these bonds
# and to relax them when they occur as defects.
HOMOELEMENTAL_BOND: dict[str, float] = {
    "Mo": 2.78, "W": 2.78, "Nb": 2.92, "Ta": 2.92, "Ti": 2.94,
    "Zr": 3.20, "Hf": 3.18, "Pt": 2.78, "Sn": 3.16, "V": 2.62,
    "S": 2.10, "Se": 2.40, "Te": 2.76,
}

# Element pairs whose covalent-radii sum is a misleading bond cutoff.
#
# Two metallic radii overshoot the in-plane metal-metal spacing of a
# layered dichalcogenide: Mo + Mo + 0.30 is 3.38 Å while MoS2's lattice
# constant is 3.16, so every metal read as bonded to its six in-plane
# neighbours and came out 12-coordinate. A real M-M bond is the 2.8 Å
# 1T' dimer, and cutting between the two is what separates a genuine
# defect bond from the lattice repeat.
#
# Every metal *pair*, not just same-element ones: an alloy such as
# Mo(1-x)W(x)S2 has Mo next to W at the lattice spacing, and leaving that
# pair to the radii sum (3.46 Å against a 3.16 Å lattice) put the metals
# back at coordination 10.
_METALS = {symbol: length for symbol, length in HOMOELEMENTAL_BOND.items()
           if symbol not in ("S", "Se", "Te")}
BOND_CUTOFF_OVERRIDE: dict[tuple[str, str], float] = {
    (first, second): 0.5 * (first_length + second_length) + 0.20
    for first, first_length in _METALS.items()
    for second, second_length in _METALS.items()
}
