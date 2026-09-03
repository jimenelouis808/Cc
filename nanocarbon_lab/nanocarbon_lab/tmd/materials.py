"""Lattice parameters for transition-metal dichalcogenides.

A TMD is not a surface one atom thick. It is a **sandwich**: a plane of
metal atoms between two planes of chalcogen, X-M-X, held to its
neighbours by van der Waals forces alone. Every builder in this package
follows from that one fact, and it is why none of the carbon machinery
transfers directly -- a graphene sheet has one atom per lattice site,
an MoS2 layer has three.

Geometry is pinned by two numbers per material:

``a``
    In-plane lattice constant, the M-M nearest-neighbour distance.
``h``
    Vertical separation of the two chalcogen planes.

The metal-chalcogen bond length follows rather than being stored
separately, ``d = sqrt(a^2/3 + h^2/4)``, because storing all three
invites them to disagree. For MoS2 that gives 2.404 Å against the 2.41 Å
usually quoted -- the residual is the experimental spread, not an error
in the relation.

**Phases.** The two that matter are distinguished by where the *bottom*
chalcogen plane sits, and nothing else:

``2H``
    Trigonal prismatic. Both chalcogen planes sit over the same
    sublattice, so the top and bottom X are eclipsed. Semiconducting;
    the ground state for the group-6 materials (MoS2, WS2, ...).
``1T``
    Octahedral. The bottom plane sits over the third sublattice, so the
    two X planes are staggered by 60 deg. Metallic; the ground state for
    the group-4 and group-10 materials (TiS2, PtS2, ...).

That is the whole difference, and it is the distinction people usually
mean by "hexagonal versus the other phase". Note that **no TMD has a
tetragonal phase**: 2H, 1T and 1T' all sit on a hexagonal (or, for 1T',
a distorted hexagonal) lattice. What varies is the coordination
polyhedron around the metal, not the crystal system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Phase = Literal["2H", "1T", "1T'"]
Stacking = Literal["2H", "3R", "AA"]

#: Phases whose metal sits in a trigonal prism rather than an octahedron.
PRISMATIC_PHASES = frozenset({"2H"})


@dataclass(frozen=True)
class TMDMaterial:
    """Lattice parameters for one MX2 compound.

    Attributes
    ----------
    metal, chalcogen
        Element symbols.
    a
        In-plane lattice constant (Å) -- the metal-metal spacing.
    h
        Chalcogen-chalcogen vertical separation within one layer (Å).
    interlayer
        Metal-plane to metal-plane distance in the bulk crystal (Å).
        This is ``c/2`` for a 2H cell (two layers per period) and ``c``
        for a 1T cell (one layer), so it is stored directly rather than
        as ``c``, where the factor is a standing invitation to a
        factor-of-two error.
    natural_phase
        The phase this compound actually adopts. Building another one is
        allowed -- metastable phases are a real research subject -- but
        the builders record which was asked for.
    """

    metal: str
    chalcogen: str
    a: float
    h: float
    interlayer: float
    natural_phase: Phase = "2H"

    @property
    def formula(self) -> str:
        return f"{self.metal}{self.chalcogen}2"

    @property
    def bond_length(self) -> float:
        """Metal-chalcogen bond length (Å), derived from ``a`` and ``h``."""
        return math.sqrt(self.a**2 / 3.0 + self.h**2 / 4.0)

    @property
    def vdw_gap(self) -> float:
        """Chalcogen-plane to chalcogen-plane gap between layers (Å).

        The empty space a neighbouring layer sees, as opposed to the
        metal-to-metal ``interlayer``. About 3.1 Å for MoS2, close to
        graphite's 3.35 Å for the same reason: it is a van der Waals
        contact between two closed-shell sheets.
        """
        return self.interlayer - self.h


# Experimental room-temperature values, monolayer `a` and `h`, bulk
# `interlayer`. Sources agree to a few thousandths of an Ångström; where
# they differ the value consistent with the quoted M-X bond was kept.
MATERIALS: dict[str, TMDMaterial] = {
    # --- group 6: semiconducting, 2H ground state
    "MoS2": TMDMaterial("Mo", "S", a=3.160, h=3.130, interlayer=6.147),
    "MoSe2": TMDMaterial("Mo", "Se", a=3.288, h=3.338, interlayer=6.465),
    "MoTe2": TMDMaterial("Mo", "Te", a=3.522, h=3.604, interlayer=6.982),
    "WS2": TMDMaterial("W", "S", a=3.153, h=3.145, interlayer=6.160),
    "WSe2": TMDMaterial("W", "Se", a=3.282, h=3.340, interlayer=6.480),
    "WTe2": TMDMaterial("W", "Te", a=3.496, h=3.600, interlayer=7.035),
    # --- group 5: metallic, 2H, charge-density-wave superconductors
    "NbSe2": TMDMaterial("Nb", "Se", a=3.442, h=3.350, interlayer=6.276),
    "TaS2": TMDMaterial("Ta", "S", a=3.316, h=3.140, interlayer=6.050),
    # --- group 4 and 10: 1T ground state
    "TiS2": TMDMaterial("Ti", "S", a=3.407, h=2.925, interlayer=5.695,
                        natural_phase="1T"),
    "ZrS2": TMDMaterial("Zr", "S", a=3.662, h=2.930, interlayer=5.813,
                        natural_phase="1T"),
    "HfS2": TMDMaterial("Hf", "S", a=3.635, h=2.920, interlayer=5.837,
                        natural_phase="1T"),
    "PtS2": TMDMaterial("Pt", "S", a=3.542, h=2.700, interlayer=5.043,
                        natural_phase="1T"),
    "SnS2": TMDMaterial("Sn", "S", a=3.648, h=2.980, interlayer=5.899,
                        natural_phase="1T"),
}


def get_material(name: str) -> TMDMaterial:
    """Look up a material by formula, e.g. ``"MoS2"``.

    Raises
    ------
    KeyError
        With the list of known compounds, because a typo here is far
        more likely than a genuinely missing material.
    """
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(
            f"Unknown material {name!r}. Known: {', '.join(sorted(MATERIALS))}."
        ) from None


def sublattice_offsets(phase: Phase) -> tuple[tuple[float, float],
                                              tuple[float, float],
                                              tuple[float, float]]:
    """In-plane fractional positions of (metal, top X, bottom X).

    The hexagonal cell has three high-symmetry columns -- A at (0, 0),
    B at (1/3, 2/3) and C at (2/3, 1/3). The metal always takes A. The
    phase is entirely a statement about which column the chalcogens use:

    * ``2H`` puts both on B, eclipsed, giving a trigonal prism;
    * ``1T`` puts the top on B and the bottom on C, staggered by 60 deg,
      giving an octahedron.

    ``1T'`` shares 1T's columns and adds a metal-metal dimerisation on
    top, applied by the builder rather than expressible here.
    """
    site_a = (0.0, 0.0)
    site_b = (1.0 / 3.0, 2.0 / 3.0)
    site_c = (2.0 / 3.0, 1.0 / 3.0)
    if phase == "2H":
        return site_a, site_b, site_b
    if phase in ("1T", "1T'"):
        return site_a, site_b, site_c
    raise ValueError(f"Unknown phase {phase!r}; expected '2H', '1T' or \"1T'\".")


def coordination_geometry(phase: Phase) -> str:
    """Human-readable coordination of the metal in this phase."""
    return "trigonal prismatic" if phase in PRISMATIC_PHASES else "octahedral"


__all__ = [
    "MATERIALS",
    "PRISMATIC_PHASES",
    "Phase",
    "Stacking",
    "TMDMaterial",
    "coordination_geometry",
    "get_material",
    "sublattice_offsets",
]
