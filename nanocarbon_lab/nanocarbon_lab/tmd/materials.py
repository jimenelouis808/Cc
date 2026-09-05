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
    "NbS2": TMDMaterial("Nb", "S", a=3.310, h=3.130, interlayer=5.945),
    "TaSe2": TMDMaterial("Ta", "Se", a=3.436, h=3.330, interlayer=6.350),
    # --- group 5, 1T: itinerant magnetism and CDW order
    "VS2": TMDMaterial("V", "S", a=3.174, h=2.974, interlayer=5.755,
                       natural_phase="1T"),
    "VSe2": TMDMaterial("V", "Se", a=3.358, h=3.125, interlayer=6.107,
                        natural_phase="1T"),
    "VTe2": TMDMaterial("V", "Te", a=3.640, h=3.454, interlayer=6.582,
                        natural_phase="1T"),
    # --- group 4 and 10: 1T ground state
    "TiS2": TMDMaterial("Ti", "S", a=3.407, h=2.925, interlayer=5.695,
                        natural_phase="1T"),
    "TiSe2": TMDMaterial("Ti", "Se", a=3.540, h=3.050, interlayer=6.008,
                         natural_phase="1T"),
    "TiTe2": TMDMaterial("Ti", "Te", a=3.777, h=3.285, interlayer=6.498,
                         natural_phase="1T"),
    "ZrS2": TMDMaterial("Zr", "S", a=3.662, h=2.930, interlayer=5.813,
                        natural_phase="1T"),
    "ZrSe2": TMDMaterial("Zr", "Se", a=3.770, h=3.195, interlayer=6.128,
                         natural_phase="1T"),
    "ZrTe2": TMDMaterial("Zr", "Te", a=3.950, h=3.485, interlayer=6.630,
                         natural_phase="1T"),
    "HfS2": TMDMaterial("Hf", "S", a=3.635, h=2.920, interlayer=5.837,
                        natural_phase="1T"),
    "HfSe2": TMDMaterial("Hf", "Se", a=3.748, h=3.162, interlayer=6.159,
                         natural_phase="1T"),
    "HfTe2": TMDMaterial("Hf", "Te", a=3.949, h=3.420, interlayer=6.651,
                         natural_phase="1T"),
    # Pt: note the unusually tight van der Waals gap (~2.4 Å against
    # MoS2's 3.0). That is real and is why the PtX2 band gap depends so
    # strongly on layer count -- do not "correct" it to look like the rest.
    "PtS2": TMDMaterial("Pt", "S", a=3.542, h=2.700, interlayer=5.043,
                        natural_phase="1T"),
    "PtSe2": TMDMaterial("Pt", "Se", a=3.728, h=2.660, interlayer=5.081,
                         natural_phase="1T"),
    "PtTe2": TMDMaterial("Pt", "Te", a=4.026, h=2.747, interlayer=5.221,
                         natural_phase="1T"),
    "SnS2": TMDMaterial("Sn", "S", a=3.648, h=2.980, interlayer=5.899,
                        natural_phase="1T"),
    "SnSe2": TMDMaterial("Sn", "Se", a=3.811, h=3.266, interlayer=6.141,
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


def available_metals() -> tuple[str, ...]:
    """Transition metals (and Sn) the table covers, in periodic order.

    Ordered by group rather than alphabetically, because the group is
    what decides the phase: group 6 is 2H and semiconducting, groups 4
    and 10 are 1T and metallic or semiconducting.
    """
    order = ("Ti", "Zr", "Hf", "V", "Nb", "Ta", "Mo", "W", "Pt", "Sn")
    present = {m.metal for m in MATERIALS.values()}
    listed = tuple(symbol for symbol in order if symbol in present)
    # Anything added to MATERIALS without being placed in `order` still
    # shows up, rather than silently vanishing from every dropdown.
    return listed + tuple(sorted(present - set(listed)))


def available_chalcogens() -> tuple[str, ...]:
    """Chalcogens the table covers, lightest first."""
    order = ("S", "Se", "Te")
    present = {m.chalcogen for m in MATERIALS.values()}
    listed = tuple(symbol for symbol in order if symbol in present)
    return listed + tuple(sorted(present - set(listed)))


def chalcogens_for(metal: str) -> tuple[str, ...]:
    """Which chalcogens this metal has a tabulated compound with."""
    found = {m.chalcogen for m in MATERIALS.values() if m.metal == metal}
    return tuple(x for x in available_chalcogens() if x in found)


def metals_for(chalcogen: str) -> tuple[str, ...]:
    """Which metals this chalcogen has a tabulated compound with."""
    found = {m.metal for m in MATERIALS.values() if m.chalcogen == chalcogen}
    return tuple(m for m in available_metals() if m in found)


def material_for(metal: str, chalcogen: str) -> TMDMaterial:
    """Look a compound up by its two elements rather than by formula.

    The formula is a presentation detail; what a user actually chooses is
    a metal and a chalcogen, and this is the lookup that matches. It is
    deliberately **not** a constructor: an MX2 that is not in the table
    is one whose lattice constants this package does not know, and
    inventing them from covalent radii would produce a structure that
    looks authoritative and is not.

    Parameters
    ----------
    metal, chalcogen
        Element symbols, e.g. ``"Mo"`` and ``"S"``.

    Returns
    -------
    TMDMaterial
        The tabulated compound.

    Raises
    ------
    KeyError
        Naming what *is* available for that metal, or for that chalcogen,
        so the next guess is an informed one. A missing pair is usually
        either a compound that does not form a layered MX2 at all (SnTe2)
        or one whose real structure is too distorted for the ideal
        1T/2H cells here (ReS2 and ReSe2 form diamond chains, NbTe2 and
        TaTe2 a different distortion).
    """
    for name, material in MATERIALS.items():
        if material.metal == metal and material.chalcogen == chalcogen:
            del name
            return material

    if metal not in available_metals():
        raise KeyError(
            f"Unknown metal {metal!r}. Available: "
            f"{', '.join(available_metals())}."
        )
    if chalcogen not in available_chalcogens():
        raise KeyError(
            f"Unknown chalcogen {chalcogen!r}. Available: "
            f"{', '.join(available_chalcogens())}."
        )
    raise KeyError(
        f"No tabulated {metal}{chalcogen}2. {metal} is available with "
        f"{', '.join(chalcogens_for(metal))}; {chalcogen} is available with "
        f"{', '.join(metals_for(chalcogen))}. A missing pair either does not "
        f"form a layered MX2 or is too distorted for the ideal 1T/2H cells "
        f"used here."
    )


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
    "available_chalcogens",
    "available_metals",
    "chalcogens_for",
    "coordination_geometry",
    "get_material",
    "material_for",
    "metals_for",
    "sublattice_offsets",
]
