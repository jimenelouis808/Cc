"""Monolayers, multilayers and bulk crystals of MX2 dichalcogenides.

Everything here is **exact crystallography**, not a relaxed model: atoms
are placed on their ideal Wyckoff positions from the measured lattice
constants in :mod:`nanocarbon_lab.tmd.materials`. That is deliberate.
These are the structures people hand to DFT or to a Stillinger-Weber MD
run as a *starting point*, and a builder that pre-relaxed them with an
approximate force field would only add error the real calculation is
about to remove.

The one place this package does deform a structure is the nanotube,
where rolling is a genuine geometric operation with a strain that has to
be reported; see :mod:`nanocarbon_lab.tmd.nanotube`.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.constants import DEFAULT_VACUUM_2D
from ..utils.geometry import center_in_cell
from .materials import (
    Phase,
    Stacking,
    TMDMaterial,
    coordination_geometry,
    get_material,
    sublattice_offsets,
)

# Metal displacement along the dimerisation direction in the 1T' phase, as
# a fraction of `a`. Alternate metals move toward each other, so the M-M
# spacing alternates a*(1 - 2s) and a*(1 + 2s): at s = 0.057 that turns
# MoS2's uniform 3.16 Å into 2.80 and 3.52 Å, matching the ~2.8 Å dimer
# reported for 1T'-MoS2. The real distortion also buckles the chalcogen
# planes; this reproduces the metal-metal zigzag chains, which is the part
# that opens the gap and the part people draw.
DIMER_SHIFT = 0.057


def hexagonal_cell(material: TMDMaterial, height: float) -> np.ndarray:
    """The 3x3 cell matrix for a hexagonal TMD lattice of the given height."""
    a = material.a
    return np.array(
        [
            [a, 0.0, 0.0],
            [-a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0],
            [0.0, 0.0, height],
        ]
    )


def _layer_sites(
    material: TMDMaterial, phase: Phase, z0: float, flip: bool
) -> list[tuple[str, np.ndarray]]:
    """Fractional in-plane sites plus absolute z for one X-M-X sandwich.

    ``flip`` swaps the metal and chalcogen columns, which is what turns
    an A layer into the A' layer of 2H stacking -- the neighbouring layer
    is the same sandwich rotated 180 degrees, so its metal sits above the
    first layer's chalcogen.
    """
    metal_xy, top_xy, bottom_xy = sublattice_offsets(phase)
    if flip:
        # The AA' relation of 2H is a 6_3 screw: rotate the sandwich 180
        # deg about c, then translate by (1/3, 2/3). Both halves matter.
        # Negating alone rotates about the metal site, which leaves the
        # metal exactly where it was and stacks metal directly on metal --
        # that is AA, not AA'. With the translation the metal lands on the
        # B column, i.e. directly above the chalcogen of the layer below,
        # which is what makes 2H the structure it is.
        def screw(site: tuple[float, float]) -> tuple[float, float]:
            return (-site[0] + 1.0 / 3.0, -site[1] + 2.0 / 3.0)

        metal_xy = screw(metal_xy)
        top_xy = screw(top_xy)
        bottom_xy = screw(bottom_xy)

    half = material.h / 2.0
    return [
        (material.metal, np.array([metal_xy[0], metal_xy[1], z0])),
        (material.chalcogen, np.array([top_xy[0], top_xy[1], z0 + half])),
        (material.chalcogen, np.array([bottom_xy[0], bottom_xy[1], z0 - half])),
    ]


def _stacking_shift(stacking: Stacking, index: int) -> tuple[float, float, bool]:
    """In-plane fractional shift and flip flag for layer ``index``.

    ``2H``
        Alternating layers are rotated 180 deg (the AA' sequence of bulk
        MoS2), so the metal of one layer faces the chalcogen of the next.
    ``3R``
        Every layer is shifted by (1/3, 2/3) with no rotation, giving the
        rhombohedral three-layer repeat.
    ``AA``
        No shift and no rotation. Not a ground state for any TMD, but it
        is the reference for twisted and artificially stacked bilayers.
    """
    if stacking == "2H":
        return 0.0, 0.0, bool(index % 2)
    if stacking == "3R":
        return (index % 3) / 3.0, (2.0 * (index % 3)) / 3.0, False
    if stacking == "AA":
        return 0.0, 0.0, False
    raise ValueError(f"Unknown stacking {stacking!r}; expected '2H', '3R' or 'AA'.")


def build_tmd_layers(
    material: str | TMDMaterial = "MoS2",
    n_layers: int = 1,
    phase: Phase = "2H",
    stacking: Stacking = "2H",
    nx: int = 1,
    ny: int = 1,
    vacuum: float = DEFAULT_VACUUM_2D,
    periodic_z: bool = False,
) -> Atoms:
    """Build a monolayer, a few-layer stack, or a bulk crystal.

    Parameters
    ----------
    material
        Formula (``"MoS2"``) or a :class:`~nanocarbon_lab.tmd.materials.
        TMDMaterial`.
    n_layers
        Number of X-M-X sandwiches. ``1`` is a monolayer, ``2`` a
        bilayer, and with ``periodic_z=True`` the natural repeat (2 for
        2H, 3 for 3R, 1 for AA) is a bulk crystal.
    phase
        ``"2H"``, ``"1T"`` or ``"1T'"``. This is the coordination inside
        each layer, and is independent of how layers stack.
    stacking
        How successive layers are placed: ``"2H"`` (AA', alternating
        180 deg rotation), ``"3R"`` (rhombohedral), ``"AA"`` (eclipsed).
        Ignored for a monolayer.
    nx, ny
        Supercell repetitions in the two in-plane lattice directions.
    vacuum
        Vacuum padding along z for a slab. Ignored when ``periodic_z``.
    periodic_z
        ``True`` makes the cell periodic in all three directions -- a
        bulk crystal rather than a slab. The c axis is then
        ``n_layers * interlayer`` with no vacuum.

    Returns
    -------
    ase.Atoms
        With ``info`` carrying the material, phase, stacking, layer count,
        coordination geometry and the derived bond length.

    Raises
    ------
    ValueError
        For a non-positive layer count or supercell size.
    """
    if isinstance(material, str):
        material = get_material(material)
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1.")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1.")

    symbols: list[str] = []
    fractional: list[np.ndarray] = []
    for index in range(n_layers):
        z0 = index * material.interlayer
        dx, dy, flip = _stacking_shift(stacking, index) if n_layers > 1 else (
            0.0, 0.0, False)
        for element, site in _layer_sites(material, phase, z0, flip):
            symbols.append(element)
            fractional.append(np.array([site[0] + dx, site[1] + dy, site[2]]))

    # Fractional in-plane, absolute in z: convert with the 2D cell only.
    height = (n_layers * material.interlayer if periodic_z
              else (n_layers - 1) * material.interlayer + material.h + vacuum)
    cell = hexagonal_cell(material, height)
    positions = np.array([
        frac[0] * cell[0] + frac[1] * cell[1] + np.array([0.0, 0.0, frac[2]])
        for frac in fractional
    ])

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell,
                  pbc=(True, True, periodic_z))
    if phase == "1T'":
        # 1T' is not a decoration of the 1T cell, it is a doubling of it:
        # the dimerisation pairs neighbouring metals, and a cell holding
        # one metal has no neighbour to pair with. Double first, then
        # distort, then let the caller's supercell repeat the result.
        atoms = atoms.repeat((2, 1, 1))
        _apply_1t_prime_distortion(atoms, material)
    atoms = atoms.repeat((nx, ny, 1))
    if not periodic_z:
        center_in_cell(atoms, axes=(2,))

    atoms.info.update(
        {
            "structure_type": "tmd_bulk" if periodic_z else "tmd_slab",
            "material": material.formula,
            "metal": material.metal,
            "chalcogen": material.chalcogen,
            "phase": phase,
            "stacking": stacking if n_layers > 1 else "n/a",
            "n_layers": n_layers,
            "coordination": coordination_geometry(phase),
            "a": material.a,
            "h": material.h,
            "interlayer": material.interlayer,
            "bond_length": material.bond_length,
            "vdw_gap": material.vdw_gap,
            "supercell": (nx, ny),
        }
    )
    if phase == "1T'":
        atoms.info["phase_note"] = (
            "idealised 1T': metals dimerised into zigzag chains, chalcogen "
            "planes left flat. The real phase also buckles them, so relax "
            "before using this for anything quantitative."
        )
    return atoms


def _apply_1t_prime_distortion(atoms: Atoms, material: TMDMaterial) -> None:
    """Pair the metals into zigzag chains, in place, on a doubled cell.

    1T' is 1T with the metals dimerised along one in-plane direction,
    which is what opens the gap and makes the phase a quantum spin Hall
    candidate. Alternate metals move toward each other along ``a1``, so
    the uniform M-M spacing splits into a short bond and a long gap.

    Which metal moves which way is decided by its **fractional** position
    along ``a1``, not its Cartesian x: the hexagonal cell is sheared, so
    successive rows sit at different x for the same lattice column and a
    Cartesian test pairs the wrong atoms.

    The real structure also buckles the chalcogen planes, so this is the
    idealised form; ``atoms.info['phase_note']`` says so.
    """
    cell = np.array(atoms.cell)
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    fractional = positions @ np.linalg.inv(cell)
    step = DIMER_SHIFT * material.a * (cell[0] / np.linalg.norm(cell[0]))

    for index, symbol in enumerate(symbols):
        if symbol != material.metal:
            continue
        # In the doubled cell the two metal columns sit at f1 = 0 and 0.5.
        column = round(fractional[index][0] * 2.0) % 2
        positions[index] += step if column == 0 else -step
    atoms.set_positions(positions)


def build_tmd_monolayer(material: str | TMDMaterial = "MoS2",
                        phase: Phase = "2H", **kwargs) -> Atoms:
    """A single X-M-X sandwich. Convenience wrapper over
    :func:`build_tmd_layers`."""
    kwargs.pop("n_layers", None)
    return build_tmd_layers(material, n_layers=1, phase=phase, **kwargs)


def build_tmd_bulk(material: str | TMDMaterial = "MoS2", phase: Phase = "2H",
                   stacking: Stacking = "2H", **kwargs) -> Atoms:
    """The bulk crystal, one full period along c.

    The number of layers is the stacking's own repeat -- two for 2H,
    three for 3R, one for AA -- so the result is the conventional cell
    rather than an arbitrary slab made periodic.
    """
    repeat = {"2H": 2, "3R": 3, "AA": 1}[stacking]
    kwargs.pop("n_layers", None)
    kwargs.pop("periodic_z", None)
    return build_tmd_layers(material, n_layers=repeat, phase=phase,
                            stacking=stacking, periodic_z=True, **kwargs)


__all__ = [
    "DIMER_SHIFT",
    "build_tmd_bulk",
    "build_tmd_layers",
    "build_tmd_monolayer",
    "hexagonal_cell",
]
