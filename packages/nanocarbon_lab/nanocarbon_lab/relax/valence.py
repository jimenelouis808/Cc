"""Relaxing a structure that already exists, with the valence force field.

**The engine was already here and there was no way in.**
:func:`~nanocarbon_lab.builders.fullerene_mesh.relax_shell` is the real
thing — bond stretching, a *true* angle term and short-range non-bonded
repulsion, minimised with L-BFGS-B on exact analytic gradients — and
eleven builders call it while they build. But it takes raw positions and
an explicit bond set, which a builder has and a person holding an
``.xyz`` does not. The only relaxation this package exported was
:func:`~nanocarbon_lab.relax.optimize.harmonic_pre_relax`, which is bond
springs alone and is documented upstairs as exactly the thing that lets
a sheet pyramidalise and fold.

So this module is a way in, not a second force field. What it adds on
top of the engine is the three decisions a caller would otherwise have
to get right by hand, each of which has a way of being quietly wrong:

1. **Where the bonds come from.** Never from distances if the structure
   carries its own list. On a curved shell many *non*-bonded atoms sit
   closer than a bond length, so a covalent-radius cutoff invents bonds
   across a neck and the angle term then tries to flatten a saddle that
   is supposed to be a saddle. Builders in this package record
   ``atoms.info["bonds"]``; that is preferred, and the fallback is
   checked rather than trusted.
2. **What angle to aim at.** A single 120° is right for a graphitic
   wall and wrong for a diamond-like node. The target is read from each
   centre's own coordination, so a structure with both gets both.
3. **What to do about the cell.** The engine's neighbour list is a
   ``cKDTree`` with ``boxsize``, which handles orthorhombic cells and
   cannot do a triclinic one. A triclinic periodic cell is refused with
   that reason, rather than relaxed against wrong minimum images.

What this is NOT, and the limit matters more than the feature: a
valence force field is a **geometry cleaner, not an energy model**. It
has one rest length, one angle target per coordination and no
conjugation, so it will happily report a strained isomer and a relaxed
one as equally fine. It cannot rank stabilities, cannot find a
reconstruction and cannot break or make a bond — the topology it is
handed is the topology it returns. For energies, hand the cleaned
geometry to a real potential; :func:`write_relaxation_request` opens
that door.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from ase import Atoms

from ..builders.capped_cnt import geometry_report
from ..builders.fullerene_mesh import relax_shell
from ..utils.geometry import guess_bonds

#: Ideal bond angle (degrees) at a centre of each coordination number.
#: Two-coordinate carbon is linear (sp), three is trigonal planar (sp2)
#: and four is tetrahedral (sp3). A centre with more neighbours than
#: this has no single target and is handled by :data:`NO_ANGLE_TARGET`.
ANGLE_FOR_COORDINATION: dict[int, float] = {
    2: 180.0,
    3: 120.0,
    4: math.degrees(math.acos(-1.0 / 3.0)),
}

#: Coordination numbers with no usable single angle target. Their angles
#: are left out of the angle term entirely and held apart by the
#: non-bonded repulsion instead — the VSEPR argument `relax_shell`
#: already documents for a six-coordinate metal. Writing them in with
#: some averaged target is worse than leaving them out: it pulls a
#: perfectly good octahedron toward an angle no octahedron has.
NO_ANGLE_TARGET: float = float("nan")

#: Highest coordination a carbon atom can really have. A derived bond
#: list that puts more neighbours than this on a carbon has invented at
#: least one bond, and relaxing against it moves atoms that were fine.
MAX_CARBON_COORDINATION: int = 4


def bonds_from_atoms(
    atoms: Atoms, tolerance: float = 0.30
) -> tuple[list[tuple[int, int]], str, list[str]]:
    """The bond list to relax against, and where it came from.

    Returns ``(bonds, source, warnings)`` with ``source`` one of
    ``"info"`` (the builder recorded it) or ``"distances"`` (guessed
    here). A guessed list is **checked** before it is returned: any
    carbon with more than :data:`MAX_CARBON_COORDINATION` neighbours, or
    any atom with none, is reported as a warning, because those are the
    two ways a radius cutoff fails on a curved structure and neither is
    visible in the relaxed output.

    Parameters
    ----------
    atoms
        The structure. ``atoms.info["bonds"]`` is used when present.
    tolerance
        Slack (Å) over the sum of covalent radii, for the fallback only.
    """
    stored = atoms.info.get("bonds")
    if stored is not None and len(stored):
        bonds = [(int(a), int(b)) if a < b else (int(b), int(a))
                 for a, b in stored]
        return sorted(set(bonds)), "info", []

    bonds = sorted({(int(i), int(j)) if i < j else (int(j), int(i))
                    for i, j, _ in guess_bonds(atoms, tolerance=tolerance)})
    warnings: list[str] = []
    counts = np.zeros(len(atoms), dtype=int)
    for a, b in bonds:
        counts[a] += 1
        counts[b] += 1
    symbols = np.asarray(atoms.get_chemical_symbols())
    crowded = np.flatnonzero((symbols == "C")
                             & (counts > MAX_CARBON_COORDINATION))
    if crowded.size:
        warnings.append(
            f"{crowded.size} carbon atoms came out with more than "
            f"{MAX_CARBON_COORDINATION} neighbours (worst: "
            f"{int(counts[crowded].max())}). A radius cutoff invents bonds "
            "across a neck on a curved structure; pass `bonds` explicitly "
            "or lower `tolerance`."
        )
    lonely = int(np.count_nonzero(counts == 0))
    if lonely:
        warnings.append(
            f"{lonely} atoms came out with no bonds at all; they take no "
            "valence force and will not move."
        )
    return bonds, "distances", warnings


def angle_targets(
    angle_triplets: np.ndarray,
    coordination: np.ndarray,
    override: Mapping[int, float] | None = None,
) -> np.ndarray:
    """One target angle (degrees) per triplet, from the centre's coordination.

    ``angle_triplets`` is the ``(m, 3)`` array of ``(i, centre, k)`` that
    the engine builds from the bond list; the centre is column 1. A
    centre whose coordination has no single ideal angle gets
    :data:`NO_ANGLE_TARGET`, which the caller drops.

    Parameters
    ----------
    angle_triplets
        ``(m, 3)`` integer array, centre in column 1.
    coordination
        Per-atom neighbour count.
    override
        Optional ``{coordination: degrees}`` replacing the defaults.
    """
    table = dict(ANGLE_FOR_COORDINATION)
    if override:
        table.update({int(k): float(v) for k, v in override.items()})
    centres = angle_triplets[:, 1]
    out = np.full(len(angle_triplets), NO_ANGLE_TARGET, dtype=float)
    for number, degrees in table.items():
        out[coordination[centres] == number] = degrees
    return out


def relax_structure(
    atoms: Atoms,
    bonds: Iterable[Sequence[int]] | None = None,
    equilibrium: float = 1.42,
    angle_override: Mapping[int, float] | None = None,
    k_bond: float = 40.0,
    k_angle: float = 15.0,
    k_repel: float = 25.0,
    repel_cutoff: float = 2.2,
    repel_skin: float = 2.0,
    outer_cycles: int = 3,
    max_iterations: int = 3000,
    tolerance: float = 0.30,
) -> Atoms:
    """Clean a structure's geometry with the valence force field.

    The topology is **frozen**: the bond list passed in (or found) is
    held for the whole minimisation, so this can never change a ring
    census — it only moves atoms. Bond lengths go to ``equilibrium`` and
    each angle to whatever its centre's coordination asks for.

    Parameters
    ----------
    atoms
        Structure to relax. Modified in place and returned.
    bonds
        Explicit bonded pairs. Omit to take ``atoms.info["bonds"]``, or
        to guess from covalent radii when the structure carries none —
        see :func:`bonds_from_atoms` for why guessing is the last
        resort.
    equilibrium
        Rest bond length (Å); 1.42 is graphitic carbon.
    angle_override
        ``{coordination: degrees}`` replacing :data:`ANGLE_FOR_COORDINATION`.
    k_bond, k_angle, k_repel
        Force constants, in the engine's consistent arbitrary units.
        ``k_angle`` must stay a substantial fraction of ``k_bond``: too
        soft an angle term is what lets a wall fold through itself.
    repel_cutoff, repel_skin
        Non-bonded repulsion range, and the extra radius its neighbour
        list is built with. The skin is not optional on anything long
        and floppy; the engine's docstring has the measurement.
    outer_cycles, max_iterations
        Neighbour-list rebuilds, and L-BFGS iterations per rebuild.
    tolerance
        Covalent-radius slack for guessing bonds, when it comes to that.

    Returns
    -------
    ase.Atoms
        The same object, relaxed, with the before/after measurements and
        every warning under ``atoms.info["valence_relax"]``.

    Raises
    ------
    ValueError
        If the structure is periodic in a triclinic cell. The engine's
        neighbour list is a ``cKDTree`` with ``boxsize``, which is
        orthorhombic-only; relaxing anyway would use wrong minimum
        images across the seams and the result would look fine.
    """
    warnings: list[str] = []
    if bonds is None:
        bond_list, source, found = bonds_from_atoms(atoms, tolerance=tolerance)
        warnings += found
    else:
        bond_list = sorted({(int(a), int(b)) if a < b else (int(b), int(a))
                            for a, b in bonds})
        source = "caller"
    if not bond_list:
        raise ValueError(
            "no bonds to relax against: the structure carries no "
            "`info['bonds']` and none were guessed from its distances."
        )

    box = _box_for(atoms)

    counts = np.zeros(len(atoms), dtype=int)
    for a, b in bond_list:
        counts[a] += 1
        counts[b] += 1

    before = geometry_report(atoms.get_positions(), bond_list, box=box)

    # The angle targets, and the triplets that have none. The engine
    # builds its own triplet array from the bonds in sorted order, which
    # is exactly what `_angle_triplets` reproduces here -- the two must
    # agree index for index or every target lands on the wrong angle.
    triplets = _angle_triplets(bond_list)
    targets = angle_targets(triplets, counts, override=angle_override)
    usable = np.isfinite(targets)
    if not usable.all():
        unusable = int((~usable).sum())
        warnings.append(
            f"{unusable} angles sit on a centre whose coordination has no "
            "single ideal angle; they are left to the non-bonded "
            "repulsion instead of being pulled to an average that no "
            "such centre has."
        )
        # Nothing to pull them to, so the angle term must not see them:
        # a NaN target would poison the whole gradient.
        targets = np.where(usable, targets, 0.0)

    # The angle targets are CARBON's. Three-coordinate means 120 deg
    # here, and on a dichalcogenide that is simply false: a chalcogen
    # with three metal neighbours sits near 82 deg, and a six-coordinate
    # metal has no single angle at all. Pulling an MX2 to carbon's
    # targets does not fail, it returns a wrecked sandwich with
    # excellent-looking bond lengths -- which is the whole class of
    # failure this package's notes keep warning about. So say it, here,
    # where every caller sees it, rather than in each caller.
    foreign = sorted({symbol for symbol in atoms.get_chemical_symbols()
                      if symbol not in ("C", "H")})
    if foreign and angle_override is None:
        warnings.append(
            f"this structure contains {', '.join(foreign)} and the angle "
            "targets used are carbon's (120 deg at three neighbours, "
            "109.47 at four). For a dichalcogenide or a crystal that is "
            "wrong and the relaxation will deform it while the bond "
            "lengths still look right. Pass `angle_override`, or relax "
            "the carbon part alone, or send it to an external code with "
            "`write_relaxation_request`."
        )

    positions = relax_shell(
        atoms.get_positions().astype(float),
        set(bond_list),
        equilibrium=equilibrium,
        angle_deg=targets,
        k_bond=k_bond,
        k_angle=np.where(usable, k_angle, 0.0),
        k_repel=k_repel,
        repel_cutoff=repel_cutoff,
        repel_skin=repel_skin,
        box=box,
        outer_cycles=outer_cycles,
        max_iterations=max_iterations,
    )
    moved = float(np.linalg.norm(positions - atoms.get_positions(), axis=1).max())
    atoms.set_positions(positions)
    after = geometry_report(positions, bond_list, box=box)

    atoms.info["valence_relax"] = {
        "bond_source": source,
        "bonds": len(bond_list),
        "angles": int(usable.sum()),
        "angles_without_target": int((~usable).sum()),
        "equilibrium": equilibrium,
        "largest_move": moved,
        "before": before,
        "after": after,
        "warnings": warnings,
    }
    return atoms


def _box_for(atoms: Atoms) -> np.ndarray | None:
    """The engine's ``box``: the cell edges, or ``None`` for a free structure.

    A zero edge is how the engine is told an axis is not periodic, which
    is what a sheet's vacuum direction needs, so a partly periodic cell
    comes back with zeros on its free axes rather than being refused.
    """
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if not pbc.any():
        return None
    cell = np.asarray(atoms.cell)
    off_diagonal = cell - np.diag(np.diag(cell))
    if not np.allclose(off_diagonal, 0.0, atol=1e-8):
        raise ValueError(
            "this structure is periodic in a triclinic cell, and the "
            "relaxer's neighbour list (cKDTree with boxsize) is "
            "orthorhombic-only. Relaxing it anyway would take the wrong "
            "minimum image across every seam and the result would look "
            "fine. Rebuild it in an orthorhombic cell, or relax it with "
            "an external calculator (see write_relaxation_request)."
        )
    return np.where(pbc, np.abs(np.diag(cell)), 0.0)


def _angle_triplets(bonds: Sequence[tuple[int, int]]) -> np.ndarray:
    """``(m, 3)`` ``(i, centre, k)`` triplets, in the engine's own order.

    A copy of what the engine builds internally, because the targets
    computed here are matched to it positionally. Kept beside the call
    that needs it rather than imported, so that if the engine's ordering
    ever changes the test that pins them together fails here.
    """
    from collections import defaultdict

    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in sorted(bonds):
        neighbours[int(a)].append(int(b))
        neighbours[int(b)].append(int(a))
    triplets: list[tuple[int, int, int]] = []
    for centre, ns in neighbours.items():
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                triplets.append((ns[i], centre, ns[j]))
    return np.array(sorted(triplets), dtype=int).reshape(-1, 3)
