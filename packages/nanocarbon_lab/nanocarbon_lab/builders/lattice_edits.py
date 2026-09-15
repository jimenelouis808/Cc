"""Defects and corrugation for the builders that place atoms on a lattice.

The meshed builders -- capped tube, coil, junction, schwarzite -- edit their
**dual mesh** before any atom exists. Rotating a mesh edge there produces the
5-7-7-5 pattern by construction, and the force-field relaxation that turns the
mesh into atoms is part of the build anyway, so the defect comes out relaxed
for free.

The lattice builders have no mesh. :func:`~nanocarbon_lab.builders.cnt.build_cnt`
and :func:`~nanocarbon_lab.builders.nanoribbon.build_nanoribbon` place atoms
straight onto the graphene lattice, which is why they shipped without any of
the modification controls the capped tube has. This module does the same job
on the finished structure.

The order is the whole point: **edit, relax, then corrugate.**

A Stone-Wales rotation applied on its own is topologically the textbook
defect and geometrically a clash -- measured on a pristine (6,6) tube, one
rotation leaves four non-bonded pairs closer than 2 Å, and on a zigzag ribbon
five, because rotating two atoms by 90 deg without letting their neighbours
move swings them straight into the next ring. The local relaxation is what
turns one into the other. A divacancy is worse than a clash without help: it
leaves four two-coordinated atoms staring across a hole, which is a hole, not
the 5-8-5 everyone means by the word. Reconstructing it -- pairing those
dangling atoms and relaxing with the new bonds in the graph -- is what closes
the octagon.
"""

from __future__ import annotations

import warnings
from collections import Counter, deque
from typing import Any

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND
from ..utils.geometry import guess_bonds
from ..utils.rng import make_rng
from ..validation.quality import sp2_quality
from . import fullerene_mesh as fm
from .capped_cnt import geometry_report

#: Minimum centre-to-centre distance between two edits (Å). Below this
#: their strain fields overlap and the relaxation resolves them as one
#: larger, unintended defect rather than two of the requested kind.
MIN_DEFECT_SEPARATION = 6.0

#: C-H bond length used when a passivated ribbon is relaxed, matching the
#: value ASE's ``graphene_nanoribbon`` places the hydrogens at.
CH_BOND = 1.09


def periodic_box(atoms: Atoms) -> np.ndarray:
    """Cell edges for minimum-image measurement, ``0`` on a free axis.

    The convention ``fullerene_mesh.minimum_image`` and ``cKDTree`` already
    share: a zero edge means "not periodic along this axis". Without it a
    tube's seam bond measures a cell length rather than 1.42 Å, and every
    quality verdict on a periodic structure comes back "broken" for a
    structure that is perfect.
    """
    return np.where(np.asarray(atoms.get_pbc()), atoms.cell.lengths(), 0.0)


def _local_frames(
    positions: np.ndarray,
    table: list[set[int]],
    box: np.ndarray,
) -> np.ndarray:
    """A consistently oriented unit normal per atom.

    The normal is the thin direction of the atom's own neighbourhood --
    itself, its neighbours and its next-nearest neighbours -- read off as
    the smallest-eigenvalue eigenvector of their covariance. A plane
    through ten atoms rather than three is what makes this survive a
    defect: the obvious estimate, the cross product of two bond vectors,
    is fine on a flat sheet and unreliable exactly where it matters. On a
    (6,6) tube the atoms beside a reconstructed divacancy pucker enough
    that one such normal came out pointing *into* the tube, the cyclic
    order round it reversed, and the face trace walked out of one ring and
    into the next -- reporting a 17-membered ring where a pentagon and two
    hexagons live.

    Signs are then made to agree by propagating one atom's choice through
    the bond graph, since a covariance eigenvector has no sign of its own.
    """
    n_atoms = len(table)
    normals = np.zeros((n_atoms, 3))
    for index in range(n_atoms):
        shell = {index} | table[index]
        for neighbour in table[index]:
            shell |= table[neighbour]
        members = sorted(shell)
        if len(members) < 3:
            normals[index] = np.array([0.0, 0.0, 1.0])
            continue
        offsets = np.array([
            fm.minimum_image(positions[k] - positions[index], box) for k in members
        ])
        offsets = offsets - offsets.mean(axis=0)
        _, _, right = np.linalg.svd(offsets, full_matrices=False)
        normals[index] = right[-1]

    # Orient consistently: a breadth-first sweep flipping any normal that
    # points against the one it was reached from. The graph can have more
    # than one connected component, so every atom gets a chance to seed.
    seen: set[int] = set()
    for root in range(n_atoms):
        if root in seen:
            continue
        seen.add(root)
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbour in table[current]:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                if float(np.dot(normals[current], normals[neighbour])) < 0.0:
                    normals[neighbour] = -normals[neighbour]
                queue.append(neighbour)
    return normals


def ring_census(
    positions: np.ndarray,
    bonds: list[tuple[int, int]],
    box: np.ndarray | None = None,
    max_size: int = 12,
) -> dict[int, int]:
    """Ring-size histogram of an sp2 net, by tracing the faces of its graph.

    Not "the shortest cycle through each bond", which is the obvious
    approach and quietly wrong here: every bond of a 5-8-5 divacancy is
    shared with one of its two pentagons, so the shortest cycle through it
    is the pentagon and the octagon is never reported at all. Tracing faces
    finds it, because each directed bond belongs to exactly one face.

    The trace turns consistently at every atom -- always the next neighbour
    round the local surface normal -- which is what makes the walk close on
    a ring rather than wander. On a closed net the faces are the rings and
    nothing else; an open one (a ribbon, a finite tube) also traces its
    boundary as one long cycle, which ``max_size`` discards.

    Parameters
    ----------
    positions
        ``(n, 3)`` atom positions; the cyclic order is geometric.
    bonds
        Minimum-image bond pairs, so a periodic seam is one ring and not
        two dangling paths.
    box
        Cell edges, ``0`` on a free axis, as :func:`periodic_box` returns.
    max_size
        Rings longer than this are boundary walks, not rings.
    """
    n_atoms = int(max((max(pair) for pair in bonds), default=-1)) + 1
    table = _neighbours(bonds, n_atoms)
    normals = _local_frames(np.asarray(positions, dtype=float), table, box)

    order: dict[int, list[int]] = {}
    for index, neighbours in enumerate(table):
        others = sorted(neighbours)
        if len(others) < 2:
            order[index] = others
            continue
        normal = normals[index]
        first = fm.minimum_image(positions[others[0]] - positions[index], box)
        axis_u = first - normal * float(np.dot(first, normal))
        axis_u = axis_u / (np.linalg.norm(axis_u) + 1e-12)
        axis_v = np.cross(normal, axis_u)
        angles = []
        for k in others:
            offset = fm.minimum_image(positions[k] - positions[index], box)
            angles.append(float(np.arctan2(np.dot(offset, axis_v),
                                           np.dot(offset, axis_u))))
        order[index] = [k for _, k in sorted(zip(angles, others, strict=True))]

    rings: list[list[int]] = []
    unvisited = {(i, j) for pair in bonds for i, j in (pair, pair[::-1])}
    # Each directed bond has exactly one successor, so the walks are the
    # orbits of a permutation and cannot run away; the cap is only there so
    # a malformed graph fails instead of hanging.
    cap = 2 * len(bonds) + 2
    while unvisited:
        start = next(iter(unvisited))
        walk: list[int] = []
        current = start
        while current in unvisited and len(walk) <= cap:
            unvisited.discard(current)
            source, target = current
            walk.append(source)
            ring = order[target]
            if len(ring) < 2:
                break
            current = (target, ring[(ring.index(source) + 1) % len(ring)])
            if current == start:
                rings.append(walk)
                break
        else:
            continue
    return dict(Counter(len(r) for r in rings if len(r) <= max_size))


def _neighbours(bonds: list[tuple[int, int]], n_atoms: int) -> list[set[int]]:
    table: list[set[int]] = [set() for _ in range(n_atoms)]
    for i, j in bonds:
        table[i].add(j)
        table[j].add(i)
    return table


def _carbon_bonds(atoms: Atoms) -> list[tuple[int, int]]:
    """C-C bonds a whole ring in from any edge -- the only sites these edits mean.

    Both carbons, *and* every carbon they touch, must have three carbon
    neighbours. Requiring it of the two alone is not enough: a bond one row
    in from a ribbon edge passes that test, and removing it leaves the edge
    carbon behind it with a single neighbour -- an atom on a stalk, which
    the relaxation then swings out to 140 deg and the quality gate reports,
    correctly, as broken. Counting carbon neighbours rather than all of
    them is what keeps a passivated edge from looking three-coordinated
    because of its hydrogen.
    """
    bonds = [(i, j) for i, j, _ in guess_bonds(atoms)]
    symbols = atoms.get_chemical_symbols()
    table = _neighbours(bonds, len(atoms))
    carbon_coordination = [
        sum(1 for k in neighbours if symbols[k] == "C") if symbols[index] == "C" else 0
        for index, neighbours in enumerate(table)
    ]
    return [
        (i, j) for i, j in bonds
        if carbon_coordination[i] == 3 and carbon_coordination[j] == 3
        and all(carbon_coordination[k] == 3 for k in (table[i] | table[j]) - {i, j})
    ]


def _spread_sites(
    atoms: Atoms,
    count: int,
    rng: np.random.Generator,
    what: str,
) -> list[tuple[int, int]]:
    """Pick ``count`` C-C bonds no two of which are within one strain field."""
    candidates = _carbon_bonds(atoms)
    if not candidates:
        raise ValueError(
            f"No three-coordinated C-C bond to place a {what} on: the "
            "structure is too small, or every carbon is on an edge."
        )
    positions = atoms.get_positions()
    box = periodic_box(atoms)
    order = rng.permutation(len(candidates))
    chosen: list[tuple[int, int]] = []
    centres: list[np.ndarray] = []
    for index in order:
        i, j = candidates[int(index)]
        centre = positions[i] + 0.5 * fm.minimum_image(positions[j] - positions[i], box)
        if any(
            np.linalg.norm(fm.minimum_image(centre - other, box))
            < MIN_DEFECT_SEPARATION
            for other in centres
        ):
            continue
        chosen.append((i, j))
        centres.append(centre)
        if len(chosen) == count:
            return chosen
    raise ValueError(
        f"Only {len(chosen)} of {count} {what} sites fit with "
        f"{MIN_DEFECT_SEPARATION:.0f} Å between them. Build a longer or wider "
        "structure, or ask for fewer."
    )


def _rotate_bond(positions: np.ndarray, i: int, j: int,
                 normals: np.ndarray, box: np.ndarray) -> None:
    """Rotate one bond 90 deg about the local surface normal, in place.

    The normals come from :func:`_local_frames`, which has already made
    their signs agree across the structure. Computing them here from each
    atom's own bond vectors instead is where this first went wrong: the
    cross product's sign follows the arbitrary order the neighbours come
    out of a set in, so the two ends' normals could arrive antiparallel,
    average to nearly zero, and normalise to a direction with no meaning.
    When that direction happened to lie along the bond, the rotation moved
    nothing at all and the "defect" was a pristine lattice.
    """
    offset = fm.minimum_image(positions[j] - positions[i], box)
    midpoint = positions[i] + 0.5 * offset
    axis = normals[i] + normals[j]
    length = float(np.linalg.norm(axis))
    axis = axis / length if length > 1e-6 else np.array([0.0, 0.0, 1.0])
    # Rodrigues at 90 deg: cos is 0 and sin is 1, so the matrix collapses
    # to the outer product plus the cross-product term.
    x, y, z = axis
    rotation = np.array([
        [x * x, x * y - z, x * z + y],
        [y * x + z, y * y, y * z - x],
        [z * x - y, z * y + x, z * z],
    ])
    half = rotation @ (0.5 * offset)
    positions[i] = midpoint - half
    positions[j] = midpoint + half


def _reconstruct_divacancy(
    remap: dict[int, int],
    left: list[int],
    right: list[int],
) -> list[tuple[int, int]]:
    """Close the four dangling bonds around a divacancy into the 5-8-5.

    The new bonds join the two remaining neighbours of **the same** removed
    atom -- not a neighbour of one to a neighbour of the other, which is
    the plausible-sounding pairing and gives the wrong structure. Read it
    off the hexagons: ``a1`` and ``a2`` are two of the three corners of the
    hexagon whose third corner was the removed atom ``A``, so bonding them
    closes that hexagon short by one and makes a pentagon. Bonding ``a1``
    to a neighbour of ``B`` instead closes a hexagon short by *two* and
    makes a square -- which is what this produced when it paired by
    distance across the hole (measured: rings 4, 8 and 10 where 5, 8, 5
    belong).

    With both pentagons made, what is left of the two hexagons that shared
    the removed bond is a single eight-membered ring. That is the 5-8-5.
    """
    return [
        (remap[left[0]], remap[left[1]]),
        (remap[right[0]], remap[right[1]]),
    ]


def _carbon_geometry(
    positions: np.ndarray,
    bonds: list[tuple[int, int]],
    symbols: list[str],
    box: np.ndarray,
) -> dict[str, float | int]:
    """The sp2 report for the **carbon skeleton**, hydrogens left out.

    A passivated ribbon's C-H bonds are 1.09 Å and its hydrogens sit about
    1.8 Å from the carbons next along the edge. Both are correct, and both
    are outside a window that describes sp2 carbon, so measuring them
    alongside the C-C bonds reported a perfectly good ribbon as broken on
    the strength of its passivation. The window is about the skeleton, so
    the measurement is too.
    """
    carbons = [index for index, symbol in enumerate(symbols) if symbol == "C"]
    remap = {old: new for new, old in enumerate(carbons)}
    pairs = [
        (remap[i], remap[j]) for i, j in bonds
        if symbols[i] == "C" and symbols[j] == "C"
    ]
    return geometry_report(np.asarray(positions)[carbons], pairs, box=box)


def apply_lattice_edits(
    atoms: Atoms,
    defects: list[dict[str, Any]] | None = None,
    roughness: float = 0.0,
    bond: float = CC_BOND,
    seed: int | None = None,
    relax_iterations: int = 1200,
) -> Atoms:
    """Put defects and corrugation on a structure built from an exact lattice.

    Parameters
    ----------
    atoms
        A pristine lattice structure (not mutated). Its ``pbc`` and cell
        are honoured: a seam bond is measured and relaxed across the seam.
    defects
        Specs as the rest of the framework spells them:
        ``{"type": "stone_wales", "count": n}`` and
        ``{"type": "divacancy", "count": n}``. Divacancies are cut first,
        since they renumber the atoms a rotation would otherwise be
        holding indices into.
    roughness
        RMS out-of-plane corrugation (Å), applied **after** relaxation so
        it survives it -- see
        :func:`~nanocarbon_lab.builders.fullerene_mesh.apply_surface_roughness`.
    bond
        Equilibrium C-C length for the relaxation.
    seed
        Chooses the defect sites and the corrugation.
    relax_iterations
        Cap on the L-BFGS-B iterations. The pristine lattice starts at the
        force field's own minimum, so only the defect neighbourhoods
        actually move and this converges long before the cap.

    Returns
    -------
    ase.Atoms
        A copy with ``info["ring_counts"]``, ``info["bonds"]``,
        ``info["geometry"]`` and ``info["defects"]`` refreshed to describe
        what is really there.
    """
    specs = list(defects or [])
    n_sw = sum(int(s.get("count", 0)) for s in specs if s.get("type") == "stone_wales")
    n_dv = sum(int(s.get("count", 0)) for s in specs if s.get("type") == "divacancy")
    unknown = {str(s.get("type")) for s in specs} - {"stone_wales", "divacancy"}
    if unknown:
        raise ValueError(
            f"Unknown defect type(s) {sorted(unknown)} for a lattice structure; "
            "only 'stone_wales' and 'divacancy' are defined here."
        )
    if roughness < 0.0:
        raise ValueError("roughness must be non-negative.")
    if not (n_sw or n_dv or roughness > 0.0):
        return atoms

    rng = make_rng(seed)
    out = atoms.copy()
    out.info = {**atoms.info}
    log: list[dict[str, Any]] = list(atoms.info.get("defects", []))
    box = periodic_box(out)
    extra_bonds: list[tuple[int, int]] = []

    # --- divacancies. Every site is chosen, and every dangling atom
    #     recorded, in one pass over the pristine structure: picking them
    #     one at a time would let the second land next to the first, since
    #     the separation test only sees the sites of its own call.
    if n_dv:
        sites = _spread_sites(out, n_dv, rng, "divacancy")
        positions = out.get_positions()
        table = _neighbours([(a, b) for a, b, _ in guess_bonds(out)], len(out))
        removed = {index for pair in sites for index in pair}
        keep = [index for index in range(len(out)) if index not in removed]
        remap = {old: new for new, old in enumerate(keep)}
        for i, j in sites:
            extra_bonds += _reconstruct_divacancy(
                remap, sorted(table[i] - removed), sorted(table[j] - removed))
            log.append({"type": "divacancy", "removed": [int(i), int(j)],
                        "reconstructed": "5-8-5"})
        info = {**out.info}
        out = out[keep]
        out.info = info

    # --- Stone-Wales rotations, all sites chosen before any atom moves so
    #     the separation test sees the pristine geometry
    if n_sw:
        sites = _spread_sites(out, n_sw, rng, "Stone-Wales")
        positions = out.get_positions()
        table = _neighbours([(a, b) for a, b, _ in guess_bonds(out)], len(out))
        normals = _local_frames(positions, table, box)
        for i, j in sites:
            _rotate_bond(positions, i, j, normals, box)
            log.append({"type": "stone_wales", "bond": [int(i), int(j)]})
        out.set_positions(positions)

    # --- one relaxation for all of it
    bond_pairs = {
        (min(a, b), max(a, b)) for a, b, _ in guess_bonds(out)
    } | {(min(a, b), max(a, b)) for a, b in extra_bonds}
    symbols = out.get_chemical_symbols()
    equilibrium = {
        pair: (bond if symbols[pair[0]] == symbols[pair[1]] == "C" else CH_BOND)
        for pair in bond_pairs
    }
    positions = fm.relax_shell(
        out.get_positions(),
        bond_pairs,
        equilibrium=equilibrium,
        box=box,
        max_iterations=relax_iterations,
    )
    if roughness > 0.0:
        positions = fm.apply_surface_roughness(
            positions, bond_pairs, roughness, rng, equilibrium=bond, box=box
        )
        log.append({"type": "roughness", "sigma": float(roughness)})
    out.set_positions(positions)

    bonds = sorted(bond_pairs)
    counts = ring_census(positions, bonds, box=box)
    geometry = _carbon_geometry(positions, bonds, symbols, box)
    family = "sp2" if set(counts) <= {6} else "haeckelite"
    verdict, why = sp2_quality(geometry, family)
    if verdict == "broken":
        # A warning, not an exception: unlike a sweep that tears its wall,
        # this structure is intact and its topology is exactly what was
        # asked for -- it is the crowding that is unphysical, and the
        # caller can see from the numbers whether it matters to them.
        warnings.warn(
            f"{len(log)} edits leave the sheet strained past the sp2 window: "
            f"{why} Space them further apart, or build a larger structure.",
            stacklevel=2,
        )
    out.info.update({
        "bonds": [[int(i), int(j)] for i, j in bonds],
        "ring_counts": counts,
        "geometry": geometry,
        "defects": log,
        # A flat octagon's interior angle is 135 deg exactly, and a
        # pentagon's 108: a correct 5-8-5 sits on both edges of the
        # pristine sp2 window and is reported broken by it. The wider
        # window the haeckelite lattices are judged against is the one
        # that describes a sheet whose non-hexagons are deliberate, so a
        # structure that now has some is judged there.
        "quality_family": family,
    })
    return out
