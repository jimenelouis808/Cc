"""Realistic capped / defected carbon nanotubes ("elongated fullerenes").

:func:`build_cnt` (see :mod:`nanocarbon_lab.builders.cnt`) produces an
open, infinitely-periodic tube -- exact chirality, but no ends, so no
convex dome and no way to place a genuinely closed-shell defect like a
divacancy. :func:`build_capped_cnt` instead builds a **finite, fully
closed** shell: a straight or gently bent cylindrical body terminated at
both ends by hemispherical fullerene domes, with pentagon/heptagon/octagon
defects composed in on request. Every ring in the returned structure is
guaranteed topologically consistent with Euler's polyhedron theorem (see
:mod:`nanocarbon_lab.builders.fullerene_mesh` for the construction and
why this is provably true, not just usually true) -- this is the entry
point for "picture-realistic" capped nanotubes intended for rendering
(Blender export, journal-cover style images) rather than for exact-(n, m)
periodic DFT/MD cells.
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell
from ..utils.rng import make_rng
from . import fullerene_mesh as fm

DefectKind = Literal["stone_wales", "divacancy"]


class DefectSpec(TypedDict, total=False):
    """One defect request for :func:`build_capped_cnt`.

    ``type`` -- ``"stone_wales"`` (2 pentagons + 2 heptagons, an in-plane
    90 deg bond rotation) or ``"divacancy"`` (1 octagon + 2 pentagons, a
    reconstructed two-atom vacancy). ``count`` -- how many independent
    instances to place (default 1), spaced apart automatically.
    """

    type: DefectKind
    count: int


def _neighbourhood(
    adjacency: dict[int, set[int]], seeds: set[int], radius: int
) -> set[int]:
    """BFS ball of the given ``radius`` around ``seeds`` in ``adjacency``."""
    frontier = set(seeds)
    visited = set(seeds)
    for _ in range(radius):
        nxt: set[int] = set()
        for v in frontier:
            nxt |= adjacency.get(v, set())
        frontier = nxt - visited
        visited |= nxt
    return visited


def _apply_defects(
    mesh: fm.Mesh,
    defects: list[DefectSpec],
    rng: np.random.Generator,
    separation: int,
) -> tuple[fm.Mesh, list[dict]]:
    """Apply a sequence of mesh-level defects, keeping them well separated.

    Each defect is placed on a random interior edge whose local
    neighbourhood (out to ``separation`` mesh hops) is still pristine
    (untouched hexagons only), so defects never merge into an unintended
    larger/compound ring and stay far enough apart for
    :func:`fullerene_mesh.relax_shell` to settle cleanly.
    """
    log: list[dict] = []
    exclude: set[int] = set()
    for spec in defects:
        kind = spec["type"]
        count = spec.get("count", 1)
        for _ in range(count):
            adjacency = fm.mesh_adjacency(mesh[1])
            degree = {v: len(ns) for v, ns in adjacency.items()}
            edge = fm.pick_interior_edge(
                mesh[1], rng, exclude=exclude, required_degree=degree
            )
            if edge is None:
                raise RuntimeError(
                    f"Could not find a clear site for a {kind!r} defect "
                    "(mesh too small or too crowded with existing defects). "
                    "Use a larger n_body_rings/freq or fewer defects."
                )
            u, v = edge
            if kind == "stone_wales":
                mesh = fm.edge_flip(mesh, u, v)
                touched = {u, v}
            elif kind == "divacancy":
                mesh, remap = fm.contract_edge(mesh, u, v)
                exclude = {remap[x] for x in exclude if x in remap}
                touched = {remap[u]}
            else:
                raise ValueError(f"Unknown defect type {kind!r}.")
            adjacency = fm.mesh_adjacency(mesh[1])
            exclude |= _neighbourhood(adjacency, touched, radius=separation)
            log.append({"type": kind, "site": sorted(int(x) for x in touched)})
    return mesh, log


def build_capped_cnt(
    n_body_rings: int = 8,
    freq: int = 3,
    radius: float = 6.5,
    bond: float = CC_BOND,
    bend_angle: float = 0.0,
    defects: Optional[list[DefectSpec]] = None,
    defect_separation: int = 3,
    vacuum: float = DEFAULT_VACUUM_1D,
    relax_steps: int = 2500,
    seed: Optional[int] = None,
) -> Atoms:
    """Build a finite, fully-capped carbon nanotube (an elongated fullerene).

    Two hemispherical fullerene domes (6 pentagons each, the rest
    hexagons -- see :mod:`nanocarbon_lab.builders.fullerene_mesh`) cap a
    straight or gently bent cylindrical hexagonal body. Every atom ends up
    3-coordinate and every ring size is exactly known and Euler-consistent
    by construction (``sum(6 - ring_size) == 12`` over the whole shell,
    always -- assert on ``atoms.info["ring_counts"]`` in a test if you
    want to double-check a specific build).

    Parameters
    ----------
    n_body_rings
        Number of lattice rings in the cylindrical body seed (>= 2).
        Controls tube length: roughly proportional to ``n_body_rings``
        for fixed ``freq``.
    freq
        Geodesic subdivision frequency (>= 1). Controls tube diameter and
        hexagon density: higher ``freq`` gives a wider tube with finer
        (smaller, more numerous) hexagons for the same physical radius.
    radius
        Target body radius in Å (the caps are hemispheres of the same
        radius). Actual radius after relaxation may drift by a few
        percent as bond lengths settle to ``bond``.
    bond
        Equilibrium C-C bond length (Å) used both to set the initial
        scale and as the relaxation target.
    bend_angle
        Total bend of the cylindrical body, in radians, swept as a
        circular arc (0 = straight). Purely a smooth elastic bend (no
        extra defects); combine with a ``"stone_wales"`` or
        ``"divacancy"`` entry in ``defects`` for a topologically-marked
        kink, as seen in TEM images of real nanotube elbows.
    defects
        List of :class:`DefectSpec` (``{"type": ..., "count": ...}``)
        requests, applied in order. ``"stone_wales"`` adds a 5-7-7-5
        pair (2 pentagons + 2 heptagons); ``"divacancy"`` adds a
        reconstructed 5-8-5 (1 octagon + 2 pentagons). Sites are chosen
        at random (seeded) pristine-hexagon locations in the cylindrical
        body, kept apart by ``defect_separation``.
    defect_separation
        Minimum mesh-hop distance enforced between defects (and between
        defects and the poles), so rings from different defects never
        overlap or merge.
    vacuum
        Vacuum padding (Å) added around the finite structure's bounding
        box on all three axes.
    relax_steps
        Steepest-descent steps for :func:`fullerene_mesh.relax_shell`.
        Increase for larger/more defected/more bent structures if the
        reported bond-length spread in ``atoms.info["bond_length"]`` is
        wider than desired.
    seed
        RNG seed controlling defect site selection. Required for
        reproducible builds when ``defects`` is non-empty.

    Returns
    -------
    ase.Atoms
        Finite (non-periodic) structure. ``atoms.info`` carries:

        - ``structure_type``: ``"capped_cnt"``
        - ``ring_counts``: ``{ring_size: count}`` over the whole shell
        - ``rings``: list of atom-index lists, one per ring
        - ``bonds``: sorted list of ``[i, j]`` bonded atom-index pairs
        - ``bond_length``: ``{min, mean, max, std}`` after relaxation
        - ``defect_log``: list of ``{type, site}`` records in placement order
        - ``radius``, ``bend_angle``, ``n_body_rings``, ``freq``, ``bond``, ``seed``

    Raises
    ------
    ValueError
        For non-physical parameters (``n_body_rings < 2``, ``freq < 1``,
        ``radius <= 0``).
    RuntimeError
        If a requested defect cannot find a sufficiently separated,
        pristine site.
    """
    if n_body_rings < 2:
        raise ValueError("n_body_rings must be >= 2.")
    if freq < 1:
        raise ValueError("freq must be >= 1.")
    if radius <= 0:
        raise ValueError("radius must be positive.")

    rng = make_rng(seed)
    mesh = fm.seed_capsule_mesh(n_body_rings)
    mesh = fm.subdivide_mesh(mesh, freq)

    defect_log: list[dict] = []
    if defects:
        mesh, defect_log = _apply_defects(mesh, defects, rng, defect_separation)

    raw_pos, bonds, rings = fm.dual_honeycomb(mesh)
    ring_counts = fm.ring_size_histogram(rings)
    deficit = sum((6 - size) * count for size, count in ring_counts.items())
    if deficit != 12:
        raise RuntimeError(
            f"Internal error: ring deficit sum = {deficit}, expected 12 "
            "(broken mesh topology)."
        )

    # Seed-mesh body half-length in the same raw units as raw_pos: the
    # z-extent of the antiprism ring stack, pulled in slightly so cap
    # atoms (from the pole triangle fans) are unambiguously classified.
    seed_half_length = (n_body_rings - 1) / 2.0 * 0.92
    positions = fm.capsule_project(
        raw_pos, radius=radius, half_length=seed_half_length, bend_angle=bend_angle
    )

    bond_lengths = np.array(
        [np.linalg.norm(positions[a] - positions[b]) for a, b in bonds]
    )
    positions *= bond / bond_lengths.mean()
    positions = fm.relax_shell(positions, bonds, equilibrium=bond, steps=relax_steps)

    bond_lengths = np.array(
        [np.linalg.norm(positions[a] - positions[b]) for a, b in bonds]
    )

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions, pbc=False)
    extents = positions.max(axis=0) - positions.min(axis=0)
    atoms.set_cell(np.diag(extents + vacuum))
    center_in_cell(atoms, axes=(0, 1, 2))

    atoms.info.update(
        {
            "structure_type": "capped_cnt",
            "n_body_rings": n_body_rings,
            "freq": freq,
            "radius": radius,
            "bond": bond,
            "bend_angle": bend_angle,
            "seed": seed,
            "ring_counts": {int(k): int(v) for k, v in ring_counts.items()},
            "rings": [[int(a) for a in r] for r in rings],
            "bonds": sorted([int(a), int(b)] for a, b in bonds),
            "bond_length": {
                "min": float(bond_lengths.min()),
                "mean": float(bond_lengths.mean()),
                "max": float(bond_lengths.max()),
                "std": float(bond_lengths.std()),
            },
            "defect_log": defect_log,
        }
    )
    return atoms
