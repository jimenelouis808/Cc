"""Realistic capped / defected carbon nanotubes ("elongated fullerenes").

:func:`build_cnt` (see :mod:`nanocarbon_lab.builders.cnt`) produces an
open, infinitely-periodic tube -- exact chirality, but no ends, so no
convex dome and no way to place a genuinely closed-shell defect like a
divacancy. :func:`build_capped_cnt` instead builds a **finite, fully
closed** shell: a straight or gently bent cylindrical body terminated at
both ends by hemispherical fullerene domes, with pentagon/heptagon/octagon
defects composed in on request.

Two invariants are enforced, and both are checked by the test suite:

* **Topology.** Every ring size is consistent with Euler's polyhedron
  theorem by construction -- see
  :mod:`nanocarbon_lab.builders.fullerene_mesh` for why this is provable
  rather than merely likely.
* **Geometry.** The relaxed structure has genuine sp2 geometry: bond
  lengths within a few thousandths of 1.42 Å, bond angles clustered on
  120 deg (dipping to ~108 deg inside cap pentagons, which is correct),
  and no non-bonded contact closer than a bond length.

The construction order matters and is deliberate:

1. seed polyhedron -> subdivide (sets diameter and hexagon density),
2. apply defect edits **on the mesh** (see ``fullerene_mesh``),
3. project + Laplacian-smooth the *mesh* onto the target capsule, so the
   triangulation -- and therefore the honeycomb dual -- is near-uniform
   before any atom exists,
4. take the dual to get atoms/bonds/rings, scale to ``bond``,
5. relax with the valence force field,
6. optionally bend, then re-relax with the ends restrained.

Step 3 is the one that is easy to get wrong: taking the dual first and
projecting the *atoms* afterwards leaves the hexagons badly unequal, and
on structures beyond ~1000 atoms the optimiser cannot recover -- the
shell folds through itself. Smoothing the mesh first is what makes the
large, defected and bent cases converge.
"""

from __future__ import annotations

import warnings
from typing import Literal, TypedDict

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell
from ..utils.rng import make_rng
from . import centerline as cl
from . import fullerene_mesh as fm

DefectKind = Literal["stone_wales", "divacancy"]

# Beyond this the uniform elastic bend model stops being physical: a real
# nanotube buckles into a localised kink rather than straining smoothly.
MAX_PHYSICAL_BEND = 1.0

# Axial length contributed by one body ring, as a multiple of the tube
# radius. Measured across freq 2-4 and 10-20 rings, where the ratio is
# 1.067-1.088; the small excess over 1 is the two end caps.
RING_RISE = 1.07


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
    larger ring and stay far enough apart for the relaxation to settle
    cleanly.
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


def _bend_positions(positions: np.ndarray, angle: float) -> np.ndarray:
    """Sweep a straight, z-aligned structure onto a circular arc.

    Arc length is set equal to the original axial span, so the bend is
    (to first order) isometric along the tube axis and bond lengths on
    the neutral surface are preserved; the inner wall is compressed and
    the outer wall stretched by ``+-r_tube / arc_radius``, exactly as in
    an elastically bent tube.
    """
    z = positions[:, 2]
    half = (z.max() - z.min()) / 2.0
    if half <= 0:
        return positions
    arc_radius = (2.0 * half) / angle
    theta = z / arc_radius
    ct, st = np.cos(theta), np.sin(theta)
    bent = np.empty_like(positions)
    bent[:, 0] = (arc_radius + positions[:, 0]) * st
    bent[:, 1] = positions[:, 1]
    bent[:, 2] = (arc_radius + positions[:, 0]) * ct - arc_radius
    return bent


def geometry_report(
    positions: np.ndarray,
    bonds: list[tuple[int, int]],
    box: float | None = None,
) -> dict[str, float | int]:
    """Measure how close a structure is to ideal sp2 geometry.

    ``box`` set makes every distance minimum-image, so a periodic cell is
    measured across its seams rather than reporting box-length "bonds".

    Returns bond-length and bond-angle statistics plus the number of
    non-bonded contacts below 2.0 Å (there should be none: the shortest
    genuinely non-bonded distance in graphitic carbon is the ~2.46 Å 1-3
    separation). Attached to every built structure as
    ``atoms.info["geometry"]`` so callers can assert on quality rather
    than trusting the builder.
    """
    from collections import defaultdict

    from scipy.spatial import cKDTree

    pos = np.asarray(positions, dtype=float)
    lengths = np.array([
        np.linalg.norm(fm.minimum_image(pos[b] - pos[a], box)) for a, b in bonds
    ])

    nbrs: dict[int, list[int]] = defaultdict(list)
    for a, b in bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    angles = []
    for centre, ns in nbrs.items():
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                v1 = fm.minimum_image(pos[ns[i]] - pos[centre], box)
                v2 = fm.minimum_image(pos[ns[j]] - pos[centre], box)
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    angles = np.array(angles)

    bonded = set()
    for a, b in bonds:
        bonded.add((a, b))
        bonded.add((b, a))
    tree = cKDTree(pos) if box is None else cKDTree(np.mod(pos, box), boxsize=box)
    close = tree.query_pairs(r=2.0, output_type="ndarray")
    n_clashes = sum(
        1 for a, b in close if (int(a), int(b)) not in bonded
    ) if len(close) else 0

    return {
        "bond_min": float(lengths.min()),
        "bond_mean": float(lengths.mean()),
        "bond_max": float(lengths.max()),
        "bond_std": float(lengths.std()),
        "angle_min": float(angles.min()),
        "angle_mean": float(angles.mean()),
        "angle_max": float(angles.max()),
        "angle_std": float(angles.std()),
        "n_close_contacts": int(n_clashes),
    }


def build_capped_cnt(
    n_body_rings: int = 8,
    freq: int = 3,
    target_radius: float | None = None,
    bond: float = CC_BOND,
    bend_angle: float = 0.0,
    shape: cl.Shape = "straight",
    waviness: float = 0.7,
    max_strain: float = cl.DEFAULT_MAX_STRAIN,
    shape_points: int = 9,
    helix_turns: float = 1.5,
    helix_radius: float | None = None,
    helix_pitch: float | None = None,
    defects: list[DefectSpec] | None = None,
    defect_separation: int = 3,
    vacuum: float = DEFAULT_VACUUM_1D,
    smoothing_iterations: int = 60,
    relax_iterations: int = 3000,
    seed: int | None = None,
) -> Atoms:
    """Build a finite, fully-capped carbon nanotube (an elongated fullerene).

    Two hemispherical fullerene domes (6 pentagons each) cap a straight or
    gently bent cylindrical hexagonal body, with optional Stone-Wales and
    divacancy defects. The result has both correct ring topology and
    genuine sp2 geometry (see the module docstring).

    Parameters
    ----------
    n_body_rings
        Lattice rings along the body (>= 2). Controls tube length.
    freq
        Geodesic subdivision frequency (>= 1). **Controls the diameter**,
        which is quantised by the lattice: the body circumference must fit
        a whole number of hexagons, so the radius is
        ``5 * freq * sqrt(3) * bond / (2*pi)`` -- about ``1.96 * freq`` Å
        for graphitic carbon. Ignored when ``target_radius`` is given.
    target_radius
        Convenience alternative to ``freq``: the nearest realisable
        ``freq`` is chosen for this radius (Å). The radius you actually
        get is reported as ``atoms.info["radius"]``; it can differ by up
        to ~1 Å because the lattice is quantised, exactly as a real
        ``(n, m)`` nanotube's diameter is fixed by its chiral indices.
    bond
        Equilibrium C-C bond length (Å), the relaxation target.
    bend_angle
        Total bend of the body in radians (0 = straight). The bend is
        imposed geometrically after the straight relaxation and then
        re-relaxed with both caps restrained, so bond lengths recover
        while the bend is retained. Values above
        :data:`MAX_PHYSICAL_BEND` are rejected: a real nanotube buckles
        into a localised kink rather than straining uniformly, which this
        smooth-arc model does not represent.
    defects
        List of :class:`DefectSpec` (``{"type": ..., "count": ...}``).
        ``"stone_wales"`` adds a 5-7-7-5 pair (2 pentagons + 2 heptagons);
        ``"divacancy"`` adds a reconstructed 5-8-5 (1 octagon + 2
        pentagons). Sites are chosen at random (seeded) pristine-hexagon
        locations, kept apart by ``defect_separation``.
    defect_separation
        Minimum mesh-hop distance enforced between defects.
    vacuum
        Vacuum padding (Å) around the structure's bounding box.
    smoothing_iterations
        Laplacian-on-capsule sweeps applied to the mesh before taking the
        dual (see module docstring, step 3).
    relax_iterations
        L-BFGS iterations per relaxation cycle.
    seed
        RNG seed controlling defect site selection.

    Returns
    -------
    ase.Atoms
        Finite (non-periodic) structure. ``atoms.info`` carries
        ``ring_counts``, ``rings``, ``bonds``, ``geometry`` (see
        :func:`geometry_report`), ``defect_log``, ``radius``,
        ``length``, ``bend_angle``, and the build parameters.

    Raises
    ------
    ValueError
        For non-physical parameters.
    RuntimeError
        If a requested defect cannot find a sufficiently separated site,
        or if the internal Euler check fails.
    """
    if n_body_rings < 2:
        raise ValueError("n_body_rings must be >= 2.")
    if target_radius is not None:
        freq = fm.freq_for_radius(target_radius, bond=bond)
    if freq < 1:
        raise ValueError("freq must be >= 1.")
    if bond <= 0:
        raise ValueError("bond must be positive.")
    if bend_angle > 0 and shape != "straight":
        raise ValueError(
            f"bend_angle and shape={shape!r} cannot be combined. bend_angle "
            "sweeps a z-aligned tube onto a planar arc, but after a shape "
            "sweep the tube's axis no longer follows z, so applying both "
            "flattens the structure. Use shape='arc' for a simple bend."
        )
    if not 0.0 <= bend_angle <= MAX_PHYSICAL_BEND:
        raise ValueError(
            f"bend_angle must be in [0, {MAX_PHYSICAL_BEND}] rad; got {bend_angle}. "
            "Beyond that a real nanotube buckles into a kink rather than "
            "bending smoothly, which this model does not represent."
        )

    # A helix given real dimensions sizes the *tube* rather than the other
    # way round: the coil's arc length says how much tube is needed, so
    # n_body_rings is derived instead of trusted. Without this the path
    # would be rescaled to whatever tube the caller happened to ask for,
    # and the requested coil radius and pitch would silently not hold.
    explicit_helix = shape == "helix" and helix_radius is not None
    if explicit_helix:
        pitch = helix_pitch if helix_pitch is not None else 2.0 * helix_radius
        arc = cl.helix_arc_length(helix_radius, pitch, helix_turns)
        n_body_rings = max(4, int(round(arc / (RING_RISE * fm.radius_for_freq(freq, bond)))))

    rng = make_rng(seed)
    mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(n_body_rings), freq)

    defect_log: list[dict] = []
    if defects:
        mesh, defect_log = _apply_defects(mesh, defects, rng, defect_separation)

    # Work in units where the capsule radius is 1; the true scale is set
    # afterwards by normalising the mean bond length to `bond`.
    half_length = (n_body_rings - 1) / 2.0
    mesh = fm.smooth_mesh_on_capsule(
        mesh, radius=1.0, half_length=half_length, iterations=smoothing_iterations
    )

    positions, bond_set, rings = fm.dual_honeycomb(mesh)
    bonds = sorted(bond_set)
    ring_counts = fm.ring_size_histogram(rings)
    deficit = sum((6 - size) * count for size, count in ring_counts.items())
    if deficit != 12:
        raise RuntimeError(
            f"Internal error: ring deficit sum = {deficit}, expected 12 "
            "(broken mesh topology)."
        )

    lengths = np.array([np.linalg.norm(positions[a] - positions[b]) for a, b in bonds])
    positions = positions * (bond / lengths.mean())
    positions = fm.relax_shell(
        positions, bond_set, equilibrium=bond, max_iterations=relax_iterations
    )

    # Measure radius/length on the straight tube: once bent, the structure
    # is no longer z-aligned and an axial slice would report the bend's arc
    # radius rather than the tube's own.
    straight = positions - positions.mean(axis=0)
    tube_length = float(straight[:, 2].max() - straight[:, 2].min())
    mid_slice = np.abs(straight[:, 2]) < 0.15 * max(tube_length, 1e-9)
    tube_radius = (
        float(np.linalg.norm(straight[mid_slice][:, :2], axis=1).mean())
        if mid_slice.any()
        else float(fm.radius_for_freq(freq, bond))
    )

    swept_strain = 0.0
    if shape != "straight":
        if not 0.0 <= waviness <= 1.0:
            raise ValueError(f"waviness must be in [0, 1]; got {waviness}.")
        if explicit_helix:
            # Dimensions were asked for explicitly, so they are honoured and
            # the resulting strain is reported rather than silently trimmed
            # away -- shrinking the coil would give the caller a different
            # structure than the one they specified.
            control = cl.helix_control_points(helix_radius, pitch, helix_turns)
            swept_strain = tube_radius * cl.helix_curvature(helix_radius, pitch)
            if swept_strain > cl.ARTISTIC_STRAIN_LIMIT:
                warnings.warn(
                    f"A coil of radius {helix_radius:.0f} Å around a "
                    f"{tube_radius:.1f} Å tube strains the outer wall by "
                    f"{swept_strain:.0%}, past the {cl.ARTISTIC_STRAIN_LIMIT:.0%} "
                    "sp2 limit. Real carbon nanocoils have coil radii of "
                    "hundreds of Å for this reason; widen helix_radius or "
                    "use a thinner tube (lower freq) for a physical result.",
                    UserWarning, stacklevel=2,
                )
        else:
            control = cl.shape_control_points(
                shape, rng, n_points=shape_points,
                amplitude=waviness, turns=helix_turns,
            )
            # Trim the path to what the lattice can physically survive, then
            # let the caller see what was actually achieved.
            if max_strain > cl.ARTISTIC_STRAIN_LIMIT:
                warnings.warn(
                    f"max_strain={max_strain:.0%} exceeds "
                    f"{cl.ARTISTIC_STRAIN_LIMIT:.0%}; bonds will stretch past the "
                    "sp2 range and the structure is no longer physically "
                    "meaningful (still fine for illustration).",
                    UserWarning, stacklevel=2,
                )
            control, swept_strain = cl.fit_to_strain_budget(
                control, tube_length, tube_radius, max_strain=max_strain
            )
        anchors = np.arange(len(positions))
        positions = cl.sweep_along_path(positions, control)
        positions = fm.relax_shell(
            positions, bond_set, equilibrium=bond,
            anchors=anchors, anchor_targets=positions, k_anchor=3.0,
            max_iterations=relax_iterations,
        )

    if bend_angle > 0:
        axial = positions[:, 2]
        lo, hi = axial.min(), axial.max()
        span = hi - lo
        # Restrain the outer 12% at each end so the imposed bend survives
        # relaxation instead of springing straight again.
        anchors = np.where(
            (axial < lo + 0.12 * span) | (axial > hi - 0.12 * span)
        )[0]
        positions = _bend_positions(positions, bend_angle)
        positions = fm.relax_shell(
            positions,
            bond_set,
            equilibrium=bond,
            anchors=anchors,
            anchor_targets=positions[anchors],
            max_iterations=relax_iterations,
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
            "bond": bond,
            "bend_angle": bend_angle,
            "shape": shape,
            "waviness": waviness,
            "path_strain": swept_strain,
            "helix_radius": helix_radius if explicit_helix else None,
            "helix_pitch": pitch if explicit_helix else None,
            "helix_turns": helix_turns if explicit_helix else None,
            "seed": seed,
            "radius": tube_radius,
            "radius_ideal": float(fm.radius_for_freq(freq, bond)),
            "length": tube_length,
            "ring_counts": {int(k): int(v) for k, v in ring_counts.items()},
            "rings": [[int(a) for a in r] for r in rings],
            "bonds": [[int(a), int(b)] for a, b in bonds],
            "geometry": geometry_report(positions, bonds),
            "defect_log": defect_log,
        }
    )
    return atoms
