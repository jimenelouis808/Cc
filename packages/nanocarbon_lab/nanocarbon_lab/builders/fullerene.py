"""Closed-cage fullerenes and the concentric onions built from them.

A fullerene is the ``half_length = 0`` limit of the capped tube: a sphere
rather than a capsule, so the two 6-pentagon caps meet with no cylindrical
body between them. Everything else -- the triangulated dual, the Euler
budget, the valence force field -- is the machinery already used by
:mod:`nanocarbon_lab.builders.capped_cnt`, so the same guarantees hold:
mesh vertex degree *is* carbon ring size, and ``sum(6 - ring_size) = 12``
falls out rather than being imposed.

What is specific here is the **seed**, because a sphere's seed decides
which fullerene you get. Both seeds below are triangulations of the
sphere whose vertices are the future carbon rings:

* the **icosahedron** (12 vertices, all degree 5) subdivided ``freq``
  times gives the Goldberg class-I series GP(f,0) -- C20, C80, C180,
  C320, C500;
* the **pentakis dodecahedron** (32 vertices: 12 degree-5 at the
  icosahedral directions, 20 degree-6 at the dodecahedral ones)
  subdivided ``freq`` times gives the class-II series GP(f,f) -- C60,
  C240, C540, C960.

The second family matters more than it looks. C60 itself is not in the
class-I series at any frequency, so an icosahedron seed cannot produce
the one fullerene everybody actually wants; and its radii, 3.55 Å per
frequency step, are what make a **nano-onion** possible at the graphitic
3.4 Å spacing. The class-I series steps by only ~2.05 Å, which no choice
of shells lands on 3.4.

Both seeds are built as convex hulls of points on the unit sphere, which
is the honest way to get a triangulation of a point set on a sphere: the
hull of points on a sphere *is* their Delaunay triangulation, so the
connectivity cannot be got wrong by hand-listing faces.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell
from ..utils.rng import make_rng
from . import fullerene_mesh as fm
from .capped_cnt import geometry_report

# Graphite interlayer spacing; what an onion's shells should sit at.
GRAPHITIC_GAP = 3.4

# Golden ratio, which places both seeds' vertices.
PHI = (1.0 + np.sqrt(5.0)) / 2.0

FullereneFamily = str  # "C60" (class II, GP(f,f)) or "C20" (class I, GP(f,0))

# Atoms in the smallest cage of each family; a frequency-f cage has
# `base * freq**2` atoms, because subdividing a triangulation by f
# multiplies its triangle count -- and the dual's atom count -- by f**2.
FAMILY_BASE_ATOMS = {"C20": 20, "C60": 60}


def icosahedron_mesh() -> fm.Mesh:
    """Regular icosahedron: 12 vertices of degree 5, 20 triangles.

    Its dual is the dodecahedron, i.e. C20 -- twelve pentagons and not a
    single hexagon, which is the smallest closed fullerene cage Euler
    allows.
    """
    raw = []
    for sign_a in (-1.0, 1.0):
        for sign_b in (-1.0, 1.0):
            raw.append((0.0, sign_a * 1.0, sign_b * PHI))
            raw.append((sign_a * 1.0, sign_b * PHI, 0.0))
            raw.append((sign_b * PHI, 0.0, sign_a * 1.0))
    return _hull_mesh(np.array(raw, dtype=float))


def pentakis_dodecahedron_mesh() -> fm.Mesh:
    """Pentakis dodecahedron: 32 vertices (12 of degree 5, 20 of degree 6).

    Its dual is the truncated icosahedron -- 12 pentagons, 20 hexagons,
    60 atoms: **C60**, the football. Built as the convex hull of the
    dodecahedron's 20 vertices together with the icosahedron's 12, all
    normalised onto the unit sphere, which puts a degree-5 vertex in the
    centre of each dodecahedral face exactly as the "kis" construction
    requires.
    """
    dodecahedron = []
    for sign_a in (-1.0, 1.0):
        for sign_b in (-1.0, 1.0):
            dodecahedron.append((0.0, sign_a / PHI, sign_b * PHI))
            dodecahedron.append((sign_a / PHI, sign_b * PHI, 0.0))
            dodecahedron.append((sign_b * PHI, 0.0, sign_a / PHI))
            for sign_c in (-1.0, 1.0):
                dodecahedron.append((sign_a, sign_b, sign_c))
    points = np.vstack([
        _normalise(np.unique(np.round(dodecahedron, 9), axis=0)),
        _normalise(icosahedron_mesh()[0]),
    ])
    return _hull_mesh(points)


def _normalise(points: np.ndarray) -> np.ndarray:
    return points / np.linalg.norm(points, axis=1)[:, None]


def _hull_mesh(points: np.ndarray) -> fm.Mesh:
    """Triangulate points on a sphere via their convex hull.

    Triangles come back with arbitrary winding, so each is flipped to face
    outward. Consistent orientation is not cosmetic: the dual walks the
    triangles around each vertex to order a ring's atoms, and a mesh with
    mixed winding orders them into a self-crossing polygon.
    """
    from scipy.spatial import ConvexHull

    points = _normalise(np.asarray(points, dtype=float))
    hull = ConvexHull(points)
    faces = []
    for tri in hull.simplices:
        a, b, c = points[tri]
        outward = np.cross(b - a, c - a) @ (a + b + c)
        faces.append(tri if outward > 0 else tri[::-1])
    return points, np.array(faces, dtype=int)


def build_fullerene(
    freq: int = 1,
    family: FullereneFamily = "C60",
    bond: float = CC_BOND,
    roughness: float = 0.0,
    vacuum: float = DEFAULT_VACUUM_1D,
    smoothing_iterations: int = 60,
    relax_iterations: int = 3000,
    seed: int | None = 0,
) -> Atoms:
    """Build a closed icosahedral fullerene cage.

    Parameters
    ----------
    freq
        Geodesic subdivision frequency (>= 1). The cage has
        ``FAMILY_BASE_ATOMS[family] * freq**2`` atoms, so this is the size
        control: ``family="C60"`` gives C60, C240, C540, C960 at
        ``freq`` 1, 2, 3, 4.
    family
        ``"C60"`` for the class-II series GP(f,f) -- the football and its
        larger relatives, radii stepping ~3.55 Å per frequency. ``"C20"``
        for the class-I series GP(f,0) -- C20, C80, C180, C320, radii
        stepping ~2.05 Å.
    bond
        Equilibrium C-C bond length (Å).
    roughness
        RMS out-of-plane corrugation (Å) applied after relaxation; ``0``
        leaves an ideal cage.
    vacuum
        Vacuum padding around the cage (Å).
    smoothing_iterations, relax_iterations
        As for :func:`nanocarbon_lab.builders.capped_cnt.build_capped_cnt`.
    seed
        RNG seed, used only by ``roughness``.

    Returns
    -------
    ase.Atoms
        The cage, with ``ring_counts`` (always exactly 12 pentagons),
        ``radius``, ``bonds``, ``rings`` and ``geometry`` in
        ``atoms.info``.

    Raises
    ------
    ValueError
        For ``freq < 1`` or an unknown ``family``.
    RuntimeError
        If the ring budget does not come to 12, which would mean the seed
        or the subdivision had broken the topology.
    """
    if freq < 1:
        raise ValueError("freq must be >= 1.")
    if family not in FAMILY_BASE_ATOMS:
        raise ValueError(
            f"family must be one of {sorted(FAMILY_BASE_ATOMS)}; got {family!r}. "
            "'C60' is the class-II series (C60, C240, C540...); 'C20' the "
            "class-I one (C20, C80, C180...). C60 is not reachable from the "
            "class-I seed at any frequency."
        )

    seed_mesh = (
        pentakis_dodecahedron_mesh() if family == "C60" else icosahedron_mesh()
    )
    mesh = fm.subdivide_mesh(seed_mesh, freq)
    # half_length=0 makes the capsule a sphere; everything downstream is
    # the capped-tube pipeline unchanged.
    mesh = fm.smooth_mesh_on_capsule(
        mesh, radius=1.0, half_length=0.0, iterations=smoothing_iterations
    )

    positions, bond_set, rings = fm.dual_honeycomb(mesh)
    bonds = sorted(bond_set)
    ring_counts = fm.ring_size_histogram(rings)
    deficit = sum((6 - size) * count for size, count in ring_counts.items())
    if deficit != 12:
        raise RuntimeError(
            f"Internal error: ring deficit sum = {deficit}, expected 12 "
            "(broken cage topology)."
        )

    lengths = np.array([np.linalg.norm(positions[a] - positions[b]) for a, b in bonds])
    positions = positions * (bond / lengths.mean())
    positions = fm.relax_shell(
        positions, bond_set, equilibrium=bond, max_iterations=relax_iterations
    )

    rng = make_rng(seed)
    if roughness > 0:
        positions = fm.apply_surface_roughness(
            positions, bond_set, roughness, rng, equilibrium=bond
        )

    centred = positions - positions.mean(axis=0)
    radius = float(np.linalg.norm(centred, axis=1).mean())

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions, pbc=False)
    extents = positions.max(axis=0) - positions.min(axis=0)
    atoms.set_cell(np.diag(extents + vacuum))
    center_in_cell(atoms, axes=(0, 1, 2))
    atoms.info.update(
        {
            "structure_type": "fullerene",
            "family": family,
            "freq": freq,
            "formula": f"C{len(positions)}",
            "radius": radius,
            "bond": bond,
            "roughness": roughness,
            "ring_counts": {int(k): int(v) for k, v in ring_counts.items()},
            "rings": [[int(a) for a in r] for r in rings],
            "bonds": [[int(a), int(b)] for a, b in bonds],
            "geometry": geometry_report(positions, bonds),
        }
    )
    return atoms


def build_nano_onion(
    n_shells: int = 3,
    inner_freq: int = 1,
    freq_step: int = 1,
    family: FullereneFamily = "C60",
    bond: float = CC_BOND,
    roughness: float = 0.0,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
    **shell_kwargs,
) -> Atoms:
    """Build a carbon nano-onion: concentric fullerene cages.

    The classic onion is C60@C240@C540 -- the ``family="C60"`` series at
    ``freq`` 1, 2, 3. That series is the right one because its radius
    steps ~3.55 Å per frequency, within a tenth of an Ångström of
    graphite's 3.4 Å interlayer spacing, so consecutive cages nest at
    very nearly the physical separation with no scaling fudge. The
    class-I ``"C20"`` series steps only ~2.05 Å and no choice of
    ``freq_step`` lands near 3.4.

    As with :func:`nanocarbon_lab.builders.assemblies.build_multiwall_cnt`,
    the shells are relaxed independently and then nested. The covalent
    force field has no dispersion term, so relaxing the assembly as a
    whole would simply collapse the cages into one another; what holds an
    onion together is exactly the term the model does not have. The
    achieved spacing is therefore **measured** and reported rather than
    assumed.

    Parameters
    ----------
    n_shells
        Number of concentric cages (>= 1).
    inner_freq
        Frequency of the innermost cage (>= 1). ``1`` with
        ``family="C60"`` starts the onion at C60 itself.
    freq_step
        Frequency increment between shells. ``1`` gives ~3.55 Å spacing
        for the C60 family, which is the physical one.
    family, bond, roughness, vacuum, seed
        As for :func:`build_fullerene`; applied to every shell.
    **shell_kwargs
        Further :func:`build_fullerene` arguments applied to every shell.

    Returns
    -------
    ase.Atoms
        The onion, with per-shell radii, the mean ``shell_spacing``, the
        summed ring counts, and combined ``bonds``/``rings`` with shell
        offsets applied.

    Raises
    ------
    ValueError
        For ``n_shells < 1`` or a non-positive ``freq_step``.
    """
    if n_shells < 1:
        raise ValueError("n_shells must be >= 1.")
    if freq_step < 1:
        raise ValueError("freq_step must be >= 1.")

    positions: list[np.ndarray] = []
    bonds: list[list[int]] = []
    rings: list[list[int]] = []
    ring_counts: dict[int, int] = {}
    radii: list[float] = []
    formulas: list[str] = []
    shell_ranges: list[tuple[int, int]] = []
    offset = 0

    for shell in range(n_shells):
        cage = build_fullerene(
            freq=inner_freq + shell * freq_step, family=family, bond=bond,
            roughness=roughness, seed=seed, **shell_kwargs,
        )
        shell_pos = cage.get_positions()
        shell_pos = shell_pos - shell_pos.mean(axis=0)
        positions.append(shell_pos)
        bonds += [[a + offset, b + offset] for a, b in cage.info["bonds"]]
        rings += [[a + offset for a in ring] for ring in cage.info["rings"]]
        for size, count in cage.info["ring_counts"].items():
            ring_counts[size] = ring_counts.get(size, 0) + count
        radii.append(float(cage.info["radius"]))
        formulas.append(cage.info["formula"])
        shell_ranges.append((offset, offset + len(cage)))
        offset += len(cage)

    merged = np.vstack(positions)
    atoms = Atoms(symbols=["C"] * len(merged), positions=merged, pbc=False)
    extents = merged.max(axis=0) - merged.min(axis=0)
    atoms.set_cell(np.diag(extents + vacuum))
    center_in_cell(atoms, axes=(0, 1, 2))

    spacings = [radii[i + 1] - radii[i] for i in range(len(radii) - 1)]
    atoms.info.update(
        {
            "structure_type": "nano_onion",
            "family": family,
            "n_shells": n_shells,
            "shell_radii": radii,
            "shell_formulas": formulas,
            "formula": "@".join(formulas),
            "shell_spacing": float(np.mean(spacings)) if spacings else 0.0,
            "bond": bond,
            "roughness": roughness,
            "ring_counts": {int(k): int(v) for k, v in ring_counts.items()},
            "rings": rings,
            "bonds": bonds,
            "geometry": _onion_geometry(merged, bonds, shell_ranges),
        }
    )
    return atoms


def _onion_geometry(
    positions: np.ndarray,
    bonds: list[list[int]],
    shell_ranges: list[tuple[int, int]],
) -> dict[str, float | int]:
    """Geometry report plus the closest approach *between* shells.

    Measured between known shell index ranges, never by excluding bonded
    neighbours: every cage is full of shorter non-bonded intra-shell
    distances (2.30 Å across a pentagon, 2.84 Å across a hexagon), so an
    exclusion-based metric would report those and mean nothing. A single
    cage has no second shell and honestly reports ``nan``.
    """
    from scipy.spatial import cKDTree

    report = dict(geometry_report(positions, [tuple(b) for b in bonds]))
    closest = float("inf")
    for i, (start_a, end_a) in enumerate(shell_ranges):
        others = [
            index
            for j, (start_b, end_b) in enumerate(shell_ranges)
            if j != i
            for index in range(start_b, end_b)
        ]
        if not others:
            continue
        tree = cKDTree(positions[others])
        distances, _ = tree.query(positions[start_a:end_a], k=1)
        closest = min(closest, float(distances.min()))
    report["min_wall_separation"] = closest if np.isfinite(closest) else float("nan")
    return report


__all__ = [
    "FAMILY_BASE_ATOMS",
    "GRAPHITIC_GAP",
    "build_fullerene",
    "build_nano_onion",
    "icosahedron_mesh",
    "pentakis_dodecahedron_mesh",
]
