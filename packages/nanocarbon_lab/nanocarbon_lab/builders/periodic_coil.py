"""A coiled nanotube as a genuinely periodic cell, for DFT.

A helix of pitch ``p`` maps onto itself under a translation of ``p``
along its axis: one turn, closed on the z-torus, is a real unit cell and
not a fragment with its ends waved at. That is what makes a plane-wave
calculation of a nanocoil possible at all — the alternative is a finite
segment whose two dangling ends dominate whatever you were trying to
measure.

Two details make the difference between a cell and a torn mesh, and both
were found by measurement rather than by reasoning:

**The implicit field has to be periodic to machine precision.**
:func:`~nanocarbon_lab.builders.implicit.tube_along_path` resamples the
centreline by arc length, so unless one turn is an exact whole number of
samples the field at ``z = 0`` and ``z = p`` differ — by 0.05 Å at a
loose spacing, which is enormous next to the interpolation marching cubes
does. The weld then fails and the mesh comes back with hundreds of
boundary edges. Choosing the sample spacing commensurate with one turn
drops the mismatch to 2e-13 and the boundary edges to zero.

**The wall is not a rolled lattice, and should not be.** Sweeping a
finished all-hexagon tube along a helix stretches its outer wall: at 6.5%
path strain that is 1.51 Å bonds, and no relaxation removes them, because
a pure-hexagon tube bent onto an arc *must* have a longer outer wall.
Meshing the surface instead lets the remesher put pentagons on the
compressed inner wall and heptagons on the stretched outer one, which is
how real coiled nanotubes relieve the strain. The result looks less tidy
and is the more physical structure.

One period of a helical tube is topologically a torus, so a correct mesh
comes back at genus 1. That is the check this module runs before it hands
anything back.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..analyse.curvature import disclination_check
from ..utils.constants import CC_BOND
from ..utils.rng import make_rng
from . import fullerene_mesh as fm
from . import implicit as im
from . import remesh as rm
from .capped_cnt import geometry_report

__all__ = ["LITERATURE_RATIO", "build_periodic_coil", "coil_centreline",
           "curvature_check", "helix_period_samples"]

#: Outer coil diameter over tubular diameter, as two independent studies
#: of single-wall coils report it. Popović et al., Contemp. Mater. III-1
#: (2012) 51, find D/d ~ 3.5 across both of their classes of helically
#: coiled tube; Liu et al., Nanoscale Res. Lett. 5 (2010) 478, Table 2,
#: give (5,5) 26.38/6.80 = 3.88, (6,6) 28.85/8.16 = 3.54, (7,7)
#: 33.58/9.52 = 3.53 and (8,8) 39.21/10.88 = 3.60. A coil built far from
#: this band is not wrong -- multi-wall coils reach 10 and more -- but it
#: is not the single-wall geometry those calculations relaxed to, and the
#: builder says so rather than letting the reader assume.
LITERATURE_RATIO = (3.4, 4.0)

#: Turns of helix sampled either side of the cell. The distance field near
#: the seam has to see the neighbouring turns, or the surface bulges where
#: the sampled centreline runs out.
_CONTEXT_TURNS = 2.0


def helix_period_samples(coil_radius: float, pitch: float,
                         target_spacing: float = 0.3,
                         sides: int | None = None) -> float:
    """Sample spacing commensurate with one turn, near ``target_spacing``.

    Returns the spacing that divides one turn's arc length a whole number
    of times. Without this the field is not periodic and the periodic
    weld has nothing to match.
    """
    if sides is None:
        arc = float(np.hypot(2.0 * np.pi * coil_radius, pitch))
    else:
        # A polygon's turn is shorter than the helix through its corners,
        # and the spacing has to divide the length the tube is actually
        # swept along or the seam stops matching to machine precision.
        chord = 2.0 * coil_radius * np.sin(np.pi / sides)
        arc = float(sides * np.hypot(chord, pitch / sides))
    divisions = max(1, round(arc / target_spacing))
    return arc / divisions


def coil_centreline(coil_radius: float, pitch: float, sides: int | None,
                    sign: float, first_turn: float, last_turn: float,
                    samples: int, corner_radius: float = 0.0) -> np.ndarray:
    """The curve the tube is swept along, smooth or polygonal.

    A real coil is not a smooth helix. Seen down the axis it is a
    **polygon**: Liu et al. show the (6,6) coil's top view as a hexagonal
    torus, and say it coincides with what is seen experimentally, because
    the wall relieves its strain at a few sharp knees rather than
    everywhere at once. ``sides`` builds that: the centreline runs
    straight between ``sides`` vertices per turn, and the curvature -- and
    with it the pentagons and heptagons -- concentrates at the corners
    instead of smearing around the whole turn.

    ``sides=None`` keeps the smooth helix, which is the right shape for a
    loosely coiled multi-wall tube and the wrong one for the single-wall
    structures the literature relaxes.
    """
    turn = np.linspace(first_turn, last_turn, samples)
    if sides is None:
        angle = 2.0 * np.pi * turn
        return np.column_stack([
            coil_radius * np.cos(angle),
            sign * coil_radius * np.sin(angle),
            pitch * turn,
        ])
    if sides < 3:
        raise ValueError(f"a polygonal coil needs at least 3 sides; got {sides}")
    # Corner index and the fraction of the way to the next corner. The
    # vertices sit on the helix; the path between them is a chord.
    scaled = turn * sides
    index = np.floor(scaled)
    frac = (scaled - index)[:, None]

    def vertex(k: np.ndarray) -> np.ndarray:
        angle = 2.0 * np.pi * k / sides
        return np.column_stack([
            coil_radius * np.cos(angle),
            sign * coil_radius * np.sin(angle),
            pitch * k / sides,
        ])

    path = vertex(index) * (1.0 - frac) + vertex(index + 1.0) * frac
    if corner_radius <= 0.0:
        return path
    # A mathematically sharp corner is not a knee and cannot be meshed:
    # marching cubes at any affordable resolution leaves the crease open
    # and the periodic seam then has nothing to weld to. A real knee bends
    # over a few bonds, so the corner is rounded over that length. The
    # filter is uniform in the turn parameter, which is what the screw
    # symmetry acts on, so the path stays exactly periodic.
    step = (last_turn - first_turn) / max(samples - 1, 1)
    length_per_turn = sides * np.hypot(2.0 * coil_radius * np.sin(np.pi / sides),
                                       pitch / sides)
    window = int(round(corner_radius / (length_per_turn * step)))
    window = max(1, window) * 2 + 1              # odd, so it is centred
    kernel = np.ones(window) / window
    smoothed = np.column_stack([
        np.convolve(path[:, axis], kernel, mode="same") for axis in range(3)
    ])
    # The convolution is wrong within half a window of either end, where
    # it has nothing to average against. Those samples are context the
    # caller discards, but they still feed the distance field, so they
    # keep their unsmoothed values rather than being pulled inwards.
    half = window // 2
    smoothed[:half] = path[:half]
    smoothed[-half:] = path[-half:]
    return smoothed


def curvature_check(positions: np.ndarray, rings, axis_xy: np.ndarray) -> dict:
    """Where the pentagons and heptagons ended up, against where they belong.

    On a coiled tube the surface is torus-like, so its Gaussian curvature
    is **positive on the outer equator and negative on the inner one**.
    A pentagon is a +60° disclination and a heptagon a -60° one, so the
    pentagons belong on the outside and the heptagons on the inside --
    both papers say it in those words, and it is the one statement about
    a coil's structure that can be checked on a finished model without
    running anything.

    This does not move anything. A ring on the wrong side is a defect of
    the model and gets reported as one; hiding it would make the census
    look like agreement with the literature when it is not.
    """
    if not rings:
        return {}
    radius = {}
    for ring in rings:
        centre = positions[list(ring)].mean(axis=0)[:2] - axis_xy
        radius.setdefault(len(ring), []).append(float(np.hypot(*centre)))
    if 6 not in radius:
        return {}
    middle = float(np.mean(radius[6]))
    five = np.array(radius.get(5, []))
    seven = np.array(radius.get(7, []))
    outside = int((five > middle).sum())
    inside = int((seven < middle).sum())
    total = len(five) + len(seven)
    return {
        "hexagon_radius": round(middle, 3),
        "pentagons": len(five),
        "pentagons_outside": outside,
        "heptagons": len(seven),
        "heptagons_inside": inside,
        "placed_correctly": (outside + inside) / total if total else 1.0,
        "pentagon_radius": round(float(five.mean()), 3) if five.size else None,
        "heptagon_radius": round(float(seven.mean()), 3) if seven.size else None,
    }


def path_period_samples(path: np.ndarray, context_turns: float,
                        total_turns: float,
                        target_spacing: float = 0.3) -> float:
    """Sample spacing that divides ONE turn of ``path`` a whole number of
    times, measured on the path itself.

    The analytic length of a helix or a polygon is the wrong number the
    moment the corners are rounded, and the field is then not periodic:
    :func:`helix_period_samples` is kept for the smooth case it is exact
    for, but what the sweep is given has to be measured.
    """
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    per_turn = len(steps) / total_turns
    start = int(round(context_turns * per_turn))
    stop = int(round((context_turns + 1.0) * per_turn))
    arc = float(steps[start:stop].sum())
    divisions = max(1, round(arc / target_spacing))
    return arc / divisions


def build_periodic_coil(
    coil_radius: float = 15.0,
    pitch: float = 8.0,
    tube_radius: float = 2.0,
    sides: int | None = None,
    corner_radius: float | None = None,
    bond: float = CC_BOND,
    handedness: int = 1,
    vacuum: float = 10.0,
    resolution: int = 64,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 80,
    place_curvature: bool = False,
    relax_iterations: int = 3000,
    seed: int | None = 0,
) -> Atoms:
    """Build one period of a coiled nanotube, periodic along z.

    Parameters
    ----------
    coil_radius
        Distance from the helix axis to the tube's centre (Å).
    pitch
        Rise of one full turn (Å), and therefore the cell length along z.
        Must clear two tube walls plus a graphitic gap, or successive
        turns merge into one solid.
    tube_radius
        Radius of the tube itself (Å).
    bond
        Target C-C length (Å); sets the remesh edge and the relaxation.
    handedness
        ``+1`` right-handed, ``-1`` left-handed.
    vacuum
        Padding (Å) between the coil and the cell walls in x and y. Those
        two directions are not periodic; the padding keeps the surface
        clear of them so their weld is a no-op.
    resolution
        Grid points across the longest cell axis. Too coarse and the seam
        does not weld, which is reported rather than returned.
    place_curvature
        Move the disclinations to where the surface's own Gaussian
        curvature asks for them, by Stone-Wales flips, after the remesh.
        It cannot change HOW MANY there are -- flips preserve
        ``sum(6 - deg)`` and any flip raising the defect count is
        refused -- only where they sit. Measured on the junctions:
        94.3 -> 98.4% of disclinations on the correct side of the
        curvature on a Y, 90.0 -> 93.8% on an L, 95.7 -> 97.9% on an X.
    anneal_sweeps
        Stays at 80, unlike the other meshed builders, because with the
        flip-length guard in :data:`~nanocarbon_lab.builders.remesh.FLIP_MAX_EDGE`
        annealing helps here rather than hurting: 88 % of the rings on
        the correct side of the curvature against 74 % without it, and
        an angle sum of 342.7 deg against 336.7.

        **Without that guard it was catastrophic and looked like
        something else entirely.** A flip cannot move a vertex, so on a
        mesh as coarse as a 3 Å tube carries -- 185 vertices, under
        eight rings around the tube -- it reached clear across the tube
        and the wall folded through itself: the tube ran 0.04-7.15 Å
        about a requested 3.0, with an angle sum of 320.8 deg, past
        tetrahedral and so not carbon at all. Four other diagnoses were
        measured and wrong first; CLAUDE.md keeps them.
    remesh_iterations, anneal_sweeps, relax_iterations, seed
        Passed through to the remesh and relaxation, as for the other
        implicit builders.

    Returns
    -------
    ase.Atoms
        ``pbc=(False, False, True)`` with the cell's z equal to ``pitch``.

    Raises
    ------
    ValueError
        If the pitch cannot keep successive turns apart, or if the mesh
        comes back open or at the wrong genus -- both mean the cell is
        not periodic, and a structure that is quietly not periodic is
        worse than none.
    """
    if coil_radius <= 0.0 or pitch <= 0.0 or tube_radius <= 0.0:
        raise ValueError("coil_radius, pitch and tube_radius must be positive.")
    notes: list[str] = []
    # Two different things were being refused as one. Turns that OVERLAP
    # are impossible -- the surface would self-intersect and the mesh is
    # meaningless. Turns merely closer than a graphitic gap are TIGHT, and
    # tight is what the published single-wall coils are: Liu et al.'s
    # relaxed (7,7) has a pitch of 12.11 Å around a 9.52 Å tube, a gap of
    # 2.59 Å, well inside 3.4. Refusing that refused a structure somebody
    # had already relaxed with DFT.
    if pitch <= 2.0 * tube_radius:
        raise ValueError(
            f"pitch={pitch:.1f} Å is not more than the {2.0 * tube_radius:.1f} Å "
            "the tube itself occupies, so successive turns would pass through "
            "each other. This is not a tight coil, it is an impossible one."
        )
    gap = pitch - 2.0 * tube_radius
    if gap < 3.4:
        notes.append(
            f"turns are {gap:.2f} Å apart wall to wall, less than the 3.4 Å "
            "graphitic gap: the walls are in van der Waals contact and any "
            "energy from this cell includes that interaction. The relaxed "
            "single-wall coils in the literature are like this too"
        )

    # The spacing is worked out AFTER the path exists, from the path. A
    # rounded polygon is shorter than the polygon through its corners,
    # and a spacing taken from the ideal shape no longer divides one turn
    # a whole number of times -- which is the one thing this whole module
    # depends on. Measured: it put 216 boundary edges into a seam that
    # had none.
    spacing = None
    turns = 1.0 + 2.0 * _CONTEXT_TURNS
    # A whole number of samples per turn only helps if the sampling starts
    # at a turn boundary, so the context is a whole number of turns too.
    sign = 1.0 if handedness >= 0 else -1.0
    # Samples per turn coprime with the side count, so no polygon corner
    # lands exactly on a sample. On a corner the tangent is two-valued;
    # the swept frame flips there and the surface tears. Measured, that
    # tore the seam for exactly the side counts that divide the grid --
    # 5 and 10 of 4000 per turn, while 6, 7, 8 and 12 welded cleanly --
    # and it got WORSE at higher resolution, which is how it was told
    # apart from a mesh that is merely too coarse. The grid still starts
    # on a turn boundary and still holds a whole number of samples per
    # turn, which is what the periodicity actually needs.
    per_turn = 4000
    while sides and per_turn % sides == 0:
        per_turn += 1
    path = coil_centreline(
        coil_radius, pitch, sides, sign,
        first_turn=-_CONTEXT_TURNS, last_turn=1.0 + _CONTEXT_TURNS,
        samples=int(round(turns * per_turn)) + 1,
        corner_radius=(0.0 if corner_radius is None else corner_radius),
    )
    # Analytic where the shape is exactly known, measured where it is not.
    # The analytic length is exact for a helix and for a sharp polygon, and
    # wrong the moment a corner is rounded.
    spacing = (helix_period_samples(coil_radius, pitch, sides=sides)
               if not corner_radius
               else path_period_samples(path, _CONTEXT_TURNS, turns))
    field, _lower, _upper = im.tube_along_path(
        path, radius=tube_radius, sample_spacing=spacing)

    half = coil_radius + tube_radius + vacuum
    cell = np.array([2.0 * half, 2.0 * half, pitch], dtype=float)

    def centred(points: np.ndarray) -> np.ndarray:
        """The mesher samples ``[0, L]``; the helix is centred on the axis."""
        shifted = np.array(points, dtype=float, copy=True)
        shifted[..., 0] -= half
        shifted[..., 1] -= half
        return field(shifted)

    mesh = rm.periodic_marching_cubes_mesh(centred, cell, resolution=resolution)
    stats = rm.mesh_statistics(mesh)
    if stats.get("boundary_edges", 0):
        raise ValueError(
            f"The z seam did not weld: {stats['boundary_edges']} boundary edges "
            f"at resolution {resolution}. The cell is not periodic. Raise the "
            "resolution, or widen the pitch so the wall is better resolved."
        )

    rng = make_rng(seed)
    mesh = rm.isotropic_remesh(
        mesh, centred, target_edge=float(np.sqrt(3.0) * bond),
        iterations=remesh_iterations, box=cell,
        anneal_sweeps=anneal_sweeps, rng=rng,
        place_curvature=place_curvature,
    )
    positions, bond_set, rings = fm.dual_honeycomb(mesh, box=cell)
    positions = fm.relax_shell(
        positions, bond_set, equilibrium=bond, box=cell,
        max_iterations=relax_iterations,
    )
    positions = np.mod(positions, cell)

    # Euler is necessary and not sufficient: a mesh can come back
    # topologically consistent and still be geometrically torn, which is
    # what a 2.2 Å "bond" means. Measured on a 25 Å coil that passed the
    # ring budget and would have been handed back as a DFT cell, so this
    # check is here for a case that actually happened rather than a
    # hypothetical one. Same thresholds as the schwarzites: far outside
    # anything strain can explain.
    quality = geometry_report(positions, sorted(bond_set), box=cell)
    if quality["bond_max"] > 1.80 or quality["n_close_contacts"] > 0:
        raise ValueError(
            "The relaxed network is torn, not merely strained: bonds span "
            f"{quality['bond_min']:.2f}-{quality['bond_max']:.2f} Å with "
            f"{quality['n_close_contacts']} non-bonded contacts under 2 Å. "
            "The surface has features finer than a carbon ring somewhere -- "
            "raise the resolution, or use a thicker tube or a tighter coil."
        )

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  pbc=(False, False, True))
    atoms.set_cell(np.diag(cell))

    bonds = sorted(bond_set)
    counts: dict[int, int] = {}
    for ring in rings:
        counts[len(ring)] = counts.get(len(ring), 0) + 1
    # Euler, as the last word on whether this is a cell or a tear. On a
    # closed trivalent net, sum(6 - n) over the rings is 12(1 - genus),
    # and one period of a helical tube is a torus, so it must come to
    # zero. A mesh that pinched through itself during remeshing passes
    # every local check and fails this one.
    deficit = sum((6 - size) * count for size, count in counts.items())
    expected = 12 * (1 - int(stats.get("genus", 1)))
    if deficit != expected:
        raise ValueError(
            f"Ring budget {deficit} does not match the {expected} a genus-"
            f"{stats.get('genus')} surface owes: the mesh is torn, not merely "
            f"strained. Census {dict(sorted(counts.items()))}."
        )

    axis_xy = atoms.get_positions()[:, :2].mean(axis=0)
    placement = curvature_check(atoms.get_positions(), rings, axis_xy)
    if placement and placement["placed_correctly"] < 1.0:
        wrong = (placement["pentagons"] - placement["pentagons_outside"]
                 + placement["heptagons"] - placement["heptagons_inside"])
        notes.append(
            f"{wrong} of the {placement['pentagons'] + placement['heptagons']} "
            "non-hexagons sit on the wrong side of the wall -- a pentagon "
            "towards the axis or a heptagon away from it. The curvature of a "
            "torus-like surface puts the positive disclinations outside and "
            "the negative ones inside, so these are defects of the model, not "
            "of the coil"
        )

    # The two published studies of single-wall coils agree on one number,
    # and it is the one an experimentalist measures, so the builder says
    # where this coil sits against it rather than leaving it to be assumed.
    outer_diameter = 2.0 * (coil_radius + tube_radius)
    tubular_diameter = 2.0 * tube_radius
    ratio = outer_diameter / tubular_diameter
    if not LITERATURE_RATIO[0] <= ratio <= LITERATURE_RATIO[1]:
        notes.append(
            f"D/d = {ratio:.2f}, outside the {LITERATURE_RATIO[0]}-"
            f"{LITERATURE_RATIO[1]} band that Popović et al. (2012) and Liu "
            "et al. (2010) both report for relaxed single-wall coils. Fine "
            "for a multi-wall coil; not the geometry those calculations "
            "settled at"
        )

    atoms.info.update({
        "structure_type": "periodic_coil",
        "coil_radius": coil_radius,
        "pitch": pitch,
        "tube_radius": tube_radius,
        "sides": sides,
        "outer_diameter": round(outer_diameter, 3),
        "tubular_diameter": round(tubular_diameter, 3),
        "diameter_ratio": round(ratio, 3),
        "wall_gap": round(pitch - 2.0 * tube_radius, 3),
        # The axis-based test above asks which side of the torus each
        # ring falls on, which needs an axis and so works here and
        # nowhere else. The intrinsic one fits the surface instead, so
        # it holds the coil to the SAME claim by a second and
        # independent route -- and it is the one every other curved
        # builder here is now held to as well.
        "curvature_check": placement,
        "notes": notes,
        "handedness": int(sign),
        "bond": bond,
        "period_axis": 2,
        "cell": [float(v) for v in cell],
        "mesh_genus": int(stats.get("genus", -1)),
        "genus": int(stats.get("genus", -1)),
        "ring_deficit": int(deficit),
        "euler": 1 - int(stats.get("genus", 1)),
        "disclination_check": disclination_check(
            positions, rings, sorted(bonds),
            box=np.array([0.0, 0.0, float(cell[2])])),
        "ring_counts": {int(k): int(v) for k, v in sorted(counts.items())},
        "rings": [[int(a) for a in r] for r in rings],
        "bonds": [[int(a), int(b)] for a, b in bonds],
        "geometry": quality,
        "seed": seed,
    })
    return atoms
