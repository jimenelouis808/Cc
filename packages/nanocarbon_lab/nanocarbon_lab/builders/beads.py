"""Graphitic spheres joined by necks: a chain, a ring or a net of them.

What this is, stated against the two neighbours it is easy to confuse
it with. A **nano-onion** (``fullerene.py``) is concentric shells of one
centre. A **supernetwork** hangs a *tube* on every edge of a graph, so
its nodes are where tubes meet and there is no sphere anywhere. This
hangs a **sphere on every vertex** and a short neck on every edge, which
is the "beads on a string" / peanut / interconnected-cage morphology --
one closed carbon surface throughout, not cages touching.

**It is one field, not two primitives unioned.** A zero-length capsule
*is* a sphere, so a bead and a neck are both segments handed to
:func:`implicit.graph_field` with different radii, and the smooth union
that already blends struts at a supernetwork node is what makes the
neck flare into the bead instead of meeting it at a crease. A crease is
not a thing a hexagonal net can tile, which is the reason that union is
smooth in the first place.

**The ring budget is fixed by the graph before anything is meshed**, by
the same identity :mod:`supernetwork` uses: the wall is the boundary of
a thickened graph, so ``chi = 2 * (V - E)`` and ``sum(6 - n) = 12 *
(V - E)``. A chain of ``n`` beads is a tree -- ``V - E = 1`` -- so its
budget is **+12**, the fullerene's twelve pentagons, however many beads
it has: a sausage is still sphere-like. A closed ring of ``n`` beads has
``V = E`` and a budget of **0**, which is a torus, and that is the same
statement :mod:`toroid` makes about a bent tube. Those two are checked
rather than asserted, and they are an *independent* check: ``_finish``
already tests the census against the **mesh's** own Euler
characteristic, which catches a torn mesh and cannot catch a sound mesh
that closed around the wrong graph.

The geometry has one rule and it is not a preference. The neck must be
**narrower than the bead** or there is no neck -- the union returns one
fat lozenge and the structure is a bumpy tube under a bead chain's
name -- and the beads must be **far enough apart** that the two spheres
do not simply merge. Both are refused rather than returned.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.rng import make_rng
from . import implicit as im
from . import remesh as rm
from .junction import MAX_GRID_RESOLUTION, _finish

__all__ = ["BEAD_LAYOUTS", "DEFAULT_SPACING_FRACTION",
           "MIN_SPACING_FRACTION", "bead_graph",
           "build_bead_chain"]

#: How the beads are arranged. ``chain`` is a tree (budget +12, sphere-
#: like); ``ring`` closes it (budget 0, a torus); ``cross`` is a central
#: bead with arms, still a tree.
BEAD_LAYOUTS = ("chain", "ring", "cross")

#: A neck wider than this fraction of the bead stops being a neck: the
#: smooth union returns one lozenge rather than two spheres joined.
#: Measured on a 7 Å bead -- at 0.85 the waist is 6.3 Å against the
#: bead's 7.0 and the shape reads as a corrugated tube.
MAX_NECK_FRACTION = 0.75

#: Two spheres closer than twice their radius INTERSECT, so there is no
#: waist at all and the neck has nothing to join: the surface comes out
#: a lumpy tube. The first version of this builder defaulted to 1.7 and
#: allowed 1.25, both under that floor, and the render showed a sausage
#: with dents -- sound carbon, correct census, not a bead chain.
MIN_SPACING_FRACTION = 2.0

#: Far enough that the waist is a real neck rather than a dimple.
DEFAULT_SPACING_FRACTION = 2.4


def bead_graph(layout: str, count: int, spacing: float,
               arms: int = 4) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Vertex positions and edges for a bead arrangement.

    Returns ``(vertices, edges)`` in Å, centred on the origin. The graph
    is the whole topological input: ``V - E`` fixes the ring budget
    before any geometry exists.

    Parameters
    ----------
    layout
        One of :data:`BEAD_LAYOUTS`.
    count
        Number of beads. For ``cross`` this is the arms' length in beads,
        so the structure holds ``arms * count + 1``.
    spacing
        Centre-to-centre distance between neighbouring beads, in Å.
    arms
        Arms for the ``cross`` layout, spread evenly in the xy plane.
    """
    if count < 2:
        raise ValueError("A bead structure needs at least 2 beads.")
    if spacing <= 0.0:
        raise ValueError("spacing must be positive.")

    if layout == "chain":
        vertices = np.zeros((count, 3))
        vertices[:, 2] = (np.arange(count) - 0.5 * (count - 1)) * spacing
        edges = [(i, i + 1) for i in range(count - 1)]
    elif layout == "ring":
        if count < 3:
            raise ValueError("A ring of beads needs at least 3.")
        # The radius follows from the spacing: n chords of length
        # `spacing` close a circle at R = spacing / (2 sin(pi/n)).
        radius = spacing / (2.0 * np.sin(np.pi / count))
        angle = np.arange(count) * 2.0 * np.pi / count
        vertices = np.stack(
            [radius * np.cos(angle), radius * np.sin(angle),
             np.zeros(count)], axis=1)
        edges = [(i, (i + 1) % count) for i in range(count)]
    elif layout == "cross":
        if arms < 2:
            raise ValueError("A cross needs at least 2 arms.")
        vertices = [np.zeros(3)]
        edges = []
        for arm in range(arms):
            theta = arm * 2.0 * np.pi / arms
            direction = np.array([np.cos(theta), np.sin(theta), 0.0])
            previous = 0
            for step in range(1, count + 1):
                vertices.append(direction * spacing * step)
                edges.append((previous, len(vertices) - 1))
                previous = len(vertices) - 1
        vertices = np.stack(vertices)
        vertices -= vertices.mean(axis=0)
    else:
        raise ValueError(
            f"Unknown layout {layout!r}. Available: "
            f"{', '.join(BEAD_LAYOUTS)}.")
    return np.asarray(vertices, dtype=float), edges


def build_bead_chain(
    layout: str = "chain",
    count: int = 3,
    bead_radius: float = 7.0,
    neck_radius: float = 3.0,
    spacing: float | None = None,
    arms: int = 4,
    blend: float = 3.0,
    bond: float = CC_BOND,
    grid_resolution: int = 72,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 0,
    place_curvature: bool = True,
    wall_anchor: float = 0.0,
    roughness: float = 0.0,
    relax_iterations: int = 3000,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
) -> Atoms:
    """Build graphitic spheres joined by necks into one closed surface.

    Parameters
    ----------
    layout
        One of :data:`BEAD_LAYOUTS`. ``chain`` and ``cross`` are trees and
        come out sphere-like (budget +12); ``ring`` closes and comes out
        toroidal (budget 0).
    count
        Beads, or arm length in beads for ``cross``.
    bead_radius
        Sphere radius in Å, before the wall relaxes.
    neck_radius
        Radius of the neck joining two beads. Must stay under
        :data:`MAX_NECK_FRACTION` of the bead, or the union returns one
        lozenge and there is no bead chain.
    spacing
        Centre-to-centre distance. Left ``None`` it is
        ``1.7 * bead_radius``, which leaves a visible waist at the
        default proportions. It is refused below ``1.25 * bead_radius``,
        where the two spheres merge before the neck does any work.
    blend
        Smooth-union radius (Å) where a neck meets a bead. This is what
        makes the join a flare rather than a crease, and a crease is not
        tileable by a hexagonal net.
    place_curvature
        **On here**, unlike the junction, and for the reason
        :mod:`nanoparticle.coat` gives: a bead is convex, so pentagons
        belong on it and the remesher's stray heptagons are pure
        defects. A junction saddles and wants the opposite.

    Returns
    -------
    ase.Atoms
        One closed carbon surface. ``atoms.info`` carries the usual
        census keys plus ``bead_count``, ``ring_budget`` and the
        parameters.

    Raises
    ------
    ValueError
        If the neck is not narrower than the bead, or the beads are
        close enough to merge.
    RuntimeError
        If the mesh does not close, or its census misses the budget the
        graph fixed.
    """
    if bead_radius <= 0.0 or neck_radius <= 0.0:
        raise ValueError("bead_radius and neck_radius must be positive.")
    if neck_radius > MAX_NECK_FRACTION * bead_radius:
        raise ValueError(
            f"neck_radius {neck_radius:.2f} A is more than "
            f"{MAX_NECK_FRACTION:.0%} of bead_radius {bead_radius:.2f} A, "
            "so the smooth union returns one lozenge rather than beads "
            "joined by a neck. Narrow the neck or widen the bead.")
    if spacing is None:
        spacing = DEFAULT_SPACING_FRACTION * bead_radius
    if spacing < MIN_SPACING_FRACTION * bead_radius:
        raise ValueError(
            f"spacing {spacing:.2f} A is under "
            f"{MIN_SPACING_FRACTION} * bead_radius "
            f"({MIN_SPACING_FRACTION * bead_radius:.2f} A): two spheres "
            f"of radius {bead_radius:.1f} A that close together INTERSECT "
            "each other, so there is no waist for a neck to be, and what "
            "comes out is a bumpy tube under a bead chain's name.")

    vertices, edges = bead_graph(layout, count, spacing, arms=arms)
    budget = 12 * (len(vertices) - len(edges))

    # A ring only has a hole if its beads leave one. Measured at bead 7,
    # neck 3, blend 3: a hole of +4.9 A closes as a torus (chi 0), +2.0 A
    # is refused by the quality gate for having features finer than a
    # carbon ring, and **+1.4 A silently closes as a SPHERE** -- a sound
    # structure, every bond in range, that is not the ring that was
    # asked for. That last one is why this is checked up front rather
    # than left to the census: by then ten seconds have been spent.
    if layout == "ring":
        ring_radius = spacing / (2.0 * np.sin(np.pi / count))
        hole = ring_radius - bead_radius
        if hole < blend:
            raise ValueError(
                f"{count} beads of radius {bead_radius:.1f} A at spacing "
                f"{spacing:.1f} A sit on a ring of radius "
                f"{ring_radius:.1f} A, leaving a hole of {hole:.1f} A -- "
                f"under the blend ({blend:.1f} A), so the beads merge "
                f"across the middle and the surface closes as a sphere "
                f"rather than a torus. Use more beads or a larger "
                f"spacing.")

    # One segment per bead (zero length -> a sphere) and one per neck.
    # Zero-length segments are what make this a single field instead of
    # a sphere field unioned with a tube field.
    segments = [np.stack([vertex, vertex]) for vertex in vertices]
    radii = [bead_radius] * len(vertices)
    for first, second in edges:
        segments.append(np.stack([vertices[first], vertices[second]]))
        radii.append(neck_radius)
    segments = np.stack(segments)

    extent = float(np.abs(vertices).max() + bead_radius + blend + 2.0)
    box = np.array([2.0 * extent] * 3)
    field = im.graph_field(segments, box, (False, False, False),
                           tube_radius=np.asarray(radii), blend=blend)

    # Same rule as the junction: track the voxel to the NECK, which is
    # the narrowest feature. Sized to the bead, a thin neck facets into
    # a prism and the census comes back wrong for a reason that looks
    # like physics.
    voxel_target = min(1.4, 0.25 * neck_radius)
    needed = int(np.ceil(2.0 * extent / voxel_target))
    resolution = int(np.clip(needed, grid_resolution, MAX_GRID_RESOLUTION))
    mesh = rm.marching_cubes_mesh(field, extent, resolution=resolution)
    rng = make_rng(seed)
    mesh = rm.isotropic_remesh(
        mesh, field, target_edge=np.sqrt(3.0) * bond,
        iterations=remesh_iterations, anneal_sweeps=anneal_sweeps,
        place_curvature=place_curvature, rng=rng,
    )
    atoms = _finish(
        mesh,
        field=field,
        wall_anchor=wall_anchor,
        bond=bond,
        relax_iterations=relax_iterations,
        vacuum=vacuum,
        roughness=roughness,
        rng=rng,
        info={
            "structure_type": "bead_chain",
            "bead_layout": layout,
            "bead_count": int(len(vertices)),
            "neck_count": int(len(edges)),
            "ring_budget": int(budget),
            "bead_radius": bead_radius,
            "neck_radius": neck_radius,
            "spacing": spacing,
            "blend": blend,
            "bond": bond,
            "anneal_sweeps": anneal_sweeps,
            "place_curvature": place_curvature,
            "roughness": roughness,
        },
    )

    # The independent check. `_finish` already tested the census against
    # the MESH's own Euler characteristic, which catches a torn mesh and
    # cannot catch a sound mesh that closed around the wrong graph -- a
    # ring whose hole filled in is exactly that, and it reads +12 against
    # a budget of 0 with every bond in range. Same reasoning as
    # SuperGraph.ring_budget, met from the other side.
    census = atoms.info.get("ring_counts", {})
    measured = sum((6 - size) * number for size, number in census.items())
    if measured != budget:
        raise RuntimeError(
            f"This {layout} of {len(vertices)} beads is a graph with "
            f"V - E = {len(vertices) - len(edges)}, so its ring budget is "
            f"{budget:+d}; the surface that came out carries "
            f"{measured:+d} ({dict(sorted(census.items()))}). The mesh "
            f"closed around a different shape than the graph describes -- "
            f"usually beads merging into each other.")
    return atoms
