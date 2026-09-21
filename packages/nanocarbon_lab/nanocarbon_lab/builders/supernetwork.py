"""Networks whose *edges* are nanotubes and whose *vertices* are junctions.

This is the hierarchy of Romo-Herrera, Terrones, Terrones, Dag and
Meunier, *Nano Lett.* **7** (2007) 570: pick a 1D block, use point-group
operations to make a multi-terminal node, then use the node as the new
building block and let translation operations generate the architecture.
Super-graphene, super-square, super-cubic and super-diamond are the four
they build; the same two steps generate a superfullerene or an
icosahedral cage, because nothing in them is specific to a crystal.

So the abstraction here is a **graph with an embedding**. A
:class:`SuperGraph` is a set of vertices, a set of edges between them,
and — when the thing is periodic — which images of the cell those edges
reach into. Everything else is one builder:

* every edge becomes a capsule of the chosen tube radius,
* the capsules are joined with a smooth union, so each vertex becomes a
  node with real curvature rather than a crease,
* the zero level set is meshed, remeshed to a graphitic edge length, and
  dualised into carbon.

What that route gives, and the reason for building these implicitly
rather than by gluing junctions together, is that **the ring statistics
are derived**. Nobody says a three-way node needs heptagons; a node is a
saddle, a saddle carries negative Gaussian curvature, and negative
curvature comes out of the remesher as degree-7 and degree-8 vertices,
which the dual renders as heptagons and octagons. Euler then has the
last word: ``sum(6 - n) = 6*chi``, with chi read from the mesh rather
than assumed, which is the same check the paper's Supporting Information
S2 describes as dictating the non-hexagonal rings.

**The edges are found from the geometry, not written down.** A net's
edge list with its periodic images is exactly the sort of table that is
easy to get subtly wrong and impossible to spot afterwards: a missing
image makes one node three-coordinate instead of four and the structure
still builds. Here the positions are declared and the bonds come from
the minimum image convention, which means the coordination number can be
CHECKED against what the net is supposed to have -- and it is, for every
entry in the catalogue, in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND
from ..utils.rng import make_rng
from . import implicit as im
from . import remesh as rm
from .junction import _finish

__all__ = [
    "ICOSAHEDRON",
    "SUPERLATTICES",
    "build_supernetwork",
    "icosahedral_cage",
    "SuperGraph",
    "edges_from_positions",
    "supergraph_from_atoms",
]

#: How far apart two vertices may be, as a fraction of the shortest
#: vertex-vertex distance, and still count as joined. Every net here has
#: a clean gap between its first and second neighbour shells -- the
#: tightest is fcc at 1.41 -- so a tenth is generous and unambiguous.
EDGE_TOLERANCE = 0.10


@dataclass(frozen=True)
class SuperGraph:
    """Vertices, edges and a cell: the skeleton a supernetwork hangs on.

    ``nodes`` are fractional coordinates of the cell for a periodic net
    and ångström for a finite cage. ``shape`` is the cell's proportions,
    multiplied by the ``scale`` given at build time; ``pbc`` says which
    of the three directions actually repeat, so a 2D net gets vacuum in
    the third rather than a spurious period.
    """

    name: str
    nodes: np.ndarray
    edges: tuple[tuple[int, int, tuple[int, int, int]], ...]
    pbc: tuple[bool, bool, bool]
    shape: tuple[float, float, float] = (1.0, 1.0, 1.0)
    note: str = ""
    reference: str = ""

    @property
    def periodic(self) -> bool:
        return any(self.pbc)

    @property
    def coordination(self) -> int:
        """Edges per vertex. Raises if the net is not uniform, because a
        net that is meant to be uniform and is not has a missing edge."""
        counts = np.zeros(len(self.nodes), dtype=int)
        for start, end, _ in self.edges:
            counts[start] += 1
            counts[end] += 1
        unique = set(counts.tolist())
        if len(unique) != 1:
            raise ValueError(
                f"{self.name}: vertices have {sorted(unique)} edges each, not "
                "all the same. For a uniform net that means an edge is "
                "missing -- most likely a periodic image."
            )
        return int(counts[0])

    @property
    def ring_budget(self) -> int:
        """``sum(6 - ring_size)`` the finished network must come out with.

        The wall of a network of tubes is the boundary of a thickened
        graph, and for that surface ``chi = 2 * (V - E)`` -- one handle
        per independent cycle of the graph. With ``sum(6 - n) = 6 * chi``
        the budget is ``12 * (V - E)``, fixed by the skeleton alone and
        known before anything is meshed.

        That makes it an **independent** check rather than a restatement.
        :func:`~nanocarbon_lab.builders.junction._finish` already tests
        the census against the mesh's own Euler characteristic, which
        catches a torn mesh; it cannot catch a mesh that closed perfectly
        around the wrong graph. A blend wide enough to merge two struts
        into one, or a neck the grid pinched shut, gives a flawless
        surface of the wrong genus -- and this is what notices.

        It reproduces the two constants
        :mod:`~nanocarbon_lab.builders.network` hardcodes: cubic
        ``12 * (1 - 3) = -24`` and diamond ``12 * (8 - 16) = -96``.
        """
        return 12 * (len(self.nodes) - len(self.edges))

    def cell(self, scale: float) -> np.ndarray:
        return np.array(self.shape, dtype=float) * float(scale)

    def segments(self, scale: float) -> np.ndarray:
        """``(n_edges, 2, 3)`` endpoints in Å, images included."""
        box = self.cell(scale)
        positions = (np.asarray(self.nodes, dtype=float) * box
                     if self.periodic else np.asarray(self.nodes, dtype=float))
        out = np.empty((len(self.edges), 2, 3), dtype=float)
        for index, (start, end, offset) in enumerate(self.edges):
            out[index, 0] = positions[start]
            out[index, 1] = positions[end] + np.asarray(offset, float) * box
        if not self.periodic:
            return out
        # Each strut is stored with whichever image the search happened to
        # find first, and for a one-node net that can be the -x neighbour
        # as easily as the +x one. Geometrically the same infinite net --
        # and NOT the same field, because the 27-image replication that
        # makes the field periodic is then centred one cell off: measured
        # against the proven cubic field, struts written backwards moved
        # the surface by 0.55 Å near the faces. Sliding each strut so its
        # MIDPOINT sits in the cell puts the replication back on centre.
        middle = out.mean(axis=1)
        for axis, repeats in enumerate(self.pbc):
            if not repeats or box[axis] <= 0:
                continue
            shift = np.floor(middle[:, axis] / box[axis]) * box[axis]
            out[:, :, axis] -= shift[:, None]
        return out

    def strut_lengths(self, scale: float) -> np.ndarray:
        segments = self.segments(scale)
        return np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)


def edges_from_positions(
    nodes: np.ndarray,
    shape: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
    tolerance: float = EDGE_TOLERANCE,
) -> tuple[tuple[int, int, tuple[int, int, int]], ...]:
    """Join every pair at the shortest vertex-vertex distance.

    The distance is the minimum image one in the periodic directions, so
    an edge that leaves through a face comes back with the offset that
    says which image it reached -- which is what the field needs to be
    continuous there.

    Working this out rather than tabulating it is the difference between
    a coordination number that can be checked and one that has to be
    believed.
    """
    nodes = np.asarray(nodes, dtype=float)
    box = np.array(shape, dtype=float)
    images = [range(-1, 2) if repeat else (0,) for repeat in pbc]
    shifts = np.array([(i, j, k)
                       for i in images[0] for j in images[1] for k in images[2]],
                      dtype=float)

    candidates: list[tuple[float, int, int, tuple[int, int, int]]] = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            for shift in shifts:
                if i == j and not shift.any():
                    continue
                delta = nodes[j] + shift * box - nodes[i]
                distance = float(np.linalg.norm(delta))
                if distance > 1e-9:
                    candidates.append(
                        (distance, i, j, tuple(int(s) for s in shift)))
    if not candidates:
        raise ValueError("a graph needs at least two vertices, or one image")

    shortest = min(item[0] for item in candidates)
    limit = shortest * (1.0 + tolerance)
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    edges: list[tuple[int, int, tuple[int, int, int]]] = []
    for distance, i, j, shift in candidates:
        if distance > limit:
            continue
        # i->j with shift and j->i with -shift are the same edge. Keep one.
        mirror = (j, i, tuple(-s for s in shift))
        if mirror in seen:
            continue
        seen.add((i, j, shift))
        edges.append((i, j, shift))
    return tuple(edges)


def _periodic_graph(name: str, nodes, shape, pbc, note, reference="") -> SuperGraph:
    nodes = np.asarray(nodes, dtype=float)
    return SuperGraph(
        name=name, nodes=nodes,
        edges=edges_from_positions(nodes * np.array(shape, float), shape, pbc),
        pbc=pbc, shape=shape, note=note, reference=reference,
    )


_PAPER = ("Romo-Herrera, Terrones, Terrones, Dag & Meunier, "
          "Nano Lett. 7 (2007) 570")

#: The nets the hierarchy generates, keyed by the name the paper uses.
SUPERLATTICES: dict[str, SuperGraph] = {
    "super-square": _periodic_graph(
        "super-square", [(0.0, 0.0, 0.5)], (1.0, 1.0, 1.0), (True, True, False),
        "2D, four tubes per node at 90°. The sharpest node in the family: "
        "a right angle is further from what sp2 carbon wants at a branch "
        "than anything else here.", _PAPER),
    "super-graphene": _periodic_graph(
        "super-graphene",
        # The four-vertex rectangular cell of a honeycomb, in fractions of
        # a cell that is a wide and a*sqrt(3) deep.
        [(0.0, 0.0, 0.5), (0.5, 1.0 / 6.0, 0.5),
         (0.5, 0.5, 0.5), (0.0, 2.0 / 3.0, 0.5)],
        (1.0, float(np.sqrt(3.0)), 1.0), (True, True, False),
        "2D, three tubes per node at 120°. The node the paper finds most "
        "stable, because 120° is the angle a graphitic branch adopts by "
        "itself.", _PAPER),
    "super-cubic": _periodic_graph(
        "super-cubic", [(0.0, 0.0, 0.0)], (1.0, 1.0, 1.0), (True, True, True),
        "3D, six tubes per node along the axes.", _PAPER),
    "super-diamond": _periodic_graph(
        "super-diamond",
        [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0),
         (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
         (0.75, 0.25, 0.75), (0.75, 0.75, 0.25)],
        (1.0, 1.0, 1.0), (True, True, True),
        "3D, four tubes per node at the tetrahedral 109.47°, which is what "
        "an sp2 branch wants. Gentler nodes than the cubic net at the cost "
        "of eight of them per cell.", _PAPER),
    "super-fcc": _periodic_graph(
        "super-fcc",
        [(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
        (1.0, 1.0, 1.0), (True, True, True),
        "3D, TWELVE tubes per node. Included because it is the densest "
        "sphere packing and the obvious thing to ask for, and it is "
        "reported rather than recommended: twelve tubes meeting at one "
        "point leaves no room for a wall between them at any radius that "
        "is still a tube.", ""),
}


def supergraph_from_atoms(atoms: Atoms, scale: float = 1.0,
                          name: str = "") -> SuperGraph:
    """Turn a finished carbon structure into the skeleton of a bigger one.

    This IS the hierarchy: a C60 cage is sixty three-coordinate vertices
    and ninety edges, so scaling it up and hanging a nanotube on every
    edge gives a superfullerene; an icosahedron gives a twelve-vertex,
    five-coordinate cage. Nothing about the step knows which structure it
    was handed, which is the point the paper makes when it calls the node
    "the new building block".

    ``scale`` multiplies the vertex positions, and it is the only knob
    that matters: at scale 1 the vertices are a bond apart and there is
    no room for a tube between them.
    """
    positions = np.asarray(atoms.get_positions(), dtype=float) * float(scale)
    positions = positions - positions.mean(axis=0)
    shape = (1.0, 1.0, 1.0)
    return SuperGraph(
        name=name or f"super-{atoms.info.get('structure_type', 'cage')}",
        nodes=positions,
        edges=edges_from_positions(positions, shape, (False, False, False)),
        pbc=(False, False, False), shape=shape,
        note=f"built from a {len(atoms)}-atom "
             f"{atoms.info.get('structure_type', 'structure')} scaled "
             f"{scale:g} times",
    )


#: A regular icosahedron's twelve vertices, from the golden ratio. Five
#: edges meet at each, which is the most a cage of tubes can ask for and
#: still leave wall between them.
_PHI = (1.0 + 5.0 ** 0.5) / 2.0
ICOSAHEDRON = np.unique(np.round(np.array(
    [p for b in ((0, 1, _PHI), (0, -1, _PHI), (0, 1, -_PHI), (0, -1, -_PHI))
     for p in ((b[0], b[1], b[2]), (b[1], b[2], b[0]), (b[2], b[0], b[1]))],
    dtype=float), 9), axis=0)


def icosahedral_cage(scale: float = 12.0) -> SuperGraph:
    """Twelve vertices, thirty edges, five tubes meeting at each.

    ``scale`` is the circumradius in ångström over the unit icosahedron,
    so the edge comes out at ``2 * scale`` (the golden-ratio vertices
    here have edge 2 and circumradius ``sqrt(1 + phi^2)``, and are
    normalised on the latter).

    The default is 12, not the 6 that looks natural, and the reason is
    worth stating because the failure is quiet in one direction and loud
    in the other: **each vertex eats about ``tube_radius + blend`` of
    either end of every edge it touches**. At scale 6 the edges are 12 Å
    and the builder's own default tube wants 18, so the cage is refused
    outright -- a default that cannot be built with the neighbouring
    defaults. At scale 9 it passes the length test and then relaxes into
    something the quality gate throws out, which costs a minute to find
    out. Measured clean: scale 12 with ``tube_radius=4, blend=3`` gives
    4318 atoms at 1.420 +- 0.023 Å with no close contacts, and scale 15
    with ``tube_radius=5, blend=3.5`` gives 6670; both meet the -216 ring
    budget exactly.
    """
    nodes = ICOSAHEDRON * float(scale)
    return SuperGraph(
        name="super-icosahedron", nodes=nodes,
        edges=edges_from_positions(nodes, (1.0, 1.0, 1.0),
                                   (False, False, False)),
        pbc=(False, False, False),
        note="a finite cage: five tubes per vertex, the densest vertex a "
             "closed cage of tubes can have.",
    )


def build_supernetwork(
    graph: SuperGraph | str = "super-graphene",
    scale: float = 40.0,
    tube_radius: float = 5.0,
    blend: float = 4.0,
    bond: float = CC_BOND,
    vacuum: float = 12.0,
    grid_resolution: int = 72,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 0,
    relax_iterations: int = 3000,
    roughness: float = 0.0,
    seed: int | None = 0,
) -> Atoms:
    """Hang a nanotube on every edge of ``graph`` and weld the vertices.

    Parameters
    ----------
    graph
        A :class:`SuperGraph`, or the name of one in
        :data:`SUPERLATTICES`. Finite cages come from
        :func:`icosahedral_cage` or :func:`supergraph_from_atoms`.
    scale
        Cell edge in Å for a periodic net. A finite cage carries its own
        scale in its vertex positions and ignores this.
    tube_radius
        Radius of every tube. Free rather than quantised, because the
        wall is meshed rather than rolled: the lattice adapts to the
        radius instead of the radius to the lattice.
    blend
        Smooth-union radius at the vertices. Too small leaves a crease
        with curvature no hexagonal net can tile; too large rounds the
        vertex into a sphere and the tubes stop being tubes.
    vacuum
        Padding in Å on directions that do not repeat.

    Returns
    -------
    ase.Atoms
        ``pbc`` matching the graph, the ring census and Euler budget in
        ``atoms.info``, and the graph's own description alongside.

    Raises
    ------
    ValueError
        If the edges are too short to leave a tube between the vertices.
    RuntimeError
        If no grid resolution gives a closed surface.
    """
    if isinstance(graph, str):
        try:
            graph = SUPERLATTICES[graph]
        except KeyError:
            raise ValueError(
                f"unknown net {graph!r}; the catalogue has "
                f"{list(SUPERLATTICES)}, and a cage comes from "
                "icosahedral_cage() or supergraph_from_atoms()") from None
    if tube_radius <= 0 or blend <= 0:
        raise ValueError("tube_radius and blend must be positive.")

    segments = graph.segments(scale)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    eaten = 2.0 * (tube_radius + blend)
    if lengths.min() <= eaten:
        raise ValueError(
            f"{graph.name}: the shortest edge is {lengths.min():.1f} Å and "
            f"each vertex eats about {tube_radius + blend:.1f} Å of either "
            "end, so nothing recognisable as a tube is left between them. "
            "Use a larger scale, a narrower tube, or a smaller blend."
        )

    if graph.periodic:
        box = graph.cell(scale)
        # A direction that does not repeat gets vacuum instead of a
        # period, and the structure is centred in it so the periodic
        # mesher's weld of that face finds nothing to join and is a
        # no-op -- which is how a 2D net stays two-dimensional.
        span = segments.reshape(-1, 3)
        for axis, repeats in enumerate(graph.pbc):
            if repeats:
                continue
            width = float(np.ptp(span[:, axis])) + 2.0 * (tube_radius + vacuum)
            box[axis] = width
        shifted = segments.copy()
        for axis, repeats in enumerate(graph.pbc):
            if not repeats:
                middle = 0.5 * (span[:, axis].min() + span[:, axis].max())
                shifted[:, :, axis] += 0.5 * box[axis] - middle
        field = im.graph_field(shifted, box, graph.pbc, tube_radius, blend)
        def mesher(resolution):
            """Weld the cell's faces to their opposite numbers."""
            return rm.periodic_marching_cubes_mesh(field, box,
                                                   resolution=resolution)

        finish_box = box
    else:
        centre = segments.reshape(-1, 3).mean(axis=0)
        shifted = segments - centre
        extent = float(np.abs(shifted).max()) + tube_radius + vacuum
        box = np.array([2.0 * extent] * 3)
        field = im.graph_field(shifted, box, (False, False, False),
                              tube_radius, blend)
        def mesher(resolution):
            """A finite cage closes on itself; there is nothing to weld."""
            return rm.marching_cubes_mesh(field, extent,
                                          resolution=resolution)

        finish_box = None

    # The same grid retry as the schwarzite and the network: whether a
    # neck is resolved depends on how the surface falls between sample
    # points, so one (scale, resolution) pair can tear where both its
    # neighbours are fine.
    failures: list[str] = []
    for attempt, resolution in enumerate(
        (grid_resolution, grid_resolution + 8, grid_resolution + 16)
    ):
        mesh = mesher(resolution)
        stats = rm.mesh_statistics(mesh)
        if graph.periodic and stats["boundary_edges"]:
            failures.append(
                f"resolution {resolution}: periodic weld left "
                f"{stats['boundary_edges']} boundary edges")
            continue
        try:
            atoms = _finish(
                rm.isotropic_remesh(
                    mesh, field, target_edge=float(np.sqrt(3.0) * bond),
                    iterations=remesh_iterations,
                    box=finish_box, anneal_sweeps=anneal_sweeps,
                    rng=make_rng(seed),
                ),
                bond=bond,
                relax_iterations=relax_iterations,
                vacuum=0.0 if graph.periodic else vacuum,
                box=finish_box,
                roughness=roughness,
                rng=make_rng(seed),
                info={
                    "structure_type": "supernetwork",
                    "network_kind": graph.name,
                    "scale": float(scale),
                    "tube_radius": float(tube_radius),
                    "blend": float(blend),
                    "bond": float(bond),
                    "n_nodes": len(graph.nodes),
                    "n_struts": len(graph.edges),
                    "node_coordination": graph.coordination,
                    "strut_length": round(float(lengths.mean()), 3),
                    "pbc": [bool(v) for v in graph.pbc],
                    "note": graph.note,
                    "reference": graph.reference,
                    "grid_resolution": resolution,
                    "grid_retries": attempt,
                    "seed": seed,
                },
            )
            if graph.periodic:
                atoms.set_pbc(graph.pbc)
            budget = graph.ring_budget
            census = atoms.info["ring_counts"]
            deficit = sum((6 - size) * count
                          for size, count in census.items())
            if deficit != budget:
                raise RuntimeError(
                    f"{graph.name}: the finished wall has sum(6-n) = "
                    f"{deficit}, but the skeleton's own topology fixes it "
                    f"at {budget} ({len(graph.nodes)} vertices, "
                    f"{len(graph.edges)} edges). The mesh closed cleanly, "
                    "so it is not torn -- it is a surface around a "
                    "different graph than the one asked for. A blend wide "
                    "enough to merge two struts, or a grid coarse enough "
                    "to pinch a neck shut, does exactly this. Try a "
                    "smaller blend, a narrower tube or a finer grid."
                )
            atoms.info["ring_budget"] = budget
            return atoms
        except RuntimeError as exc:
            failures.append(f"resolution {resolution}: {exc}")

    raise RuntimeError(
        f"Could not build {graph.name!r} at scale={scale:.1f} Å, "
        f"tube_radius={tube_radius:.1f} Å after {len(failures)} grid "
        "resolutions:\n  " + "\n  ".join(failures)
        + "\nA larger scale or a narrower tube gives the necks more room."
    )
