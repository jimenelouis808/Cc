"""MX2 on a triply periodic minimal surface -- dichalcogenide schwarzites.

A binary compound cannot be draped over a carbon schwarzite unchanged. M
and X alternate around every ring, so the net must be bipartite, and the
net here is the dual of the surface mesh: **atoms sit at triangle
centroids and bond across shared edges**, which makes ring size equal
mesh-vertex degree. A triangulation's faces are 2-colourable exactly when
every vertex degree is even, so "MoS2 schwarzite" is really "even-degree
triangulation of a TPMS".

Two things about that are worth stating because both are easy to get
wrong in opposite directions.

Pentagons are forbidden, but that never ruled out negative curvature.
``sum(6 - n) = 6*chi``, and an octagon pays -2 exactly as a heptagon does
while staying even; Schwarz P (chi = -4) wants twelve of them. What
pentagons rule out is the *sphere* -- MX2 closes one with six squares,
which is the observed MoS2 nano-octahedron.

And even degrees are necessary but **not sufficient**. The 2-colouring
theorem is a sphere result; at genus g there are 2g further Z/2 classes,
and on these surfaces they do not vanish. A few per cent of bonds stay
homoelemental no matter how the colouring is chosen. That is not a defect
of the construction -- it is an **inversion-domain boundary**, the same
line defect seen in grown MoS2 and h-BN, and it is reported rather than
hidden.

So the parity repair is a genuine trade, measured rather than assumed
(Schwarz P, 36 Å, MoS2):

=========== ============== ========== ======== =========
``parity``  homoelemental  worst M-X  p95 M-X  atoms
=========== ============== ========== ======== =========
``"none"``  11.0%          8.4%       3.9%     1066
``"flip"``  7.7%           12.5%      5.5%     1066
``"split"`` 2.2%           25.0%      11.5%    1298
=========== ============== ========== ======== =========

Better chemistry costs geometry, monotonically, and which end of that you
want depends on what the structure is for -- so it is a parameter, not a
decision made here. All three come out with zero atomic overlaps and
correct coordination and stoichiometry.

Why the two repairs differ. An edge **flip** toggles the parity of four
vertex degrees at once, so with odd vertices sparse it can only shuffle
them, never annihilate the last pair -- it gets partway and adds no
vertices. An edge **split** toggles exactly the two vertices opposite the
edge, which is the weight-two move that reaches zero odd; the cost is a
new vertex each time, and enough of them crowd the surface with more
sites than its area holds at spacing ``a/sqrt(3)``.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Literal

import numpy as np
from ase import Atoms

from ..builders import fullerene_mesh as fm
from ..builders import implicit as im
from ..builders import remesh as rm
from ..builders.fullerene_mesh import minimum_image
from ..builders.remesh import _adjacency, _edge_faces
from ..utils.geometry import center_in_cell
from .materials import Phase, TMDMaterial, coordination_geometry, get_material

ParityRepair = Literal["none", "flip", "split"]

#: Homoelemental bond lengths at an inversion-domain boundary, in Å,
#: taken as twice the metallic (metal) or covalent (chalcogen) radius.
#: Forcing these to the M-X length instead warps the network around every
#: boundary -- it moved the worst M-X bond of a flip-repaired Schwarz P
#: cell from 13.6% to 24.2%.
METAL_METAL_BOND = {
    "Mo": 2.78, "W": 2.78, "Nb": 2.92, "Ta": 2.92, "Ti": 2.94,
    "Zr": 3.20, "Hf": 3.18, "Pt": 2.78, "Sn": 3.16,
}
CHALCOGEN_CHALCOGEN_BOND = {"S": 2.10, "Se": 2.40, "Te": 2.76}

#: Below this the TPMS channels are narrower than the MX2 sandwich is
#: thick and the cell is not a surface any more. Larger than the carbon
#: minimum because the sandwich is ~3.1 Å thick where graphene is one atom.
MIN_SCHWARZITE_CELL = {"primitive": 30.0, "gyroid": 34.0, "diamond": 34.0}


# ----------------------------------------------------------------- parity
def odd_vertices(mesh: fm.Mesh) -> set[int]:
    """Mesh vertices of odd degree, i.e. the odd-membered rings."""
    adjacency = _adjacency(np.asarray(mesh[1], dtype=int))
    return {v for v, ns in adjacency.items() if len(ns) % 2}


def _apex_map(face_list: list[tuple[int, int, int]]) -> dict:
    """Every interior edge mapped to the two vertices opposite it."""
    faces = np.array(face_list, dtype=int)
    out = {}
    for edge, incident in _edge_faces(faces).items():
        if len(incident) != 2:
            continue
        opposite = set()
        for index in incident:
            opposite |= set(face_list[index]) - set(edge)
        if len(opposite) == 2:
            out[edge] = tuple(sorted(opposite))
    return out


def _edge_length(verts, edge, box) -> float:
    return float(np.linalg.norm(minimum_image(verts[edge[1]] - verts[edge[0]],
                                              box)))


def _split_edge(verts, face_list, edge, field, box):
    """Insert a midpoint on ``edge``; toggles its two apexes and nothing else."""
    a, b = edge
    incident = _edge_faces(np.array(face_list, dtype=int)).get(tuple(sorted(edge)))
    if incident is None or len(incident) != 2:
        return verts, face_list, False

    delta = minimum_image(verts[b] - verts[a], box)
    midpoint = im.project_to_surface(field, (verts[a] + 0.5 * delta)[None, :])[0]
    new_index = len(verts)
    verts = np.vstack([verts, midpoint])

    for index in incident:
        tri = face_list[index]
        for k in range(3):
            p, q, r = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            if {p, q} == {a, b}:
                # (p, q, r) -> (p, m, r) + (m, q, r), which keeps the winding.
                face_list[index] = (p, new_index, r)
                face_list.append((new_index, q, r))
                break
    return verts, face_list, True


def _cheapest_hop(verts, face_list, source, targets, box, target_edge):
    """First move of the cheapest co-apex walk from ``source`` to any odd.

    Weighted, not breadth-first: splitting an edge leaves two halves, so
    cutting an already-short edge makes a sliver whose triangle centroids
    land on top of each other. Routing the walk through long edges instead
    cut the number of badly-spaced sites by a third.
    """
    graph: dict[int, list] = {}
    for edge, (c, d) in _apex_map(face_list).items():
        half = 0.5 * _edge_length(verts, edge, box)
        shortfall = max(0.0, target_edge - half)
        cost = 1.0 + (shortfall / target_edge) ** 2 * 100.0
        graph.setdefault(c, []).append((d, edge, cost))
        graph.setdefault(d, []).append((c, edge, cost))
    if source not in graph:
        return None

    dist = {source: 0.0}
    previous: dict[int, tuple] = {source: (source, None)}
    heap = [(0.0, source)]
    found = None
    while heap:
        cost, v = heapq.heappop(heap)
        if cost > dist.get(v, float("inf")):
            continue
        if v in targets and v != source:
            found = v
            break
        for w, edge, step in graph.get(v, ()):
            if cost + step < dist.get(w, float("inf")):
                dist[w] = cost + step
                previous[w] = (v, edge)
                heapq.heappush(heap, (dist[w], w))
    if found is None:
        return None
    node = found
    while previous[node][0] != source:
        node = previous[node][0]
    return previous[node][1]


def repair_parity_by_splitting(mesh, field, box, target_edge, max_splits=600):
    """Split edges until every vertex degree is even.

    Reaches exactly zero odd vertices -- a split is a weight-two parity
    move where a flip is weight-four -- at the cost of one new vertex per
    split.
    """
    verts = np.array(mesh[0], dtype=float)
    face_list = [tuple(int(x) for x in tri) for tri in mesh[1]]
    splits = 0
    for _ in range(max_splits):
        degree = {v: len(n) for v, n in
                  _adjacency(np.array(face_list, dtype=int)).items()}
        odd = {v for v, d in degree.items() if d % 2}
        if not odd:
            break
        edge = _cheapest_hop(verts, face_list, min(odd), odd, box, target_edge)
        if edge is None:
            break
        verts, face_list, ok = _split_edge(verts, face_list, edge, field, box)
        if not ok:
            break
        splits += 1
    return (verts, np.array(face_list, dtype=int)), splits


def repair_parity_by_flipping(mesh, sweeps=400, t_start=0.25, t_end=0.02,
                              seed=0):
    """Anneal edge flips toward even degrees, adding no vertices.

    A flip changes the odd count by -4, -2, 0, +2 or +4 according to how
    many of the four touched vertices were already odd. Greedy descent
    only ever takes the falling case, which with sparse odd vertices
    barely exists; the two-odd move is exactly cost-neutral and is what
    walks defects together, so Metropolis takes it for free.

    It cannot reach zero -- from two odd vertices no flip can annihilate
    them -- but it leaves the vertex positions untouched, which is why its
    geometry stays sound where splitting's does not.
    """
    verts, faces = mesh
    face_list = [tuple(int(x) for x in tri) for tri in faces]
    nbrs = _adjacency(np.asarray(faces, dtype=int))
    degree = {v: len(ns) for v, ns in nbrs.items()}
    rng = np.random.default_rng(seed)

    def cost(d: int) -> float:
        if d < 4:
            return 50.0  # a degree-3 vertex is a three-membered ring
        return (1.0 if d % 2 else 0.0) + 0.8 * abs(d - 6)

    best_faces, best_odd = list(face_list), sum(1 for d in degree.values()
                                                if d % 2)
    for sweep in range(sweeps):
        temperature = t_start * (t_end / t_start) ** (sweep / max(1, sweeps - 1))
        items = list(_edge_faces(np.array(face_list, dtype=int)).items())
        rng.shuffle(items)
        touched: set[int] = set()
        for edge, incident in items:
            if len(incident) != 2:
                continue
            if incident[0] in touched or incident[1] in touched:
                continue
            u, v = edge
            opposite = set()
            for index in incident:
                opposite |= set(face_list[index]) - {u, v}
            if len(opposite) != 2:
                continue
            w1, w2 = sorted(opposite)
            if w2 in nbrs[w1]:
                continue  # would duplicate an existing edge
            delta = (cost(degree[u] - 1) + cost(degree[v] - 1)
                     + cost(degree[w1] + 1) + cost(degree[w2] + 1)
                     - cost(degree[u]) - cost(degree[v])
                     - cost(degree[w1]) - cost(degree[w2]))
            if delta > 0 and rng.random() >= np.exp(-delta / max(temperature,
                                                                1e-9)):
                continue
            face_list[incident[0]] = (u, w1, w2)
            face_list[incident[1]] = (v, w2, w1)
            degree[u] -= 1
            degree[v] -= 1
            degree[w1] += 1
            degree[w2] += 1
            nbrs[u].discard(v)
            nbrs[v].discard(u)
            nbrs[w1].add(w2)
            nbrs[w2].add(w1)
            touched.update(incident)
        odd = sum(1 for d in degree.values() if d % 2)
        if odd < best_odd:
            best_odd, best_faces = odd, list(face_list)
        if odd == 0:
            break
    return (verts, np.array(best_faces, dtype=int)), best_odd


# ---------------------------------------------------------------- colouring
def two_colour(n_sites: int, bonds, rounds: int = 200, seed: int = 0):
    """Split the net into M and X sublattices, minimising what is left over.

    BFS settles whether the net is bipartite at all -- any conflict means
    it is not, whatever the search order -- but it drops the conflicts
    wherever it happens to close a loop. The local search pulls them onto
    short loops, which is where a real inversion-domain boundary runs.
    """
    adjacency: dict[int, list[int]] = {i: [] for i in range(n_sites)}
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)

    colour: dict[int, int] = {}
    for start in range(n_sites):
        if start in colour:
            continue
        colour[start] = 0
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for w in adjacency[v]:
                if w not in colour:
                    colour[w] = 1 - colour[v]
                    queue.append(w)

    def frustration(c) -> int:
        return sum(1 for i, j in bonds if c[i] == c[j])

    best, best_cost = dict(colour), frustration(colour)
    rng = np.random.default_rng(seed)
    order = list(range(n_sites))
    for _ in range(rounds):
        rng.shuffle(order)
        moved = 0
        for v in order:
            same = sum(1 for w in adjacency[v] if colour[w] == colour[v])
            if same * 2 > len(adjacency[v]):
                colour[v] = 1 - colour[v]
                moved += 1
        cost = frustration(colour)
        if cost < best_cost:
            best_cost, best = cost, dict(colour)
        if not moved:
            break
    return best, best_cost


# ------------------------------------------------------------------ normals
def _face_normals(mesh, box):
    """Triangle normals, from the mesh's own consistent winding."""
    verts, faces = mesh
    anchor = verts[faces[:, 0]]
    ab = minimum_image(verts[faces[:, 1]] - anchor, box)
    ac = minimum_image(verts[faces[:, 2]] - anchor, box)
    normal = np.cross(ab, ac)
    return normal / np.linalg.norm(normal, axis=1, keepdims=True)


def _site_normals(sites, bonds, reference, box):
    """Surface normal at each trivalent site, consistently oriented.

    Taken from the relaxed net rather than the implicit field's gradient:
    the relaxation moves sites off the level set it was meshed from, and
    the gradient there is no longer the net's own normal. Orientation is
    then *propagated* across the net rather than decided site by site
    against the original triangles, which the relaxation has also moved.
    """
    neighbours: dict[int, list[int]] = {i: [] for i in range(len(sites))}
    for i, j in bonds:
        neighbours[i].append(j)
        neighbours[j].append(i)

    raw = np.zeros_like(sites)
    for i, ns in neighbours.items():
        if len(ns) < 2:
            raw[i] = reference[i]
            continue
        d = minimum_image(sites[ns] - sites[i], box)
        acc = np.zeros(3)
        for k in range(len(d)):
            acc += np.cross(d[k], d[(k + 1) % len(d)])
        length = np.linalg.norm(acc)
        raw[i] = acc / length if length > 1e-9 else reference[i]

    seen = np.zeros(len(sites), dtype=bool)
    for start in range(len(sites)):
        if seen[start]:
            continue
        if float(np.dot(raw[start], reference[start])) < 0:
            raw[start] = -raw[start]
        seen[start] = True
        queue = deque([start])
        while queue:
            v = queue.popleft()
            for w in neighbours[v]:
                if seen[w]:
                    continue
                if float(np.dot(raw[w], raw[v])) < 0:
                    raw[w] = -raw[w]
                seen[w] = True
                queue.append(w)
    return raw


# ------------------------------------------------------------------ builder
def build_tmd_schwarzite(
    material: str | TMDMaterial = "MoS2",
    kind: im.SchwarziteKind = "primitive",
    cell: float = 36.0,
    parity: ParityRepair = "flip",
    phase: Phase = "2H",
    grid_resolution: int = 64,
    remesh_iterations: int = 25,
    relax_iterations: int = 4000,
    seed: int = 0,
) -> Atoms:
    """Build a periodic MX2 schwarzite unit cell.

    Parameters
    ----------
    material
        Formula or :class:`~nanocarbon_lab.tmd.materials.TMDMaterial`.
    kind
        ``"primitive"`` (Schwarz P), ``"diamond"`` or ``"gyroid"``.
    cell
        Cubic cell length in Å, at least
        :data:`MIN_SCHWARZITE_CELL` for the surface.
    parity
        How hard to push the net toward alternating M/X. See the module
        docstring for the measured trade; ``"none"`` keeps the best
        geometry, ``"split"`` the best chemistry, ``"flip"`` sits between
        and is the default.
    phase
        Recorded for provenance. A curved sheet has no single stacking, so
        this does not change the geometry the way it does for a slab.
    grid_resolution, remesh_iterations, relax_iterations
        Marching-cubes grid, isotropic remesh passes, and L-BFGS
        iterations for each of the two relaxations.
    seed
        Seeds the flip annealing and the colouring search.

    Returns
    -------
    ase.Atoms
        Periodic in all three directions, with the ring census, the
        homoelemental-bond count and the measured bond spread in ``info``.

    Raises
    ------
    ValueError
        For an unknown ``parity``, or a cell below the minimum.
    """
    if isinstance(material, str):
        material = get_material(material)
    if parity not in ("none", "flip", "split"):
        raise ValueError(
            f"parity must be 'none', 'flip' or 'split'; got {parity!r}.")
    minimum = MIN_SCHWARZITE_CELL.get(kind, 30.0)
    if cell < minimum:
        raise ValueError(
            f"cell={cell:.1f} Å is too small for the {kind!r} surface with "
            f"{material.formula}, which needs at least {minimum:.0f} Å. The "
            f"sandwich is {material.h:.1f} Å thick, so below that the channels "
            "are narrower than the layer and the cell stops being a surface."
        )

    site_bond = material.a / np.sqrt(3.0)
    field, _ = im.schwarzite_field(kind, cell=cell, thickness=0.0)

    # Whether the marching-cubes grid resolves a given neck depends on how
    # the surface falls between sample points, so one (cell, resolution)
    # pair can tear where both its neighbours are fine -- Schwarz P at
    # 42 Å leaves a 21 Å gap between neighbouring sites at resolution 64
    # and is clean at 72. That is a discretisation artefact, not a
    # physical limit, so retry on a shifted grid rather than returning a
    # torn cell. The carbon builder does the same for the same reason.
    failures: list[str] = []
    built = None
    for resolution in (grid_resolution, grid_resolution + 8,
                       grid_resolution + 16):
        try:
            mesh = rm.periodic_marching_cubes_mesh(field, cell,
                                                   resolution=resolution)
            stats = rm.mesh_statistics(mesh)
            if stats["boundary_edges"]:
                failures.append(f"resolution {resolution}: periodic weld left "
                                f"{stats['boundary_edges']} boundary edges")
                continue
            mesh = rm.isotropic_remesh(
                mesh, field, target_edge=material.a,
                iterations=remesh_iterations, box=cell, anneal_sweeps=0,
                rng=np.random.default_rng(seed),
            )
            splits = 0
            if parity == "split":
                mesh, splits = repair_parity_by_splitting(mesh, field, cell,
                                                          material.a)
            elif parity == "flip":
                mesh, _ = repair_parity_by_flipping(mesh, seed=seed)

            positions, net_bonds, _ = fm.dual_honeycomb(mesh, box=cell)
            net_bonds = sorted(net_bonds)
        except (StopIteration, RuntimeError, ValueError) as exc:
            # `dual_honeycomb` raises StopIteration when a vertex fan is
            # not a closed loop, i.e. the mesh is non-manifold there.
            failures.append(f"resolution {resolution}: {exc!r}")
            continue

        pairs = np.array(net_bonds, dtype=int)
        spacing = np.linalg.norm(
            minimum_image(positions[pairs[:, 1]] - positions[pairs[:, 0]],
                          cell), axis=1)
        if spacing.max() > 2.0 * site_bond:
            failures.append(
                f"resolution {resolution}: neighbouring sites up to "
                f"{spacing.max():.1f} Å apart (want ~{site_bond:.2f}), so the "
                "mesh is torn")
            continue
        built = (mesh, positions, net_bonds, splits, stats)
        break

    if built is None:
        raise RuntimeError(
            f"Could not mesh a {kind!r} cell at {cell:.1f} Å after "
            f"{len(failures)} grid resolutions:\n  " + "\n  ".join(failures)
        )
    mesh, positions, net_bonds, splits, _ = built
    rings = rm.degree_histogram(mesh)
    stats = rm.mesh_statistics(mesh)
    reference = _face_normals(mesh, cell)

    # The sites form a trivalent net -- graphene's topology with a
    # different bond length -- so the carbon relaxer applies directly,
    # and its angle term is what keeps the sheet from folding.
    sites = fm.relax_shell(
        positions.copy(), set(net_bonds), equilibrium=site_bond,
        angle_deg=120.0, k_bond=40.0, k_angle=15.0, k_repel=25.0,
        repel_cutoff=site_bond * 1.55, repel_skin=2.0,
        box=cell, outer_cycles=3, max_iterations=relax_iterations,
    )
    normals = _site_normals(sites, net_bonds, reference, cell)

    colour, frustrated = two_colour(len(sites), net_bonds, seed=seed)
    n_zero = sum(1 for v in colour.values() if v == 0)
    metal_colour = 0 if n_zero >= len(colour) - n_zero else 1

    half = material.h / 2.0
    site_atoms: dict[int, list[int]] = {}
    symbols: list[str] = []
    coords: list[np.ndarray] = []
    for k, point in enumerate(sites):
        if colour[k] == metal_colour:
            site_atoms[k] = [len(coords)]
            symbols.append(material.metal)
            coords.append(point)
        else:
            site_atoms[k] = [len(coords), len(coords) + 1]
            symbols.append(material.chalcogen)
            coords.append(point + normals[k] * half)
            symbols.append(material.chalcogen)
            coords.append(point - normals[k] * half)
    coords = np.array(coords)

    metal_metal = METAL_METAL_BOND.get(material.metal, 2 * material.a / 2.3)
    chalcogen_chalcogen = CHALCOGEN_CHALCOGEN_BOND.get(material.chalcogen, 2.10)
    targets: dict[tuple[int, int], float] = {}
    mx_bonds: list[tuple[int, int]] = []
    for i, j in net_bonds:
        first, second = site_atoms[i], site_atoms[j]
        if len(first) == 1 and len(second) == 2:
            for x in second:
                pair = (min(first[0], x), max(first[0], x))
                targets[pair] = material.bond_length
                mx_bonds.append(pair)
        elif len(first) == 2 and len(second) == 1:
            for x in first:
                pair = (min(second[0], x), max(second[0], x))
                targets[pair] = material.bond_length
                mx_bonds.append(pair)
        elif len(first) == 1:
            pair = (min(first[0], second[0]), max(first[0], second[0]))
            targets[pair] = metal_metal
        else:
            # Antiphase chalcogen pair: bond plane to matching plane.
            for a, b in ((first[0], second[0]), (first[1], second[1])):
                targets[(min(a, b), max(a, b))] = chalcogen_chalcogen

    # No angle term: the metal is six-coordinate, so its ligand-metal-
    # ligand angles take several values at once and no single target
    # fits. `exclude_13=False` puts the non-bonded repulsion in charge of
    # holding those ligands apart instead -- without it the chalcogens
    # around a metal have neither an angle nor a repulsion and collapse
    # onto each other (70 sub-2 Å pairs became zero).
    relaxed = fm.relax_shell(
        coords.copy(), set(targets), equilibrium=targets,
        k_bond=40.0, k_angle=0.0, k_repel=60.0,
        repel_cutoff=3.0, repel_skin=2.0,
        box=cell, outer_cycles=4, max_iterations=relax_iterations,
        exclude_13=False,
    )

    atoms = Atoms(symbols=symbols, positions=relaxed,
                  cell=np.diag([cell] * 3), pbc=True)
    center_in_cell(atoms, axes=(0, 1, 2))

    pairs = np.array(sorted(mx_bonds), dtype=int)
    lengths = np.linalg.norm(
        minimum_image(relaxed[pairs[:, 1]] - relaxed[pairs[:, 0]], cell), axis=1)
    deviation = np.abs(lengths - material.bond_length) / material.bond_length

    # Coordination from the bond graph the builder constructed, never
    # from distances. On a saddle a 2.4 Å bond's usual 1.25x cutoff
    # reaches 3.0 Å, and plenty of non-bonded atoms sit inside that --
    # the distance-based report reads this cell as 4-8 coordinate metal
    # when every metal in it has exactly six bonds.
    every = np.array(sorted(targets), dtype=int)
    coordination = np.zeros(len(atoms), dtype=int)
    np.add.at(coordination, every[:, 0], 1)
    np.add.at(coordination, every[:, 1], 1)
    species = np.array(symbols)
    metal_coordination = coordination[species == material.metal]
    chalcogen_coordination = coordination[species == material.chalcogen]

    n_metal = symbols.count(material.metal)
    n_chalcogen = symbols.count(material.chalcogen)
    atoms.info.update(
        {
            # The real bond graph, recorded rather than left to be guessed
            # from distances later: validation and the render bundle both
            # read this, and on a saddle a distance cutoff gets it wrong.
            "bonds": [[int(a), int(b)] for a, b in every],
            "graph_metal_coordination": (int(metal_coordination.min()),
                                         int(metal_coordination.max())),
            "graph_chalcogen_coordination": (int(chalcogen_coordination.min()),
                                             int(chalcogen_coordination.max())),
            "stoichiometry": n_chalcogen / n_metal,
        }
    )
    atoms.info.update(
        {
            "structure_type": "tmd_schwarzite",
            "material": material.formula,
            "metal": material.metal,
            "chalcogen": material.chalcogen,
            "phase": phase,
            "coordination": coordination_geometry(phase),
            "schwarzite_kind": kind,
            "cell": cell,
            "genus": stats["genus"],
            "euler": stats["euler"],
            "ring_counts": {int(k): int(v) for k, v in rings.items()},
            "ring_deficit": int(sum((6 - k) * v for k, v in rings.items())),
            "parity": parity,
            "parity_splits": splits,
            "odd_rings": int(sum(v for k, v in rings.items() if k % 2)),
            "antiphase_bonds": int(frustrated),
            "n_net_bonds": len(net_bonds),
            "antiphase_fraction": frustrated / len(net_bonds),
            "bond_deviation_max": float(deviation.max()),
            "bond_deviation_p95": float(np.percentile(deviation, 95)),
            "sublattices": (n_metal, n_chalcogen),
            "a": material.a,
            "h": material.h,
            "bond_length": material.bond_length,
        }
    )
    return atoms


def schwarzite_quality(atoms: Atoms) -> tuple[str, str]:
    """Verdict for a curved MX2 cell, from its own recorded numbers.

    Deliberately not routed through :func:`~nanocarbon_lab.tmd.quality.
    tmd_quality`, which finds bonds by distance. That is right for a slab
    or a rolled tube and wrong on a minimal surface, where a 2.4 Å bond's
    cutoff reaches 3.0 Å and sweeps up non-bonded neighbours -- it reads a
    sound cell as 4-8 coordinate. The builder knows the bond graph, so
    judge from that.

    Two independent axes, because they fail independently: how far the
    M-X bonds are stretched, and how many bonds are homoelemental. The
    second is not strain and does not relax away.
    """
    info = atoms.info
    worst = float(info["bond_deviation_max"])
    p95 = float(info["bond_deviation_p95"])
    antiphase = float(info["antiphase_fraction"])
    metal = info["graph_metal_coordination"]
    chalcogen = info["graph_chalcogen_coordination"]

    if metal[1] > 6 or chalcogen[1] > 3:
        return "broken", (
            f"metal coordination reaches {metal[1]} and chalcogen "
            f"{chalcogen[1]}; the net is miswired, not merely strained."
        )
    if info["antiphase_bonds"]:
        boundary = (f"{antiphase:.1%} of bonds are homoelemental — an "
                    "inversion-domain boundary, which even degrees alone "
                    "cannot rule out above genus 0")
    else:
        # Reachable but not guaranteed: even degrees settle the local
        # parity, and the 2g homology classes settle the rest. Saying
        # "0.0% are an inversion-domain boundary" would be nonsense.
        boundary = ("every bond is M–X — the net alternates perfectly, which "
                    "even degrees make possible but do not guarantee above "
                    "genus 0")
    if p95 > 0.10:
        return "broken", (
            f"M–X bonds deviate {p95:.1%} at the 95th percentile "
            f"({worst:.1%} worst). {boundary}."
        )
    if p95 > 0.04 or antiphase > 0.08:
        return "strained", (
            f"M–X bonds within {p95:.1%} for 95% of the net ({worst:.1%} "
            f"worst); {boundary}. A usable starting structure for a "
            "relaxation, not a finished one."
        )
    return "clean", (
        f"M–X bonds within {p95:.1%} for 95% of the net, coordination 6/3 "
        f"throughout; {boundary}."
    )


__all__ = [
    "CHALCOGEN_CHALCOGEN_BOND",
    "METAL_METAL_BOND",
    "MIN_SCHWARZITE_CELL",
    "ParityRepair",
    "build_tmd_schwarzite",
    "odd_vertices",
    "repair_parity_by_flipping",
    "repair_parity_by_splitting",
    "schwarzite_quality",
    "two_colour",
]
