"""Why there is no MX2 schwarzite builder yet — reproducible in one run.

This is a negative result kept as runnable code, so the next attempt
starts from the measurement rather than from the argument.

The claim to test. Atoms sit at triangle centroids and bond across shared
mesh edges, so the net is the mesh's dual: **vertex degree == ring size**.
A binary M/X net must alternate around every ring, i.e. the dual graph
must be 2-colourable, and a triangulation's faces are 2-colourable
exactly when every vertex degree is even. So "MoS2 schwarzite" reduces to
"even-degree triangulation of a triply periodic minimal surface".

Note what this does *not* say. Pentagons are forbidden, but negative
curvature never needed them: `sum(6 - n) = 6*chi`, and an octagon pays -2
just as a heptagon does, while staying even. Schwarz P (chi = -4) wants
twelve octagons. The chemistry permits this; h-BN schwarzites in the
literature are built exactly this way.

Three things come out of running this:

1. Parity is solvable. An edge *flip* toggles four vertex degrees at once,
   so greedy flipping stalls with odd vertices sparse. An edge *split*
   toggles exactly the two opposite vertices -- a weight-two move -- so
   odd vertices can be walked together and annihilated in pairs. Result:
   every degree even, rings 4/6/8/10 only, `sum(6-n)` exactly 6*chi, and
   X/M = 2.000.

2. Even degrees are necessary but not sufficient above genus 0. The
   2-colourability theorem is a sphere result; at genus g there are 2g
   further Z/2 classes, and here they do not vanish. About 2% of bonds
   stay homoelemental no matter how the colouring is chosen -- which is a
   real inversion-domain boundary, the same defect seen in grown MoS2 and
   h-BN, not an artefact of the method.

3. The blocker is geometric. Each split inserts a vertex, so the repaired
   mesh carries more sites than the surface area holds at spacing
   a/sqrt(3). Triangle centroids then land on top of each other, and no
   optimiser fixes a density mismatch: relaxation, projection, Laplacian
   smoothing and alternating remesh-with-repair were each measured and
   none closes it.

Closing it needs a parity-preserving isotropic remesher: splits *and*
collapses paired so the site count tracks the area while degrees stay
even. Until then, building this would export a cell with hundreds of
sub-Angstrom atomic overlaps.

Run with::

    python -m nanocarbon_lab.examples.ex06_tmd_schwarzite_parity
"""

from __future__ import annotations

from collections import deque

import numpy as np

from ..builders import fullerene_mesh as fm
from ..builders import implicit as im
from ..builders import remesh as rm
# Private helpers: this is a diagnostic of the mesh internals, which is
# the one place reaching into them is the point rather than a shortcut.
from ..builders.remesh import _adjacency, _edge_faces
from ..tmd.materials import get_material


def apex_map(face_list: list[tuple[int, int, int]]) -> dict:
    """Map every interior edge to the two vertices opposite it."""
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


def split_edge(verts, face_list, edge, field, box):
    """Insert a midpoint on ``edge``; toggles the parity of its two apexes.

    ``m`` joins a, b and both apexes, so it arrives at degree 4 (even);
    a and b swap a neighbour for m and keep their degree; only c and d
    gain one. That is what makes this a weight-two move where a flip is
    weight-four.
    """
    a, b = edge
    incident = _edge_faces(np.array(face_list, dtype=int)).get(tuple(sorted(edge)))
    if incident is None or len(incident) != 2:
        return verts, face_list, False

    delta = verts[b] - verts[a]
    if box is not None:
        delta -= box * np.round(delta / box)  # shortest image across the seam
    midpoint = im.project_to_surface(field, (verts[a] + 0.5 * delta)[None, :])[0]
    new_index = len(verts)
    verts = np.vstack([verts, midpoint])

    for index in incident:
        tri = face_list[index]
        for k in range(3):
            p, q, r = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            if {p, q} == {a, b}:
                # (p, q, r) -> (p, m, r) + (m, q, r) keeps the winding.
                face_list[index] = (p, new_index, r)
                face_list.append((new_index, q, r))
                break
    return verts, face_list, True


def repair_parity(mesh, field, box, max_splits=500):
    """Split until every vertex degree is even."""
    verts = np.array(mesh[0], dtype=float)
    face_list = [tuple(int(x) for x in tri) for tri in mesh[1]]

    splits = 0
    for _ in range(max_splits):
        degree = {v: len(n) for v, n in
                  _adjacency(np.array(face_list, dtype=int)).items()}
        odd = {v for v, d in degree.items() if d % 2}
        if not odd:
            break
        graph: dict[int, list] = {}
        for edge, (c, d) in apex_map(face_list).items():
            graph.setdefault(c, []).append((d, edge))
            graph.setdefault(d, []).append((c, edge))

        source = min(odd)
        step = _first_hop(graph, source, odd)
        if step is None:
            break
        verts, face_list, ok = split_edge(verts, face_list, step, field, box)
        if not ok:
            break
        splits += 1
    return (verts, np.array(face_list, dtype=int)), splits


def _first_hop(graph, source, targets):
    """First edge of a shortest co-apex path from source to another odd."""
    if source not in graph:
        return None
    previous = {source: (source, None)}
    queue = deque([source])
    found = None
    while queue:
        v = queue.popleft()
        if v in targets and v != source:
            found = v
            break
        for w, edge in graph.get(v, ()):
            if w not in previous:
                previous[w] = (v, edge)
                queue.append(w)
    if found is None:
        return None
    node = found
    while previous[node][0] != source:
        node = previous[node][0]
    return previous[node][1]


def colour_net(n_sites, bonds, rounds=200, seed=0):
    """2-colour the net, then locally minimise the leftover frustration.

    BFS alone settles bipartiteness (any conflict at all means the graph
    is not bipartite, whatever the search order), but it drops the
    conflicts wherever it happens to close a loop. The local search pulls
    them onto short loops, which is where a real domain boundary sits.
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

    def frustration(c):
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


def investigate(kind="primitive", cell=36.0, material="MoS2"):
    mat = get_material(material)
    site_spacing = mat.a / np.sqrt(3.0)
    field, _ = im.schwarzite_field(kind, cell=cell, thickness=0.0)
    mesh = rm.periodic_marching_cubes_mesh(field, cell, resolution=64)
    mesh = rm.isotropic_remesh(
        mesh, field, target_edge=mat.a, iterations=25, box=cell,
        anneal_sweeps=0, rng=np.random.default_rng(0),
    )

    print(f"\n=== {mat.formula} on the {kind} surface, {cell:.0f} Å cell ===")
    before = rm.degree_histogram(mesh)
    print(f"  as remeshed : rings {dict(sorted(before.items()))}")
    print(f"                {sum(c for d, c in before.items() if d % 2)} odd "
          "(pentagons and heptagons — legal in carbon, not in MX2)")

    mesh, splits = repair_parity(mesh, field, cell)
    after = rm.degree_histogram(mesh)
    odd = sum(c for d, c in after.items() if d % 2)
    deficit = sum((6 - d) * c for d, c in after.items())
    stats = rm.mesh_statistics(mesh)
    chi = stats["euler"]
    print(f"\n  [1] parity  : {splits} splits -> {odd} odd. "
          f"rings {dict(sorted(after.items()))}")
    print(f"                sum(6-n) = {deficit}, 6*chi = {6 * chi}  "
          f"{'✓' if deficit == 6 * chi else '✗'}")

    positions, bonds, _ = fm.dual_honeycomb(mesh, box=cell)
    bonds = sorted(bonds)
    colour, frustrated = colour_net(len(positions), bonds)
    n_zero = sum(1 for v in colour.values() if v == 0)
    n_one = len(colour) - n_zero
    print(f"\n  [2] colour  : {frustrated} of {len(bonds)} bonds stay "
          f"homoelemental ({100.0 * frustrated / len(bonds):.2f}%)")
    print(f"                genus {stats['genus']} leaves "
          f"{2 * stats['genus']} Z/2 classes that even degrees do not fix")
    print(f"                sublattices {n_zero}/{n_one} -> "
          f"X/M = {2.0 * n_one / n_zero:.3f}")

    delta = positions[np.array(bonds)[:, 1]] - positions[np.array(bonds)[:, 0]]
    delta -= cell * np.round(delta / cell)
    spacing = np.linalg.norm(delta, axis=1)
    print(f"\n  [3] geometry: site spacing {spacing.min():.3f} / "
          f"{spacing.mean():.3f} / {spacing.max():.3f} Å "
          f"(want {site_spacing:.3f})")
    print(f"                {int((spacing < 0.5 * site_spacing).sum())} bonds "
          "under half the target — the splits added sites the surface area")
    print("                cannot hold, which is why this is not shipped.")


def main():
    investigate("primitive", 36.0)
    investigate("gyroid", 40.0)
    print("\nNext step: a parity-preserving isotropic remesher — splits and")
    print("collapses paired so the site count tracks the area while every")
    print("degree stays even. See CLAUDE.md for what was already ruled out.")


if __name__ == "__main__":
    main()
