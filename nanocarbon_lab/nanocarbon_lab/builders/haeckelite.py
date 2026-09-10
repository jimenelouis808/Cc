"""Haeckelites and other 2D carbon allotropes, by design rather than by menu.

A haeckelite is graphene's honeycomb with pentagons and heptagons tiled
into it periodically. Terrones' three originals -- R5,7, H5,6,7 and
O5,6,7 -- are the famous ones, but they are three points in a space, and
the point of this module is to let you move around that space rather than
pick from a list.

**The design move is the Stone-Wales rotation, applied to the mesh.**
That qualification is the whole module. Rotating a bond turns the four
rings around it into two pentagons and two heptagons, and it is tempting
to do that directly on the atoms' bond graph: swap one neighbour between
the two atoms and the degrees all stay at three. It was tried here first
and it is wrong. A Stone-Wales move is a move on an *embedded* graph, and
performed as bare rewiring it silently produces a graph that no longer
embeds in the torus at all. The symptom is not an exception: the topology
looks perfect -- 3-regular, right number of bonds -- and the relaxation
returns a sheet with 2.3 Å bonds that no ring census can describe.

So the work happens one level down, on the **triangulated dual**, exactly
as `fullerene_mesh` does for closed shells: each mesh vertex is a ring,
each triangle an atom, each edge a bond.
:func:`~nanocarbon_lab.builders.fullerene_mesh.edge_flip` takes a
triangulation to a triangulation, so the result always embeds, and the
ring census is then read straight off the vertex degrees rather than
perceived.

That is what makes the whole family safe to generate. On a torus the
Euler characteristic is zero, so a 2D periodic sheet must satisfy

    sum over rings of (6 - n) == 0

exactly -- the flat analogue of the +12 a fullerene owes and the 6*chi a
schwarzite owes. An edge flip drops two vertex degrees by one and raises
two by one, paying zero, so **it cannot break the budget however many you
apply, wherever you put them**. A pattern nobody has published is as
sound as R5,7.

**The topology comes from the mesh; the geometry does not.** This is the
second half of the same lesson, and it cost as much to learn. The obvious
next step, having flipped the triangulation, is to relax the *mesh* and
then put an atom at each triangle's centroid -- the rule `tmd/curved.py`
records for its site net. On a closed shell that works. On a flat sheet it
cannot, for a reason that is geometric rather than numerical: equilateral
triangles meeting five at a vertex sum to 300 deg and seven to 420, so a
**flat** triangulation carrying pentagons and heptagons can never have
equal edges. Asked for them anyway, the mesh relaxation sat exactly still
-- the flipped lattice is a genuine minimum of the edge springs, with the
long diagonals' forces cancelling to 2.6e-13 by symmetry -- and handed the
dual 0.82 Å bonds that no later relaxation could undo.

So the mesh supplies only the rewiring, and the geometry is built the way
a Stone-Wales rotation is actually drawn: from perfect graphene, with each
rotated **dimer turned 90 deg in the plane** about its own midpoint. A
single defect built this way relaxes to 1.32-1.48 Å bonds and
103.5-136.7 deg angles at graphene's own cell, which is the published
5-7-7-5 geometry. The turn has two senses and only one matches the
rewiring the flip performed; the other puts each new partner 2.40 Å away
instead of 1.51, so the sense is chosen per dimer by measuring both.

**Two rotations must not overlap, and a flip must land on clean hexagons.**
Both limits were found by building lattices that violated them. Dimers
sharing a bond each turn an atom the other one needs, and the pair ends up
3.4 Å apart -- so rotations are required to be pairwise non-adjacent.
Separately, a flip whose four touched mesh vertices are not all still
degree 6 stacks its degree changes onto an existing defect, which is sound
topology but not a haeckelite: a run without that rule returned squares
and nonagons.

Together they cap how dense a pattern this route can reach -- and that cap
turns out to be exactly where the geometry stops working. Across 105
combinations of cell size, pattern and seed, **every** lattice the rules
admit passes the sp2 gate in ``build_haeckelite``, and the ones they turn
down are the ones that came back at 1.225-1.663 Å when the rules were
absent. So the gate has not fired through the public API; it stays as a
guard, not as the mechanism. The reachable range is up to 67%
non-hexagonal at 1.27-1.57 Å.

**The sheet is kept flat, deliberately.** The force field has bond and
angle terms but no flexural one, because a sheet's resistance to bending
comes from its pi system. Long-wavelength wrinkling therefore costs almost
nothing in it, and given the freedom the cell search took it: a 4x4 R5,7
came back at 1.51 Å^2 per atom against graphene's 2.62, having bought low
bond strain by crumpling into a smaller footprint. Relaxing in-plane
removes a degree of freedom the force field cannot price. Any buckling
amplitude this module could report would be an artefact, so it reports
none.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND
from . import fullerene_mesh as fm

#: In-plane strain per cycle beyond which the rescaling is judged to have
#: converged. 1e-4 is far below anything the geometry report resolves.
CELL_TOLERANCE: float = 1e-4

#: Vacuum gap (Å) above and below the sheet, added to its own thickness
#: the way every other builder here adds it. The framework's guardrail for
#: a 2D structure is 12 Å.
DEFAULT_VACUUM: float = 15.0

#: The window a finished lattice has to fall inside. Every bound is
#: measured, and which one does the work is not what it looks like.
#:
#: A single Stone-Wales defect relaxes here to 1.32-1.48 Å and
#: 103.5-136.7 deg, matching the published 5-7-7-5 geometry. Dense
#: patterns concentrate real strain and come out at 1.245-1.528 Å and
#: 99.8-140.3 deg -- sound by every other measure: exact census, zero
#: contacts, graphene's area per atom. A *frustrated* arrangement,
#: meanwhile, converges to 1.225-1.663 Å and 97.1-144.0 deg.
#:
#: So the **ceiling discriminates and the floor does not**: the two
#: populations' short bonds overlap (1.245 against 1.225) while their long
#: bonds do not (1.528 against 1.663). The floor is therefore set just
#: above a C-C triple bond, where carbon stops being carbon at all, and
#: the real judgement is left to the ceiling and the angles. A lattice
#: that passes with a bond under `BOND_SOFT_FLOOR` is warned about
#: instead, because that much compression is this force field's artefact
#: -- it has one rest length for every bond -- and not a prediction.
BOND_FLOOR: float = 1.22
BOND_SOFT_FLOOR: float = 1.30
BOND_CEILING: float = 1.60
ANGLE_FLOOR: float = 95.0
ANGLE_CEILING: float = 145.0


def triangular_torus_mesh(m: int, n: int, spacing: float
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A periodic triangular mesh on a torus, in a **rectangular** cell.

    This is graphene's dual: every vertex is a ring, every triangle is an
    atom, every edge is a bond. Starting from it rather than from the
    honeycomb is what makes the whole design space safe, because
    :func:`~nanocarbon_lab.builders.fullerene_mesh.edge_flip` is a move on
    a *triangulation* and therefore always returns a triangulation --
    which is to say, always returns something that still embeds in the
    torus.

    The rectangular setting is the two-site description of the triangular
    lattice: sites at ``(i, j)`` and at ``(i + 1/2, j + 1/2)``. The
    natural rhombic cell would make the honeycomb dual hexagonal, and a
    hexagonal cell needs the full matrix for minimum image, which the
    relaxer's componentwise convention cannot do.

    Parameters
    ----------
    m, n
        Repeats along x and y. The dual honeycomb then holds ``4*m*n``
        atoms.
    spacing
        Triangular lattice constant (Å). The dual honeycomb's bond comes
        out at ``spacing / sqrt(3)``.

    Returns
    -------
    (vertices, triangles, box)
        ``box`` is the three cell lengths; z is generous rather than 0
        because the mesh is flat and ``dual_honeycomb`` wraps into it.
    """
    if m < 3 or n < 2:
        raise ValueError(
            f"m must be at least 3 and n at least 2, got {m} and {n}. At "
            "m = 2 a vertex's two x-neighbours are the same vertex through "
            "opposite images, so four edges collapse onto two and the "
            "triangulation stops being a simple graph: the degrees read 5 "
            "instead of 6 and some edges border four triangles, which is "
            "not a surface and which `edge_flip` cannot act on. Along y the "
            "two sublattices are offset by half a cell, so n = 2 is fine."
        )

    height = math.sqrt(3.0) * spacing
    vertices = np.zeros((2 * m * n, 3), dtype=float)

    def vertex(i: int, j: int, site: int) -> int:
        return 2 * ((j % n) * m + (i % m)) + site

    for j in range(n):
        for i in range(m):
            vertices[vertex(i, j, 0)] = [i * spacing, j * height, 0.0]
            vertices[vertex(i, j, 1)] = [(i + 0.5) * spacing,
                                         (j + 0.5) * height, 0.0]

    # Four triangles per cell, which is 4*m*n in total -- exactly the
    # number of atoms the dual will have. Each is equilateral with side
    # `spacing`; a test pins that rather than trusting the arithmetic.
    triangles: list[tuple[int, int, int]] = []
    for j in range(n):
        for i in range(m):
            centre = vertex(i, j, 1)
            triangles.append((centre, vertex(i, j, 0), vertex(i + 1, j, 0)))
            triangles.append((centre, vertex(i + 1, j + 1, 0),
                              vertex(i, j + 1, 0)))
            triangles.append((centre, vertex(i + 1, j, 0),
                              vertex(i + 1, j, 1)))
            triangles.append((centre, vertex(i + 1, j, 1),
                              vertex(i + 1, j + 1, 0)))

    box = np.array([m * spacing, n * height, 1000.0])
    return vertices, np.array(triangles, dtype=int), box


def mesh_edges(triangles: np.ndarray) -> list[tuple[int, int]]:
    """Every edge of the triangulation, once, sorted."""
    edges: set[tuple[int, int]] = set()
    for triangle in triangles:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            first, second = int(triangle[a]), int(triangle[b])
            edges.add((min(first, second), max(first, second)))
    return sorted(edges)


def flippable(triangles: np.ndarray, first: int, second: int) -> bool:
    """Whether flipping this edge leaves a valid triangulation.

    Two conditions. The edge must border exactly two triangles, which on
    a closed surface every edge does. And the two opposite vertices must
    not already share an edge -- flipping onto an existing edge would
    give the mesh a doubled edge, and the dual of that is two rings
    sharing two bonds, which is not a surface.
    """
    incident = [tuple(int(x) for x in t) for t in triangles
                if first in t and second in t]
    if len(incident) != 2:
        return False
    opposite: set[int] = set()
    for triangle in incident:
        opposite |= set(triangle) - {first, second}
    if len(opposite) != 2:
        return False
    left, right = sorted(opposite)
    return not any(left in t and right in t for t in triangles)


def _neighbours(bonds: set[tuple[int, int]], n_atoms: int) -> list[list[int]]:
    table: list[list[int]] = [[] for _ in range(n_atoms)]
    for first, second in bonds:
        table[first].append(second)
        table[second].append(first)
    return table


def _vertex_degrees(triangles: np.ndarray, n_vertices: int) -> np.ndarray:
    """Degree of every mesh vertex, which *is* the size of its dual ring."""
    degrees = np.zeros(n_vertices, dtype=int)
    for first, second in mesh_edges(triangles):
        degrees[first] += 1
        degrees[second] += 1
    return degrees


def _dimer_of(triangles: np.ndarray, labels: list[int],
              first: int, second: int
              ) -> tuple[int, int, int, int, list[int], list[int]] | None:
    """The two atoms a flip of mesh edge ``(first, second)`` rotates.

    The two triangles sharing the edge are the two atoms of the
    Stone-Wales dimer, so the flip's dual is a rotation of exactly those
    two. ``labels`` carries each triangle's identity through earlier
    flips, because :func:`fullerene_mesh.edge_flip` appends its two new
    triangles at the end rather than replacing in place -- so triangle
    *indices* move and the atom behind them has to be tracked.

    Returns ``(atom_a, atom_b, slot_a, slot_b, incident, touched)`` or
    ``None`` if the edge is not a clean interior edge.
    """
    face_list = [tuple(int(x) for x in f) for f in triangles]
    incident = [k for k, f in enumerate(face_list)
                if first in f and second in f]
    if len(incident) != 2:
        return None
    touched = set()
    for k in incident:
        touched |= set(face_list[k])
    opposite = sorted(touched - {first, second})
    if len(opposite) != 2:
        return None
    left, _ = opposite
    # edge_flip appends (u, w1, w2) then (v, w2, w1); the first keeps w1,
    # so it inherits whichever incident triangle already held w1.
    holds_left = incident[0] if left in face_list[incident[0]] else incident[1]
    holds_right = incident[1] if holds_left == incident[0] else incident[0]
    return (labels[holds_left], labels[holds_right],
            holds_left, holds_right, incident, sorted(touched))


def _turn_dimers(positions: np.ndarray, dimers: list[tuple[int, int]],
                 bonds: set[tuple[int, int]], box: np.ndarray,
                 bond: float) -> np.ndarray:
    """Turn each Stone-Wales dimer 90 deg in the plane about its midpoint.

    The sense is chosen per dimer by measuring both: a 90 deg turn has two
    of them, and only the one matching the rewiring the mesh flip
    performed brings each new partner to 1.51 Å. The other puts it at
    2.40 Å, which is what a first version did -- silently, because the
    topology was impeccable either way.
    """
    table: dict[int, list[int]] = {}
    for first, second in bonds:
        table.setdefault(first, []).append(second)
        table.setdefault(second, []).append(first)
    out = np.array(positions, dtype=float, copy=True)
    for atom_a, atom_b in dimers:
        half = fm.minimum_image(out[atom_b] - out[atom_a], box) / 2.0
        midpoint = out[atom_a] + half
        turned = np.array([-half[1], half[0], 0.0])
        outer = [(atom_a, n) for n in table[atom_a] if n != atom_b]
        outer += [(atom_b, n) for n in table[atom_b] if n != atom_a]
        best: tuple[float, float] | None = None
        for sense in (1.0, -1.0):
            out[atom_a] = midpoint - sense * turned
            out[atom_b] = midpoint + sense * turned
            cost = sum(
                (float(np.linalg.norm(fm.minimum_image(
                    out[q] - out[p], box))) - bond) ** 2
                for p, q in outer)
            if best is None or cost < best[0]:
                best = (cost, sense)
        assert best is not None
        out[atom_a] = midpoint - best[1] * turned
        out[atom_b] = midpoint + best[1] * turned
    return out


def _edge_direction(vertices: np.ndarray, box: np.ndarray,
                    first: int, second: int) -> str:
    """Which of the triangular lattice's three directions an edge runs along."""
    delta = fm.minimum_image(vertices[second] - vertices[first], box)
    angle = math.degrees(math.atan2(delta[1], delta[0])) % 180.0
    if angle < 30.0 or angle >= 150.0:
        return "a"
    if angle < 90.0:
        return "b"
    return "c"


#: How a pattern chooses which mesh edges to flip. Each is a rule over
#: the supercell, so the same name gives a consistent lattice at any
#: size -- which is what makes a design reproducible and a sweep
#: meaningful.
PATTERNS = ("r57", "stripes", "sparse", "random", "none")


def select_flips(vertices: np.ndarray, triangles: np.ndarray,
                 box: np.ndarray, m: int, n: int,
                 pattern: str = "r57", period: int = 2,
                 density: float = 0.15,
                 seed: int | None = 0) -> list[tuple[int, int]]:
    """Which mesh edges a pattern flips.

    Parameters
    ----------
    pattern
        ``"r57"``      every eligible edge of one lattice direction, the
                       densest 5-7 tiling this construction reaches;
        ``"stripes"``  the same but only every ``period``-th row, so bands
                       of pentagons and heptagons alternate with graphene;
        ``"sparse"``   one flip every ``period`` cells in **both**
                       directions -- an isolated 5-7-7-5 in a graphene
                       matrix, repeated as a superlattice, which is what
                       defect-engineered graphene actually looks like;
        ``"random"``   a fraction ``density`` of the eligible edges,
                       seeded -- the setting for lattices nobody has drawn;
        ``"none"``     nothing, so the builder returns graphene and every
                       downstream check has a baseline.

    Returns
    -------
    list of (u, v)
        Mesh edges to flip, **vertex-disjoint and re-validated in order**.
        Two flips sharing a vertex would each be chosen against a mesh the
        other had already changed; and even disjoint ones can make a later
        flip illegal, so each is re-checked against the mesh as it stands
        when its turn comes rather than against the mesh at selection time.
    """
    if pattern not in PATTERNS:
        raise ValueError(
            f"Unknown pattern {pattern!r}; expected one of {list(PATTERNS)}."
        )
    if pattern == "none":
        return []
    if period < 1:
        raise ValueError(f"period must be at least 1, got {period}.")
    if pattern == "random" and not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}.")

    height = box[1] / max(1, n)
    width = box[0] / max(1, m)
    # One lattice direction only. Picking one is what makes the result a
    # lattice rather than a scatter; the other two are equivalent by
    # symmetry.
    candidates = [(u, v) for u, v in mesh_edges(triangles)
                  if _edge_direction(vertices, box, u, v) == "a"]

    chosen: list[tuple[int, int]] = []
    for first, second in candidates:
        midpoint = vertices[first] + 0.5 * fm.minimum_image(
            vertices[second] - vertices[first], box)
        row = int(math.floor(midpoint[1] / height + 1e-6)) % max(1, n)
        column = int(math.floor(midpoint[0] / width + 1e-6)) % max(1, m)
        if pattern in ("r57", "random"):
            chosen.append((first, second))
        elif pattern == "stripes" and row % period == 0:
            chosen.append((first, second))
        elif (pattern == "sparse"
              and row % period == 0 and column % period == 0):
            chosen.append((first, second))

    if pattern == "random":
        rng = np.random.default_rng(seed)
        wanted = int(round(density * len(chosen)))
        order = rng.permutation(len(chosen))
        chosen = sorted(chosen[int(k)] for k in order[:wanted])

    used: set[int] = set()
    disjoint: list[tuple[int, int]] = []
    for first, second in chosen:
        if first in used or second in used:
            continue
        used.update((first, second))
        disjoint.append((first, second))
    return disjoint


def apply_flips(vertices: np.ndarray, triangles: np.ndarray,
                mesh_box: np.ndarray, flips: list[tuple[int, int]],
                bonds: set[tuple[int, int]], n_atoms: int
                ) -> tuple[np.ndarray, list[int], list[tuple[int, int]], int]:
    """Apply the flips a pattern asked for, keeping only the legal ones.

    Three rules, each of which a build violated first:

    * the edge must still be flippable against the mesh **as it stands**,
      not as it stood when the pattern was drawn;
    * all four mesh vertices the flip touches must still be degree 6, so
      the result is a clean 5-7-7-5 rather than a degree change stacked
      onto an existing defect -- without this the census came back with
      squares and nonagons;
    * the two atoms it rotates must not be, or be bonded to, an atom
      another accepted flip already rotates. Two dimers sharing a bond
      each turn an atom the other needs and the pair lands 3.4 Å apart.

    Returns ``(triangles, labels, dimers, refused)``, where ``labels[k]``
    is the atom behind final triangle ``k`` and ``dimers`` are the pairs
    to turn, in graphene's own atom indexing.
    """
    labels = list(range(len(triangles)))
    dimers: list[tuple[int, int]] = []
    blocked: set[int] = set()
    table = _neighbours(bonds, n_atoms)
    refused = 0
    for first, second in flips:
        if not flippable(triangles, first, second):
            refused += 1
            continue
        found = _dimer_of(triangles, labels, first, second)
        if found is None:
            refused += 1
            continue
        atom_a, atom_b, slot_a, slot_b, incident, touched = found
        if atom_a in blocked or atom_b in blocked:
            refused += 1
            continue
        degrees = _vertex_degrees(triangles, len(vertices))
        if any(degrees[k] != 6 for k in touched):
            refused += 1
            continue
        keep = [k for k in range(len(triangles)) if k not in incident]
        _, triangles = fm.edge_flip((vertices, triangles), first, second)
        labels = [labels[k] for k in keep] + [labels[slot_a], labels[slot_b]]
        dimers.append((atom_a, atom_b))
        for atom in (atom_a, atom_b):
            blocked.add(atom)
            blocked.update(table[atom])
    return triangles, labels, dimers, refused


def strain_score(positions: np.ndarray, bonds: set[tuple[int, int]],
                 box: np.ndarray, bond: float,
                 angle_deg: float = 120.0) -> float:
    """How far a relaxed sheet is from ideal sp2, as one number.

    Mean squared bond strain plus mean squared angle deviation, weighted
    the way the force field weights them. **Both terms are needed.** Bond
    strain alone cannot find the right cell: squeeze a sheet and it
    buckles rather than compressing its bonds, so the bonds stay at
    1.42 Å while the cell collapses -- which is exactly what a
    mean-bond-matching rule did here, contracting a 4x4 cell by 24% and
    tearing the sheet while every bond looked perfect. Buckling costs
    *angle* energy, and that is what the second term sees.
    """
    pairs = sorted(bonds)
    first = np.array([a for a, _ in pairs])
    second = np.array([b for _, b in pairs])
    lengths = np.linalg.norm(
        fm.minimum_image(positions[second] - positions[first], box), axis=1)
    bond_term = float(np.mean(((lengths - bond) / bond) ** 2))

    table = _neighbours(bonds, len(positions))
    deviations = []
    target = math.radians(angle_deg)
    for centre, neighbours in enumerate(table):
        for i in range(len(neighbours)):
            for j in range(i + 1, len(neighbours)):
                u = fm.minimum_image(positions[neighbours[i]] - positions[centre], box)
                v = fm.minimum_image(positions[neighbours[j]] - positions[centre], box)
                cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                deviations.append(math.acos(float(np.clip(cosine, -1.0, 1.0))) - target)
    angle_term = float(np.mean(np.square(deviations))) if deviations else 0.0

    # A sheet folded onto itself can have excellent bonds and angles
    # everywhere -- they are local, and folding is not. Without this veto
    # the cell search happily picked a collapsed candidate over a sound
    # one, because locally the wreckage scored better. The framework's
    # own rule, in reverse: zero close contacts does not prove a
    # structure is fine, but a close contact does prove it is not.
    from .capped_cnt import geometry_report

    report = geometry_report(positions, sorted(bonds),
                             box=np.array([box[0], box[1], 0.0]))
    if int(report["n_close_contacts"]):
        return float("inf")
    return 40.0 * bond_term + 15.0 * angle_term


def _relax_at(positions: np.ndarray, bonds: set[tuple[int, int]],
              box: np.ndarray, bond: float, iterations: int) -> np.ndarray:
    return fm.relax_shell(positions.copy(), bonds, equilibrium=bond,
                          box=box, max_iterations=iterations)


def _embeds(positions: np.ndarray, bonds: set[tuple[int, int]],
            box: np.ndarray, expected: dict[int, int]) -> bool:
    """Whether the relaxed sheet still embeds the mesh it came from.

    The mesh census is exact, so any candidate whose own faces disagree
    with it has torn -- and a tear does not need a close contact to
    happen, which is why the contact veto alone was not enough. Widening
    the cell search found a *lower-scoring* cell whose sheet traced
    {5: 35, 7: 32} against the mesh's {5: 36, 7: 36}: better local bonds
    and angles, on a surface that had come apart.
    """
    from ..analyse.rings import trace_faces

    sheet = Atoms(symbols=["C"] * len(positions), positions=positions,
                  pbc=(True, True, False))
    sheet.set_cell([box[0], box[1], 40.0])
    sheet.positions[:, 2] += 20.0
    faces, boundary = trace_faces(sheet, np.asarray(sorted(bonds), dtype=int),
                                  max_size=12)
    if boundary:
        return False
    traced: dict[int, int] = {}
    for face in faces:
        traced[len(face)] = traced.get(len(face), 0) + 1
    return traced == expected


def _fit_cell(positions: np.ndarray, bonds: set[tuple[int, int]],
              box: np.ndarray, bond: float, iterations: int,
              expected: dict[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the cell the topology actually wants, by relaxing at each.

    Coordinate descent on the two in-plane constants, wide then fine.
    Every candidate is a full relaxation at a fixed cell, scored by
    :func:`strain_score` and **required to still embed the mesh**. That
    is slower than a closed-form rule and it is the reason this converges
    at all: the score is the quantity the physics minimises, and any
    cheaper proxy the sheet can satisfy by wrinkling instead.

    **The cell barely moves, and that is the result rather than the
    premise.** An earlier version searched +-36% per axis because a dense
    pattern looked as though it wanted a far smaller cell -- but that was
    the sheet crumpling out of plane to shrink its own footprint, which
    the force field cannot price and which relaxing in-plane now prevents.
    Measured across the patterns, every lattice lands at 2.62-2.68 Å^2 per
    atom against graphene's 2.619, and for an unrotated sheet the search
    does not move at all. The span is left wider than those numbers need,
    since it only costs time and a build is a couple of seconds; what must
    not be relaxed is the flatness, or the old 1.51 Å^2-per-atom wreck
    comes back.

    Every candidate is still **required to still embed the mesh**, and
    that check is not redundant with the contact veto: widening the search
    once found a *lower-scoring* cell whose sheet traced {5: 35, 7: 32}
    against the mesh's {5: 36, 7: 36} -- better local bonds and angles, on
    a surface that had come apart.
    """
    box = np.array(box, dtype=float)
    best_box = box.copy()
    best_positions = _relax_at(positions, bonds, best_box, bond, iterations)
    best = strain_score(best_positions, bonds, best_box, bond)
    if best <= CELL_TOLERANCE:
        # Already at the cell the topology wants -- graphene's own, for an
        # unrotated sheet. Nothing further could do better than tie.
        best_positions[:, :2] = np.mod(best_positions[:, :2], best_box[:2])
        return best_positions, best_box, best

    def evaluate(factors: np.ndarray) -> None:
        nonlocal best, best_box, best_positions
        trial_box = box * np.array([factors[0], factors[1], 1.0])
        start = positions * np.array([factors[0], factors[1], 1.0])
        relaxed = _relax_at(start, bonds, trial_box, bond, iterations)
        score = strain_score(relaxed, bonds, trial_box, bond)
        if score < best and _embeds(relaxed, bonds, trial_box, expected):
            best, best_box, best_positions = score, trial_box, relaxed

    current = np.array([1.0, 1.0])
    for span in (0.36, 0.12, 0.04):
        for axis in (0, 1):
            for step in np.linspace(-span, span, 9):
                if abs(step) < 1e-9:
                    continue
                trial = current.copy()
                trial[axis] = current[axis] * (1.0 + step)
                before = best
                evaluate(trial)
                if best < before:
                    current = np.array([best_box[0] / box[0],
                                        best_box[1] / box[1]])

    best_positions[:, :2] = np.mod(best_positions[:, :2], best_box[:2])
    return best_positions, best_box, best


def build_haeckelite(nx: int = 4, ny: int = 4,
                     pattern: str = "r57",
                     period: int = 2,
                     density: float = 0.15,
                     bond: float = CC_BOND,
                     vacuum: float = DEFAULT_VACUUM,
                     relax_iterations: int = 3000,
                     seed: int | None = 0) -> Atoms:
    """Design a 2D carbon allotrope by patterning Stone-Wales rotations.

    Parameters
    ----------
    nx, ny
        Repeats of graphene's 4-atom rectangular cell, so the sheet holds
        ``4 * nx * ny`` atoms whatever the pattern -- a rotation moves
        bonds, never atoms.
    pattern, period, density, seed
        Which bonds to rotate; see :func:`select_bonds`.
    bond
        Target C-C length (Å) for the relaxation.
    vacuum
        Padding (Å) above and below the sheet.
    relax_iterations
        L-BFGS iterations per cycle of the variable-cell relaxation.

    Returns
    -------
    ase.Atoms
        Periodic in x and y, with ``info`` carrying the bond graph, the
        rings, the ring census, ``euler`` (which **must** be 0), the
        pattern that produced it and the measured geometry.

    Raises
    ------
    ValueError
        In three cases, and the third is the one that matters most.

        If the sheet comes out with a ring census whose ``sum(6 - n)`` is
        not zero -- impossible from the rotations themselves, each of which
        pays zero, so it means the mesh was not a triangulation.

        If the traced faces of the finished sheet disagree with the mesh's
        census, which means the geometry tore during relaxation and the
        ring list no longer describes the thing beside it.

        And if the relaxed geometry is outside the sp2 window
        (``BOND_FLOOR``-``BOND_CEILING`` Å,
        ``ANGLE_FLOOR``-``ANGLE_CEILING`` deg). This is not a relaxation
        that needs more iterations: a frustrated dense pattern converges,
        reproducibly and at every cell in the search, to 1.225-1.663 Å. Not
        every arrangement of pentagons and heptagons relaxes flat, and a
        sound census with an impossible geometry is exactly the combination
        this framework exists not to hand back.

    Warns
    -----
    If the shortest bond is under ``BOND_SOFT_FLOOR`` (1.30 Å) while
    everything else passes. A dense pattern really does compress some
    bonds, and this force field -- one rest length for every bond -- cannot
    price that, so the number is its artefact rather than a prediction.

    If the pattern applied flips but the sheet came back all hexagons,
    which means the pattern selected nothing at this cell size.

    Notes
    -----
    The ring census is **traced, not assumed**: the faces of the finished
    sheet are read off its own embedding by
    :func:`~nanocarbon_lab.analyse.rings.trace_faces`, so the reported
    pentagons and heptagons are the ones actually present rather than the
    ones the pattern intended to make.

    **A pattern is a request, not a guarantee.** ``n_flips`` and
    ``n_flips_refused`` both go into ``info``, and the gap between them is
    usually large: the rules in :func:`apply_flips` reject most candidates
    on a small cell. ``r57`` at nx=ny=6 applies 12 of the 36 edges it
    names, which is 67% non-hexagonal -- a real, dense lattice, but not the
    100% the name suggests. Read the census, not the pattern name.

    The reachable range, measured across the patterns at nx=ny=4 and 6: up
    to **67% non-hexagonal**, bonds 1.27-1.57 Å, angles 101-141 deg, zero
    close contacts, 2.62-2.68 Å^2 per atom against graphene's 2.619, and
    every census exact with ``sum(6-n) = 0``. Denser than that and the
    geometry gate refuses.
    """
    spacing = bond * math.sqrt(3.0)
    vertices, triangles, mesh_box = triangular_torus_mesh(nx, ny, spacing)

    flips = select_flips(vertices, triangles, mesh_box, nx, ny,
                         pattern=pattern, period=period,
                         density=density, seed=seed)

    # Graphene first, exactly: its dual is the geometry every rotation
    # starts from, and its bond graph is what the rotations rewire.
    flat_positions, flat_bonds, _ = fm.dual_honeycomb(
        (vertices, triangles), box=mesh_box)
    box = np.array([mesh_box[0], mesh_box[1], 0.0])

    triangles, labels, dimers, refused = apply_flips(
        vertices, triangles, mesh_box, flips, flat_bonds,
        len(flat_positions))
    applied = len(dimers)

    # The final topology, relabelled into graphene's atom indexing so the
    # rotated positions and the rewired bonds describe the same atoms.
    _, bonds_final, rings_final = fm.dual_honeycomb(
        (vertices, triangles), box=mesh_box)
    bonds = {tuple(sorted((labels[a], labels[b]))) for a, b in bonds_final}
    rings = [[labels[i] for i in ring] for ring in rings_final]

    # The ring census is known **exactly** from the mesh: a vertex of
    # degree d becomes a ring of d atoms, so nothing has to be perceived
    # and sum(6 - n) is zero identically on a torus. It is checked
    # anyway, against the finished sheet, further down.
    counts_from_mesh: dict[int, int] = {}
    for ring in rings:
        counts_from_mesh[len(ring)] = counts_from_mesh.get(len(ring), 0) + 1
    mesh_deficit = int(sum(6 - len(ring) for ring in rings))
    if mesh_deficit != 0:
        raise ValueError(
            f"The triangulation gives sum(6-n) = {mesh_deficit:+d}, but a "
            "torus owes exactly 0. An edge flip cannot change that, so the "
            "mesh was not a valid triangulation to begin with."
        )

    positions = _turn_dimers(flat_positions, dimers, bonds, box, bond)
    positions, box, score = _fit_cell(positions, bonds, box, bond,
                                      relax_iterations, counts_from_mesh)

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  pbc=(True, True, False))
    # Vacuum is the gap, added to the sheet's own thickness, which is the
    # convention every other builder here uses and what `check_vacuum`
    # measures. Setting the cell length *to* the vacuum quietly failed the
    # 10 Å guardrail as soon as a sheet had any thickness at all.
    thickness = float(positions[:, 2].max() - positions[:, 2].min())
    atoms.set_cell([box[0], box[1], thickness + vacuum])
    atoms.positions[:, 2] += (thickness + vacuum) / 2.0 - thickness / 2.0

    pair_array = np.asarray(sorted(bonds), dtype=int)
    atoms.info["bonds"] = [[int(a), int(b)] for a, b in pair_array]

    from ..analyse.rings import trace_faces

    # Traced as a *check*, not as the source. The rings come from the
    # mesh, where they are exact; if the sheet's own embedding disagrees
    # with them, the relaxation tore it and the ring list no longer
    # describes the geometry beside it.
    faces, boundary = trace_faces(atoms, pair_array, max_size=12)
    traced: dict[int, int] = {}
    for face in faces:
        traced[len(face)] = traced.get(len(face), 0) + 1
    counts = counts_from_mesh
    deficit = mesh_deficit

    if boundary or traced != counts:
        raise ValueError(
            f"The mesh says the rings are {counts}, but the relaxed sheet "
            f"traces {traced} with {boundary} boundary walk(s). An edge flip "
            "cannot change the census, so the topology is not at fault: the "
            "geometry tore during relaxation and no longer embeds the mesh. "
            "Try a lower density, a longer period or a larger cell."
        )

    atoms.info.update({
        "builder": "haeckelite",
        # Which window `sp2_quality` should judge this against. A heptagon's
        # interior angle is 128.6 deg before any strain, so the default sp2
        # window calls every sound haeckelite BROKEN.
        "quality_family": "haeckelite",
        "pattern": pattern,
        "n_flips": applied,
        "n_flips_refused": refused,
        "rings": [[int(i) for i in ring] for ring in rings],
        "ring_counts": dict(sorted(counts.items())),
        "euler": deficit,
        "genus": 1,               # a 2D periodic sheet is a torus
        "cell_a": float(box[0]),
        "cell_b": float(box[1]),
        "bond": float(bond),
        "seed": seed,
        "strain_score": round(score, 6),
    })
    from .capped_cnt import geometry_report

    geometry = geometry_report(
        positions, sorted(bonds), box=np.array([box[0], box[1], 0.0]))
    atoms.info["geometry"] = geometry

    fraction = sum(count for size, count in counts.items() if size != 6)
    atoms.info["non_hexagonal_fraction"] = round(
        fraction / max(1, len(rings)), 4)

    # The gate. A census can be perfect, the Euler budget exactly zero and
    # every contact clear while the bonds are still not carbon's -- a
    # densely and randomly patterned sheet converges, reproducibly and at
    # every cell tried, to 1.225-1.663 Å. Returning that under a
    # material's name is the failure this framework exists to avoid, so it
    # is refused with the numbers attached rather than warned about.
    if (geometry["bond_min"] < BOND_FLOOR
            or geometry["bond_max"] > BOND_CEILING
            or geometry["angle_min"] < ANGLE_FLOOR
            or geometry["angle_max"] > ANGLE_CEILING):
        raise ValueError(
            f"Pattern {pattern!r} at nx={nx}, ny={ny} gives a sound topology "
            f"-- rings {dict(sorted(counts.items()))}, sum(6-n) = {deficit:+d} "
            f"-- on a geometry that is not carbon: bonds "
            f"{geometry['bond_min']:.3f}-{geometry['bond_max']:.3f} Å and "
            f"angles {geometry['angle_min']:.1f}-{geometry['angle_max']:.1f} "
            f"deg, against the sp2 window {BOND_FLOOR}-{BOND_CEILING} Å and "
            f"{ANGLE_FLOOR}-{ANGLE_CEILING} deg. This is a converged "
            "minimum, not an unfinished relaxation: the arrangement is "
            "frustrated, and packing pentagons and heptagons this densely "
            "only relaxes cleanly for particular periodic arrangements. "
            "Lower the density, lengthen the period, or enlarge the cell. "
            f"({applied} flip(s) applied, {refused} refused.)"
        )

    if geometry["bond_min"] < BOND_SOFT_FLOOR:
        warnings.warn(
            f"Shortest bond is {geometry['bond_min']:.3f} Å, below the "
            f"{BOND_SOFT_FLOOR} Å this framework expects of sp2 carbon. The "
            "lattice is sound by every other measure -- the census is exact, "
            "there are no close contacts -- and this is the force field's "
            "own limit rather than a structural fault: it carries one rest "
            "length for every bond, and a dense pentagon-heptagon pattern "
            "really does put some bonds under compression. Re-relax with a "
            "proper calculator before quoting a geometry.",
            stacklevel=2,
        )

    if pattern != "none" and fraction == 0:
        warnings.warn(
            f"Pattern {pattern!r} flipped {applied} edge(s) but the "
            "relaxed sheet is all hexagons. At this cell size the pattern "
            "selects nothing; raise nx/ny or the density.",
            stacklevel=2,
        )
    return atoms


def describe_haeckelite(atoms: Atoms) -> str:
    """One-line summary of what the pattern actually produced."""
    info: dict[str, Any] = atoms.info
    counts = info.get("ring_counts", {})
    census = ", ".join(f"{size}:{count}" for size, count in sorted(counts.items()))
    return (
        f"{info.get('pattern', '?')} pattern, {info.get('n_flips', 0)} "
        f"flip(s): {len(atoms)} atoms, rings {census}, "
        f"sum(6-n) = {info.get('euler', 0):+d}, cell "
        f"{info.get('cell_a', 0):.2f} x {info.get('cell_b', 0):.2f} Å, "
        f"{100 * info.get('non_hexagonal_fraction', 0):.0f}% non-hexagonal."
    )


__all__ = [
    "ANGLE_CEILING",
    "ANGLE_FLOOR",
    "BOND_CEILING",
    "BOND_SOFT_FLOOR",
    "BOND_FLOOR",
    "DEFAULT_VACUUM",
    "PATTERNS",
    "apply_flips",
    "strain_score",
    "build_haeckelite",
    "describe_haeckelite",
    "flippable",
    "mesh_edges",
    "select_flips",
    "triangular_torus_mesh",
]
