"""Superstructures as polyhedral frames: octahedral nodes, prism arms.

The knee route (:mod:`.knee`) builds a node by trimming straight arms
against each other, and it is exact only when the arms balance -- the
convex-hull argument in its notes rules out every finite cage. The mesh
route builds anything and gets an amorphous wall. This module is a third
answer, and it is the Goldberg-Coxeter construction of :mod:`.goldberg`
carried onto a thickened graph:

* every **node** is a truncated octahedron -- the cell of the bcc lattice,
  whose six square faces point along <100> and eight hexagonal faces along
  <111>;
* every **arm** is a prism whose cross-section is the face it leaves by: a
  square tube along <100>, a hexagonal tube along <111>;
* faces with no arm are closed -- hexagons fanned from their centre (a
  flat patch, no defect), squares split on a diagonal (two corners left
  short: two pentagons, on the outside, where the node is convex).

Counting triangles round each coarse vertex gives the rings, and they come
out as exactly the minimum the topology asks for, all isolated:

=====================  ==========================  ================
node                   census per node             same as
=====================  ==========================  ================
6 <100> arms (P)       24 heptagons                knee P
4 <100> arms, planar   4 pentagons + 16 heptagons  knee X / super-square
4 <111> arms (D)       12 heptagons                knee tetrahedral
cube corner, 3 <100>   6 pentagons + 12 heptagons  **new: cages**
=====================  ==========================  ================

The last row is the point. A cage corner is a convex node whose arms do
not balance, which the knee route cannot build at all; here its three
closed squares carry six pentagons on the outer, positively curved side,
and its twelve heptagons sit in the necks. Every graph whose edges run
along <100> or <111> can be built this way -- cubic frames of any size,
the tesseract's projection, diamond cages -- and every one is then refined
with :func:`.goldberg.subdivide` into flat graphene patches.

Arm length is free (a whole number of prism rows), and ``rows=0`` fuses
two octahedra face to face, which gives back the Goldberg octagon maps.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND
from .goldberg import (
    Mesh,
    _Builder,
    _truncated_octahedron,
    finish_refined,
    root_three,
    subdivide,
    taubin_smooth,
)

#: The octahedron is built at a = 4: vertices are permutations of
#: (0, +-1, +-2), square faces sit 2 from the centre and hexagonal ones
#: sqrt(3). Every coarse edge is sqrt(2).
_A = 4.0
_EDGE = np.sqrt(2.0)
#: Row spacing along an arm, so the strip triangles are near-equilateral.
ROW_SPACING = _EDGE * np.sqrt(3.0) / 2.0
_FACE_OFFSET = {4: 2.0, 6: np.sqrt(3.0)}


def _direction_kind(vector: np.ndarray) -> int:
    """4 for a <100> arm, 6 for a <111> one; raises for anything else."""
    unit = vector / np.linalg.norm(vector)
    for axis in range(3):
        if abs(abs(unit[axis]) - 1.0) < 1e-6:
            return 4
    if np.allclose(np.abs(unit), 1.0 / np.sqrt(3.0), atol=1e-6):
        return 6
    raise ValueError(
        f"an arm along {np.round(unit, 3).tolist()} leaves the octahedron "
        "by no face: frames take <100> (square) and <111> (hexagonal) "
        "arms only.")


def frame_map(points: np.ndarray, edges: list[tuple[int, int, tuple]],
              box: np.ndarray | None = None, rows: int | None = None
              ) -> tuple[Mesh, int]:
    """The coarse map of a frame: octahedral nodes and prism arms.

    Parameters
    ----------
    points
        Node centres, in the octahedron's own units (``a = 4``).
    edges
        ``(i, j, shift)``: an arm from node ``i`` to node ``j`` translated
        by the integer cell vector ``shift``.
    box
        Cell edges for a periodic frame, ``0`` on an aperiodic axis, or
        ``None`` for a finite one.
    rows
        Prism rows per arm. ``None`` takes each arm's own length divided by
        :data:`ROW_SPACING`; ``0`` would fuse the faces, which needs the
        nodes to touch.

    Returns
    -------
    (mesh, budget)
        The closed triangulated surface and its ``sum(6-n) = 12(V - E)``.
    """
    points = np.asarray(points, dtype=float)
    cell = np.zeros(3) if box is None else np.asarray(box, dtype=float)
    build = _Builder(cell)
    used: dict[int, set[tuple]] = {i: set() for i in range(len(points))}

    for i, j, shift in edges:
        far = points[j] + np.asarray(shift, dtype=float) * cell
        vector = far - points[i]
        sides = _direction_kind(vector)
        unit = vector / np.linalg.norm(vector)
        length = float(np.linalg.norm(vector)) - 2.0 * _FACE_OFFSET[sides]
        if length < -1e-6:
            raise ValueError(
                f"nodes {i} and {j} are {np.linalg.norm(vector):.2f} apart, "
                "closer than two octahedra can sit.")
        for node, direction in ((i, unit), (j, -unit)):
            key = tuple(np.round(direction * (np.sqrt(3.0) if sides == 6
                                              else 1.0)).astype(int))
            if key in used[node]:
                raise ValueError(f"node {node} has two arms along {key}.")
            used[node].add(key)
        count = rows if rows is not None else int(round(length / ROW_SPACING))
        if length > 1e-6 and count < 1:
            count = 1
        if length <= 1e-6:
            continue
        # The face this arm leaves node i by, ordered round the axis.
        _, hexagons, squares = _truncated_octahedron(points[i], _A)
        faces = squares if sides == 4 else hexagons
        normal, ring = max(faces, key=lambda f: float(np.dot(f[0], unit)
                                                      / np.linalg.norm(f[0])))
        ring = [np.asarray(v, dtype=float) for v in ring]
        rings = [[v + unit * length * k / count for v in ring]
                 for k in range(count + 1)]
        ids = [[build.vertex(v) for v in r] for r in rings]
        n = len(ring)
        centre = points[i]
        for k in range(count):
            for a in range(n):
                b = (a + 1) % n
                quad = (ids[k][a], ids[k][b], ids[k + 1][b], ids[k + 1][a])
                p0 = rings[k][a]
                outward = p0 - centre - np.dot(p0 - centre, unit) * unit
                tri1 = (quad[0], quad[1], quad[2])
                tri2 = (quad[0], quad[2], quad[3])
                cross = np.cross(rings[k][b] - p0, rings[k + 1][b] - p0)
                flip = np.dot(cross, outward) < 0
                for tri in (tri1, tri2):
                    build.tris.append(tri[::-1] if flip else tri)

    for node, centre in enumerate(points):
        _, hexagons, squares = _truncated_octahedron(centre, _A)
        for normal, ring in hexagons:
            key = tuple(np.round(normal).astype(int))
            if key not in used[node]:
                build.fan([np.asarray(v, float) for v in ring], normal)
        for normal, ring in squares:
            key = tuple(np.round(normal).astype(int))
            if key not in used[node]:
                build.split([np.asarray(v, float) for v in ring], normal)

    mesh = build.mesh()
    budget = 12 * (len(points) - len(edges))
    return mesh, budget


# --------------------------------------------------------------------------
# The catalogue: each entry returns (points, edges, box, pbc) for a row
# count, the node spacing following from it.
# --------------------------------------------------------------------------
def _square_step(rows: int) -> float:
    """Centre-to-centre distance of two nodes joined by a <100> arm."""
    return 2.0 * _FACE_OFFSET[4] + rows * ROW_SPACING


def _diagonal_step(rows: int) -> float:
    """Centre-to-centre distance along <111>."""
    return 2.0 * _FACE_OFFSET[6] + rows * ROW_SPACING


def _super_cubic(rows: int):
    step = _square_step(rows)
    edges = [(0, 0, tuple(np.eye(3, dtype=int)[k])) for k in range(3)]
    return np.zeros((1, 3)), edges, np.full(3, step), (True, True, True)


def _super_square(rows: int):
    step = _square_step(rows)
    edges = [(0, 0, (1, 0, 0)), (0, 0, (0, 1, 0))]
    return np.zeros((1, 3)), edges, np.array([step, step, 0.0]), \
        (True, True, False)


def _super_diamond(rows: int):
    # Diamond sites in a cube of 8 units have <111> bonds of 2 sqrt(3);
    # scale so a bond is the diagonal step.
    scale = _diagonal_step(rows) / (2.0 * np.sqrt(3.0))
    fcc = np.array([[0, 0, 0], [0, 4, 4], [4, 0, 4], [4, 4, 0]], dtype=float)
    sites = np.vstack([fcc, fcc + 2.0])
    edges = []
    for i in range(4):
        for j in range(4, 8):
            for shift in product((-1, 0, 1), repeat=3):
                d = sites[j] + 8.0 * np.array(shift) - sites[i]
                if abs(np.linalg.norm(d) - 2.0 * np.sqrt(3.0)) < 1e-9:
                    edges.append((i, j, shift))
    return sites * scale, edges, np.full(3, 8.0 * scale), (True, True, True)


def _cube_cage(rows: int):
    half = 0.5 * _square_step(rows)
    corners = np.array(list(product((-half, half), repeat=3)))
    edges = [(i, j, (0, 0, 0)) for i in range(8) for j in range(i + 1, 8)
             if np.sum(np.abs(corners[i] - corners[j]) > 1e-9) == 1]
    return corners, edges, None, (False, False, False)


def _tesseract(rows: int):
    """The 4-cube's usual projection: a cube inside a cube, joined on <111>."""
    inner = 0.5 * _square_step(rows)
    outer = inner + _diagonal_step(rows) / np.sqrt(3.0)
    cube = np.array(list(product((-1.0, 1.0), repeat=3)))
    points = np.vstack([cube * inner, cube * outer])
    edges = []
    for offset in (0, 8):
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(cube[i] != cube[j]) == 1:
                    edges.append((offset + i, offset + j, (0, 0, 0)))
    edges += [(i, i + 8, (0, 0, 0)) for i in range(8)]
    return points, edges, None, (False, False, False)


def _adamantane_cage(rows: int):
    """Diamond's own ten-node cage: four branch nodes, six bends."""
    scale = _diagonal_step(rows) / (2.0 * np.sqrt(3.0))
    fcc = np.array([[0, 0, 0], [0, 4, 4], [4, 0, 4], [4, 4, 0],
                    [4, 4, 8], [4, 8, 4], [8, 4, 4]], dtype=float)
    branch = np.array([[2, 2, 2], [2, 6, 6], [6, 2, 6], [6, 6, 2]],
                      dtype=float)
    # Keep the six fcc sites with two branch neighbours: the cage.
    bends = [p for p in fcc
             if sum(abs(np.linalg.norm(p - b) - 2 * np.sqrt(3)) < 1e-9
                    for b in branch) == 2]
    points = np.vstack([branch, np.array(bends)])
    edges = [(i, j, (0, 0, 0)) for i in range(len(points))
             for j in range(i + 1, len(points))
             if abs(np.linalg.norm(points[i] - points[j])
                    - 2 * np.sqrt(3)) < 1e-9]
    return points * scale, edges, None, (False, False, False)


#: Frames by name: ``name -> (generator, description)``.
FRAMES: dict[str, tuple] = {
    "super-cubic": (_super_cubic, "six square tubes per node (Schwarz P)"),
    "super-square": (_super_square, "a sheet of four-way crossings"),
    "super-diamond": (_super_diamond,
                      "four hexagonal tubes per node (Schwarz D)"),
    "cube cage": (_cube_cage, "eight corners, twelve square tubes"),
    "tesseract": (_tesseract, "a cube in a cube, joined along <111>"),
    "adamantane cage": (_adamantane_cage,
                        "diamond's ten-node cage of hexagonal tubes"),
}


def frame_atom_count(kind: str, m: int = 2, rows: int = 2,
                     class_two: bool = False) -> int:
    """Atoms the build will have, exactly: coarse triangles times T."""
    points, edges, box, _ = FRAMES[kind][0](rows)
    (_, tris), _ = frame_map(points, edges, box)
    return len(tris) * (3 * m * m if class_two else m * m)


def build_frame(
    kind: str = "cube cage",
    m: int = 2,
    rows: int = 2,
    class_two: bool = False,
    bond: float = CC_BOND,
    vacuum: float = 12.0,
    relax: bool = True,
    relax_iterations: int = 3000,
    smooth: int = 20,
) -> Atoms:
    """A superstructure of octahedral nodes and prism arms, refined to graphene.

    Parameters
    ----------
    kind
        A key of :data:`FRAMES`.
    m
        Goldberg-Coxeter index: every coarse triangle becomes ``m**2``, so
        the tube circumference is ``4m`` (square arms) or ``6m``
        (hexagonal arms) rings, and the defects are ``m`` rings apart.
    rows
        Prism rows per arm, before refinement. ``0`` fuses the octahedra.
    class_two
        Refine by ``(m, m)`` instead of ``(m, 0)``: three times the atoms.
    bond, vacuum, relax, relax_iterations
        As for every builder here.
    smooth
        Taubin rounds on the refined mesh before the dual: the prisms'
        folds are 90 or 120 deg and would otherwise be bent out as bond
        strain. ``0`` keeps the flat-faced polyhedron.

    Returns
    -------
    ase.Atoms
        With the census, the budget ``12(V - E)``, the geometry and the
        placement check in ``info``.
    """
    if kind not in FRAMES:
        raise ValueError(f"unknown frame {kind!r}; built here: "
                         f"{sorted(FRAMES)}.")
    if m < 1 or rows < 0:
        raise ValueError("m must be >= 1 and rows >= 0.")
    points, edges, box, pbc = FRAMES[kind][0](rows)
    coarse, budget = frame_map(points, edges, box)
    refine_box = None if box is None else np.asarray(box, dtype=float)
    fine = subdivide(coarse, m, refine_box)
    if class_two:
        fine = root_three(fine, refine_box)
    if smooth:
        fine = taubin_smooth(fine, refine_box, iterations=smooth)
    atoms = finish_refined(fine, refine_box, pbc, budget, bond=bond,
                           relax=relax, relax_iterations=relax_iterations,
                           vacuum=vacuum, label=kind,
                           cell_scan=box is not None)
    atoms.info.update({
        "builder": "frame",
        "structure_type": "supernetwork",
        "kind": kind,
        "gc_indices": (int(m), int(m) if class_two else 0),
        "arm_rows": int(rows),
        "n_nodes": len(points),
        "n_struts": len(edges),
        "ring_budget": int(budget),
        "euler": 2 * (len(points) - len(edges)),
        "description": FRAMES[kind][1],
    })
    return atoms


def describe_frame(atoms: Atoms) -> str:
    """One line: kind, size, census and the law it meets."""
    info = atoms.info
    census = ", ".join(f"{s}: {c}" for s, c in info["ring_counts"].items())
    return (f"{info['kind']} ({info['description']}), GC{info['gc_indices']}, "
            f"{info['arm_rows']} rows per arm: {len(atoms)} atoms, rings "
            f"{{{census}}}, sum(6-n) = {info['ring_deficit']:+d} = "
            f"12(V - E) for V = {info['n_nodes']}, E = {info['n_struts']}")
