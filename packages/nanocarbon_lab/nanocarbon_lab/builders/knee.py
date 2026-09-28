"""Knee toroids: straight hexagon walls, pentagons outside, heptagons in.

:func:`~nanocarbon_lab.builders.toroid.build_toroid` meshes a torus
implicitly and lets the remesher choose the ring sizes; it returns a sound
but **amorphous** wall -- 68 pentagons and 68 heptagons at R=20, r=5, only
93% of them on the curvature side they belong on, pentagons touching
pentagons and heptagons touching heptagons.
:func:`~nanocarbon_lab.builders.toroid.build_polyhex_toroid` bends a
finished lattice and is perfectly clean -- *all* hexagons -- but it can only
stretch, so a (5,5) needs a 43 Å ring and 2200 atoms to stay under an 8%
outer-wall strain. That module's own docstring named the route that was
missing: *"they use knees with a pentagon outside and a heptagon inside,
which is a third route this module does not yet have."*

This is that route, and it is exact.

The smallest one, :data:`SOUND_SHAPE`, is 492 atoms: **12 pentagons, 222
hexagons, 12 heptagons**, ``sum(6-n) = 0``, every pentagon on the outer
equator and every heptagon on the inner one, not one of them touching
another, bonds 1.400-1.444 Å and angles 106.6-121.5 deg with no non-bonded
contact under 2 Å. The 106.6 deg is the interior angle of a pentagon, which
is the right answer rather than a strained one.

**The law.** A 5-7 pair turns a tube axis by 30 deg, and a torus asks for
exactly twelve of them. The second half of that is a theorem: the positive
curvature of *any* torus integrates to ``4 * pi``, since with
``K = cos(t) / (r (R + r cos t))`` and ``dA = r (R + r cos t) dt dphi`` the
element is just ``cos(t) dt dphi``, and a pentagon is a 60 deg disclination
carrying ``pi / 3``. Twelve pairs, 360 degrees, 30 degrees each -- and 30
deg is the angle of the published elbow, which is how the two halves check
each other.

So the pairs per knee is not free: ``knees * PAIRS_PER_KNEE`` has to be 12.
A mitred knee carries **two** pairs, because it is mirror-symmetric about
the plane of the torus and every defect it makes has a partner on the other
side of that plane -- a mitre knee is a *double* elbow. Two pairs times
**six knees** is the law exactly, and that is the default. Other counts
build and are told what they cost: eight knees spend 16 pairs on 360 deg,
which is 22.5 deg a pair, and the curvature is over-corrected by a third.

**Why a stack of circular rings cannot do any of this.** Stack rings of
``k`` vertices and stitch each band with a word of ``k`` a's and ``k`` b's.
The vertex ``b_t`` of the upper ring receives ``m + 1`` neighbours from the
lower one, where ``m`` is the number of a's consumed while the b-pointer
sits on ``t``; every upper vertex has degree 6 only if ``m = 1``
everywhere, which forces the alternating word, which leaves the *lower*
ring uniform too. Any irregularity therefore puts a defect on **both**
rings, and the pentagon and the heptagon come out as a dipole at the same
azimuth -- an axial dislocation, not a knee. Measured: a 12-vertex,
24-section torus with four irregular bands gives an exact
``{5: 4, 6: 412, 7: 4}`` census with all four pentagons on the outer
equator *and* all four heptagons there as well.

**What works is a mitre with an exact seam.** Cut a triangulated cylinder
with an oblique plane and the boundary is not a ring but a zigzag that
climbs on one side and falls on the other, so the two arms of a knee can
meet with their defects at *opposite* azimuths. Three details make it
exact, and each was found by measuring rather than by reasoning:

- **The cut boundary is slid onto the mitre plane before the seam is
  made.** The two arms are mirror images through that plane, so once their
  boundaries lie in it they coincide point for point and the join is an
  identity: no averaging, no distortion, and the arms stay equilateral to
  2.46 Å against a 2.46 Å target. Merging the raw cut boundaries instead
  leaves mesh edges from 1.9 to 5.7 Å, and no relaxation recovers from
  that -- it is what made every earlier version of this module produce
  walls at 1.2-1.6 Å.
- **The rows climb by half a step, always the same way, and the mirrored
  arm is a real reflection** -- coordinates *and* triangle winding.
  Alternating the stagger ``0, 1/2, 0, 1/2 ...``, or just flipping the sign
  of the climb, looks equivalent and is not: the band joins vertex ``j`` to
  ``j`` and ``j-1``, so the second edge then spans 1.5 steps and every
  third edge comes out at 4.19 Å.
- **The knee count is even**, because that reflection reverses the climb,
  so the arms alternate handedness -- the same alternation a Dunlap torus
  makes with zigzag and armchair segments -- and an odd ring cannot close.

**The seam offset chooses the knee**, which is the one genuinely free
choice: turning one side by a single position gives two pentagons outside
and two heptagons inside, and turning it by none gives a **square outside
and an octagon inside** (``knee="octagon"``: same 492 atoms,
``{4: 12, 6: 222, 8: 12}``, bonds 1.375-1.471 Å, angles down to 88.4 deg,
which is a square's interior angle). Both obey ``sum(6-n) = 0``. The
pentagon knee is the default because a four-membered ring is poor sp2
carbon, not because the other one is wrong.

**Not every (knees, circumference, arm_rows) triple is exact**, because the
wedge the mitre removes has to be a whole number of lattice steps. The ones
that are get found by building and counting -- :func:`clean_circumferences`
and :func:`clean_shapes` do exactly that, and leaving the knobs at ``None``
makes the builder use them. At six knees there are 54 of them between
R/r 2 and 8.

The census is **by construction**: the mesh is built, counted and checked
before a single atom exists, and
:func:`~nanocarbon_lab.builders.fullerene_mesh.relax_shell` is handed an
explicit bond graph it cannot alter. Geometry is relaxed; topology is not
negotiable.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D

#: Mesh edge length in units of the C-C bond. The honeycomb dual of an
#: equilateral triangulation of edge ``a`` has bonds of ``a / sqrt(3)``, so
#: the mesh must be built ``sqrt(3)`` times coarser than the bond it is
#: meant to produce.
MESH_EDGE = float(np.sqrt(3.0))

#: The sp2 bond band the rest of this package judges a wall by (see
#: :func:`~nanocarbon_lab.builders.capped_cnt.geometry_report`). A knee
#: toroid's census is exact by construction, but its *shape* is not
#: automatically sound, and the builder says so rather than letting the
#: caller assume it.
SP2_BOND_RANGE = (1.30, 1.55)

#: Fewest knees that still close into a ring rather than a rosette. Below
#: this the bend per knee passes 90 deg and the mitre planes cut away more
#: of each arm than they leave. The count must also be **even**: the arms
#: alternate handedness (see :func:`_ring_stack`), so an odd ring cannot
#: close on itself.
MIN_KNEES = 4

#: Pentagon-heptagon pairs one mitred knee carries. It is two and not one
#: because the knee is mirror-symmetric about the plane of the torus, so
#: every defect it makes has a partner on the other side of that plane --
#: a mitre knee is a *double* Dunlap elbow. A one-pair knee does not exist
#: in this family: searched over knee count, circumference, arm length,
#: seam offset and azimuthal phase, nothing ever gave one.
PAIRS_PER_KNEE = 2

#: Degrees of axis turn one pentagon-heptagon pair buys. It follows from
#: the other two constants -- ``360 / (CURVATURE_PENTAGONS)`` -- and it is
#: the angle of the published elbow, which is the check that the law is
#: the right one.
TURN_PER_PAIR = 30.0

#: How far one side of a seam is turned before it is joined. ``0`` gives a
#: square outside the bend and an octagon inside; one step either way gives
#: the two pentagons and two heptagons a knee is supposed to carry.
SEAM_SHIFT = -1

#: The smallest shape that is exact **and** measures sound: 492 atoms,
#: R/r 4.02, bonds 1.400-1.444 Å, angles 106.6-121.5 deg (the 106.6 being
#: the interior angle of a cap pentagon, which is the right answer rather
#: than a strained one) and no non-bonded contact under 2 Å.
SOUND_SHAPE = {"knees": 6, "circumference": 8, "arm_rows": 7}

#: Circumferences tried when ``circumference`` is left to the builder.
#: Below 8 the tube is narrower than the wedge a knee removes; above 32 the
#: structure is large enough that the caller should be choosing on purpose.
CANDIDATE_CIRCUMFERENCES = tuple(range(8, 33))

#: Arm row counts tried when ``arm_rows`` is left to the builder. Odd only
#: (see the module docstring), and short arms first so the smallest
#: structure that works is the one returned.
CANDIDATE_ROWS = (5, 7, 9, 11, 13, 15)

#: Pentagons an ideal torus wants on its outer equator, and heptagons on its
#: inner one. On **any** torus the positive curvature integrates to exactly
#: ``4 * pi``: with ``K = cos(t) / (r * (R + r cos t))`` and
#: ``dA = r (R + r cos t) dt dphi`` the element is just ``cos(t) dt dphi``,
#: so the outer half carries ``2 * pi * 2 = 4 * pi`` whatever the radii are.
#: One pentagon is a 60 deg disclination and carries ``pi / 3``, so the
#: curvature-exact count is ``4 * pi / (pi / 3) = 12`` -- and because a knee
#: built this way carries two, **six knees** is the curvature-neutral toroid.
#: More knees over-correct (eight give 16 of each) and fewer cannot close.
CURVATURE_PENTAGONS = 12

#: ``R / r`` aimed for when the builder is left to choose the shape. The
#: published toroidal carbons sit between 3 and 6 (see
#: :data:`~nanocarbon_lab.builders.toroid.LITERATURE_ASPECT`); the middle of
#: that band is what "choose for me" should mean, because the smallest shape
#: that closes cleanly is often a fat torus around a nearly shut hole -- at
#: 6 knees the shortest arms give ``R/r`` 2.56, below the aspect the other
#: toroid builder refuses outright.
PREFERRED_ASPECT = 4.5


def _ring_stack(circumference: int, rows: int, radius: float, spacing: float,
                mirror: bool = False) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """A triangulated cylinder: every interior vertex degree 6, exactly.

    The rows climb by half a step, always the same way, and the band joins
    vertex ``j`` to ``j`` and ``j-1`` to match. ``mirror`` reflects the
    whole thing -- coordinates **and** triangle winding -- rather than
    flipping the sign of the climb on its own, which would leave the band
    joining the wrong pair and put a 1.5-step edge in every row. The
    mirrored stack is what a mitre needs on the far side of a knee, so the
    arms alternate handedness, which is why the knee count must be even.
    """
    verts: list[list[float]] = []
    idx: dict[tuple[int, int], int] = {}
    tris: list[tuple[int, int, int]] = []
    for i in range(rows):
        for j in range(circumference):
            # Half a step per row, always the SAME way. Alternating the
            # stagger (0, 1/2, 0, 1/2 ...) looks equivalent and is not: the
            # band always joins vertex j to j and j-1, so on the rows where
            # the stagger runs the other way that second edge spans 1.5
            # steps. Measured on a plain k=10 stack: every third edge came
            # out at 4.194 A against a 2.460 A target, and the dual of that
            # mesh cannot be relaxed into sp2 geometry no matter how round
            # the surface is.
            angle = 2.0 * np.pi * (j + 0.5 * i) / circumference
            idx[(i, j)] = len(verts)
            verts.append([radius * np.cos(angle), radius * np.sin(angle),
                          (i - (rows - 1) / 2.0) * spacing])
    k = circumference
    for i in range(rows - 1):
        ai = bi = 0
        for char in ["a", "b"] * k:
            a0, a1 = idx[(i, ai % k)], idx[(i, (ai + 1) % k)]
            b0, b1 = idx[(i + 1, bi % k)], idx[(i + 1, (bi + 1) % k)]
            # Wound so every directed edge appears exactly once, which is
            # what `dual_honeycomb`'s ring walk needs: checked on a closed
            # stack, this winding gives 0 repeated directed edges and the
            # other one 720.
            if char == "a":
                tris.append((a0, a1, b0))
                ai += 1
            else:
                tris.append((a0, b1, b0))
                bi += 1
    points = np.asarray(verts, dtype=float)
    if mirror:
        points[:, 1] *= -1.0
        tris = [(c, b, a) for a, b, c in tris]
    return points, tris


def _frame(axis: np.ndarray, outward: np.ndarray) -> np.ndarray:
    """Right-handed frame with ``axis`` as z and ``outward`` projected to x."""
    e3 = axis / np.linalg.norm(axis)
    e1 = outward - e3 * float(np.dot(outward, e3))
    e1 /= np.linalg.norm(e1)
    return np.column_stack([e1, np.cross(e3, e1), e3])


def _boundary_cycles(tris) -> list[list[int]] | None:
    """Ordered boundary cycles, or ``None`` if the boundary is not manifold."""
    count: dict[tuple[int, int], int] = defaultdict(int)
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            count[(min(a, b), max(a, b))] += 1
    rims = [e for e, c in count.items() if c == 1]
    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in rims:
        neighbours[a].append(b)
        neighbours[b].append(a)
    if any(len(v) != 2 for v in neighbours.values()):
        return None
    seen: set[int] = set()
    cycles: list[list[int]] = []
    for start in neighbours:
        if start in seen:
            continue
        cycle, previous, current = [start], None, start
        seen.add(start)
        while True:
            onward = [x for x in neighbours[current] if x != previous]
            if not onward or onward[0] == start:
                break
            previous, current = current, onward[0]
            cycle.append(current)
            seen.add(current)
        cycles.append(cycle)
    return cycles


def _reflect(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Mirror ``vector`` in the plane with this unit ``normal``."""
    return vector - 2.0 * float(vector @ normal) * normal


def dual_open(vertices: np.ndarray, tris) -> tuple[np.ndarray, set, list, list[int]]:
    """The honeycomb dual of a mesh that is allowed to have rims.

    :func:`~nanocarbon_lab.builders.fullerene_mesh.dual_honeycomb` derives
    its bonds from the rings and its rings by walking each vertex's faces
    in a cycle, so a boundary vertex -- whose faces do not close a cycle
    -- makes it raise. A finite coil has two rims by definition, exactly
    as a nanocone has one, so it needs a dual that keeps them.

    Bonds come from **face adjacency** instead: two triangles sharing an
    edge are bonded, which is the same relation the ring walk encodes and
    is defined with or without a boundary. Rings are returned for the
    interior vertices only, and the rim vertices are reported separately
    -- their rings are genuinely incomplete, not missing.

    Returns ``(positions, bonds, rings, rim_atoms)``.
    """
    faces = np.asarray(tris, dtype=int)
    positions = vertices[faces].mean(axis=1)

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, t in enumerate(tris):
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_faces[(min(a, b), max(a, b))].append(index)
    bonds: set[tuple[int, int]] = set()
    for sharing in edge_faces.values():
        if len(sharing) == 2:
            a, b = sharing
            bonds.add((a, b) if a < b else (b, a))

    boundary = {x for e, sharing in edge_faces.items() if len(sharing) != 2
                for x in e}
    vertex_faces: dict[int, list[int]] = defaultdict(list)
    for index, t in enumerate(tris):
        for x in t:
            vertex_faces[int(x)].append(index)

    rings: list[list[int]] = []
    for x, incident in vertex_faces.items():
        if x in boundary:
            continue
        members = {f: set(int(v) for v in tris[f]) for f in incident}
        around: dict[int, list[int]] = defaultdict(list)
        for a in incident:
            for b in incident:
                if a != b and len(members[a] & members[b]) == 2:
                    around[a].append(b)
        if any(len(v) != 2 for v in around.values()):
            continue
        start = incident[0]
        ring, previous, current = [start], None, start
        while True:
            onward = [y for y in around[current] if y != previous]
            if not onward or onward[0] == start:
                break
            previous, current = current, onward[0]
            ring.append(current)
        if len(ring) == len(incident):
            rings.append(ring)

    rim = sorted({index for index, t in enumerate(tris)
                  if any(int(x) in boundary for x in t)})
    two_coordinate = sorted(
        index for index in rim
        if sum(1 for b in bonds if index in b) < 3)
    return positions, bonds, rings, two_coordinate


def knee_path_mesh(
    points: np.ndarray,
    circumference: int,
    tube_radius: float,
    spacing: float,
    seam_shift: int = SEAM_SHIFT,
    closed: bool = True,
    references: np.ndarray | None = None,
    period: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int]], list[int]]:
    """Mitre a tube along **any** equal-step polyline: ring, helix or arc.

    ``points`` are the corners, evenly spaced; an arm runs between each
    consecutive pair and a knee sits at every interior corner. A closed
    polygon gives a toroid, a helix gives a coil, an open arc gives a
    bent tube with two free rims.

    **The reflection works for any equal-step path**, which is what makes
    one routine cover all of them. With ``u`` and ``v`` the two unit
    directions at a corner and ``n`` proportional to ``u + v``, the mitre
    plane's reflection sends ``u`` to ``-v`` -- expand it and the
    ``(1 + u.v)`` cancels -- so it sends the previous corner to the next
    one whenever the steps are equal. The two arms of a knee are then
    exact mirror images whatever the path is doing.

    That also fixes the azimuthal reference, and it has to: an arbitrary
    choice per arm would leave the seams misaligned on anything that is
    not a regular polygon. ``references[q]`` is carried from arm to arm
    **through the same reflection**, so the mirror relation the seam
    relies on holds by construction rather than by symmetry. It is the
    mitre's analogue of the rotation-minimising frame
    :mod:`~nanocarbon_lab.builders.centerline` sweeps with, and it is
    needed for the same reason -- a frame chosen fresh at each step
    accumulates a twist the structure did not ask for.

    Returns ``(vertices, triangles, owner)``, ``owner[i]`` naming the arm
    each vertex came from.
    """
    points = np.asarray(points, dtype=float)
    n_points = len(points)
    if period is not None:
        # A screw-periodic path -- a coil. The corner after the last one is
        # the first, moved along by one period, so the wrap is a real mitre
        # knee rather than two rims welded afterwards. Welding them instead
        # left the two boundaries 1.8-3.8 A apart, a full lattice step, which
        # is a seam the relaxation then has to carry.
        closed = True
        period = np.asarray(period, dtype=float)

    def corner(q: int) -> np.ndarray:
        if period is None:
            return points[q % n_points]
        return points[q % n_points] + (q // n_points) * period

    n_arms = n_points if closed else n_points - 1
    if n_arms < 2:
        raise ValueError("a mitred path needs at least two arms.")
    if closed and n_arms % 2:
        raise ValueError(
            f"a closed path needs an even number of arms ({n_arms} given). "
            "The reflection through a mitre plane reverses the half-step the "
            "rows climb by, so the arms alternate handedness and an odd ring "
            "cannot close on itself."
        )

    steps = np.array([corner(q + 1) - corner(q) for q in range(n_arms)])
    lengths = np.linalg.norm(steps, axis=1)
    if float(lengths.max() - lengths.min()) > 1e-6 * float(lengths.mean()):
        raise ValueError(
            f"the path's steps are not equal ({lengths.min():.3f} to "
            f"{lengths.max():.3f} A). The mitre reflection sends one corner "
            "to the next only when they are, so an uneven path would leave "
            "the two sides of a knee out of register."
        )
    axes = steps / lengths[:, None]

    knee_at = (list(range(n_points)) if closed
               else list(range(1, n_points - 1)))
    normals: dict[int, np.ndarray] = {}
    for q in knee_at:
        incoming = axes[(q - 1) % n_arms]
        outgoing = axes[q % n_arms]
        normal = incoming + outgoing
        size = float(np.linalg.norm(normal))
        if size < 1e-9:
            raise ValueError(
                f"corner {q} doubles the path back on itself, so the mitre "
                "plane is undefined."
            )
        normals[q] = normal / size

    if references is not None:
        refs = np.asarray(references, dtype=float)
    else:
        seed = np.array([0.0, 0.0, 1.0])
        if abs(float(seed @ axes[0])) > 0.9:
            seed = np.array([1.0, 0.0, 0.0])
        seed = seed - axes[0] * float(seed @ axes[0])
        seed /= np.linalg.norm(seed)
        carried = [seed]
        for q in range(1, n_arms):
            carried.append(_reflect(carried[-1], normals[q]))
        refs = np.asarray(carried)

    # The stack has to reach past the mitre on the long side of every arm,
    # so the plane and not the stack's own last row is what ends it: the
    # outer side of an arm runs `step + 2 r tan(turn/2)`. The built count is
    # ODD, because the reflection sends row i to row rows-1-i and the rows
    # climb by half a step -- an even count lands the seam between rows on
    # one side and on a row on the other.
    turns = [float(np.arccos(np.clip(axes[(q - 1) % n_arms] @ axes[q % n_arms],
                                     -1.0, 1.0))) for q in knee_at] or [0.0]
    overhang = tube_radius * np.tan(max(turns) / 2.0)
    rows_built = int(np.ceil((float(lengths.mean()) + 2.0 * overhang) / spacing)) + 3
    rows_built += 1 - rows_built % 2

    verts: list[list[float]] = []
    tris: list[tuple[int, int, int]] = []
    owner: list[int] = []
    for q in range(n_arms):
        middle = 0.5 * (corner(q) + corner(q + 1))
        local, local_tris = _ring_stack(circumference, rows_built,
                                        tube_radius, spacing,
                                        mirror=bool(q % 2))
        placed = middle + local @ _frame(axes[q], refs[q]).T
        keep = np.ones(len(placed), dtype=bool)
        if q in normals:
            keep &= (placed - corner(q)) @ normals[q] >= 0.0
        elif not closed:
            keep &= (placed - corner(q)) @ axes[q] >= -0.5 * spacing
        end = (q + 1) % n_points
        if end in normals:
            keep &= (placed - corner(q + 1)) @ normals[end] <= 0.0
        elif not closed:
            keep &= (placed - corner(q + 1)) @ axes[q] <= 0.5 * spacing
        offset = len(verts)
        verts.extend(placed.tolist())
        owner.extend([q] * len(placed))
        tris.extend((a + offset, b + offset, c + offset)
                    for a, b, c in local_tris
                    if keep[a] and keep[b] and keep[c])

    vertices = np.asarray(verts, dtype=float)
    used = sorted({x for t in tris for x in t})
    remap = {x: i for i, x in enumerate(used)}
    vertices = vertices[used]
    owner = [owner[x] for x in used]
    tris = [(remap[a], remap[b], remap[c]) for a, b, c in tris]

    cycles = _boundary_cycles(tris)
    expected = 2 * len(knee_at) + (0 if closed else 2)
    if cycles is None or len(cycles) != expected:
        got = "non-manifold" if cycles is None else f"{len(cycles)}"
        raise ValueError(
            f"The mitred arms left {got} boundary cycles where {expected} "
            "were expected (two per knee, plus the free rims of an open "
            "path). The arms are too short for the bend: lengthen the step "
            "or turn less at each corner."
        )
    sites = np.array([corner(q) for q in range(n_arms + 1)])
    by_corner: dict[int, list[list[int]]] = {}
    for cycle in cycles:
        centre = vertices[cycle].mean(axis=0)
        q = int(np.argmin(np.linalg.norm(sites - centre, axis=1))) % n_points
        if q in normals:
            by_corner.setdefault(q, []).append(cycle)
    if (len(by_corner) != len(knee_at)
            or any(len(v) != 2 for v in by_corner.values())):
        raise ValueError(
            "The boundary cycles did not fall two to a knee, so the arms do "
            "not meet where the path says they should. Lengthen the step."
        )

    for q, pair in by_corner.items():
        for cycle in pair:
            for x in cycle:
                axis = axes[owner[x]]
                # The wrap knee's plane is at the corner nearest that arm,
                # which for the last arm is one period along.
                here = min((corner(q), corner(q + n_points)),
                           key=lambda c, x=x: float(
                               np.linalg.norm(vertices[x] - c)))
                step = float((here - vertices[x]) @ normals[q]
                             / (axis @ normals[q]))
                vertices[x] = vertices[x] + step * axis

    merge: dict[int, int] = {}
    for q, (first, second) in by_corner.items():
        incoming = (q - 1) % n_arms
        lower, upper = ((list(first), list(second))
                        if owner[first[0]] == incoming
                        else (list(second), list(first)))
        if len(lower) != len(upper):
            raise ValueError(
                f"Knee {q} has {len(lower)} vertices on one side of the mitre "
                f"and {len(upper)} on the other, so the seam cannot pair up."
            )
        # At the wrap knee of a screw-periodic path the incoming arm sits
        # one period along, so the two sides are compared -- and merged --
        # modulo that period. The seam is then shared rather than welded,
        # and everything downstream reads it by minimum image.
        offset = np.zeros(3) if (period is None or q != 0) else period
        gap = (vertices[lower] + offset)[:, None, :] - vertices[upper][None, :, :]
        distances = np.linalg.norm(gap, axis=2)
        nearest = distances.argmin(axis=1)
        if len(set(nearest.tolist())) != len(lower):
            raise ValueError(
                f"Knee {q}: the two boundaries did not land on each other, so "
                "the seam has no one-to-one partner. Try another step length."
            )
        ordered = [upper[int(x)] for x in nearest]
        n = len(ordered)
        for t, low in enumerate(lower):
            high = ordered[(t + seam_shift) % n]
            keep_at, drop_at = min(low, high), max(low, high)
            merge[drop_at] = keep_at
            middle_point = 0.5 * (vertices[low] + offset + vertices[high])
            vertices[keep_at] = (middle_point - offset if keep_at == low
                                 else middle_point)

    def root(x: int) -> int:
        while x in merge:
            x = merge[x]
        return x

    tris = [tuple(root(x) for x in t) for t in tris]
    tris = [t for t in tris if len(set(t)) == 3]
    live = sorted({x for t in tris for x in t})
    relabel = {x: i for i, x in enumerate(live)}
    owner = [owner[x] for x in live]
    return (vertices[live],
            [(relabel[a], relabel[b], relabel[c]) for a, b, c in tris],
            owner)


def knee_polygon_mesh(
    knees: int,
    circumference: int,
    arm_rows: int,
    tube_radius: float,
    spacing: float,
    seam_shift: int = SEAM_SHIFT,
) -> tuple[np.ndarray, list[tuple[int, int, int]], float]:
    """Triangulate a closed regular polygon of mitred arms.

    Returns ``(vertices, triangles, centreline_radius)``. The arms are
    exactly equilateral away from the knees (2.46 Å against a 2.46 Å
    target), because the cut boundary is **slid onto the mitre plane**
    before the seam is made: the two arms of a knee are mirror images
    through that plane, so once their boundaries lie in it they coincide
    point for point and the join is an identity rather than an average.

    ``seam_shift`` then turns one side by that many positions along the
    seam, and it is what chooses the knee:

    ===============  ==========================================
    ``seam_shift``   what the knee carries
    ===============  ==========================================
    ``0``            a square outside and an octagon inside
    ``-1`` / ``1``   two pentagons outside, two heptagons inside
    ===============  ==========================================

    Both obey ``sum(6-n) = 0``; the square is a poor ring for sp2 carbon,
    so the builder uses the pentagon knee and keeps the other reachable.

    Raises
    ------
    ValueError
        If ``arm_rows`` is even, or if the boundary cycles do not pair up
        two per knee -- which means the mitre planes cut the arms apart
        rather than trimming them.
    """
    if arm_rows % 2 == 0:
        raise ValueError(
            f"arm_rows must be odd ({arm_rows} is even). The reflection "
            "through a mitre plane sends row i to row rows-1-i, and the rows "
            "climb by half a step, so an even count lands the seam between "
            "rows on one side and on a row on the other."
        )
    arm_length = arm_rows * spacing
    centre_radius = arm_length / (2.0 * np.tan(np.pi / knees))
    # Rows of slack at each end, so the mitre planes are what trim the arm.
    # The outer side of an arm is `arm_length + 2 r tan(pi/knees)` long and
    # the inner one that much shorter -- exactly the (R+r)/(R-r) ratio a
    # torus needs -- so a stack cut to the nominal length is truncated
    # outside and the outer equator comes back too sparse.
    pad = int(np.ceil(tube_radius * np.tan(np.pi / knees) / spacing)) + 1
    corners = np.array([
        [centre_radius * np.cos(2.0 * np.pi * q / knees),
         centre_radius * np.sin(2.0 * np.pi * q / knees), 0.0]
        for q in range(knees)
    ])
    axes = np.array([corners[(q + 1) % knees] - corners[q] for q in range(knees)])
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    # The mitre plane at corner q bisects the two travel directions, so its
    # normal is their sum -- the same plane a mitred pipe elbow is cut on.
    normals = np.array([axes[q - 1] + axes[q] for q in range(knees)])
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    verts: list[list[float]] = []
    tris: list[tuple[int, int, int]] = []
    owner: list[int] = []
    for q in range(knees):
        middle = 0.5 * (corners[q] + corners[(q + 1) % knees])
        outward = np.array([middle[0], middle[1], 0.0])
        outward /= np.linalg.norm(outward)
        local, local_tris = _ring_stack(circumference, arm_rows + 2 * pad,
                                        tube_radius, spacing,
                                        mirror=bool(q % 2))
        placed = middle + local @ _frame(axes[q], outward).T
        lo_p, lo_n = corners[q], normals[q]
        hi_p, hi_n = corners[(q + 1) % knees], normals[(q + 1) % knees]
        keep = (((placed - lo_p) @ lo_n >= 0.0)
                & ((placed - hi_p) @ hi_n <= 0.0))
        offset = len(verts)
        verts.extend(placed.tolist())
        owner.extend([q] * len(placed))
        tris.extend((a + offset, b + offset, c + offset)
                    for a, b, c in local_tris
                    if keep[a] and keep[b] and keep[c])

    vertices = np.asarray(verts, dtype=float)
    used = sorted({x for t in tris for x in t})
    remap = {x: i for i, x in enumerate(used)}
    vertices = vertices[used]
    owner = [owner[x] for x in used]
    tris = [(remap[a], remap[b], remap[c]) for a, b, c in tris]

    cycles = _boundary_cycles(tris)
    if cycles is None or len(cycles) != 2 * knees:
        got = "non-manifold" if cycles is None else f"{len(cycles)}"
        raise ValueError(
            f"The mitred arms left {got} boundary cycles where {2 * knees} "
            "were expected (two per knee, one from each arm). The arms are "
            "too short for the bend: raise arm_rows or knees."
        )
    by_corner: dict[int, list[list[int]]] = {}
    for cycle in cycles:
        centre = vertices[cycle].mean(axis=0)
        q = int(np.argmin(np.linalg.norm(corners - centre, axis=1)))
        by_corner.setdefault(q, []).append(cycle)
    if len(by_corner) != knees or any(len(v) != 2 for v in by_corner.values()):
        raise ValueError(
            "The boundary cycles did not fall two to a knee, so the arms do "
            "not meet where the polygon says they should. Raise arm_rows."
        )

    # Slide every boundary vertex along its own arm's axis onto the mitre
    # plane. This is what makes the join exact.
    for q, pair in by_corner.items():
        for cycle in pair:
            for x in cycle:
                axis = axes[owner[x]]
                step = float((corners[q] - vertices[x]) @ normals[q]
                             / (axis @ normals[q]))
                vertices[x] = vertices[x] + step * axis

    merge: dict[int, int] = {}
    for q, (first, second) in by_corner.items():
        incoming = (q - 1) % knees
        lower, upper = ((list(first), list(second))
                        if owner[first[0]] == incoming
                        else (list(second), list(first)))
        if len(lower) != len(upper):
            raise ValueError(
                f"Knee {q} has {len(lower)} vertices on one side of the mitre "
                f"and {len(upper)} on the other, so the seam cannot pair up."
            )
        # After the slide the two boundaries are the same points, so the
        # mirror partner is simply the nearest one -- no search, no
        # ambiguity, and the pairing is exact before the shift is applied.
        # After the slide the two boundaries are the same points, so the
        # mirror partner is simply the nearest one -- no search, no
        # ambiguity, and the pairing is exact before the shift is applied.
        distances = np.linalg.norm(
            vertices[lower][:, None, :] - vertices[upper][None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        if len(set(nearest.tolist())) != len(lower):
            raise ValueError(
                f"Knee {q}: the two boundaries did not land on each other, so "
                "the seam has no one-to-one partner. Try another arm length."
            )
        ordered = [upper[int(x)] for x in nearest]
        n = len(ordered)
        for t, low in enumerate(lower):
            high = ordered[(t + seam_shift) % n]
            keep_at, drop_at = min(low, high), max(low, high)
            merge[drop_at] = keep_at
            vertices[keep_at] = 0.5 * (vertices[low] + vertices[high])

    def root(x: int) -> int:
        while x in merge:
            x = merge[x]
        return x

    tris = [tuple(root(x) for x in t) for t in tris]
    tris = [t for t in tris if len(set(t)) == 3]
    vertices, tris = _compact(vertices, tris)
    return vertices, tris, float(centre_radius)


def _compact(vertices: np.ndarray, tris):
    used = sorted({x for t in tris for x in t})
    remap = {x: i for i, x in enumerate(used)}
    return vertices[used], [(remap[a], remap[b], remap[c]) for a, b, c in tris]


def mesh_census(vertices: np.ndarray, tris) -> tuple[dict[int, int], dict[int, int], int]:
    """Vertex degrees of a mesh -- which *are* the ring sizes of its dual.

    Returns ``(histogram, degree_by_vertex, n_non_manifold_edges)``. A closed
    mesh has no non-manifold edges; anything else means the surface did not
    close and the dual cannot be taken.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)
    count: dict[tuple[int, int], int] = defaultdict(int)
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            adjacency[a].add(b)
            adjacency[b].add(a)
            count[(min(a, b), max(a, b))] += 1
    broken = sum(1 for c in count.values() if c != 2)
    degrees = {x: len(adjacency[x]) for x in range(len(vertices))}
    histogram: dict[int, int] = {}
    for d in degrees.values():
        histogram[d] = histogram.get(d, 0) + 1
    return histogram, degrees, broken


def equator_rows(vertices: np.ndarray, centre_radius: float,
                  tube_radius: float) -> tuple[int, int]:
    """Vertices within 30 deg of the outer equator, and of the inner one.

    The number that says whether the disclinations are carrying the
    *metric* or only the curvature. A torus's outer parallel is
    ``(R+r)/(R-r)`` times longer than its inner one, so a net with uniform
    bonds needs that many times more rows out there.
    """
    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    angle = np.arctan2(vertices[:, 2], radius - centre_radius)
    outer = int((np.abs(angle) < np.pi / 6.0).sum())
    inner = int((np.abs(np.abs(angle) - np.pi) < np.pi / 6.0).sum())
    return outer, inner


def defect_contacts(vertices: np.ndarray, tris,
                    skip_rim: bool = False) -> dict[str, int]:
    """Edges between two non-hexagons, split by whether the signs agree.

    ``like`` is what a lobe of pentagons or a fused heptagon pair IS, and
    ``fused`` is a pentagon sharing an edge with a heptagon -- the axial
    dislocation a stack of circular rings cannot avoid. A clean wall has
    zero of both: every disclination surrounded entirely by hexagons.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)
    count: dict[tuple[int, int], int] = defaultdict(int)
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            adjacency[a].add(b)
            adjacency[b].add(a)
            count[(min(a, b), max(a, b))] += 1
    degrees = {x: len(adjacency[x]) for x in adjacency}
    # A rim vertex has a low degree because its ring is cut off, not
    # because it is a disclination, so on an open mesh it must not be
    # counted as one -- otherwise a coil's two rims read as 16 lobes.
    rim = ({x for e, c in count.items() if c != 2 for x in e}
           if skip_rim else set())
    like = fused = 0
    for x, neighbours in adjacency.items():
        if degrees[x] == 6 or x in rim:
            continue
        for y in neighbours:
            if y <= x or degrees[y] == 6 or y in rim:
                continue
            same = (degrees[x] > 6) == (degrees[y] > 6)
            like += int(same)
            fused += int(not same)
    return {"like": like, "fused": fused}


def _verdict(vertices, tris, knees, centre_radius):
    """``(ok, census, placed_fraction)`` for a candidate mesh."""
    census, degrees, broken = mesh_census(vertices, tris)
    if broken:
        return False, census, 0.0
    if set(census) - {5, 6, 7}:
        return False, census, 0.0
    if census.get(5, 0) != 2 * knees or census.get(7, 0) != 2 * knees:
        return False, census, 0.0
    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    fives = [x for x, d in degrees.items() if d == 5]
    sevens = [x for x, d in degrees.items() if d == 7]
    right = (sum(1 for x in fives if radius[x] > centre_radius)
             + sum(1 for x in sevens if radius[x] < centre_radius))
    return True, census, right / float(len(fives) + len(sevens))


def shape_radii(knees: int, arm_rows: int, circumference: int,
                bond: float = CC_BOND) -> tuple[float, float]:
    """``(major_radius, minor_radius)`` a shape will have, without building it."""
    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    minor = MESH_EDGE * bond / (2.0 * np.sin(np.pi / circumference))
    major = arm_rows * spacing / (2.0 * np.tan(np.pi / knees))
    return float(major), float(minor)


def clean_shapes(
    knees: int = 8,
    bond: float = CC_BOND,
    rows_candidates=CANDIDATE_ROWS,
    circ_candidates=CANDIDATE_CIRCUMFERENCES,
) -> list[tuple[int, int]]:
    """``(arm_rows, circumference)`` pairs that give an exact census.

    Shortest arms and narrowest tube first, so the first entry is the
    smallest structure that works. Both knobs matter: at 12 knees no
    circumference closes cleanly with 7-row arms below 27, but 13-row arms
    bring it down to 19.
    """
    out: list[tuple[int, int]] = []
    for rows in rows_candidates:
        if rows % 2 == 0:
            continue
        for k in clean_circumferences(knees, rows, bond, circ_candidates):
            out.append((int(rows), int(k)))
    return out


def clean_circumferences(
    knees: int = 8,
    arm_rows: int = 7,
    bond: float = CC_BOND,
    candidates=CANDIDATE_CIRCUMFERENCES,
) -> list[int]:
    """Circumferences that give an exact census for this many knees.

    The mitre has to remove a whole number of lattice steps, so only some
    ``(knees, circumference)`` pairs close cleanly. Rather than deriving the
    condition, this builds each candidate and counts: a circumference is
    returned only if the mesh is closed, its census is exactly
    ``{5: 2*knees, 7: 2*knees}`` with every other vertex degree 6, **and**
    every pentagon sits outside the centreline with every heptagon inside.
    """
    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    good: list[int] = []
    for k in candidates:
        radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / k))
        try:
            vertices, tris, centre_radius = knee_polygon_mesh(
                knees, k, arm_rows, radius, spacing)
        except ValueError:
            continue
        ok, _, placed = _verdict(vertices, tris, knees, centre_radius)
        if ok and placed == 1.0:
            good.append(int(k))
    return good


def build_knee_toroid(
    knees: int = 6,
    circumference: int | None = None,
    arm_rows: int | None = None,
    knee: str = "pentagon",
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_1D,
    relax: bool = True,
    relax_iterations: int = 400,
) -> Atoms:
    """A toroid of straight arms and mitred knees: 5s outside, 7s inside.

    Parameters
    ----------
    knees
        Number of knees, so the bend per knee is ``360 / knees`` degrees.
        Sets how round the torus is: the arms between knees are straight
        all-hexagon tube. Each knee carries two pentagons and two heptagons,
        so **six** is the curvature-exact count -- see
        :data:`CURVATURE_PENTAGONS`.
    circumference
        Mesh vertices around the tube, which sets the tube radius. ``None``
        asks :func:`clean_circumferences` for the smallest one that gives an
        exact census at this ``knees`` and ``arm_rows``.
    arm_rows
        Rows of mesh per arm; must be **odd** (see the module docstring).
        Sets the arm length and therefore the major radius. ``None`` asks
        :func:`clean_shapes` for the smallest pair that works.
    bond, vacuum
        C-C length and the padding around the finished molecule (Å).
    relax, relax_iterations
        Whether to relax the geometry with the valence force field, and for
        how long. The bond graph is passed in explicitly, so relaxation can
        only move atoms -- the ring census is fixed before this runs.

    Returns
    -------
    ase.Atoms
        A finite molecule with the census, the two radii, the bend per knee
        and the disclination placement in ``atoms.info``.

    Raises
    ------
    ValueError
        If ``knees`` is below :data:`MIN_KNEES`, if ``arm_rows`` is even, if
        no circumference gives an exact census at this ``knees``, or if the
        requested one does not -- in which case the message names the ones
        that do.

    Examples
    --------
    >>> atoms = build_knee_toroid(knees=8, circumference=12, relax=False)
    >>> atoms.info["ring_counts"][5], atoms.info["ring_counts"][7]
    (16, 16)
    >>> atoms.info["disclinations_placed"]
    1.0
    """
    if knees < MIN_KNEES:
        raise ValueError(
            f"knees must be at least {MIN_KNEES} ({knees} given): below that "
            "the bend per knee passes 90 deg and the mitre planes cut away "
            "more of each arm than they leave."
        )
    if knee not in ("pentagon", "octagon"):
        raise ValueError(
            f"knee must be 'pentagon' or 'octagon', not {knee!r}. The seam "
            "either turns by one position, which puts two pentagons on the "
            "outer elbow and two heptagons on the inner one, or by none, "
            "which puts a square outside and an octagon inside."
        )
    if knees % 2:
        raise ValueError(
            f"knees must be even ({knees} given). The reflection through a "
            "mitre plane reverses the half-step the rows climb by, so the "
            "arms alternate handedness and an odd ring cannot close on "
            "itself -- the same alternation a Dunlap torus makes with "
            "zigzag and armchair segments."
        )
    if arm_rows is not None and arm_rows % 2 == 0:
        raise ValueError(
            f"arm_rows must be odd ({arm_rows} given) -- see the module "
            "docstring: an even count pairs same-parity rows across the seam "
            "and the census comes back with octagons."
        )

    if arm_rows is None:
        shapes = clean_shapes(knees, bond)
        if circumference is not None:
            shapes = [(r, k) for r, k in shapes if k == circumference]
        if not shapes:
            raise ValueError(
                f"No odd arm length in {CANDIDATE_ROWS} gives an exact census "
                f"at {knees} knees"
                + (f" and circumference {circumference}" if circumference
                   else "")
                + ". The wedge a knee removes has to be a whole number of "
                "lattice steps, and at this bend none of them is: try a "
                "different number of knees."
            )
        # Nearest to the middle of the published aspect band, not simply
        # the smallest: the smallest shape that closes is often a fat torus
        # around a nearly shut hole.
        def distance(shape: tuple[int, int]) -> tuple[float, int, int]:
            major, minor = shape_radii(knees, shape[0], shape[1], bond)
            return (abs(major / minor - PREFERRED_ASPECT), shape[0], shape[1])

        arm_rows, circumference = min(shapes, key=distance)

    if circumference is None:
        options = clean_circumferences(knees, arm_rows, bond)
        if not options:
            raise ValueError(
                f"No circumference in {CANDIDATE_CIRCUMFERENCES[0]}-"
                f"{CANDIDATE_CIRCUMFERENCES[-1]} gives an exact census at "
                f"{knees} knees. The wedge a knee removes has to be a whole "
                "number of lattice steps, and at this bend none of them is: "
                "try a different number of knees."
            )
        circumference = options[0]

    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    tube_radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / circumference))
    shift = SEAM_SHIFT if knee == "pentagon" else 0
    vertices, tris, centre_radius = knee_polygon_mesh(
        knees, circumference, arm_rows, tube_radius, spacing, seam_shift=shift)
    ok, census, placed = _verdict(vertices, tris, knees, centre_radius)
    if knee == "octagon":
        ok = True                       # a square/octagon knee is its own census
    if not ok:
        options = clean_circumferences(knees, arm_rows, bond)
        names = ", ".join(str(k) for k in options) or "none in 10-24"
        raise ValueError(
            f"{knees} knees at circumference {circumference} gives the census "
            f"{dict(sorted(census.items()))}, not the exact "
            f"{{5: {2 * knees}, 7: {2 * knees}}} a knee toroid must have. The "
            "wedge a knee removes has to be a whole number of lattice steps. "
            f"Circumferences that do work here: {names}."
        )

    contacts = defect_contacts(vertices, tris)
    outer_rows, inner_rows = equator_rows(vertices, centre_radius, tube_radius)
    rows_needed = ((centre_radius + tube_radius)
                   / (centre_radius - tube_radius))

    from .capped_cnt import geometry_report
    from .fullerene_mesh import dual_honeycomb, relax_shell

    faces = np.asarray(tris, dtype=int)
    positions, bonds, rings = dual_honeycomb((vertices, faces))
    if relax:
        positions = relax_shell(positions, bonds, equilibrium=bond,
                                max_iterations=relax_iterations)

    ring_counts: dict[int, int] = {}
    for ring in rings:
        ring_counts[len(ring)] = ring_counts.get(len(ring), 0) + 1
    deficit = sum((6 - size) * count for size, count in ring_counts.items())

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  pbc=(False, False, False))
    span = positions.max(axis=0) - positions.min(axis=0)
    atoms.set_cell(span + 2.0 * vacuum)
    atoms.center()

    minor = float(tube_radius)
    atoms.info.update({
        "builder": "knee_toroid",
        "structure_type": "toroid",
        "knees": int(knees),
        "knee": knee,
        "bend_per_knee_deg": round(360.0 / knees, 2),
        "pairs_per_knee": PAIRS_PER_KNEE,
        # A pair buys 30 deg. Six knees of two pairs turn the axis all the
        # way round and spend exactly the 12 pentagons a torus asks for;
        # any other knee count misses one or the other.
        "turn_per_pair_deg": round(360.0 / (knees * PAIRS_PER_KNEE), 2),
        "turn_per_pair_law_deg": TURN_PER_PAIR,
        "circumference": int(circumference),
        "arm_rows": int(arm_rows),
        "major_radius": round(float(centre_radius), 3),
        "minor_radius": round(minor, 3),
        "aspect_ratio": round(float(centre_radius / minor), 3),
        "ring_counts": dict(sorted(ring_counts.items())),
        "ring_deficit": int(deficit),
        "disclinations_placed": round(float(placed), 4),
        # A torus wants exactly 12 pentagons outside whatever its radii;
        # this says whether the knee count matches that budget.
        "curvature_pentagons": CURVATURE_PENTAGONS,
        "curvature_balance": round(
            ring_counts.get(5, 0) / float(CURVATURE_PENTAGONS), 3),
        # Zero of either is what "clean wall" means: no lobe of pentagons,
        # no fused heptagon pair, no 5-7 dislocation. Measured, not assumed.
        "like_sign_pairs": int(contacts["like"]),
        "fused_dipoles": int(contacts["fused"]),
        # Curvature is not the whole story: the outer parallel of a torus
        # is (R+r)/(R-r) times longer than the inner one, so a net with
        # uniform bonds needs that many times more rows out there. A mitre
        # merge pairs the two boundary cycles one for one, so it cannot
        # change a row count at all -- these two numbers are what says so.
        "equator_rows": [int(outer_rows), int(inner_rows)],
        "equator_row_ratio": round(outer_rows / max(inner_rows, 1), 3),
        "equator_row_ratio_needed": round(float(rows_needed), 3),
        # The real atom indices per ring, not a census. `dopants/rings.py`
        # places a heteroatom on a named ring size and RAISES without
        # this, rather than falling back on perceiving rings by distance
        # -- which is the failure `fullerene_mesh` exists to prevent.
        "rings": [[int(x) for x in ring] for ring in rings],
        "bonds": sorted(bonds),
        "relaxed": bool(relax),
        # Measured on the finished atoms, so a caller can assert on the
        # geometry rather than trust the builder -- the same contract the
        # capped tube and the fullerene keep.
        "geometry": geometry_report(positions, sorted(bonds)),
    })
    quality = atoms.info["geometry"]
    low, high = SP2_BOND_RANGE
    if relax and not (low <= quality["bond_min"]
                      and quality["bond_max"] <= high):
        warnings.warn(
            f"The census is exact -- {dict(sorted(ring_counts.items()))}, "
            f"every disclination on its own curvature side and none of them "
            f"touching -- but the relaxed wall runs "
            f"{quality['bond_min']:.3f}-{quality['bond_max']:.3f} Å against "
            f"the sp2 band {low}-{high}, with bond angles down to "
            f"{quality['angle_min']:.0f} deg. A knee here turns the axis by "
            f"{360.0 / knees:.0f} deg over a tube of radius {minor:.1f} Å. "
            "The reason is measured and in `atoms.info`: the outer equator "
            f"carries {outer_rows} rows against {inner_rows} inside, a ratio "
            f"of {outer_rows / max(inner_rows, 1):.2f} where the metric asks "
            f"for {rows_needed:.2f}. A mitre merge pairs the two boundary "
            "cycles one for one, so it cannot create or destroy a row -- the "
            "disclinations land on the right curvature side and carry none "
            "of the metric. A bent all-hexagon stack at the same aspect "
            "measures no better (1.223-1.647 Å), so this is the aspect "
            "ratio, not the knees.",
            stacklevel=2,
        )

    if deficit != 0:                                    # pragma: no cover
        warnings.warn(
            f"The census {dict(sorted(ring_counts.items()))} gives sum(6-n) = "
            f"{deficit:+d}, but a torus is genus 1 and the budget is exactly "
            "0. The dual did not agree with the mesh, which should not "
            "happen -- please report the parameters.",
            stacklevel=2,
        )
    return atoms


#: Arm directions for the primitive (Schwarz P) node: six, along the cube
#: axes. A node of ``c`` arms is a sphere with ``c`` holes, so
#: ``chi = 2 - c`` and ``sum(6-n) = 6(2-c)`` -- see :func:`node_budget`.
PRIMITIVE_AXES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                           [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)

#: How far apart two seam representatives may be and still be the same
#: corner of the cube, and how close the two binding planes' magnitudes
#: must be for a vertex to count as sitting on both arcs. Both are in Å at
#: the mesh's own scale, and both are checked by the census rather than
#: tuned: a wrong value shows up as a seam with unequal sides.
CORNER_TOLERANCE = 1.2
CORNER_SPAN = 0.6

#: How far the two mouths of a periodic cell may miss each other.
MOUTH_TOLERANCE = 1.2


def node_budget(arms: int) -> int:
    """``sum(6 - n)`` a node of ``arms`` arms carries: ``6 * (2 - arms)``.

    A node is a sphere with one hole per arm, so ``chi = 2 - arms`` and
    Gauss-Bonnet in ring units gives the rest. It reproduces the genus
    table :func:`~nanocarbon_lab.builders.junction.build_schwarzite`
    quotes from nothing but counting arms: Schwarz P is one six-arm node
    per cubic cell (**-24**), the gyroid eight three-arm ones (**-48**)
    and Schwarz D eight four-arm ones (**-96**). A knee is the ``arms=2``
    case and pays 0, which is why a toroid's pentagons and heptagons come
    out in equal numbers.
    """
    return 6 * (2 - int(arms))


def collapse_degree_three(vertices: np.ndarray, tris):
    """Remove every degree-3 vertex and close its link with one triangle.

    Where three arms meet -- the eight corners of the cube on a primitive
    node -- each arm contributes a single edge to the shared vertex, so it
    comes out with degree 3: a three-membered ring, which is not carbon.
    The move that fixes it is exact rather than a repair. A degree-3
    vertex's three neighbours are already adjacent to one another, being
    its link, so deleting it and filling the hole adds no edge: one
    vertex, three edges and three faces go and one face comes back, which
    leaves ``chi`` untouched, while each of the three neighbours drops a
    degree.

    Measured on the primitive node: ``{3: 8, 6: 696, 8: 24}`` becomes
    ``{6: 696, 7: 24}``, because the neighbours of those eight vertices
    are exactly the twenty-four octagons. That is the Schwarz P census --
    hexagons and heptagons, nothing else.
    """
    tris = [tuple(int(x) for x in t) for t in tris]
    while True:
        adjacency: dict[int, set[int]] = defaultdict(set)
        incident: dict[int, int] = defaultdict(int)
        edges: dict[tuple[int, int], int] = defaultdict(int)
        for t in tris:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                adjacency[a].add(b)
                adjacency[b].add(a)
                edges[(min(a, b), max(a, b))] += 1
            for x in t:
                incident[x] += 1
        # A rim vertex has a low degree because its ring is cut off, not
        # because three arms met there, and its link is an arc rather than a
        # cycle -- so filling it WOULD add an edge and change `chi`. A closed
        # cell has no rim, so this changes nothing for the schwarzite; it is
        # what makes the move safe on an open node such as a junction.
        rim = {x for e, c in edges.items() if c != 2 for x in e}
        victim = next((v for v, ns in adjacency.items()
                       if len(ns) == 3 and incident[v] == 3
                       and v not in rim), None)
        if victim is None:
            break
        around = [t for t in tris if victim in t]
        keep = [x for x in around[0] if x != victim]
        third = (adjacency[victim] - set(keep)).pop()
        tris = [t for t in tris if victim not in t]
        tris.append((keep[0], keep[1], third))
    live = sorted({x for t in tris for x in t})
    relabel = {x: i for i, x in enumerate(live)}
    return vertices[live], [(relabel[a], relabel[b], relabel[c])
                            for a, b, c in tris]


def _link_cycle(tris, victim: int) -> list[int] | None:
    """The neighbours of ``victim`` in cyclic order, or ``None``.

    Each incident triangle contributes one step of the walk, so an
    interior vertex's link closes and a rim vertex's does not. Returning
    ``None`` rather than a partial arc is what keeps the callers from
    retriangulating a hole that is really a boundary.
    """
    onward: dict[int, int] = {}
    for t in tris:
        if victim not in t:
            continue
        a, b, c = t
        while a != victim:
            a, b, c = b, c, a
        onward[b] = c
    if not onward:
        return None
    start = next(iter(onward))
    cycle, x = [start], onward[start]
    while x != start:
        if x not in onward or len(cycle) > len(onward):
            return None
        cycle.append(x)
        x = onward[x]
    return cycle if len(cycle) == len(onward) else None


def collapse_degree_four(vertices: np.ndarray, tris):
    """Annihilate a 4-8 dipole by removing the square, not by flipping it.

    The Y node comes out of :func:`node_mesh` with the right heptagons in
    the right places and, on each seam, one extra **4-8 pair sitting side
    by side**. That pair is neutral -- ``(6-4) + (6-8) = 0`` -- so it
    costs the Gauss-Bonnet budget nothing and no census check sees it; it
    is a dislocation, and it is what stands between this node and the
    textbook junction.

    **No edge flip can remove it, and that is arithmetic rather than a
    search failure.** A flip drops its two endpoints a degree and raises
    its two opposites, so taking the 8 down takes a neighbouring hexagon
    down with it and bringing the 4 up brings another hexagon up:
    ``sum|deg - 6|`` is 4 before and 4 after, every time. Measured, greedy
    descent over every flip on this mesh finds not one improving move, and
    a plateau anneal across four seeds stays put. This is the same wall
    the Dunlap toroid met -- gliding a dislocation preserves it, and
    separating or annihilating it needs **climb**, which changes the
    vertex count and therefore cannot be a flip at all.

    So climb: delete the degree-4 vertex and fill the quadrilateral its
    link leaves with two triangles. That removes one vertex, four edges
    and four faces and returns two, leaving ``chi`` untouched, and drops
    every one of the four neighbours a degree. The **diagonal decides
    which two get it back**, and choosing it is the whole move: the link
    comes out in the cyclic order ``6, 7, 6, 8``, and the diagonal joining
    the two hexagons returns their degree and leaves the 8 at 7 and the 4
    gone. Measured on a ``k = 14`` Y node, one step takes
    ``{4: 6, 6: 234, 7: 6, 8: 6}`` to ``{4: 5, 6: 235, 7: 6, 8: 5}`` --
    the dipole gone and nothing else touched. The other diagonal gives
    ``{4: 5, 5: 2, 6: 232, 7: 6, 8: 6}`` and is refused.

    The choice is made by measurement rather than by that pattern: both
    diagonals are tried, a diagonal that is already an edge is skipped
    (it would be a duplicate), and the move is taken only when
    ``sum|deg - 6|`` over the **interior** strictly falls. A rim vertex's
    link is an arc, so it is never a candidate.
    """
    tris = [tuple(int(x) for x in t) for t in tris]

    def misfit(faces) -> int:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for t in faces:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                counts[(min(a, b), max(a, b))] += 1
        rim = {x for e, c in counts.items() if c != 2 for x in e}
        degree: dict[int, int] = defaultdict(int)
        for a, b in counts:
            degree[a] += 1
            degree[b] += 1
        return sum(abs(d - 6) for x, d in degree.items() if x not in rim)

    while True:
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for t in tris:
            for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
                counts[(min(a, b), max(a, b))] += 1
        rim = {x for e, c in counts.items() if c != 2 for x in e}
        degree: dict[int, int] = defaultdict(int)
        for a, b in counts:
            degree[a] += 1
            degree[b] += 1
        edges = set(counts)
        before = misfit(tris)

        taken = None
        for victim in sorted(x for x, d in degree.items()
                             if d == 4 and x not in rim):
            cycle = _link_cycle(tris, victim)
            if cycle is None or len(cycle) != 4:
                continue
            survivors = [t for t in tris if victim not in t]
            for shift in (0, 1):
                a, b, c, d = cycle[shift:] + cycle[:shift]
                if (min(a, c), max(a, c)) in edges:
                    continue
                candidate = survivors + [(a, b, c), (a, c, d)]
                if misfit(candidate) < before:
                    taken = candidate
                    break
            if taken is not None:
                break
        if taken is None:
            break
        tris = taken

    live = sorted({x for t in tris for x in t})
    relabel = {x: i for i, x in enumerate(live)}
    return vertices[live], [(relabel[a], relabel[b], relabel[c])
                            for a, b, c in tris]


def _union_find(n: int):
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    return find, union


def fill_triangular_holes(tris):
    """Close every three-vertex boundary cycle with one triangle.

    Where three arms meet, the trim leaves one of two things, and which
    depends on the angles between them. On the **primitive** node -- six
    arms along the cube axes -- the three meeting arms share a single
    vertex, which comes out with degree 3 and is removed by
    :func:`collapse_degree_three`. On a **Y** node -- three arms at 120
    deg in a plane -- they leave a triangular **hole** instead, and the
    node measures ``chi = -3`` with boundary cycles ``[3, 3, k, k, k]``:
    the three mouths plus two holes.

    Filling such a hole adds no edge, because all three already exist, so
    **no vertex changes degree**; only ``F`` rises, and ``chi`` with it.
    Measured on a three-arm node at ``k = 14``: ``chi`` goes from -3 to
    -1, the boundary becomes the three mouths alone, and the interior
    census reads ``{4: 6, 6: 150, 7: 6, 8: 6}`` for
    ``sum(6-n) = -6`` -- exactly :func:`node_budget` for three arms, which
    it was not before the fill.

    The two repairs are the same accident with two faces, so a node runs
    both: fill the holes, then collapse what is left.
    """
    tris = [tuple(int(x) for x in t) for t in tris]
    while True:
        cycles = _boundary_cycles(tris)
        if not cycles:
            break
        hole = next((c for c in cycles if len(c) == 3), None)
        if hole is None:
            break
        a, b, c = hole
        # Wind it against the face that already uses one of its edges, or
        # the new triangle faces the wrong way and the walk crosses itself.
        neighbour = next(t for t in tris if a in t and b in t)
        forward = (neighbour.index(b) - neighbour.index(a)) % 3 == 1
        tris.append((b, a, c) if forward else (a, b, c))
    return tris


def node_mesh(
    axes: np.ndarray,
    circumference: int,
    arm_rows: int,
    tube_radius: float,
    spacing: float,
    corner_span: float = CORNER_SPAN,
    corner_tol: float = CORNER_TOLERANCE,
    phases=None,
    mirrors=None,
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """A node of arms along **any** directions, trimmed and seamed.

    The generalisation of :func:`primitive_node_mesh` off the cube axes,
    and what a junction needs: three arms at 120 deg in a plane is a Y,
    four at 109.47 deg is a diamond node. The trim is by **dominance** --
    a vertex keeps the arm whose axis it lies furthest along -- and the
    surface that separates two arms is then the bisector plane of their
    axes through the origin, which is a plane for any pair and not only
    for perpendicular ones. So nothing new is needed off the cube.

    Returns ``(vertices, triangles)`` **open**: the arms' mouths are left
    as rims, because a junction is open at its arms the way a nanocone is
    at its base. The caller closes the three-vertex holes at the meeting
    points with :func:`fill_triangular_holes` and takes the dual with
    :func:`dual_open`.

    Raises
    ------
    ValueError
        If the trim, the mouths or a seam does not come out in register.
    """
    axes = np.asarray(axes, dtype=float)
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    reach = (arm_rows - 1) * spacing
    verts: list[list[float]] = []
    tris: list[tuple[int, int, int]] = []
    owner: list[int] = []
    phases = [0.0] * len(axes) if phases is None else list(phases)
    mirrors = [False] * len(axes) if mirrors is None else list(mirrors)
    for a, axis in enumerate(axes):
        outward = np.array([0.0, 0.0, 1.0])
        if abs(float(outward @ axis)) > 0.9:
            outward = np.array([1.0, 0.0, 0.0])
        local, local_tris = _ring_stack(circumference, arm_rows, tube_radius,
                                        spacing, mirror=bool(mirrors[a]))
        turn = 2.0 * np.pi * float(phases[a]) / circumference
        spin = np.array([[np.cos(turn), -np.sin(turn), 0.0],
                         [np.sin(turn), np.cos(turn), 0.0],
                         [0.0, 0.0, 1.0]])
        placed = local @ spin.T @ _frame(axis, outward).T + axis * (reach / 2.0)
        offset = len(verts)
        verts.extend(placed.tolist())
        owner.extend([a] * len(placed))
        tris.extend((x + offset, y + offset, z + offset)
                    for x, y, z in local_tris)
    vertices = np.asarray(verts, dtype=float)

    along = vertices @ axes.T
    keep = ((along.argmax(axis=1) == np.asarray(owner))
            & (along.max(axis=1) > 0.0))
    tris = [t for t in tris if keep[t[0]] and keep[t[1]] and keep[t[2]]]
    if not tris:
        raise ValueError("the dominance trim left nothing of the node.")
    used = sorted({x for t in tris for x in t})
    remap = {x: i for i, x in enumerate(used)}
    vertices = vertices[used]
    owner = [owner[x] for x in used]
    tris = [(remap[a], remap[b], remap[c]) for a, b, c in tris]

    cycles = _boundary_cycles(tris)
    if cycles is None:
        raise ValueError("the trimmed node's boundary is not manifold.")

    def reach_of(cycle) -> float:
        return float((vertices[cycle] @ axes[owner[cycle[0]]]).mean())

    mouths = [c for c in cycles if reach_of(c) > 0.9 * reach]
    inner = [c for c in cycles if c not in mouths]
    if len(mouths) != len(axes):
        raise ValueError(
            f"the node came out with {len(mouths)} mouths where "
            f"{len(axes)} were expected. Lengthen the arms."
        )

    find, union = _union_find(len(vertices))
    seam_vertices = sorted({x for c in inner for x in c})
    seams: dict[tuple[int, int], dict[int, list[int]]] = {}
    for x in seam_vertices:
        a = owner[x]
        projection = vertices[x] @ axes.T
        others = [j for j in range(len(axes)) if j != a]
        facing = max(others, key=lambda j: projection[j])
        target = projection[facing]
        normal = axes[a] - axes[facing]
        normal = normal / np.linalg.norm(normal)
        step = -float(vertices[x] @ normal) / float(axes[a] @ normal)
        vertices[x] = vertices[x] + step * axes[a]
        for j in others:
            if target - projection[j] >= corner_span:
                continue
            key = (min(a, j), max(a, j))
            seams.setdefault(key, {}).setdefault(a, []).append(x)

    for (a, b), sides in seams.items():
        if len(sides) != 2:
            raise ValueError(f"seam {a}-{b} has only one side.")
        here, there = sides[a], sides[b]
        if len(here) != len(there):
            raise ValueError(
                f"seam {a}-{b} has {len(here)} vertices on one side and "
                f"{len(there)} on the other, so it cannot pair up."
            )
        distances = np.linalg.norm(
            vertices[here][:, None, :] - vertices[there][None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        if len(set(nearest.tolist())) != len(here):
            raise ValueError(f"seam {a}-{b} has no one-to-one partner.")
        for t, x in enumerate(here):
            union(x, there[int(nearest[t])])

    representatives = sorted({find(x) for x in seam_vertices})
    points = vertices[representatives]
    spread = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    for i in range(len(representatives)):
        for j in range(i + 1, len(representatives)):
            if spread[i, j] < corner_tol:
                union(representatives[i], representatives[j])

    tris = [tuple(find(x) for x in t) for t in tris]
    tris = [t for t in tris if len(set(t)) == 3]
    live = sorted({x for t in tris for x in t})
    relabel = {x: i for i, x in enumerate(live)}
    return (vertices[live],
            [(relabel[a], relabel[b], relabel[c]) for a, b, c in tris])


def primitive_node_mesh(
    circumference: int,
    arm_rows: int,
    tube_radius: float,
    spacing: float,
) -> tuple[np.ndarray, list[tuple[int, int, int]], float]:
    """One Schwarz P cell: a six-arm node closed on the 3-torus.

    The arms run along the cube axes from a common origin and are trimmed
    by **dominance** -- a vertex keeps the arm whose axis it lies furthest
    along -- which for perpendicular axes is exactly the set of bisector
    planes, the same mitre the knee uses. The locus equidistant from two
    perpendicular lines through a point is a plane, so nothing new is
    needed to cut a node that was not already needed to cut a bend.

    The cell closes **by periodicity, not by capping**: in the crystal the
    ``+x`` mouth of one node *is* the ``-x`` mouth of its neighbour, so
    gluing them to each other leaves a genuinely closed surface with no
    boundary term in Gauss-Bonnet. Six arms give ``chi = -4``, genus 3,
    and ``sum(6-n) = -24``.

    Returns ``(vertices, triangles, cell)``. The eight cube corners come
    out as degree-3 vertices and are removed by
    :func:`collapse_degree_three`; the caller does that, because the
    intermediate is worth being able to look at.

    Raises
    ------
    ValueError
        If the trim, the seams or the periodic mouths do not come out in
        register, with which of the three it was.
    """
    reach = (arm_rows - 1) * spacing
    axes = PRIMITIVE_AXES
    verts: list[list[float]] = []
    tris: list[tuple[int, int, int]] = []
    owner: list[int] = []
    for a, axis in enumerate(axes):
        outward = (np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9
                   else np.array([1.0, 0.0, 0.0]))
        local, local_tris = _ring_stack(circumference, arm_rows, tube_radius,
                                        spacing)
        placed = local @ _frame(axis, outward).T + axis * (reach / 2.0)
        offset = len(verts)
        verts.extend(placed.tolist())
        owner.extend([a] * len(placed))
        tris.extend((x + offset, y + offset, z + offset)
                    for x, y, z in local_tris)
    vertices = np.asarray(verts, dtype=float)

    dominant = np.argmax(np.abs(vertices), axis=1)
    sign = np.sign(vertices[np.arange(len(vertices)), dominant])
    wanted = np.array([int(np.argmax(np.abs(ax))) for ax in axes])
    wanted_sign = np.array([np.sign(ax[int(np.argmax(np.abs(ax)))])
                            for ax in axes])
    held = np.asarray(owner)
    keep = (dominant == wanted[held]) & (sign == wanted_sign[held])
    tris = [t for t in tris if keep[t[0]] and keep[t[1]] and keep[t[2]]]
    if not tris:
        raise ValueError("the dominance trim left nothing of the node.")
    used = sorted({x for t in tris for x in t})
    remap = {x: i for i, x in enumerate(used)}
    vertices = vertices[used]
    owner = [owner[x] for x in used]
    tris = [(remap[a], remap[b], remap[c]) for a, b, c in tris]

    cycles = _boundary_cycles(tris)
    if cycles is None:
        raise ValueError("the trimmed node's boundary is not manifold.")

    def reach_of(cycle) -> float:
        axis = int(np.argmax(np.abs(axes[owner[cycle[0]]])))
        return float(np.abs(vertices[cycle][:, axis]).mean())

    mouths = [c for c in cycles if reach_of(c) > 0.9 * reach]
    inner = [c for c in cycles if c not in mouths]
    if len(mouths) != len(axes):
        raise ValueError(
            f"the node came out with {len(mouths)} mouths where "
            f"{len(axes)} were expected. Lengthen the arms."
        )

    find, union = _union_find(len(vertices))
    seam_vertices = sorted({x for c in inner for x in c})
    seams: dict[tuple[int, int], dict[int, list[int]]] = {}
    for x in seam_vertices:
        a = owner[x]
        own_axis = int(np.argmax(np.abs(axes[a])))
        others = [c for c in range(3) if c != own_axis]
        sizes = [abs(vertices[x][c]) for c in others]
        target = max(sizes)
        # Slide onto the plane that binds, along this arm's own axis.
        vertices[x][own_axis] = np.sign(vertices[x][own_axis]) * target
        # A CORNER of the cube has both magnitudes equal and belongs to
        # two arcs; filing it under one leaves the other seam an endpoint
        # short, which is where the unequal sides came from.
        for other, size in zip(others, sizes, strict=True):
            if target - size >= CORNER_SPAN:
                continue
            facing = np.sign(vertices[x][other]) or 1.0
            b = next(j for j in range(len(axes))
                     if int(np.argmax(np.abs(axes[j]))) == other
                     and np.sign(axes[j][other]) == facing)
            key = (min(a, b), max(a, b))
            seams.setdefault(key, {}).setdefault(a, []).append(x)

    for (a, b), sides in seams.items():
        if len(sides) != 2:
            raise ValueError(f"seam {a}-{b} has only one side.")
        here, there = sides[a], sides[b]
        if len(here) != len(there):
            raise ValueError(
                f"seam {a}-{b} has {len(here)} vertices on one side and "
                f"{len(there)} on the other, so it cannot pair up."
            )
        distances = np.linalg.norm(
            vertices[here][:, None, :] - vertices[there][None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        if len(set(nearest.tolist())) != len(here):
            raise ValueError(f"seam {a}-{b} has no one-to-one partner.")
        for t, x in enumerate(here):
            union(x, there[int(nearest[t])])

    # The eight cube corners: three arms bring a vertex each and all three
    # are the same point. They fall outside the pairwise seams, so they are
    # grouped by proximity along the (+-1, +-1, +-1) diagonals.
    representatives = sorted({find(x) for x in seam_vertices})
    points = vertices[representatives]
    spread = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    for i in range(len(representatives)):
        for j in range(i + 1, len(representatives)):
            if spread[i, j] < CORNER_TOLERANCE:
                union(representatives[i], representatives[j])

    cell = 2.0 * reach
    for first, second in ((0, 1), (2, 3), (4, 5)):
        near = next((c for c in mouths if owner[c[0]] == first), None)
        far = next((c for c in mouths if owner[c[0]] == second), None)
        if near is None or far is None:
            raise ValueError("a mouth has no opposite number.")
        axis = int(np.argmax(np.abs(axes[first])))
        moved = vertices[far].copy()
        moved[:, axis] += cell
        distances = np.linalg.norm(
            vertices[near][:, None, :] - moved[None, :, :], axis=2)
        nearest = distances.argmin(axis=1)
        if len(set(nearest.tolist())) != len(near):
            raise ValueError("the two mouths do not pair one to one.")
        if float(distances[np.arange(len(near)), nearest].max()) > MOUTH_TOLERANCE:
            raise ValueError(
                "the two mouths do not land on each other, so the cell "
                "would not be periodic. Try another arm length."
            )
        for t, x in enumerate(near):
            union(x, far[int(nearest[t])])

    tris = [tuple(find(x) for x in t) for t in tris]
    tris = [t for t in tris if len(set(t)) == 3]
    live = sorted({x for t in tris for x in t})
    relabel = {x: i for i, x in enumerate(live)}
    return (vertices[live],
            [(relabel[a], relabel[b], relabel[c]) for a, b, c in tris],
            float(cell))


#: The junctions this route builds, as unit axes. A **Y** is three arms
#: at 120 deg in a plane; a **tetrahedral** node is four at 109.47 deg,
#: which is the building block of a Schwarz D cell exactly as the
#: six-arm cube node is of a Schwarz P one. A *planar* X -- four arms at
#: 90 deg in a plane -- is deliberately absent: it builds, and it comes
#: out with ``chi = -4`` where a sphere with four holes has -2, because
#: the four corner unions close a tunnel through the middle. That is a
#: real surface and it is not the X junction anybody means.
JUNCTION_AXES: dict[str, np.ndarray] = {
    "y": np.array([[np.cos(a), np.sin(a), 0.0]
                   for a in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)]),
    "tetrahedral": np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                             [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    / np.sqrt(3.0),
}


def clean_junction_shapes(
    kind: str = "y",
    bond: float = CC_BOND,
    circumferences=tuple(range(8, 25)),
    rows_candidates=(7, 9, 11, 13),
) -> list[tuple[int, int]]:
    """``(arm_rows, circumference)`` pairs giving the exact junction census.

    A pair qualifies when the node closes, the interior census is
    hexagons plus heptagons alone -- no square, no octagon, no pentagon,
    a junction being a saddle everywhere -- and the heptagon count is
    exactly ``-node_budget(arms)``.
    """
    axes = JUNCTION_AXES[kind]
    budget = node_budget(len(axes))
    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    good: list[tuple[int, int]] = []
    for rows in rows_candidates:
        for k in circumferences:
            radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / k))
            try:
                vertices, tris = node_mesh(axes, k, rows, radius, spacing)
                tris = fill_triangular_holes(tris)
                vertices, tris = collapse_degree_three(vertices, tris)
                vertices, tris = collapse_degree_four(vertices, tris)
            except (ValueError, StopIteration, KeyError):
                continue
            census = _interior_census(tris)
            if set(census) - {6, 7} or census.get(7, 0) != -budget:
                continue
            good.append((int(rows), int(k)))
    return good


def _interior_census(tris) -> dict[int, int]:
    """Ring sizes of the interior alone, a rim vertex's ring being cut off."""
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            counts[(min(a, b), max(a, b))] += 1
    rim = {x for e, c in counts.items() if c != 2 for x in e}
    degree: dict[int, int] = defaultdict(int)
    for a, b in counts:
        degree[a] += 1
        degree[b] += 1
    census: dict[int, int] = {}
    for x, d in degree.items():
        if x in rim:
            continue
        census[d] = census.get(d, 0) + 1
    return dict(sorted(census.items()))


def build_knee_junction(
    kind: str = "y",
    circumference: int = 14,
    arm_rows: int = 9,
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_1D,
    relax: bool = True,
    relax_iterations: int = 3000,
) -> Atoms:
    """A nanotube junction with the six heptagons and nothing else.

    :func:`~nanocarbon_lab.builders.junction.build_junction` meshes the
    junction implicitly and lets the remesher choose the rings; measured,
    a Y comes back with 50 pentagons and 38 heptagons where six heptagons
    pay the whole budget -- sound, and an amorphous wall. This builds the
    node as the lattice object it is: three arms trimmed by **dominance**,
    their seams mitred and welded, the triangular holes where the arms
    meet filled, and the leftover dislocations **climbed out**.

    The census is then what Gauss-Bonnet asks for and no more. A node of
    ``c`` arms is a sphere with ``c`` holes, so ``chi = 2 - c`` and
    ``sum(6-n) = 6(2-c)``, and with no pentagons available -- a junction
    saddles everywhere, so positive curvature would be wrong -- it is paid
    in heptagons alone:

    ==============  =======  =======  =====================
    kind            arms     chi      census
    ==============  =======  =======  =====================
    ``y``           3        -1       ``{6: N, 7: 6}``
    ``tetrahedral`` 4        -2       ``{6: N, 7: 12}``
    ==============  =======  =======  =====================

    Measured at every circumference from 10 to 20 and every arm length
    tried, exactly -- six heptagons for the Y, twelve for the tetrahedral
    node, and not one square, octagon or pentagon beside them. Six is the
    number the published Y junctions carry.

    **The dislocations are what the climb removes, and no flip could.**
    The welded seams each come out carrying a neutral 4-8 pair, which
    costs the budget nothing and which every census check passes over.
    :func:`collapse_degree_four` is the move that takes them out; its
    docstring has the arithmetic showing a flip cannot.

    The mouths are left **open**, as a nanocone's rim is, and recorded in
    ``info["rim_atoms"]``.

    Parameters
    ----------
    kind
        ``"y"`` or ``"tetrahedral"`` -- see :data:`JUNCTION_AXES`.
    circumference, arm_rows
        Mesh vertices around each arm, and rows along it: the tube radius
        and the arm length. Not every pair closes --
        :func:`clean_junction_shapes` is the list that does.
    bond
        C-C length (Å).
    vacuum
        Padding around the finite structure (Å).
    relax, relax_iterations
        Whether to relax with the valence force field. The bond graph is
        explicit, so the census is fixed before this runs.

    Returns
    -------
    ase.Atoms
        Finite, with the census, the budget and the measured geometry in
        ``atoms.info``.

    Raises
    ------
    ValueError
        If the kind is unknown, or the shape does not give the exact
        census, naming the ones that do.
    """
    if kind not in JUNCTION_AXES:
        raise ValueError(
            f"unknown junction kind {kind!r}; the ones this route builds "
            f"are {sorted(JUNCTION_AXES)}."
        )
    axes = JUNCTION_AXES[kind]
    arms = len(axes)
    budget = node_budget(arms)
    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    tube_radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / circumference))
    try:
        vertices, tris = node_mesh(axes, circumference, arm_rows, tube_radius,
                                   spacing)
        tris = fill_triangular_holes(tris)
        vertices, tris = collapse_degree_three(vertices, tris)
        vertices, tris = collapse_degree_four(vertices, tris)
    except (ValueError, StopIteration, KeyError) as problem:
        shapes = clean_junction_shapes(kind, bond)
        raise ValueError(
            f"circumference {circumference} with {arm_rows}-row arms does "
            f"not close as a {kind} node ({problem}). Pairs that do, as "
            f"(arm_rows, circumference): {shapes or 'none found'}."
        ) from problem

    census = _interior_census(tris)
    deficit = sum((6 - size) * count for size, count in census.items())
    if set(census) - {6, 7} or census.get(7, 0) != -budget:
        shapes = clean_junction_shapes(kind, bond)
        raise ValueError(
            f"circumference {circumference} with {arm_rows}-row arms gives "
            f"the census {census} and sum(6-n) = {deficit:+d}, not the "
            f"hexagons-plus-{-budget}-heptagons a {arms}-arm node must have "
            "(it saddles everywhere, so it can carry no pentagon at all). "
            f"Pairs that do, as (arm_rows, circumference): "
            f"{shapes or 'none found'}."
        )

    from .capped_cnt import geometry_report
    from .fullerene_mesh import relax_shell

    positions, bonds, rings, rim = dual_open(vertices, tris)
    if relax:
        positions = relax_shell(positions, bonds, equilibrium=bond,
                                max_iterations=relax_iterations)

    ring_counts: dict[int, int] = {}
    for ring in rings:
        ring_counts[len(ring)] = ring_counts.get(len(ring), 0) + 1

    # The intrinsic check, and the one that is not a restatement of the
    # census: fit the surface over each ring and ask the sign of K there.
    # A heptagon is a -60 deg disclination, so it belongs in negative
    # curvature. Measured 100% on both kinds, mean sign exactly -1.00.
    from ..analyse.curvature import disclination_check
    check = disclination_check(positions, rings, sorted(bonds))

    span = positions.max(axis=0) - positions.min(axis=0)
    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  cell=np.diag(span + 2.0 * vacuum), pbc=(False, False, False))
    atoms.center()
    atoms.info.update({
        "builder": "knee_junction",
        "structure_type": "junction",
        "kind": kind,
        "arms": int(arms),
        "euler": int(2 - arms),
        "ring_budget": int(budget),
        "circumference": int(circumference),
        "arm_rows": int(arm_rows),
        "tube_radius": round(float(tube_radius), 3),
        "arm_length": round(float((arm_rows - 1) * spacing), 3),
        "ring_counts": dict(sorted(ring_counts.items())),
        "ring_deficit": int(sum((6 - s) * c for s, c in ring_counts.items())),
        "mesh_census": census,
        # A junction is a saddle everywhere, so a pentagon on one is never
        # right. This is what the implicit route cannot get to zero: it
        # returns fifty on a comparable Y.
        "pentagons": int(ring_counts.get(5, 0)),
        "disclination_check": check,
        "disclinations_placed": check["agreement"],
        # The mouths are two-coordinate by construction, as a nanocone's
        # rim is.
        "rim_atoms": [int(x) for x in rim],
        "terminal_atoms": [int(x) for x in rim],
        # The real atom indices per ring, not a census. `dopants/rings.py`
        # places a heteroatom on a named ring size and RAISES without
        # this, rather than falling back on perceiving rings by distance
        # -- which is the failure `fullerene_mesh` exists to prevent.
        "rings": [[int(x) for x in ring] for ring in rings],
        "bonds": sorted(bonds),
        "relaxed": bool(relax),
        "geometry": geometry_report(positions, sorted(bonds)),
    })
    return atoms


def describe_knee_junction(atoms: Atoms) -> str:
    """One line: the arms, the census and whether it is the exact one."""
    info = atoms.info
    counts = info.get("ring_counts", {})
    census = ", ".join(f"{s}:{c}" for s, c in sorted(counts.items()))
    mesh = info.get("mesh_census", {})
    exact = (set(mesh) <= {6, 7}
             and mesh.get(7, 0) == -int(info.get("ring_budget", 0)))
    verdict = ("exactly the heptagons Gauss-Bonnet asks for and nothing else"
               if exact else "not the exact census")
    return (
        f"knee junction ({info.get('kind', '?')}): {info.get('arms', 0)} arms, "
        f"{len(atoms)} atoms, tube radius "
        f"{info.get('tube_radius', 0):.2f} Å, rings {census}, sum(6-n) = "
        f"{info.get('ring_deficit', 0):+d} against a budget of "
        f"{info.get('ring_budget', 0):+d}; {verdict}."
    )


#: The eight sites of the conventional cubic diamond cell, as fractions
#: of its edge, and which way their arms point: the A sublattice looks
#: out along +(1,1,1) and its family, the B sublattice along the
#: negatives. That is the diamond structure, and it is why a node of four
#: arms at 109.47 deg is the piece a Schwarz D cell is made of.
DIAMOND_SITES: tuple[tuple[tuple[float, float, float], int], ...] = (
    ((0.00, 0.00, 0.00), +1), ((0.00, 0.50, 0.50), +1),
    ((0.50, 0.00, 0.50), +1), ((0.50, 0.50, 0.00), +1),
    ((0.25, 0.25, 0.25), -1), ((0.25, 0.75, 0.75), -1),
    ((0.75, 0.25, 0.75), -1), ((0.75, 0.75, 0.25), -1),
)

#: How far two glued mouths may miss each other, in Å, before the cell is
#: refused. A mouth is a ring of the mesh edge, ~2.5 Å, so this is well
#: inside one step and cannot accept a neighbour's ring by mistake.
MOUTH_REGISTER = 0.6


def diamond_cell_mesh(
    circumference: int,
    arm_rows: int,
    tube_radius: float,
    spacing: float,
) -> tuple[np.ndarray, list[tuple[int, int, int]], float]:
    """One Schwarz D cell: eight tetrahedral nodes, closed on the 3-torus.

    The D surface is the diamond lattice thickened into a wall, so its
    piece is the four-arm node :func:`node_mesh` already builds -- the
    same construction as :func:`primitive_node_mesh`'s six-arm one, with
    the nodes kept separate and glued rather than trimmed against each
    other. Each arm is **half** the strut to a neighbour, so two mouths
    meeting set the cell: ``2 * reach`` is the diamond bond ``a*sqrt(3)/4``
    and therefore ``a = 8 * reach / sqrt(3)``.

    **The budget is fixed before anything is built.** Gluing two boundary
    circles adds nothing to ``chi`` -- a circle has ``chi = 0`` -- so a
    cell of ``n`` nodes of ``c`` arms has ``chi = n(2-c)``, here
    ``8 * (2-4) = -16``, genus 9, and ``sum(6-n) = 6*chi = -96``. Measured
    at ``circumference=10, arm_rows=5``: 1408 triangles in a 39.4 Å cell,
    census ``{6: 592, 7: 96}``, ``sum(6-n) = -96``, no boundary edge and
    none shared by other than two faces. Ninety-six heptagons and nothing
    else -- no pentagon, which a minimal surface can have none of, and no
    square or octagon either.

    **This is the conventional cell, not the primitive one.** The
    rhombohedral primitive cell holds two nodes and is genus 3, which is
    what Lenosky's D216 is; it is not orthorhombic, and
    :func:`~nanocarbon_lab.builders.fullerene_mesh.minimum_image` takes an
    orthorhombic box only, so a bond across its seam would read as a
    cell-length stretch. The cubic cell is four primitive cells of the
    same surface, and it is the one that can be measured correctly.

    **Not every circumference works, and the node says so first.** At
    ``circumference=12`` the cell closes with ``{6: N, 9: 32}`` --
    nonagons, which is the tetrahedral node's own census at that size
    rather than anything the gluing did. :func:`clean_schwarzite_shapes`
    with ``kind="diamond"`` is the list that comes out exact.

    Returns ``(vertices, triangles, cell)`` with the cell **closed**: no
    rim, so :func:`mesh_census` reads every vertex.

    Raises
    ------
    ValueError
        If a node does not close, the mouths do not come out in pairs, or
        two glued mouths are out of register.
    """
    axes = JUNCTION_AXES["tetrahedral"]
    reach = (arm_rows - 1) * spacing
    cell = 8.0 * reach / np.sqrt(3.0)

    verts: list[list[float]] = []
    tris: list[tuple[int, int, int]] = []
    for fraction, sign in DIAMOND_SITES:
        local, local_tris = node_mesh(sign * axes, circumference, arm_rows,
                                      tube_radius, spacing)
        local_tris = fill_triangular_holes(local_tris)
        local, local_tris = collapse_degree_three(local, local_tris)
        local, local_tris = collapse_degree_four(local, local_tris)
        offset = len(verts)
        verts.extend((local + np.asarray(fraction) * cell).tolist())
        tris.extend((x + offset, y + offset, z + offset)
                    for x, y, z in local_tris)
    vertices = np.asarray(verts, dtype=float)

    cycles = _boundary_cycles(tris)
    if cycles is None:
        raise ValueError("a node's boundary is not manifold.")
    wanted = len(DIAMOND_SITES) * len(axes)
    if len(cycles) != wanted:
        raise ValueError(
            f"the cell came out with {len(cycles)} mouths where {wanted} "
            "were expected, so a node did not close."
        )

    find, union = _union_find(len(vertices))
    centres = np.array([vertices[c].mean(axis=0) for c in cycles])
    paired: set[int] = set()
    for i in range(len(cycles)):
        if i in paired:
            continue
        delta = centres - centres[i]
        delta -= cell * np.round(delta / cell)
        distance = np.linalg.norm(delta, axis=1)
        distance[i] = np.inf
        for j in paired:
            distance[j] = np.inf
        j = int(distance.argmin())
        if distance[j] > MOUTH_REGISTER:
            raise ValueError(
                f"mouth {i} has no partner: the nearest is "
                f"{distance[j]:.2f} Å away, past the {MOUTH_REGISTER} Å a "
                "weld allows."
            )
        paired.update((i, j))
        here = vertices[cycles[i]]
        # The partner may be in a neighbouring cell; carry the translation
        # that brought its centre into register, not a wrap of each vertex.
        shift = cell * np.round((centres[i] - centres[j]) / cell)
        there = vertices[cycles[j]] + shift
        spread = np.linalg.norm(here[:, None, :] - there[None, :, :], axis=2)
        nearest = spread.argmin(axis=1)
        if len(set(nearest.tolist())) != len(cycles[i]):
            raise ValueError(
                f"mouths {i} and {j} have no one-to-one partner, so they "
                "are out of phase rather than merely apart."
            )
        worst = float(spread[np.arange(len(cycles[i])), nearest].max())
        if worst > MOUTH_REGISTER:
            raise ValueError(
                f"mouths {i} and {j} are out of register by {worst:.2f} Å."
            )
        for q, x in enumerate(cycles[i]):
            union(x, cycles[j][int(nearest[q])])

    tris = [tuple(find(x) for x in t) for t in tris]
    tris = [t for t in tris if len(set(t)) == 3]
    live = sorted({x for t in tris for x in t})
    relabel = {x: i for i, x in enumerate(live)}
    return (vertices[live],
            [(relabel[a], relabel[b], relabel[c]) for a, b, c in tris],
            float(cell))


#: The schwarzite cells this route builds, as (mesh routine, node arms,
#: nodes per cell, genus). The budget follows from the last three:
#: gluing two boundary circles adds nothing to chi, so a cell of `n`
#: nodes of `c` arms has chi = n(2-c) and sum(6-n) = 6*chi.
SCHWARZITE_CELLS: dict[str, tuple[int, int, int]] = {
    #     arms, nodes, genus
    "primitive": (6, 1, 3),
    "diamond": (4, 8, 9),
}

#: ``(circumference, arm_rows)`` per cell kind: the shape whose relaxed
#: geometry measured best, not a round number. A P cell wants a wide arm
#: and a long one; a D cell wants the opposite, because eight nodes in one
#: cube leave far less room between them.
DEFAULT_SCHWARZITE_SHAPE: dict[str, tuple[int, int]] = {
    "primitive": (20, 9),
    "diamond": (18, 5),
}


def _cell_mesh(kind: str, circumference: int, arm_rows: int,
               tube_radius: float, spacing: float):
    """The mesh routine for a cell kind, both of one signature."""
    if kind == "primitive":
        return primitive_node_mesh(circumference, arm_rows, tube_radius,
                                   spacing)
    return diamond_cell_mesh(circumference, arm_rows, tube_radius, spacing)


def schwarzite_budget(kind: str) -> int:
    """``sum(6-n)`` for a cell kind, from its nodes and their arms alone."""
    arms, nodes, _ = SCHWARZITE_CELLS[kind]
    return nodes * node_budget(arms)


def clean_schwarzite_shapes(
    bond: float = CC_BOND,
    circumferences=tuple(range(12, 29)),
    rows_candidates=(7, 9, 11, 13, 15),
    kind: str = "primitive",
) -> list[tuple[int, int]]:
    """``(arm_rows, circumference)`` pairs that give the exact census.

    The wedge the dominance trim removes has to be a whole number of
    lattice steps, exactly as the toroid's mitre does, so the pairs that
    close cleanly are found by building and counting rather than derived.
    A pair qualifies only when the cell is closed, the census matches the
    kind's own genus, and the
    census is hexagons plus **exactly** ``-schwarzite_budget(kind)``
    heptagons -- 24 for a P cell, 96 for a D one -- and no pentagons,
    which a minimal surface can have none of, and no octagons.
    """
    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    good: list[tuple[int, int]] = []
    for rows in rows_candidates:
        for k in circumferences:
            radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / k))
            try:
                vertices, tris, _ = _cell_mesh(kind, k, rows, radius, spacing)
                vertices, tris = collapse_degree_three(vertices, tris)
            except (ValueError, StopIteration, KeyError):
                continue
            census, _, broken = mesh_census(vertices, tris)
            if broken or set(census) - {6, 7}:
                continue
            if census.get(7, 0) != -schwarzite_budget(kind):
                continue
            good.append((int(rows), int(k)))
    return good


def build_knee_schwarzite(
    circumference: int | None = None,
    arm_rows: int | None = None,
    bond: float = CC_BOND,
    relax: bool = True,
    relax_iterations: int = 3000,
    kind: str = "primitive",
) -> Atoms:
    """A Schwarz P schwarzite with the census a minimal surface must have.

    :func:`~nanocarbon_lab.builders.junction.build_schwarzite` meshes the
    surface implicitly and lets the remesher choose; measured at a 32 Å
    cell it returns 910 atoms and ``{5: 33, 6: 361, 7: 57}`` -- the budget
    right at ``sum(6-n) = -24`` and the arrangement wrong, ninety
    disclinations where twenty-four suffice and **thirty-three of them
    pentagons on a surface that has no positive curvature anywhere**.

    This builds the node instead: six arms trimmed by dominance, their
    seams mitred, the eight cube corners collapsed, and the cell closed by
    gluing each mouth to its opposite number. Measured at
    ``circumference=20, arm_rows=9``: 968 atoms in a 31.9 Å cell, rings
    ``{6: 456, 7: 24}``, ``sum(6-n) = -24``, bonds 1.417-1.540 Å, angles
    from 114.2 deg and no non-bonded contact under 2 Å.

    The cell is **rescaled to its own mean bond before relaxing**, for the
    same reason ``junction._finish`` relaxes the cell with the atoms: the
    mesh is coarser than the surface area asks for, so everything comes
    out stretched by the same factor and the force field cannot fix a
    uniform scale with the cell held.

    Parameters
    ----------
    circumference, arm_rows
        Mesh vertices around each arm, and rows along it. Together they
        set the cell. Not every pair closes cleanly --
        :func:`clean_schwarzite_shapes` is the list that does.
    bond
        C-C length (Å).
    relax, relax_iterations
        Whether to relax with the valence force field, under the minimum
        image convention. The bond graph is explicit, so the census is
        fixed before this runs.

    Returns
    -------
    ase.Atoms
        Periodic in all three directions, with the census, the budget and
        the measured geometry in ``atoms.info``.

    Raises
    ------
    ValueError
        If the pair does not give the exact census, naming the ones that
        do.
    """
    if kind not in SCHWARZITE_CELLS:
        raise ValueError(
            f"unknown schwarzite kind {kind!r}; the ones this route builds "
            f"are {sorted(SCHWARZITE_CELLS)}."
        )
    # The shape that measured best for this kind, since what suits a P
    # cell does not suit a D one: eight nodes in a cube leave far less
    # room between them than one node does.
    default_k, default_rows = DEFAULT_SCHWARZITE_SHAPE[kind]
    circumference = default_k if circumference is None else int(circumference)
    arm_rows = default_rows if arm_rows is None else int(arm_rows)
    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    tube_radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / circumference))
    arms, nodes, genus = SCHWARZITE_CELLS[kind]
    budget = schwarzite_budget(kind)
    surface = "Schwarz P" if kind == "primitive" else "Schwarz D"
    try:
        vertices, tris, cell = _cell_mesh(kind, circumference, arm_rows,
                                          tube_radius, spacing)
        vertices, tris = collapse_degree_three(vertices, tris)
    except (ValueError, StopIteration, KeyError) as problem:
        shapes = clean_schwarzite_shapes(bond, kind=kind)
        raise ValueError(
            f"circumference {circumference} with {arm_rows}-row arms does "
            f"not close as a {surface} cell ({problem}). Pairs that do, as "
            f"(arm_rows, circumference): {shapes or 'none found'}."
        ) from problem

    census, _, broken = mesh_census(vertices, tris)
    deficit = sum((6 - size) * count for size, count in census.items())
    if broken or set(census) - {6, 7} or deficit != budget:
        shapes = clean_schwarzite_shapes(bond, kind=kind)
        raise ValueError(
            f"circumference {circumference} with {arm_rows}-row arms gives "
            f"the census {dict(sorted(census.items()))} and sum(6-n) = "
            f"{deficit:+d}, not the hexagons-plus-{-budget}-heptagons a "
            f"{surface} cell must have (it is a minimal surface, so it can "
            "have no pentagons at all). Pairs that do, as "
            f"(arm_rows, circumference): {shapes or 'none found'}."
        )

    from .capped_cnt import geometry_report
    from .fullerene_mesh import dual_honeycomb, minimum_image, relax_shell

    positions, bonds, rings = dual_honeycomb(
        (vertices, np.asarray(tris, dtype=int)), box=cell)
    pairs = np.asarray(sorted(bonds), dtype=int)
    raw = np.array([np.linalg.norm(minimum_image(positions[j] - positions[i],
                                                 cell))
                    for i, j in pairs])
    scale = bond / float(raw.mean())
    positions, cell = positions * scale, cell * scale
    if relax:
        positions = relax_shell(positions, bonds, equilibrium=bond,
                                max_iterations=relax_iterations, box=cell)

    ring_counts: dict[int, int] = {}
    for ring in rings:
        ring_counts[len(ring)] = ring_counts.get(len(ring), 0) + 1

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  cell=[cell, cell, cell], pbc=(True, True, True))
    atoms.wrap()
    atoms.info.update({
        "builder": "knee_schwarzite",
        "structure_type": "schwarzite",
        "kind": kind,
        "surface": surface,
        "arms": int(arms),
        "nodes": int(nodes),
        "genus": int(genus),
        "euler": 2 - 2 * genus,
        "ring_budget": int(budget),
        "circumference": int(circumference),
        "arm_rows": int(arm_rows),
        "cell_length": round(float(cell), 3),
        "tube_radius": round(float(tube_radius * scale), 3),
        "ring_counts": dict(sorted(ring_counts.items())),
        "ring_deficit": int(sum((6 - s) * c for s, c in ring_counts.items())),
        # A minimal surface saddles everywhere, so a pentagon on one is
        # never right. This is the number the implicit route cannot get to
        # zero: it returns 33 at a comparable cell.
        "pentagons": int(ring_counts.get(5, 0)),
        # The real atom indices per ring, not a census. `dopants/rings.py`
        # places a heteroatom on a named ring size and RAISES without
        # this, rather than falling back on perceiving rings by distance
        # -- which is the failure `fullerene_mesh` exists to prevent.
        "rings": [[int(x) for x in ring] for ring in rings],
        "bonds": sorted(bonds),
        "relaxed": bool(relax),
        # Judged by the plain sp2 window: a heptagon's interior angle is
        # 128.6 deg before any strain, and this cell's widest is 122.4, so
        # it does not need the haeckelite family's looser band.
        "geometry": geometry_report(positions, sorted(bonds), box=cell),
    })
    return atoms


def describe_knee_schwarzite(atoms: Atoms) -> str:
    """One line: the cell, the census and whether it is pentagon-free."""
    info = atoms.info
    counts = info.get("ring_counts", {})
    census = ", ".join(f"{s}:{c}" for s, c in sorted(counts.items()))
    clean = ("no pentagons, as a minimal surface must have"
             if not info.get("pentagons")
             else f"{info['pentagons']} pentagons, which a minimal surface "
                  "should have none of")
    return (
        f"knee schwarzite ({info.get('kind', '?')}): {len(atoms)} atoms in a "
        f"{info.get('cell_length', 0):.1f} Å cell, genus "
        f"{info.get('genus', 0)}, rings {census}, sum(6-n) = "
        f"{info.get('ring_deficit', 0):+d} against a budget of "
        f"{info.get('ring_budget', 0):+d}; {clean}."
    )


#: Coil diameter over tube diameter for the published single-wall coils.
#: Popovic finds it for both of their classes and Liu's Table 2 gives 3.53
#: to 3.88 for the (5,5) through (8,8). A coil outside that band is not
#: wrong -- multi-wall coils reach ten and more -- but it is not the
#: geometry those calculations relaxed to, and the builder says so.
LITERATURE_COIL_ASPECT = (3.5, 3.9)


def build_knee_coil(
    coil_radius: float = 12.0,
    pitch: float = 12.0,
    sides_per_turn: int = 8,
    turns: int = 2,
    circumference: int = 8,
    knee: str = "pentagon",
    bond: float = CC_BOND,
    vacuum: float = DEFAULT_VACUUM_1D,
    relax: bool = True,
    relax_iterations: int = 3000,
) -> Atoms:
    """A nanocoil of straight hexagon arms and mitred knees.

    The same construction as :func:`build_knee_toroid` on a **helix**
    instead of a ring, which works because the mitre reflection sends one
    corner to the next on any equal-step path (see
    :func:`knee_path_mesh`). Liu et al. show a real coil seen down its
    axis as a **polygon**: the wall relieves its strain at a few knees,
    not everywhere at once, and this builds that object directly.

    The point of it is tightness. `nanocoil.build_nanocoil` winds a
    finished lattice, so it can only stretch: at a 25 Å coil radius it
    **refuses outright** (bond 1.60 Å) and wants about 51 Å before the
    wall fits an 8% budget. Here the pentagons and heptagons absorb the
    bend instead, so a 12 Å coil -- ``D/d`` 3.73, inside the published
    single-wall band -- comes out at 1.331-1.527 Å with no non-bonded
    contact under 2 Å.

    The coil is **finite and open**, as :func:`nanocoil.build_nanocoil`
    is: its two rims are two-coordinate and recorded in
    ``info["rim_atoms"]``. It is not welded into a period, because a
    helix has holonomy -- carrying the azimuthal frame round one period
    by the mitre reflections does not return it to where it started, so
    the two rims land out of register with each other. Closing it needs
    that holonomy to come out a whole number of lattice steps, which is a
    condition on the geometry rather than a detail of the code.

    Parameters
    ----------
    coil_radius, pitch
        Helix radius and rise per turn (Å).
    sides_per_turn
        Knees per turn, so the axis turns ``360 / sides_per_turn`` degrees
        at each one. This is the polygon Liu et al. photograph.
    turns
        How many turns to build.
    circumference
        Mesh vertices around the tube, which sets the tube radius.
    knee
        ``"pentagon"`` for two pentagons on the outside of each bend and
        two heptagons on the inside, ``"octagon"`` for a square outside
        and an octagon inside. See :func:`knee_polygon_mesh`.
    bond, vacuum
        C-C length and the padding around the finished molecule (Å).
    relax, relax_iterations
        Whether to relax with the valence force field. The bond graph is
        explicit, so the census is fixed before this runs.

    Returns
    -------
    ase.Atoms
        With the census, the placement, ``D/d`` and the rims in
        ``atoms.info``.

    Raises
    ------
    ValueError
        If the pitch is not clear of the tube, if ``knee`` is unknown, or
        if the path's knees are too sharp for the arms to meet -- in which
        case the message says to widen the coil or use more sides.
    """
    if knee not in ("pentagon", "octagon"):
        raise ValueError(
            f"knee must be 'pentagon' or 'octagon', not {knee!r}."
        )
    if turns < 1 or sides_per_turn < 3:
        raise ValueError(
            f"a coil needs at least one turn and three sides a turn "
            f"({turns} and {sides_per_turn} given)."
        )
    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    tube_radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / circumference))
    if pitch <= 2.0 * tube_radius:
        raise ValueError(
            f"a pitch of {pitch:.1f} Å does not clear a tube of radius "
            f"{tube_radius:.2f} Å, so consecutive turns would pass through "
            "each other. Raise the pitch above "
            f"{2.0 * tube_radius:.1f} Å or narrow the tube."
        )

    angle = 2.0 * np.pi * np.arange(sides_per_turn * turns + 1) / sides_per_turn
    points = np.column_stack([coil_radius * np.cos(angle),
                              coil_radius * np.sin(angle),
                              pitch * angle / (2.0 * np.pi)])
    shift = SEAM_SHIFT if knee == "pentagon" else 0
    vertices, tris, _ = knee_path_mesh(points, circumference, tube_radius,
                                       spacing, seam_shift=shift, closed=False)

    census, degrees, broken = mesh_census(vertices, tris)
    contacts = defect_contacts(vertices, tris, skip_rim=True)
    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_count[(min(a, b), max(a, b))] += 1
    on_rim = {x for e, c in edge_count.items() if c != 2 for x in e}
    fives = [x for x, d in degrees.items() if d == 5 and x not in on_rim]
    sevens = [x for x, d in degrees.items() if d == 7 and x not in on_rim]
    outside = sum(1 for x in fives if radius[x] > coil_radius)
    inside = sum(1 for x in sevens if radius[x] < coil_radius)
    placed = ((outside + inside) / float(len(fives) + len(sevens))
              if fives or sevens else 1.0)

    from .capped_cnt import geometry_report
    from .fullerene_mesh import relax_shell

    positions, bonds, rings, rim = dual_open(vertices, tris)
    if relax:
        positions = relax_shell(positions, bonds, equilibrium=bond,
                                max_iterations=relax_iterations)

    ring_counts: dict[int, int] = {}
    for ring in rings:
        ring_counts[len(ring)] = ring_counts.get(len(ring), 0) + 1

    atoms = Atoms(symbols=["C"] * len(positions), positions=positions,
                  pbc=(False, False, False))
    span = positions.max(axis=0) - positions.min(axis=0)
    atoms.set_cell(span + 2.0 * vacuum)
    atoms.center()

    aspect = coil_radius / tube_radius
    knees = sides_per_turn * turns - 1
    atoms.info.update({
        "builder": "knee_coil",
        "structure_type": "nanocoil",
        "knee": knee,
        "knees": int(knees),
        "sides_per_turn": int(sides_per_turn),
        "turns": int(turns),
        "bend_per_knee_deg": round(360.0 / sides_per_turn, 2),
        "pairs_per_knee": PAIRS_PER_KNEE,
        "coil_radius": round(float(coil_radius), 3),
        "pitch": round(float(pitch), 3),
        "tube_radius": round(float(tube_radius), 3),
        # D/d, the number the single-wall coil papers report.
        "coil_aspect": round(float(aspect), 3),
        "literature_coil_aspect": list(LITERATURE_COIL_ASPECT),
        "circumference": int(circumference),
        "ring_counts": dict(sorted(ring_counts.items())),
        "mesh_census": dict(sorted(census.items())),
        "disclinations_placed": round(float(placed), 4),
        "like_sign_pairs": int(contacts["like"]),
        "fused_dipoles": int(contacts["fused"]),
        # The rims are two-coordinate by construction, as a nanocone's is.
        "rim_atoms": [int(x) for x in rim],
        "terminal_atoms": [int(x) for x in rim],
        # The real atom indices per ring, not a census. `dopants/rings.py`
        # places a heteroatom on a named ring size and RAISES without
        # this, rather than falling back on perceiving rings by distance
        # -- which is the failure `fullerene_mesh` exists to prevent.
        "rings": [[int(x) for x in ring] for ring in rings],
        "bonds": sorted(bonds),
        "relaxed": bool(relax),
        "geometry": geometry_report(positions, sorted(bonds)),
    })
    if broken and not rim:                              # pragma: no cover
        warnings.warn(
            "The coil mesh is not manifold and has no rim to explain it.",
            stacklevel=2,
        )
    low, high = LITERATURE_COIL_ASPECT
    if not low <= aspect <= high:
        warnings.warn(
            f"D/d = {aspect:.2f} is outside the {low}-{high} band the "
            "published single-wall coils sit in. The structure is not wrong "
            "-- multi-wall coils reach ten and more -- but it is not the "
            "geometry those calculations relaxed to.",
            stacklevel=2,
        )
    return atoms


def describe_knee_coil(atoms: Atoms) -> str:
    """One line: what was wound, and where the disclinations went."""
    info = atoms.info
    counts = info.get("ring_counts", {})
    census = ", ".join(f"{s}:{c}" for s, c in sorted(counts.items()))
    placed = info.get("disclinations_placed")
    where = ("" if placed is None
             else f", {100 * placed:.0f}% of disclinations on the curvature "
                  "side they belong on")
    like, fused = info.get("like_sign_pairs"), info.get("fused_dipoles")
    clean = ("" if like is None
             else (", every one isolated in hexagons"
                   if not like and not fused
                   else f", {like} like-sign and {fused} fused contacts"))
    return (
        f"knee coil: {info.get('turns', 0)} turns of "
        f"{info.get('sides_per_turn', 0)} sides, {info.get('knees', 0)} knees "
        f"of {info.get('bend_per_knee_deg', 0):.1f} deg, R="
        f"{info.get('coil_radius', 0):.1f} r={info.get('tube_radius', 0):.2f} "
        f"Å (D/d {info.get('coil_aspect', 0):.2f}), {len(atoms)} atoms, rings "
        f"{census}{where}{clean}."
    )


def describe_knee_toroid(atoms: Atoms) -> str:
    """One line: what was built, and where the disclinations went."""
    info = atoms.info
    counts = info.get("ring_counts", {})
    census = ", ".join(f"{s}:{c}" for s, c in sorted(counts.items()))
    placed = info.get("disclinations_placed")
    where = ("" if placed is None
             else f", {100 * placed:.0f}% of disclinations on the curvature "
                  "side they belong on")
    like, fused = info.get("like_sign_pairs"), info.get("fused_dipoles")
    clean = ("" if like is None
             else (", every one isolated in hexagons"
                   if not like and not fused
                   else f", {like} like-sign and {fused} fused contacts"))
    turn = info.get("turn_per_pair_deg")
    law = ("" if turn is None
           else f" {info.get('pairs_per_knee', 0)} pairs each, "
                f"{turn:.1f} deg a pair against the {TURN_PER_PAIR:.0f} the "
                "law asks for," if abs(turn - TURN_PER_PAIR) > 0.05
           else f" {info.get('pairs_per_knee', 0)} pairs each at exactly "
                f"{TURN_PER_PAIR:.0f} deg a pair,")
    return (
        f"knee toroid: {info.get('knees', 0)} knees of "
        f"{info.get('bend_per_knee_deg', 0):.1f} deg,{law} R="
        f"{info.get('major_radius', 0):.1f} r={info.get('minor_radius', 0):.1f} "
        f"Å (R/r {info.get('aspect_ratio', 0):.2f}), {len(atoms)} atoms, "
        f"rings {census}, sum(6-n) = {info.get('ring_deficit', 0):+d}"
        f"{where}{clean}."
    )
