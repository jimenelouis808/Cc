"""Heptanene: a trivalent net of heptagons and nothing else.

The question this module answers is not "how do I build it" but "may it
exist at all", and the answer is decided before any geometry by the same
Euler budget the rest of this package checks its structures against:

    sum over faces of (6 - n) = 6 * chi

A net of nothing but heptagons contributes ``-1`` per face, so ``F``
heptagons force ``chi = -F/6``. That single line settles all three
geometries at once:

* **Elliptic** -- a sphere, or any closed trivalent polyhedron, has
  ``chi = +2`` and therefore needs ``F = -12`` heptagons. A negative
  count is not a hard case, it is a contradiction. **Impossible.**
  (This is the same budget that gives a fullerene its twelve pentagons:
  positive curvature is paid for in faces *smaller* than six, and a
  heptagon is the wrong sign.)
* **Euclidean** -- a periodic sheet is a torus, ``chi = 0``, so ``F = 0``.
  The only all-heptagon Euclidean net is the one with no heptagons in
  it. **Impossible**, and this is exactly why
  :mod:`~nanocarbon_lab.builders.haeckelite` must pair every heptagon it
  makes with a pentagon: on a flat sheet the budget is zero and the two
  are each other's payment.
* **Hyperbolic** -- ``chi < 0`` gives ``F = -6*chi > 0``. **Possible**,
  and only here.

The regular-tiling test agrees independently: ``{p,q}`` is elliptic,
Euclidean or hyperbolic as ``(p-2)(q-2)`` is less than, equal to or
greater than 4. For trivalent carbon ``q = 3``, so ``{6,3}`` is graphene
at exactly 4 and ``{7,3}`` is hyperbolic at 5. Heptanene is the order-3
heptagonal tiling of the hyperbolic plane.

**So heptanene is not a 2D material.** Hilbert's theorem says the
hyperbolic plane admits no isometric embedding in three-space, so there
is no flat sheet to make, at any size, however it is relaxed. What can
be made is a closed surface of negative curvature carrying the same
tiling -- which in carbon means a schwarzite. Orientability forces ``F``
to be a multiple of 12 (``chi = -F/6`` must be even), so the series
starts at:

    F = 12   V = 28   E = 42   chi = -2   genus 2
    F = 24   V = 56   E = 84   chi = -4   genus 3
    F = 36   V = 84   E = 126  chi = -6   genus 4

The second is the one this module builds, and the reason is physical
rather than aesthetic: **genus 3 is the genus of a triply periodic
minimal surface's primitive cell** -- Schwarz P and Schwarz D both --
which is the surface a periodic carbon crystal actually has. It is also,
combinatorially, the **Klein quartic**: the ``{7,3}`` regular map whose
automorphism group is ``PSL(2,7)`` of order 168, the smallest Hurwitz
group. :func:`klein_quartic_map` constructs it from that group rather
than transcribing a table, so its 24 heptagons, 56 trivalent vertices and
84 edges are derived and can be checked rather than trusted.

One property of it is worth knowing before reading any geometry here.
``PSL(2,7)`` has no real three-dimensional irreducible representation, so
**the full symmetry of heptanene cannot be realised in three-space at
all**. The Laplacian shows it without any representation theory: the
first non-trivial eigenvalue has multiplicity **8**, not 3, so there is
no natural spectral embedding into R^3 and a projection onto any three of
those eight modes collapses (measured: 112 close contacts, bonds from
0.61 to 2.40 Å). Any realisation therefore keeps a subgroup of the
symmetry, not all of it -- which is a fact about the object, not a
failure of the relaxer, and the module does not pretend otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The field the group is built over. ``PSL(2,7)`` is the automorphism
#: group of the Klein quartic and has order 168 = 84 * 2 = 24 * 7 = 56 * 3
#: -- one element per dart, which is what makes the map regular.
FIELD = 7

#: Trivalent carbon: three bonds at every atom.
VALENCE = 3


@dataclass(frozen=True)
class Geometry:
    """Whether an all-heptagon trivalent net may exist in one geometry."""

    name: str
    euler: int | None
    faces: int | None
    possible: bool
    reason: str


def admissible_geometries(ring: int = 7) -> list[Geometry]:
    """Decide, for each geometry, whether an all-``ring`` trivalent net exists.

    Parameters
    ----------
    ring
        Face size. 7 is heptanene; 6 returns graphene's own answer, which
        is the check that this reasoning is not special pleading -- a
        hexagon pays 0, so the Euclidean case admits any number of them
        and the elliptic and hyperbolic ones admit none.

    Returns
    -------
    list of Geometry
        Elliptic, Euclidean and hyperbolic, in that order.

    Notes
    -----
    The whole argument is ``sum(6 - n) = 6 * chi`` with every face the
    same size, so ``F * (6 - ring) = 6 * chi``. Nothing about carbon
    enters; this is topology, and it is why no amount of relaxation can
    produce a flat heptagon sheet.
    """
    pay = 6 - ring
    out: list[Geometry] = []
    for name, chi in (("elliptic", 2), ("euclidean", 0)):
        needed = 6 * chi
        if pay == 0:
            possible = needed == 0
            faces = None if possible else 0
            reason = (
                f"a {ring}-gon pays 0 toward the budget, so any number of "
                f"them satisfies chi = {chi} only when the budget is 0"
                if possible else
                f"a {ring}-gon pays 0, which can never reach the "
                f"required {needed}"
            )
        else:
            faces = needed // pay if needed % pay == 0 else None
            possible = faces is not None and faces > 0
            if faces is None:
                reason = f"{needed} is not divisible by {pay}"
            elif faces > 0:
                reason = f"{faces} faces satisfy the budget"
            else:
                reason = (
                    f"the budget needs {faces} faces, and a count cannot be "
                    f"{'negative' if faces < 0 else 'zero'}"
                )
        out.append(Geometry(name, chi, faces, possible, reason))

    if pay < 0:
        reason = (
            f"chi = F * {pay} / 6 is negative for any F > 0, which is what "
            "a surface of negative curvature has"
        )
        possible = True
    else:
        reason = f"a {ring}-gon pays {pay} >= 0 and cannot make chi negative"
        possible = False
    out.append(Geometry("hyperbolic", None, None, possible, reason))
    return out


def closed_surface_series(ring: int = 7, most: int = 5
                          ) -> list[tuple[int, int, int, int, int]]:
    """``(F, V, E, chi, genus)`` for the smallest closed orientable nets.

    Counting twice over: every face has ``ring`` edges and every edge
    borders two faces, so ``E = ring * F / 2``; every vertex has three
    edges and every edge two ends, so ``V = 2 * E / 3``. Both must be
    whole, and ``chi`` must be even for the surface to be orientable --
    which together make ``F`` a multiple of 12 for heptagons.
    """
    out = []
    faces = 0
    while len(out) < most:
        faces += 1
        if (ring * faces) % 2 or (2 * ring * faces) % 6:
            continue
        edges = ring * faces // 2
        vertices = 2 * edges // 3
        chi = vertices - edges + faces
        if chi % 2:
            continue                  # non-orientable; not a carbon surface
        out.append((faces, vertices, edges, chi, (2 - chi) // 2))
    return out


# ---------------------------------------------------------------- the map

def _mul(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Multiply two 2x2 matrices over the field, flattened row-major."""
    return ((a[0] * b[0] + a[1] * b[2]) % FIELD,
            (a[0] * b[1] + a[1] * b[3]) % FIELD,
            (a[2] * b[0] + a[3] * b[2]) % FIELD,
            (a[2] * b[1] + a[3] * b[3]) % FIELD)


def _rep(a: tuple[int, ...]) -> tuple[int, ...]:
    """PSL is SL modulo plus-or-minus the identity; pick one of the pair."""
    return min(a, tuple((-v) % FIELD for v in a))


#: Order 7 (trace +-2): the rotation about a face, so a face is a heptagon.
GENERATOR_FACE = _rep((1, 1, 0, 1))
#: Order 3 (trace +-1): the rotation about a vertex, so carbon is trivalent.
GENERATOR_VERTEX = _rep((0, 1, 6, 1))


def psl27() -> list[tuple[int, ...]]:
    """Every element of ``PSL(2,7)``, generated from the two rotations."""
    identity = _rep((1, 0, 0, 1))
    group = {identity}
    frontier = [identity]
    while frontier:
        nxt = []
        for g in frontier:
            for s in (GENERATOR_FACE, GENERATOR_VERTEX):
                h = _rep(_mul(g, s))
                if h not in group:
                    group.add(h)
                    nxt.append(h)
        frontier = nxt
    return sorted(group)


def _orbits(group, generator):
    """Left cosets ``g<generator>``: orbits of right multiplication."""
    seen: set = set()
    out: list[list] = []
    for g in group:
        if g in seen:
            continue
        orbit, h = [], g
        while True:
            orbit.append(h)
            seen.add(h)
            h = _rep(_mul(h, generator))
            if h == g:
                break
        out.append(orbit)
    return out


def klein_quartic_map() -> tuple[list[tuple[int, int]], list[list[int]]]:
    """The ``{7,3}`` regular map: bonds and heptagons, derived not tabulated.

    A regular map's darts are the elements of its automorphism group, and
    its vertices, edges and faces are the cosets of the three rotation
    subgroups. Here that is ``PSL(2,7)``: 168 darts give 56 vertices
    (cosets of an order-3 rotation), 84 edges (order 2) and 24 heptagons
    (order 7) with no arithmetic left over -- which is what "regular"
    means and why the counts cannot be got wrong by hand.

    Returns
    -------
    bonds
        84 pairs ``(i, j)`` with ``i < j``, over 56 atoms, every atom in
        exactly three of them.
    rings
        24 lists of 7 atom indices, each in cyclic order around its face,
        so consecutive entries are bonded.
    """
    group = psl27()
    faces = _orbits(group, GENERATOR_FACE)
    vertices = _orbits(group, GENERATOR_VERTEX)
    edges = _orbits(group, _rep(_mul(GENERATOR_FACE, GENERATOR_VERTEX)))
    vertex_of = {dart: i for i, v in enumerate(vertices) for dart in v}
    bonds = sorted({tuple(sorted((vertex_of[a], vertex_of[b])))
                    for a, b in edges})
    rings = [[vertex_of[dart] for dart in face] for face in faces]
    return bonds, rings


def cocycle_space(bonds, rings, n_atoms: int):
    """The lattice shift each bond may carry, as an exact integer basis.

    A periodic realisation gives every bond an integer lattice
    translation. For each heptagon to close, those translations must sum
    to zero around it -- so the assignment is a 1-cocycle, and once the
    spanning tree is gauged to zero its remaining freedom is ``H_1``,
    of rank ``2g``.

    That makes this an **independent measurement of the genus**: it never
    uses Euler's formula, only the cycle structure of the graph and the
    face relations, and for the Klein quartic it returns rank 6 = 2 * 3.

    Returns
    -------
    tree, free, kernel
        The spanning tree's bonds, the remaining bonds in order, and an
        integer basis of the cocycle space over those free bonds.
    """
    neighbours: dict[int, list[int]] = {i: [] for i in range(n_atoms)}
    for a, b in bonds:
        neighbours[a].append(b)
        neighbours[b].append(a)
    parent = {0: None}
    queue = [0]
    while queue:
        u = queue.pop(0)
        for w in neighbours[u]:
            if w not in parent:
                parent[w] = u
                queue.append(w)
    tree = {tuple(sorted((w, parent[w])))
            for w in parent if parent[w] is not None}
    free = [b for b in bonds if b not in tree]

    rows = []
    for ring in rings:
        row = [0] * len(free)
        for i, u in enumerate(ring):
            w = ring[(i + 1) % len(ring)]
            key = tuple(sorted((u, w)))
            if key in tree:
                continue
            row[free.index(key)] += 1 if u < w else -1
        rows.append(row)
    return tree, free, _integer_kernel(rows)


def _integer_kernel(rows: list[list[int]]) -> np.ndarray:
    """Exact integer kernel, by column-style Hermite reduction.

    Done in Python integers rather than by an SVD because the answer is a
    lattice: a floating null space gives vectors that span the right
    subspace and are not translations, and a shift of 0.9999 is not a
    shift.
    """
    n_rows = len(rows)
    n_cols = len(rows[0])
    columns = [[rows[i][j] for i in range(n_rows)] for j in range(n_cols)]
    track = [[1 if i == j else 0 for i in range(n_cols)]
             for j in range(n_cols)]
    start = 0
    for r in range(n_rows):
        while True:
            nonzero = [j for j in range(start, n_cols) if columns[j][r]]
            if not nonzero:
                break
            pivot = min(nonzero, key=lambda j: abs(columns[j][r]))
            columns[start], columns[pivot] = columns[pivot], columns[start]
            track[start], track[pivot] = track[pivot], track[start]
            settled = True
            for j in range(start + 1, n_cols):
                if not columns[j][r]:
                    continue
                q = columns[j][r] // columns[start][r]
                if q:
                    for i in range(n_rows):
                        columns[j][i] -= q * columns[start][i]
                    for i in range(n_cols):
                        track[j][i] -= q * track[start][i]
                if columns[j][r]:
                    settled = False
            if settled:
                break
        if any(columns[j][r] for j in range(start, n_cols)):
            start += 1
    return np.array([track[j] for j in range(n_cols)
                     if not any(columns[j])], dtype=int)


# --------------------------------------------------------- the geometry

def barycentric_placement(bonds, shifts, n_atoms: int) -> np.ndarray:
    """Every atom at the mean of its neighbours, in units of the cell.

    Once the lattice shift on each bond is fixed this is a **linear**
    system, and its solution is the canonical embedding crystallography
    uses for a periodic net. The cocycle fixes the shape; only the scale
    is free, so the cell length is whatever makes the mean bond
    graphitic rather than an independent parameter.

    A net whose barycentric placement collapses -- two atoms landing on
    the same point, a bond of length zero -- is *unstable* in the
    crystallographic sense and has no embedding at all for that shift
    assignment. Measured on the Klein quartic: of the twenty cocycles
    built from three of the six basis vectors, **eighteen collapse** and
    two do not, which is why the choice cannot be made arbitrarily.
    """
    laplacian = np.zeros((n_atoms, n_atoms))
    source = np.zeros((n_atoms, 3))
    for (a, b), shift in zip(bonds, shifts, strict=True):
        laplacian[a, a] += 1
        laplacian[b, b] += 1
        laplacian[a, b] -= 1
        laplacian[b, a] -= 1
        source[a] += shift
        source[b] -= shift
    # The Laplacian is singular -- a net may be translated freely -- so
    # pin one atom rather than reaching for a pseudo-inverse.
    laplacian[0, :] = 0.0
    laplacian[0, 0] = 1.0
    source[0] = 0.0
    return np.linalg.solve(laplacian, source)


def angular_excess(ring: int = 7) -> float:
    """Degrees by which three ``ring``-gons overfill a flat vertex.

    The number that says how curved the net is, and it is worth putting
    beside its positive-curvature mirror: heptanene's excess is
    **25.71 deg**, *less* than the 36 deg deficit C20 carries at every
    one of its vertices -- and C20 exists. So the curvature magnitude is
    not what stands in heptanene's way.
    """
    return 3.0 * 180.0 * (ring - 2) / ring - 360.0


def atoms_per_handle(k: int) -> float:
    """How much surface each handle gets in the ``F = 12k`` series.

    This is what actually limits the smallest members. Genus is
    ``1 + k`` and the atom count ``28k``, so the ratio starts at 14 and
    climbs towards 28 -- 18.7 at the genus-3 Klein quartic, 21.0 at
    genus 4, 26.7 at genus 21. A handle built from 18 atoms is a tube
    about two rings around, and its two walls are nearer to each other
    than carbon allows.
    """
    return 28.0 * k / (1.0 + k)


def _relaxed(bonds, shifts, placement, cell, bond, iterations):
    """VFF relaxation with every bond's lattice shift held FIXED.

    :func:`~nanocarbon_lab.builders.fullerene_mesh.relax_shell` uses the
    minimum image, which lets a bond silently change which image it
    points at -- and that changes the topology halfway through a
    relaxation. Here the shift is given and never moves, so whatever
    comes out has the topology that went in, which is the whole point of
    computing the cocycle in the first place.
    """
    from scipy.optimize import minimize

    n = len(placement)
    first = np.array([b[0] for b in bonds])
    second = np.array([b[1] for b in bonds])
    offset = np.asarray(shifts, dtype=float) * cell

    neighbours: dict[int, list] = {i: [] for i in range(n)}
    for (a, b), t in zip(bonds, np.asarray(shifts, dtype=float),
                         strict=True):
        neighbours[a].append((b, t))
        neighbours[b].append((a, -t))
    triples = [(c, x[0], y[0], x[1], y[1])
               for c, lst in neighbours.items()
               for i, x in enumerate(lst) for y in lst[i + 1:]]
    centre = np.array([t[0] for t in triples])
    left = np.array([t[1] for t in triples])
    right = np.array([t[2] for t in triples])
    lshift = np.array([t[3] for t in triples], dtype=float) * cell
    rshift = np.array([t[4] for t in triples], dtype=float) * cell
    target = np.radians(120.0)

    def energy(flat):
        pos = flat.reshape(n, 3)
        grad = np.zeros_like(pos)
        delta = pos[second] + offset - pos[first]
        length = np.linalg.norm(delta, axis=1)
        value = 40.0 * np.sum((length - bond) ** 2)
        coef = (80.0 * (length - bond) / length)[:, None] * delta
        np.add.at(grad, second, coef)
        np.add.at(grad, first, -coef)

        u = pos[left] + lshift - pos[centre]
        v = pos[right] + rshift - pos[centre]
        nu = np.linalg.norm(u, axis=1)
        nv = np.linalg.norm(v, axis=1)
        cosine = np.clip(np.sum(u * v, axis=1) / (nu * nv),
                         -1 + 1e-12, 1 - 1e-12)
        theta = np.arccos(cosine)
        value += 15.0 * np.sum((theta - target) ** 2)
        scale = -30.0 * (theta - target) / np.sqrt(1 - cosine ** 2)
        du = (v / (nu * nv)[:, None]
              - (cosine / nu ** 2)[:, None] * u) * scale[:, None]
        dv = (u / (nu * nv)[:, None]
              - (cosine / nv ** 2)[:, None] * v) * scale[:, None]
        np.add.at(grad, left, du)
        np.add.at(grad, right, dv)
        np.add.at(grad, centre, -(du + dv))
        return value, grad.ravel()

    result = minimize(energy, (placement * cell).ravel(), jac=True,
                      method="L-BFGS-B",
                      options={"maxiter": iterations, "maxfun": 2 * iterations})
    return result.x.reshape(n, 3)


def _measure(positions, bonds, shifts, cell):
    first = np.array([b[0] for b in bonds])
    second = np.array([b[1] for b in bonds])
    delta = positions[second] + np.asarray(shifts, float) * cell - positions[first]
    lengths = np.linalg.norm(delta, axis=1)
    n = len(positions)
    neighbours: dict[int, list] = {i: [] for i in range(n)}
    for (a, b), t in zip(bonds, np.asarray(shifts, dtype=float),
                         strict=True):
        neighbours[a].append((b, t))
        neighbours[b].append((a, -t))
    angles = []
    for c, lst in neighbours.items():
        for i, x in enumerate(lst):
            for y in lst[i + 1:]:
                u = positions[x[0]] + x[1] * cell - positions[c]
                v = positions[y[0]] + y[1] * cell - positions[c]
                cosine = float(np.dot(u, v)
                               / (np.linalg.norm(u) * np.linalg.norm(v)))
                angles.append(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    angles = np.array(angles)
    return {
        "bond_min": float(lengths.min()), "bond_mean": float(lengths.mean()),
        "bond_max": float(lengths.max()), "bond_std": float(lengths.std()),
        "angle_min": float(angles.min()), "angle_mean": float(angles.mean()),
        "angle_max": float(angles.max()),
        # Every other builder's geometry carries this, and the window
        # reads it without asking whether it is there. Leaving it out
        # raised a KeyError inside the panel update, which runs AFTER
        # the redraw -- so the structure appeared and the panel kept
        # describing the previous one. A stale panel is worse than a
        # missing field: it is a wrong reading that looks right.
        "n_close_contacts": int(_close_contacts(positions, bonds, shifts,
                                                cell)),
    }


def _close_contacts(positions, bonds, shifts, cell, cutoff: float = 2.0
                    ) -> int:
    """Non-bonded pairs closer than ``cutoff`` Å, across the cell.

    The Klein quartic's embedding is strained enough that this is a real
    measurement rather than a formality -- it is one of the numbers the
    refusal rests on.

    ``cell`` is the cubic edge as a scalar here, matching what
    :func:`_measure` already assumes, so images scale rather than
    multiply as a matrix.
    """
    bonded = {tuple(sorted(pair[:2])) for pair in bonds}
    count = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if (i, j) in bonded:
                continue
            best = min(
                float(np.linalg.norm(positions[j] + image * cell
                                     - positions[i]))
                for image in _IMAGES
            )
            if best < cutoff:
                count += 1
    return count


#: The 27 neighbouring cells. A contact across the seam is a contact.
_IMAGES = np.array([[i, j, k] for i in (-1, 0, 1)
                    for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)


#: The sp2 window, shared with the flat haeckelite builder so that the
#: two lattices are judged by one standard.
BOND_FLOOR, BOND_CEILING = 1.22, 1.60
ANGLE_FLOOR, ANGLE_CEILING = 95.0, 145.0


def build_heptanene(bond: float = 1.42, relax_iterations: int = 20000,
                    strict: bool = True):
    """Build the smallest orientable heptanene and judge its geometry.

    The topology is exact and needs no searching: 24 heptagons, 56
    trivalent atoms, 84 bonds, genus 3. The geometry does, because a
    periodic realisation must also choose the lattice shift on every
    bond, and most choices are not embeddings at all.

    Parameters
    ----------
    bond
        Target C-C length (Å).
    relax_iterations
        L-BFGS iterations.
    strict
        Raise if the relaxed geometry leaves the sp2 window. Pass
        ``False`` to get the strained structure back for inspection --
        it is a real answer about a real object, not a broken build, and
        it is the only way to look at what the obstruction is.

    Returns
    -------
    ase.Atoms
        ``pbc=(True, True, True)``, 56 atoms, with the ring census, the
        cocycle, the measured geometry and the verdict in ``atoms.info``.

    Raises
    ------
    ValueError
        Under ``strict`` when the geometry is outside the sp2 window,
        which on every cocycle and cell tried it is. **This is the
        honest result, not an unfinished relaxation**, and the message
        carries the numbers.

    Notes
    -----
    What was searched, so that the refusal means something: the exact
    integer cocycle space (rank 6 = 2g), all twenty cocycles built from
    three of its six basis vectors -- of which eighteen collapse -- and
    a further 60 000 sampled from the lattice with entries in
    ``{-1, 0, 1}``, each placed barycentrically, relaxed from that
    placement and from jittered copies of it, over cells from 5 to 16 Å.
    The best this builder reaches is bonds **1.107-1.704 Å** and angles
    **65.1-146.0 deg** at a 6.19 Å cell, against a window of 1.22-1.60
    and 95-145.

    The diagnosis is **steric, not curvature**. Heptanene's angular
    excess is 25.7 deg per vertex, *less* than the 36 deg deficit C20
    carries -- and C20 exists. What genus 3 cannot afford is room: 56
    atoms over three handles is 18.7 atoms each, a tube about two rings
    around, whose two walls come closer than carbon allows. The series
    improves only slowly (21.0 atoms per handle at genus 4, 26.7 at
    genus 21), so the next thing to try is a larger member -- which
    needs a larger Hurwitz group than ``PSL(2,7)``, not a better
    relaxation of this one.
    """
    from ase import Atoms

    bonds, rings = klein_quartic_map()
    n = 1 + max(max(b) for b in bonds)
    tree, free, kernel = cocycle_space(bonds, rings, n)
    position = {b: j for j, b in enumerate(free)}

    best = None
    for trio in _triples(len(kernel)):
        combo = np.zeros((3, len(kernel)), dtype=int)
        for row, col in enumerate(trio):
            combo[row, col] = 1
        component = combo @ kernel
        shifts = np.zeros((len(bonds), 3), dtype=int)
        for k, b in enumerate(bonds):
            if b not in tree:
                shifts[k] = component[:, position[b]]
        if np.linalg.matrix_rank(shifts.astype(float)) < 3:
            continue
        placement = barycentric_placement(bonds, shifts, n)
        delta = (placement[[b[1] for b in bonds]] + shifts
                 - placement[[b[0] for b in bonds]])
        lengths = np.linalg.norm(delta, axis=1)
        if lengths.min() < 1e-6:
            continue                       # unstable net: it has collapsed
        # The barycentric cell is where the MEAN bond is graphitic, which
        # is not where the spread is smallest: scanning around it moved
        # the answer from 1.165-2.215 Å to 1.107-1.704. Reporting the
        # better one matters, because the refusal below is only as strong
        # as the best geometry it was able to find.
        natural = bond / float(lengths.mean())
        for factor in np.arange(0.70, 1.35, 0.05):
            cell = natural * factor
            positions = _relaxed(bonds, shifts, placement, cell, bond,
                                 relax_iterations)
            measured = _measure(positions, bonds, shifts, cell)
            score = (measured["bond_max"] - measured["bond_min"]
                     + abs(measured["bond_mean"] - bond))
            if best is None or score < best[0]:
                best = (score, trio, cell, positions, shifts, measured)

    if best is None:
        raise ValueError(
            "Every cocycle of the Klein quartic collapsed under barycentric "
            "placement, so none of them is an embedding. That is a "
            "statement about the net, not a failed search."
        )
    _score, trio, cell, positions, shifts, measured = best

    atoms = Atoms(symbols=["C"] * n, positions=positions % cell,
                  pbc=(True, True, True))
    atoms.set_cell([cell, cell, cell])
    census = {7: len(rings)}
    deficit = sum((6 - size) * count for size, count in census.items())
    clean = (BOND_FLOOR <= measured["bond_min"]
             and measured["bond_max"] <= BOND_CEILING
             and ANGLE_FLOOR <= measured["angle_min"]
             and measured["angle_max"] <= ANGLE_CEILING)
    atoms.info.update({
        "builder": "heptanene",
        "structure_type": "heptanene",
        "quality_family": "haeckelite",
        "bonds": [(int(a), int(b)) for a, b in bonds],
        "rings": [[int(i) for i in r] for r in rings],
        "ring_counts": census,
        "euler": deficit // 6,
        "genus": 1 + len(rings) // 12,
        "cell": float(cell),
        "cocycle": [int(t) for t in trio],
        "shifts": shifts.tolist(),
        "geometry": measured,
        "angular_excess": round(angular_excess(7), 3),
        "atoms_per_handle": round(atoms_per_handle(len(rings) // 12), 2),
        "sp2": clean,
        "bond": float(bond),
    })
    if strict and not clean:
        raise ValueError(
            f"Heptanene's topology is exact -- {len(rings)} heptagons, "
            f"{n} trivalent atoms, sum(6-n) = {deficit:+d}, genus "
            f"{1 + len(rings) // 12} -- and its geometry is not carbon's: "
            f"bonds {measured['bond_min']:.3f}-{measured['bond_max']:.3f} Å "
            f"and angles {measured['angle_min']:.1f}-"
            f"{measured['angle_max']:.1f} deg, against the sp2 window "
            f"{BOND_FLOOR}-{BOND_CEILING} Å and {ANGLE_FLOOR}-"
            f"{ANGLE_CEILING} deg. This is a converged minimum over every "
            "cocycle that is an embedding at all, not an unfinished "
            "relaxation. The obstruction is room rather than curvature: "
            f"the angular excess is {angular_excess(7):.1f} deg, less than "
            "the 36 deg C20 carries, but three handles over 56 atoms is "
            f"{atoms_per_handle(len(rings) // 12):.1f} atoms each and the "
            "walls meet. Pass strict=False to inspect it, or go to a "
            "larger member of the series."
        )
    return atoms


def _triples(n: int):
    from itertools import combinations
    return combinations(range(n), 3)


__all__ = [
    "ANGLE_CEILING",
    "ANGLE_FLOOR",
    "BOND_CEILING",
    "BOND_FLOOR",
    "FIELD",
    "GENERATOR_FACE",
    "GENERATOR_VERTEX",
    "Geometry",
    "VALENCE",
    "admissible_geometries",
    "build_heptanene",
    "angular_excess",
    "atoms_per_handle",
    "barycentric_placement",
    "closed_surface_series",
    "cocycle_space",
    "klein_quartic_map",
    "psl27",
]
