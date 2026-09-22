"""Topological coordinates: positions from the adjacency matrix alone.

Very often the only thing known about a nanostructure is which atom is
bonded to which. István László's review (*Theor Chem Acc* **134**, 104,
2015) collects the methods that turn that into Cartesian coordinates,
and the idea behind all of them is the same: certain eigenvectors of the
adjacency matrix are **bi-lobal** -- deleting the vertices where they
vanish and the edges across which they change sign leaves exactly two
components -- and those behave like the first standing waves on the
surface, so they can be read as angles.

Two cases have a clean answer, and this module implements both:

* **Spherical** (Fowler & Manolopoulos; Pisanski & Shawe-Taylor). Three
  bi-lobal eigenvectors, scaled by ``1/sqrt(lambda_1 - lambda_k)``, are
  the ``x``, ``y``, ``z`` of a fullerene. For most fullerenes they are
  simply eigenvectors 2, 3 and 4.
* **Toroidal** (László et al.). Three are provably *not* enough -- the
  torus always comes out flat from some direction -- and **four** are.
  A torus is the product of two circles, so the four split into two
  degenerate pairs, one giving the angle round the ring and the other
  the angle round the tube.

**Which four was measured here, not assumed.** Taking a toroidal polyhex
this package can already build exactly -- a (5,5) tube bent over 60
periods, 1200 atoms, all hexagons -- throwing its coordinates away and
keeping only its bonds, the bi-lobal eigenvectors in decreasing
eigenvalue order are 1, 2, 13, 14, ... and:

===========  ==========  =================  ==================
pair         eigenvalue  agrees with        agrees with
                         the ring angle     the tube angle
===========  ==========  =================  ==================
(1, 2)       2.9973      **1.0000**         0.0000
(13, 14)     2.8699      0.0000             **1.0000**
(15, 16)     2.8672      0.0019             0.0001
(21, 22)     2.8591      0.0000             0.0000
===========  ==========  =================  ==================

So it is the two highest-eigenvalue *degenerate bi-lobal pairs*, and the
agreement is 1.0000 -- exact up to an offset, not approximate. Every
other pair is orthogonal to both angles. That table is the reason to
believe the recipe, and re-deriving it costs one build.

What this does **not** do is the general case. László is explicit that
there is none: a 1165-atom nanotube junction needs sixteen bi-lobal
eigenvectors, and the method only works for structures related to the
sphere. The general answer is a matrix ``W``, built from a harmonic
potential over first and second neighbours, whose null space holds
``X``, ``Y`` and ``Z`` -- and constructing ``W`` exactly needs the
coordinates it is meant to produce. This package met that wall from the
other side: `heptanene`'s Laplacian has an **8-fold degenerate** first
non-trivial eigenvalue, so no three of its eigenvectors embed it at all.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

#: Two eigenvalues closer than this are treated as degenerate, which is
#: what makes a *pair*. The gap either side is ~0.13 on the test torus,
#: so this is three orders of magnitude clear of the nearest real gap.
DEGENERACY = 1e-6

#: Below this an eigenvector component counts as zero for the bi-lobal
#: test. Machine noise on a 1200-atom adjacency sits near 1e-16.
ZERO = 1e-9


def adjacency_matrix(n_atoms: int, bonds) -> np.ndarray:
    """The 0/1 adjacency matrix of a bond list."""
    matrix = np.zeros((n_atoms, n_atoms))
    for i, j in bonds:
        matrix[int(i), int(j)] = 1.0
        matrix[int(j), int(i)] = 1.0
    return matrix


def is_bilobal(vector: np.ndarray, bonds, n_atoms: int) -> bool:
    """Whether deleting the zeros and the sign-changing edges leaves two
    components.

    This is László's definition verbatim, and it is a *graph* test rather
    than a numerical one: it asks whether the eigenvector has exactly two
    lobes, which is what makes it readable as an angle.
    """
    alive = [i for i in range(n_atoms) if abs(vector[i]) > ZERO]
    position = {vertex: index for index, vertex in enumerate(alive)}
    neighbours = defaultdict(list)
    for i, j in bonds:
        i, j = int(i), int(j)
        if i in position and j in position and vector[i] * vector[j] > 0:
            neighbours[position[i]].append(position[j])
            neighbours[position[j]].append(position[i])
    seen: set[int] = set()
    components = 0
    for start in range(len(alive)):
        if start in seen:
            continue
        components += 1
        if components > 2:
            return False
        stack = [start]
        while stack:
            here = stack.pop()
            if here in seen:
                continue
            seen.add(here)
            stack.extend(neighbours[here])
    return components == 2


def bilobal_spectrum(bonds, n_atoms: int, most: int | None = None):
    """Eigenvalues, eigenvectors and which of them are bi-lobal.

    Sorted by **decreasing** eigenvalue, which is László's convention for
    the adjacency matrix (increasing for the Laplacian).
    """
    matrix = adjacency_matrix(n_atoms, bonds)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(-values)
    values, vectors = values[order], vectors[:, order]
    limit = n_atoms if most is None else min(most, n_atoms)
    flags = [is_bilobal(vectors[:, k], bonds, n_atoms) for k in range(limit)]
    return values, vectors, flags


def bilobal_pairs(values: np.ndarray, flags, tolerance: float = DEGENERACY):
    """Degenerate pairs of bi-lobal eigenvectors, best eigenvalue first.

    A pair is what an angle needs: two eigenvectors of the *same*
    eigenvalue, which behave as the cosine and sine of one harmonic.
    """
    indices = [k for k, flag in enumerate(flags) if flag]
    pairs = []
    used: set[int] = set()
    for a_index, first in enumerate(indices):
        if first in used:
            continue
        for second in indices[a_index + 1:]:
            if second in used:
                continue
            if abs(values[first] - values[second]) <= tolerance:
                pairs.append((first, second))
                used.update((first, second))
                break
    return pairs


def toroidal_coordinates(bonds, n_atoms: int, bond: float = 1.42,
                         most: int | None = None) -> np.ndarray:
    """Place a toroidal graph from four bi-lobal eigenvectors.

    Three are provably not enough -- Graovac et al. showed the torus goes
    flat from some direction -- so the two highest-eigenvalue degenerate
    bi-lobal pairs are used, the first for the angle round the ring and
    the second for the angle round the tube. The two radii are then the
    only free numbers, and they are fitted so the mean bond comes out at
    ``bond``.

    Raises
    ------
    ValueError
        If fewer than two degenerate bi-lobal pairs exist, which means
        the graph is not toroidal in the sense this method needs.
    """
    values, vectors, flags = bilobal_spectrum(bonds, n_atoms, most=most)
    pairs = bilobal_pairs(values, flags)
    if len(pairs) < 2:
        raise ValueError(
            f"found {len(pairs)} degenerate bi-lobal pair(s); a torus needs "
            "two -- one for the ring and one for the tube. This graph is "
            "not toroidal in the sense the method requires."
        )
    (ring_a, ring_b), (tube_a, tube_b) = pairs[0], pairs[1]
    phi = np.arctan2(vectors[:, ring_b], vectors[:, ring_a])
    theta = np.arctan2(vectors[:, tube_b], vectors[:, tube_a])

    # Only the two radii are left, and the bonds fix them. A coarse scan
    # then a refinement: the objective is smooth and one-dimensional once
    # the ratio is set, but it is cheap enough not to be clever about.
    first = np.array([int(i) for i, _ in bonds])
    second = np.array([int(j) for _, j in bonds])

    def place(major: float, minor: float) -> np.ndarray:
        radial = major + minor * np.cos(theta)
        return np.column_stack([radial * np.cos(phi), radial * np.sin(phi),
                                minor * np.sin(theta)])

    def error(scale: np.ndarray) -> float:
        points = place(scale[0], scale[1])
        lengths = np.linalg.norm(points[second] - points[first], axis=1)
        return float(np.mean((lengths - bond) ** 2))

    best = None
    for major in np.linspace(2.0, 400.0, 120):
        for ratio in np.linspace(0.02, 0.45, 60):
            value = error(np.array([major, major * ratio]))
            if best is None or value < best[0]:
                best = (value, major, major * ratio)
    from scipy.optimize import minimize

    refined = minimize(error, np.array([best[1], best[2]]),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-12})
    return place(refined.x[0], refined.x[1])


def spherical_coordinates(bonds, n_atoms: int, bond: float = 1.42,
                          most: int | None = None) -> np.ndarray:
    """Place a sphere-like graph from three bi-lobal eigenvectors.

    Fowler & Manolopoulos's construction: the three are scaled by
    ``1/sqrt(lambda_1 - lambda_k)``, which is what stops the highest
    harmonics dominating, and the whole is then scaled so the mean bond
    is ``bond``.
    """
    values, vectors, flags = bilobal_spectrum(bonds, n_atoms, most=most)
    chosen = [k for k, flag in enumerate(flags) if flag][:3]
    if len(chosen) < 3:
        raise ValueError(
            f"found {len(chosen)} bi-lobal eigenvector(s); a sphere needs "
            "three. This graph is not sphere-like in the sense the method "
            "requires."
        )
    scale = 1.0 / np.sqrt(np.maximum(values[0] - values[chosen], 1e-12))
    points = vectors[:, chosen] * scale
    first = np.array([int(i) for i, _ in bonds])
    second = np.array([int(j) for _, j in bonds])
    lengths = np.linalg.norm(points[second] - points[first], axis=1)
    return points * (bond / float(np.mean(lengths)))


__all__ = ["DEGENERACY", "ZERO", "adjacency_matrix", "bilobal_pairs",
           "bilobal_spectrum", "is_bilobal", "spherical_coordinates",
           "toroidal_coordinates"]
