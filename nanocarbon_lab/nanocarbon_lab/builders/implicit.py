"""Implicit (signed-distance) surfaces for junctions and schwarzites.

Everything here returns a scalar field ``f(points) -> values`` whose zero
level set is the carbon surface. That single abstraction covers both
families the builder needs:

* **Junctions** (L, T, Y, X) -- a smooth union of capsules radiating from
  the origin. The smooth-union blend radius is what creates the flared,
  negatively-curved neck at the branch; a hard union would leave a
  crease that no honeycomb can tile cleanly.
* **Schwarzites** -- triply periodic minimal surfaces (Schwarz P, Schwarz
  D, gyroid) written in their standard trigonometric approximations.
  These are the classical templates for "negative curvature carbon":
  where a fullerene closes with 12 pentagons, a schwarzite opens into a
  periodic sponge whose saddle points are tiled with heptagons and
  octagons.

The zero level set is meshed by :mod:`nanocarbon_lab.builders.remesh`,
which does not care where the field came from -- so a user-supplied
callable works exactly like the presets below.

Gaussian curvature sign, and therefore ring size, follows from the
surface itself: convex caps take pentagons, saddles take heptagons, and
the Euler characteristic of whatever surface you mesh fixes the totals
(``sum(6 - ring_size) = 12 * (1 - genus)``). Nothing here needs to know
that -- it falls out of the topology.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

Field = Callable[[np.ndarray], np.ndarray]

JunctionKind = Literal["L", "T", "Y", "X", "cross3d"]
SchwarziteKind = Literal["primitive", "diamond", "gyroid"]


def capsule(start, end, radius: float) -> Field:
    """Signed distance to a capsule (a cylinder with hemispherical ends)."""
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    ab = b - a
    denom = float(ab @ ab)

    def field(points: np.ndarray) -> np.ndarray:
        ap = points - a
        t = np.clip((ap @ ab) / denom, 0.0, 1.0)[..., None]
        return np.linalg.norm(ap - t * ab, axis=-1) - radius

    return field


def smooth_union(first: Field, second: Field, blend: float) -> Field:
    """Polynomial smooth minimum of two fields.

    A plain ``min`` (hard union) leaves a sharp crease where two tubes
    meet; the crease has concentrated negative curvature that a hexagonal
    net cannot tile without collapsing into tiny rings. Blending over a
    finite radius spreads that curvature into a smooth neck, which the
    remesher can cover with a handful of heptagons -- the structure real
    Y-junction nanotubes adopt.
    """

    def field(points: np.ndarray) -> np.ndarray:
        d1, d2 = first(points), second(points)
        h = np.clip(0.5 + 0.5 * (d2 - d1) / blend, 0.0, 1.0)
        return d2 * (1.0 - h) + d1 * h - blend * h * (1.0 - h)

    return field


def _junction_directions(kind: JunctionKind) -> list[np.ndarray]:
    """Unit vectors along which a junction's arms radiate."""
    if kind == "L":
        dirs = [(1, 0, 0), (0, 0, 1)]
    elif kind == "T":
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 0, 1)]
    elif kind == "Y":
        dirs = [
            (np.cos(np.radians(a)), 0.0, np.sin(np.radians(a)))
            for a in (90, 210, 330)
        ]
    elif kind == "X":
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
    elif kind == "cross3d":
        dirs = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    else:
        raise ValueError(
            f"Unknown junction kind {kind!r}; expected one of "
            "'L', 'T', 'Y', 'X', 'cross3d'."
        )
    return [np.array(d, dtype=float) / np.linalg.norm(d) for d in dirs]


def junction_field(
    kind: JunctionKind,
    tube_radius: float = 6.0,
    arm_length: float = 22.0,
    blend: float = 4.0,
) -> tuple[Field, float]:
    """Field for a capped multi-arm nanotube junction.

    Parameters
    ----------
    kind
        ``"L"`` (2 arms, 90 deg), ``"T"`` (3 arms), ``"Y"`` (3 arms at
        120 deg), ``"X"`` (4 arms), ``"cross3d"`` (6 arms along +-x/y/z).
    tube_radius
        Radius of each arm in Å.
    arm_length
        Distance from the junction centre to each arm's tip, in Å.
    blend
        Smooth-union radius (Å). Larger values give a rounder, more
        gently curved neck; too small reintroduces a crease the
        honeycomb cannot tile.

    Returns
    -------
    (field, extent)
        ``extent`` is the half-width of a bounding box that safely
        contains the whole surface. Meshing inside a smaller box clips
        the arms and leaves boundary edges -- an open, non-manifold mesh
        whose dual is not a valid carbon network.
    """
    if tube_radius <= 0 or arm_length <= 0 or blend <= 0:
        raise ValueError("tube_radius, arm_length and blend must be positive.")

    directions = _junction_directions(kind)
    field = capsule((0.0, 0.0, 0.0), directions[0] * arm_length, tube_radius)
    for direction in directions[1:]:
        field = smooth_union(
            field,
            capsule((0.0, 0.0, 0.0), direction * arm_length, tube_radius),
            blend,
        )
    # Clear the arm tips (arm_length + radius) plus a margin for the grid.
    extent = arm_length + tube_radius + max(6.0, blend)
    return field, extent


def schwarzite_field(
    kind: SchwarziteKind = "primitive",
    cell: float = 30.0,
    thickness: float = 0.0,
) -> tuple[Field, float]:
    """Field for a triply periodic minimal surface (a schwarzite template).

    These are the standard trigonometric approximations to the Schwarz P
    ("primitive"), Schwarz D ("diamond") and gyroid surfaces. Meshing one
    period gives a sponge-like carbon network of negative Gaussian
    curvature -- heptagons and octagons instead of the pentagons that
    close a fullerene.

    Parameters
    ----------
    kind
        Which surface family.
    cell
        Length of one period in Å.
    thickness
        Shifts the level set off zero, thinning or thickening the
        channels. Zero is the minimal surface itself.

    Returns
    -------
    (field, extent)
        ``extent`` is the half-width of a box holding roughly one period.

    Notes
    -----
    The mesh produced here is a **finite fragment**, closed off by the
    box, not a periodic cell: the builder caps it so the result is a
    genuine closed molecule. Its Euler characteristic (and therefore its
    heptagon/octagon budget) is measured from the mesh rather than
    assumed.
    """
    k = 2.0 * np.pi / cell

    def field(points: np.ndarray) -> np.ndarray:
        x, y, z = points[..., 0] * k, points[..., 1] * k, points[..., 2] * k
        if kind == "primitive":
            value = np.cos(x) + np.cos(y) + np.cos(z)
        elif kind == "diamond":
            value = (
                np.sin(x) * np.sin(y) * np.sin(z)
                + np.sin(x) * np.cos(y) * np.cos(z)
                + np.cos(x) * np.sin(y) * np.cos(z)
                + np.cos(x) * np.cos(y) * np.sin(z)
            )
        elif kind == "gyroid":
            value = (
                np.sin(x) * np.cos(y)
                + np.sin(y) * np.cos(z)
                + np.sin(z) * np.cos(x)
            )
        else:
            raise ValueError(
                f"Unknown schwarzite kind {kind!r}; expected 'primitive', "
                "'diamond' or 'gyroid'."
            )
        return value - thickness

    return field, cell * 0.5


def normalize_to_distance(field: Field, eps: float = 1e-3) -> Field:
    """Rescale a field into an approximate signed distance (``f / |grad f|``).

    Combining fields with ``min``/``max`` only behaves if they share
    units. The triply periodic surfaces are trigonometric: their values
    run over roughly [-3, 3] regardless of the cell size, while a ball's
    signed distance is measured in Å. Intersecting the two directly
    produced a surface that ran straight through the clipping sphere and
    out of the sampling box, giving hundreds of boundary edges. Dividing
    by the gradient magnitude is the standard first-order fix: it makes
    the zero level set unchanged but the values near it read as distance.
    """

    def normalised(points: np.ndarray) -> np.ndarray:
        value = field(points)
        grad = np.empty(points.shape[:-1] + (3,))
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = eps
            grad[..., axis] = (field(points + offset) - field(points - offset)) / (
                2 * eps
            )
        magnitude = np.linalg.norm(grad, axis=-1)
        magnitude = np.where(magnitude < 1e-9, 1e-9, magnitude)
        return value / magnitude

    return normalised


def intersect_with_ball(field: Field, radius: float, softness: float = 0.0) -> Field:
    """Clip a field to a ball, so an infinite surface becomes a finite blob.

    Triply periodic surfaces extend forever; intersecting with a sphere
    yields a closed fragment that can be turned into an actual molecule.
    ``softness`` blends the clip so the rim is rounded rather than a hard
    edge (a hard edge tiles badly, exactly as for junction creases).

    ``field`` must be in distance units -- pass it through
    :func:`normalize_to_distance` first if it is not, or the clip will not
    hold and the fragment will run out of the sampling box.
    """

    def clipped(points: np.ndarray) -> np.ndarray:
        ball = np.linalg.norm(points, axis=-1) - radius
        inner = field(points)
        if softness <= 0:
            return np.maximum(inner, ball)
        # Polynomial smooth max. The blend must fall back to `ball` where
        # the ball is the binding constraint (h -> 0 far outside it);
        # mixing the two the other way round leaves the clip inert and the
        # surface simply runs out of the sampling box.
        h = np.clip(0.5 - 0.5 * (ball - inner) / softness, 0.0, 1.0)
        return ball * (1.0 - h) + inner * h + softness * h * (1.0 - h)

    return clipped


def gradient(field: Field, points: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference gradient of a field, used to project onto it."""
    grad = np.empty_like(points)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = eps
        grad[:, axis] = (field(points + offset) - field(points - offset)) / (2 * eps)
    return grad


def project_to_surface(
    field: Field, points: np.ndarray, iterations: int = 3
) -> np.ndarray:
    """Newton-project points onto the zero level set of ``field``.

    Each step moves a point by ``-f / |grad f|`` along the gradient. Used
    after every smoothing pass in the remesher so vertices stay on the
    surface instead of drifting off it (unconstrained Laplacian
    smoothing shrinks a closed surface toward a point).
    """
    result = points.copy()
    for _ in range(max(1, iterations)):
        value = field(result)
        grad = gradient(field, result)
        norm_sq = np.sum(grad * grad, axis=1)
        norm_sq = np.where(norm_sq < 1e-12, 1e-12, norm_sq)
        result -= (value / norm_sq)[:, None] * grad
    return result
