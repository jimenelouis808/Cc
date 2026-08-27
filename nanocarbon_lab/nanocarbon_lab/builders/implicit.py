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


def tube_along_path(
    path: np.ndarray,
    radius: float,
    sample_spacing: float = 0.4,
    margin: float | None = None,
) -> tuple[Field, np.ndarray, np.ndarray]:
    """Signed distance to a tube of constant ``radius`` following ``path``.

    This is what lets an arbitrarily curved tube -- a coil, an S-bend, a
    random meander -- go through the *implicit* route instead of being
    swept geometrically. The difference is not cosmetic. Sweeping bends a
    finished all-hexagon lattice, so the outer wall is simply stretched:
    at 6.5% path strain that means 1.51 Å bonds, and no amount of
    relaxation removes it, because a pure-hexagon tube bent onto an arc
    *must* have a longer outer wall. Meshing the bent surface instead lets
    the remesher pick the ring sizes the curvature actually calls for --
    pentagons on the compressed inner wall, heptagons on the stretched
    outer wall -- which is how real bent and coiled nanotubes relieve the
    strain, and the bonds come back to graphitic length.

    Distance is evaluated against a dense point sampling of the path
    rather than against its segments. Resampling at ``sample_spacing``
    bounds the error: a query point at distance ``d`` from the true tube
    axis sees at worst ``sqrt(d^2 + (spacing/2)^2) - d``, which for a 6 Å
    tube at 0.4 Å spacing is under 0.004 Å -- three orders of magnitude
    below a bond, and far cheaper than an exact segment query over a
    hundred-segment path at every one of half a million grid points.

    Parameters
    ----------
    path
        ``(n, 3)`` polyline of centreline points in Å, in order. It is
        resampled internally, so the input spacing does not matter as long
        as it already resolves the shape.
    radius
        Tube radius in Å.
    sample_spacing
        Resampling step along the path (Å). Must be well under ``radius``.
    margin
        Extra padding on the returned bounding box beyond
        ``radius``. Defaults to ``max(4, radius)``; the box **must** clear
        the surface or marching cubes returns an open mesh.

    Returns
    -------
    (field, lower, upper)
        The field, and the corner points of an axis-aligned box that
        safely contains the whole tube. The box is deliberately not a
        cube: a coil is a wide, flat object, and sampling it inside a cube
        would spend most of the grid on empty space.

    Raises
    ------
    ValueError
        For fewer than two path points, a non-positive radius, or a
        sampling step that does not resolve the tube.
    """
    from scipy.spatial import cKDTree

    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError("path must be an (n, 3) array with n >= 2.")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if sample_spacing <= 0 or sample_spacing > 0.5 * radius:
        raise ValueError(
            f"sample_spacing={sample_spacing} must be positive and well below "
            f"radius={radius}; use at most {0.5 * radius:.2f}."
        )

    # Resample to uniform arc length so the KD-tree covers the path evenly
    # however unevenly the caller spaced their control points.
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(arc[-1])
    if total <= 0:
        raise ValueError("path has zero length.")
    n_samples = max(2, int(np.ceil(total / sample_spacing)) + 1)
    targets = np.linspace(0.0, total, n_samples)
    samples = np.stack(
        [np.interp(targets, arc, path[:, axis]) for axis in range(3)], axis=-1
    )
    tree = cKDTree(samples)

    def field(points: np.ndarray) -> np.ndarray:
        flat = points.reshape(-1, 3)
        distance, _ = tree.query(flat, k=1)
        return (distance - radius).reshape(points.shape[:-1])

    pad = radius + (max(4.0, radius) if margin is None else margin)
    return field, samples.min(axis=0) - pad, samples.max(axis=0) + pad


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
    field: Field, points: np.ndarray, iterations: int = 3,
    max_step: float | None = None,
) -> np.ndarray:
    """Newton-project points onto the zero level set of ``field``.

    Each step moves a point by ``-f / |grad f|`` along the gradient. Used
    after every smoothing pass in the remesher so vertices stay on the
    surface instead of drifting off it (unconstrained Laplacian
    smoothing shrinks a closed surface toward a point).

    Parameters
    ----------
    field, points, iterations
        The field, the points to project, and how many Newton steps.
    max_step
        Cap on how far a single step may move a point (Å). Leave ``None``
        for an unlimited step.

        The cap matters wherever a surface has two sheets facing each
        other across a gap -- neighbouring turns of a coil, arms of a
        junction folded back, the necks of a small schwarzite cell.
        "Nearest point on the surface" is not the same as "nearest point
        on *this* sheet", so once smoothing nudges a vertex past the
        midline of the gap, an unlimited Newton step lands it on the
        facing sheet. The mesh stays closed and keeps its genus, so
        nothing downstream notices; the damage only shows up hundreds of
        steps later as a patch of carbon fused to the wall one full turn
        away. Capping the step below half the gap makes that jump
        impossible while leaving ordinary projection (which moves a
        fraction of an edge length) untouched.
    """
    result = points.copy()
    for _ in range(max(1, iterations)):
        value = field(result)
        grad = gradient(field, result)
        norm_sq = np.sum(grad * grad, axis=1)
        norm_sq = np.where(norm_sq < 1e-12, 1e-12, norm_sq)
        step = -(value / norm_sq)[:, None] * grad
        if max_step is not None:
            length = np.linalg.norm(step, axis=1)
            scale = np.where(length > max_step, max_step / np.maximum(length, 1e-12), 1.0)
            step *= scale[:, None]
        result += step
    return result
