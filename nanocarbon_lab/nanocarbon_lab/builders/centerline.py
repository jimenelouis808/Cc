"""Curved centrelines for sweeping a nanotube along an arbitrary 3D path.

A straight capped tube is swept onto a space curve so it can meander,
coil or S-bend, which is what makes a cover image read as a nanotube
rather than a rod. Three pieces make that work correctly:

* **Arc-length parameterisation.** The path is resampled so equal steps
  along it cover equal distance. Without this the tube stretches where
  the spline runs fast and bunches where it runs slow.
* **A rotation-minimizing frame** (:func:`rotation_minimizing_frames`),
  not a Frenet frame. The Frenet normal is undefined where curvature
  passes through zero and flips by 180 deg either side of an inflection
  point -- and a random meander is full of inflection points, so a
  Frenet sweep would tear the tube apart at each one.
* **A strain budget.** The outer wall of a bent tube of radius
  ``r_tube`` following a path of curvature ``kappa`` is stretched by
  approximately ``r_tube * kappa``. Left unchecked this is what destroys
  the structure: measured on a 3200-atom tube, a path at ~5% strain
  relaxes to clean sp2 geometry, ~10% is still sound, and ~100% produces
  2.2 Å "bonds" and overlapping atoms. :func:`fit_to_strain_budget`
  therefore scales a path's lateral deviation until it respects a
  physical limit, instead of letting the caller request something a real
  nanotube could not survive.

Real nanotubes buckle into a localised kink past a critical curvature
rather than straining smoothly, so the default budget is deliberately
conservative; the sweep here models the pre-buckling elastic regime.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline

Shape = Literal["straight", "arc", "random", "helix", "s_curve"]

# Outer-wall strain (r_tube * kappa) that still relaxes to clean sp2
# geometry. Measured, not guessed -- see the module docstring.
DEFAULT_MAX_STRAIN = 0.08

# Past this, bonds stretch beyond the sp2 range (measured: 20% strain gives
# 1.63 Å bonds). Structures stay visually intact and are fine for artwork,
# but are no longer physically meaningful -- the builder warns.
ARTISTIC_STRAIN_LIMIT = 0.15


def random_control_points(
    n_points: int,
    amplitude: float,
    rng: np.random.Generator,
    n_modes: int = 3,
) -> np.ndarray:
    """Smooth random meander built from a few low-frequency modes.

    The obvious construction -- a random walk that turns by a random angle
    at each step -- does **not** work here. Its turns point in uncorrelated
    directions and cancel, so the path wiggles at high frequency while
    going essentially straight. Because the strain budget is spent on peak
    *curvature*, those cancelling wiggles consume the whole budget and
    :func:`fit_to_strain_budget` then flattens the path almost to a line:
    the tube comes out straight no matter how high ``amplitude`` is set.

    Instead the lateral offset is a sum of ``n_modes`` sine modes with
    random phases and amplitudes falling as ``1/k``, evaluated
    independently for x and y. Few modes means low curvature per unit of
    lateral travel, so the budget buys visible, coherent bending rather
    than noise.

    Parameters
    ----------
    n_points
        Control points to return (also sets the path's nominal length).
    amplitude
        0-1 scale on the lateral excursion.
    rng
        Seeded generator.
    n_modes
        Sine modes summed per axis. 1-2 gives a single sweeping bend,
        3-4 an organic meander; higher starts to cancel again.

    Returns
    -------
    numpy.ndarray
        ``(n_points, 3)`` control points running along +z.
    """
    n_points = max(4, n_points)
    t = np.linspace(0.0, 1.0, n_points)
    offsets = np.zeros((n_points, 2))
    for axis in range(2):
        for k in range(1, max(1, n_modes) + 1):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            weight = rng.normal() / k
            offsets[:, axis] += weight * np.sin(np.pi * k * t + phase)
    # Zero the ends so the path starts and finishes on the axis, keeping
    # the meander in the middle where it reads clearly.
    offsets -= np.outer(1.0 - t, offsets[0]) + np.outer(t, offsets[-1])
    lateral = amplitude * n_points * 0.30
    return np.stack(
        [offsets[:, 0] * lateral, offsets[:, 1] * lateral, t * n_points], axis=1
    )


def shape_control_points(
    shape: Shape,
    rng: np.random.Generator,
    n_points: int = 9,
    amplitude: float = 1.0,
    turns: float = 1.5,
) -> np.ndarray:
    """Control points for one of the built-in centreline shapes.

    ``straight`` -- a line (no sweep). ``arc`` -- a single planar bend.
    ``s_curve`` -- two opposing bends. ``helix`` -- a coil of ``turns``
    turns. ``random`` -- a seeded meander via :func:`random_control_points`.
    All are returned unit-scaled; amplitude is trimmed later to respect
    the strain budget.
    """
    t = np.linspace(0.0, 1.0, max(4, n_points))
    if shape == "straight":
        return np.stack([np.zeros_like(t), np.zeros_like(t), t * n_points], axis=1)
    if shape == "arc":
        return np.stack([amplitude * t**2 * n_points * 0.25,
                         np.zeros_like(t), t * n_points], axis=1)
    if shape == "s_curve":
        return np.stack([amplitude * np.sin(2 * np.pi * t) * n_points * 0.12,
                         np.zeros_like(t), t * n_points], axis=1)
    if shape == "helix":
        angle = 2 * np.pi * turns * t
        radius = amplitude * n_points * 0.12
        return np.stack([radius * np.cos(angle), radius * np.sin(angle),
                         t * n_points], axis=1)
    if shape == "random":
        return random_control_points(n_points, amplitude, rng)
    raise ValueError(f"Unknown shape {shape!r}.")


def arclength_sampler(
    control_points: np.ndarray, samples: int = 3000
) -> tuple[Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], float]:
    """Fit a cubic spline and return an arc-length sampler plus total length.

    Returns ``(sample, total_length)`` where ``sample(s)`` gives
    ``(positions, unit_tangents)`` at arc-length coordinates ``s``.
    """
    if len(control_points) < 4:
        raise ValueError("Need at least 4 control points for a cubic spline.")
    knots = np.arange(len(control_points), dtype=float)
    spline = CubicSpline(knots, control_points, axis=0)

    dense_t = np.linspace(0.0, len(control_points) - 1, samples)
    dense_p = spline(dense_t)
    steps = np.linalg.norm(np.diff(dense_p, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(cumulative[-1])

    def sample(target_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target_s = np.clip(np.asarray(target_s, dtype=float), 0.0, total)
        t = np.interp(target_s, cumulative, dense_t)
        positions = spline(t)
        tangents = spline(t, 1)
        tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True)
        return positions, tangents

    return sample, total


def rotation_minimizing_frames(
    positions: np.ndarray, tangents: np.ndarray
) -> np.ndarray:
    """Transport a normal along a curve with minimal twist (double reflection).

    Implements the double-reflection method of Wang, Juttler, Zheng and
    Liu (2008). The alternative -- a Frenet frame -- is undefined at
    inflection points and flips 180 deg across them, which on a meandering
    path shears the swept tube apart; this transports the normal smoothly
    instead, and is exact to machine precision on orthonormality.

    Parameters
    ----------
    positions, tangents
        ``(n, 3)`` samples along the curve; ``tangents`` must be unit.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` unit normals, each perpendicular to its tangent. The
        binormal is ``cross(tangent, normal)``.
    """
    n = len(positions)
    normals = np.zeros_like(positions)

    seed = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(seed, tangents[0])) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    first = seed - np.dot(seed, tangents[0]) * tangents[0]
    normals[0] = first / np.linalg.norm(first)

    for i in range(n - 1):
        v1 = positions[i + 1] - positions[i]
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-18:
            normals[i + 1] = normals[i]
            continue
        reflected_n = normals[i] - (2.0 / c1) * np.dot(v1, normals[i]) * v1
        reflected_t = tangents[i] - (2.0 / c1) * np.dot(v1, tangents[i]) * v1
        v2 = tangents[i + 1] - reflected_t
        c2 = float(np.dot(v2, v2))
        if c2 < 1e-18:
            normals[i + 1] = reflected_n
        else:
            normals[i + 1] = reflected_n - (2.0 / c2) * np.dot(v2, reflected_n) * v2
        normals[i + 1] /= np.linalg.norm(normals[i + 1])
    return normals


def max_curvature(control_points: np.ndarray, total_length: float) -> float:
    """Peak curvature (1/Å) of the path rescaled to ``total_length``."""
    sample, total = arclength_sampler(control_points)
    if total <= 0:
        return 0.0
    scaled = control_points * (total_length / total)
    sample, total = arclength_sampler(scaled)
    s = np.linspace(0.0, total, 1500)
    positions, _ = sample(s)
    d1 = np.gradient(positions, s, axis=0)
    d2 = np.gradient(d1, s, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    speed = np.where(speed < 1e-12, 1e-12, speed)
    return float((np.linalg.norm(np.cross(d1, d2), axis=1) / speed**3).max())


def fit_to_strain_budget(
    control_points: np.ndarray,
    total_length: float,
    tube_radius: float,
    max_strain: float = DEFAULT_MAX_STRAIN,
    iterations: int = 24,
) -> tuple[np.ndarray, float]:
    """Flatten a path toward straight until it respects a strain budget.

    Outer-wall strain is ``tube_radius * kappa``. Control points are
    interpolated toward the straight line joining the endpoints -- a
    factor of 0 gives a straight path (zero curvature), 1 the original --
    and bisection finds the largest factor whose peak strain stays within
    ``max_strain``. This is why a "wild" random path yields a bent but
    intact nanotube rather than one with 2 Å bonds: the request is scaled
    to what the lattice can actually survive.

    Returns
    -------
    (control_points, achieved_strain)
    """
    if max_strain <= 0:
        raise ValueError("max_strain must be positive.")
    start, end = control_points[0], control_points[-1]
    t = np.linspace(0.0, 1.0, len(control_points))[:, None]
    straight = start + t * (end - start)

    def strain_of(factor: float) -> float:
        candidate = straight + factor * (control_points - straight)
        return tube_radius * max_curvature(candidate, total_length)

    if strain_of(1.0) <= max_strain:
        return control_points, strain_of(1.0)

    low, high = 0.0, 1.0
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        if strain_of(mid) <= max_strain:
            low = mid
        else:
            high = mid
    fitted = straight + low * (control_points - straight)
    return fitted, strain_of(low)


def sweep_along_path(
    positions: np.ndarray,
    control_points: np.ndarray,
    frame_samples: int = 2000,
) -> np.ndarray:
    """Bend a straight, z-aligned structure onto a 3D centreline.

    The path is rescaled so its arc length equals the structure's axial
    span, so the tube is neither stretched nor compressed along its axis;
    each atom's transverse offset ``(x, y)`` is then re-expressed in the
    rotation-minimizing frame at the matching arc length.

    Parameters
    ----------
    positions
        ``(n, 3)`` straight structure, long axis along z.
    control_points
        Path control points (any scale -- normalised internally).
    frame_samples
        Frames computed along the path before per-atom interpolation.

    Returns
    -------
    numpy.ndarray
        ``(n, 3)`` swept positions.
    """
    z = positions[:, 2]
    z_min, z_max = z.min(), z.max()
    length = float(z_max - z_min)
    if length <= 0:
        return positions.copy()

    _, raw_total = arclength_sampler(control_points)
    if raw_total <= 0:
        return positions.copy()
    sample, total = arclength_sampler(control_points * (length / raw_total))

    grid = np.linspace(0.0, total, frame_samples)
    path_p, path_t = sample(grid)
    normals = rotation_minimizing_frames(path_p, path_t)
    binormals = np.cross(path_t, normals)

    fractional = np.interp(z - z_min, grid, np.arange(frame_samples))
    lower = np.clip(np.floor(fractional).astype(int), 0, frame_samples - 2)
    weight = (fractional - lower)[:, None]

    def blend(field: np.ndarray, normalise: bool) -> np.ndarray:
        out = field[lower] * (1.0 - weight) + field[lower + 1] * weight
        if normalise:
            out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out

    centre = blend(path_p, normalise=False)
    normal = blend(normals, normalise=True)
    binormal = blend(binormals, normalise=True)
    return centre + positions[:, 0:1] * normal + positions[:, 1:2] * binormal
