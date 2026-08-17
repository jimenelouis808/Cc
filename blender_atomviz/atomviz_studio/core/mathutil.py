"""Pure geometry helpers: bounding volumes, camera framing, arc generation.

Everything here works on plain ``(x, y, z)`` tuples so it can be unit tested
without Blender. The Blender-facing modules convert to ``mathutil.Vector``
only at the very last step.
"""

from __future__ import annotations

import math
import random

Vec3 = tuple[float, float, float]


# --------------------------------------------------------------------------
# Vector basics
# --------------------------------------------------------------------------
def add(a: Vec3, b: Vec3) -> Vec3:
    """Component-wise sum."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    """Component-wise difference ``a - b``."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a: Vec3, factor: float) -> Vec3:
    """Multiply every component by *factor*."""
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def length(a: Vec3) -> float:
    """Euclidean norm."""
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def normalize(a: Vec3) -> Vec3:
    """Unit vector; returns ``(0, 0, 1)`` for a zero-length input."""
    n = length(a)
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product ``a x b``."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: Vec3, b: Vec3) -> float:
    """Dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    """Linear interpolation between two points."""
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def perpendicular(a: Vec3) -> Vec3:
    """Return an arbitrary unit vector orthogonal to *a*."""
    reference = (0.0, 0.0, 1.0) if abs(normalize(a)[2]) < 0.9 else (1.0, 0.0, 0.0)
    return normalize(cross(a, reference))


# --------------------------------------------------------------------------
# Bounding volumes
# --------------------------------------------------------------------------
def bbox(points: object) -> tuple[Vec3, Vec3]:
    """Axis-aligned bounding box of *points* as ``(min_corner, max_corner)``.

    Raises:
        ValueError: if *points* is empty.
    """
    pts = list(points)  # type: ignore[arg-type]
    if not pts:
        raise ValueError("bbox() needs at least one point")
    xs, ys, zs = zip(*pts)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def bbox_center(low: Vec3, high: Vec3) -> Vec3:
    """Centre of an axis-aligned bounding box."""
    return lerp(low, high, 0.5)


def bbox_size(low: Vec3, high: Vec3) -> Vec3:
    """Extent of an axis-aligned bounding box along each axis."""
    return sub(high, low)


def bounding_radius(points: object, center: Vec3 | None = None) -> float:
    """Radius of a sphere centred on *center* enclosing all *points*."""
    pts = list(points)  # type: ignore[arg-type]
    if not pts:
        raise ValueError("bounding_radius() needs at least one point")
    if center is None:
        center = bbox_center(*bbox(pts))
    return max(length(sub(p, center)) for p in pts)


# --------------------------------------------------------------------------
# Camera framing
# --------------------------------------------------------------------------
def sensor_fov(focal_mm: float, sensor_mm: float = 36.0) -> float:
    """Horizontal/vertical field of view in radians for a given focal length."""
    if focal_mm <= 0.0:
        raise ValueError("focal length must be positive")
    return 2.0 * math.atan(sensor_mm / (2.0 * focal_mm))


def fit_distance(radius: float, fov: float, margin: float = 1.15) -> float:
    """Distance at which a sphere of *radius* fills a *fov* view.

    Args:
        radius: Bounding-sphere radius of the subject.
        fov: Field of view in radians (use the **narrower** of the two axes).
        margin: Extra breathing room; ``1.0`` means "touching the frame edge".

    Returns:
        Distance from the subject centre to the camera.
    """
    half = max(1e-4, min(math.pi / 2 - 1e-4, fov / 2.0))
    return (radius / math.sin(half)) * max(1.0, margin)


def orbit_position(center: Vec3, distance: float, azimuth_deg: float, elevation_deg: float) -> Vec3:
    """Point on a sphere around *center*.

    ``azimuth=0, elevation=0`` sits on ``-Y`` (Blender's front view);
    azimuth rotates counter-clockwise seen from ``+Z``.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(max(-89.9, min(89.9, elevation_deg)))
    horizontal = distance * math.cos(el)
    offset = (
        horizontal * math.sin(az),
        -horizontal * math.cos(az),
        distance * math.sin(el),
    )
    return add(center, offset)


def look_at_euler(location: Vec3, target: Vec3) -> Vec3:
    """XYZ Euler angles (radians) aiming a Blender camera/spot at *target*.

    Blender cameras and spot lights look down their local ``-Z`` with ``+Y``
    up, which is exactly the ``to_track_quat('-Z', 'Y')`` convention; this is
    the closed-form equivalent so it can be tested without ``mathutils``.
    """
    d = normalize(sub(target, location))
    rot_x = math.acos(max(-1.0, min(1.0, -d[2])))
    rot_z = math.atan2(d[1], d[0]) - math.pi / 2.0
    return (rot_x, 0.0, rot_z)


def frame_camera(
    points: object,
    focal_mm: float = 50.0,
    azimuth_deg: float = 35.0,
    elevation_deg: float = 18.0,
    margin: float = 1.25,
    aspect: float = 1.0,
    sensor_mm: float = 36.0,
) -> tuple[Vec3, Vec3, Vec3, float]:
    """Compute a camera transform that frames *points*.

    Args:
        points: World-space positions of the subject.
        focal_mm: Camera focal length.
        azimuth_deg: Orbit angle around ``+Z``.
        elevation_deg: Height above the horizon, in degrees.
        margin: Framing margin (``1.0`` = tight).
        aspect: Render aspect ratio ``width / height``; portrait covers
            (``aspect < 1``) need the camera pulled back further.
        sensor_mm: Sensor width in millimetres.

    Returns:
        ``(location, rotation_euler, target, focus_distance)``.
    """
    center = bbox_center(*bbox(points))
    radius = max(1e-3, bounding_radius(points, center))
    fov = sensor_fov(focal_mm, sensor_mm)
    if aspect < 1.0:  # portrait: the horizontal FOV is the limiting one
        fov = 2.0 * math.atan(math.tan(fov / 2.0) * aspect)
    distance = fit_distance(radius, fov, margin)
    location = orbit_position(center, distance, azimuth_deg, elevation_deg)
    return location, look_at_euler(location, center), center, distance


def fibonacci_sphere(count: int, radius: float = 1.0) -> list[Vec3]:
    """*count* near-uniformly distributed points on a sphere of *radius*.

    Used to place lasers and rim lights without them clumping together.
    """
    if count <= 0:
        return []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    points: list[Vec3] = []
    for i in range(count):
        y = 1.0 - (2.0 * i) / max(1, count - 1) if count > 1 else 0.0
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden * i
        points.append((math.cos(theta) * r * radius, y * radius, math.sin(theta) * r * radius))
    return points


# --------------------------------------------------------------------------
# Lightning / arc geometry
# --------------------------------------------------------------------------
def fractal_path(
    start: Vec3,
    end: Vec3,
    subdivisions: int = 5,
    chaos: float = 0.16,
    rng: random.Random | None = None,
) -> list[Vec3]:
    """Midpoint-displacement polyline between two points.

    This is the classic recursive lightning construction: every segment is
    split at its midpoint, which is then pushed in a random direction
    perpendicular-ish to the segment. Displacement halves at each level, so
    the result is self-similar and reads as an electric arc.

    Args:
        start: First point.
        end: Last point.
        subdivisions: Number of recursive splits (``5`` -> 33 points).
        chaos: Displacement of the first split as a fraction of the total
            length. ``0`` returns a straight line.
        rng: Seeded :class:`random.Random` for reproducible arcs.

    Returns:
        A list of ``2**subdivisions + 1`` points, ``start`` first, ``end`` last.
    """
    rng = rng or random.Random()
    subdivisions = max(0, min(9, subdivisions))
    points = [start, end]
    span = length(sub(end, start))
    if span < 1e-9:
        return points
    displacement = span * max(0.0, chaos)

    for _ in range(subdivisions):
        refined: list[Vec3] = []
        for a, b in zip(points, points[1:]):
            mid = lerp(a, b, 0.5)
            axis = normalize(sub(b, a))
            u = perpendicular(axis)
            v = cross(axis, u)
            angle = rng.uniform(0.0, 2.0 * math.pi)
            amount = rng.uniform(-1.0, 1.0) * displacement
            offset = add(scale(u, math.cos(angle) * amount), scale(v, math.sin(angle) * amount))
            refined.extend([a, add(mid, offset)])
        refined.append(points[-1])
        points = refined
        displacement *= 0.5
    return points


def branch_points(
    path: object,
    count: int,
    rng: random.Random | None = None,
    min_fraction: float = 0.15,
    max_fraction: float = 0.85,
) -> list[int]:
    """Pick *count* interior indices of *path* to spawn secondary arcs from."""
    pts = list(path)  # type: ignore[arg-type]
    n = len(pts)
    if n < 4 or count <= 0:
        return []
    rng = rng or random.Random()
    low = max(1, int(n * min_fraction))
    high = min(n - 2, int(n * max_fraction))
    if high <= low:
        return []
    candidates = list(range(low, high))
    rng.shuffle(candidates)
    return sorted(candidates[:count])
