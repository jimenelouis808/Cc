"""Geometry: bounding volumes, camera framing and arc generation."""

from __future__ import annotations

import math
import random

import pytest

from atomviz_studio.core import mathutil as M

CUBE = [
    (-1.0, -1.0, -1.0), (1.0, -1.0, -1.0), (1.0, 1.0, -1.0), (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0),
]


def test_bbox_and_center():
    low, high = M.bbox(CUBE)
    assert low == (-1.0, -1.0, -1.0)
    assert high == (1.0, 1.0, 1.0)
    assert M.bbox_center(low, high) == (0.0, 0.0, 0.0)
    assert M.bbox_size(low, high) == (2.0, 2.0, 2.0)
    assert M.bounding_radius(CUBE) == pytest.approx(math.sqrt(3))


def test_bbox_requires_points():
    with pytest.raises(ValueError):
        M.bbox([])
    with pytest.raises(ValueError):
        M.bounding_radius([])


def test_normalize_handles_zero_vector():
    assert M.normalize((0.0, 0.0, 0.0)) == (0.0, 0.0, 1.0)
    assert M.length(M.normalize((3.0, 4.0, 0.0))) == pytest.approx(1.0)


def test_perpendicular_is_orthogonal():
    for vector in [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 2.0, 3.0)]:
        assert M.dot(M.normalize(vector), M.perpendicular(vector)) == pytest.approx(0.0, abs=1e-9)


def test_fit_distance_grows_with_radius_and_narrow_fov():
    fov = M.sensor_fov(50.0)
    assert M.fit_distance(2.0, fov) > M.fit_distance(1.0, fov)
    narrow = M.sensor_fov(200.0)
    assert M.fit_distance(1.0, narrow) > M.fit_distance(1.0, fov)


def test_sensor_fov_rejects_bad_focal():
    with pytest.raises(ValueError):
        M.sensor_fov(0.0)


def test_orbit_position_conventions():
    center = (0.0, 0.0, 0.0)
    front = M.orbit_position(center, 10.0, 0.0, 0.0)
    assert front == pytest.approx((0.0, -10.0, 0.0), abs=1e-9)
    side = M.orbit_position(center, 10.0, 90.0, 0.0)
    assert side == pytest.approx((10.0, 0.0, 0.0), abs=1e-9)
    top = M.orbit_position(center, 10.0, 0.0, 89.9)
    assert top[2] == pytest.approx(10.0, abs=1e-2)


def _apply_euler(euler, vector):
    """Rotate *vector* by an XYZ Euler the way Blender does (Rz @ Ry @ Rx)."""
    rx, ry, rz = euler
    x, y, z = vector
    # Rx
    y, z = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
    # Ry
    x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)
    # Rz
    x, y = x * math.cos(rz) - y * math.sin(rz), x * math.sin(rz) + y * math.cos(rz)
    return (x, y, z)


@pytest.mark.parametrize(
    "location",
    [(0.0, -5.0, 0.0), (5.0, 0.0, 0.0), (0.0, 0.0, 7.0), (3.0, -4.0, 5.0), (-2.0, 6.0, -3.0)],
)
def test_look_at_euler_points_camera_minus_z_at_target(location):
    target = (0.0, 0.0, 0.0)
    euler = M.look_at_euler(location, target)
    forward = _apply_euler(euler, (0.0, 0.0, -1.0))
    expected = M.normalize(M.sub(target, location))
    assert forward == pytest.approx(expected, abs=1e-9)


def test_frame_camera_frames_the_subject():
    location, rotation, center, distance = M.frame_camera(CUBE, focal_mm=50.0, margin=1.2)
    assert center == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert M.length(M.sub(location, center)) == pytest.approx(distance, abs=1e-9)
    forward = _apply_euler(rotation, (0.0, 0.0, -1.0))
    assert forward == pytest.approx(M.normalize(M.sub(center, location)), abs=1e-9)


def test_portrait_aspect_pulls_the_camera_back():
    square = M.frame_camera(CUBE, aspect=1.0)[3]
    portrait = M.frame_camera(CUBE, aspect=2480 / 3508)[3]
    assert portrait > square


def test_fibonacci_sphere_is_uniform_radius():
    points = M.fibonacci_sphere(24, 3.0)
    assert len(points) == 24
    for point in points:
        assert M.length(point) == pytest.approx(3.0, abs=1e-6)
    assert M.fibonacci_sphere(0) == []


def test_fractal_path_endpoints_and_length():
    start, end = (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)
    path = M.fractal_path(start, end, subdivisions=4, chaos=0.2, rng=random.Random(1))
    assert path[0] == start
    assert path[-1] == end
    assert len(path) == 2**4 + 1
    # A displaced arc is always longer than the straight line it spans.
    total = sum(M.length(M.sub(b, a)) for a, b in zip(path, path[1:]))
    assert total > M.length(M.sub(end, start))


def test_fractal_path_is_seeded():
    a = M.fractal_path((0, 0, 0), (5, 5, 0), rng=random.Random(7))
    b = M.fractal_path((0, 0, 0), (5, 5, 0), rng=random.Random(7))
    c = M.fractal_path((0, 0, 0), (5, 5, 0), rng=random.Random(8))
    assert a == b
    assert a != c


def test_fractal_path_degenerate_span():
    path = M.fractal_path((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), subdivisions=3)
    assert path == [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]


def test_zero_chaos_stays_on_the_line():
    path = M.fractal_path((0, 0, 0), (4, 0, 0), subdivisions=3, chaos=0.0, rng=random.Random(3))
    assert all(abs(p[1]) < 1e-12 and abs(p[2]) < 1e-12 for p in path)


def test_branch_points_are_interior_and_seeded():
    path = M.fractal_path((0, 0, 0), (10, 0, 0), subdivisions=4, rng=random.Random(2))
    picks = M.branch_points(path, 3, random.Random(5))
    assert len(picks) == 3
    assert all(0 < index < len(path) - 1 for index in picks)
    assert picks == M.branch_points(path, 3, random.Random(5))
    assert M.branch_points(path, 0) == []
    assert M.branch_points([(0, 0, 0), (1, 1, 1)], 2) == []
