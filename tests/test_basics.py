import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from checks.xrd_clearance import footprint, max_obstacle_height, min_two_theta_for_footprint  # noqa: E402
from params import COUPON, ORING, XRD, hole, oring_gland  # noqa: E402

R, DIV = XRD["goniometer_radius"], XRD["divergence_deg"]


def test_oring_gland():
    assert oring_gland(2.0, "face") == {"depth": 1.5, "width": 3.0}
    assert oring_gland(2.0, "radial") == {"depth": 1.6, "width": 2.8}
    with pytest.raises(ValueError):
        oring_gland(2.0, "axial")


def test_hole_offset():
    assert hole(3.0) > 3.0


def test_footprint_small_angle_limit():
    # huella ≈ R·α / sin θ para α pequeño
    approx = R * math.radians(DIV) / math.sin(math.radians(10))
    assert footprint(20, R, DIV) == pytest.approx(approx, rel=0.01)


def test_footprint_shrinks_with_angle():
    assert footprint(10, R, DIV) > footprint(20, R, DIV) > footprint(40, R, DIV)


def test_obstacle_height():
    assert max_obstacle_height(1.0, 20, R, DIV) == 0.0          # dentro de la huella
    hs = [max_obstacle_height(d, 20, R, DIV) for d in (10, 20, 30)]
    assert hs == sorted(hs) and hs[0] > 0


def test_min_two_theta_inverts_footprint():
    tt = min_two_theta_for_footprint(15.0, R, DIV)
    assert footprint(tt, R, DIV) == pytest.approx(15.0, abs=0.01)


def test_coupon_builds():
    from coupons.tolerance_coupon import build, layout

    solid = build().val()
    assert solid.isValid()
    lay = layout()
    assert len(lay["holes"]) == len(COUPON["hole_nominals"]) * len(COUPON["hole_offsets"])
    bb = solid.BoundingBox()
    assert bb.zmax == pytest.approx(0.0, abs=1e-6)
    assert bb.zmin == pytest.approx(-COUPON["thickness"], abs=1e-6)
    # las ranuras no rompen el fondo de la placa
    assert max(d for _, (_, _, d) in lay["rings"]) < COUPON["thickness"]
    assert ORING["face_cs"] > 0
