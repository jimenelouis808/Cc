import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import params as P  # noqa: E402
from checks import walls  # noqa: E402
from checks.raman_clearance import collision_volume  # noqa: E402
from modules import base_flat, chamber_ring  # noqa: E402
from modules._geom import bolt_stack, sample_seal, variant  # noqa: E402


@pytest.fixture(scope="module")
def ring():
    return chamber_ring.build()


@pytest.fixture(scope="module")
def base():
    return base_flat.build()


def test_parts_are_valid_single_solids(ring, base):
    for part in (ring, base):
        solid = part.val()
        assert solid.isValid()
        assert len(solid.Solids()) == 1


def test_reference_plane(ring, base):
    """z = 0 es la cara del WE: la base termina ahí y el anillo empieza ahí."""
    assert base.val().BoundingBox().zmax == pytest.approx(0.0, abs=1e-6)
    assert ring.val().BoundingBox().zmin == pytest.approx(0.0, abs=1e-6)
    assert ring.val().BoundingBox().zmax == pytest.approx(P.CHAMBER_RING["h"], abs=1e-6)


def test_exposed_area_is_oring_id():
    s = sample_seal(P)
    assert s["bore_d"] == P.WE_FLAT["exposed_d"]
    assert s["area_mm2"] == pytest.approx(3.14159 * 9, rel=1e-4)
    assert s["step_d"] < P.WE_FLAT["sample_w"]


def test_bolt_does_not_protrude_and_engages_nut():
    st = bolt_stack(P, P.CHAMBER_RING["h"] - P.CELL["bolt_head_h"], -P.BASE_FLAT["t"])
    assert st["z_end"] > -P.BASE_FLAT["t"]                      # no sobresale
    assert st["nut_top"] - P.CELL["nut_h"] >= st["z_end"] - 1e-9  # rosca completa
    assert st["nut_top"] < 0


def test_all_wall_checks_pass():
    bad = [str(r) for r in walls.run() if not r.ok]
    assert not bad, "\n".join(bad)


def test_raman_objective_clears_ring(ring):
    assert collision_volume(ring) < 1e-6


def test_checks_catch_bad_variant():
    """Tornillos pegados al borde: el check de pared seca debe fallar."""
    p = variant(CELL={"bolt_inset": 2.5})
    names = [r.name for r in walls.ring_checks(p) if not r.ok]
    assert "tornillos ↔ exterior lateral" in names


def test_raman_check_catches_narrow_well():
    p = variant(CHAMBER_RING={"well_d": 20.0})
    assert collision_volume(chamber_ring.build(p), p) > 0


def test_ports_open_into_well(ring):
    """Los puertos restan material (no quedaron fuera del cuerpo)."""
    p_closed = variant(RING_PORTS=())
    v_closed = chamber_ring.build(p_closed).val().Volume()
    assert ring.val().Volume() < v_closed
