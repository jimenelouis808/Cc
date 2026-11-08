import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import params as P  # noqa: E402
from checks import walls  # noqa: E402
from checks import raman_clearance  # noqa: E402
from checks.raman_clearance import collision_volume  # noqa: E402
from modules import base_flat, chamber_ring, lid_raman  # noqa: E402
from modules._optics import light_radius, z_tip  # noqa: E402
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
    assert s["area_mm2"] == pytest.approx(3.14159265 * P.WE_FLAT["exposed_d"] ** 2 / 4)
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


def test_raman_check_catches_low_ring():
    """Un anillo con pozo estrecho invade la nariz del objetivo."""
    p = variant(CHAMBER_RING={"well_d": 12.0, "h": 13.0})
    assert collision_volume(chamber_ring.build(p), p) > 0


# --- tapa Raman ---------------------------------------------------------------

@pytest.fixture(scope="module")
def lid():
    return lid_raman.build()


def test_lid_is_valid_single_solid(lid):
    s = lid.val()
    assert s.isValid() and len(s.Solids()) == 1


def test_focus_shift():
    """z_tip = WD + t(1 − 1/n) por cada capa plana; sin capas, z_tip = WD."""
    p0 = variant(RAMAN={"window_t": 0.0}, CELL={"thin_gap_raman": 0.0})
    assert z_tip(p0, 0.0) == pytest.approx(P.RAMAN["working_distance"])
    assert z_tip(P) < P.RAMAN["working_distance"] + P.RAMAN["window_t"] + P.CELL["thin_gap_raman"]


def test_light_cone_is_continuous_and_refracts():
    g, t = P.CELL["thin_gap_raman"], P.RAMAN["window_t"]
    assert light_radius(P, 0.0) == 0.0
    # más abierto en aire que en vidrio para el mismo Δz
    d_glass = light_radius(P, g + t) - light_radius(P, g)
    d_air = light_radius(P, g + 2 * t) - light_radius(P, g + t)
    assert d_air > d_glass > 0


def test_nose_bottom_is_at_gap(lid):
    zmin = lid.val().BoundingBox().zmin
    assert zmin == pytest.approx(P.CELL["thin_gap_raman"], abs=1e-3)
    s_lo, s_hi = lid_raman.shim_range(P)
    assert lid_raman.gap_for_shim(P, s_lo) == pytest.approx(P.CELL["gap_range_raman"][0])
    assert lid_raman.gap_for_shim(P, s_hi) == pytest.approx(P.CELL["gap_range_raman"][1])
    moved = lid_raman.build(P, shim=s_hi).val().BoundingBox().zmin
    assert moved == pytest.approx(P.CELL["gap_range_raman"][1], abs=1e-3)


def test_raman_clearance_all_parts_all_gaps():
    bad = [(n, v) for n, v in raman_clearance.run() if v >= raman_clearance.TOL]
    assert not bad


def test_barrel_sized_objective_tip_is_rejected():
    """Con la cara frontal del tamaño del barril (25 mm) el émbolo no cabe en el pozo."""
    p = variant(RAMAN={"objective_tip_diameter": 25.0})
    bad = [r.name for r in walls.lid_raman_checks(p) if not r.ok]
    assert any("émbolo ↔ cámara" in n or "interferencia" in n for n in bad)


def test_small_exposed_area_breaks_nose_wall():
    """Con exposed_d = 6 (valor original) la nariz no deja pasar NA 0.5 con pared de 2 mm."""
    p = variant(WE_FLAT={"exposed_d": 6.0})
    bad = [r.name for r in walls.lid_raman_checks(p) if not r.ok]
    assert "émbolo: pared hueco ↔ electrolito" in bad


def test_ports_open_into_well(ring):
    """Los puertos restan material (no quedaron fuera del cuerpo)."""
    p_closed = variant(RING_PORTS=())
    v_closed = chamber_ring.build(p_closed).val().Volume()
    assert ring.val().Volume() < v_closed


# --- cámara para pinza de Pt ----------------------------------------------------

def test_clamp_chamber_valid_and_lid_compatible():
    from modules import chamber_clamp

    s = chamber_clamp.build().val()
    assert s.isValid() and len(s.Solids()) == 1
    bb = s.BoundingBox()
    assert bb.zmax == pytest.approx(P.CHAMBER_RING["h"], abs=1e-6)   # misma cara que el anillo
    assert bb.xmax - bb.xmin == pytest.approx(P.CELL["body_w"], abs=1e-6)


def test_clamp_sample_face_is_reference_plane():
    from modules import chamber_clamp

    smp = chamber_clamp.sample().BoundingBox()
    assert smp.zmax == pytest.approx(0.0, abs=1e-9)
    assert smp.xmin < 0 < smp.xmax                                    # el eje óptico cae en la tira
    ped = chamber_clamp.pedestal().BoundingBox()
    assert ped.zmax == pytest.approx(smp.zmin, abs=1e-9)              # apoya en el pedestal


def test_clamp_checks_pass():
    bad = [str(r) for r in walls.clamp_checks() + walls.lid_raman_checks(chamber="clamp")
           if not r.ok]
    assert not bad, "\n".join(bad)


def test_wide_clip_is_rejected():
    """Una pinza que cerrada no pasa por el puerto WE debe marcarse."""
    p = variant(WE_CLAMP={"clip_w": 4.5})
    bad = [r.name for r in walls.clamp_checks(p) if not r.ok]
    assert "pinza cerrada cabe por el puerto WE (holgura)" in bad
