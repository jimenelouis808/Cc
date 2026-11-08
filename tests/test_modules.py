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


# --- celda XRD -------------------------------------------------------------------

@pytest.fixture(scope="module")
def xrd_body():
    from modules import chamber_xrd
    return chamber_xrd.build()


def test_xrd_parts_valid(xrd_body):
    from modules import lid_xrd
    for part in (xrd_body, lid_xrd.build(), lid_xrd.build_cover()):
        s = part.val()
        assert s.isValid() and len(s.Solids()) == 1


def test_xrd_top_is_the_reference_plane(xrd_body):
    from modules import chamber_xrd
    assert xrd_body.val().BoundingBox().zmax == pytest.approx(0.0, abs=1e-6)
    assert chamber_xrd.sample().BoundingBox().zmax == pytest.approx(0.0, abs=1e-9)


def test_xrd_nothing_on_the_beam_faces():
    """Ningún puerto en ±x; si alguien lo pone, el cuerpo se niega a construirse."""
    from modules import chamber_xrd
    bad = list(P.XRD_PORTS) + [{"name": "x", "face": "+x", "offset": 0.0, "z": -8.0,
                                "kind": "fitting", "through_d": 2.0}]
    with pytest.raises(ValueError):
        chamber_xrd.features(variant(XRD_PORTS=tuple(bad)))


def test_xrd_checks_pass():
    bad = [str(r) for r in walls.xrd_checks() if not r.ok]
    assert not bad, "\n".join(bad)


def test_xrd_flat_frame_is_pushed_out_of_the_beam():
    """Sin bisel (borde de espesor completo) el marco se aleja hasta que el haz pasa."""
    from modules import chamber_xrd
    flat = variant(LID_XRD={"edge_t": P.LID_XRD["t"]})
    assert chamber_xrd.layout(flat)["lid"]["fx_in"] > chamber_xrd.layout()["lid"]["fx_in"] + 5


def test_xrd_beam_check_catches_an_obstacle():
    """Un bloque de 1 mm a 12 mm del centro, dentro del corredor, tapa el haz a 10°."""
    import cadquery as cq
    from checks.xrd_clearance import invasion
    block = cq.Workplane("XY").add(cq.Solid.makeBox(2, 4, 1, cq.Vector(11, -2, 0)))
    assert invasion(block, P) > 0.1


def test_xrd_frame_profile_stays_under_the_beam_limit():
    from checks.xrd_clearance import allowed_height
    from modules import chamber_xrd, lid_xrd
    lay = chamber_xrd.layout()
    d = lay["lid"]["fx_in"]
    while d < lay["x_half"]:
        assert lid_xrd.frame_profile(P, d) <= allowed_height(P, d) + 1e-9
        d += 0.5


def test_xrd_seal_encloses_the_clip_pocket():
    from modules import chamber_xrd
    lay = chamber_xrd.layout()
    assert lay["lid"]["gy1_in"] > lay["pocket_y1"] + P.CHAMBER_XRD["cover_ledge"]


# --- celda Chladni ---------------------------------------------------------------

def test_chladni_parts_valid():
    from modules import chladni
    for part in (chladni.build_wall(), chladni.build_bottom(), chladni.build_lid()):
        s = part.val()
        assert s.isValid() and len(s.Solids()) == 1


def test_chladni_plate_face_is_reference_plane():
    from modules import chladni
    bb = chladni.plate().BoundingBox()
    assert bb.zmax == pytest.approx(0.0, abs=1e-9)
    assert bb.zmin == pytest.approx(-P.CHLADNI["plate_h"], abs=1e-9)


def test_chladni_checks_pass():
    bad = [str(r) for r in walls.chladni_checks() if not r.ok]
    assert not bad, "\n".join(bad)


def test_piezo_coupling_physics():
    """∫∇²w del modo fundamental sobre toda la placa es 0 (pendiente nula en el borde
    empotrado): un piezo centrado acopla fuerte; uno en ≈ L/4 casi nada."""
    from sim.chladni_estimate import coupling
    L, d = P.CHLADNI["plate_L"], P.CHLADNI["piezo_d"]
    centred = abs(coupling(1, 1, (0.0, 0.0), d, L))
    quarter = abs(coupling(1, 1, (-L / 4, -L / 4), d, L))
    assert centred > 0.3 and quarter < 0.1
    # (2,1) tiene línea nodal en x = 0: un piezo centrado no lo excita
    assert abs(coupling(2, 1, (0.0, 0.0), d, L)) < 0.02
    # en la diagonal, (2,1) y (1,2) acoplan igual: sólo su combinación simétrica
    c = P.CHLADNI["piezo_center"]
    assert coupling(2, 1, c, d, L) == pytest.approx(coupling(1, 2, c, d, L), rel=1e-6)


def test_wet_modes_are_below_dry_ones():
    from sim.chladni_estimate import mode_table
    for row in mode_table():
        assert row["f_liquido_hz"] < row["f_seco_hz"]


# --- celda EPD / DEP -------------------------------------------------------------

def test_epd_parts_valid():
    from modules import epd
    for part in (epd.build_base(), epd.build_frame()):
        s = part.val()
        assert s.isValid() and len(s.Solids()) == 1


def test_epd_checks_pass():
    bad = [str(r) for r in walls.epd_checks() if not r.ok]
    assert not bad, "\n".join(bad)


def test_epd_gaps_decrease_downstream_and_fingers_alternate():
    from modules import epd
    pat = epd.ide_rects()
    gaps = [s["gap"] for s in pat["segments"]]
    assert gaps == sorted(gaps, reverse=True)
    for seg in pat["segments"]:
        xs = seg["fingers"]
        pitch = P.EPD["finger_w"] + seg["gap"]
        assert all(b - a == pytest.approx(pitch) for a, b in zip(xs, xs[1:]))


def test_epd_mask_is_written(tmp_path):
    from modules import epd
    path = epd.mask_dxf(path=str(tmp_path / "m.dxf"))
    assert os.path.getsize(path) > 1000


def test_epd_mask_check_catches_too_fine_a_gap():
    p = variant(EPD={"gaps": (0.5, 0.05)})
    bad = [r.name for r in walls.epd_checks(p) if not r.ok]
    assert "máscara: puente mínimo" in bad


def test_epd_solvent_check_catches_pmma():
    p = variant(EPD={"wetted_materials": ("vidrio", "PMMA")})
    bad = [r.name for r in walls.epd_checks(p) if not r.ok]
    assert any("disolventes" in n for n in bad)


def test_dep_capture_scales_as_v2_over_g3():
    from sim.epd_dep import capture_table
    base = capture_table()
    hi = capture_table(variant(EPD_SIM={"v_rms": 2 * P.EPD_SIM["v_rms"]}))
    d = P.EPD_SIM["particle_d_nm"][0]
    assert hi[0]["C"][d] == pytest.approx(4 * base[0]["C"][d], rel=1e-9)
    g0, g1 = base[0]["gap_um"], base[-1]["gap_um"]
    assert base[-1]["C"][d] / base[0]["C"][d] == pytest.approx((g0 / g1) ** 3, rel=1e-9)
