"""The literature database and its dispersion corrections."""

from __future__ import annotations

import json

import pytest

from ramancarbon.core.spectrum import laser_energy_ev
from ramancarbon.database import load_database
from ramancarbon.database.loader import DATA_DIR, DatabaseError


def test_every_band_carries_a_source_and_a_confidence(db):
    for band in db.bands.values():
        assert band.source, f"{band.key} has no source"
        assert band.confidence in {"high", "medium", "low", "unknown"}


def test_every_material_carries_a_source(db):
    for material in db.materials.values():
        assert material.source, f"{material.key} has no source"


def test_band_windows_bracket_their_positions(db):
    for band in db.bands.values():
        low, high = band.window
        assert low <= band.position <= high, band.key


def test_dispersion_moves_the_d_band_the_known_amount(db):
    """~50 cm-1/eV: 22 cm-1 between 532 and 633 nm."""
    d = db.band("D")
    at_532 = d.position_at(laser_energy_ev(532.0))
    at_633 = d.position_at(laser_energy_ev(633.0))
    assert at_532 - at_633 == pytest.approx(18.5, abs=1.0)


def test_the_2d_band_disperses_twice_as_fast_as_d(db):
    """It is an overtone; anything else would be unphysical."""
    assert db.band("2D").dispersion == pytest.approx(2.0 * db.band("D").dispersion)


def test_the_g_band_does_not_disperse(db):
    assert db.band("G").dispersion == 0.0
    assert not db.band("G").is_dispersive


def test_rbm_has_no_meaningful_reference_position(db):
    """Its frequency is set by the diameter, so a 'shift' means nothing."""
    assert not db.band("RBM").position_is_reference
    assert db.band("RBM").multi_valued
    assert db.band("D").position_is_reference


def test_window_translates_with_the_laser(db):
    d = db.band("D")
    low_532, high_532 = d.window_at(laser_energy_ev(532.0))
    low_785, high_785 = d.window_at(laser_energy_ev(785.0))
    assert low_785 < low_532
    # The width is scatter between samples, not a laser effect.
    assert (high_785 - low_785) == pytest.approx(high_532 - low_532)


def test_unknown_keys_list_the_alternatives(db):
    with pytest.raises(DatabaseError, match="known bands"):
        db.band("Z")
    with pytest.raises(DatabaseError, match="known"):
        db.material("unobtainium")


def test_rbm_default_exists(db):
    assert db.rbm_parameterisation().key == db.rbm_default


def test_metallic_g_splitting_constant_is_larger(db):
    """It carries an extra electron-phonon contribution."""
    constants = db.g_splitting
    assert constants["C_metallic"] > constants["C_semiconducting"]


def test_defect_type_ratios_are_ordered_as_published(db):
    r = db.defect_type_ratios
    assert r["sp3"] > r["vacancy"] > r["boundary"] > r["on_site_substitutional"]


def test_all_json_files_are_valid_and_documented():
    for path in sorted(DATA_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "_comment" in data, f"{path.name} lacks an explanatory header"


def test_database_can_be_pointed_at_a_copy(tmp_path):
    """A specialist must be able to use a house-calibrated database."""
    import shutil

    target = tmp_path / "data"
    shutil.copytree(DATA_DIR, target)
    payload = json.loads((target / "rbm.json").read_text(encoding="utf-8"))
    payload["parameterisations"][0]["A"] = 999.0
    (target / "rbm.json").write_text(json.dumps(payload), encoding="utf-8")
    other = load_database(target)
    assert other.rbm[payload["parameterisations"][0]["key"]].A == 999.0
