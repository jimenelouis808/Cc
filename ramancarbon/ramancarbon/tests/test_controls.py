"""Linked parameters, exclusion ranges, manual baselines, undo and settings."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ramancarbon.core.baseline import anchor_baseline, suggest_anchors
from ramancarbon.core.history import (
    History,
    Preferences,
    config_directory,
)
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.examples.demo_data import make_demo
from ramancarbon.models.constraints import (
    ConstraintError,
    Link,
    order,
    parse_link,
    parse_links,
    validate,
)
from ramancarbon.models.fitting import FitModel, PeakSpec, fit_model

X = np.arange(1100.0, 1800.0, 1.0)


def lorentzian(centre, height, fwhm):
    half = fwhm / 2.0
    return height * half ** 2 / ((X - centre) ** 2 + half ** 2)


@pytest.fixture
def three_bands():
    y = (lorentzian(1350.0, 300.0, 60.0) + lorentzian(1580.0, 500.0, 45.0)
         + lorentzian(1620.0, 120.0, 45.0) + 20.0)
    noise = np.random.default_rng(0).normal(0.0, 2.0, X.size)
    return Spectrum(shift=X, intensity=y + noise, laser_nm=532.0, name="tres")


def model(**kwargs):
    return FitModel(
        peaks=[PeakSpec("D", centre=1350.0, height=300.0, fwhm=60.0),
               PeakSpec("G", centre=1580.0, height=500.0, fwhm=45.0),
               PeakSpec("D'", centre=1620.0, height=120.0, fwhm=50.0)],
        window=(1150.0, 1750.0), **kwargs,
    )


# -- parsing links -----------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [("D3.centre = D.centre + 100", (1.0, 100.0)),
     ("G-.fwhm = G.fwhm", (1.0, 0.0)),
     ("p2.height = 0.5 * p1.height", (0.5, 0.0)),
     ("a.centre = b.centre - 12.5", (1.0, -12.5)),
     ("x.fwhm=y.fwhm", (1.0, 0.0))],
)
def test_the_link_syntax_reads_what_it_should(text, expected):
    link = parse_link(text)
    assert (link.factor, link.offset) == expected


def test_a_link_round_trips_through_its_own_text():
    for text in ("D3.centre = D.centre + 100", "p2.height = 0.5·p1.height",
                 "G-.fwhm = G.fwhm"):
        assert str(parse_link(text.replace("·", " * "))) == text


def test_a_link_that_is_not_understood_says_what_the_form_is():
    with pytest.raises(ConstraintError, match="La forma es"):
        parse_link("D3.centre = D.centre * 2")
    with pytest.raises(ConstraintError, match="La forma es"):
        parse_link("tonterías")


def test_comments_and_blank_lines_are_skipped():
    links = parse_links(["", "# una nota", "a.fwhm = b.fwhm  # otra"])
    assert len(links) == 1


def test_a_cycle_is_refused_with_the_cycle_in_the_message():
    with pytest.raises(ConstraintError, match="ciclo"):
        order([parse_link("a.centre = b.centre"),
               parse_link("b.centre = a.centre")])


def test_chains_are_allowed_and_come_back_in_order():
    links = order(parse_links(["c.centre = b.centre + 5",
                               "b.centre = a.centre + 5"]))
    assert [link.target_peak for link in links] == ["b", "c"]


def test_a_link_to_a_component_that_is_not_there_names_the_ones_that_are():
    available = {"D": ("centre", "height", "fwhm")}
    with pytest.raises(ConstraintError, match="hay: D"):
        validate([parse_link("X.centre = D.centre")], available)


def test_a_link_to_a_parameter_that_does_not_exist_is_refused():
    available = {"D": ("centre", "height", "fwhm")}
    with pytest.raises(ConstraintError, match="no tiene el parámetro"):
        validate([parse_link("D.eta = D.centre")], available)


def test_a_parameter_cannot_be_linked_twice():
    available = {"a": ("centre",), "b": ("centre",), "c": ("centre",)}
    with pytest.raises(ConstraintError, match="ya está"):
        validate([parse_link("a.centre = b.centre"),
                  parse_link("a.centre = c.centre")], available)


def test_a_parameter_cannot_be_linked_to_itself():
    with pytest.raises(ConstraintError, match="a sí mismo"):
        validate([Link("a", "centre", "a", "centre")],
                 {"a": ("centre",)})


# -- fitting with links ------------------------------------------------

def test_a_linked_width_really_is_the_same_width(three_bands):
    result = fit_model(three_bands, model(links=("D'.fwhm = G.fwhm",)))
    assert result.peak("D'").fwhm == pytest.approx(result.peak("G").fwhm,
                                                   abs=1e-9)


def test_a_linked_centre_keeps_its_offset_exactly(three_bands):
    result = fit_model(three_bands, model(links=("D'.centre = G.centre + 40",)))
    separation = result.peak("D'").centre - result.peak("G").centre
    assert separation == pytest.approx(40.0, abs=1e-9)


def test_a_linked_fit_has_fewer_free_parameters(three_bands):
    free = fit_model(three_bands, model())
    linked = fit_model(three_bands, model(links=("D'.fwhm = G.fwhm",)))
    assert linked.n_parameters == free.n_parameters - 1


def test_a_link_with_a_factor_holds(three_bands):
    result = fit_model(three_bands, model(links=("D'.height = 0.25 * G.height",)))
    assert result.peak("D'").height == pytest.approx(
        0.25 * result.peak("G").height, rel=1e-9)


def test_the_links_travel_with_the_result(three_bands):
    result = fit_model(three_bands, model(links=("D'.fwhm = G.fwhm",)))
    assert result.links == ("D'.fwhm = G.fwhm",)
    assert "Ligaduras" in result.summary()


def test_a_linked_fit_is_almost_as_good_as_a_free_one_when_the_link_is_true(
        three_bands):
    """D' and G really do have the same width in this spectrum."""
    free = fit_model(three_bands, model())
    linked = fit_model(three_bands, model(links=("D'.fwhm = G.fwhm",)))
    assert linked.r_squared > free.r_squared - 1e-4


def test_two_components_cannot_share_a_name():
    with pytest.raises(ValueError, match="mismo nombre"):
        FitModel(peaks=[PeakSpec("D", centre=1350.0),
                        PeakSpec("D", centre=1580.0)],
                 window=(1150.0, 1750.0))


def test_a_bad_link_is_caught_when_the_model_is_built():
    with pytest.raises(ConstraintError):
        model(links=("Z.fwhm = G.fwhm",))


# -- exclusion ranges --------------------------------------------------

@pytest.fixture
def with_spike():
    y = (lorentzian(1350.0, 300.0, 60.0) + lorentzian(1580.0, 500.0, 45.0)
         + 20.0)
    y = y.copy()
    y[np.abs(X - 1450.0) < 4] += 4000.0
    return Spectrum(shift=X, intensity=y, laser_nm=532.0, name="rayo")


def two_bands():
    return FitModel(
        peaks=[PeakSpec("D", centre=1350.0, height=300.0, fwhm=60.0),
               PeakSpec("G", centre=1580.0, height=500.0, fwhm=45.0)],
        window=(1150.0, 1750.0),
    )


def test_a_spike_wrecks_a_fit_and_excluding_it_does_not(with_spike):
    spoiled = fit_model(with_spike, two_bands())
    clean = fit_model(with_spike, two_bands(), exclude=[(1440.0, 1460.0)])
    assert abs(spoiled.peak("D").centre - 1350.0) > 5.0
    assert clean.peak("D").centre == pytest.approx(1350.0, abs=0.5)
    assert clean.peak("G").height == pytest.approx(500.0, rel=0.02)


def test_what_was_excluded_travels_with_the_result(with_spike):
    result = fit_model(with_spike, two_bands(), exclude=[(1440.0, 1460.0)])
    assert result.excluded == ((1440.0, 1460.0),)
    assert "Excluido" in result.summary()


def test_the_residual_still_covers_the_excluded_range(with_spike):
    """It is what shows whether the exclusion was justified."""
    result = fit_model(with_spike, two_bands(), exclude=[(1440.0, 1460.0)])
    assert result.residual.shape == result.x.shape
    inside = np.abs(result.x - 1450.0) < 4
    assert np.max(np.abs(result.residual[inside])) > 1000.0


def test_the_statistics_ignore_the_excluded_range(with_spike):
    """An R-squared that counts a region the fit was never asked to
    reproduce is not a measure of anything."""
    result = fit_model(with_spike, two_bands(), exclude=[(1440.0, 1460.0)])
    assert result.r_squared > 0.999


def test_excluding_everything_is_refused(with_spike):
    with pytest.raises(ValueError, match="quedan"):
        fit_model(with_spike, two_bands(), exclude=[(1100.0, 1800.0)])


def test_an_exclusion_outside_the_window_is_simply_ignored(with_spike):
    result = fit_model(with_spike, two_bands(), exclude=[(900.0, 950.0)])
    assert result.excluded == ()


# -- manual baselines --------------------------------------------------

def test_a_baseline_through_anchors_passes_near_them():
    spectrum = make_demo("MWCNT", fluorescence=700.0, seed=1)
    anchors = suggest_anchors(spectrum.shift, spectrum.intensity, 8)
    baseline = anchor_baseline(spectrum.shift, spectrum.intensity, anchors)
    for anchor in anchors:
        index = int(np.argmin(np.abs(spectrum.shift - anchor)))
        assert abs(baseline[index] - spectrum.intensity[index]) < 30.0


def test_an_anchor_takes_a_local_median_not_one_noisy_point():
    x = np.arange(0.0, 1000.0, 1.0)
    y = np.full_like(x, 100.0)
    y[500] = 900.0                      # one bad sample right on an anchor
    baseline = anchor_baseline(x, y, [100.0, 500.0, 900.0], half_width=10.0)
    assert baseline[500] == pytest.approx(100.0, abs=1.0)


def test_the_shape_preserving_interpolation_does_not_ring():
    """A cubic spline undershoots a step by 207 counts where PCHIP
    undershoots by 123."""
    x = np.arange(0.0, 1000.0, 1.0)
    step = np.where(x < 400.0, 500.0, 120.0)
    anchors = [10.0, 150.0, 300.0, 380.0, 430.0, 600.0, 800.0, 990.0]
    pchip = anchor_baseline(x, step, anchors, kind="pchip")
    spline = anchor_baseline(x, step, anchors, kind="spline")
    assert np.min(pchip - step) > np.min(spline - step)


def test_a_smooth_background_is_recovered_by_all_three():
    x = np.arange(1000.0, 2000.0, 1.0)
    background = 200.0 + 300.0 * np.exp(-(x - 1000.0) / 300.0)
    peak = 900.0 * np.exp(-0.5 * ((x - 1500.0) / 40.0) ** 2)
    anchors = [1010.0, 1200.0, 1350.0, 1660.0, 1850.0, 1990.0]
    for kind in ("pchip", "spline", "linear"):
        estimated = anchor_baseline(x, background + peak, anchors, kind=kind)
        assert np.max(np.abs(estimated - background)) < 15.0


def test_two_anchors_are_a_straight_line():
    x = np.arange(0.0, 100.0, 1.0)
    y = 5.0 + 0.5 * x
    assert np.allclose(anchor_baseline(x, y, [1.0, 98.0]), y, atol=1.0)


def test_fewer_than_two_anchors_is_refused():
    with pytest.raises(ValueError, match="dos anclas"):
        anchor_baseline([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [2.0])


def test_an_anchor_outside_the_spectrum_is_refused_by_name():
    with pytest.raises(ValueError, match="fuera del espectro"):
        anchor_baseline([1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [2.0, 40.0])


def test_an_unknown_interpolation_is_refused():
    with pytest.raises(ValueError, match="interpolación desconocida"):
        anchor_baseline([1.0, 2.0, 3.0, 4.0], [1.0] * 4, [1.0, 2.0, 4.0],
                        kind="bezier")


def test_suggested_anchors_are_inside_the_spectrum_and_ordered():
    spectrum = make_demo("MWCNT", seed=2)
    anchors = suggest_anchors(spectrum.shift, spectrum.intensity, 10)
    assert anchors == sorted(anchors)
    assert all(spectrum.shift[0] <= a <= spectrum.shift[-1] for a in anchors)


def test_the_manual_baseline_is_reachable_through_the_dispatcher():
    from ramancarbon.core.baseline import estimate_baseline

    spectrum = make_demo("MWCNT", fluorescence=500.0)
    anchors = suggest_anchors(spectrum.shift, spectrum.intensity, 6)
    baseline = estimate_baseline(spectrum, method="anclas", anchors=anchors)
    assert baseline.shape == spectrum.shift.shape


# -- undo and redo -----------------------------------------------------

def test_undo_and_redo_walk_the_states():
    history = History({"a": 1})
    history.push({"a": 2}, "dos")
    history.push({"a": 3}, "tres")
    assert history.undo() == {"a": 2}
    assert history.undo() == {"a": 1}
    assert not history.can_undo
    assert history.redo() == {"a": 2}


def test_an_unchanged_state_is_not_recorded():
    history = History({"a": 1})
    assert history.push({"a": 2}, "cambio")
    assert not history.push({"a": 2}, "lo mismo")
    assert len(history) == 2


def test_pushing_after_an_undo_discards_the_redo_branch():
    history = History({"a": 1})
    history.push({"a": 2})
    history.push({"a": 3})
    history.undo()
    history.push({"a": 9}, "otra rama")
    assert not history.can_redo
    assert history.current == {"a": 9}


def test_the_stack_has_a_bottom():
    history = History({"n": 0}, depth=5)
    for index in range(1, 20):
        history.push({"n": index})
    assert len(history) == 5
    assert history.current == {"n": 19}


def test_undoing_past_the_start_is_an_error():
    history = History({"a": 1})
    with pytest.raises(IndexError, match="deshacer"):
        history.undo()
    with pytest.raises(IndexError, match="rehacer"):
        history.redo()


def test_the_states_are_copied_not_referenced():
    state = {"peaks": [1, 2, 3]}
    history = History(state)
    state["peaks"].append(4)
    assert history.current == {"peaks": [1, 2, 3]}


def test_every_state_carries_a_label():
    history = History({"a": 0})
    history.push({"a": 1}, "quitar fondo")
    assert history.undo_label() == "quitar fondo"
    assert "quitar fondo" in history.labels()


def test_reset_forgets_everything():
    history = History({"a": 1})
    history.push({"a": 2})
    history.reset({"a": 99})
    assert not history.can_undo and not history.can_redo
    assert history.current == {"a": 99}


# -- preferences -------------------------------------------------------

def test_preferences_round_trip_through_a_file(tmp_path):
    preferences = Preferences.load(tmp_path / "p.json")
    preferences.set("laser_nm", 633.0)
    preferences.set("palette", "oscuro")
    assert preferences.save() == tmp_path / "p.json"

    reopened = Preferences.load(tmp_path / "p.json")
    assert reopened.loaded
    assert reopened.get("laser_nm") == 633.0
    assert reopened.get("palette") == "oscuro"


def test_a_missing_file_gives_the_defaults_without_complaint(tmp_path):
    preferences = Preferences.load(tmp_path / "no_hay.json")
    assert not preferences.loaded
    assert not preferences.problem
    assert preferences.get("laser_nm") == 532.0


def test_a_corrupt_file_is_survived_with_a_reason(tmp_path):
    path = tmp_path / "roto.json"
    path.write_text("{esto no es json", encoding="utf-8")
    preferences = Preferences.load(path)
    assert preferences.get("laser_nm") == 532.0
    assert "no se han podido leer" in preferences.problem


def test_a_file_that_is_json_but_not_an_object_is_survived(tmp_path):
    path = tmp_path / "lista.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert "objeto JSON" in Preferences.load(path).problem


def test_settings_a_newer_version_wrote_are_kept(tmp_path):
    """Two versions of the program sharing a preferences file must not
    delete each other's settings."""
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"laser_nm": 785.0, "algo_del_futuro": 3}),
                    encoding="utf-8")
    preferences = Preferences.load(path)
    preferences.save()
    assert json.loads(path.read_text(encoding="utf-8"))["algo_del_futuro"] == 3


def test_saving_where_it_cannot_be_written_is_not_fatal(tmp_path):
    blocked = tmp_path / "archivo"
    blocked.write_text("soy un archivo", encoding="utf-8")
    preferences = Preferences.load(tmp_path / "p.json")
    assert preferences.save(blocked / "sub" / "p.json") is None
    assert preferences.problem


def test_recent_files_have_no_duplicates_and_the_newest_is_first(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("x", encoding="utf-8")
    second.write_text("x", encoding="utf-8")
    preferences = Preferences()
    preferences.remember_file(first)
    preferences.remember_file(second)
    preferences.remember_file(first)
    assert preferences.values["recent_files"][0] == str(first.resolve())
    assert len(preferences.values["recent_files"]) == 2


def test_a_recent_file_that_is_gone_is_not_offered_but_not_forgotten(tmp_path):
    path = tmp_path / "a.txt"
    path.write_text("x", encoding="utf-8")
    preferences = Preferences()
    preferences.remember_file(path)
    path.unlink()
    assert preferences.existing_recent() == []
    assert preferences.values["recent_files"], "sigue en la lista por si vuelve"


def test_resetting_puts_the_factory_settings_back():
    preferences = Preferences()
    preferences.set("laser_nm", 785.0)
    preferences.reset(["laser_nm"])
    assert preferences.get("laser_nm") == 532.0
    preferences.set("palette", "oscuro")
    preferences.reset()
    assert preferences.get("palette") == "claro"


def test_the_configuration_directory_is_named_after_the_program(monkeypatch):
    monkeypatch.delenv("RAMANCARBON_CONFIG", raising=False)
    assert config_directory().name == "ramancarbon"
    monkeypatch.setenv("RAMANCARBON_CONFIG", "/tmp/en_otro_sitio")
    assert str(config_directory()) == "/tmp/en_otro_sitio"
