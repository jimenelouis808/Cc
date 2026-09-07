"""Electrochemistry: capacitance, mechanism, impedance, catalysis."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.echem.curve import (
    ChargeDischarge,
    CurveError,
    Electrode,
    Impedance,
    Voltammogram,
)
from ramancarbon.echem.cv import analyse_cv, analyse_rate_study, capacitance
from ramancarbon.echem.eis import (
    CircuitError,
    cpe_to_capacitance,
    fit_circuit,
    kramers_kronig,
    parse_circuit,
)
from ramancarbon.echem.evaluate import (
    analyse_catalysis,
    classify_storage,
    tafel_analysis,
)
from ramancarbon.echem.gcd import analyse_cycling, analyse_gcd
from ramancarbon.echem.io import read_cv, read_eis, read_gcd, write_cv, write_eis, write_gcd
from ramancarbon.echem.report import analyse_sample
from ramancarbon.examples.demo_data import (
    ECHEM_DEMOS,
    cv_rate_series,
    make_cv_demo,
    make_eis_demo,
    make_gcd_demo,
    make_lsv_demo,
)


# -- the data model ---------------------------------------------------


def test_a_scan_rate_in_millivolts_is_caught():
    """The single most common error in this whole analysis, and it is a
    factor of a thousand."""
    v = np.linspace(0.0, 0.5, 100)
    with pytest.raises(CurveError, match="mV/s"):
        Voltammogram(v, np.ones(100) * 1e-3, scan_rate=50.0)


def test_non_finite_data_are_refused():
    v = np.linspace(0.0, 0.5, 100)
    bad = np.ones(100)
    bad[7] = np.nan
    with pytest.raises(CurveError):
        Voltammogram(v, bad, scan_rate=0.05)


def test_rhe_conversion_needs_the_ph():
    electrode = Electrode(reference="Ag/AgCl_3M")
    converted, reason = electrode.to_rhe(np.array([0.0]))
    assert converted is None and "pH" in reason

    electrode = Electrode(reference="Ag/AgCl_3M", ph=14.0)
    converted, reason = electrode.to_rhe(np.array([0.0]))
    assert converted is not None
    assert converted[0] == pytest.approx(0.210 + 0.05916 * 14.0, abs=1e-4)


def test_reference_electrodes_differ_by_the_amounts_that_matter():
    """Ag/AgCl 3 M and SCE are 31 mV apart; confusing them shifts a whole
    Tafel plot."""
    kwargs = {"ph": 0.0}
    a = Electrode(reference="Ag/AgCl_3M", **kwargs).to_rhe(np.array([0.0]))[0][0]
    b = Electrode(reference="SCE", **kwargs).to_rhe(np.array([0.0]))[0][0]
    assert abs(b - a) == pytest.approx(0.031, abs=0.002)


def test_missing_resistance_is_said_out_loud_not_ignored():
    _, note = Electrode().ir_correct(np.zeros(5), np.zeros(5))
    assert "SIN corrección" in note


def test_impedance_keeps_the_physical_sign():
    frequency = np.logspace(5, -2, 20)
    z = 10.0 - 1j * 50.0 / frequency
    spectrum = Impedance(frequency, z)
    assert np.all(spectrum.z.imag < 0), "una respuesta capacitiva tiene Z'' < 0"


# -- capacitance ------------------------------------------------------


def test_capacitance_round_trips_the_value_it_was_generated_from():
    curve = make_cv_demo("condensador", capacitance=0.05, seed=1)
    loop, sweep = capacitance(curve.cycles()[-1])
    assert loop.farads == pytest.approx(0.05, rel=0.03)
    assert sweep.farads == pytest.approx(0.05, rel=0.03)


def test_the_vertex_trim_does_not_bias_the_result():
    """Trimming the vertices and then dividing by the FULL window made
    every capacitance 4 % too small."""
    from ramancarbon.echem.cv import VERTEX_TRIM, integrate_charge

    curve = make_cv_demo("condensador", capacitance=0.05, seed=1).cycles()[-1]
    _, _, span = integrate_charge(curve)
    assert span < curve.span
    assert capacitance(curve)[0].farads == pytest.approx(0.05, rel=0.03)


def test_both_capacitance_conventions_are_reported():
    result = analyse_cv(make_cv_demo("condensador", seed=1))
    assert result.capacitance_loop and result.capacitance_sweep
    assert "2νΔV" in result.capacitance_loop.convention
    assert "νΔV" in result.capacitance_sweep.convention
    assert "factor 2" in result.summary()


def test_specific_capacitance_needs_the_active_mass():
    curve = make_cv_demo("condensador", seed=1)
    curve.electrode = Electrode()
    result = analyse_cv(curve)
    assert result.capacitance_loop.specific_f_per_g is None
    assert any("masa de material ACTIVO" in w for w in result.warnings)


def test_the_first_cycle_is_not_used_silently():
    result = analyse_cv(make_cv_demo("condensador", cycles=3, seed=1))
    assert any("PRIMERO nunca es representativo" in w for w in result.warnings)


# -- galvanostatic ----------------------------------------------------


def test_gcd_recovers_the_capacitance_and_the_ir_drop():
    curve = make_gcd_demo("condensador", capacitance=0.05, resistance=2.0,
                          current=1e-3, seed=1)
    result = analyse_gcd(curve)
    assert result.capacitance_f == pytest.approx(0.05, rel=0.05)
    assert result.resistance_ohm == pytest.approx(2.0, rel=0.2)


def test_the_coulombic_efficiency_comes_back():
    curve = make_gcd_demo("condensador", efficiency=0.9, seed=1)
    result = analyse_gcd(curve)
    assert result.coulombic_efficiency == pytest.approx(0.9, rel=0.03)


def test_a_plateau_is_reported_as_battery_behaviour():
    result = analyse_gcd(make_gcd_demo("bateria", seed=1))
    assert result.discharges[-1].linearity < 0.97
    assert any("meseta" in w for w in result.warnings)
    assert result.capacity_mah_per_g is not None


def test_energy_is_integrated_and_not_assumed():
    """½CV² is exact for a linear discharge and wrong for an off-centre
    plateau; the integral ∫V dq is right for both.

    The plateau has to be off-centre for the test to mean anything: a
    plateau at the middle of the window has a mean discharge potential of
    exactly ΔV/2, so ½CV² gives the right answer there by coincidence.
    Real battery plateaus are not centred."""
    capacitor = analyse_gcd(make_gcd_demo("condensador", seed=1)).discharges[-1]
    battery = analyse_gcd(make_gcd_demo("bateria", seed=1)).discharges[-1]
    assert capacitor.energy_half_cv2 == pytest.approx(capacitor.energy_j, rel=0.06)
    assert abs(battery.energy_half_cv2 - battery.energy_j) > 0.08 * battery.energy_j


def test_a_large_ir_drop_is_diagnosed():
    result = analyse_gcd(make_gcd_demo("condensador", resistance=80.0, seed=1))
    assert any("limitada por resistencia" in w for w in result.warnings)


def test_cycling_reports_fade_per_cycle_not_just_retention():
    curves = [make_gcd_demo("condensador", capacitance=0.05 * (1 - 0.01 * i), seed=i)
              for i in range(12)]
    result = analyse_cycling(curves)
    assert result.retention == pytest.approx(0.89, abs=0.03)
    assert result.fade_per_cycle is not None
    assert result.fade_per_cycle == pytest.approx(0.01, abs=0.003)


# -- mechanism --------------------------------------------------------


@pytest.mark.parametrize("kind", ECHEM_DEMOS)
def test_every_mechanism_is_classified_correctly(kind):
    verdict = classify_storage(
        analyse_cv(make_cv_demo(kind, seed=1)),
        analyse_gcd(make_gcd_demo(kind, seed=1)),
        analyse_rate_study(cv_rate_series(kind, seed=2)),
    )
    expected = {
        "condensador": "EDLC",
        "pseudocondensador": "pseudocapacitive",
        "bateria": "battery",
    }[kind]
    assert verdict.mechanism == expected, verdict.summary()
    assert verdict.confidence in ("media", "alta")


def test_a_battery_electrode_refuses_farads_per_gram():
    verdict = classify_storage(
        analyse_cv(make_cv_demo("bateria", seed=1)),
        analyse_gcd(make_gcd_demo("bateria", seed=1)),
    )
    assert not verdict.farads_are_appropriate
    assert "NO debe informarse en F/g" in verdict.summary()
    assert "mAh/g" in verdict.report_as or "C/g" in verdict.report_as


def test_a_b_value_near_one_does_not_argue_for_the_double_layer():
    """Both EDLC and pseudocapacitive give b ≈ 1; treating that as evidence
    for the double layer classified every pseudocapacitor as an EDLC."""
    rates = analyse_rate_study(cv_rate_series("pseudocondensador", seed=2))
    assert np.median([v.b for v in rates.b_values]) > 0.85
    verdict = classify_storage(
        analyse_cv(make_cv_demo("pseudocondensador", seed=1)), None, rates
    )
    assert verdict.mechanism == "pseudocapacitive"


def test_no_evidence_gives_no_verdict():
    verdict = classify_storage()
    assert verdict.mechanism == "desconocido"
    assert verdict.confidence == "ninguna"


# -- rate study -------------------------------------------------------


def test_a_diffusion_limited_peak_shows_a_low_b():
    rates = analyse_rate_study(cv_rate_series("bateria", seed=2))
    assert min(v.b for v in rates.b_values) < 0.7


def test_too_narrow_a_rate_range_is_flagged():
    rates = analyse_rate_study(cv_rate_series("condensador", rates=(0.02, 0.05), seed=2))
    assert any("una década" in w for w in rates.warnings)


def test_the_ecsa_carries_its_own_factor_of_three():
    rates = analyse_rate_study(cv_rate_series("condensador", seed=2))
    assert rates.ecsa_cm2 is not None
    low, high = rates.ecsa_range_cm2
    assert high / low > 2.0
    assert any("peor determinada" in w for w in rates.warnings)


def test_the_double_layer_window_must_be_declared_non_faradaic():
    rates = analyse_rate_study(cv_rate_series("condensador", seed=2))
    assert any("libre de corriente faradaica" in w for w in rates.warnings)


# -- impedance --------------------------------------------------------


def test_circuit_strings_parse():
    tree = parse_circuit("R0-(R1|Q1)-Wo1")
    assert [e.name for e in tree.elements()] == ["R0", "R1", "Q1", "Wo1"]
    with pytest.raises(CircuitError):
        parse_circuit("R0-(R1|Q1")
    with pytest.raises(CircuitError):
        parse_circuit("Z9")


def test_a_parallel_combination_is_smaller_than_either_branch():
    omega = np.array([100.0])
    tree = parse_circuit("R0|R1")
    for element, value in zip(tree.elements(), (10.0, 30.0)):
        element.values = [value]
    assert tree.impedance(omega)[0].real == pytest.approx(7.5)


def test_the_reflective_warburg_has_the_right_limits():
    """45° at high frequency, R/3 in the real part at low."""
    element = parse_circuit("Wo1").elements()[0]
    element.values = [100.0, 10.0]
    high = element.impedance(np.array([2 * np.pi * 1e4]))[0]
    low = element.impedance(np.array([2 * np.pi * 1e-4]))[0]
    assert high.real == pytest.approx(-high.imag, rel=0.02)
    assert low.real == pytest.approx(100.0 / 3.0, rel=0.05)


@pytest.mark.parametrize(
    "circuit", ["R0-(R1|C1)", "R0-(R1|Q1)", "R0-(R1|Q1)-Q2", "R0-(R1|Q1)-Wo1"]
)
def test_circuit_fits_recover_the_parameters_they_were_generated_from(circuit):
    spectrum = make_eis_demo(circuit, seed=3)
    fit = fit_circuit(spectrum, circuit)
    truth = spectrum.metadata["true_values"]
    for label, value in fit.values().items():
        if label in truth:
            assert value == pytest.approx(truth[label], rel=0.06), label
    assert fit.residual_pct < 1.5


def test_a_degenerate_circuit_is_flagged_by_its_uncertainties():
    """A Warburg in series inside a branch that is in parallel with a CPE
    is exactly reproducible by the CPE alone at n = 0.5. The fit finds an
    equally good, quite different answer — and says the parameters are not
    determined."""
    spectrum = make_eis_demo("R0-(R1-W1|Q1)", seed=3)
    fit = fit_circuit(spectrum, "R0-(R1-W1|Q1)")
    assert fit.residual_pct < 1.5
    assert any("no está determinado por los datos" in w for w in fit.warnings)


def test_clean_data_pass_kramers_kronig():
    for circuit in ("R0-(R1|Q1)", "R0-(R1|Q1)-Q2"):
        assert kramers_kronig(make_eis_demo(circuit, seed=3)).passes, circuit


def test_a_cell_that_drifts_fails_kramers_kronig():
    base = make_eis_demo("R0-(R1|Q1)", seed=5)
    drift = np.linspace(1.0, 1.6, base.n)
    result = kramers_kronig(Impedance(base.frequency, base.z * drift))
    assert not result.passes
    assert result.runs_ratio < 0.6
    assert "cambió MIENTRAS medías" in result.message


def test_a_cpe_is_not_reported_as_a_capacitance():
    fit = fit_circuit(make_eis_demo("R0-(R1|Q1)", seed=3), "R0-(R1|Q1)")
    assert any("NO son faradios" in w for w in fit.warnings)
    value = cpe_to_capacitance(2e-4, 0.88, 40.0)
    assert 0.0 < value < 2e-4


# -- catalysis --------------------------------------------------------


@pytest.mark.parametrize("slope", [40.0, 60.0, 120.0])
def test_the_tafel_slope_comes_back(slope):
    result = analyse_catalysis(make_lsv_demo("OER", tafel_mv_per_decade=slope, seed=1))
    assert result.tafel is not None and result.tafel.valid
    assert result.tafel.slope_mv_per_decade == pytest.approx(slope, rel=0.05)
    assert result.tafel.decades >= 1.0
    assert result.tafel.exchange_current_density == pytest.approx(1e-5, rel=0.25)


def test_a_short_current_range_refuses_a_tafel_slope():
    eta = np.linspace(0.20, 0.24, 40)
    j = 1e-5 * 10.0 ** (eta / 0.06)
    result = tafel_analysis(eta, j)
    assert not result.valid
    assert "décadas" in result.reason


def test_the_overpotential_at_the_benchmark_is_found():
    result = analyse_catalysis(make_lsv_demo("OER", tafel_mv_per_decade=60.0, seed=1))
    assert result.overpotential_at_benchmark is not None
    assert 0.05 < result.overpotential_at_benchmark < 0.6
    assert any("CONVENIO" in w for w in result.warnings)


def test_her_and_oer_use_different_equilibrium_potentials():
    her = analyse_catalysis(make_lsv_demo("HER", seed=1), reaction="HER")
    oer = analyse_catalysis(make_lsv_demo("OER", seed=1), reaction="OER")
    assert her.overpotential_at_benchmark is not None
    assert oer.overpotential_at_benchmark is not None
    assert her.overpotential_at_benchmark == pytest.approx(
        oer.overpotential_at_benchmark, abs=0.02
    )


def test_no_ir_correction_is_reported_as_a_bias():
    curve = make_lsv_demo("OER", seed=1)
    curve.electrode.resistance_ohm = None
    result = analyse_catalysis(curve)
    assert any("sesgado" in w for w in result.warnings)


def test_stability_is_always_raised():
    result = analyse_catalysis(make_lsv_demo("OER", seed=1))
    assert any("estabilidad" in w for w in result.warnings)


# -- io ---------------------------------------------------------------


def test_cv_round_trips_through_a_file(tmp_path):
    curve = make_cv_demo("condensador", seed=1)
    path = write_cv(curve, tmp_path / "cv.txt")
    back = read_cv(path)
    assert back.scan_rate == pytest.approx(curve.scan_rate)
    assert np.allclose(back.current, curve.current, atol=1e-12)


def test_gcd_round_trips_through_a_file(tmp_path):
    curve = make_gcd_demo("condensador", seed=1)
    back = read_gcd(write_gcd(curve, tmp_path / "gcd.txt"))
    assert back.n == curve.n
    assert np.allclose(back.potential, curve.potential, atol=1e-6)


def test_eis_round_trips_and_keeps_the_sign(tmp_path):
    spectrum = make_eis_demo("R0-(R1|Q1)", seed=3)
    back = read_eis(write_eis(spectrum, tmp_path / "eis.txt"))
    assert np.allclose(back.z, spectrum.z, rtol=1e-5)
    assert np.mean(back.z.imag) < 0


def test_a_negated_imaginary_column_is_detected(tmp_path):
    spectrum = make_eis_demo("R0-(R1|Q1)", seed=3)
    path = tmp_path / "neg.txt"
    path.write_text(
        "freq/Hz\tZre/ohm\t-Zim/ohm\n"
        + "\n".join(
            f"{f}\t{z.real}\t{-z.imag}" for f, z in zip(spectrum.frequency, spectrum.z)
        ),
        encoding="utf-8",
    )
    back = read_eis(path)
    assert np.allclose(back.z, spectrum.z, rtol=1e-5)


def test_a_cv_without_a_scan_rate_refuses(tmp_path):
    path = tmp_path / "bare.txt"
    path.write_text("0.0\t1e-3\n0.1\t1e-3\n0.2\t1e-3\n0.3\t1e-3\n0.4\t1e-3\n0.5\t1e-3\n")
    with pytest.raises(Exception, match="velocidad"):
        read_cv(path)


# -- the whole thing --------------------------------------------------


def test_analyse_sample_ties_the_measurements_together():
    result = analyse_sample(
        "demo",
        cv=make_cv_demo("pseudocondensador", seed=1),
        rate_series=cv_rate_series("pseudocondensador", seed=2),
        gcd=make_gcd_demo("pseudocondensador", seed=1),
        eis=make_eis_demo("R0-(R1|Q1)-Q2", seed=3),
    )
    assert result.storage is not None
    text = result.report()
    assert text.index("MECANISMO") < text.index("VOLTAMPEROMETRÍA")
    row = result.to_dict()
    assert row["mecanismo"] == "pseudocapacitive"
    assert row["C_gcd_F"] is not None


def test_disagreeing_capacitances_are_pointed_out():
    """The CV integral catches all the faradaic current; the galvanostatic
    discharge only what comes back."""
    result = analyse_sample(
        "demo",
        cv=make_cv_demo("pseudocondensador", seed=1),
        gcd=make_gcd_demo("pseudocondensador", seed=1),
    )
    assert any("veces la de la curva galvanostática" in w for w in result.warnings)


def test_a_battery_gets_the_farads_caveat_at_the_top_level():
    result = analyse_sample(
        "demo",
        cv=make_cv_demo("bateria", seed=1),
        gcd=make_gcd_demo("bateria", seed=1),
        rate_series=cv_rate_series("bateria", seed=2),
    )
    assert any("mecanismo es de tipo BATERÍA" in w for w in result.warnings)
