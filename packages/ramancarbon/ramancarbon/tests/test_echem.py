"""Electrochemistry: capacitance, mechanism, impedance, catalysis."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.echem.curve import (
    CurveError,
    Electrode,
    Impedance,
    Voltammogram,
)
from ramancarbon.echem.cv import (_current_at, analyse_cv,
                                  analyse_rate_study, capacitance)
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


def test_a_trimmed_window_is_divided_by_the_span_it_integrated():
    """Trimming the vertices and then dividing by the FULL window made
    every capacitance 4 % too small."""
    from ramancarbon.echem.cv import integrate_charge

    curve = make_cv_demo("condensador", capacitance=0.05, seed=1).cycles()[-1]
    _, _, span = integrate_charge(curve, trim=0.05)
    assert span < curve.span, "un recorte explícito tiene que reducir el vano"
    assert capacitance(curve)[0].farads == pytest.approx(0.05, rel=0.03)


def test_nothing_is_trimmed_when_there_is_no_transient():
    """The 2 % trim used to be unconditional, and on a record with no
    switching transient it threw away up to 8 % of the charge."""
    from ramancarbon.echem.cv import integrate_charge, vertex_transients

    curve = make_cv_demo("condensador", capacitance=0.05, seed=1).cycles()[-1]
    mask, reason = vertex_transients(curve)
    assert not mask.any()
    assert "sin transitorio" in reason
    _, _, span = integrate_charge(curve)
    assert span == pytest.approx(curve.span)


def test_a_planted_switching_transient_is_found():
    """And the detector has to earn its keep: a real spike must go."""
    from ramancarbon.echem.cv import vertex_transients

    curve = make_cv_demo("condensador", capacitance=0.05, seed=1).cycles()[-1]
    vertex = int(np.argmax(curve.potential))
    curve.current[vertex] *= 40.0
    mask, reason = vertex_transients(curve)
    assert mask[vertex], reason
    assert mask.sum() == 1, "sólo el punto plantado"


def test_a_cycle_that_reaches_both_limits_can_still_be_incomplete():
    """A record stopped three quarters of the way back down has the full
    span and three quarters of the charge. Measured on the user's
    ten-cycle file, the trailing cycle spanned 1.0003 of a 1.0003 V
    window and returned 5.87 F/g against 7.17 for the one before it."""
    curve = make_cv_demo("condensador", capacitance=0.05, cycles=3, seed=1)
    cut = int(0.90 * curve.n)
    truncated = Voltammogram(
        potential=curve.potential[:cut],
        current=curve.current[:cut],
        scan_rate=curve.scan_rate,
        electrode=curve.electrode,
        cycle=None if curve.cycle is None else curve.cycle[:cut],
    )
    last = truncated.cycles()[-1]
    assert last.span == pytest.approx(truncated.span, rel=0.02), (
        "el ciclo truncado alcanza los dos extremos, que es lo que hace "
        "que una prueba de vano sola lo acepte"
    )
    assert last.traversed < 1.8 * last.span
    complete = [c.name for c in truncated.complete_cycles()]
    assert last.name not in complete


def test_the_real_clock_is_preferred_over_the_reconstructed_one():
    from ramancarbon.echem.cv import elapsed_time

    curve = make_cv_demo("condensador", capacitance=0.05, seed=1).cycles()[-1]
    axis, origin = elapsed_time(curve)
    assert "reconstruido" in origin
    with_clock = Voltammogram(
        potential=curve.potential,
        current=curve.current,
        scan_rate=curve.scan_rate,
        electrode=curve.electrode,
        time=axis,
    )
    measured, origin = elapsed_time(with_clock)
    assert "reloj" in origin
    assert measured == pytest.approx(axis)


def test_a_join_is_not_counted_as_sweeping():
    """A branch of a voltammogram is generally two disjoint runs, and
    ``cumsum(|dE|)`` across the join counted a potential the electrode
    never swept: measured, 39.99 s of sweeping instead of 20.00."""
    from ramancarbon.echem.cv import elapsed_time

    curve = make_cv_demo("condensador", capacitance=0.05, seed=1).cycles()[-1]
    whole, _ = elapsed_time(curve)
    half = curve.n // 2
    keep = np.ones(curve.n, dtype=bool)
    keep[half:half + 50] = False
    gapped = curve._subset(keep, "con hueco")
    spliced, origin = elapsed_time(gapped)
    assert "empalme" in origin
    assert spliced[-1] < whole[-1]


def test_the_two_conventions_agree_on_a_balanced_loop():
    from ramancarbon.echem.cv import coulombic_efficiency

    curve = make_cv_demo("condensador", capacitance=0.05, seed=1).cycles()[-1]
    assert coulombic_efficiency(curve) == pytest.approx(1.0, abs=0.02)
    loop, sweep = capacitance(curve)
    assert sweep.farads == pytest.approx(loop.farads, rel=0.03)


def test_the_net_charge_measures_the_parasitic_current():
    """``Q_ox − Q_red = ∫i dt`` is the charge that did not come back, so
    its slope against the cycle's duration is the leak — exactly, with
    an intercept of zero as the check on the model."""
    from ramancarbon.echem.cv import leakage_from_rates

    leak = 2e-6
    curves = []
    for rate in (0.005, 0.01, 0.02, 0.05):
        curve = make_cv_demo("condensador", capacitance=0.05,
                             scan_rate=rate, seed=1).cycles()[-1]
        curve.current = curve.current + leak
        curves.append(curve)

    fit = leakage_from_rates(curves)
    assert fit is not None
    assert fit.leakage_a == pytest.approx(leak, rel=0.05)
    assert fit.r_squared > 0.999
    assert abs(fit.offset_c) < 0.02 * max(fit.oxidation_charges)


def test_two_rates_are_not_enough_to_fit_a_leak():
    from ramancarbon.echem.cv import leakage_from_rates

    curves = [
        make_cv_demo("condensador", capacitance=0.05, scan_rate=rate,
                     seed=1).cycles()[-1]
        for rate in (0.005, 0.05)
    ]
    assert leakage_from_rates(curves) is None


def test_a_one_sided_leak_fools_the_loop_convention_not_the_reduction():
    """This is the real case: a current that only flows while the
    potential goes up adds to Q_ox and not to Q_red, so it inflates the
    loop convention by more the slower the scan — the user's 11.33 F/g at
    5 mV/s against 5.81 at 50, which reads as a transport limitation and
    is not one."""
    from ramancarbon.echem.cv import coulombic_efficiency, reversible_capacitance

    # Twice the capacitive current of the slow scan, which is the regime
    # the real files are in: at 5 mV/s a 0.11 mF electrode draws 0.55 µA
    # and leaks 0.41 µA.
    leak = 5e-4
    curves = []
    for rate in (0.005, 0.05):
        curve = make_cv_demo("condensador", capacitance=0.05,
                             scan_rate=rate, seed=1).cycles()[-1]
        rising = np.gradient(curve.potential) > 0
        curve.current = curve.current + leak * rising
        curves.append(curve)

    slow, fast = (capacitance(c)[0].farads for c in curves)
    assert slow > 1.5 * fast, "el convenio del lazo tiene que dejarse engañar"
    honest = [reversible_capacitance(c).farads for c in curves]
    assert honest[0] == pytest.approx(honest[1], rel=0.05)
    assert coulombic_efficiency(curves[0]) < coulombic_efficiency(curves[1])


def test_a_leak_withdraws_the_b_value_as_evidence_of_diffusion():
    """b = 0.53 classified a leaky capacitor as a battery electrode and
    told the user to report it in mAh/g. A constant current dilutes a
    capacitive one by more the slower the scan, which bends b downwards
    with no diffusion anywhere."""
    from ramancarbon.echem.cv import analyse_rate_study

    leak = 5e-4
    series = []
    for rate in (0.005, 0.01, 0.02, 0.05):
        curve = make_cv_demo("condensador", capacitance=0.05,
                             scan_rate=rate, seed=1).cycles()[-1]
        rising = np.gradient(curve.potential) > 0
        curve.current = curve.current + leak * rising
        series.append(curve)

    study = analyse_rate_study(series)
    assert study.b_values
    assert min(b.b for b in study.b_values) < 0.8, (
        "la fuga tiene que doblar el b hacia abajo, que es la premisa"
    )
    verdict = classify_storage(rates=study)
    assert verdict.mechanism != "battery"
    assert any("no cuenta como difusión" in text for text in verdict.against)


def test_the_three_laws_for_the_current_are_compared_on_equal_terms():
    from ramancarbon.echem.cv import compare_current_laws

    series = [
        make_cv_demo("condensador", capacitance=0.05, scan_rate=rate,
                     seed=1).cycles()[-1]
        for rate in (0.005, 0.01, 0.02, 0.05)
    ]
    laws = compare_current_laws(series)
    assert laws
    for law in laws:
        assert set(law.residuals) == {"potencia", "dunn", "fuga"}
        # A clean capacitor: the leak law must find no leak worth having.
        assert abs(law.leakage_a) < 0.02 * law.capacitance_f * 0.05
    assert compare_current_laws(series[:3]) == [], (
        "tres velocidades no bastan para tres leyes de dos parámetros"
    )


def test_the_cycle_to_report_can_be_chosen_by_hand():
    curve = make_cv_demo("condensador", capacitance=0.05, cycles=4, seed=1)
    default, why = curve.reporting_cycle()
    assert "ÚLTIMO completo" in why

    curve.preferred_cycle = 2
    chosen, why = curve.reporting_cycle()
    assert "elegido a mano" in why
    assert chosen.name == curve.cycles()[1].name
    # And it reaches the capacitance, which is the point of putting it on
    # the curve rather than in a function argument.
    assert capacitance(curve)[0].note.startswith("ciclo 2")

    curve.preferred_cycle = 99
    with pytest.raises(CurveError, match="se ha pedido el ciclo 99"):
        curve.reporting_cycle()


def test_choosing_an_incomplete_cycle_says_so():
    curve = make_cv_demo("condensador", capacitance=0.05, cycles=3, seed=1)
    cut = int(0.90 * curve.n)
    truncated = Voltammogram(
        potential=curve.potential[:cut], current=curve.current[:cut],
        scan_rate=curve.scan_rate, electrode=curve.electrode,
        cycle=None if curve.cycle is None else curve.cycle[:cut],
    )
    truncated.preferred_cycle = len(truncated.cycles())
    _, why = truncated.reporting_cycle()
    assert "no está completo" in why


def test_the_dunn_coefficients_cannot_be_negative():
    """A capacitive current that subtracts from the measured one is not a
    capacitive current, and drawn unclipped it shades a region outside
    the voltammogram — which is the first thing anyone notices about the
    figure and the last thing the percentages admit."""
    from ramancarbon.echem.cv import dunn_analysis
    from ramancarbon.examples.demo_data import cv_rate_series

    series = cv_rate_series("hibrido", seed=2)
    constrained = dunn_analysis(series)
    assert np.all(constrained.k1_anodic >= 0.0)
    assert np.all(constrained.k2_anodic >= 0.0)

    free = dunn_analysis(series, nonnegative=False)
    assert free.k1.shape == constrained.k1.shape


def test_the_capacitive_band_is_drawn_inside_the_loop():
    from ramancarbon.echem.cv import dunn_analysis
    from ramancarbon.examples.demo_data import cv_rate_series

    series = cv_rate_series("hibrido", seed=2)
    analysis = dunn_analysis(series)
    fastest = max(series, key=lambda c: c.scan_rate)
    anodic, cathodic = analysis.capacitive_inside(fastest)[:2]
    for branch, side in ((anodic, "media"), (cathodic, "catodica")):
        measured = np.array(
            [_current_at(fastest, float(p), side) for p in analysis.potentials],
            dtype=float)
        good = np.isfinite(measured)
        assert np.all(np.abs(branch[good]) <= np.abs(measured[good]) + 1e-15)


def test_a_pure_pseudocapacitor_is_still_fully_capacitive():
    """The constraint must not buy a readable figure with a biased
    number: clipping k₁ at min(|i|/ν) looked right and returned 91 % on
    an electrode that is 100 % capacitive by construction."""
    from ramancarbon.echem.cv import dunn_analysis
    from ramancarbon.examples.demo_data import cv_rate_series

    analysis = dunn_analysis(cv_rate_series("pseudocondensador", seed=2))
    assert min(analysis.fractions.values()) > 0.95


def test_a_leak_correction_has_to_come_from_outside_the_rate_fit():
    from ramancarbon.echem.cv import leakage_from_rates, without_parasitic

    leak = 2e-6
    series = []
    for rate in (0.005, 0.01, 0.02, 0.05):
        curve = make_cv_demo("condensador", capacitance=0.05,
                             scan_rate=rate, seed=1).cycles()[-1]
        curve.current = curve.current + leak
        series.append(curve)

    measured = leakage_from_rates(series)
    fixed, note = without_parasitic(series, measured.leakage_a)
    assert "balance de carga" in note
    assert fixed[0].current == pytest.approx(
        series[0].current - measured.leakage_a, abs=1e-12)
    assert fixed[0].metadata["corriente_parasita_restada_a"] == pytest.approx(
        measured.leakage_a)
    # The corrected series must balance, which is what says the leak was
    # the whole of the imbalance.
    assert leakage_from_rates(fixed).leakage_a == pytest.approx(0.0, abs=0.05 * leak)


def test_a_repeat_at_the_same_rate_is_used_as_a_drift_control():
    from ramancarbon.echem.cv import analyse_rate_study, drift_check

    series = [
        make_cv_demo("condensador", capacitance=0.05, scan_rate=rate,
                     seed=1).cycles()[-1]
        for rate in (0.05, 0.005, 0.01, 0.02, 0.05)
    ]
    series[-1].current = series[-1].current * 0.8  # 20 % lost
    check = drift_check(series)
    assert check is not None
    assert check.rate == pytest.approx(0.05)
    assert check.fade_percent == pytest.approx(20.0, abs=2.0)
    assert check.degraded

    study = analyse_rate_study(series)
    assert len(study.rates) == 4, "la velocidad repetida no entra dos veces"
    assert study.drift is check or study.drift is not None
    assert any("CAMBIÓ" in w for w in study.warnings)


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
        # Broad overlapping redox plus a diffusive TAIL is still a
        # pseudocapacitor. What makes an electrode a battery is a narrow,
        # dominant, diffusion-limited pair -- a phase transition -- and
        # that is the next entry.
        "hibrido": "pseudocapacitive",
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


# -- the circuit library ------------------------------------------------

def _from_circuit(text, values, noise=0.003, seed=1, decades=(5, -2), n=60):
    """A synthetic spectrum built from a circuit with known parameters."""
    from ramancarbon.echem.eis import parse_circuit

    frequency = np.logspace(decades[0], decades[1], n)
    tree = parse_circuit(text)
    for element in tree.elements():
        element.values = list(values[element.name])
    z = tree.impedance(2.0 * np.pi * frequency)
    rng = np.random.default_rng(seed)
    z = z * (1.0 + rng.normal(0.0, noise, z.size)
             + 1j * rng.normal(0.0, noise, z.size))
    return Impedance(frequency=frequency, z=z)


def test_every_bundled_circuit_parses_and_says_what_it_is_for():
    """A name and a string are not enough. Every equivalent circuit fits
    SOMETHING; what separates them is what the parameters mean
    afterwards, so each one carries its case and its trap."""
    from ramancarbon.echem.eis import LIBRARY, TEMPLATES, parse_circuit

    assert len(TEMPLATES) >= 20
    for template in TEMPLATES:
        parse_circuit(template.circuit)
        assert template.use.strip(), template.name
        assert template.caution.strip(), template.name
        assert template.origin == "programa"
    assert set(LIBRARY) == {t.name for t in TEMPLATES}
    assert len({t.circuit for t in TEMPLATES}) == len(TEMPLATES), (
        "two names for the same circuit")


def test_the_porous_line_gives_a_third_of_the_ionic_resistance():
    """De Levie / Bisquert, and the factor of three is the whole point.
    At low frequency the pore is charged along its length, so its real
    part settles at R_ion/3, NOT at R_ion — and R_ion is the quantity
    that limits a supercapacitor's power. Fitting the 45-degree region
    with a Warburg instead reports a diffusion coefficient for an ion
    that is not diffusing anywhere."""
    from ramancarbon.echem.eis import _z_t

    omega = 2.0 * np.pi * np.logspace(-3, 6, 400)
    z = _z_t(omega, 30.0, 0.8, 1.0)
    assert np.isfinite(z).all(), "the coth overflowed at the high-frequency end"
    assert z.real[0] == pytest.approx(10.0, rel=0.01)
    # And the low-frequency tail is the wall capacitance, |Z''| = 1/(wC).
    assert (-z.imag[0]) * omega[0] * 0.8 == pytest.approx(1.0, rel=0.01)
    # 45 degrees at high frequency: a distributed RC, not a capacitor.
    assert np.degrees(np.angle(z[-1])) == pytest.approx(-45.0, abs=1.0)


def test_a_porous_electrode_recovers_its_own_ionic_resistance():
    from ramancarbon.echem.eis import fit_circuit

    spectrum = _from_circuit("R0-T1", {"R0": (1.5,), "T1": (36.0, 0.8, 0.95)})
    fit = fit_circuit(spectrum, "poroso")
    assert fit.converged
    values = fit.values()
    assert values["T1.Ri"] == pytest.approx(36.0, rel=0.05)
    assert values["T1.Q"] == pytest.approx(0.8, rel=0.05)
    assert values["T1.n"] == pytest.approx(0.95, abs=0.02)


def test_a_gerischer_is_not_a_cpe():
    """Both make a depressed arc and both fit. The difference is that
    the Gerischer's k is a rate constant in s-1 that can be checked
    against a kinetic measurement, and a CPE's n is not."""
    from ramancarbon.echem.eis import _z_g, fit_circuit

    omega = 2.0 * np.pi * np.logspace(-3, 6, 400)
    z = _z_g(omega, 20.0, 5.0)
    assert z.real[0] == pytest.approx(20.0 / np.sqrt(5.0), rel=0.01)
    assert np.degrees(np.angle(z[-1])) == pytest.approx(-45.0, abs=1.0)

    spectrum = _from_circuit(
        "R0-(G1|Q1)", {"R0": (2.0,), "G1": (15.0, 3.0), "Q1": (5e-4, 0.9)})
    fit = fit_circuit(spectrum, "gerischer")
    assert fit.values()["G1.Z0"] == pytest.approx(15.0, rel=0.05)
    assert fit.values()["G1.k"] == pytest.approx(3.0, rel=0.1)


def test_a_fixed_parameter_stays_put_and_is_named():
    """Fixing is not free: the parameter stops contributing a degree of
    freedom, so every OTHER uncertainty comes out smaller. A reader who
    cannot see which were held cannot read the column next to them."""
    from ramancarbon.echem.eis import fit_circuit

    spectrum = _from_circuit(
        "R0-(R1|Q1)", {"R0": (2.0,), "R1": (40.0,), "Q1": (2e-4, 0.88)})
    held = fit_circuit(spectrum, "randles_cpe",
                       initial={"Q1.n": 0.80}, fixed=["Q1.n"])

    assert held.values()["Q1.n"] == 0.80, "the fit moved a fixed parameter"
    assert held.fixed == ["Q1.n"]
    assert "Q1.n" not in held.errors, "a held parameter has no uncertainty"
    assert "FIJADO" in held.summary()
    assert "fijados" in held.summary()

    # Fixing it at the wrong value does not wreck the fit: the error
    # moves into its neighbours instead, which is exactly why it has to
    # be declared.
    free = fit_circuit(spectrum, "randles_cpe")
    assert free.values()["Q1.n"] == pytest.approx(0.88, abs=0.02)
    assert held.residual_pct > free.residual_pct


def test_fixing_everything_is_refused_and_so_is_a_name_that_is_not_there():
    from ramancarbon.echem.eis import fit_circuit

    spectrum = _from_circuit("R0-(R1|C1)", {"R0": (2.0,), "R1": (40.0,),
                                            "C1": (1e-4,)})
    with pytest.raises(CircuitError, match="fijados todos"):
        fit_circuit(spectrum, "randles", fixed=["R0.R", "R1.R", "C1.C"])
    with pytest.raises(CircuitError, match="no hay ningún parámetro"):
        fit_circuit(spectrum, "randles", fixed=["R7.R"])


def test_user_circuits_live_outside_the_package(tmp_path, monkeypatch):
    """Same rule as the user's Raman phases: a package upgrade replaces
    what is inside the package, and a circuit somebody worked out for
    their own cell would go with it."""
    from ramancarbon.echem import eis

    monkeypatch.setattr(eis, "user_circuit_file",
                        lambda: tmp_path / "circuitos_usuario.json")

    assert eis.user_circuits() == {}
    eis.save_user_circuit("mi_celda", "R0-(R1|Q1)-T1")
    assert eis.user_circuits()["mi_celda"] == "R0-(R1|Q1)-T1"
    assert eis.circuit_for("mi_celda") == "R0-(R1|Q1)-T1"

    names = [t.name for t in eis.circuit_templates()]
    assert "mi_celda" in names and "randles_cpe" in names
    mine = next(t for t in eis.circuit_templates() if t.name == "mi_celda")
    assert mine.origin == "usuario"

    # It is checked when it is saved, not a week later in the middle of
    # an analysis.
    with pytest.raises(CircuitError):
        eis.save_user_circuit("roto", "R0-(R1|Q1")
    with pytest.raises(CircuitError):
        eis.save_user_circuit("   ", "R0")

    assert eis.delete_user_circuit("mi_celda") is True
    assert eis.delete_user_circuit("randles_cpe") is False, (
        "a bundled circuit must not be deletable: the deletion would be "
        "written to a file the next upgrade replaces and would come back")


def test_an_unreadable_user_circuit_file_is_ignored_not_fatal(tmp_path, monkeypatch):
    from ramancarbon.echem import eis

    path = tmp_path / "circuitos_usuario.json"
    path.write_text("{ esto no es json,, }", encoding="utf-8")
    monkeypatch.setattr(eis, "user_circuit_file", lambda: path)
    assert eis.user_circuits() == {}
    assert [t.name for t in eis.circuit_templates()][:1] != []


def test_the_section_lets_you_edit_a_circuit_and_hold_its_parameters(tmp_path,
                                                                     monkeypatch):
    """What the user asked for: more circuits in the list, and the list
    editable rather than a set of presets to accept or leave."""
    from ramancarbon.echem import eis
    from ramancarbon.gui.echem_state import EchemSession

    monkeypatch.setattr(eis, "user_circuit_file",
                        lambda: tmp_path / "circuitos_usuario.json")
    session = EchemSession()
    assert len(session.circuit_choices()) >= 20

    # A name picked from the list fills the box with its string, which is
    # how the notation gets learnt.
    assert session.set_circuit("poroso")
    assert session.circuit_text() == "R0-T1"
    use, caution = session.circuit_note()
    assert "poroso" in use.lower() and caution

    # And a string typed by hand is accepted on the same footing.
    assert session.set_circuit("R0-(R1|Q1)-T1")
    assert session.circuit_text() == "R0-(R1|Q1)-T1"
    labels = [label for label, _, _ in session.circuit_parameters()]
    assert labels == ["R0.R", "R1.R", "Q1.Q", "Q1.n", "T1.Ri", "T1.Q", "T1.n"]

    # Nonsense is refused where it was typed, not ten seconds into an
    # analysis.
    assert session.set_circuit("R0-(R1|Q1") is False
    assert session.circuit_text() == "R0-(R1|Q1)-T1", "a bad edit was kept"
    assert session.messages and session.messages[-1][0] == "error"

    session.set_circuit_parameter("T1.n", value=0.92, fixed=True)
    assert session.circuit_fixed == {"T1.n"}
    assert dict((label, v) for label, v, _ in session.circuit_parameters())["T1.n"] == 0.92

    # Changing the circuit clears them, because R1.R in one circuit is
    # not R1.R in another.
    assert session.set_circuit("randles_cpe")
    assert session.circuit_fixed == set()
    assert session.circuit_initial == {}

    assert session.save_circuit("mi_celda", "R0-(R1|Q1)-T1")
    assert "mi_celda" in session.circuit_choices()
    assert session.delete_circuit("mi_celda")
    assert session.delete_circuit("randles_cpe") is False


def test_the_fitted_table_says_which_parameters_were_held():
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession()
    session.eis = _from_circuit(
        "R0-(R1|Q1)", {"R0": (2.0,), "R1": (40.0,), "Q1": (2e-4, 0.88)})
    session.set_circuit("randles_cpe")
    session.set_circuit_parameter("Q1.n", value=0.88, fixed=True)
    assert session.analyse() is not None

    rows = {row[0]: row for row in session.circuit_rows()}
    assert rows["Q1.n"][2] == "fijado" and rows["Q1.n"][3] == "sí"
    assert rows["R1.R"][3] == ""
    assert float(rows["R1.R"][1]) == pytest.approx(40.0, rel=0.05)


# -- the section's new tabs ---------------------------------------------

def _demo_session():
    from ramancarbon.echem.curve import Electrode
    from ramancarbon.examples.demo_data import (
        cv_rate_series,
        make_cv_demo,
        make_eis_demo,
        make_gcd_demo,
    )
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession()
    session.electrode = Electrode(mass_mg=2.0, area_cm2=1.0, ph=14.0,
                                  resistance_ohm=2.0)
    session.add_curves(
        cv=make_cv_demo("pseudocondensador", seed=1),
        rate_series=cv_rate_series("pseudocondensador", seed=2),
        gcd=make_gcd_demo("pseudocondensador", seed=1),
        gcd_series=[make_gcd_demo("pseudocondensador", current=c, seed=k)
                    for k, c in enumerate((0.5e-3, 2e-3, 5e-3), start=10)],
        eis=make_eis_demo("R0-(R1|Q1)-Q2", seed=3),
    )
    return session


def test_a_ragone_point_per_discharge_curve_and_the_times_check_out():
    """E and P both per KILOGRAM, and the consistency P = E·3600/t needs
    no closed case: it just has to give back the discharge time. Reported
    as W/g instead, the power axis of every Ragone plot moves three
    decades."""
    session = _demo_session()
    points = session.ragone_points()
    assert len(points) == 4, [p[2] for p in points]

    times = sorted(3600.0 * energy / power for energy, power, _ in points)
    assert all(1.0 < t < 600.0 for t in times), times
    # 0.5 to 5 mA is a factor of ten in current, so a factor of ten in
    # time.
    assert times[-1] / times[0] == pytest.approx(10.0, rel=0.25)
    # And the energy is nearly flat while the power spans the range: that
    # is what a Ragone plot of a capacitor looks like.
    energies = [e for e, _, _ in points]
    powers = [p for _, p, _ in points]
    assert max(powers) / min(powers) > 10.0
    assert max(energies) / min(energies) < 1.5


def test_the_drt_table_gives_a_capacitance_per_process():
    """Two processes with the same resistance and time constants a decade
    apart are different things, and tau/R is the column that says which."""
    session = _demo_session()
    assert session.analyse() is not None
    rows = session.drt_rows()
    assert rows, "the DRT found nothing on the demo spectrum"
    for tau, resistance, capacitance_mf, share in rows:
        assert float(tau) > 0 and float(resistance) > 0
        assert float(capacitance_mf) == pytest.approx(
            1e3 * float(tau) / float(resistance), rel=1e-3)
        assert share.endswith("%")
    assert sum(float(r[3].rstrip(" %")) for r in rows) == pytest.approx(
        100.0, abs=1.0)


def test_the_drt_knows_where_the_data_stop():
    """Outside the measured range of time constants gamma is not
    determined by the data and does not sit quietly at zero: the
    unmeasured slow end collects whatever the diffusion tail implies. On
    the demo spectrum that is a spike seven hundred times the real peaks,
    and a plot autoscaled to it is a flat line with a wall at one end."""
    from ramancarbon.echem.eis import drt
    from ramancarbon.examples.demo_data import make_eis_demo

    spectrum = make_eis_demo("R0-(R1|Q1)-Q2", seed=3)
    result = drt(spectrum)
    low, high = result.measured_s
    assert low == pytest.approx(1.0 / (2 * np.pi * spectrum.frequency.max()),
                                rel=1e-6)
    assert high == pytest.approx(1.0 / (2 * np.pi * spectrum.frequency.min()),
                                 rel=1e-6)
    assert all(low <= peak <= high for peak in result.peaks_s), (
        "a peak was reported outside the measured range")

    tau = np.asarray(result.tau_s)
    inside = (tau >= low) & (tau <= high)
    assert result.gamma.max() > 5.0 * result.gamma[inside].max(), (
        "this spectrum no longer has the pile-up the plot limits guard "
        "against; pick another one for this test")


def test_the_three_capacitances_are_reported_with_their_conditions():
    """The same electrode measured three ways is not the same
    measurement. What a device delivers is the GCD value; the EIS one is
    measured with 10 mV about a fixed point where nothing is
    rate-limited, and is an upper bound the device never sees."""
    session = _demo_session()
    assert session.analyse() is not None
    rows = session.capacitance_rows()
    methods = [row[0] for row in rows]
    assert {"CV", "GCD", "EIS"} <= set(methods), methods
    for _, condition, farads, _ in rows:
        assert condition.strip(), "a capacitance without its condition"
        assert float(farads) > 0

    spread = session.result.capacitance.spread
    assert spread is not None and spread > 0.3
    assert any("difieren" in w for w in session.result.capacitance.warnings), (
        "a spread above 30 % has to be said out loud")
    assert any("GCD" in w for w in session.result.capacitance.warnings), (
        "and which of the three is the one a device delivers")


def test_the_complex_capacitance_comes_from_the_data_not_a_circuit():
    session = _demo_session()
    result = session.analyse()
    complex_c = result.complex_capacitance
    assert complex_c is not None
    assert complex_c.relaxation_s is not None
    # tau_0 = 1/(2 pi f_0) at the maximum of C''.
    peak = int(np.argmax(complex_c.imaginary))
    assert complex_c.relaxation_frequency_hz == pytest.approx(
        complex_c.frequency[peak], rel=1e-6)
    assert complex_c.relaxation_s == pytest.approx(
        1.0 / (2 * np.pi * complex_c.frequency[peak]), rel=1e-6)
    # And it is positive, which it is not if the code was written against
    # the -Z'' of a Nyquist plot.
    assert complex_c.low_frequency_capacitance > 0


def test_a_gcd_series_is_ordered_by_its_own_current():
    """By the MEDIAN of |i|, not its maximum: the rest step between the
    branches is at zero and the switch between them can overshoot, and
    either would order the series by an artefact."""
    from ramancarbon.gui.echem_state import EchemSession, _typical_current
    from ramancarbon.examples.demo_data import make_gcd_demo

    session = EchemSession()
    for current in (5e-3, 0.5e-3, 2e-3):
        session.gcd_series.append(make_gcd_demo("condensador", current=current))
    session.gcd_series.sort(key=_typical_current)
    assert [round(1e3 * _typical_current(c), 1) for c in session.gcd_series] == [
        0.5, 2.0, 5.0]

    # Immune to an overshoot at a reversal, which a maximum is not: the
    # first version filtered on a fraction of the MAXIMUM, so a single
    # 50 A spike threw away every real point and the "typical current"
    # came out as the spike.
    curve = make_gcd_demo("condensador", current=3e-3)
    curve.current[0] = 50.0
    assert _typical_current(curve) == pytest.approx(3e-3, rel=0.05)


# -- the choices a rate study needs -------------------------------------

def test_dunn_can_be_run_on_either_branch_or_the_whole_cycle():
    """The anodic sweep alone is the usual published choice, and it is a
    CHOICE: on a material whose oxidation and reduction are not mirror
    images the two halves give different coefficients, and that
    difference is information."""
    from ramancarbon.echem.cv import SWEEP_CHOICES, dunn_analysis
    from ramancarbon.examples.demo_data import cv_rate_series

    curves = cv_rate_series("hibrido", seed=2)
    results = {}
    for key, label in SWEEP_CHOICES:
        analysis = dunn_analysis(curves, sweep=key)
        assert analysis.sweep == key
        assert label.split(":")[0] in analysis.summary() or key in analysis.summary()
        results[key] = analysis.fractions

    slowest = min(results["media"])
    assert results["media"][slowest] != pytest.approx(
        results["catodica"][slowest], abs=1e-6), (
        "the two branches gave identical answers; this demo is symmetric "
        "and cannot show what the option is for")

    with pytest.raises(CurveError, match="rama desconocida"):
        dunn_analysis(curves, sweep="diagonal")


def test_the_hybrid_demo_has_a_diffusive_part_to_separate():
    """The plain pseudocapacitor is surface-confined by construction, so
    Dunn correctly returns ~100 % for it and there is nothing to see. A
    demonstration of the method needs an electrode that has both."""
    from ramancarbon.echem.cv import dunn_analysis
    from ramancarbon.examples.demo_data import cv_rate_series

    plain = dunn_analysis(cv_rate_series("pseudocondensador", seed=2))
    hybrid = dunn_analysis(cv_rate_series("hibrido", seed=2))

    assert min(plain.fractions.values()) > 0.95
    slow = min(hybrid.fractions)
    fast = max(hybrid.fractions)
    assert 0.55 < hybrid.fractions[slow] < 0.9, hybrid.fractions
    # And it RISES with scan rate, which is the whole signature: the
    # diffusive term grows as sqrt(nu) while the surface one grows as nu.
    assert hybrid.fractions[fast] > hybrid.fractions[slow] + 0.08


def test_three_capacitance_conventions_and_the_spread_is_the_diagnostic():
    """On an ideal capacitor they agree to about a per cent; on a plateau
    the delta-V convention over-reports by exactly what the curve bends."""
    from ramancarbon.echem.gcd import analyse_gcd
    from ramancarbon.examples.demo_data import make_gcd_demo

    ideal = analyse_gcd(make_gcd_demo("condensador", capacitance=0.05, seed=1))
    branch = ideal.discharges[-1]
    values = [v for v in branch.capacitances().values() if v]
    assert len(values) == 3
    # The slope convention is the closed case: I/|dV/dt| on a straight
    # discharge must give back the capacitance the demo was built from.
    assert branch.capacitance_slope_f == pytest.approx(0.05, rel=0.02)
    assert branch.capacitance_spread < 0.05
    assert not any("convenios" in w for w in ideal.warnings)

    curved = analyse_gcd(make_gcd_demo("bateria", seed=1))
    bent = curved.discharges[-1]
    assert bent.capacitance_spread > 0.10
    assert bent.capacitance_f > bent.capacitance_energy_f, (
        "the delta-V convention must be the one that over-reports on a "
        "plateau; that is why it needs saying")
    assert any("convenios" in w for w in curved.warnings)


def test_the_overpotential_is_reported_at_more_than_one_current_density():
    """A catalyst that is excellent at 10 mA/cm2 and collapses at 100 is
    one nobody can use, and only the pair says so. What the curve did not
    reach is reported as not reached, never extrapolated."""
    from ramancarbon.echem.evaluate import BENCHMARK_DENSITIES, analyse_catalysis
    from ramancarbon.examples.demo_data import make_lsv_demo

    curve = make_lsv_demo("OER", seed=1)
    curve.electrode = Electrode(area_cm2=1.0, ph=14.0, resistance_ohm=2.0)
    result = analyse_catalysis(curve, reaction="OER")

    assert set(result.overpotentials) == set(BENCHMARK_DENSITIES)
    reached = {k: v for k, v in result.overpotentials.items() if v is not None}
    assert reached, result.overpotentials
    # Monotonic: more current costs more overpotential.
    levels = sorted(reached)
    for low, high in zip(levels, levels[1:]):
        assert reached[high] > reached[low]
    assert result.overpotentials[10.0] == pytest.approx(
        result.overpotential_at_benchmark, rel=1e-6)

    # And the polarisation curve itself is carried, so the section can
    # draw the shape and not only its logarithm.
    assert result.curve_j is not None and result.curve_eta is not None
    assert result.curve_j.size == result.curve_eta.size > 10
    assert np.all(np.diff(result.curve_j) >= 0), "sorted by current density"


def test_dunn_carries_both_branches_for_the_published_figure():
    """The figure the field prints is the CLOSED voltammogram with the
    surface-controlled region shaded between the two capacitive curves
    and the diffusive part filling out to the measured trace on both
    branches. Half a cycle with a line on it is the same arithmetic and
    not the same figure."""
    from ramancarbon.echem.cv import dunn_analysis
    from ramancarbon.examples.demo_data import cv_rate_series

    curves = cv_rate_series("hibrido", seed=2)
    analysis = dunn_analysis(curves, sweep="media")

    assert analysis.k1_anodic is not None and analysis.k1_cathodic is not None
    slow = min(analysis.rates)
    anodic, cathodic = analysis.branch_currents(slow)
    assert anodic is not None and cathodic is not None
    assert anodic.shape == cathodic.shape == analysis.potentials.shape

    # The two branches straddle zero: the anodic capacitive current is
    # positive and the cathodic one negative, which is what makes the
    # shaded band a band and not a line.
    assert np.median(anodic) > 0 > np.median(cathodic)

    # And the sweep choice does not change what the figure can draw.
    other = dunn_analysis(curves, sweep="catodica")
    assert other.k1_anodic is not None and other.k1_cathodic is not None
    assert other.k1_anodic == pytest.approx(analysis.k1_anodic)


def test_an_overpotential_below_the_measured_range_is_not_invented():
    """np.interp CLAMPS: asked for a density below the measured range it
    returns the first value without saying so, and a guard on the upper
    end alone lets that through. On a real HER curve whose active branch
    starts at 250 mA/cm2 that reported the SAME overpotential at 10, 50
    and 100 -- the potential at the very end of the sweep, three times."""
    from ramancarbon.echem.evaluate import analyse_catalysis

    n = 300
    potential = np.linspace(0.0, -0.12, n)
    density = np.linspace(250.0, 900.0, n)        # mA/cm2, never below 250
    curve = Voltammogram(
        potential=potential, current=-density * 1e-3, scan_rate=0.005,
        electrode=Electrode(area_cm2=1.0, reference="RHE",
                            resistance_ohm=0.0),
        name="HER parcial")
    result = analyse_catalysis(curve, reaction="HER")

    assert result.overpotentials == {10.0: None, 50.0: None, 100.0: None}
    assert result.overpotential_at_benchmark is None
    assert result.onset_overpotential is None
    assert any("por encima de los 10" in w for w in result.warnings)

    # A curve that DOES span the benchmarks still reports them, and they
    # are distinct.
    wide = Voltammogram(
        potential=np.linspace(0.0, -0.30, n),
        current=-np.linspace(0.5, 300.0, n) * 1e-3, scan_rate=0.005,
        electrode=Electrode(area_cm2=1.0, reference="RHE",
                            resistance_ohm=0.0),
        name="HER completo")
    spanning = analyse_catalysis(wide, reaction="HER")
    values = [v for v in spanning.overpotentials.values() if v is not None]
    assert len(values) == 3 and len(set(np.round(values, 6))) == 3


def test_an_impossible_current_density_is_called_out():
    """2e5 mA/cm2 is 200 A/cm2. No laboratory electrode does that: it is
    amps taken for milliamps, or the wrong area, and the iR correction is
    multiplied by the same factor."""
    from ramancarbon.echem.evaluate import ABSURD_CURRENT_DENSITY, analyse_catalysis

    n = 200
    curve = Voltammogram(
        potential=np.linspace(0.0, -0.2, n),
        current=-np.linspace(10.0, 240.0, n),     # AMPS, i.e. 1000x too big
        scan_rate=0.005,
        electrode=Electrode(area_cm2=1.0, reference="RHE", resistance_ohm=0.0),
        name="unidades mal")
    result = analyse_catalysis(curve, reaction="HER")
    assert max(result.curve_j) > ABSURD_CURRENT_DENSITY
    assert any("A/cm²" in w and "unidades" in w for w in result.warnings)


# -- ZARC, and the capacitance a CPE is not ---------------------------


class TestTheZarc:
    """Same circuit as R|Q, written in the parameters that mean something."""

    R, TAU, GAMMA = 47.0, 2.3e-3, 0.82

    @property
    def q(self) -> float:
        return self.TAU ** self.GAMMA / self.R

    def test_it_is_exactly_a_resistance_in_parallel_with_a_cpe(self):
        from ramancarbon.echem.eis import parse_circuit

        omega = np.geomspace(1e-2, 1e6, 300)
        arc = parse_circuit("Za1")
        arc.elements()[0].values = [self.R, self.TAU, self.GAMMA]
        pair = parse_circuit("(R1|Q1)")
        by_name = {e.name: e for e in pair.elements()}
        by_name["R1"].values = [self.R]
        by_name["Q1"].values = [self.q, self.GAMMA]
        assert np.allclose(arc.impedance(omega), pair.impedance(omega),
                           rtol=1e-12, atol=0.0)

    def test_its_capacitance_is_tau_over_r_with_no_conversion(self):
        """The identity that makes the parameterisation worth having."""
        from ramancarbon.echem.eis import hsu_mansfeld_capacitance

        direct = self.TAU / self.R
        converted = hsu_mansfeld_capacitance(self.q, self.GAMMA, self.R).farads
        assert converted == pytest.approx(direct, rel=1e-10)

    def test_a_non_ideal_inductance_reduces_to_an_ideal_one(self):
        from ramancarbon.echem.eis import parse_circuit

        omega = np.geomspace(1e-2, 1e6, 100)
        modified = parse_circuit("La1")
        modified.elements()[0].values = [3e-7, 1.0]
        ideal = parse_circuit("L1")
        ideal.elements()[0].values = [3e-7]
        assert np.allclose(modified.impedance(omega), ideal.impedance(omega))

    def test_the_two_letter_symbols_parse_before_the_one_letter_ones(self):
        """``La`` must not be read as ``L`` followed by a stray ``a``."""
        from ramancarbon.echem.eis import parse_circuit

        names = [e.name for e in parse_circuit("La0-R0-Za1-Za2").elements()]
        assert names == ["La0", "R0", "Za1", "Za2"]


class TestEffectiveCapacitance:
    """Which relation, with which resistance, and what it assumes."""

    def test_hsu_mansfeld_is_not_divided_by_the_resistance(self):
        """The bug this replaced, pinned by an identity rather than a number.

        ``cpe_to_capacitance`` computed ``(Q·R^(1−n))^(1/n) / R``: the
        right expression with one spurious division, so every effective
        capacitance the package produced was too small by a factor of
        exactly R. The old test asserted only ``0 < C < Q``, which the
        wrong value satisfies as comfortably as the right one.

        A ZARC's capacitance is τ/R by construction, so converting its
        equivalent CPE has one correct answer and the factor of R shows
        up immediately.
        """
        from ramancarbon.echem.eis import cpe_to_capacitance

        resistance, tau, gamma = 47.0, 2.3e-3, 0.82
        q = tau ** gamma / resistance
        value = cpe_to_capacitance(q, gamma, resistance)
        assert value == pytest.approx(tau / resistance, rel=1e-10)
        assert value == pytest.approx(4.89362e-05, rel=1e-4)

    def test_brug_needs_the_series_resistance_and_says_so(self):
        from ramancarbon.echem.eis import CircuitError, brug_capacitance

        with pytest.raises(CircuitError, match="SERIE"):
            brug_capacitance(2e-4, 0.88, r_series=0.0)

    def test_brug_is_smaller_than_hsu_mansfeld(self):
        """Because it uses R_s ∥ R_ct, which is smaller than R_ct."""
        from ramancarbon.echem.eis import (
            brug_capacitance, hsu_mansfeld_capacitance)

        t, p, r_s, r_ct = 2e-4, 0.88, 5.0, 40.0
        brug = brug_capacitance(t, p, r_s, r_ct).farads
        hsu = hsu_mansfeld_capacitance(t, p, r_ct).farads
        assert 0.0 < brug < hsu

    def test_blocking_is_the_limit_of_faradaic_as_rct_grows(self):
        """The two Brug cases are one relation, so they have to meet."""
        from ramancarbon.echem.eis import brug_capacitance

        t, p, r_s = 2e-4, 0.88, 5.0
        blocking = brug_capacitance(t, p, r_s).farads
        huge = brug_capacitance(t, p, r_s, r_charge_transfer=1e12).farads
        assert huge == pytest.approx(blocking, rel=1e-6)

    def test_the_report_shows_the_arithmetic_and_names_t_and_p(self):
        from ramancarbon.echem.eis import brug_capacitance

        text = brug_capacitance(2e-4, 0.88, 5.0, 40.0).describe()
        assert "T (= Q)" in text and "P (= n)" in text
        assert "1/R_s + 1/R_ct" in text
        assert "R_s ∥ R_ct" in text
        # And what it assumes, because the two relations are applied
        # interchangeably in the literature and do not agree.
        assert "A LO LARGO" in text


class TestReadingTheDRT:
    """Each peak, what it is likely to be, and the circuit it translates to."""

    @staticmethod
    @pytest.fixture(scope="class")
    def two_processes():
        from ramancarbon.echem.eis import drt

        return drt(make_eis_demo("R0-(R1|Q1)-(R2|Q2)", seed=5))

    def test_a_peak_reports_its_capacitance_as_tau_over_r(self):
        from ramancarbon.echem.eis import describe_drt_peak

        _, _, capacitance = describe_drt_peak(1e-3, 50.0)
        assert capacitance == pytest.approx(2e-5)

    def test_every_time_constant_lands_in_exactly_one_band(self):
        from ramancarbon.echem.eis import DRT_PROCESSES, describe_drt_peak

        for tau in (1e-9, 1e-5, 1e-3, 0.1, 10.0, 1e4):
            process, _, _ = describe_drt_peak(tau, 10.0)
            assert process != "fuera de los intervalos tabulados"
        # The bands tile the line without gaps or overlaps.
        edges = [(low, high) for low, high, _, _ in DRT_PROCESSES]
        for (_, high), (low, _) in zip(edges, edges[1:]):
            assert high == low

    def test_the_interpretation_names_the_confirming_experiment(self, two_processes):
        """A prompt without a test is a guess with a citation."""
        text = two_processes.interpret()
        assert "C = τ/R" in text
        assert "cambia el potencial" in text or "cambia la temperatura" in text
        assert "ninguna de estas asignaciones sale" in text

    def test_two_processes_translate_into_two_arcs(self, two_processes):
        from ramancarbon.echem.eis import circuit_from_drt

        circuit, notes = circuit_from_drt(two_processes)
        assert circuit == "R0-Za1-Za2"
        assert any("TRADUCCIÓN" in note for note in notes)
        assert any("45°" in note for note in notes)

    def test_peaks_closer_than_a_factor_of_three_are_merged(self):
        from ramancarbon.echem.eis import DRTResult, circuit_from_drt

        close = DRTResult(
            tau_s=np.array([1.0]), gamma=np.array([1.0]), regularisation=1.0,
            fitted=np.array([1.0 + 0j]), residual=0.0,
            peaks_s=[1e-3, 2e-3], peak_resistances=[10.0, 12.0],
        )
        circuit, notes = circuit_from_drt(close)
        assert circuit == "R0-Za1"
        assert any("se funden en uno" in note for note in notes)
        assert any("λ y no" in note for note in notes)

    def test_no_resolved_peak_proposes_the_circuit_that_claims_least(self):
        from ramancarbon.echem.eis import DRTResult, circuit_from_drt

        empty = DRTResult(
            tau_s=np.array([1.0]), gamma=np.array([1.0]), regularisation=1.0,
            fitted=np.array([1.0 + 0j]), residual=0.0,
        )
        circuit, notes = circuit_from_drt(empty)
        assert circuit == "R0-Q1"
        assert any("no hay nada que traducir" in note for note in notes)

    def test_the_translated_circuit_fits_as_well_as_the_one_it_came_from(self):
        """The end-to-end claim: DRT to circuit to numbers.

        ``R0-Za1-Za2`` and ``R0-(R1|Q1)-(R2|Q2)`` are the same circuit, so
        the test is not that the ZARC chain fits better — it cannot — but
        that reading the DRT first lands the fit in the right basin
        without anyone guessing a circuit off the Nyquist plot.
        """
        from ramancarbon.echem.eis import (
            circuit_from_drt, drt, drt_starting_values, fit_circuit)

        spectrum = make_eis_demo("R0-(R1|Q1)-(R2|Q2)", seed=5)
        distribution = drt(spectrum)
        circuit, _ = circuit_from_drt(distribution)
        seeded = fit_circuit(
            spectrum, circuit,
            initial=drt_starting_values(distribution, circuit))
        reference = fit_circuit(spectrum, "R0-(R1|Q1)-(R2|Q2)")
        assert seeded.chi_squared == pytest.approx(
            reference.chi_squared, rel=0.5)
        values = seeded.values()
        assert values["Za1.tau"] < values["Za2.tau"]


# -- choosing a circuit, and seeing the one you chose -----------------


class TestDrawingACircuit:
    """The string and the shape stop agreeing where it matters."""

    def test_a_parallel_branch_is_drawn_stacked(self):
        from ramancarbon.echem.eis import draw_circuit

        drawing = draw_circuit("R0-(R1|Q1)")
        assert "┬" in drawing and "┴" in drawing
        assert "R1" in drawing and "Q1" in drawing
        # And a legend saying what each symbol is, because a reader who
        # has to decode the glyphs has gained nothing over the string.
        assert "CPE" in drawing and "resistencia" in drawing

    def test_the_inductive_loop_template_really_makes_one(self):
        """The bug the drawing found, pinned by the physics.

        ``adsorcion_inductiva`` shipped as ``R0-(R1|Q1|R2-L1)``. In this
        notation ``|`` binds tighter than ``-``, so that is
        ``R0-((R1|Q1|R2)-L1)``: an inductor in SERIES, which lifts Z″ at
        the top of the spectrum and produces no low-frequency loop at
        all. Measured on the template's own description of itself — HER
        and OER give an adsorption loop almost always — it gave zero
        points of inductive loop and a Z″ maximum of +8·10⁵ at 16 kHz,
        which is the series reactance running away.

        An inductive loop is not "Z″ > 0": a series inductor does that
        too. It is Z″ > 0 **while Z′ recedes**, which is what a loop in
        the Nyquist plane means and what this asserts.
        """
        from ramancarbon.echem.eis import LIBRARY, parse_circuit

        values = {"R0": [5.0], "R1": [40.0], "Q1": [2e-4, 0.9],
                  "R2": [25.0], "L1": [8.0]}
        omega = np.geomspace(1e-3, 1e5, 600)
        tree = parse_circuit(LIBRARY["adsorcion_inductiva"])
        for element in tree.elements():
            element.values = list(values[element.name])
        z = tree.impedance(omega)
        # Walk from high frequency to low, as a sweep does.
        real, imaginary = z.real[::-1], z.imag[::-1]
        looping = (imaginary[1:] > 1e-9) & (np.diff(real) < 0.0)
        assert looping.sum() > 50
        # And the loop is at LOW frequency, not at the top of the sweep.
        peak = int(np.argmax(imaginary))
        assert omega[::-1][peak] < 100.0

    def test_parallel_binds_tighter_than_series(self):
        """The precedence that made the template wrong, stated as a test."""
        from ramancarbon.echem.eis import parse_circuit

        implicit = parse_circuit("R1|Q1|R2-L1")
        explicit = parse_circuit("(R1|Q1|R2)-L1")
        assert implicit.kind == explicit.kind == "series"
        assert len(implicit.children) == len(explicit.children) == 2


class TestComparingCircuits:
    """Rank them, and say why the ranking is not the answer."""

    @staticmethod
    @pytest.fixture(scope="class")
    def comparison():
        from ramancarbon.echem.eis import compare_circuits

        return compare_circuits(
            make_eis_demo("R0-(R1|Q1)-(R2|Q2)", seed=5),
            circuits=["R0-Q1", "R0-(R1|Q1)", "R0-Za1-Za2"],
        )

    def test_the_too_simple_circuit_ranks_last(self, comparison):
        order = comparison.ranked()
        assert order[-1].circuit == "R0-Q1"
        assert comparison.best().circuit == "R0-Za1-Za2"

    def test_equivalent_circuits_score_the_same(self, comparison):
        """R0-Za1 and R0-(R1|Q1) are one circuit, so they cannot differ."""
        from ramancarbon.echem.eis import compare_circuits

        spectrum = make_eis_demo("R0-(R1|Q1)", seed=2)
        both = compare_circuits(spectrum, circuits=["R0-Za1", "R0-(R1|Q1)"])
        first, second = both.candidates
        assert first.fit is not None and second.fit is not None
        assert first.fit.chi_squared == pytest.approx(
            second.fit.chi_squared, rel=1e-6)
        assert first.free_parameters == second.free_parameters

    def test_it_counts_the_parameters_that_came_back_determined(self, comparison):
        """The column that decides, because BIC alone cannot."""
        for candidate in comparison.ranked():
            assert 0 <= candidate.determined <= candidate.free_parameters

    def test_the_report_refuses_to_let_the_bic_decide_alone(self, comparison):
        text = comparison.describe()
        assert "determinados" in text
        assert "ajusta SIEMPRE al menos igual de bien" in text
        assert "APROPIADO para el material" in text


class TestCircuitConcerns:
    """Structural checks, run before any fit."""

    def test_a_series_chain_cannot_make_a_semicircle(self):
        from ramancarbon.echem.eis import circuit_concerns

        assert any("PARALELO" in c for c in circuit_concerns("R0-Q1"))

    def test_an_element_that_is_itself_an_arc_counts(self):
        """The false positive the first version had on its own new element.

        A ZARC IS a resistance in parallel with a CPE, written as one
        symbol, so looking for a ``|`` in the string flagged ``R0-Za1``
        — the circuit the DRT translation produces — as unable to make
        an arc.
        """
        from ramancarbon.echem.eis import circuit_concerns

        assert not any("PARALELO" in c for c in circuit_concerns("R0-Za1"))
        assert not any("PARALELO" in c for c in circuit_concerns("R0-T1"))

    def test_an_ideal_capacitor_is_flagged_but_not_refused(self):
        from ramancarbon.echem.eis import circuit_concerns

        concerns = circuit_concerns("R0-(R1|C1)")
        assert any("IDEAL" in c for c in concerns)

    def test_a_circuit_without_a_resistance_cannot_set_the_intercept(self):
        from ramancarbon.echem.eis import circuit_concerns

        assert any("resistencia" in c for c in circuit_concerns("Q1|C1"))

    def test_an_unreadable_circuit_says_so_instead_of_raising(self):
        from ramancarbon.echem.eis import circuit_concerns

        assert circuit_concerns("R0-(R1|Q1")[0].startswith("no se puede leer")


class TestTheCircuitEditorFlow:
    """DRT to circuit to numbers to a file, through the session."""

    @staticmethod
    def _session():
        from ramancarbon.gui.echem_state import EchemSession

        session = EchemSession()
        session.eis = make_eis_demo("R0-(R1|Q1)-(R2|Q2)", seed=5)
        session.analyse()
        return session

    def test_refitting_replaces_only_the_impedance_part(self):
        """Choosing a circuit costs one fit, not the whole analysis."""
        session = self._session()
        before = session.result
        assert session.set_circuit("R0-Za1-Za2")
        assert session.refit_circuit()
        # Same result object, new impedance fit: nothing else depends on
        # the circuit, so nothing else is recomputed.
        assert session.result is before
        assert session.result.eis is not None
        assert session.result.eis.fit is not None
        assert session.result.eis.fit.circuit == "R0-Za1-Za2"

    def test_refitting_without_a_spectrum_says_so(self):
        from ramancarbon.gui.echem_state import EchemSession

        session = EchemSession()
        assert not session.refit_circuit()
        assert any("impedancia" in text for _, text in session.messages)

    def test_the_drawing_carries_the_fitted_values(self):
        session = self._session()
        session.set_circuit("R0-Za1-Za2")
        session.refit_circuit()
        drawing = session.circuit_drawing()
        assert "Za1" in drawing and "Za2" in drawing
        assert "Za1.tau" in drawing

    def test_a_zarc_needs_no_conversion_and_the_report_says_so(self):
        session = self._session()
        session.set_circuit("R0-Za1-Za2")
        session.refit_circuit()
        text = session.effective_capacitance_report()
        assert "sin conversión" in text
        assert "C = τ/R" in text
        # And no note about two relations, because neither was used.
        assert "LAS DOS NO SON LA MISMA MEDIDA" not in text

    def test_a_cpe_gets_both_relations_and_the_warning(self):
        session = self._session()
        session.set_circuit("R0-(R1|Q1)")
        session.refit_circuit()
        text = session.effective_capacitance_report()
        assert "Hsu-Mansfeld" in text and "Brug" in text
        assert "LAS DOS NO SON LA MISMA MEDIDA" in text

    def test_the_exported_curve_keeps_the_physical_sign_of_z_imaginary(self):
        """The -Z'' of a Nyquist plot is a drawing convention."""
        session = self._session()
        session.set_circuit("R0-(R1|Q1)")
        session.refit_circuit()
        rows = session.fit_curve_rows()
        assert len(rows) == session.eis.n
        assert len(rows[0]) == len(session.FIT_CURVE_COLUMNS)
        # Capacitive, so Z'' is negative as stored.
        assert float(rows[len(rows) // 2][2]) < 0.0

    def test_exporting_writes_the_circuit_and_the_sign_convention(self, tmp_path):
        session = self._session()
        session.set_circuit("R0-(R1|Q1)")
        session.refit_circuit()
        destination = tmp_path / "ajuste.csv"
        assert session.save_fit_curve(destination)
        text = destination.read_text(encoding="utf-8")
        assert "# circuito: R0-(R1|Q1)" in text
        assert "signo FÍSICO" in text
        assert "f (Hz),Zr_obs" in text

    def test_exporting_without_a_fit_refuses_with_a_reason(self, tmp_path):
        from ramancarbon.gui.echem_state import EchemSession

        session = EchemSession()
        assert not session.save_fit_curve(tmp_path / "nada.csv")
        assert any("curva ajustada" in text for _, text in session.messages)
