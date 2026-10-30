"""Electrochemistry: capacitance, mechanism, impedance, catalysis."""

from __future__ import annotations

import math

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
    KNEE_PHASE_DEG,
    analyse_eis,
    cpe_to_capacitance,
    fit_circuit,
    hsu_mansfeld_capacitance,
    kramers_kronig,
    parse_circuit,
    read_shape,
    recommend_circuits,
    series_cpe_capacitance,
    storage_element,
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
    # Either wording: the direct test (the three-term fit, when the
    # series is wide enough) or the fallback (the laws tying). Both say
    # the same thing and both name the parasitic current.
    assert any("parásita" in text for text in verdict.against), verdict.against


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


def test_an_electrolyte_preset_fills_the_ph_it_needs():
    """Without a pH there is no RHE scale and no overpotential, and a pH
    typed from memory is 59 mV per unit of error downstream."""
    electrode = Electrode(reference="Ag/AgCl_sat")
    note = electrode.use_electrolyte("KOH_0.5M")
    assert electrode.electrolyte == "KOH 0.5 M"
    assert electrode.ph == pytest.approx(13.56)
    assert "nominal" in note.lower()

    acid = Electrode()
    acid.use_electrolyte("H2SO4_0.5M")
    assert acid.ph == pytest.approx(0.3)

    unknown = Electrode(ph=7.0)
    unknown.use_electrolyte("mi mezcla rara")
    assert unknown.electrolyte == "mi mezcla rara"
    assert unknown.ph == 7.0, "un electrolito desconocido no inventa un pH"


def test_a_disc_diameter_gives_its_area():
    electrode = Electrode()
    assert electrode.use_disc(3.0) == pytest.approx(0.070686, rel=1e-4)
    assert electrode.area_cm2 == pytest.approx(0.070686, rel=1e-4)


def test_the_potential_axis_defaults_to_what_was_measured():
    electrode = Electrode(reference="Ag/AgCl_sat", ph=13.56)
    measured = np.array([-0.6, 0.4])

    values, label, _ = electrode.on_scale(measured)
    assert np.allclose(values, measured)
    assert "Ag/AgCl" in label

    values, label, why = electrode.on_scale(measured, "RHE")
    assert values == pytest.approx(measured + 0.9992, abs=1e-3)
    assert label == "E / V vs RHE"
    assert "0.9992" in why


def test_a_refused_conversion_still_returns_a_usable_axis():
    """to_rhe correctly returns None without a pH, and a figure that
    forgets to check draws nothing. This one falls back and says so."""
    electrode = Electrode(reference="Ag/AgCl_sat", ph=None)
    measured = np.array([-0.6, 0.4])
    values, label, why = electrode.on_scale(measured, "RHE")
    assert np.allclose(values, measured)
    assert "Ag/AgCl" in label, "la etiqueta tiene que decir lo que se dibujó"
    assert "no se ha podido" in why


def test_a_fitted_constant_needs_the_charge_balance_to_agree():
    """A constant the i(ν) fit likes must also show up where it cannot
    hide. On the synthetic hybrid, which has no parasitic current at
    all, the fitted one came out opposite in sign to the other two
    terms, the signed sum nearly cancelled, and the share read 92 %."""
    from ramancarbon.echem.cv import analyse_rate_study
    from ramancarbon.examples.demo_data import cv_rate_series

    study = analyse_rate_study(cv_rate_series("hibrido", seed=2))
    verdict = classify_storage(rates=study)
    assert not any("corriente parásita que es el" in text
                   for text in verdict.against), verdict.against


def test_a_repeated_rate_enters_a_fit_once():
    from ramancarbon.echem.cv import unique_rates

    series = [
        make_cv_demo("condensador", capacitance=0.05, scan_rate=rate,
                     seed=1).cycles()[-1]
        for rate in (0.05, 0.005, 0.01, 0.02, 0.05)
    ]
    kept = unique_rates(series)
    assert len(kept) == 4
    assert kept[0] is series[-1], "se queda la ÚLTIMA de cada velocidad"


def test_the_analysis_choices_reach_the_result():
    from ramancarbon.echem.report import analyse_sample
    from ramancarbon.examples.demo_data import cv_rate_series

    series = cv_rate_series("hibrido", seed=2)
    coarse = analyse_sample(rate_series=series, dunn_points=12)
    assert coarse.dunn.potentials.size == 12

    free = analyse_sample(rate_series=series, dunn_nonnegative=False)
    constrained = analyse_sample(rate_series=series)
    assert np.all(constrained.dunn.k1_anodic >= 0.0)
    assert free.dunn.k1.shape == constrained.dunn.k1.shape


def test_subtracting_the_parasitic_current_is_announced():
    from ramancarbon.echem.report import analyse_sample

    leak = 5e-4
    series = []
    for rate in (0.005, 0.01, 0.02, 0.05):
        curve = make_cv_demo("condensador", capacitance=0.05,
                             scan_rate=rate, seed=1).cycles()[-1]
        rising = np.gradient(curve.potential) > 0
        curve.current = curve.current + leak * rising
        series.append(curve)

    plain = analyse_sample(rate_series=series)
    assert plain.parasitic_note == ""
    fixed = analyse_sample(rate_series=series, subtract_parasitic=True)
    assert "balance de carga" in fixed.parasitic_note
    assert any("SERIE CORREGIDA" in w for w in fixed.warnings)
    # The corrected series must be flatter in rate than the measured one:
    # that is the whole point of being able to ask.
    before = plain.rates.capacitances
    after = fixed.rates.capacitances
    assert after[-1] / after[0] > before[-1] / before[0]


def test_the_session_starts_on_the_electrode_most_runs_use():
    """Defaults so the common case needs no typing — and the mass is
    deliberately NOT one of them: there is no sensible value, a wrong one
    is invisible, and a blank one makes the F/g disappear, which is the
    failure that can be seen."""
    from ramancarbon.gui.echem_state import EchemSession

    electrode = EchemSession().electrode
    assert electrode.reference == "Ag/AgCl_sat"
    assert electrode.area_cm2 == pytest.approx(0.070686, rel=1e-4)
    assert electrode.electrolyte == "KOH 0.5 M"
    assert electrode.ph == pytest.approx(13.56)
    assert electrode.mass_mg is None


def test_the_session_exports_every_table_it_shows(tmp_path):
    from ramancarbon.gui.echem_state import TABLE_COLUMNS, EchemSession
    from ramancarbon.examples.demo_data import cv_rate_series

    session = EchemSession()
    session.electrode.mass_mg = 1.0
    session.rate_series = list(cv_rate_series("hibrido", seed=2))
    session.cv = session.rate_series[-1]
    session.retag_electrode()
    session.analyse()

    written = session.save_tables(tmp_path)
    assert "informe.txt" in written
    for name, columns in TABLE_COLUMNS.items():
        path = tmp_path / f"{name}.csv"
        assert path.exists(), name
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0] == ",".join(columns)
        assert len(lines) > 1
        assert all(len(row.split(",")) == len(columns) for row in lines[1:])


def test_the_session_draws_the_axis_the_user_chose():
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession()
    session.cv = make_cv_demo("condensador", seed=1)
    session.retag_electrode()

    values, label, _ = session.potential_axis(session.cv)
    assert np.allclose(values, session.cv.potential)
    assert "Ag/AgCl" in label

    session.potential_scale_choice = "RHE"
    values, label, _ = session.potential_axis(session.cv)
    assert label == "E / V vs RHE"
    assert np.all(values > session.cv.potential)


def _series_with_three_terms(rates, capacitance_f, diffusive, parasitic):
    """Voltammograms whose current obeys k_cap·ν + k_diff·√ν + i_par.

    Built by hand rather than from ``make_cv_demo``: that one has a
    series resistance, so its current at a fixed potential is not
    exactly ``C·ν`` near the vertices, and the fit then absorbs the
    difference into the constant — which is the term under test.
    """
    series = []
    for rate in rates:
        up = np.linspace(0.0, 0.6, 300)
        potential = np.concatenate([up, up[::-1]])
        surface = capacitance_f * rate + diffusive * math.sqrt(rate)
        current = np.concatenate([
            np.full(up.size, surface), np.full(up.size, -surface)
        ]) + parasitic
        series.append(Voltammogram(potential=potential, current=current,
                                   scan_rate=rate,
                                   name=f"{1e3 * rate:g} mV/s"))
    return series


def test_the_three_terms_separate_when_the_series_is_wide_enough():
    from ramancarbon.echem.cv import fit_full_current

    series = _series_with_three_terms(
        (0.005, 0.01, 0.02, 0.05, 0.1), 50e-6, 10e-6, 0.4e-6)
    models = fit_full_current(series)
    assert models
    middle = models[len(models) // 2]
    assert middle.capacitance_f == pytest.approx(50e-6, rel=0.15)
    assert middle.diffusive == pytest.approx(10e-6, rel=0.2)
    assert middle.leakage_a == pytest.approx(0.4e-6, abs=0.15e-6)
    assert all(middle.determined().values()), middle


def test_a_parasitic_current_may_be_cathodic():
    """At the negative end of this window the electrode sits in the
    oxygen-reduction region, so the constant is NEGATIVE. Forcing it
    positive made the fit put 11 µF of capacitance where there are 90
    and call the rest a leak."""
    from ramancarbon.echem.cv import fit_full_current

    series = _series_with_three_terms(
        (0.005, 0.01, 0.02, 0.05, 0.1), 50e-6, 10e-6, -0.8e-6)
    middle = fit_full_current(series)[4]
    assert middle.leakage_a < 0.0
    assert middle.leakage_a == pytest.approx(-0.8e-6, abs=0.2e-6)
    assert middle.determined()["i_par"]


def test_three_parameters_need_more_than_three_rates():
    from ramancarbon.echem.cv import fit_full_current

    series = _series_with_three_terms((0.005, 0.02, 0.1), 50e-6, 10e-6, 0.4e-6)
    assert fit_full_current(series) == [], (
        "tres puntos y tres parámetros no dejan ningún grado de libertad, "
        "y un error típico de cero grados de libertad no es un error típico"
    )


def test_a_dunn_coefficient_is_judged_against_its_own_uncertainty():
    """On an ideal capacitor the diffusive term is zero by construction,
    so 'negative by more than 5 % of its own scale' compares noise with
    noise: it fired on 7 of 120 coefficients. The test is now against
    each coefficient's own error, with the level divided by how many
    coefficients are tested."""
    from ramancarbon.echem.cv import dunn_analysis
    from ramancarbon.examples.demo_data import cv_rate_series

    for seed in (0, 1, 2, 3):
        analysis = dunn_analysis(cv_rate_series("condensador", seed=seed))
        assert not [w for w in analysis.warnings if "quedado en CERO" in w], seed


def test_the_current_is_read_off_the_reporting_cycle():
    """It was read off the whole file, so the b-value and the Dunn
    separation came from a different cycle than the capacitance quoted
    beside them."""
    from ramancarbon.echem.cv import _current_at

    curve = make_cv_demo("condensador", capacitance=0.05, cycles=4, seed=1)
    last = curve.cycles()[-1]
    potential = float(np.median(curve.potential))
    assert _current_at(curve, potential) == pytest.approx(
        _current_at(last, potential))


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
    # A factor of ten in current buys a factor of ten in power, and a
    # LITTLE less: the energy is referenced to each branch's own lower end,
    # and the IR jump pushes that end up in proportion to the current, so
    # the fastest discharge loses slightly more of its window than the
    # slowest. 9.997 rather than 10 is that cost, and asserting strictly
    # more than ten asserted that it is free.
    assert max(powers) / min(powers) == pytest.approx(10.0, rel=0.02)
    assert max(powers) / min(powers) < 10.0
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


# -- the shape of the spectrum, read before any circuit ---------------


def _blocking_porous(points: int = 40, seed: int = 4):
    """A porous carbon: series R, a pore line, no closed arc.

    `make_eis_demo` builds it from the same element code the fitter uses,
    so what goes in is what has to come out.
    """
    return make_eis_demo(
        "R0-T1",
        values={"R0.R": 5.0, "T1.Ri": 60.0, "T1.Q": 2e-4, "T1.n": 0.88},
        frequencies=(0.1, 1e5), points=points, noise_pct=0.3, seed=seed,
    )


def test_the_knee_is_not_the_maximum_of_minus_z_imaginary():
    """The bug this replaced, reproduced on a blocking porous electrode.

    −Z″ of anything with a capacitive tail rises monotonically to the
    end of the sweep, so its maximum is the LAST measured point. On the
    user's three real carbons that returned 0.0999 Hz in all three and
    called it a time constant of 10 s.
    """
    spectrum = _blocking_porous()
    imaginary = -spectrum.z.imag
    assert int(np.argmax(imaginary)) == spectrum.n - 1, (
        "este espectro tiene que subir monótona hasta el final, que es "
        "lo que hacía fallar al criterio anterior"
    )

    shape = read_shape(spectrum)
    assert shape.arc_apex_hz is None
    assert shape.knee_hz is not None
    lowest = float(spectrum.frequency[-1])
    assert shape.knee_hz > 100.0 * lowest, (
        f"el codo salió en {shape.knee_hz:.4g} Hz, demasiado cerca del "
        f"extremo del barrido ({lowest:.4g} Hz)"
    )
    # And it is where the phase really crosses, bracketed by two points.
    below = spectrum.frequency[spectrum.phase_deg <= KNEE_PHASE_DEG]
    above = spectrum.frequency[spectrum.phase_deg > KNEE_PHASE_DEG]
    assert float(below.max()) <= shape.knee_hz <= float(above.min())


def test_a_spectrum_without_a_closed_arc_says_so():
    shape = read_shape(_blocking_porous())
    assert shape.arc_apex_hz is None
    assert any("transferencia de carga" in note for note in shape.notes)
    assert shape.capacitive_tail


def test_an_arc_that_closes_is_found_and_is_not_the_last_point():
    spectrum = make_eis_demo("R0-(R1|Q1)", seed=3)
    shape = read_shape(spectrum)
    assert shape.arc_apex_hz is not None
    assert shape.arc_apex_hz > float(spectrum.frequency[-1])
    # 1/(R C) with C from the CPE, which is where the apex of R|Q sits.
    expected = 1.0 / (2.0 * math.pi * 40.0
                      * hsu_mansfeld_capacitance(2e-4, 0.88, 40.0).farads)
    assert shape.arc_apex_hz == pytest.approx(expected, rel=0.5)


def test_the_series_resistance_carries_how_it_was_obtained():
    result = analyse_eis(_blocking_porous(), circuit=None)
    assert result.series_resistance is not None
    assert result.series_resistance_how, (
        "una corrección iR con una resistencia sin explicar es un "
        "parámetro libre"
    )


def test_a_porous_shape_asks_for_the_transmission_line_not_randles():
    spectrum = _blocking_porous()
    suggestions, shape = recommend_circuits(spectrum)
    names = [s.name for s in suggestions]
    assert "poroso" in names
    assert "randles_cpe" not in names, (
        "randles_cpe era el circuito por defecto y es el modelo "
        "equivocado aquí: su R1 se va a la cota"
    )
    assert shape.fortyfive_decades > 0.5
    for suggestion in suggestions:
        assert suggestion.why


def test_the_default_circuit_no_longer_pins_a_resistance_to_its_bound():
    """What the old default did, and that the new path does not do it."""
    spectrum = _blocking_porous()
    pinned = fit_circuit(spectrum, "R0-(R1|Q1)")
    assert pinned.values()["R1.R"] > 1e8, (
        "sin cota pegada no hay nada que demostrar; este espectro tiene "
        "que forzar a randles_cpe a mandar su R al infinito"
    )

    result = analyse_eis(spectrum)
    assert result.fit is not None
    values = result.fit.values()
    assert all(value < 1e8 for label, value in values.items()
               if label.endswith(".R") or label.endswith(".Ri"))


def test_the_charge_transfer_resistance_is_read_off_the_structure():
    """Not off the order of the parameter dictionary.

    ``R0-T1`` has two resistances —R0 and the pore's Ri— and neither is
    a charge-transfer resistance. Taking "the second .R" would have
    reported one anyway for a nested circuit.
    """
    assert storage_element("R0-T1") == "T1"
    result = analyse_eis(_blocking_porous(), circuit="R0-T1")
    assert result.charge_transfer_resistance is None
    assert result.ionic_resistance_ohm == pytest.approx(60.0, rel=0.25)


def test_the_stored_capacitance_comes_from_the_element_that_stores_it():
    """And the arc's CPE is a different number, named differently.

    Measured on the user's 800 °C carbon: the arc's CPE converts to
    0.092 mF and the pore wall to 0.216 mF, where the model-free C′ at
    the same frequency is 0.212 mF. Taking the first CPE in the circuit
    was wrong by a factor of 2.3.
    """
    assert storage_element("R0-(R1|Q1)-T1") == "T1"
    assert storage_element("R0-(R1|Q1)-Q2") == "Q2"
    assert storage_element("R0-(R1|Q1)") is None

    spectrum = _blocking_porous()
    result = analyse_eis(spectrum, circuit="R0-T1")
    assert result.storage_capacitance is not None
    # Q · ω^(n−1) at the lowest measured frequency, with the planted Q.
    omega = float(spectrum.omega[-1])
    expected = 2e-4 * omega ** (0.88 - 1.0)
    assert result.storage_capacitance.farads == pytest.approx(expected, rel=0.1)
    assert "serie" in result.storage_capacitance.method


def test_the_stored_capacitance_is_checked_against_c_prime():
    result = analyse_eis(_blocking_porous(), circuit="R0-T1")
    assert "C′" in result.storage_check
    assert result.capacitance_f is not None
    assert result.capacitance_frequency_hz is not None
    # The deliverable number is the model-free one, by the package's own
    # rule: C′ carries the real part of the whole chain.
    spectrum = _blocking_porous()
    omega = float(spectrum.omega[-1])
    measured = float(-spectrum.z.imag[-1] / (omega * abs(spectrum.z[-1]) ** 2))
    assert result.capacitance_f == pytest.approx(measured, rel=1e-9)


def test_a_series_cpe_capacitance_needs_its_frequency():
    one = series_cpe_capacitance(2e-4, 0.88, 0.1)
    ten = series_cpe_capacitance(2e-4, 0.88, 1.0)
    assert one.farads > ten.farads, (
        "con n < 1 la capacitancia de un CPE en serie BAJA al subir la "
        "frecuencia; si no, el exponente está al revés"
    )
    assert "0.1" in one.method
    assert one.steps and one.caution
    ideal = series_cpe_capacitance(3e-4, 1.0, 7.0)
    assert ideal.farads == pytest.approx(3e-4), (
        "con n = 1 un CPE ES un condensador y su capacitancia no puede "
        "depender de la frecuencia"
    )


def test_dropping_a_real_inductance_makes_the_pore_resistance_unmeasurable():
    """Measured on the user's 900 °C spectrum, reproduced synthetically.

    Its Z″ is positive at 100 kHz — cable inductance. Fitted without an
    L in the circuit, R0 absorbs it and T1.Ri collapsed from 0.79 Ω
    (± 13 %) to 2·10⁻⁵ Ω (± 3.5·10⁷ %): the quantity that limits the
    device's power stopped being measurable, and nothing about the
    residual said so.
    """
    spectrum = make_eis_demo(
        "L0-R0-T1",
        values={"L0.L": 1e-6, "R0.R": 20.0, "T1.Ri": 0.8, "T1.Q": 3e-4,
                "T1.n": 0.90},
        frequencies=(0.1, 1e5), points=40, noise_pct=0.2, seed=7,
    )
    shape = read_shape(spectrum)
    assert shape.inductive_top
    assert any("inductancia" in note for note in shape.notes)
    names = [s.name for s in recommend_circuits(spectrum, shape)[0]]
    assert any("inductancia" in name or "completo" in name for name in names)

    without = fit_circuit(spectrum, "R0-T1")
    with_it = fit_circuit(spectrum, "L0-R0-T1")
    bad = without.errors["T1.Ri"] / abs(without.values()["T1.Ri"])
    good = with_it.errors["T1.Ri"] / abs(with_it.values()["T1.Ri"])
    assert good < 0.5 <= bad, (
        f"Ri sale al {100 * good:.0f} % con la L dentro y al "
        f"{100 * bad:.0f} % sin ella"
    )
    assert with_it.values()["T1.Ri"] == pytest.approx(0.8, rel=0.25)
    assert without.values()["T1.Ri"] < 0.01 * 0.8, (
        "sin la L, Ri no sale solo impreciso: se desploma a cero, que es "
        "la línea de transmisión degenerando en un CPE pelado"
    )


def test_the_drt_says_when_its_peaks_are_a_distributed_element():
    """A pore has no discrete time constant, and the DRT gives it peaks.

    The module said this in prose; `drt` has the spectrum, so it can say
    it with the measured number of decades.
    """
    from ramancarbon.echem.eis import drt

    result = drt(_blocking_porous(points=50))
    assert result.peaks_s, "sin picos no hay nada que avisar"
    assert any("décadas cerca de −45" in w for w in result.warnings)



# -- the composite, multi-panel figure --------------------------------


def _session_with_spectra(count: int = 3):
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession(name="serie")
    for index in range(count):
        spectrum = _blocking_porous(seed=4 + index)
        spectrum.name = f"muestra {index + 1}"
        session.eis_series.append(spectrum)
    session.eis = session.eis_series[0]
    return session


def test_only_the_panels_with_data_are_offered():
    from ramancarbon.gui.echem_state import EchemSession

    empty = EchemSession()
    assert empty.figure_panels() == []

    session = _session_with_spectra()
    keys = [panel.key for panel in session.figure_panels()]
    assert keys, "con espectros cargados tiene que ofrecer algo"
    assert all(key.startswith("eis_") for key in keys), (
        "sin voltamperogramas no se puede ofrecer ningún panel de CV: un "
        "panel vacío en una figura de seis se lee como un fallo y no hay "
        "forma de distinguirlo de uno cuyos datos no dijeran nada"
    )
    for panel in session.figure_panels():
        assert panel.label and panel.note


def test_the_panels_keep_the_order_they_are_offered_in():
    """Not the order they were ticked.

    A figure's letters run (a), (b), (c) down the page, and someone who
    ticks the residual before the Nyquist does not mean the residual to
    be panel (a).
    """
    session = _session_with_spectra()
    offered = [panel.key for panel in session.figure_panels()]
    session.set_figure_panels([offered[2], offered[0]])
    assert session.composite.panels == [offered[0], offered[2]]


def test_a_composite_figure_is_a_value_that_round_trips():
    from ramancarbon.gui.figures_echem import CompositeFigure

    spec = CompositeFigure(panels=["eis_nyquist", "eis_bode"], columns=1,
                           width_in=3.3, dpi=600, letters=False,
                           title="Figura 2")
    again = CompositeFigure.from_dict(spec.to_dict())
    assert again == spec
    assert again.rows == 2
    assert again.figsize() == (3.3, 2 * spec.panel_height_in)


def test_the_figure_exports_the_numbers_of_every_panel():
    """Including the ones that carry a twin axis.

    Zipping the chosen panels against ``figure.axes`` of the composite
    looks equivalent and is not: a panel with a twin adds a second entry
    to that list, so every panel after the first Bode plot read another
    panel's lines and two came back with nothing but their header.
    """
    session = _session_with_spectra(2)
    session.set_figure_panels(["eis_nyquist", "eis_bode",
                               "eis_capacitancia", "eis_residuo"])
    rows = session.figure_data_rows()
    assert set(rows) == set(session.composite.panels)
    for key, table in rows.items():
        assert table[0] == ("serie", "x", "y")
        assert len(table) > 1, f"{key} salió sin datos"
        names = {name for name, _, _ in table[1:]}
        # The residual panel names the circuit in its legend too, so the
        # check is that both samples are in there and nothing else.
        for sample in ("muestra 1", "muestra 2"):
            assert any(name.startswith(sample) for name in names), (
                f"{key} no lleva ninguna serie de {sample}")


def test_the_figure_is_saved_at_its_own_width(tmp_path):
    from PIL import Image

    from ramancarbon.gui.theme import PALETTES

    session = _session_with_spectra(2)
    session.set_figure_panels(["eis_nyquist", "eis_bode"])
    session.composite.columns = 2
    session.composite.width_in = 7.0
    session.composite.panel_height_in = 3.0
    session.composite.dpi = 100
    target = tmp_path / "figura.png"
    assert session.save_figure(target, PALETTES["claro"])
    # bbox_inches="tight" trims, so this is a band and not an equality;
    # what it has to catch is a figure saved at the canvas's size
    # instead of the one that was asked for.
    width, height = Image.open(target).size
    assert 500 <= width <= 800, width
    assert 200 <= height <= 400, height


def test_a_panel_that_fails_does_not_take_the_figure_with_it():
    import matplotlib
    from matplotlib.figure import Figure

    from ramancarbon.gui.figures_echem import (PANELS_BY_KEY, PanelSpec,
                                               draw_composite)
    from ramancarbon.gui.theme import PALETTES, matplotlib_style

    def explode(axis, session, spec, palette):
        raise RuntimeError("a propósito")

    session = _session_with_spectra(1)
    session.set_figure_panels(["eis_nyquist"])
    session.composite.panels = ["eis_nyquist", "roto"]
    PANELS_BY_KEY["roto"] = PanelSpec("roto", "Roto", "eis", explode)
    try:
        with matplotlib.rc_context(matplotlib_style(PALETTES["claro"])):
            figure = Figure(figsize=(6, 6))
            problems = draw_composite(figure, session.composite, session,
                                      PALETTES["claro"])
        assert len(problems) == 1 and "a propósito" in problems[0]
        assert len(figure.axes) >= 2, (
            "construir una figura de seis paneles y recibir una traza en "
            "vez de los cinco que sí salieron es el cambio equivocado"
        )
    finally:
        del PANELS_BY_KEY["roto"]


# -- the impedance series ---------------------------------------------


def test_the_section_compares_several_spectra(tmp_path):
    from ramancarbon.gui.echem_state import (EIS_COMPARISON_COLUMNS,
                                             EchemSession)

    session = EchemSession()
    for index in range(2):
        path = tmp_path / f"m{index}.csv"
        write_eis(_blocking_porous(seed=4 + index), path)
        assert session.add_eis_to_series(path, label=f"{800 + 100 * index} °C")
    assert session.eis_series_choices() == ["800 °C", "900 °C"]
    assert session.eis is session.eis_series[0], (
        "el primero tiene que quedar seleccionado o el editor de circuito "
        "se queda sin espectro sobre el que trabajar"
    )
    rows = session.eis_comparison_rows()
    assert rows[0] == EIS_COMPARISON_COLUMNS
    assert len(rows) == 3
    assert [row[0] for row in rows[1:]] == ["800 °C", "900 °C"]

    assert session.use_eis_from_series("900 °C")
    assert session.eis is session.eis_series[1]
    assert not session.use_eis_from_series("1020 °C")


def test_clearing_the_series_releases_the_chosen_spectrum(tmp_path):
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession()
    path = tmp_path / "m.csv"
    write_eis(_blocking_porous(), path)
    assert session.add_eis_to_series(path, label="una")
    session.clear("eis_series")
    assert session.eis_series == []
    assert session.eis is None, (
        "el espectro elegido era uno de la serie; dejarlo apuntando a un "
        "objeto que ya no está en ninguna lista es la forma de que la "
        "pestaña siga enseñando lo que se acaba de borrar"
    )


def test_the_per_spectrum_analyses_are_cached_and_invalidated(tmp_path):
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession()
    path = tmp_path / "m.csv"
    write_eis(_blocking_porous(), path)
    assert session.add_eis_to_series(path, label="una")
    first = session.eis_analyses()
    assert first is session.eis_analyses(), (
        "cada panel de impedancia pide el mismo análisis y cada uno cuesta "
        "una comparación de circuitos: una figura de seis paneles lo "
        "correría seis veces por espectro"
    )
    assert session.add_eis_to_series(path, label="otra")
    assert session.eis_analyses() is not first
    assert len(session.eis_analyses()) == 2


# -- choosing the circuit from the window ------------------------------


def test_the_section_defaults_to_choosing_the_circuit_from_the_data():
    from ramancarbon.echem.eis import AUTO_CIRCUIT
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession()
    assert session.circuit == AUTO_CIRCUIT
    assert session.circuit_choices()[0] == AUTO_CIRCUIT
    use, caution = session.circuit_note()
    assert use and caution
    assert session.circuit_parameters() == [], (
        "antes del ajuste no hay circuito que listar: cuál se ajusta no se "
        "sabe hasta leer el espectro"
    )
    assert "analiza el espectro" in session.circuit_drawing()


def test_the_chosen_circuit_can_be_taken_over_by_hand(tmp_path):
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession()
    path = tmp_path / "m.csv"
    write_eis(_blocking_porous(), path)
    assert session.load_eis(path)
    assert not session.adopt_chosen_circuit(), (
        "sin análisis todavía no hay nada que adoptar"
    )
    assert session.analyse() is not None
    chosen = session.result.eis.fit.circuit
    assert session.adopt_chosen_circuit()
    assert session.circuit == chosen
    labels = [label for label, _, _ in session.circuit_parameters()]
    assert labels == list(session.result.eis.fit.values())
    assert chosen in session.circuit_drawing()


# -- the figure's style, and the language of its labels ----------------


def test_the_axis_labels_default_to_english_and_can_be_spanish():
    """Two audiences, not one.

    The window, the report and every message stay Spanish — they are for
    the person running the experiment. The figure is for whoever reads
    the paper, and this group publishes in English.
    """
    from ramancarbon.gui.figures_echem import LABELS, CompositeFigure

    spec = CompositeFigure()
    assert spec.language == "en"
    assert spec.text("duration") == "cycle duration / s"
    assert CompositeFigure(language="es").text("duration") == \
        "duración del ciclo / s"
    # Every key has to exist in both tables, or a figure comes out
    # half-translated, which is the most visible way one looks unfinished.
    assert set(LABELS["en"]) == set(LABELS["es"])


def test_a_label_full_of_latex_braces_still_substitutes():
    """`str.format` cannot be used here and the leak label proved it.

    Almost every label is LaTeX and LaTeX is made of braces:
    `$\\langle i_{par}\\rangle$` through `format` raises
    `KeyError: 'par'`.
    """
    from ramancarbon.gui.figures_echem import CompositeFigure

    spec = CompositeFigure()
    # The braces SURVIVE — they are the LaTeX — and nothing raises.
    assert r"i_{par}" in spec.text("leak_fit")
    assert spec.text("potential", reference="Ag/AgCl") == "E / V vs Ag/AgCl"
    assert spec.text("current_at", rate="5") == "|i| at 5 mV/s  /  µA"


def test_the_grid_goes_off_when_it_is_asked_to():
    """`grid(False, linewidth=…)` turns the grid ON, with a warning.

    matplotlib reads "you gave me line properties, so you must want a
    grid". The suite's style sets ``axes.grid: True`` for the on-screen
    figures, so every exported figure came out carrying one from code
    that had asked for it to go away.
    """
    import matplotlib
    from matplotlib.figure import Figure

    from ramancarbon.gui.figures_echem import draw_composite, every_axes
    from ramancarbon.gui.theme import PALETTES, matplotlib_style

    palette = PALETTES["claro"]
    assert matplotlib_style(palette)["axes.grid"] is True, (
        "si el estilo ya no enciende la rejilla, esta prueba no prueba nada"
    )
    session = _session_with_spectra(1)
    session.set_figure_panels(["eis_nyquist", "eis_bode"])
    for wanted in (False, True):
        session.composite.grid = wanted
        with matplotlib.rc_context(matplotlib_style(palette)):
            figure = Figure(figsize=(6, 6))
            draw_composite(figure, session.composite, session, palette)
            drawn = any(line.get_visible()
                        for axes in every_axes(figure)
                        for line in axes.get_xgridlines() + axes.get_ygridlines())
        assert drawn is wanted, f"grid={wanted} salió dibujada={drawn}"


def test_a_preset_moves_the_layout_and_leaves_the_panels_alone():
    from ramancarbon.gui.figures_echem import PRESET_LABELS

    session = _session_with_spectra(1)
    session.set_figure_panels(["eis_nyquist", "eis_bode"])
    before = list(session.composite.panels)
    assert session.use_figure_preset("articulo_1col")
    assert session.composite.width_in == pytest.approx(3.4)
    assert session.composite.columns == 1
    assert session.composite.panels == before, (
        "un preajuste dice a dónde va la figura, no qué lleva dentro"
    )
    assert not session.use_figure_preset("no_existe")
    assert {key for key, _ in PRESET_LABELS} <= set(
        __import__("ramancarbon.gui.figures_echem", fromlist=["PRESETS"]).PRESETS)


def test_the_standard_figure_is_one_action():
    """And it picks what the field publishes, not the first of the list."""
    session = _session_with_spectra(2)
    chosen = session.standard_figure()
    assert chosen, "con espectros cargados tiene que elegir algo"
    assert chosen[0] == "eis_nyquist"
    assert "eis_bode" in chosen
    assert chosen == session.composite.panels


def test_the_nyquist_panel_keeps_its_aspect_and_insets_the_corner():
    """The aspect is the panel.

    A box whose axes run 0–1400 against 0–8000 draws the 45° region of a
    porous electrode almost vertical, and anyone reading a transport
    regime off it is reading the axes. The inset is how the corner gets
    shown without giving that up.
    """
    import matplotlib
    from matplotlib.figure import Figure

    from ramancarbon.gui.figures_echem import draw_composite
    from ramancarbon.gui.theme import PALETTES, matplotlib_style

    palette = PALETTES["claro"]
    session = _session_with_spectra(2)
    session.set_figure_panels(["eis_nyquist"])
    with matplotlib.rc_context(matplotlib_style(palette)):
        figure = Figure(figsize=(4, 4))
        draw_composite(figure, session.composite, session, palette)
    main = figure.axes[0]
    assert main.get_xlim() == main.get_ylim()
    assert main.get_aspect() == 1.0
    # An inset is a CHILD of the axes it sits in, not an entry of
    # `figure.axes`.
    assert main.child_axes, "falta el recuadro del rincón"
    inset = main.child_axes[0]
    assert inset.get_xlim()[1] < main.get_xlim()[1]
    assert inset.get_aspect() == 1.0


def test_the_combined_panel_gives_the_phase_its_own_two_axes():
    """`twiny` keeps the LEFT y, which here is −Z″ in ohms.

    So the phase, which runs −90 to 0, drew as a flat line on the floor
    of the Nyquist. It needs a twinx for the y and a twiny OF THAT for
    the frequency.
    """
    import matplotlib
    from matplotlib.figure import Figure

    from ramancarbon.gui.figures_echem import draw_composite
    from ramancarbon.gui.theme import PALETTES, matplotlib_style

    palette = PALETTES["claro"]
    session = _session_with_spectra(1)
    session.set_figure_panels(["eis_nyquist_bode"])
    with matplotlib.rc_context(matplotlib_style(palette)):
        figure = Figure(figsize=(4, 4))
        draw_composite(figure, session.composite, session, palette)
    phase_axes = [a for a in figure.axes
                  if a.get_ylim()[0] > a.get_ylim()[1] < 0.0]
    assert phase_axes, "ningún eje lleva la fase invertida de +5 a −95"
    for axes in phase_axes:
        low, high = sorted(axes.get_ylim())
        assert low >= -100.0 and high <= 10.0, (
            f"la fase se está dibujando contra {axes.get_ylim()}, que son "
            "ohmios y no grados"
        )


# -- Mott-Schottky ----------------------------------------------------


def _mott_schottky_truth(density_cm3=5e17, flat_band=-0.25, epsilon=48.0,
                         area_cm2=0.5, temperature=298.15):
    """The slope a given carrier density has to produce, in F⁻² V⁻¹."""
    from ramancarbon.echem.mott_schottky import (ELEMENTARY_CHARGE, EPSILON_0)

    return 2.0 / (epsilon * EPSILON_0 * (area_cm2 * 1e-4) ** 2
                  * ELEMENTARY_CHARGE * density_cm3 * 1e6)


def _mott_schottky_series(density_cm3=5e17, flat_band=-0.25, kind="n",
                          epsilon=48.0, area_cm2=0.5, points=25, seed=3):
    """A synthetic ``(E, C)`` series with a KNOWN density and E_fb."""
    from ramancarbon.echem.mott_schottky import (BOLTZMANN, ELEMENTARY_CHARGE,
                                                 ROOM_TEMPERATURE)

    thermal = BOLTZMANN * ROOM_TEMPERATURE / ELEMENTARY_CHARGE
    slope = _mott_schottky_truth(density_cm3, flat_band, epsilon, area_cm2)
    potential = np.linspace(0.0, 0.8, points)
    if kind == "n":
        inverse = slope * (potential - flat_band - thermal)
    else:
        # the sweep has to stop BEFORE the intercept: past
        # ``flat_band - kT/e`` the line is negative and 1/sqrt is not real.
        potential = np.linspace(flat_band - 0.8,
                                flat_band - thermal - 0.02, points)
        inverse = -slope * (potential - flat_band + thermal)
    rng = np.random.default_rng(seed)
    capacitance = (1.0 / np.sqrt(inverse)) * (
        1.0 + rng.normal(0.0, 0.004, inverse.size))
    return potential, capacitance


def test_mott_schottky_recovers_the_density_and_the_flat_band():
    """The closed case: a line built from a known N and E_fb."""
    from ramancarbon.echem.mott_schottky import mott_schottky

    potential, capacitance = _mott_schottky_series()
    fit = mott_schottky(potential, capacitance, 1000.0, 48.0, 0.5)
    assert fit.carrier_type == "n"
    assert fit.carrier_density_cm3 == pytest.approx(5e17, rel=0.05)
    assert fit.flat_band_v == pytest.approx(-0.25, abs=0.01)
    assert fit.r_squared > 0.99


def test_a_p_type_electrode_comes_back_as_p():
    """The SIGN of the slope is the type, and kT/e changes sign with it."""
    from ramancarbon.echem.mott_schottky import mott_schottky

    potential, capacitance = _mott_schottky_series(flat_band=0.9, kind="p")
    fit = mott_schottky(potential, capacitance, 1000.0, 48.0, 0.5)
    assert fit.carrier_type == "p"
    assert fit.slope < 0
    assert fit.flat_band_v == pytest.approx(0.9, abs=0.01)
    assert fit.carrier_density_cm3 == pytest.approx(5e17, rel=0.05)


def test_the_thermal_term_is_taken_off_and_it_is_not_nothing():
    """25.7 mV at room temperature, and it is part of the intercept."""
    from ramancarbon.echem.mott_schottky import mott_schottky

    potential, capacitance = _mott_schottky_series()
    fit = mott_schottky(potential, capacitance, 1000.0, 48.0, 0.5)
    crossing = -fit.intercept / fit.slope
    assert fit.thermal_term_v == pytest.approx(0.0257, abs=0.0005)
    assert fit.flat_band_v == pytest.approx(crossing - fit.thermal_term_v,
                                            abs=1e-9)


def test_the_density_goes_as_one_over_epsilon_and_area_squared():
    """Both prefactors, because both are things people get wrong.

    ``N ∝ 1/(ε A²)``: an ε from the wrong phase or the geometric area of
    a rough electrode carries the carrier density away with it, and
    neither shows up in the residual.
    """
    from ramancarbon.echem.mott_schottky import mott_schottky

    potential, capacitance = _mott_schottky_series()
    base = mott_schottky(potential, capacitance, 1000.0, 48.0, 0.5)
    half_epsilon = mott_schottky(potential, capacitance, 1000.0, 24.0, 0.5)
    double_area = mott_schottky(potential, capacitance, 1000.0, 48.0, 1.0)
    assert half_epsilon.carrier_density_cm3 == pytest.approx(
        2.0 * base.carrier_density_cm3, rel=1e-9)
    assert double_area.carrier_density_cm3 == pytest.approx(
        0.25 * base.carrier_density_cm3, rel=1e-9)
    # And the flat band potential needs NEITHER of them.
    assert half_epsilon.flat_band_v == pytest.approx(base.flat_band_v,
                                                     abs=1e-9)
    assert double_area.flat_band_v == pytest.approx(base.flat_band_v, abs=1e-9)


def test_the_linear_region_is_found_not_assumed():
    """A Mott-Schottky plot is straight only in depletion.

    Past flat band the curve bends over; fitting the whole sweep returns
    the slope of a line through points that are not on one.
    """
    from ramancarbon.echem.mott_schottky import mott_schottky

    potential, capacitance = _mott_schottky_series(points=30)
    # Bend the last third, as a real electrode does when it starts to
    # conduct.
    bent = capacitance.copy()
    bent[20:] *= np.linspace(1.0, 2.2, 10)
    fit = mott_schottky(potential, bent, 1000.0, 48.0, 0.5)
    assert fit.points < potential.size, "se ha ajustado el barrido entero"
    assert fit.window[1] < float(potential[-1])
    assert fit.carrier_density_cm3 == pytest.approx(5e17, rel=0.15)


def test_a_single_frequency_is_reported_as_not_a_measurement():
    from ramancarbon.echem.mott_schottky import _check_the_frequencies
    from ramancarbon.echem.mott_schottky import (MottSchottkyResult,
                                                 mott_schottky)

    potential, capacitance = _mott_schottky_series()
    result = MottSchottkyResult(
        fits=[mott_schottky(potential, capacitance, 1000.0, 48.0, 0.5)])
    _check_the_frequencies(result)
    assert any("una frecuencia" in w.lower() for w in result.warnings)


def test_frequency_dispersion_is_the_result_not_a_detail():
    """E_fb moving with frequency IS the finding, so it is reported."""
    from ramancarbon.echem.mott_schottky import (MottSchottkyResult,
                                                 _check_the_frequencies,
                                                 mott_schottky)

    fits = []
    for index, flat_band in enumerate((-0.25, -0.38, -0.50)):
        potential, capacitance = _mott_schottky_series(flat_band=flat_band,
                                                       seed=3 + index)
        fits.append(mott_schottky(potential, capacitance,
                                  10.0 ** (3 - index), 48.0, 0.5))
    result = MottSchottkyResult(fits=fits)
    _check_the_frequencies(result)
    assert result.flat_band_spread() == pytest.approx(0.25, abs=0.02)
    assert any("se mueve" in w for w in result.warnings)
    # And the consensus is ONE fit, never an average of the three.
    best = result.consensus()
    assert best in fits


def test_a_capacitance_near_the_helmholtz_layer_is_flagged():
    """What is measured is C_sc in SERIES with C_H.

    The equation describes it only while C_sc is far below C_H, and the
    end of the plot the intercept is read from is where that stops
    holding.
    """
    from ramancarbon.echem.mott_schottky import (HELMHOLTZ_F_PER_CM2,
                                                 mott_schottky)

    area = 1.0
    potential = np.linspace(0.0, 0.5, 20)
    # A capacitance deliberately up at the Helmholtz scale.
    capacitance = np.linspace(0.9, 0.4, 20) * HELMHOLTZ_F_PER_CM2 * area
    fit = mott_schottky(potential, capacitance, 1000.0, 48.0, area)
    assert any("Helmholtz" in w for w in fit.warnings)


def test_the_dielectric_constant_comes_from_the_database_with_its_source():
    from ramancarbon.echem.mott_schottky import (dielectric_constant,
                                                 semiconductors)

    table = semiconductors()
    assert table, "la tabla de semiconductores está vacía"
    for key, entry in table.items():
        assert entry.get("epsilon", 0) > 0, key
        assert entry.get("source"), key
        assert entry.get("confidence") in {"high", "medium", "low"}, key
    value, source = dielectric_constant("ZnO")
    assert value == pytest.approx(8.5)
    assert "ZnO" in source and "confianza" in source
    missing, reason = dielectric_constant("Vibranio")
    assert missing is None
    assert "1/ε" in reason, (
        "un material que no está tiene que decir POR QUÉ no se puede "
        "suponer una ε, no sólo que no está"
    )


def test_the_series_capacitance_is_the_one_the_equation_uses(tmp_path):
    """``C_s = −1/(ωZ″)``, not ``C′``.

    The model is a space-charge capacitor in series with the Helmholtz
    layer, so the ``C`` of the equation is the series combination the
    cell presents.
    """
    from ramancarbon.echem.mott_schottky import series_capacitance

    spectrum = make_eis_demo("R0-Q1", seed=3)
    index = spectrum.n // 2
    frequency = float(spectrum.frequency[index])
    omega = 2.0 * math.pi * frequency
    expected = -1.0 / (omega * float(spectrum.z.imag[index]))
    assert series_capacitance(spectrum, frequency) == pytest.approx(
        expected, rel=0.02)
    with pytest.raises(CurveError):
        series_capacitance(spectrum, 10.0 * float(spectrum.frequency.max()))


def test_the_bias_of_each_spectrum_is_read_from_the_file():
    """A folder of PEIS files at different biases IS the experiment.

    The potential is in the file — EC-Lab writes it in ``<Ewe>/V`` —
    so it is read rather than typed back in, which is how a point ends
    up on the wrong abscissa.
    """
    from ramancarbon.echem.mott_schottky import series_from_spectra

    spectra = []
    for bias in (-0.2, 0.0, 0.2):
        spectrum = make_eis_demo("R0-Q1", seed=3)
        spectrum.metadata["potencial_cd_v"] = bias
        spectra.append(spectrum)
    naked = make_eis_demo("R0-Q1", seed=3)
    naked.name = "sin potencial"
    spectra.append(naked)

    potential, capacitance, notes = series_from_spectra(
        spectra, float(spectra[0].frequency[spectra[0].n // 2]))
    assert list(potential) == [-0.2, 0.0, 0.2]
    assert capacitance.size == 3
    assert any("sin potencial" in note for note in notes)


def test_the_section_runs_mott_schottky_end_to_end():
    from ramancarbon.echem.curve import Electrode, Impedance
    from ramancarbon.echem.mott_schottky import (BOLTZMANN, ELEMENTARY_CHARGE,
                                                 ROOM_TEMPERATURE)
    from ramancarbon.gui.echem_state import (MOTT_SCHOTTKY_COLUMNS,
                                             EchemSession)

    epsilon, area, density, flat_band = 48.0, 0.5, 5e17, -0.25
    thermal = BOLTZMANN * ROOM_TEMPERATURE / ELEMENTARY_CHARGE
    slope = _mott_schottky_truth(density, flat_band, epsilon, area)
    session = EchemSession(name="semiconductor")
    session.electrode = Electrode(area_cm2=area)
    session.ms_material = "TiO2_anatasa"
    frequency = np.logspace(5, 1, 40)
    for bias in np.linspace(0.0, 0.8, 17):
        space_charge = 1.0 / math.sqrt(slope * (bias - flat_band - thermal))
        z = 25.0 + 1.0 / (2j * math.pi * frequency * space_charge)
        session.ms_series.append(Impedance(
            frequency=frequency, z=z, electrode=session.electrode,
            name=f"{bias:+.3f} V",
            metadata={"potencial_cd_v": float(bias)}))

    result = session.mott_schottky()
    assert result is not None and result.available
    assert len(result.fits) == 3, "tres frecuencias por defecto, no una"
    best = result.consensus()
    assert best.carrier_type == "n"
    assert best.carrier_density_cm3 == pytest.approx(density, rel=0.05)
    assert best.flat_band_v == pytest.approx(flat_band, abs=0.01)

    rows = session.mott_schottky_rows()
    assert rows[0] == MOTT_SCHOTTKY_COLUMNS
    assert len(rows) == 4
    assert session.mott_schottky() is result, "el resultado no se cacheó"

    # And the panel is offered now that there is a series.
    assert "mott_schottky" in {p.key for p in session.figure_panels()}
    assert "mott_schottky" in session.standard_figure()


# -- galvanostatic: the audit --------------------------------------------


def _gcd_curve(window=(-0.6, 0.4), charge_a=20e-6, discharge_a=-20e-6,
               capacitance=45e-6, resistance=80.0, cycles=3,
               points_per_branch=200, mass_mg=0.02):
    """An IDEAL capacitor charged and discharged over an arbitrary window.

    The window is the point: it straddles zero, like every three-electrode
    measurement referenced to Ag/AgCl, and the energy of an ideal
    capacitor over it is exactly ``½CΔV²`` whatever the reference
    electrode happens to be sitting at.
    """
    from ramancarbon.echem.curve import ChargeDischarge, Electrode

    low, high = window
    span = high - low
    times, potentials, currents = [], [], []
    clock = 0.0
    for _ in range(cycles):
        for current in (charge_a, discharge_a):
            duration = capacitance * span / abs(current)
            t = np.linspace(0.0, duration, points_per_branch)
            ramp = (t / duration) * span
            base = low + ramp if current > 0 else high - ramp
            # The IR jump: the potential sits I*R away from the ideal ramp
            # for as long as the current flows, and jumps at every switch.
            base = base + current * resistance
            times.append(clock + t)
            potentials.append(base)
            currents.append(np.full(t.size, current))
            clock += duration
    return ChargeDischarge(
        time=np.concatenate(times),
        potential=np.concatenate(potentials),
        current=np.concatenate(currents),
        electrode=Electrode(mass_mg=mass_mg, area_cm2=0.0707,
                            electrolyte="KOH_3M"),
        name="gcd sintetico",
    )


def test_the_energy_is_referenced_to_the_branch_floor_not_to_zero():
    """``∫V dq`` on the ABSOLUTE axis is not an energy on a window that
    straddles zero, and every test in this file used to miss it because
    the demo window starts at 0.0 V, where the two agree exactly.

    Closed case: an ideal capacitor over −0.6 to +0.4 V stores
    ``½CΔV²`` = ½·45 µF·1 V² = 22.5 µJ. Integrated against the absolute
    axis the same discharge comes out at a fifth of that, because the
    mean potential over the window is −0.1 V instead of +0.5 V.
    """
    from ramancarbon.core.compat import trapezoid
    from ramancarbon.echem.gcd import analyse_gcd

    curve = _gcd_curve(window=(-0.6, 0.4), capacitance=45e-6,
                       resistance=0.0)
    branch = analyse_gcd(curve).discharges[-1]
    assert branch.energy_j == pytest.approx(0.5 * 45e-6 * 1.0 ** 2, rel=0.02)

    # And what the absolute axis would have given, for the record.
    halves = [h for h in __import__(
        "ramancarbon.echem.gcd", fromlist=["half_cycles"]
    ).half_cycles(curve) if h.kind == "descarga"]
    last = halves[-1]
    q = abs(last.current_a) * (last.time - last.time[0])
    absolute = abs(float(trapezoid(last.potential, q)))
    assert absolute < 0.3 * branch.energy_j


def test_the_energy_floor_does_not_depend_on_the_reference_electrode():
    """The same capacitor on two potential axes has to store the same
    energy. That is the whole argument for the floor, and an absolute
    integral fails it by construction."""
    from ramancarbon.echem.gcd import analyse_gcd

    here = analyse_gcd(_gcd_curve(window=(-0.6, 0.4), resistance=0.0))
    shifted = analyse_gcd(_gcd_curve(window=(0.0, 1.0), resistance=0.0))
    assert here.discharges[-1].energy_j == pytest.approx(
        shifted.discharges[-1].energy_j, rel=1e-6)
    assert here.energy_wh_per_kg == pytest.approx(
        shifted.energy_wh_per_kg, rel=1e-6)


def test_the_ir_resistance_divides_by_the_current_STEP():
    """The jump spans ``|I_charge| + |I_discharge|``, which is ``2|I|``
    only when the two are equal.

    The user's fifth file charges at 60 µA and discharges at 20, and the
    old denominator returned twice the resistance its other four files
    agree on. Here the planted resistance is 80 Ω both ways.
    """
    from ramancarbon.echem.gcd import analyse_gcd

    symmetric = analyse_gcd(_gcd_curve(charge_a=20e-6, discharge_a=-20e-6,
                                       resistance=80.0))
    assert symmetric.resistance_ohm == pytest.approx(80.0, rel=0.05)

    asymmetric = analyse_gcd(_gcd_curve(charge_a=60e-6, discharge_a=-20e-6,
                                        resistance=80.0))
    assert asymmetric.resistance_ohm == pytest.approx(80.0, rel=0.05)
    # Dividing by 2|I_discharge| would have doubled it.
    assert asymmetric.resistance_ohm < 1.5 * 80.0
    assert "µA" in asymmetric.resistance_how


def test_an_asymmetric_pair_is_named_as_such():
    """Two different currents is not a cycle, and the efficiency computed
    from it does not measure reversibility."""
    from ramancarbon.echem.gcd import analyse_gcd

    result = analyse_gcd(_gcd_curve(charge_a=60e-6, discharge_a=-20e-6))
    assert result.asymmetric is True
    assert result.charge_current_a == pytest.approx(60e-6, rel=0.01)
    assert any("simétrico" in w for w in result.warnings)
    assert analyse_gcd(_gcd_curve()).asymmetric is False


def test_a_three_point_half_cycle_still_gives_its_charge_and_its_window():
    """A real GCPL saves on a potential condition and writes three points
    per half cycle. The whole measurement used to be thrown away: the
    splitter built a ``ChargeDischarge`` per branch, which refuses fewer
    than ten points, so it raised from inside itself on all five of the
    user's files."""
    from ramancarbon.echem.gcd import MIN_SHAPE_POINTS, analyse_gcd

    curve = _gcd_curve(points_per_branch=3, cycles=4)
    result = analyse_gcd(curve)          # used to raise CurveError
    branch = result.discharges[-1]
    assert branch.points == 3
    # q and C by dV need only the endpoints.
    assert branch.charge_c > 0
    assert branch.capacitance_f is not None
    # The shape-based numbers are refused, with a reason.
    assert branch.capacitance_slope_f is None
    assert branch.capacitance_energy_f is None
    assert branch.refusals and str(MIN_SHAPE_POINTS) in branch.refusals[0]


def test_segments_no_longer_raises_on_a_short_branch():
    """``segments()`` returns curves, so its floor has to be the curve
    minimum. Five was below it."""
    from ramancarbon.echem.curve import MIN_CURVE_POINTS

    curve = _gcd_curve(points_per_branch=3, cycles=4)
    assert curve.segments() == []        # nothing long enough to be a curve
    longer = _gcd_curve(points_per_branch=MIN_CURVE_POINTS, cycles=2)
    assert len(longer.segments()) == 4


def test_the_coulombic_partner_is_the_charge_BEFORE_the_discharge():
    """Indexing a separate list of charges by position is right only when
    the file starts with a charge."""
    from ramancarbon.echem.gcd import analyse_gcd

    curve = _gcd_curve(cycles=3)
    # Start the file on a discharge by dropping the first charge branch.
    start = 200
    from ramancarbon.echem.curve import ChargeDischarge
    trimmed = ChargeDischarge(
        time=curve.time[start:], potential=curve.potential[start:],
        current=curve.current[start:], electrode=curve.electrode,
        name="empieza en descarga")
    result = analyse_gcd(trimmed)
    assert result.branches[0].kind == "descarga"
    # The reported discharge is the last one, and its partner is the
    # charge immediately before it, so the efficiency stays near one.
    assert result.coulombic_efficiency == pytest.approx(1.0, abs=0.05)


def _leaky_curve(current=20e-6, leak=0.2e-6, capacitance=45e-6,
                 window=(-0.6, 0.4), cycles=3):
    """A cell with a known parasitic current.

    The leak supplies extra charge on the way out, so the discharge runs
    longer than the charge by exactly ``leak·T/I`` with ``T`` the whole
    cycle. That is the identity the analysis has to invert.
    """
    from ramancarbon.echem.curve import ChargeDischarge, Electrode

    low, high = window
    span = high - low
    t_charge = capacitance * span / current
    # Solve t_dis from  I*t_dis - I*t_ch = leak*(t_ch + t_dis).
    t_discharge = (current + leak) * t_charge / (current - leak)
    times, potentials, currents = [], [], []
    clock = 0.0
    for _ in range(cycles):
        for signed, duration in ((current, t_charge),
                                 (-current, t_discharge)):
            t = np.linspace(0.0, duration, 200)
            ramp = (t / duration) * span
            potentials.append(low + ramp if signed > 0 else high - ramp)
            times.append(clock + t)
            currents.append(np.full(t.size, signed))
            clock += duration
    return ChargeDischarge(
        time=np.concatenate(times), potential=np.concatenate(potentials),
        current=np.concatenate(currents),
        electrode=Electrode(mass_mg=0.02, area_cm2=0.0707,
                            electrolyte="KOH_3M"),
        name="con fuga")


def test_the_coulombic_excess_is_reported_as_a_leakage_CURRENT():
    """A leak is a current, so the SAME leak is a big percentage at a
    small current and a small one at a large current. A percentage cannot
    be compared across currents; a current can.

    Closed case: a planted 0.2 µA leak has to come back as 0.2 µA at
    every applied current.
    """
    from ramancarbon.echem.gcd import analyse_gcd

    for current in (5e-6, 20e-6, 40e-6):
        result = analyse_gcd(_leaky_curve(current=current, leak=0.2e-6))
        assert result.leakage_a == pytest.approx(0.2e-6, rel=0.02), current
        # And the percentage it shows up as does depend on the current,
        # which is the whole reason the current is the thing reported.
        assert result.coulombic_efficiency > 1.0
    slow = analyse_gcd(_leaky_curve(current=5e-6, leak=0.2e-6))
    fast = analyse_gcd(_leaky_curve(current=40e-6, leak=0.2e-6))
    assert (slow.coulombic_efficiency - 1.0) > 4 * (
        fast.coulombic_efficiency - 1.0)

    # An ideal symmetric cycle leaks nothing.
    assert abs(analyse_gcd(_gcd_curve(resistance=0.0)).leakage_a) < 1e-9


def test_the_leak_is_divided_by_the_WHOLE_cycle_not_by_the_charge():
    """A parasitic reaction runs the whole time, not only while the cell
    is being charged. On a symmetric cycle the two denominators look
    equivalent because the halves are the same length; on an asymmetric
    one they are not, and that is what settled it.

    Over the user's five files the charge-only denominator gives 0.534,
    0.448, 0.390, 0.383 and **0.915** µA, with the asymmetric file 2.4×
    above the rest; the whole cycle gives 0.254, 0.219, 0.193, 0.190 and
    0.226, which lands it in the middle. All five agreeing is the check.
    """
    from ramancarbon.echem.gcd import analyse_gcd

    result = analyse_gcd(_leaky_curve(current=20e-6, leak=0.2e-6))
    branch = result.discharges[-1]
    charge = [b for b in result.branches if b.kind == "carga"][-1]
    excess = branch.charge_c - charge.charge_c
    whole = excess / (charge.duration_s + branch.duration_s)
    charge_only = excess / charge.duration_s
    assert result.leakage_a == pytest.approx(whole, rel=1e-9)
    assert result.leakage_a != pytest.approx(charge_only, rel=0.1)


def test_cycling_from_one_file_gives_a_point_per_cycle():
    """``analyse_cycling`` takes a LIST and gives one point per file. A
    GCPL holds its cycles inside ONE file — the user's hold 1001 each —
    and all of that collapsed to a single number."""
    from ramancarbon.echem.gcd import analyse_cycling, cycling_from_curve

    curve = _gcd_curve(cycles=12, points_per_branch=20)
    per_cycle = cycling_from_curve(curve)
    assert len(per_cycle.cycles) == 12
    assert len(per_cycle.capacitance_f) == 12
    assert per_cycle.retention == pytest.approx(1.0, abs=0.02)
    assert "un solo archivo" in per_cycle.source

    # The list route still gives one point per measurement.
    whole = analyse_cycling([curve, curve])
    assert len(whole.cycles) == 2
    assert "una por ciclo" in whole.source


def test_cycling_sees_a_real_fade_and_names_its_shape():
    """A capacitance that decays as √cycles is a film growing and slowing
    itself down. The user's 5 µA file does exactly that, losing 80 % of
    its capacitance over 1001 cycles."""
    from ramancarbon.echem.gcd import cycling_from_curve
    from ramancarbon.echem.kinetics import fade_model

    pieces = []
    for cycle in range(40):
        fraction = 1.0 - 0.5 * np.sqrt(cycle / 40.0)
        pieces.append(_gcd_curve(capacitance=45e-6 * fraction, cycles=1,
                                 points_per_branch=20, resistance=0.0))
    from ramancarbon.echem.curve import ChargeDischarge
    clock, times = 0.0, []
    for piece in pieces:
        times.append(piece.time - piece.time[0] + clock)
        clock = times[-1][-1] + 1.0
    joined = ChargeDischarge(
        time=np.concatenate(times),
        potential=np.concatenate([p.potential for p in pieces]),
        current=np.concatenate([p.current for p in pieces]),
        electrode=pieces[0].electrode, name="con desvanecimiento")
    result = cycling_from_curve(joined)
    assert len(result.cycles) == 40
    assert result.retention == pytest.approx(0.5, abs=0.05)
    shape = fade_model(result.cycles, result.capacity_c)
    assert shape.model == "raiz"


def test_a_cycling_curve_that_changes_current_is_not_a_retention():
    """Mixing a rate test into a cycling curve makes its retention
    meaningless, and that has to be said rather than plotted."""
    from ramancarbon.echem.curve import ChargeDischarge
    from ramancarbon.echem.gcd import cycling_from_curve

    slow = _gcd_curve(charge_a=5e-6, discharge_a=-5e-6, cycles=6,
                      points_per_branch=20)
    fast = _gcd_curve(charge_a=40e-6, discharge_a=-40e-6, cycles=6,
                      points_per_branch=20)
    joined = ChargeDischarge(
        time=np.concatenate([slow.time, fast.time + slow.time[-1] + 1.0]),
        potential=np.concatenate([slow.potential, fast.potential]),
        current=np.concatenate([slow.current, fast.current]),
        electrode=slow.electrode, name="dos corrientes")
    result = cycling_from_curve(joined)
    assert any("no es la misma en todos los ciclos" in w
               for w in result.warnings)


def test_the_power_ceiling_comes_from_the_resistance_not_the_discharge():
    """``ΔV²/(4·ESR)`` is what a cell of this resistance could deliver
    into a matched load. It is an upper bound on the Ragone power axis,
    and no measured discharge reaches it."""
    from ramancarbon.echem.gcd import analyse_gcd

    result = analyse_gcd(_gcd_curve(resistance=80.0))
    assert result.max_power_w_per_kg is not None
    assert result.power_w_per_kg is not None
    assert result.max_power_w_per_kg > result.power_w_per_kg


def _gcd_session(currents=(5e-6, 10e-6, 20e-6, 40e-6), asymmetric=None):
    """A session holding a rate series, optionally with a broken one."""
    from ramancarbon.echem.curve import Electrode
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession(name="serie")
    session.electrode = Electrode(mass_mg=0.02, area_cm2=0.0707,
                                  electrolyte="KOH_3M")
    for current in currents:
        curve = _gcd_curve(charge_a=current, discharge_a=-current,
                           cycles=3, points_per_branch=60)
        curve.electrode = session.electrode
        session.gcd_series.append(curve)
    if asymmetric is not None:
        charge, discharge = asymmetric
        curve = _gcd_curve(charge_a=charge, discharge_a=-discharge,
                           cycles=3, points_per_branch=60)
        curve.electrode = session.electrode
        session.gcd_series.append(curve)
    session._forget_gcd()
    from ramancarbon.gui.echem_state import _discharge_current
    session.gcd_series.sort(key=_discharge_current)
    return session


def test_a_rate_point_is_labelled_by_its_DISCHARGE_current():
    """The median over the whole record returns a current neither branch
    used. The user's fifth file charges at 60 µA and discharges at 20
    with the same number of points in each, so the median is **40** —
    exactly between the two, and wrong for a rate point, which is at the
    discharge current because that is where a device is measured."""
    from ramancarbon.gui.echem_state import (_discharge_current,
                                             _typical_current)

    curve = _gcd_curve(charge_a=60e-6, discharge_a=-20e-6,
                       points_per_branch=60)
    assert _discharge_current(curve) == pytest.approx(20e-6, rel=0.01)
    assert _typical_current(curve) != pytest.approx(20e-6, rel=0.1)


def test_an_asymmetric_curve_stays_out_of_the_rate_curve_by_default():
    """It has no single current density, so it cannot be a point on a
    plot whose x axis is current. The default keeps it in the tables and
    out of the curves."""
    session = _gcd_session(asymmetric=(60e-6, 20e-6))
    assert session.gcd_asymmetry == "aparte"
    assert len(session.gcd_asymmetric()) == 1
    # Five curves loaded, four on the rate plot and four on the Ragone.
    assert len(session.gcd_curves()) == 5
    assert len(session.capacitance_vs_current()) == 4
    assert len(session.ragone_points()) == 4
    # But it IS in the table, flagged.
    rows = session.gcd_rows()
    assert len(rows) == 5
    assert sum(1 for row in rows if row[-1] == "sí") == 1
    assert any("/−" in row[0] for row in rows)


def test_the_asymmetry_policy_is_a_choice_and_all_four_work():
    session = _gcd_session(asymmetric=(60e-6, 20e-6))
    from ramancarbon.echem.gcd import ASYMMETRY_POLICIES

    counts = {}
    for policy in ASYMMETRY_POLICIES:
        session.gcd_asymmetry = policy
        counts[policy] = len(session.capacitance_vs_current())
    assert counts["aparte"] == 4
    assert counts["excluir"] == 4
    assert counts["incluir"] == 5
    assert counts["interpolar"] == 5


def test_interpolating_reports_the_measured_value_NEXT_TO_the_estimate():
    """Quietly substituting an interpolated number for a measured one
    hides whether the broken schedule changed the answer. On the user's
    file it changed it by 13 %."""
    session = _gcd_session(asymmetric=(60e-6, 20e-6))
    rows = session.gcd_interpolated()
    assert len(rows) == 1
    label, milliamps, measured, interpolated = rows[0]
    assert "/−" in label
    assert milliamps == pytest.approx(0.02, rel=0.01)
    assert measured > 0 and interpolated > 0


def test_interpolating_refuses_outside_the_measured_currents():
    """``np.interp`` clips, and clipping in silence is inventing."""
    session = _gcd_session(currents=(10e-6, 20e-6),
                           asymmetric=(200e-6, 100e-6))
    assert session.gcd_interpolated() == []


def test_the_session_reaches_the_cycling_of_each_file():
    """A GCPL holds its cycles inside one file, and the section had no
    way to see them."""
    session = _gcd_session(currents=(10e-6,))
    rows = session.gcd_cycling_rows()
    assert len(rows) == 1
    assert rows[0][1] == "3"          # three cycles in the synthetic curve
    cyclings = session.gcd_cyclings()
    assert cyclings[0] is not None
    assert len(cyclings[0].cycles) == 3


def test_the_gcd_caches_forget_when_the_series_changes():
    """A cache keyed on the identity of the curves has to be dropped when
    the list changes: answering for the wrong curve is worse than no
    cache."""
    session = _gcd_session(currents=(10e-6, 20e-6))
    assert len(session.gcd_analyses()) == 2
    session.gcd_series.append(_gcd_curve(charge_a=40e-6, discharge_a=-40e-6,
                                         points_per_branch=60))
    session.gcd_series[-1].electrode = session.electrode
    session._forget_gcd()
    assert len(session.gcd_analyses()) == 3
    session.clear("gcd_series")
    assert session.gcd_analyses() == []


# -- the figure tab: techniques, scales, the new panels ------------------


def test_every_panel_declares_a_technique_that_the_selector_offers():
    from ramancarbon.gui.figures_echem import ECHEM_PANELS, TECHNIQUES

    known = {key for key, _ in TECHNIQUES}
    for panel in ECHEM_PANELS:
        assert panel.technique in known, panel.key
    # And every technique other than "todo" has at least one panel.
    used = {panel.technique for panel in ECHEM_PANELS}
    assert known - {"todo"} == used


def test_the_catalogue_covers_gcd_and_catalysis():
    """The figure tab had no galvanostatic panel and no HER/OER panel at
    all, so a charge-discharge series could be analysed and not drawn."""
    from ramancarbon.gui.figures_echem import ECHEM_PANELS

    by_technique: dict[str, int] = {}
    for panel in ECHEM_PANELS:
        by_technique[panel.technique] = by_technique.get(panel.technique, 0) + 1
    assert by_technique.get("gcd", 0) >= 6
    assert by_technique.get("catalysis", 0) >= 2
    assert by_technique.get("eis", 0) >= 6
    assert by_technique.get("cv", 0) >= 5


def test_the_technique_filter_narrows_the_list_not_the_figure():
    session = _gcd_session()
    from ramancarbon.gui.figures_echem import available_panels

    everything = available_panels(session, "todo")
    only_gcd = available_panels(session, "gcd")
    assert {p.key for p in only_gcd} <= {p.key for p in everything}
    assert all(p.technique == "gcd" for p in only_gcd)
    assert len(only_gcd) >= 6

    # Panels already chosen survive a filter that hides them.
    session.composite.panels = ["gcd_series", "gcd_ragone"]
    session.figure_technique = "eis"
    assert session.composite.panels == ["gcd_series", "gcd_ragone"]


def test_the_technique_selector_only_offers_what_has_data():
    session = _gcd_session()
    offered = dict(session.figure_techniques())
    assert "todo" in offered and "gcd" in offered
    assert "eis" not in offered and "catalysis" not in offered


def test_a_panel_scale_can_be_forced_and_round_trips():
    """«Both axes logarithmic» has to be reachable for every panel, not
    only for the ones somebody thought to write that way."""
    from ramancarbon.gui.figures_echem import CompositeFigure

    spec = CompositeFigure(panels=["eis_bode"])
    assert spec.scale_for("eis_bode") == "auto"
    spec.scales["eis_bode"] = "loglog"
    assert spec.scale_for("eis_bode") == "loglog"
    assert spec.scale_for("no_existe") == "auto"
    # An unknown choice degrades to auto rather than reaching matplotlib.
    spec.scales["eis_bode"] = "cuadratico"
    assert spec.scale_for("eis_bode") == "auto"
    spec.scales["eis_bode"] = "loglog"
    assert CompositeFigure.from_dict(spec.to_dict()).scales == spec.scales


def test_a_log_scale_is_refused_where_it_would_empty_the_axis():
    """A phase axis runs −90 to 0. Forced logarithmic it draws nothing,
    so that one is refused and left linear."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ramancarbon.gui.figures_echem import apply_scale

    figure, axis = plt.subplots()
    axis.plot([1.0, 10.0, 100.0], [1.0, 10.0, 100.0])
    apply_scale(axis, "loglog")
    assert axis.get_xscale() == "log" and axis.get_yscale() == "log"
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot([1.0, 10.0], [-90.0, 0.0])
    apply_scale(axis, "loglog")
    assert axis.get_xscale() == "log"
    assert axis.get_yscale() == "linear"
    plt.close(figure)


def test_the_scale_test_is_on_the_data_limits_not_the_view():
    """The view carries matplotlib's autoscale margin, so an axis holding
    1 to 100 reports a lower limit near −4 and a check on that refuses
    every panel there is."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots()
    axis.plot([1.0, 10.0, 100.0], [1.0, 10.0, 100.0])
    assert axis.get_xlim()[0] < 0.0          # the margin, which is the trap
    assert axis.dataLim.intervalx[0] == pytest.approx(1.0)
    plt.close(figure)


def test_every_gcd_panel_draws_from_a_real_series():
    """Each panel in its own figure, because a twin axis steals the lines
    of the panels after it when they share one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ramancarbon.gui.figures_echem import (CompositeFigure,
                                               available_panels,
                                               draw_composite, every_axes)
    from ramancarbon.gui.theme import PALETTES

    session = _gcd_session(asymmetric=(60e-6, 20e-6))
    panels = available_panels(session, "gcd")
    assert len(panels) >= 6
    for panel in panels:
        spec = CompositeFigure(panels=[panel.key], columns=1)
        figure = plt.figure(figsize=(5.0, 3.5))
        problems = draw_composite(figure, spec, session, PALETTES["claro"])
        assert problems == [], f"{panel.key}: {problems}"
        drawn = sum(len(axes.get_lines()) for axes in every_axes(figure))
        assert drawn > 0, panel.key
        plt.close(figure)


def test_the_standard_gcd_figure_draws_whole():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ramancarbon.gui.figures_echem import draw_composite
    from ramancarbon.gui.theme import PALETTES

    session = _gcd_session(asymmetric=(60e-6, 20e-6))
    session.figure_technique = "gcd"
    chosen = session.standard_figure()
    assert "gcd_series" in chosen and "gcd_ragone" in chosen
    figure = plt.figure(figsize=session.composite.figsize())
    assert draw_composite(figure, session.composite, session,
                          PALETTES["claro"]) == []
    plt.close(figure)


def test_a_preset_resets_what_it_does_not_mention():
    """Only the draft preset asks for a grid, so moving off it has to put
    the grid back. A preset that leaves the previous one's decisions in
    place is not a preset."""
    from ramancarbon.gui.figures_echem import (PRESETS, CompositeFigure,
                                               apply_preset)

    assert len(PRESETS) >= 10
    spec = CompositeFigure(panels=["cv_dunn"])
    assert apply_preset(spec, "borrador")
    assert spec.grid is True
    assert apply_preset(spec, "articulo_2col")
    assert spec.grid is False
    assert spec.panels == ["cv_dunn"]      # panels are never touched


def test_every_preset_has_a_label_and_the_other_way_round():
    from ramancarbon.gui.figures_echem import PRESETS, PRESET_LABELS

    assert {key for key, _ in PRESET_LABELS} == set(PRESETS)
    assert len({label for _, label in PRESET_LABELS}) == len(PRESET_LABELS)


def test_every_gcd_panel_can_export_its_numbers():
    """A line with no label is ``_child0`` to matplotlib and the data
    export skips it, so a panel can draw numbers nobody can get out —
    which is the same failure the stackplot panel had, by another route.
    """
    session = _gcd_session(asymmetric=(60e-6, 20e-6))
    from ramancarbon.gui.figures_echem import available_panels

    session.composite.panels = [p.key for p in available_panels(session, "gcd")]
    rows = session.figure_data_rows()
    for key, table in rows.items():
        assert len(table) > 1, f"{key} exportó sólo su cabecera"


def test_a_sudden_step_in_the_cycling_is_not_called_degradation():
    """Degradation is gradual by definition. The user's 5 µA file has
    exactly two drops over 20 % in 1001 cycles — the first cycle, which
    is formation, and cycle 319, which loses 66 % in ONE cycle while the
    coulombic efficiency barely moves. A fade model fitted across that
    describes the step, not a mechanism."""
    from ramancarbon.echem.curve import ChargeDischarge
    from ramancarbon.echem.gcd import cycling_from_curve

    before = [_gcd_curve(capacitance=45e-6, cycles=1, points_per_branch=20,
                         resistance=0.0) for _ in range(8)]
    after = [_gcd_curve(capacitance=15e-6, cycles=1, points_per_branch=20,
                        resistance=0.0) for _ in range(8)]
    clock, times = 0.0, []
    pieces = before + after
    for piece in pieces:
        times.append(piece.time - piece.time[0] + clock)
        clock = times[-1][-1] + 1.0
    joined = ChargeDischarge(
        time=np.concatenate(times),
        potential=np.concatenate([p.potential for p in pieces]),
        current=np.concatenate([p.current for p in pieces]),
        electrode=pieces[0].electrode, name="con un escalón")
    result = cycling_from_curve(joined)
    assert any("Un salto NO es una degradación" in w
               for w in result.warnings)

    # A smooth fade of the same total size is not flagged.
    smooth = []
    for cycle in range(16):
        smooth.append(_gcd_curve(capacitance=45e-6 * (1 - 0.04 * cycle),
                                 cycles=1, points_per_branch=20,
                                 resistance=0.0))
    clock, times = 0.0, []
    for piece in smooth:
        times.append(piece.time - piece.time[0] + clock)
        clock = times[-1][-1] + 1.0
    gentle = ChargeDischarge(
        time=np.concatenate(times),
        potential=np.concatenate([p.potential for p in smooth]),
        current=np.concatenate([p.current for p in smooth]),
        electrode=smooth[0].electrode, name="sin escalón")
    assert not any("Un salto NO" in w
                   for w in cycling_from_curve(gentle).warnings)


# -- HER/OER: the barriers a single polarisation curve cannot raise -----


def _her_curve(onset_v=0.05, tafel_mv=42.0, exchange=1e-3, points=400,
               scan_rate=0.001, offset_v=0.0, resistance=0.0,
               background_ma_cm2=0.0, area_cm2=0.0707):
    """A synthetic HER curve with a KNOWN Tafel slope and onset.

    ``offset_v`` shifts the whole potential axis, which is what a
    reference electrode sitting somewhere other than the table says
    does, and the thing the onset-sign check has to catch.
    """
    from ramancarbon.echem.curve import Electrode, Voltammogram

    # eta measured from RHE, cathodic branch, as a Tafel law.
    eta = np.linspace(-0.25, 0.20, points)
    j = exchange * 10.0 ** (np.clip(eta, 0.0, None) / (tafel_mv * 1e-3))
    j = j - background_ma_cm2
    current = -j * 1e-3 * area_cm2             # amps, cathodic
    # A POSITIVE offset puts the measured potential above where it
    # belongs, which is what a reference electrode sitting somewhere
    # other than the table says does, and it makes the apparent onset
    # overpotential negative.
    potential = -eta + offset_v + resistance * current
    electrode = Electrode(mass_mg=0.02, area_cm2=area_cm2,
                          electrolyte="H2SO4_0.5M", reference="RHE")
    order = np.argsort(-potential)
    return Voltammogram(
        potential=potential[order], current=current[order],
        scan_rate=scan_rate, electrode=electrode, name="HER sintetico")


def test_the_electrode_fills_its_pH_from_its_own_electrolyte():
    """``Electrode(electrolyte="H2SO4_0.5M")`` used to store the key as
    free text and leave the pH empty, so every conversion to RHE was
    refused — on information the package was holding."""
    from ramancarbon.echem.curve import Electrode

    filled = Electrode(electrolyte="H2SO4_0.5M")
    assert filled.ph == pytest.approx(0.3)
    assert filled.electrolyte == "H2SO4 0.5 M"      # the display label

    # An explicitly given pH always wins.
    assert Electrode(electrolyte="H2SO4_0.5M", ph=1.5).ph == 1.5
    # And free text is left exactly as typed.
    odd = Electrode(electrolyte="algo que nadie tabuló")
    assert odd.ph is None
    assert odd.electrolyte == "algo que nadie tabuló"
    assert Electrode().ph is None


def test_a_negative_onset_overpotential_is_called_impossible():
    """An overpotential is measured FROM the thermodynamic potential, so
    a negative one says the reaction runs before thermodynamics allows
    it. Both of the user's curves report one (−11 and −18 mV), which is
    an offset in the potential axis that every number inherits."""
    from ramancarbon.echem.evaluate import analyse_catalysis

    shifted = analyse_catalysis(_her_curve(offset_v=0.16), reaction="HER")
    assert shifted.onset_overpotential is not None
    assert shifted.onset_overpotential < 0
    assert any("NEGATIVO" in w and "imposible" in w
               for w in shifted.warnings)

    clean = analyse_catalysis(_her_curve(offset_v=0.0), reaction="HER")
    assert not any("NEGATIVO" in w for w in clean.warnings)


def test_a_few_millivolts_of_negative_onset_is_not_an_accusation():
    """The potential axis is read to the millivolt, so a couple of
    millivolts either way is resolution rather than a claim."""
    from ramancarbon.echem.evaluate import (ONSET_SIGN_TOLERANCE,
                                            analyse_catalysis)

    assert ONSET_SIGN_TOLERANCE == pytest.approx(0.005)
    # 126 mV is this curve's own onset, so 128 leaves 2 mV of apparent
    # negative overpotential: resolution, not a claim.
    small = analyse_catalysis(_her_curve(offset_v=0.128), reaction="HER")
    assert not any("NEGATIVO" in w for w in small.warnings)


def test_without_a_resistance_the_report_says_what_would_erase_it():
    """A generic "no iR correction" warning cannot be acted on. The
    resistance that would account for the WHOLE reported overpotential
    can: on the user's curve it is 43 Ω, which an ordinary cell reaches,
    so that overpotential does not yet distinguish a good catalyst from
    a low-resistance cell."""
    from ramancarbon.echem.evaluate import analyse_catalysis

    result = analyse_catalysis(_her_curve(), reaction="HER")
    assert result.overpotential_at_benchmark is not None
    assert any("bastarían" in w and "Ω" in w for w in result.warnings)


def test_a_resistance_that_dominates_the_overpotential_is_named():
    from ramancarbon.echem.curve import Electrode
    from ramancarbon.echem.evaluate import analyse_catalysis

    curve = _her_curve()
    curve.electrode = Electrode(
        mass_mg=0.02, area_cm2=0.0707, electrolyte="H2SO4_0.5M",
        reference="RHE", resistance_ohm=60.0)
    result = analyse_catalysis(curve, reaction="HER")
    assert any("término dominante" in w for w in result.warnings)

    curve.electrode.resistance_ohm = 0.05       # negligible
    quiet = analyse_catalysis(curve, reaction="HER")
    assert not any("término dominante" in w for w in quiet.warnings)


def _her_session(rates=(0.001, 0.005)):
    from ramancarbon.echem.curve import Electrode
    from ramancarbon.gui.echem_state import EchemSession

    session = EchemSession(name="HER")
    session.electrode = Electrode(mass_mg=0.02, area_cm2=0.0707,
                                  electrolyte="H2SO4_0.5M", reference="RHE")
    session.reaction = "HER"
    for index, rate in enumerate(rates):
        curve = _her_curve(scan_rate=rate, offset_v=0.004 * index)
        curve.electrode = session.electrode
        curve.name = f"{1e3 * rate:g} mV/s"
        session.lsv_series.append(curve)
    session.lsv_series.sort(key=lambda c: c.scan_rate)
    session.lsv = session.lsv_series[0]
    session._lsv_analyses = None
    session._lsv_key = ()
    return session


def test_two_scan_rates_are_compared_and_their_spread_reported():
    """A polarisation curve is supposed to BE the steady state, so two
    scan rates have to lie on top of each other. The gap between them is
    a systematic that no single curve reports — on the user's pair it is
    5 mV at 10 mA/cm² and 8 mV at 100."""
    session = _her_session()
    rows = session.lsv_rate_rows()
    assert len(rows) == 3                      # two rates and the spread
    assert rows[-1][0] == "dispersión"
    assert rows[0][0] == "1" and rows[1][0] == "5"
    # The spread column is a number, not a dash.
    assert rows[-1][1] not in ("—", "")


def test_one_curve_alone_gets_no_spread_row():
    session = _her_session(rates=(0.001,))
    rows = session.lsv_rate_rows()
    assert len(rows) == 1
    assert rows[0][0] == "1"


def test_the_catalysis_panels_draw_without_a_full_analysis():
    """They used to read only ``session.result``, so they came back
    blank for anyone who had loaded curves and not pressed analyse —
    which from the figure tab is everybody."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ramancarbon.gui.figures_echem import (CompositeFigure,
                                               available_panels,
                                               draw_composite, every_axes)
    from ramancarbon.gui.theme import PALETTES

    session = _her_session()
    assert session.result is None
    panels = available_panels(session, "catalysis")
    assert {p.key for p in panels} == {"her_medida", "her_polarizacion",
                                       "her_tafel", "her_velocidades"}
    for panel in panels:
        spec = CompositeFigure(panels=[panel.key], columns=1)
        figure = plt.figure(figsize=(5.0, 3.5))
        assert draw_composite(figure, spec, session,
                              PALETTES["claro"]) == [], panel.key
        drawn = sum(len(a.get_lines()) for a in every_axes(figure))
        assert drawn > 0, panel.key
        plt.close(figure)


def test_the_rate_panel_needs_two_curves_to_mean_anything():
    from ramancarbon.gui.figures_echem import available_panels

    one = _her_session(rates=(0.001,))
    assert "her_velocidades" not in {p.key
                                     for p in available_panels(one, "catalysis")}


def test_the_rhe_axis_is_drawn_for_a_session_holding_only_lsv():
    """``_rhe_shift`` looked only at the voltammetry and the
    Mott-Schottky series, so a session holding nothing but polarisation
    curves got no RHE axis — refused in silence on information it was
    holding, the same failure as the electrode that would not fill its
    own pH."""
    from ramancarbon.gui.figures_echem import _rhe_shift

    session = _her_session()
    assert session.cv is None and not session.rate_series
    shift = _rhe_shift(session)
    assert shift is not None
    # reference="RHE" in the helper, so the shift is zero but PRESENT.
    assert shift == pytest.approx(0.0, abs=1e-9)

    from ramancarbon.echem.curve import Electrode
    for curve in session.lsv_series:
        curve.electrode = Electrode(mass_mg=0.02, area_cm2=0.0707,
                                    electrolyte="H2SO4_0.5M",
                                    reference="Ag/AgCl_sat")
    assert _rhe_shift(session) == pytest.approx(0.197 + 0.05916 * 0.3,
                                                abs=2e-3)


def test_the_as_measured_panel_keeps_the_potential_on_its_own_scale():
    """Every other catalysis panel plots an overpotential, which is
    already a conversion. This one is the curve the instrument wrote,
    with the conversion drawn on a second axis rather than applied."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ramancarbon.gui.figures_echem import (CompositeFigure,
                                               draw_composite, every_axes)
    from ramancarbon.gui.theme import PALETTES

    session = _her_session()
    spec = CompositeFigure(panels=["her_medida"], columns=1)
    figure = plt.figure(figsize=(5.0, 3.5))
    assert draw_composite(figure, spec, session, PALETTES["claro"]) == []
    axes = every_axes(figure)
    lines = [line for a in axes for line in a.get_lines()]
    assert len(lines) >= 2
    # The x data are potentials, not current densities: they stay inside
    # the window the sweep covered.
    xs = np.concatenate([line.get_xdata() for line in lines
                         if len(line.get_xdata()) > 5])
    assert xs.min() > -2.0 and xs.max() < 2.0
    plt.close(figure)


def test_a_platinum_like_overpotential_on_a_non_noble_catalyst_is_flagged():
    """The commonest way a non-noble electrode reports platinum's numbers
    is not synthesis: it is platinum dissolved from the COUNTER electrode
    and re-deposited on the working one. Nanograms do it, and no
    electrochemical measurement can see it afterwards, because what is
    being measured really is platinum."""
    from ramancarbon.echem.evaluate import (PLATINUM_SUSPICION_V,
                                            analyse_catalysis)

    # A curve whose benchmark lands in platinum's range.
    noble = analyse_catalysis(_her_curve(exchange=3.0, tafel_mv=32.0),
                              reaction="HER")
    assert noble.overpotential_at_benchmark is not None
    assert 0 < noble.overpotential_at_benchmark < PLATINUM_SUSPICION_V
    assert any("PLATINO" in w and "CONTRAELECTRODO" in w
               for w in noble.warnings)

    # An ordinary non-noble catalyst is not accused.
    # 50 mV/dec from 0.01 mA/cm² puts the benchmark at 150 mV, which is
    # where the good non-noble catalysts of this class report.
    ordinary = analyse_catalysis(_her_curve(exchange=1e-2, tafel_mv=50.0),
                                 reaction="HER")
    assert ordinary.overpotential_at_benchmark > PLATINUM_SUSPICION_V
    assert not any("PLATINO" in w for w in ordinary.warnings)


def test_a_cathodic_current_before_the_reaction_is_named_as_background():
    """Positive of the equilibrium the reaction cannot run, so a steady
    cathodic current there is something ELSE being reduced — in an
    unpurged aqueous cell, dissolved oxygen. It adds to the sweep, so
    every overpotential read from the curve comes out too small."""
    from ramancarbon.echem.evaluate import analyse_catalysis

    # A background of 0.3 mA/cm², which is what the user's cell shows.
    dirty = analyse_catalysis(_her_curve(background_ma_cm2=-0.3),
                              reaction="HER")
    assert dirty.background_ma_cm2 is not None
    assert dirty.background_ma_cm2 == pytest.approx(0.3, rel=0.2)
    assert any("OXÍGENO" in w for w in dirty.warnings)

    clean = analyse_catalysis(_her_curve(), reaction="HER")
    assert clean.background_ma_cm2 is None
    assert not any("OXÍGENO" in w for w in clean.warnings)



# -- stability: a hold is a RATE, not a number ---------------------------


def test_a_hold_comes_back_as_a_drift_rate():
    """An overpotential describes the instant of the measurement. What
    decides whether a catalyst is usable is whether it holds, and the
    field quotes that as a single number after ten hours — which hides
    the shape and invites a one-hour hold to be reported as a ten-hour
    one."""
    from ramancarbon.echem.evaluate import stability

    time = np.linspace(0.0, 3600.0, 400)
    eta = 0.050 + 0.030 * (time / 3600.0)        # 30 mV/h, planted
    result = stability(time, eta, -10.0)
    assert result.drift_v_per_h == pytest.approx(0.030, rel=0.02)
    assert result.duration_h == pytest.approx(1.0, rel=0.01)
    assert result.r_squared > 0.999
    assert result.hours_to_lose_100mv == pytest.approx(100 / 30, rel=0.05)
    # The settling fraction is skipped, so the start is not the first point.
    assert result.overpotential_start > 0.050


def test_a_failing_electrode_is_named_as_failing():
    """The user's hold loses 92 mV/h at 10 mA/cm². It does not finish the
    ten-hour test everybody quotes, and saying that is more useful than
    a retention computed over whatever fraction of it ran."""
    from ramancarbon.echem.evaluate import (STABILITY_FAILING_V_PER_H,
                                            stability)

    time = np.linspace(0.0, 2449.0, 500)
    eta = 0.043 + 0.0925 * (time / 2449.0)
    result = stability(time, eta, -10.0)
    assert result.drift_v_per_h > STABILITY_FAILING_V_PER_H
    assert any("no termina" in w for w in result.warnings)
    assert any("minutos" in w for w in result.warnings)

    steady = stability(np.linspace(0.0, 36000.0, 500),
                       np.full(500, 0.050), -10.0)
    assert abs(steady.drift_v_per_h) < 1e-9
    assert not any("no termina" in w for w in steady.warnings)
    assert not any("minutos" in w for w in steady.warnings)


def test_a_drift_that_is_not_a_line_says_so():
    """A loss rate that changes during the hold is averaged by a single
    slope, and the extrapolation is then worth even less."""
    from ramancarbon.echem.evaluate import stability

    time = np.linspace(0.0, 7200.0, 400)
    eta = 0.050 + 0.08 * (time / 7200.0) ** 3    # accelerating
    result = stability(time, eta, -10.0)
    assert result.r_squared < 0.9
    assert any("no es una recta" in w for w in result.warnings)


def test_stability_refuses_what_it_cannot_measure():
    from ramancarbon.echem.evaluate import stability

    with pytest.raises(ValueError, match="diez puntos"):
        stability([0.0, 1.0], [0.05, 0.06], -10.0)
    with pytest.raises(ValueError, match="no avanza"):
        stability(np.zeros(20), np.linspace(0.05, 0.06, 20), -10.0)
