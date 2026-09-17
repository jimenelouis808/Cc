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
    from ramancarbon.echem.cv import integrate_charge

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
    for tau, resistance, capacitance, share in rows:
        assert float(tau) > 0 and float(resistance) > 0
        assert float(capacitance) == pytest.approx(
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
