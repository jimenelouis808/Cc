"""States have to earn their place, and the fit has to survive a checklist.

The default region model used to be every state the database lists. On the
user's own PHI Quantera file that produced, for C 1s alone, fourteen
components and eleven pairs of parameters with |r| ≥ 0.95 — six of them at
exactly 1.000 — with binding energies printed to ±278 822, ±5 613 136 and
±203 002 971 eV. Those are not uncertainties, they are the arithmetic
saying the parameter is not determined, and they were being reported as
chemical composition.

These tests hold the two halves of the fix: the selection that decides
which states go in, and the audit that says whether the result can be
believed. The numbers in the docstrings are measured on that file.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.xps.spectrum import XPSSpectrum
from ramancarbon.xps.tables import load_xps_database


@pytest.fixture(scope="module")
def database():
    return load_xps_database()


def region(peaks, *, low=280.0, high=296.0, noise=8.0, background=200.0,
           step=0.05, pass_energy=20.0, seed=3, dwell=0.5, sweeps=4):
    """A synthetic region: Gaussians on a flat background, Poisson noise.

    ``pass_energy`` is low by default so the instrument resolution floor
    (1.5 % of it, folded with the X-ray linewidth) sits below the widths
    used here and does not fire on every component.
    """
    axis = np.arange(low, high + step / 2, step)
    signal = np.full_like(axis, background)
    for centre, fwhm, height in peaks:
        sigma = fwhm / 2.3548
        signal = signal + height * np.exp(-0.5 * ((axis - centre) / sigma) ** 2)
    rng = np.random.default_rng(seed)
    counts = rng.poisson(np.clip(signal, 0.0, None)).astype(float)
    if noise:
        counts = counts + rng.normal(0.0, noise, counts.shape)
    return XPSSpectrum(
        binding_energy=axis, counts=np.clip(counts, 0.0, None),
        photon_energy=1486.6, pass_energy=pass_energy, dwell_s=dwell,
        sweeps=sweeps, region="C 1s", name="sintético",
    )


class TestFewerComponentsThanStates:
    def test_the_default_no_longer_fits_every_state(self, database):
        """Two real peaks, fourteen catalogued states, and the model has to
        come out closer to two than to fourteen."""
        from ramancarbon.xps.selection import evidence_model

        spectrum = region([(284.5, 1.2, 4000.0), (285.3, 1.4, 1600.0)])
        catalogued = len(database.states_for("C 1s"))
        selection = evidence_model(spectrum, "C 1s", database=database)
        assert catalogued >= 14
        assert len(selection.result.components) <= 5, selection.summary()

    def test_nothing_the_data_cannot_separate_gets_through(self, database):
        from ramancarbon.xps.acceptance import DEGENERATE
        from ramancarbon.xps.selection import evidence_model

        spectrum = region([(284.5, 1.2, 4000.0), (285.3, 1.4, 1600.0),
                           (288.6, 1.5, 700.0)])
        result = evidence_model(spectrum, "C 1s", database=database).result
        worst = [pair for pair, value in result.correlations.items()
                 if abs(value) >= DEGENERATE]
        assert not worst, f"pares no separables: {worst}"

    def test_a_candidate_that_only_improves_the_residual_is_refused(self, database):
        """The checklist asks it in one line and this is the line.

        With the gate at zero every candidate that absorbs a little noise
        gets in; at six it has to pay for its parameters.
        """
        from ramancarbon.xps.selection import evidence_model

        spectrum = region([(284.5, 1.2, 4000.0)])
        strict = evidence_model(spectrum, "C 1s", database=database)
        loose = evidence_model(spectrum, "C 1s", database=database, bic_gain=0.0)
        assert len(loose.result.components) >= len(strict.result.components)
        refused = [s for s in strict.steps
                   if not s.accepted and "no paga sus parámetros" in s.reason]
        assert refused, strict.summary()

    def test_every_rejection_says_why(self, database):
        from ramancarbon.xps.selection import evidence_model

        selection = evidence_model(region([(284.5, 1.2, 4000.0)]), "C 1s",
                                   database=database)
        for step in selection.steps:
            if not step.accepted:
                assert step.reason, f"{step.state} rechazado sin motivo"


class TestPresenceIsNotAbsence:
    """Measuring a region proves an element is there. Nothing proves it is
    not, except a survey.

    Getting this wrong is not academic. Deriving the composition from the
    four narrow scans in the user's file dropped the metal-oxide O 1s
    component — from a sample whose O 1s peaks at 530.05 eV, which is
    exactly where that component sits. What was left could not reach the
    peak: two components, both pinned against their limits, reduced
    chi-squared 34.4 against 1.39 for the model that keeps it.
    """

    def _oxygen(self):
        return region([(529.9, 1.6, 3000.0), (531.5, 1.6, 1200.0),
                       (533.0, 1.7, 600.0)],
                      low=524.0, high=540.0, seed=5)

    def test_an_incomplete_list_drops_nothing(self, database):
        from ramancarbon.xps.selection import evidence_model

        spectrum = self._oxygen()
        selection = evidence_model(spectrum, "O 1s", present=["C", "O"],
                                   complete=False, database=database)
        assert selection.missing_element == []
        assert not selection.elements_known

    def test_a_survey_may_drop_a_state(self, database):
        from ramancarbon.xps.selection import evidence_model

        selection = evidence_model(self._oxygen(), "O 1s",
                                   present=["C", "O"], complete=True,
                                   database=database)
        assert selection.missing_element, selection.summary()
        assert selection.elements_known

    def test_the_metal_oxide_component_survives_without_a_survey(self, database):
        """The regression, stated as a test."""
        from ramancarbon.xps.selection import evidence_model

        spectrum = self._oxygen()
        with_filter = evidence_model(spectrum, "O 1s", present=["C", "N", "O", "S"],
                                     complete=True, database=database)
        without = evidence_model(spectrum, "O 1s", present=["C", "N", "O", "S"],
                                 complete=False, database=database)
        assert without.result.reduced_chi2 <= with_filter.result.reduced_chi2

    def test_the_summary_says_the_filter_did_not_run(self, database):
        from ramancarbon.xps.selection import evidence_model

        text = evidence_model(self._oxygen(), "O 1s", database=database).summary()
        assert "survey" in text


class TestGrowThenPrune:
    def test_a_component_that_falls_below_the_noise_later_is_removed(self, database):
        """Judging each arrival is not the same as judging what comes out.

        On the real N 1s the oxidised-nitrogen component was worth 53
        points of BIC when it arrived and, once a later component had been
        accepted too, its height had dropped under three times the noise.
        """
        from ramancarbon.xps.acceptance import MIN_SIGNAL_SIGMA
        from ramancarbon.xps.selection import evidence_model

        spectrum = region([(398.3, 1.4, 2500.0), (400.9, 1.5, 2000.0),
                           (399.6, 1.5, 900.0)],
                          low=392.0, high=410.0, seed=7)
        spectrum = XPSSpectrum(
            binding_energy=spectrum.binding_energy, counts=spectrum.counts,
            photon_energy=1486.6, pass_energy=20.0, dwell_s=0.5, sweeps=4,
            region="N 1s", name="sintético",
        )
        selection = evidence_model(spectrum, "N 1s", database=database)
        noise = spectrum.noise_estimate()
        for component in selection.result.components:
            if component.satellite:
                continue
            assert component.peak_height >= MIN_SIGNAL_SIGMA * noise, (
                f"{component.name} quedó por debajo del ruido\n"
                + selection.summary())

    def test_a_pruned_step_records_that_it_was_pruned(self, database):
        from ramancarbon.xps.selection import evidence_model

        spectrum = region([(284.5, 1.2, 4000.0), (285.3, 1.4, 1500.0)])
        selection = evidence_model(spectrum, "C 1s", database=database)
        pruned = [s for s in selection.steps
                  if not s.accepted and "al final del ajuste" in s.reason]
        # Not every spectrum needs a prune; when one happens it must say so.
        for step in pruned:
            assert np.isfinite(step.delta_bic)


class TestTheChecklist:
    def test_a_parameter_on_its_limit_is_grave(self, database):
        """Stronger than "unusual": the optimiser wanted to go further, so
        the model is wrong and not the limit."""
        from ramancarbon.xps.acceptance import audit_region
        from ramancarbon.xps.fitting import fit_region
        from ramancarbon.xps.presets import state_model

        spectrum = region([(284.5, 1.2, 4000.0)])
        model = state_model(spectrum, "C 1s", ["C-C sp2"], database=database)
        # A ceiling well under the width the data want.
        for component in model.components:
            component.fwhm_bounds = (0.20, 0.35)
            component.fwhm = 0.25
        audit = audit_region(fit_region(spectrum, model, database=database),
                             spectrum=spectrum, database=database)
        pinned = [f for f in audit.findings if f.code == "pegado-al-limite"]
        assert pinned, str(audit)
        assert audit.confidence != "ALTA"

    def test_a_component_under_the_noise_is_grave(self, database):
        from ramancarbon.xps.acceptance import audit_region
        from ramancarbon.xps.fitting import fit_region
        from ramancarbon.xps.presets import state_model

        # A region with one real peak, fitted with a second state that has
        # nothing to sit on.
        spectrum = region([(284.5, 1.2, 4000.0)], noise=30.0)
        model = state_model(spectrum, "C 1s", ["C-C sp2", "carbonate"],
                            database=database)
        audit = audit_region(fit_region(spectrum, model, database=database),
                             spectrum=spectrum, database=database)
        codes = {f.code for f in audit.findings}
        assert {"bajo-el-ruido", "area-despreciable"} & codes, str(audit)

    def test_an_unverified_element_is_a_hypothesis_not_a_result(self, database):
        from ramancarbon.xps.acceptance import audit_region
        from ramancarbon.xps.fitting import fit_region
        from ramancarbon.xps.presets import state_model

        spectrum = region([(285.8, 1.3, 3000.0)])
        model = state_model(spectrum, "C 1s", ["C-C sp2", "C-S"],
                            database=database)
        audit = audit_region(fit_region(spectrum, model, database=database),
                             spectrum=spectrum, present=["C"], complete=False,
                             database=database)
        assert any(f.code == "sin-confirmar" for f in audit.findings), str(audit)

    def test_a_survey_turns_that_into_a_contradiction(self, database):
        from ramancarbon.xps.acceptance import audit_region
        from ramancarbon.xps.fitting import fit_region
        from ramancarbon.xps.presets import state_model

        spectrum = region([(285.8, 1.3, 3000.0)])
        model = state_model(spectrum, "C 1s", ["C-C sp2", "C-S"],
                            database=database)
        audit = audit_region(fit_region(spectrum, model, database=database),
                             spectrum=spectrum, present=["C", "O"],
                             complete=True, database=database)
        assert any(f.code == "falta-el-elemento" for f in audit.findings), str(audit)

    def test_a_confirmed_element_is_not_mentioned(self, database):
        from ramancarbon.xps.acceptance import audit_region
        from ramancarbon.xps.fitting import fit_region
        from ramancarbon.xps.presets import state_model

        spectrum = region([(285.8, 1.3, 3000.0)])
        model = state_model(spectrum, "C 1s", ["C-C sp2", "C-S"],
                            database=database)
        audit = audit_region(fit_region(spectrum, model, database=database),
                             spectrum=spectrum, present=["C", "S"],
                             complete=True, database=database)
        codes = {f.code for f in audit.findings}
        assert "sin-confirmar" not in codes and "falta-el-elemento" not in codes

    def test_ambiguous_is_its_own_verdict_not_a_low_score(self, database):
        """Because it means something different: not "poorly measured" but
        "these data are equally compatible with another answer"."""
        from ramancarbon.xps.acceptance import Audit, Finding, _verdict
        from ramancarbon.xps.fitting import fit_region
        from ramancarbon.xps.presets import state_model

        spectrum = region([(284.5, 1.2, 4000.0)])
        result = fit_region(spectrum, state_model(spectrum, "C 1s",
                                                 ["C-C sp2"],
                                                 database=database),
                            database=database)
        findings = [Finding("no-separables", "grave", None, "x")]
        confidence, reason = _verdict(findings, result)
        assert confidence == "AMBIGUA"
        assert "no lo resuelve" in reason or "distintas" in reason
        assert Audit(findings, confidence, reason).grave


class TestAMissingWordIsNotAFact:
    """A VAMAS header that does not say "mono" has not said the source is
    unmonochromated.

    A PHI Quantera SXM is a monochromated instrument by construction and
    its header carries "Al 1486.6". Reading that as a bare anode put the
    resolution floor at 1.19 eV instead of 0.88 and made the acceptance
    check accuse three ordinary components of being narrower than physics
    allows. A floor used to REJECT has to be the most permissive one
    consistent with what is known.
    """

    def test_silence_means_monochromated(self):
        from ramancarbon.xps.io import _DECLARED_UNMONOCHROMATED

        assert _DECLARED_UNMONOCHROMATED("Al 1486.6") is True
        assert _DECLARED_UNMONOCHROMATED("") is True

    def test_an_explicit_statement_is_believed(self):
        from ramancarbon.xps.io import _DECLARED_UNMONOCHROMATED

        assert _DECLARED_UNMONOCHROMATED("Al non-mono") is False
        assert _DECLARED_UNMONOCHROMATED("Mg twin anode") is False

    def test_the_floor_follows(self):
        from ramancarbon.xps.fitting import resolution_floor

        mono = XPSSpectrum(np.linspace(280.0, 296.0, 200),
                           np.ones(200), photon_energy=1486.6,
                           pass_energy=55.0, monochromated=True)
        bare = XPSSpectrum(np.linspace(280.0, 296.0, 200),
                           np.ones(200), photon_energy=1486.6,
                           pass_energy=55.0, monochromated=False)
        assert resolution_floor(mono) < resolution_floor(bare)


class TestThroughTheReport:
    def test_the_analysis_carries_a_selection_and_an_audit(self):
        from ramancarbon.examples.demo_data import make_xps_demo
        from ramancarbon.xps.report import analyse_xps, build_report

        analysis = analyse_xps(make_xps_demo("NCNT_FeSe", seed=11))
        assert analysis.regions
        assert len(analysis.audits) == len(analysis.regions)
        assert len(analysis.selections) == len(analysis.regions)
        text = build_report(analysis)
        assert "Confianza del ajuste" in text

    def test_an_explicit_list_of_states_is_still_obeyed(self):
        """The argument exists so the decision can be the user's. Overruling
        it would make it useless."""
        from ramancarbon.examples.demo_data import make_xps_demo
        from ramancarbon.xps.report import analyse_xps

        analysis = analyse_xps(make_xps_demo("NCNT_FeSe", seed=11),
                               regions={"C 1s": ["C-C sp2", "C-C sp3"]})
        carbon = analysis.region("C 1s")
        assert carbon is not None
        states = {c.state for c in carbon.components if not c.satellite}
        assert states == {"C-C sp2", "C-C sp3"}
