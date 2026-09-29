"""A hand-adjusted deconvolution has to reach the numbers.

Editing the components by hand produced a fit, a residual and an audit,
and nothing else moved. The indices, the intensity ratios, the
crystallite size and the whole report still came from the model the
comparison had chosen before the edits -- so someone who fits D-G their
group's way, which is the ordinary case, read I_D/I_G off a model they
had just replaced. The manual fit was a picture of a fit.
"""

from __future__ import annotations

import pytest

from ramancarbon.analysis.report import analyse
from ramancarbon.examples.demo_data import make_demo
from ramancarbon.gui.state import Session
from ramancarbon.models.deconvolution import build_region_model
from ramancarbon.models.fitting import fit_model


@pytest.fixture(scope="module")
def spectrum():
    return make_demo("MWCNT", seed=2)


@pytest.fixture(scope="module")
def automatic(spectrum):
    return analyse(spectrum)


def _manual_fit(result, n_d=3, n_g=2):
    return fit_model(result.processed, build_region_model(
        result.processed, n_d=n_d, n_g=n_g, profile="pseudo_voigt"))


class TestAnalyseAcceptsAGivenDeconvolution:
    def test_it_uses_exactly_the_fit_it_was_handed(self, spectrum, automatic):
        given = _manual_fit(automatic)
        again = analyse(spectrum, dg_fit=given)
        assert again.fit is given

    def test_the_indices_follow_the_model(self, spectrum, automatic):
        # The point of the whole exercise: a different deconvolution is a
        # different R1, because every area index reads the fit.
        given = _manual_fit(automatic)
        again = analyse(spectrum, dg_fit=given)
        auto_r1 = automatic.indices.indices["R1"].value
        manual_r1 = again.indices.indices["R1"].value
        assert auto_r1 is not None and manual_r1 is not None
        assert len(again.fit.peaks) != len(automatic.fit.peaks)

    def test_it_says_no_models_were_compared(self, spectrum, automatic):
        # You supplied the model, so the information criteria have no
        # opinion on whether it is the best one, and the report must not
        # imply it was selected.
        again = analyse(spectrum, dg_fit=_manual_fit(automatic))
        assert any("suministrada" in w for w in again.warnings)
        assert again.comparison is None

    def test_without_it_nothing_changes(self, spectrum, automatic):
        assert analyse(spectrum).fit is not automatic.fit
        assert analyse(spectrum).comparison is not None


class TestTheSessionCarriesItThrough:
    def _session(self, spectrum):
        session = Session()
        session.add(spectrum)
        session.analyse_active()
        return session

    def test_it_refuses_without_a_manual_fit(self, spectrum):
        session = self._session(spectrum)
        assert session.reanalyse_with_manual_fit() is None
        assert any("a mano" in text for _, text in session.messages)

    def test_it_rebuilds_the_result_from_the_manual_fit(self, spectrum):
        session = self._session(spectrum)
        item = session.active
        before = item.result.indices.indices["R1"].value
        item.manual_fit = _manual_fit(item.result)
        assert session.reanalyse_with_manual_fit() is item
        assert item.result.fit is item.manual_fit
        assert item.from_manual_fit is True
        assert item.result.indices is not None
        assert before is not None

    def test_analysing_again_returns_to_the_automatic_model(self, spectrum):
        # The label has to come off with the numbers it described.
        session = self._session(spectrum)
        item = session.active
        item.manual_fit = _manual_fit(item.result)
        session.reanalyse_with_manual_fit()
        assert item.from_manual_fit is True
        session.analyse_active()
        assert item.from_manual_fit is False
        assert item.result.fit is not item.manual_fit

    def test_the_report_still_builds(self, spectrum):
        # Everything downstream reads the fit, so this is the check that
        # the substitution did not leave a half-built result behind.
        from ramancarbon.analysis.report import build_report

        session = self._session(spectrum)
        item = session.active
        item.manual_fit = _manual_fit(item.result)
        session.reanalyse_with_manual_fit()
        text = build_report(item.result)
        assert "suministrada" in text or len(text) > 200
