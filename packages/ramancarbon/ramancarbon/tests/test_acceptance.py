"""The acceptance rules, on fits built to trip each one."""
from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.models.acceptance import (
    DEGENERATE_CORRELATION,
    FWHM_WINDOWS,
    NEGLIGIBLE_AREA_FRACTION,
    Audit,
    aicc,
    audit_fit,
)


class _Peak:
    def __init__(self, name, centre, fwhm, height, area):
        self.name, self.centre, self.fwhm = name, centre, fwhm
        self.height, self.area = height, area


class _Result:
    def __init__(self, peaks, r_squared=0.99, correlations=None,
                 success=True, message="", x=None, n_parameters=8):
        self.peaks = peaks
        self.r_squared = r_squared
        self.correlations = correlations or {}
        self.success = success
        self.message = message
        self.x = np.zeros(2000) if x is None else x
        self.n_parameters = n_parameters


def _sound():
    """A fit with nothing wrong: D and G, plausible widths, separable."""
    return _Result([_Peak("D", 1345.0, 120.0, 300.0, 40000.0),
                    _Peak("G", 1585.0, 55.0, 320.0, 20000.0)])


class TestAicc:
    def test_correction_is_positive_and_shrinks_with_data(self):
        few = aicc(100.0, n_points=50, n_parameters=10)
        many = aicc(100.0, n_points=5000, n_parameters=10)
        assert few > many > 100.0

    def test_a_model_the_data_cannot_support_is_infinite(self):
        """More parameters than points is not a model, and returning the
        uncorrected AIC would hide exactly the case AICc exists for."""
        assert aicc(100.0, n_points=10, n_parameters=10) == float("inf")
        assert aicc(100.0, n_points=11, n_parameters=10) == float("inf")


class TestWidths:
    def test_a_sound_fit_is_high_confidence(self):
        audit = audit_fit(_sound())
        assert audit.confidence == "HIGH"
        assert not audit.findings

    def test_a_d_band_wider_than_the_window_is_flagged(self):
        """The 200 cm-1 D of a real disordered carbon: reported, never
        clamped, because a broad D is a measurement."""
        result = _Result([_Peak("D", 1345.0, 200.0, 300.0, 40000.0),
                          _Peak("G", 1585.0, 55.0, 320.0, 20000.0)])
        audit = audit_fit(result)
        assert any(f.code == "fwhm-ancha" for f in audit.findings)
        assert audit.confidence == "MODERATE"

    def test_a_component_narrower_than_the_window_is_flagged(self):
        result = _Result([_Peak("D", 1345.0, 5.0, 300.0, 40000.0),
                          _Peak("G", 1585.0, 55.0, 320.0, 20000.0)])
        assert any(f.code == "fwhm-estrecha" for f in audit_fit(result).findings)

    @pytest.mark.parametrize("band", sorted(FWHM_WINDOWS))
    def test_every_window_is_ordered_and_positive(self, band):
        low, high = FWHM_WINDOWS[band]
        assert 0.0 < low < high


class TestAreaAndDegeneracy:
    def test_a_component_with_no_area_is_grave(self):
        """Table 11's "area cercana a cero": the component is not doing
        anything a different baseline would not also do."""
        tiny = NEGLIGIBLE_AREA_FRACTION * 40000.0 * 0.5
        result = _Result([_Peak("D", 1345.0, 120.0, 300.0, 40000.0),
                          _Peak("G", 1585.0, 55.0, 320.0, 20000.0),
                          _Peak("D4", 1150.0, 200.0, 2.0, tiny)])
        audit = audit_fit(result)
        assert any(f.code == "area-despreciable" and f.component == "D4"
                   for f in audit.findings)
        assert audit.confidence == "LOW"

    def test_inseparable_parameters_make_the_answer_ambiguous(self):
        """Not LOW: the fit is fine, there is simply more than one model
        the data cannot choose between. That is what AMBIGUOUS means."""
        result = _Result(
            _sound().peaks,
            correlations={("G.centre", "D'.height"): -0.99},
        )
        audit = audit_fit(result)
        assert audit.confidence == "AMBIGUOUS"
        assert any(f.code == "degenerado" for f in audit.findings)

    def test_correlation_below_the_threshold_is_left_alone(self):
        result = _Result(_sound().peaks,
                         correlations={("G.centre", "D.height"):
                                       DEGENERATE_CORRELATION - 0.05})
        assert audit_fit(result).confidence == "HIGH"


class TestBoundsAndConvergence:
    def test_a_parameter_against_its_bound_is_grave(self):
        """The optimiser wanted to go further; the model is what is
        wrong, not the bound."""
        result = _Result([_Peak("D", 1345.0, 200.0, 300.0, 40000.0),
                          _Peak("G", 1585.0, 55.0, 320.0, 20000.0)])
        audit = audit_fit(result, bounds={"D.fwhm": (20.0, 200.0)})
        pinned = [f for f in audit.findings if f.code == "pegado-al-limite"]
        assert pinned and pinned[0].component == "D"
        assert audit.confidence == "LOW"

    def test_a_parameter_inside_its_bounds_is_not_pinned(self):
        audit = audit_fit(_sound(), bounds={"D.fwhm": (20.0, 300.0),
                                            "G.fwhm": (10.0, 200.0)})
        assert not [f for f in audit.findings if f.code == "pegado-al-limite"]

    def test_a_fit_that_did_not_converge_is_low(self):
        result = _Result(_sound().peaks, success=False, message="max iter")
        audit = audit_fit(result)
        assert audit.confidence == "LOW"
        assert any(f.code == "no-converge" for f in audit.findings)

    def test_no_components_is_low_and_says_so(self):
        audit = audit_fit(_Result([]))
        assert audit.confidence == "LOW"
        assert audit.findings[0].code == "sin-componentes"


class TestRSquaredIsNotAnArgument:
    def test_a_near_perfect_r2_with_many_bands_is_noted(self):
        """Adding components always raises R-squared. The note exists so
        a reader is not handed 0.9995 as if it settled anything."""
        peaks = [_Peak(n, c, 80.0, 100.0, 10000.0) for n, c in
                 (("D4", 1180.0), ("D", 1345.0), ("D3", 1500.0),
                  ("G", 1585.0), ("D'", 1615.0))]
        audit = audit_fit(_Result(peaks, r_squared=0.9995))
        assert any(f.code == "r2-solo" for f in audit.findings)

    def test_a_poor_r2_with_no_findings_is_only_moderate(self):
        audit = audit_fit(_Result(_sound().peaks, r_squared=0.90))
        assert audit.confidence == "MODERATE"


def test_audit_renders_for_a_report():
    text = str(audit_fit(_sound()))
    assert "HIGH" in text and "sin observaciones" in text
    assert isinstance(audit_fit(_sound()), Audit)


class TestTheFitCarriesItsOwnBounds:
    """The pinned-parameter check was blind, and it was the one that mattered.

    A real 532 nm spectrum of carbon on FeSe came back from the two- and
    three-band models with the D band's FWHM at exactly 200.0 cm⁻¹ — its
    ceiling. The auditor could say the width was unusual but not that it
    was against its limit, which is a different and much stronger
    statement: a parameter at its bound means the optimiser wanted to go
    further, so the model is wrong, not the bound. It could not say it
    because the caller had to supply the bounds and nobody did. They are
    known where the fit runs, so that is where they are recorded.
    """

    def _spectrum(self):
        import numpy as np

        from ramancarbon.core.spectrum import Spectrum

        shift = np.linspace(1100.0, 1750.0, 400)
        signal = (60.0
                  + 300.0 * np.exp(-0.5 * ((shift - 1350.0) / 110.0) ** 2)
                  + 260.0 / (1.0 + ((shift - 1590.0) / 35.0) ** 2))
        return Spectrum(shift=shift, intensity=signal, laser_nm=532.0,
                        name="pegado")

    def _model(self, ceiling: float):
        from ramancarbon.models.fitting import FitModel, PeakSpec

        return FitModel(
            peaks=[
                PeakSpec(name="D", centre=1350.0, height=300.0, fwhm=100.0,
                         fwhm_bounds=(20.0, ceiling)),
                PeakSpec(name="G", centre=1590.0, height=260.0, fwhm=40.0,
                         fwhm_bounds=(10.0, 120.0)),
            ],
            window=(1100.0, 1750.0),
        )

    def test_the_bounds_are_on_the_result(self):
        from ramancarbon.models.fitting import fit_model

        result = fit_model(self._spectrum(), self._model(200.0))
        assert result.bounds["D.fwhm"] == (20.0, 200.0)
        assert result.bounds["G.fwhm"] == (10.0, 120.0)

    def test_a_parameter_at_its_ceiling_is_reported_without_being_asked(self):
        from ramancarbon.models.acceptance import audit_fit
        from ramancarbon.models.fitting import fit_model

        # A ceiling well below the width the data want: the fit has to end
        # against it.
        result = fit_model(self._spectrum(), self._model(60.0))
        audit = audit_fit(result)
        pinned = [f for f in audit.findings if f.code == "pegado-al-limite"]
        assert any(f.component == "D" for f in pinned), str(audit)
        assert audit.confidence != "HIGH"

    def test_a_comfortable_bound_is_not_reported(self):
        from ramancarbon.models.acceptance import audit_fit
        from ramancarbon.models.fitting import fit_model

        result = fit_model(self._spectrum(), self._model(400.0))
        audit = audit_fit(result)
        assert not [f for f in audit.findings
                    if f.code == "pegado-al-limite" and f.component == "D"]

    def test_an_explicit_empty_dict_still_skips_the_check(self):
        """Because a caller may have a reason to, and silence is a choice."""
        from ramancarbon.models.acceptance import audit_fit
        from ramancarbon.models.fitting import fit_model

        result = fit_model(self._spectrum(), self._model(60.0))
        audit = audit_fit(result, bounds={})
        assert not [f for f in audit.findings if f.code == "pegado-al-limite"]

    def test_fixed_parameters_are_not_in_the_bounds(self):
        """They were never free, so ending at a limit means nothing."""
        from ramancarbon.models.fitting import FitModel, PeakSpec, fit_model

        model = FitModel(
            peaks=[
                PeakSpec(name="D", centre=1350.0, height=300.0, fwhm=100.0,
                         fwhm_bounds=(20.0, 200.0), fixed=("centre",)),
                PeakSpec(name="G", centre=1590.0, height=260.0, fwhm=40.0),
            ],
            window=(1100.0, 1750.0),
        )
        result = fit_model(self._spectrum(), model)
        assert "D.centre" not in result.bounds
        assert "D.fwhm" in result.bounds
