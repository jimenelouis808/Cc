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
