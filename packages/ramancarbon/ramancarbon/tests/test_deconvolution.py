"""Deconvolution models: the bands must keep their identities."""
from __future__ import annotations

import pytest


class TestBandsDoNotTradePlaces:
    """The database's G window (1550-1610) and D' window (1595-1640)
    share fifteen wavenumbers, and nothing used to force an order.

    On a measured disordered carbon the three-band fit took the
    invitation: "G" came out at 1554 and "D'" at 1597 -- D3 territory and
    G territory. It converged, and it had the best AICc of the four
    models tried, so a model-comparison step would have recommended it.
    Every ratio computed from that fit is mislabelled.
    """

    def _carbon(self):
        import numpy as np

        from ramancarbon.core.spectrum import Spectrum

        shift = np.linspace(1000.0, 1800.0, 900)
        signal = (60.0
                  + 340.0 * np.exp(-0.5 * ((shift - 1345.0) / 75.0) ** 2)
                  + 300.0 * np.exp(-0.5 * ((shift - 1585.0) / 35.0) ** 2)
                  + 90.0 * np.exp(-0.5 * ((shift - 1615.0) / 20.0) ** 2))
        return Spectrum(shift=shift, intensity=signal)

    def test_g_and_d_prime_windows_come_out_disjoint(self):
        from ramancarbon.models.deconvolution import MIN_CENTRE_GAP, build_model

        model = build_model(self._carbon(), preset="three_band")
        specs = {spec.name: spec for spec in model.peaks}
        assert specs["G"].centre_bounds[1] <= specs["D'"].centre_bounds[0]
        gap = specs["D'"].centre_bounds[0] - specs["G"].centre_bounds[1]
        assert gap == pytest.approx(MIN_CENTRE_GAP, abs=1e-6)

    def test_each_band_keeps_the_rest_of_its_own_window(self):
        """Only the ambiguous overlap is split. A G at 1580 or a D' at
        1620 must still be reachable."""
        from ramancarbon.models.deconvolution import build_model

        specs = {s.name: s for s in
                 build_model(self._carbon(), preset="three_band").peaks}
        g_lo, g_hi = specs["G"].centre_bounds
        d_lo, d_hi = specs["D'"].centre_bounds
        assert g_lo <= 1580.0 <= g_hi
        assert d_lo <= 1620.0 <= d_hi

    def test_the_fitted_order_is_the_order_the_names_mean(self):
        from ramancarbon.models.deconvolution import build_model
        from ramancarbon.models.fitting import fit_model

        spectrum = self._carbon()
        result = fit_model(spectrum, build_model(spectrum, preset="three_band"))
        centres = {p.name: p.centre for p in result.peaks}
        assert centres["D"] < centres["G"] < centres["D'"]

    def test_the_starting_centres_stay_inside_the_new_bounds(self):
        """A seed outside its own bounds is rejected by the fitter, so
        splitting the window has to move the seed with it."""
        from ramancarbon.models.deconvolution import build_model

        for spec in build_model(self._carbon(), preset="three_band").peaks:
            low, high = spec.centre_bounds
            assert low <= spec.centre <= high


class TestSadezkyWithoutDPrime:
    """A fifth component the data do not support is not a better model.

    On a real 532 nm spectrum of carbon on FeSe the Sadezky five-band fit
    put D' at 1616 cm⁻¹ with a width of 11 cm⁻¹ — below D''s own window —
    and an area 0.5 % of the largest, which the overfitting table calls
    "area near zero". The auditor said so and there was nothing to do
    about it: every preset with D4 in it also had D'. Dropping D' was
    better on AICc and on BIC at once, with four parameters fewer.
    """

    def _soot(self):
        """Disordered carbon: G wide enough that D' is inside it."""
        import numpy as np

        from ramancarbon.core.spectrum import Spectrum

        shift = np.linspace(880.0, 1820.0, 660)
        signal = (
            70.0
            + 120.0 * np.exp(-0.5 * ((shift - 1250.0) / 150.0) ** 2)   # D4
            + 300.0 / (1.0 + ((shift - 1350.0) / 70.0) ** 2)           # D
            + 110.0 * np.exp(-0.5 * ((shift - 1520.0) / 95.0) ** 2)    # D3
            + 250.0 / (1.0 + ((shift - 1592.0) / 35.0) ** 2)           # G
        )
        rng = np.random.default_rng(20260927)
        signal = signal + rng.normal(0.0, 3.0, signal.shape)
        return Spectrum(shift=shift, intensity=signal, laser_nm=532.0,
                        name="hollin")

    def test_the_preset_exists_and_has_no_d_prime(self):
        from ramancarbon.models.deconvolution import PRESET_BANDS, PRESETS

        assert "five_band_no_dprime" in PRESETS
        bands = PRESET_BANDS["five_band_no_dprime"]
        assert bands == ("D4", "D", "D3", "G")
        assert "D'" not in bands

    def test_it_builds_and_fits(self):
        from ramancarbon.models.deconvolution import build_model
        from ramancarbon.models.fitting import fit_model

        spectrum = self._soot()
        result = fit_model(spectrum, build_model(
            spectrum, preset="five_band_no_dprime", profile="pseudo_voigt"))
        assert result.success
        assert {p.name for p in result.peaks} == {"D4", "D", "D3", "G"}

    def _fits(self):
        from ramancarbon.models.deconvolution import build_model
        from ramancarbon.models.fitting import fit_model

        spectrum = self._soot()
        window = (900.0, 1800.0)
        return {
            preset: fit_model(spectrum, build_model(
                spectrum, preset=preset, window=window,
                profile="pseudo_voigt"))
            for preset in ("five_band", "five_band_no_dprime")
        }

    def test_it_beats_the_five_band_model_on_bic(self):
        """BIC, which is the criterion the comparison defaults to.

        Not AIC. AIC's penalty is 2 per parameter however long the
        spectrum, so on six hundred points it is cheap enough that a D'
        fitting noise can pay for itself — which is the module's own
        argument for defaulting to BIC, whose penalty grows as ln(n). On
        the real 532 nm spectrum the no-D' model won both; asserting both
        here would be asserting a property of this synthetic trace.
        """
        fits = self._fits()
        assert fits["five_band_no_dprime"].bic < fits["five_band"].bic

    def test_the_d_prime_it_drops_is_the_one_the_auditor_rejects(self):
        """The reason the preset exists, stated as a test.

        If D' in the five-band fit were a real component this preset would
        be throwing away physics. It is not: its area is a rounding error
        beside the others, which is what the overfitting rules call an
        area near zero.
        """
        from ramancarbon.models.acceptance import audit_fit

        fits = self._fits()
        audit = audit_fit(fits["five_band"])
        negligible = [f for f in audit.findings
                      if f.code == "area-despreciable" and f.component == "D'"]
        assert negligible, str(audit)
        clean = audit_fit(fits["five_band_no_dprime"])
        assert not [f for f in clean.findings if f.code == "area-despreciable"]

    def test_it_is_among_the_models_compared_by_default(self):
        from ramancarbon.models.deconvolution import compare_models

        comparison = compare_models(self._soot(), profile="pseudo_voigt")
        assert "five_band_no_dprime" in comparison.results
