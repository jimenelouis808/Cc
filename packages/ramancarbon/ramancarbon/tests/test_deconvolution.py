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
