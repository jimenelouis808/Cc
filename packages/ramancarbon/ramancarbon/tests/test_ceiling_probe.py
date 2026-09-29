"""Lifting a bound is the only way to know whether it was being pushed.

A component that is genuinely as wide as its ceiling and a component that
is mopping up intensity belonging to a band the model does not have come
back looking identical: a width close to a bound. The two mean opposite
things -- the first is a measurement, the second says nothing derived
from the fit may be quoted -- so a check that cannot tell them apart has
to report both the same way, and that is what made the first version of
this check call a real 191 cm-1 D band an artefact.

The experiment that separates them: lift the ceiling and fit again.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.core.spectrum import Spectrum
from ramancarbon.models.deconvolution import (
    PROBE_MOVED_FRACTION,
    probe_width_ceilings,
)
from ramancarbon.models.fitting import FitModel, PeakSpec, fit_model


def _gaussian(x, centre, height, fwhm):
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return height * np.exp(-0.5 * ((x - centre) / sigma) ** 2)


def _spectrum(y, x):
    return Spectrum(shift=x, intensity=y, laser_nm=532.0)


def _model(fwhm_ceiling: float, start: float = 60.0) -> FitModel:
    return FitModel(
        peaks=[PeakSpec(name="A", profile="gaussian", centre=1500.0,
                        height=100.0, fwhm=start,
                        centre_bounds=(1400.0, 1600.0),
                        fwhm_bounds=(10.0, fwhm_ceiling), band="A")],
        window=(1200.0, 1800.0), background="linear",
    )


def test_a_band_narrower_than_its_ceiling_is_not_probed_at_all() -> None:
    """Nothing near a bound means nothing to ask about."""
    x = np.linspace(1200.0, 1800.0, 900)
    y = _gaussian(x, 1500.0, 100.0, 60.0)
    model = _model(300.0)
    result = fit_model(_spectrum(y, x), model)
    assert probe_width_ceilings(_spectrum(y, x), model, result) == []


def test_a_band_that_really_is_that_wide_is_cleared() -> None:
    """The true width equals the ceiling: lifting it must not move it."""
    x = np.linspace(1200.0, 1800.0, 900)
    y = _gaussian(x, 1500.0, 100.0, 200.0)
    model = _model(200.0)
    spectrum = _spectrum(y, x)
    result = fit_model(spectrum, model)
    notes = probe_width_ceilings(spectrum, model, result)
    assert len(notes) == 1
    assert "No estaba empujando" in notes[0]


def test_a_band_mopping_up_is_caught() -> None:
    """One component over data that needs two: it will take all it is given."""
    x = np.linspace(1200.0, 1800.0, 900)
    y = _gaussian(x, 1500.0, 100.0, 400.0)      # far wider than the ceiling
    model = _model(150.0)
    spectrum = _spectrum(y, x)
    result = fit_model(spectrum, model)
    notes = probe_width_ceilings(spectrum, model, result)
    assert len(notes) == 1
    assert "EMPUJANDO" in notes[0]


def test_the_probe_does_not_modify_the_model_it_was_given() -> None:
    """It is a diagnostic. The fit on screen has to stay the fit on screen."""
    x = np.linspace(1200.0, 1800.0, 900)
    y = _gaussian(x, 1500.0, 100.0, 400.0)
    model = _model(150.0)
    before = model.peaks[0].fwhm_bounds
    spectrum = _spectrum(y, x)
    probe_width_ceilings(spectrum, model, fit_model(spectrum, model))
    assert model.peaks[0].fwhm_bounds == before


def test_the_message_carries_both_numbers() -> None:
    x = np.linspace(1200.0, 1800.0, 900)
    y = _gaussian(x, 1500.0, 100.0, 400.0)
    model = _model(150.0)
    spectrum = _spectrum(y, x)
    note = probe_width_ceilings(spectrum, model, fit_model(spectrum, model))[0]
    assert "150" in note and "300" in note


def test_the_threshold_is_a_fraction_of_the_ceiling() -> None:
    assert 0.0 < PROBE_MOVED_FRACTION < 1.0


@pytest.mark.parametrize("ceiling,truth,pushing", [
    (200.0, 200.0, False),                       # the D case: sits, not pushes
    (150.0, 400.0, True),                        # the D4 case: a wall
])
def test_the_two_cases_measured_on_the_real_spectrum(
    ceiling: float, truth: float, pushing: bool
) -> None:
    """Both appeared in the SAME five-band fit of the user's spectrum.

    The D came back at 191.0 of 200 and stayed at 185.3 when the ceiling
    went to 400; the D4 came back at 250.0 of 250, 300.0 of 300 and 400.0
    of 400. A distance check sees one thing; the probe sees two.
    """
    x = np.linspace(1200.0, 1800.0, 900)
    y = _gaussian(x, 1500.0, 100.0, truth)
    model = _model(ceiling)
    spectrum = _spectrum(y, x)
    notes = probe_width_ceilings(spectrum, model, fit_model(spectrum, model))
    assert len(notes) == 1
    assert ("EMPUJANDO" in notes[0]) is pushing
