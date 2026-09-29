"""A number typed into the component table has to beat the published bound.

The bug this protects against had no traceback and no wrong number: the
user edited a centre in the deconvolution table, pressed «Ajustar», and
the fit came back identical. The components arrive carrying the bounds
the preset gave them, the typed value fell outside one, and the fitter
clipped it back before the first iteration -- so the application looked
like the button did nothing.

The rule is the one the photoemission section already applies: the user's
number wins, the bound moves to admit it, and the component stops being
evidence of the band whose window it just left.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.models.deconvolution import (
    EDITED_BOUND_MARGIN,
    admit_edited_values,
)
from ramancarbon.models.fitting import PeakSpec


def _spec(**kwargs) -> PeakSpec:
    base = {
        "name": "D3",
        "profile": "gaussian",
        "centre": 1500.0,
        "height": 100.0,
        "fwhm": 120.0,
        "centre_bounds": (1450.0, 1560.0),
        "fwhm_bounds": (60.0, 250.0),
        "band": "D3",
    }
    base.update(kwargs)
    return PeakSpec(**base)


def test_a_value_inside_its_bounds_changes_nothing() -> None:
    spec = _spec()
    assert admit_edited_values([spec]) == []
    assert spec.centre_bounds == (1450.0, 1560.0)
    assert spec.fwhm_bounds == (60.0, 250.0)


def test_a_centre_typed_above_the_window_widens_the_window() -> None:
    spec = _spec(centre=1595.0)
    notes = admit_edited_values([spec])
    assert len(notes) == 1
    low, high = spec.centre_bounds
    assert low == 1450.0                       # the far side is untouched
    assert high == 1595.0 + EDITED_BOUND_MARGIN
    assert low < spec.centre < high            # and not pinned against it


def test_a_centre_typed_below_the_window_widens_the_other_side() -> None:
    spec = _spec(centre=1400.0)
    admit_edited_values([spec])
    low, high = spec.centre_bounds
    assert high == 1560.0
    assert low == 1400.0 - EDITED_BOUND_MARGIN


def test_a_width_typed_above_its_ceiling_is_admitted_too() -> None:
    spec = _spec(fwhm=300.0)
    admit_edited_values([spec])
    assert spec.fwhm_bounds[1] == 300.0 + EDITED_BOUND_MARGIN


def test_a_width_bound_never_goes_to_zero_or_below() -> None:
    """The margin must not put a width bound at a non-positive number."""
    spec = _spec(fwhm=1.0, fwhm_bounds=(60.0, 250.0))
    admit_edited_values([spec])
    assert spec.fwhm_bounds[0] > 0.0
    assert spec.fwhm_bounds[0] <= 1.0


def test_the_note_says_the_component_is_no_longer_evidence() -> None:
    """Widening a bound in silence would be the same bug facing the other way."""
    spec = _spec(centre=1595.0)
    note = admit_edited_values([spec])[0]
    assert "D3" in note
    assert "1595" in note
    assert "ya no es prueba" in note


def test_both_parameters_report_separately() -> None:
    spec = _spec(centre=1595.0, fwhm=300.0)
    notes = admit_edited_values([spec])
    assert len(notes) == 2
    assert any("centro" in n for n in notes)
    assert any("FWHM" in n for n in notes)


def test_a_component_without_bounds_is_left_alone() -> None:
    """A hand-added component has no published window to argue with."""
    spec = PeakSpec(name="Gx1", centre=1500.0, height=10.0, fwhm=30.0)
    assert admit_edited_values([spec]) == []


@pytest.mark.parametrize("attribute", ["centre", "fwhm"])
def test_the_widened_bound_actually_reaches_the_fit(attribute: str) -> None:
    """The regression test proper: the fit has to be able to go there.

    Without the widening the optimiser starts clipped and comes back with
    the value the bound allows, which is what «no pasa nada» looked like.
    """
    from ramancarbon.core.spectrum import Spectrum
    from ramancarbon.models.fitting import FitModel, fit_model

    x = np.linspace(1400.0, 1700.0, 600)
    y = 100.0 * np.exp(-0.5 * ((x - 1595.0) / 40.0) ** 2)
    spectrum = Spectrum(shift=x, intensity=y, laser_nm=532.0)

    spec = _spec(centre=1595.0, fwhm=94.0)     # 94 = 40 cm-1 sigma, as FWHM
    if attribute == "fwhm":
        spec.centre, spec.centre_bounds = 1595.0, (1450.0, 1700.0)
    admit_edited_values([spec])
    result = fit_model(
        spectrum,
        FitModel(peaks=[spec], window=(1400.0, 1700.0), background="linear"),
    )
    assert result.peaks[0].centre == pytest.approx(1595.0, abs=3.0)
