"""A width that reaches its own ceiling is not a width.

`PINNED_TOLERANCE` is deliberately literal, and on widths that hid the
finding. On the user's real 532 nm spectrum of carbon grown on FeSe the D
band came back at 200.0 cm-1 of a 200 ceiling in the three-band model,
195.7 in the four-band, 191.0 in the five-band and 193.0 in Sadezky's
model without D' -- the same wall four times, reported once, because the
other three stop a few wavenumbers short of it.

Those three are not measurements of a width either, and the difference
between 191 and 200 here is where the optimiser gave up, not physics.
Both halves of the condition matter: the width has to be above the band's
usual range AND up against the bound the fit actually used.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from ramancarbon.models.acceptance import NO_ROOM_FRACTION, audit_fit


@dataclass
class _Peak:
    name: str
    centre: float
    fwhm: float
    area: float = 1000.0
    height: float = 100.0


@dataclass
class _Result:
    peaks: list
    bounds: dict = field(default_factory=dict)
    correlations: dict = field(default_factory=dict)
    r_squared: float = 0.99
    success: bool = True
    message: str = ""


def _audit(fwhm: float, ceiling: float | None = 200.0, name: str = "D"):
    bounds = {} if ceiling is None else {f"{name}.fwhm": (15.0, ceiling)}
    result = _Result(peaks=[_Peak(name=name, centre=1350.0, fwhm=fwhm)], bounds=bounds)
    return audit_fit(result)


def _codes(audit) -> set[str]:
    return {f.code for f in audit.findings}


def test_a_width_exactly_on_its_ceiling_is_grave() -> None:
    assert "anchura-sin-sitio" in _codes(_audit(200.0))


@pytest.mark.parametrize("width", [199.0, 195.7, 193.0, 191.0])
def test_the_widths_the_real_spectrum_produced_are_all_caught(width: float) -> None:
    """All four presets hit the same wall; all four have to say so."""
    assert "anchura-sin-sitio" in _codes(_audit(width))


def test_a_width_above_its_usual_range_but_far_from_the_bound_is_only_a_warning() -> None:
    """The broad D of a turbostratic carbon is a measurement, not an error."""
    codes = _codes(_audit(185.0, ceiling=400.0))
    assert "anchura-sin-sitio" not in codes
    assert "fwhm-ancha" in codes


def test_a_normal_width_says_nothing_about_widths() -> None:
    codes = _codes(_audit(120.0))
    assert "anchura-sin-sitio" not in codes
    assert "fwhm-ancha" not in codes


def test_a_width_at_its_ceiling_but_inside_the_usual_range_is_not_reported() -> None:
    """Both halves of the condition are required.

    A band whose published bound happens to sit inside its own plausible
    range is a database question, not a finding about this fit.
    """
    assert "anchura-sin-sitio" not in _codes(_audit(120.0, ceiling=120.0))


def test_without_recorded_bounds_the_check_cannot_fire() -> None:
    """It asks what constrained THIS fit, not what the literature says."""
    codes = _codes(_audit(195.7, ceiling=None))
    assert "anchura-sin-sitio" not in codes
    assert "fwhm-ancha" in codes


def test_the_threshold_is_the_documented_fraction() -> None:
    ceiling = 200.0
    just_inside = ceiling * (1.0 - NO_ROOM_FRACTION) + 0.1
    just_outside = ceiling * (1.0 - NO_ROOM_FRACTION) - 0.1
    assert "anchura-sin-sitio" in _codes(_audit(just_inside))
    assert "anchura-sin-sitio" not in _codes(_audit(just_outside))


def test_it_takes_the_fit_out_of_high_confidence() -> None:
    assert _audit(195.7).confidence == "LOW"


def test_the_message_names_the_number_and_the_ceiling() -> None:
    finding = next(f for f in _audit(195.7).findings if f.code == "anchura-sin-sitio")
    assert "195.7" in finding.message
    assert "200" in finding.message
