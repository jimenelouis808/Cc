"""Degenerate input, and the fabrications it used to produce.

Every case here was found by feeding the analysis something broken and
watching what came out. They are regression tests for specific failures, and
the docstrings say which.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.analysis.report import analyse
from ramancarbon.core.compat import trapezoid
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.examples.demo_data import make_demo


def _flat(value: float = 0.0) -> Spectrum:
    x = np.arange(100.0, 3200.0)
    return Spectrum(x, np.full_like(x, value), laser_nm=532.0, name="plano")


# -- fabrication from nothing -------------------------------------------
@pytest.mark.parametrize(
    "name, intensity",
    [
        ("ceros", np.zeros(3100)),
        ("constante", np.full(3100, 5.0)),
        ("ruido puro", np.random.default_rng(0).normal(100.0, 3.0, 3100)),
    ],
)
def test_a_spectrum_with_no_bands_names_no_material(name, intensity):
    """A least-squares fit always returns components. On a spectrum of
    literal zeros this used to yield "MWCNT, confianza media, I_D/I_G =
    1.43" — five bands invented out of nothing."""
    x = np.arange(100.0, 3200.0)
    result = analyse(Spectrum(x, intensity, laser_nm=532.0, name=name))
    assert result.classification.best is None
    assert result.classification.confidence == "insuficiente"
    assert result.id_ig is None


def test_refusal_puts_the_reason_in_the_label_not_the_confidence():
    """_verdict's refusal branches returned confidence and label swapped,
    so ``classification.confidence`` held a whole sentence."""
    result = analyse(_flat())
    assert result.classification.confidence == "insuficiente"
    assert "banda G" in result.classification.label


def test_classification_needs_a_g_band():
    """A spectrum covering only the 2D region was classified as graphite."""
    x = np.arange(2400.0, 3200.0)
    source = make_demo("MWCNT", seed=1)
    y = np.interp(x, source.shift, source.intensity)
    result = analyse(Spectrum(x, y, laser_nm=532.0, name="solo2D"))
    assert result.classification.best is None


def test_fitted_components_below_the_noise_are_not_assigned():
    result = analyse(_flat(5.0))
    assert result.assignment.bands == {}


# -- invalid data --------------------------------------------------------
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_intensities_are_rejected(bad):
    """These used to flow through the baseline into the fit and come out as
    a confident-looking classification."""
    x = np.arange(100.0, 3200.0)
    y = np.ones_like(x)
    y[500] = bad
    with pytest.raises(ValueError, match="no son finitas"):
        Spectrum(x, y, laser_nm=532.0)


def test_the_error_says_where_the_bad_point_is():
    x = np.arange(100.0, 3200.0)
    y = np.ones_like(x)
    y[500] = np.nan
    with pytest.raises(ValueError, match="600"):
        Spectrum(x, y)


# -- numpy compatibility -------------------------------------------------
def test_trapezoid_shim_works_on_either_numpy():
    """np.trapezoid is NumPy 2.0 only, but the package declares >=1.24."""
    x = np.linspace(0.0, 10.0, 101)
    assert trapezoid(np.ones_like(x), x) == pytest.approx(10.0)
    assert trapezoid(x, x) == pytest.approx(50.0)


def test_no_module_calls_numpy_trapezoid_directly():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = [
        path.relative_to(root)
        for path in root.rglob("*.py")
        if path.name != "compat.py"
        and "tests" not in path.parts
        and "np.trapezoid" in path.read_text(encoding="utf-8")
    ]
    assert not offenders, f"bypass the shim: {offenders}"


# -- things that should degrade, not crash ------------------------------
@pytest.mark.parametrize("low, high", [(1000.0, 1800.0), (100.0, 500.0), (1500.0, 1700.0)])
def test_truncated_ranges_do_not_crash(low, high):
    x = np.arange(low, high)
    source = make_demo("MWCNT", seed=1)
    y = np.interp(x, source.shift, source.intensity)
    result = analyse(Spectrum(x, y, laser_nm=532.0, name="recortado"))
    assert result.report()


@pytest.mark.parametrize("step", [2.0, 4.0, 8.0])
def test_coarse_sampling_still_runs(step):
    assert analyse(make_demo("MWCNT", seed=1, step=step)).report()


def test_huge_dynamic_range():
    x = np.arange(100.0, 3200.0)
    result = analyse(Spectrum(x, np.exp(x / 200.0), laser_nm=532.0, name="exp"))
    assert result.classification.confidence in {"baja", "media", "insuficiente"}
