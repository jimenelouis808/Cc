"""Changing the background model changes the answer, and that is the test.

A weight fraction comes from a scale factor, and a scale factor is fixed
by whatever intensity the background did not take. When a phase's
reflections are as broad as the background's own flexibility, which is
the ordinary case for anything nanocrystalline, the split between them is
a modelling choice wearing the clothes of a measurement.

No single refinement can show this: it converges either way. Two do.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.xrd.rietveld import (
    SENSITIVITY_ORDERS,
    BackgroundSensitivity,
    background_order_sensitivity,
)

#: The real numbers, from the user's CVD pattern of carbon grown on FeSe.
MEASURED = {
    2: (0.05342, 1.2729, {"C": 0.259, "Fe": 0.029, "Fe3C": 0.527}),
    4: (0.04902, 1.1681, {"C": 0.366, "Fe": 0.093, "Fe3C": 0.323}),
    6: (0.04387, 1.0455, {"C": 0.524, "Fe": 0.208, "Fe3C": 0.163}),
    8: (0.04053, 0.9661, {"C": 0.331, "Fe": 0.195, "Fe3C": 0.377}),
    10: (0.04001, 0.9538, {"C": 0.331, "Fe": 0.197, "Fe3C": 0.365}),
}


def _measured() -> BackgroundSensitivity:
    orders = tuple(sorted(MEASURED))
    return BackgroundSensitivity(
        orders,
        {o: MEASURED[o][0] for o in orders},
        {o: MEASURED[o][1] for o in orders},
        {o: dict(MEASURED[o][2]) for o in orders},
    )


def test_the_spread_is_the_size_of_the_choice() -> None:
    spread = _measured().spread()
    assert spread["C"] == pytest.approx(0.524 - 0.259, abs=1e-9)
    assert spread["Fe3C"] == pytest.approx(0.527 - 0.163, abs=1e-9)


def test_the_best_background_is_not_the_default() -> None:
    """Sixth order is the package default and it had the worst Rwp but one.

    That is the whole point of the scan: at the default the carbon read
    52.4 %, and at the two orders that fit better it read 33.1 % twice.
    """
    scan = _measured()
    assert scan.best_order() == 10
    assert scan.fractions[6]["C"] > scan.fractions[10]["C"]


def test_the_summary_names_the_phase_that_is_not_determined() -> None:
    text = _measured().summary()
    assert "NO está determinada por la medida" in text
    assert "C " in text and "Fe3C" in text


def test_a_phase_that_barely_moves_is_called_stable() -> None:
    orders = (4, 6, 8)
    scan = BackgroundSensitivity(
        orders,
        {o: 0.04 for o in orders},
        {o: 1.0 for o in orders},
        {4: {"Fe": 0.50}, 6: {"Fe": 0.52}, 8: {"Fe": 0.51}},
    )
    assert "estable" in scan.summary()
    assert "NO está determinada" not in scan.summary()


def test_a_phase_with_no_computable_fraction_does_not_crash_the_spread() -> None:
    orders = (4, 6)
    scan = BackgroundSensitivity(
        orders, {o: 0.04 for o in orders}, {o: 1.0 for o in orders},
        {4: {"X": None}, 6: {"X": None}},
    )
    assert scan.spread() == {"X": 0.0}
    assert "n/d" in scan.summary()


def test_the_default_orders_straddle_the_package_default() -> None:
    from ramancarbon.xrd.rietveld import DEFAULT_BACKGROUND_ORDER

    assert min(SENSITIVITY_ORDERS) < DEFAULT_BACKGROUND_ORDER
    assert max(SENSITIVITY_ORDERS) > DEFAULT_BACKGROUND_ORDER


def test_the_scan_runs_and_each_order_gets_its_own_answer() -> None:
    """End to end on a synthetic pattern, two orders to keep it short."""
    from ramancarbon.xrd.pattern import Pattern
    from ramancarbon.xrd.reference import load_library

    crystal = next(iter(load_library())).crystal
    angles = np.linspace(20.0, 60.0, 1200)
    counts = 500.0 + 40.0 * np.sin(angles / 9.0)
    pattern = Pattern(two_theta=angles, intensity=counts, name="sintético")
    scan = background_order_sensitivity(pattern, [crystal], orders=(2, 4))
    assert scan.orders == (2, 4)
    assert set(scan.r_wp) == {2, 4}
    assert set(scan.gof) == {2, 4}
    assert scan.best_order() in (2, 4)
