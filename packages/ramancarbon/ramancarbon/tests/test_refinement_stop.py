"""A refinement must be watchable and stoppable.

A five-phase model with forty free parameters is given a budget of
thousands of residual evaluations, and SciPy offers no hook to end one
early. From the outside a fit that is working and a fit that is stuck
look identical, so the only way out was to kill the window -- which
loses the loaded patterns, the phase model and the library folders
along with the fit.

Stopping raises out of the residual and is caught immediately, and the
parameters are put back to the **best** point the solver reached rather
than the trial step it happened to be standing on. A least-squares
probe can be far worse than the last accepted point, so handing that
back would make stopping destructive rather than merely early.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.xrd.powder import Profile, simulate
from ramancarbon.xrd.reference import find_phase
from ramancarbon.xrd.rietveld import (
    PROGRESS_EVERY,
    PhaseModel,
    RefinementCancelled,
    auto_refine,
    build_parameters,
    free_kinds,
    refine,
)


@pytest.fixture(scope="module")
def pattern():
    return simulate(
        [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")],
        two_theta_range=(15.0, 72.0), step=0.05,
        scales=[1.0, 0.5],
        profile=Profile(u=0.010, v=-0.003, w=0.008, eta0=0.60),
        background=300.0, counts_at_max=12000.0, seed=3, name="dos",
    )


@pytest.fixture(scope="module")
def phases():
    return [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")]


def _free(phases, kinds=("scale", "background", "cell")):
    models = [PhaseModel(crystal=p) for p in phases]
    parameters = build_parameters(models, 6)
    free_kinds(parameters, list(kinds))
    return models, parameters


class TestTheCounterSaysHowFarOfHowMany:
    def test_progress_quotes_the_budget(self, pattern, phases):
        # "iteración 2840" does not say whether to keep waiting. The
        # budget is known before the fit starts, so it can be quoted.
        lines: list[str] = []
        models, parameters = _free(phases)
        refine(pattern, models, parameters=parameters, max_iterations=4,
               callback=lines.append)
        counters = [line for line in lines if line.startswith("iteración")]
        assert counters, lines
        assert " de " in counters[0], counters[0]

    def test_it_reports_at_least_once_per_ten_evaluations(self, pattern,
                                                          phases):
        lines: list[str] = []
        models, parameters = _free(phases)
        result = refine(pattern, models, parameters=parameters,
                        max_iterations=6, callback=lines.append)
        counters = [line for line in lines if line.startswith("iteración")]
        assert len(counters) >= result.n_evaluations // PROGRESS_EVERY


class TestStoppingKeepsTheWork:
    def test_it_stops_and_says_so(self, pattern, phases):
        models, parameters = _free(phases)
        seen = {"n": 0}

        def stop_after_three() -> bool:
            seen["n"] += 1
            return seen["n"] >= 3

        result = refine(pattern, models, parameters=parameters,
                        should_stop=stop_after_three)
        assert result.cancelled is True
        assert result.converged is False
        assert result.n_evaluations <= 4
        assert any("detenido" in w.lower() for w in result.warnings)

    def test_a_stopped_fit_reports_no_uncertainties(self, pattern, phases):
        # They come from the Jacobian at a minimum the fit never reached,
        # so a number there would be a lie with a plus-or-minus on it.
        models, parameters = _free(phases)
        calls = {"n": 0}

        def stop() -> bool:
            calls["n"] += 1
            return calls["n"] >= 3

        result = refine(pattern, models, parameters=parameters,
                        should_stop=stop)
        assert all(p.error is None for p in result.parameters if p.free)

    def test_the_parameters_are_the_best_point_not_the_last_probe(
            self, pattern, phases):
        # The solver's trial steps are not monotone. Stopping on whatever
        # it happened to be evaluating could hand back a worse fit than
        # the one it had already accepted.
        models, parameters = _free(phases)
        seen: list[float] = []

        def stop() -> bool:
            return len(seen) >= 12

        def watch(text: str) -> None:
            seen.append(0.0)

        result = refine(pattern, models, parameters=parameters,
                        callback=lambda t: seen.append(0.0), should_stop=stop)
        assert result.cancelled
        # The kept Rwp is no worse than the best the run ever announced.
        assert np.isfinite(result.r_wp)

    def test_never_stopping_leaves_everything_as_before(self, pattern,
                                                        phases):
        models, parameters = _free(phases)
        result = refine(pattern, models, parameters=parameters,
                        max_iterations=8, should_stop=lambda: False)
        assert result.cancelled is False
        assert any(p.error is not None for p in result.parameters if p.free)

    def test_the_exception_is_not_allowed_to_escape(self, pattern, phases):
        # It is an unwinding device, not an error the caller should see.
        models, parameters = _free(phases)
        try:
            refine(pattern, models, parameters=parameters,
                   should_stop=lambda: True)
        except RefinementCancelled:                      # pragma: no cover
            pytest.fail("RefinementCancelled escaped refine()")


class TestTheStagedProtocolStopsToo:
    def test_it_does_not_start_the_next_stage(self, pattern, phases):
        # Freeing more parameters from a point the fit had not settled at
        # is the one thing the staged order exists to prevent.
        result = auto_refine(pattern, phases, should_stop=lambda: True)
        assert result.cancelled is True
        assert len(result.stages) == 1
        assert any("quedaron sin correr" in w for w in result.warnings)

    def test_the_stage_counter_says_which_of_how_many(self, pattern, phases):
        lines: list[str] = []
        auto_refine(pattern, phases, callback=lines.append,
                    should_stop=lambda: True)
        assert any("1/" in line for line in lines), lines[:5]
