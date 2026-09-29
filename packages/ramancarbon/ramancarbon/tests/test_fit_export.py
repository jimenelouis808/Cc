"""A refinement has to leave the program as numbers, not only as a picture.

The window could save the plot and save the report. Neither lets the fit
be redrawn: on a journal's axes, beside another sample, at a different
scale, or with the difference curve divided by sigma the way most
Rietveld figures show it. Those need the columns.
"""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.xrd.powder import Profile, simulate
from ramancarbon.xrd.reference import find_phase
from ramancarbon.xrd.rietveld import (
    FIT_COLUMNS,
    auto_refine,
    fit_columns,
    reflection_ticks,
    write_fit,
)


@pytest.fixture(scope="module")
def refined():
    phases = [find_phase("FeSe_tetragonal"), find_phase("grafito_2H")]
    pattern = simulate(
        phases, two_theta_range=(15.0, 72.0), step=0.1, scales=[1.0, 0.5],
        profile=Profile(u=0.010, v=-0.003, w=0.008, eta0=0.60),
        background=300.0, counts_at_max=12000.0, seed=3, name="dos",
    )
    return auto_refine(pattern, phases)


class TestTheColumnsAreTheFigure:
    def test_it_gives_the_six_a_rietveld_plot_needs(self, refined):
        names, table = fit_columns(refined)
        assert names == list(FIT_COLUMNS)
        assert table.shape == (refined.pattern.n, 6)

    def test_the_columns_are_the_arrays_the_window_plots(self, refined):
        # Not a recalculation, which could drift from what was shown.
        _, table = fit_columns(refined)
        assert np.allclose(table[:, 0], refined.pattern.two_theta)
        assert np.allclose(table[:, 1], refined.pattern.intensity)
        assert np.allclose(table[:, 2], refined.calculated)
        assert np.allclose(table[:, 3], refined.background)

    def test_the_difference_is_observed_minus_calculated(self, refined):
        _, table = fit_columns(refined)
        assert np.allclose(table[:, 4], table[:, 1] - table[:, 2])

    def test_sigma_comes_along_because_the_weighting_is_the_result(
            self, refined):
        # The usual difference plot is (obs - calc) / sigma, and without
        # this column that plot cannot be made from the file.
        _, table = fit_columns(refined)
        assert np.all(table[:, 5] > 0)


class TestTheTicksComeToo:
    def test_one_row_of_ticks_per_phase(self, refined):
        ticks = reflection_ticks(refined)
        assert set(ticks) == {p.current_crystal().name for p in refined.phases}
        assert all(len(angles) for angles in ticks.values())

    def test_the_zero_shift_is_already_applied(self, refined):
        # They must land where the ticks sit on the MEASURED axis, not
        # where an ideal cell would put them.
        from ramancarbon.xrd.powder import reflections

        phase = refined.phases[0]
        raw = reflections(phase.current_crystal(),
                          wavelength=refined.pattern.wavelength,
                          two_theta_range=refined.pattern.range)
        shifted = reflection_ticks(refined)[phase.current_crystal().name]
        assert np.allclose(shifted,
                           [r.two_theta + refined.zero for r in raw])

    def test_every_tick_is_inside_the_measured_range(self, refined):
        low, high = refined.pattern.range
        for angles in reflection_ticks(refined).values():
            assert np.all(angles >= low - 1.0)
            assert np.all(angles <= high + 1.0)


class TestWhatIsWrittenCanBeReadBack:
    def test_two_files_because_they_are_two_shapes(self, refined, tmp_path):
        # One row per measured point against one row per reflection.
        # Forcing both into one table means padding, and a column of
        # blanks is read differently by every plotting program.
        written = write_fit(refined, tmp_path / "ajuste.txt")
        assert len(written) == 2
        assert all(path.exists() for path in written)
        assert written[1].name == "ajuste_reflexiones.txt"

    def test_numpy_reads_the_curves_straight_back(self, refined, tmp_path):
        curves, _ = write_fit(refined, tmp_path / "a.txt")
        back = np.loadtxt(curves)
        _, table = fit_columns(refined)
        assert back.shape == table.shape
        assert np.allclose(back, table, atol=1e-5)

    def test_the_header_carries_what_the_numbers_mean(self, refined,
                                                      tmp_path):
        curves, _ = write_fit(refined, tmp_path / "a.txt")
        head = curves.read_text(encoding="utf-8")
        assert "lambda" in head
        assert "Rwp" in head and "GOF" in head
        assert "FeSe_tetragonal" in head

    def test_a_stopped_refinement_says_so_in_the_file(self, tmp_path):
        # A file that outlives the session must carry the caveat with it.
        phases = [find_phase("grafito_2H")]
        pattern = simulate(
            phases, two_theta_range=(20.0, 60.0), step=0.1, scales=[1.0],
            profile=Profile(u=0.01, v=-0.003, w=0.008, eta0=0.6),
            background=300.0, counts_at_max=9000.0, seed=1, name="uno")
        stopped = auto_refine(pattern, phases, should_stop=lambda: True)
        assert stopped.cancelled
        curves, _ = write_fit(stopped, tmp_path / "s.txt")
        assert "DETENIDO" in curves.read_text(encoding="utf-8")
