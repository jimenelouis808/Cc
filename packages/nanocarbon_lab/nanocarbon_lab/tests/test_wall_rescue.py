"""The rescue that rebuilds a collapsed wall with the wall held.

A collapsed wall is not a poor structure, it is an impossible one: an
angle sum under 328.4 deg is past tetrahedral, which no carbon reaches.
The logic is tested here against synthetic walls rather than real
builds, because a real one costs minutes and the thing worth pinning is
the decision, not the physics it decides on.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from ase import Atoms

from nanocarbon_lab.analyse.hybridisation import TETRAHEDRAL_SUM
from nanocarbon_lab.builders.junction import (
    RESCUE_ANCHORS,
    RESCUE_MARGIN,
    rescue_collapsed_wall,
)


def _wall(angle_sum: float) -> Atoms:
    """A three-bonded carbon whose angle sum is exactly `angle_sum`.

    Three bonds at equal angles `theta` about a cone: the angle between
    any two is fixed by the cone's half-angle, so one parameter sets the
    sum and the geometry stays real rather than mocked.
    """
    target = np.radians(angle_sum / 3.0)
    # For three unit vectors at polar angle p, pairwise angle a obeys
    # cos a = cos^2 p + sin^2 p * cos(120 deg).
    polar = None
    for candidate in np.linspace(0.0, np.pi / 2, 20001):
        cosine = (np.cos(candidate) ** 2
                  + np.sin(candidate) ** 2 * np.cos(np.radians(120.0)))
        if np.arccos(np.clip(cosine, -1, 1)) >= target:
            polar = candidate
            break
    assert polar is not None
    points = [[0.0, 0.0, 0.0]]
    for k in range(3):
        phi = 2 * np.pi * k / 3
        points.append([1.42 * np.sin(polar) * np.cos(phi),
                       1.42 * np.sin(polar) * np.sin(phi),
                       1.42 * np.cos(polar)])
    atoms = Atoms("C4", positions=points)
    atoms.info["bonds"] = [(0, 1), (0, 2), (0, 3)]
    return atoms


class TestTheSyntheticWall:
    """If the fixture does not hit its angle sum, nothing below means
    anything."""

    @pytest.mark.parametrize("wanted", [360.0, 340.0, 328.4, 320.0])
    def test_it_hits_the_angle_sum_asked_for(self, wanted):
        from nanocarbon_lab.analyse.hybridisation import hybridisation_report

        got = hybridisation_report(_wall(wanted).positions,
                                   _wall(wanted).info["bonds"])
        assert got["angle_sum_min"] == pytest.approx(wanted, abs=0.6)


class TestRescue:
    def test_a_sound_wall_is_returned_untouched(self):
        """No rebuild, no cost, no marker. The anchor widens bonds, so
        running it on a sound wall would be a loss."""
        sound = _wall(340.0)
        calls = []
        out = rescue_collapsed_wall(sound, lambda k: calls.append(k))
        assert out is sound
        assert not calls
        assert "wall_rescued_from" not in out.info

    def test_a_collapsed_wall_is_rebuilt_and_recorded(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rescue_collapsed_wall(_wall(320.0), lambda k: _wall(335.0))
        assert out.info["wall_rescued_from"] == pytest.approx(320.0, abs=0.6)
        assert out.info["wall_rescue_step"] == f"wall_anchor={RESCUE_ANCHORS[0]:g}"

    def test_it_escalates_until_the_wall_clears_with_margin(self):
        """super-cubic clears at 1.0; the superfullerene gets WORSE at
        1.0 and needs 2.0 to clear, 4.0 to clear comfortably."""
        seen = []

        def rebuild(strength):
            seen.append(strength)
            return _wall({1.0: 318.0, 2.0: 328.5, 4.0: 336.0}[strength])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rescue_collapsed_wall(_wall(320.0), rebuild)
        # 328.5 clears `collapsed_wall` by a tenth of a degree and must
        # NOT stop the search; only the margin does.
        assert seen == [1.0, 2.0, 4.0]
        assert out.info["wall_rescue_step"] == "wall_anchor=4"

    def test_it_stops_as_soon_as_the_margin_is_met(self):
        seen = []

        def rebuild(strength):
            seen.append(strength)
            return _wall(TETRAHEDRAL_SUM + RESCUE_MARGIN + 2.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rescue_collapsed_wall(_wall(320.0), rebuild)
        assert seen == [RESCUE_ANCHORS[0]]

    def test_a_rebuild_that_raises_does_not_lose_the_original(self):
        original = _wall(320.0)

        def rebuild(strength):
            raise RuntimeError("mesh did not weld")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rescue_collapsed_wall(original, rebuild)
        assert out is original

    def test_it_keeps_the_best_even_when_none_clears(self):
        """A wall that is still collapsed but less so is worth keeping;
        handing back the worst one would be perverse."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rescue_collapsed_wall(
                _wall(315.0),
                lambda k: _wall({1.0: 320.0, 2.0: 325.0, 4.0: 322.0}[k]))
        from nanocarbon_lab.analyse.hybridisation import hybridisation_report

        got = hybridisation_report(out.positions, out.info["bonds"])
        assert got["angle_sum_min"] == pytest.approx(325.0, abs=0.6)

    def test_it_says_so(self):
        with pytest.warns(UserWarning, match="collapsed"):
            rescue_collapsed_wall(_wall(320.0), lambda k: _wall(336.0))


class TestTheLadderCarriesMoreThanAnchors:
    """`None` in the ladder means "try a finer grid", because
    super-diamond's collapse is a resolution problem no anchor fixes."""

    def test_a_none_step_is_named_not_floated(self):
        seen = []

        def rebuild(step):
            seen.append(step)
            return _wall(336.0 if step is None else 320.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rescue_collapsed_wall(_wall(318.0), rebuild,
                                        anchors=(1.0, None))
        assert seen == [1.0, None]
        assert out.info["wall_rescue_step"] == "finer grid"

    def test_a_slow_build_is_left_alone_with_a_named_remedy(self):
        """A super-fcc cell takes 24 minutes to build once; three more
        rebuilds is an hour and a half of a window that looks hung.

        The budget is TIME, not atoms: super-fcc is 3082 atoms at 1440 s
        and a superfullerene is 7534 at 79, so an atom limit that spares
        one refuses the other for no reason."""
        slow = _wall(320.0)
        calls = []
        with pytest.warns(UserWarning, match="left to you"):
            out = rescue_collapsed_wall(slow, lambda k: calls.append(k),
                                        time_budget=60.0,
                                        seconds_spent=1440.0)
        assert out is slow
        assert not calls

    def test_a_quick_build_is_rescued_however_many_atoms(self):
        """The superfullerene: 7534 atoms and 79 s, well inside budget."""
        many = _wall(320.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rescue_collapsed_wall(many, lambda k: _wall(336.0),
                                        time_budget=900.0,
                                        seconds_spent=79.0)
        assert out.info["wall_rescued_from"] == pytest.approx(320.0, abs=0.6)
