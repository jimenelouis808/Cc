"""Tests for the anemone: a tree of tubes as one closed surface.

The skeleton tests are fast and carry most of the logic. One build is
shared by a class-scoped fixture, kept small (three short arms) so the
whole file stays in the fast suite.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nanocarbon_lab.builders.anemone import (
    MIN_TUBE_RADIUS,
    anemone_graph,
    build_anemone,
)


class TestTheSkeleton:
    def test_it_is_a_tree_whatever_the_arm_and_branch_count(self):
        """V - E = 1, which is what fixes the ring budget at +12."""
        for arms, branches in ((3, 0), (8, 0), (8, 4), (12, 6)):
            graph = anemone_graph(arms=arms, branches=branches, seed=1)
            assert graph["n_vertices"] - graph["n_edges"] == 1, (arms,
                                                                 branches)

    def test_the_arms_have_unequal_lengths_inside_the_range(self):
        graph = anemone_graph(arms=8, arm_length=(20.0, 40.0), seed=0)
        lengths = np.linalg.norm(graph["bulbs"], axis=1)
        assert lengths.min() >= 20.0 and lengths.max() <= 40.0
        assert np.ptp(lengths) > 5.0

    def test_the_arms_are_spread_and_not_flat(self):
        """Fibonacci directions: no two arms closer than a wide angle."""
        graph = anemone_graph(arms=8, seed=0)
        unit = graph["bulbs"] / np.linalg.norm(graph["bulbs"], axis=1)[:, None]
        cosines = unit @ unit.T - np.eye(8)
        assert np.degrees(np.arccos(cosines.max())) > 35.0
        # Not a flower: the directions span all three axes.
        assert np.linalg.matrix_rank(unit, tol=0.3) == 3

    def test_it_is_reproducible_from_the_seed(self):
        first = anemone_graph(seed=5)
        again = anemone_graph(seed=5)
        other = anemone_graph(seed=6)
        assert np.array_equal(first["segments"], again["segments"])
        assert not np.array_equal(first["bulbs"], other["bulbs"])

    def test_a_branch_that_does_not_fit_is_dropped_not_forced(self):
        """Many branches on short crowded arms cannot all be seated."""
        graph = anemone_graph(arms=10, arm_length=(20.0, 22.0),
                              branches=40, seed=0)
        assert graph["n_branches"] < 40
        # And the count that came back is the count in the skeleton.
        assert graph["n_edges"] == 10 + 2 * graph["n_branches"]


class TestTheRefusals:
    def test_a_tube_under_the_meshing_floor(self):
        with pytest.raises(ValueError, match="floor"):
            build_anemone(tube_radius=MIN_TUBE_RADIUS - 0.5)

    def test_a_neck_too_wide_for_its_bulb(self):
        with pytest.raises(ValueError, match="lozenge"):
            build_anemone(tube_radius=5.0, bulb_radius=6.0)

    def test_an_arm_too_short_to_leave_a_waist(self):
        with pytest.raises(ValueError, match="waist"):
            build_anemone(arm_length=(10.0, 30.0))

    def test_fewer_than_two_arms(self):
        with pytest.raises(ValueError, match="two arms"):
            build_anemone(arms=1)


@pytest.fixture(scope="module")
def anemone():
    """One small build, shared: three short arms, no branches."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_anemone(arms=3, arm_length=(22.0, 26.0),
                             branches=0, bulb_radius=6.0, seed=0)


class TestASmallBuild:

    def test_the_census_meets_the_trees_budget(self, anemone):
        census = anemone.info["ring_counts"]
        assert sum((6 - size) * count for size, count in census.items()) == 12
        assert anemone.info["ring_budget"] == 12

    def test_it_is_sound_carbon(self, anemone):
        geometry = anemone.info["geometry"]
        assert set(anemone.get_chemical_symbols()) == {"C"}
        assert 1.25 < geometry["bond_min"] and geometry["bond_max"] < 1.65
        assert geometry["n_close_contacts"] == 0
        assert not anemone.pbc.any()

    def test_every_bulb_is_found_where_its_atoms_are(self, anemone):
        """The centres are measured, and a wrong one shows as no room.

        A bounding-box shift put three of eight centres on their own
        stems, which read as a free radius under a few angstrom. A bulb
        of radius 6 leaves several, so this is the regression guard.
        """
        centres = np.asarray(anemone.info["bulb_centres"])
        radii = anemone.info["bulb_radii"]
        assert len(centres) == 3 and len(radii) == 3
        assert min(radii) > 3.0, radii
        for centre in centres:
            distance = np.linalg.norm(anemone.positions - centre, axis=1)
            shell = distance[distance < 1.15 * 6.0]
            assert len(shell) > 60
            assert 4.5 < shell.mean() < 7.5
