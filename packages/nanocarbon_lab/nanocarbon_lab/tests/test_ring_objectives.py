"""What the flip objectives can and cannot do to the ring population.

Written after a report of "isolated 5-7 pairs, 7-7 pairs, lobes of
pentagons" on structures that had been through the full refinement, and
after measuring where those actually come from. The measurements are in
the docstrings because the conclusions are not obvious and two of them
are negative.

The short version, on a Y junction of 538 mesh vertices:

    straight out of isotropic_remesh   46 non-hexagons, 10 unbudgeted
    census anneal                      32 non-hexagons,  8 unbudgeted
    place_disclinations after that      32 non-hexagons,  8 unbudgeted
                                        -- three rounds, misfit 254.5 to
                                        252.7, histogram unchanged

and on a 1200-vertex six-arm junction, five census anneals instead of one
take the count from 88 to 79 and the same-sign adjacencies from 1x5-5 and
3x7-7 to 5x5-5 and 4x7-7 -- the wrong way, and those clumps are what
shows up as a lobe.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import implicit as im
from nanocarbon_lab.builders import remesh as rm

BOND = 1.42


@pytest.fixture(scope="module")
def junction():
    """A Y junction's mesh, remeshed but not refined."""
    field, extent = im.junction_field("Y", tube_radius=6.0, arm_length=22.0,
                                      blend=4.0)
    mesh = rm.marching_cubes_mesh(field, extent=extent, resolution=70)
    mesh = rm.isotropic_remesh(mesh, field,
                               target_edge=BOND * np.sqrt(3.0), iterations=20)
    return mesh, rm.vertex_curvature_targets(mesh)


def census(mesh):
    adjacency = rm._adjacency(mesh[1])
    return sum(1 for ns in adjacency.values() if len(ns) != 6)


class TestTheCurvatureObjectiveExists:
    """It was documented and not implemented.

    ``anneal_edge_flips``'s docstring said ``curvature_misfit`` "is the
    quantity :func:`anneal_edge_flips` minimises under
    ``objective='curvature'``" while the function had no ``objective``
    parameter at all and minimised ``sum |degree - 6|``.
    """

    def test_the_objective_can_be_asked_for(self, junction):
        mesh, targets = junction
        rng = np.random.default_rng(0)
        result = rm.anneal_edge_flips(mesh, rng, sweeps=20, temperature=0.3,
                                      objective="curvature", targets=targets)
        assert census(result) > 0
        assert len(result[0]) == len(mesh[0])          # positions untouched

    def test_an_unknown_objective_is_refused(self, junction):
        mesh, _targets = junction
        with pytest.raises(ValueError, match="census|curvature"):
            rm.anneal_edge_flips(mesh, np.random.default_rng(0), sweeps=1,
                                 objective="lo-que-sea")

    def test_it_places_the_charge_better_than_the_census_does(self, junction):
        """The measured claim, and the only one the curvature objective
        gets to make: 186 against 230 of misfit on the Y junction.

        It does NOT come out with fewer non-hexagons — it comes out with
        more — so this asserts placement and nothing else.
        """
        mesh, targets = junction
        by_census = rm.anneal_edge_flips(
            mesh, np.random.default_rng(0), sweeps=200, temperature=0.5,
            restarts=2)
        by_curvature = rm.anneal_edge_flips(
            mesh, np.random.default_rng(0), sweeps=200, temperature=0.5,
            restarts=2, objective="curvature", targets=targets,
            pair_weight=6.0)
        assert (rm.curvature_misfit(by_curvature, targets)
                < rm.curvature_misfit(by_census, targets))

    def test_the_census_objective_is_untouched_by_default(self, junction):
        """Every existing caller has to keep getting what it got."""
        mesh, _targets = junction
        first = rm.anneal_edge_flips(mesh, np.random.default_rng(7),
                                     sweeps=40, temperature=0.3)
        second = rm.anneal_edge_flips(mesh, np.random.default_rng(7),
                                      sweeps=40, temperature=0.3,
                                      objective="census")
        assert np.array_equal(first[1], second[1])


class TestLikeSignClumps:
    """Two pentagons sharing an edge is a lobe; the objectives cannot see it.

    ``|5-6| + |5-6|`` is 2 whether the two pentagons are adjacent or on
    opposite sides of the structure, so nothing in the census objective
    discourages a clump — and repeated annealing measurably makes them
    worse.
    """

    def _clumped(self):
        """A flat patch of mesh with two adjacent degree-5 vertices."""
        field, extent = im.junction_field("L", tube_radius=6.0,
                                          arm_length=18.0, blend=4.0)
        mesh = rm.marching_cubes_mesh(field, extent=extent, resolution=60)
        return rm.isotropic_remesh(mesh, field,
                                   target_edge=BOND * np.sqrt(3.0),
                                   iterations=15)

    def test_the_metric_counts_like_signs_and_not_dislocations(self):
        """A 5 beside a 7 is a Stone-Wales dislocation: ordinary, neutral,
        and deliberately not charged."""
        mesh = self._clumped()
        adjacency = rm._adjacency(mesh[1])
        degrees = {v: len(ns) for v, ns in adjacency.items()}
        by_hand = 0
        for vertex, neighbours in adjacency.items():
            if degrees[vertex] == 6:
                continue
            high = degrees[vertex] > 6
            for other in neighbours:
                if other <= vertex or degrees[other] == 6:
                    continue
                if (degrees[other] > 6) == high:
                    by_hand += 1
        assert rm.like_sign_adjacencies(mesh) == by_hand

    def test_the_penalty_reduces_them(self):
        mesh = self._clumped()
        plain = mesh
        charged = mesh
        rng_a = np.random.default_rng(1)
        rng_b = np.random.default_rng(1)
        for _ in range(3):
            plain = rm.anneal_edge_flips(plain, rng_a, sweeps=60,
                                         temperature=0.3)
            charged = rm.anneal_edge_flips(charged, rng_b, sweeps=60,
                                           temperature=0.3, like_penalty=3.0)
        assert (rm.like_sign_adjacencies(charged)
                <= rm.like_sign_adjacencies(plain))

    def test_it_is_off_by_default(self):
        mesh = self._clumped()
        first = rm.anneal_edge_flips(mesh, np.random.default_rng(4), sweeps=30)
        second = rm.anneal_edge_flips(mesh, np.random.default_rng(4), sweeps=30,
                                      like_penalty=0.0)
        assert np.array_equal(first[1], second[1])


class TestTheDiagnostics:
    """The question a person actually asks is "why is there a pentagon in
    the middle of my straight tube", and one misfit number does not answer
    it."""

    def test_most_defects_are_where_the_curvature_can_pay_for_them(self, junction):
        """The reframing that came out of measuring instead of looking.

        46 non-hexagons sounds like a broken structure. Only 10 of them
        are somewhere the curvature cannot account for; the rest are the
        caps and the neck doing what Gauss-Bonnet requires.
        """
        mesh, targets = junction
        total = census(mesh)
        unbudgeted = rm.unbudgeted_disclinations(mesh, targets)
        assert 0 < len(unbudgeted) < total / 2

    def test_a_hexagon_is_never_unbudgeted(self, junction):
        mesh, targets = junction
        adjacency = rm._adjacency(mesh[1])
        for vertex in rm.unbudgeted_disclinations(mesh, targets):
            assert len(adjacency[vertex]) != 6

    def test_the_window_is_the_unit_and_not_the_vertex(self, junction):
        """A pentagon carries exactly 1. On this junction no single vertex
        is ever asked for more than 0.167, because the curvature of a cap
        is spread over the whole cap — so a per-vertex test would call
        every pentagon unjustified, including the twelve that are required."""
        _mesh, targets = junction
        assert float(np.abs(targets).max()) < 0.5
        assert float(targets.sum()) == pytest.approx(12.0, abs=0.5)

    def test_the_report_names_the_three_things(self, junction):
        mesh, targets = junction
        text = rm.ring_report(mesh, targets)
        assert "anillos" in text
        assert "curvatura no los pide" in text
        assert "cúmulos" in text


class TestWhatFlipsCannotDo:
    """The negative result, pinned so nobody spends another afternoon on it.

    A flip changes four degrees: two fall, two rise. Moving a LONE
    disclination therefore costs +2 whichever flip is chosen, because it
    must leave a partner behind. Two separated disclinations cannot
    approach each other to annihilate without passing through that
    barrier, so greedy descent stalls and Metropolis only crosses it at
    temperatures that scramble the placement — measured: the census anneal
    at T=0.45 and above comes back with the misfit at 781 to 912 against
    238 for the unrefined mesh.
    """

    def test_greedy_descent_stalls_on_an_annealed_mesh(self, junction):
        mesh, targets = junction
        annealed = rm.anneal_edge_flips(mesh, np.random.default_rng(0),
                                        sweeps=80, temperature=0.3, restarts=2)
        placed, history = rm.place_disclinations(annealed, max_rounds=12,
                                                 pair_weight=1.0)
        # It runs, it does not crash, and it barely moves: this is a
        # statement about the method, not a defect to fix by tuning.
        assert len(history) <= 6
        assert census(placed) == census(annealed)

    def test_a_flip_never_moves_a_vertex(self, junction):
        """Whatever the objective, only connectivity changes — which is why
        the curvature targets are computed once and held."""
        mesh, targets = junction
        for kwargs in ({}, {"objective": "curvature", "targets": targets}):
            result = rm.anneal_edge_flips(mesh, np.random.default_rng(3),
                                          sweeps=20, **kwargs)
            assert np.array_equal(result[0], mesh[0])

    def test_the_total_charge_is_conserved_whatever_is_done(self, junction):
        """Gauss-Bonnet is not negotiable, and an objective that broke it
        would be producing a different surface."""
        mesh, targets = junction

        def charge(m):
            return sum(6 - len(ns) for ns in rm._adjacency(m[1]).values())

        expected = charge(mesh)
        for kwargs in ({}, {"like_penalty": 2.0},
                       {"objective": "curvature", "targets": targets,
                        "pair_weight": 3.0}):
            result = rm.anneal_edge_flips(mesh, np.random.default_rng(5),
                                          sweeps=40, temperature=0.4, **kwargs)
            assert charge(result) == expected
