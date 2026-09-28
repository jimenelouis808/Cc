"""Knee toroids: the census is by construction, so the tests can be exact.

Nothing here is a tolerance on a measured quantity. The mesh is built,
counted and checked before an atom exists, and the relaxation is handed a
bond graph it cannot alter -- so every claim about the topology is an
equality, and the claims about the geometry are the package's own sp2
report rather than a promise.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nanocarbon_lab.builders.knee import (
    CURVATURE_PENTAGONS,
    MESH_EDGE,
    MIN_KNEES,
    PAIRS_PER_KNEE,
    SOUND_SHAPE,
    TURN_PER_PAIR,
    build_knee_toroid,
    clean_circumferences,
    clean_shapes,
    defect_contacts,
    describe_knee_toroid,
    knee_polygon_mesh,
    mesh_census,
)


def _mesh(knees: int, circumference: int, arm_rows: int, bond: float = 1.42,
          seam_shift: int = -1):
    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / circumference))
    return knee_polygon_mesh(knees, circumference, arm_rows, radius, spacing,
                             seam_shift=seam_shift)


@pytest.fixture(scope="module")
def toroid():
    """The law-exact toroid: six knees, two pairs each, 30 deg a pair."""
    return build_knee_toroid(**SOUND_SHAPE)


class TestTheCensusIsExact:
    def test_twelve_pentagons_and_twelve_heptagons_and_nothing_else(self, toroid):
        assert toroid.info["ring_counts"] == {5: 12, 6: 222, 7: 12}

    def test_the_euler_budget_is_exactly_zero(self, toroid):
        assert toroid.info["ring_deficit"] == 0

    def test_every_atom_is_three_coordinate(self, toroid):
        degree: dict[int, int] = {}
        for i, j in toroid.info["bonds"]:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1
        assert set(degree.values()) == {3}
        assert len(degree) == len(toroid)

    def test_two_pairs_a_knee_whatever_the_shape(self):
        for knees, circumference, rows in ((6, 8, 7), (6, 9, 7), (8, 10, 7)):
            vertices, tris, _ = _mesh(knees, circumference, rows)
            census, _, broken = mesh_census(vertices, tris)
            assert broken == 0
            assert census[5] == census[7] == PAIRS_PER_KNEE * knees


class TestTheLaw:
    """A 5-7 pair turns the axis 30 deg and a torus asks for twelve of them,
    so six knees of two pairs is the whole budget and nothing else is."""

    def test_the_default_spends_exactly_the_curvature_budget(self, toroid):
        assert toroid.info["ring_counts"][5] == CURVATURE_PENTAGONS
        assert toroid.info["curvature_balance"] == 1.0

    def test_and_turns_exactly_thirty_degrees_a_pair(self, toroid):
        assert toroid.info["turn_per_pair_deg"] == TURN_PER_PAIR
        assert (toroid.info["knees"] * toroid.info["pairs_per_knee"]
                == CURVATURE_PENTAGONS)

    def test_eight_knees_over_correct_and_the_line_says_so(self):
        atoms = build_knee_toroid(knees=8, circumference=10, arm_rows=7,
                                  relax=False)
        assert atoms.info["ring_counts"][5] == 16
        assert atoms.info["curvature_balance"] > 1.0
        assert atoms.info["turn_per_pair_deg"] == 22.5
        assert "against the 30 the law asks for" in describe_knee_toroid(atoms)


class TestTheDisclinationsGoWhereTheCurvatureIs:
    def test_every_pentagon_is_outside_and_every_heptagon_inside(self, toroid):
        assert toroid.info["disclinations_placed"] == 1.0

    def test_it_holds_across_the_clean_shapes(self):
        for rows, circumference in clean_shapes(6)[:4]:
            vertices, tris, centre_radius = _mesh(6, circumference, rows)
            census, degrees, broken = mesh_census(vertices, tris)
            assert broken == 0
            assert set(census) == {5, 6, 7}
            assert census[5] == census[7] == CURVATURE_PENTAGONS
            radius = np.hypot(vertices[:, 0], vertices[:, 1])
            assert all(radius[x] > centre_radius
                       for x, d in degrees.items() if d == 5)
            assert all(radius[x] < centre_radius
                       for x, d in degrees.items() if d == 7)


class TestEveryDisclinationIsAloneInHexagons:
    def test_no_two_pentagons_and_no_two_heptagons_touch(self, toroid):
        """The complaint this builder answers: lobes of pentagons and fused
        heptagon pairs. There are none -- not few, none."""
        assert toroid.info["like_sign_pairs"] == 0

    def test_no_pentagon_shares_an_edge_with_a_heptagon(self, toroid):
        """A fused 5-7 is an axial dislocation, which is what a stack of
        circular rings is forced into and what the mitre avoids."""
        assert toroid.info["fused_dipoles"] == 0

    def test_it_holds_across_the_clean_shapes(self):
        for knees, circumference, rows in ((6, 8, 7), (6, 9, 7), (6, 10, 11)):
            vertices, tris, _ = _mesh(knees, circumference, rows)
            assert defect_contacts(vertices, tris) == {"like": 0, "fused": 0}


class TestTheShapeIsActuallySound:
    """The census being exact does not make the shape right, so the shape is
    measured too -- on the finished atoms, by the same report the capped tube
    and the fullerene are judged by."""

    def test_the_bonds_are_sp2(self, toroid):
        quality = toroid.info["geometry"]
        assert 1.39 <= quality["bond_min"]
        assert quality["bond_max"] <= 1.45

    def test_the_smallest_bond_angle_is_a_pentagon_and_not_a_fold(self, toroid):
        """108 deg is a pentagon's interior angle; anything near 90 is a
        wall that has folded."""
        assert toroid.info["geometry"]["angle_min"] >= 105.0

    def test_the_wall_did_not_fold_through_itself(self, toroid):
        assert toroid.info["geometry"]["n_close_contacts"] == 0

    def test_the_arms_stay_equilateral_away_from_the_knees(self):
        """What the exact seam buys: sliding the cut boundary onto the mitre
        plane before joining leaves the arms untouched."""
        vertices, tris, centre_radius = _mesh(6, 8, 7)
        corners = np.array([
            [centre_radius * np.cos(np.pi * q / 3.0),
             centre_radius * np.sin(np.pi * q / 3.0), 0.0] for q in range(6)])
        edges = sorted({(min(a, b), max(a, b)) for t in tris
                        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))})
        pairs = np.asarray(edges, dtype=int)
        lengths = np.linalg.norm(vertices[pairs[:, 0]] - vertices[pairs[:, 1]],
                                 axis=1)
        middles = 0.5 * (vertices[pairs[:, 0]] + vertices[pairs[:, 1]])
        away = np.min(np.linalg.norm(middles[:, None, :] - corners[None, :, :],
                                     axis=2), axis=1) > 6.0
        target = MESH_EDGE * 1.42
        assert lengths[away].max() == pytest.approx(target, abs=0.02)
        assert lengths[away].min() == pytest.approx(target, abs=0.02)


class TestTheSeamOffsetChoosesTheKnee:
    def test_no_offset_gives_a_square_outside_and_an_octagon_inside(self):
        atoms = build_knee_toroid(**SOUND_SHAPE, knee="octagon")
        assert atoms.info["ring_counts"] == {4: 12, 6: 222, 8: 12}
        assert atoms.info["ring_deficit"] == 0

    def test_both_knees_are_the_same_size(self):
        five = build_knee_toroid(**SOUND_SHAPE, relax=False)
        eight = build_knee_toroid(**SOUND_SHAPE, knee="octagon", relax=False)
        assert len(five) == len(eight)

    def test_an_unknown_knee_is_refused(self):
        with pytest.raises(ValueError, match="pentagon"):
            build_knee_toroid(**SOUND_SHAPE, knee="heptagon")


class TestTheConstructionRulesAreEnforced:
    def test_an_even_arm_length_is_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="odd"):
            build_knee_toroid(knees=6, circumference=8, arm_rows=8)

    def test_too_few_knees_is_refused(self):
        with pytest.raises(ValueError, match="at least"):
            build_knee_toroid(knees=MIN_KNEES - 2)

    def test_an_odd_knee_count_is_refused_with_the_reason(self):
        """The arms alternate handedness, so an odd ring cannot close."""
        with pytest.raises(ValueError, match="even"):
            build_knee_toroid(knees=7)

    def test_a_circumference_that_does_not_close_names_the_ones_that_do(self):
        good = clean_circumferences(6, 7)
        assert good, "six knees with seven-row arms must have some exact shape"
        bad = next(k for k in range(8, 25) if k not in good)
        with pytest.raises(ValueError) as excinfo:
            build_knee_toroid(knees=6, circumference=bad, arm_rows=7)
        for k in good:
            assert str(k) in str(excinfo.value)

    def test_leaving_both_knobs_open_finds_a_usable_shape(self):
        atoms = build_knee_toroid(knees=6, relax=False)
        counts = atoms.info["ring_counts"]
        assert counts[5] == counts[7] == CURVATURE_PENTAGONS
        assert set(counts) == {5, 6, 7}
        assert atoms.info["arm_rows"] % 2 == 1
        assert 3.0 <= atoms.info["aspect_ratio"] <= 6.0


class TestRelaxationCannotChangeTheTopology:
    def test_the_census_is_the_same_relaxed_or_not(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loose = build_knee_toroid(**SOUND_SHAPE, relax=False)
            tight = build_knee_toroid(**SOUND_SHAPE, relax=True)
        assert loose.info["ring_counts"] == tight.info["ring_counts"]
        assert len(loose) == len(tight)


class TestWhatItSaysAboutItself:
    def test_the_description_names_the_knees_and_the_placement(self, toroid):
        line = describe_knee_toroid(toroid)
        assert "6 knees" in line
        assert "60.0 deg" in line
        assert "exactly 30 deg a pair" in line
        assert "sum(6-n) = +0" in line
        assert "100%" in line
        assert "every one isolated in hexagons" in line

    def test_the_metadata_says_where_it_came_from(self, toroid):
        assert toroid.info["builder"] == "knee_toroid"
        assert toroid.info["structure_type"] == "toroid"
        assert toroid.info["bend_per_knee_deg"] == 60.0
        assert toroid.info["knee"] == "pentagon"
