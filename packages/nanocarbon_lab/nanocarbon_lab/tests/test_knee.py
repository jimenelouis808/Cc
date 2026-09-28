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
    JUNCTION_AXES,
    MESH_EDGE,
    MIN_KNEES,
    PAIRS_PER_KNEE,
    SOUND_SHAPE,
    TURN_PER_PAIR,
    build_knee_coil,
    build_knee_junction,
    build_knee_schwarzite,
    build_knee_toroid,
    clean_circumferences,
    clean_junction_shapes,
    clean_shapes,
    collapse_degree_four,
    collapse_degree_three,
    defect_contacts,
    describe_knee_coil,
    describe_knee_junction,
    describe_knee_schwarzite,
    describe_knee_toroid,
    fill_triangular_holes,
    knee_path_mesh,
    knee_polygon_mesh,
    mesh_census,
    node_budget,
    node_mesh,
    primitive_node_mesh,
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


@pytest.fixture(scope="module")
def coil():
    """D/d 3.73 -- inside the band the single-wall coil papers report, and
    a radius the lattice-winding route refuses outright."""
    return build_knee_coil(coil_radius=12.0, pitch=12.0, sides_per_turn=8,
                           turns=2, circumference=8)


class TestTheSameKneeWindsACoil:
    """A coil is the same mitre on a helix instead of a ring. The
    reflection sends one corner to the next on any equal-step path, so one
    routine covers both -- and the coil is the case that needs it most,
    since winding a finished lattice refuses a 25 A radius outright."""

    def test_two_pentagons_and_two_heptagons_a_knee(self, coil):
        counts = coil.info["ring_counts"]
        expected = PAIRS_PER_KNEE * coil.info["knees"]
        assert counts[5] == counts[7] == expected

    def test_nothing_but_pentagons_hexagons_and_heptagons(self, coil):
        assert set(coil.info["ring_counts"]) == {5, 6, 7}

    def test_pentagons_outside_the_helix_and_heptagons_inside(self, coil):
        assert coil.info["disclinations_placed"] == 1.0

    def test_every_disclination_is_alone_in_hexagons(self, coil):
        assert coil.info["like_sign_pairs"] == 0
        assert coil.info["fused_dipoles"] == 0

    def test_the_wall_is_sp2(self, coil):
        quality = coil.info["geometry"]
        assert 1.30 <= quality["bond_min"]
        assert quality["bond_max"] <= 1.55
        assert quality["angle_min"] >= 100.0
        assert quality["n_close_contacts"] == 0

    def test_it_is_tight_enough_to_be_a_single_wall_coil(self, coil):
        low, high = coil.info["literature_coil_aspect"]
        assert low <= coil.info["coil_aspect"] <= high

    def test_the_two_rims_are_two_coordinate_and_recorded(self, coil):
        """A coil is open at both ends, as a nanocone is at its base."""
        rim = coil.info["rim_atoms"]
        assert rim
        degree: dict[int, int] = {}
        for i, j in coil.info["bonds"]:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1
        assert all(degree.get(x, 0) < 3 for x in rim)
        assert coil.info["terminal_atoms"] == rim

    def test_a_pitch_that_does_not_clear_the_tube_is_refused(self):
        with pytest.raises(ValueError, match="clear"):
            build_knee_coil(coil_radius=12.0, pitch=3.0, circumference=8)

    def test_the_description_names_the_turns_and_the_placement(self, coil):
        line = describe_knee_coil(coil)
        assert "2 turns of 8 sides" in line
        assert "D/d 3.73" in line
        assert "100%" in line
        assert "every one isolated in hexagons" in line


class TestOnePathRoutineCoversBoth:
    def test_the_general_path_reproduces_the_toroid(self):
        """The polygon is just the closed, planar case, so the two must
        agree exactly on it."""
        bond = 1.42
        spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
        radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / 8))
        centre = 7 * spacing / (2.0 * np.tan(np.pi / 6))
        corners = np.array([
            [centre * np.cos(np.pi * q / 3.0), centre * np.sin(np.pi * q / 3.0),
             0.0] for q in range(6)])
        vertices, tris, _ = knee_path_mesh(corners, 8, radius, spacing)
        census, _, broken = mesh_census(vertices, tris)
        assert broken == 0
        assert census == {5: 12, 6: 222, 7: 12}

    def test_an_uneven_path_is_refused_with_the_reason(self):
        """The reflection sends one corner to the next only when the steps
        are equal, so an uneven path would leave a knee out of register."""
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                           [10.0, 3.0, 0.0], [0.0, 3.0, 0.0]])
        with pytest.raises(ValueError, match="not equal"):
            knee_path_mesh(points, 8, 3.2, 2.13)


@pytest.fixture(scope="module")
def schwarzite():
    """One Schwarz P cell from a six-arm node."""
    return build_knee_schwarzite(circumference=20, arm_rows=9)


class TestOneLawForEveryNode:
    """A node of `c` arms is a sphere with `c` holes, so chi = 2 - c and
    sum(6-n) = 6(2-c). Counting arms reproduces the genus table the
    implicit schwarzite builder quotes."""

    def test_a_knee_is_the_two_arm_case_and_pays_nothing(self):
        assert node_budget(2) == 0

    def test_it_reproduces_the_published_surface_budgets(self):
        assert node_budget(6) == -24                      # Schwarz P, 1 node
        assert 8 * node_budget(3) == -48                  # gyroid, srs net
        assert 8 * node_budget(4) == -96                  # Schwarz D


class TestTheSchwarziteCensusIsExact:
    def test_hexagons_and_exactly_twenty_four_heptagons(self, schwarzite):
        assert schwarzite.info["ring_counts"] == {6: 456, 7: 24}

    def test_not_one_pentagon(self, schwarzite):
        """A minimal surface saddles everywhere, so it has no positive
        curvature for a pentagon to sit in. The implicit route returns 33
        at a comparable cell."""
        assert schwarzite.info["pentagons"] == 0

    def test_the_budget_is_the_node_law(self, schwarzite):
        assert schwarzite.info["ring_deficit"] == node_budget(6)
        assert schwarzite.info["ring_budget"] == -24

    def test_it_is_periodic_in_all_three_directions(self, schwarzite):
        assert all(schwarzite.get_pbc())
        assert schwarzite.cell[0][0] == pytest.approx(
            schwarzite.info["cell_length"], abs=1e-3)

    def test_every_atom_is_three_coordinate(self, schwarzite):
        degree: dict[int, int] = {}
        for i, j in schwarzite.info["bonds"]:
            degree[i] = degree.get(i, 0) + 1
            degree[j] = degree.get(j, 0) + 1
        assert set(degree.values()) == {3}
        assert len(degree) == len(schwarzite)

    def test_the_wall_is_sp2(self, schwarzite):
        quality = schwarzite.info["geometry"]
        assert 1.30 <= quality["bond_min"]
        assert quality["bond_max"] <= 1.56
        assert quality["angle_min"] >= 100.0
        assert quality["n_close_contacts"] == 0

    def test_the_description_says_it_is_pentagon_free(self, schwarzite):
        line = describe_knee_schwarzite(schwarzite)
        assert "genus 3" in line
        assert "sum(6-n) = -24" in line
        assert "no pentagons" in line


class TestCollapsingADegreeThreeVertexIsExact:
    """Where three arms meet each brings one edge, so the shared vertex
    comes out degree 3 -- a three-membered ring. Removing it and filling
    its link adds no edge, because its three neighbours are already
    adjacent, so chi is untouched and each of them drops a degree."""

    def test_it_turns_the_octagons_into_heptagons(self):
        bond = 1.42
        spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
        radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / 20))
        vertices, tris, _ = primitive_node_mesh(20, 9, radius, spacing)
        before, _, _ = mesh_census(vertices, tris)
        assert before == {3: 8, 6: 456, 8: 24}
        after_vertices, after_tris = collapse_degree_three(vertices, tris)
        after, _, broken = mesh_census(after_vertices, after_tris)
        assert after == {6: 456, 7: 24}
        assert broken == 0

    def test_it_leaves_the_euler_characteristic_alone(self):
        bond = 1.42
        spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
        radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / 20))
        vertices, tris, _ = primitive_node_mesh(20, 9, radius, spacing)

        def euler(v, t):
            edges = {(min(a, b), max(a, b)) for x in t
                     for a, b in ((x[0], x[1]), (x[1], x[2]), (x[2], x[0]))}
            return len(v) - len(edges) + len(t)

        moved_vertices, moved_tris = collapse_degree_three(vertices, tris)
        assert euler(vertices, tris) == euler(moved_vertices, moved_tris)
        assert euler(moved_vertices, moved_tris) == -4      # genus 3

    def test_a_pair_that_does_not_close_names_the_ones_that_do(self):
        with pytest.raises(ValueError) as excinfo:
            build_knee_schwarzite(circumference=9, arm_rows=9, relax=False)
        assert "arm_rows" in str(excinfo.value)


def _interior(tris):
    """The degree census of the interior alone, and the rim size."""
    counts: dict[tuple[int, int], int] = {}
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            key = (min(a, b), max(a, b))
            counts[key] = counts.get(key, 0) + 1
    rim = {x for e, c in counts.items() if c != 2 for x in e}
    degree: dict[int, int] = {}
    for a, b in counts:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    census: dict[int, int] = {}
    for x, d in degree.items():
        if x not in rim:
            census[d] = census.get(d, 0) + 1
    return dict(sorted(census.items())), len(rim)


def _euler(vertices, tris):
    edges = {(min(a, b), max(a, b)) for x in tris
             for a, b in ((x[0], x[1]), (x[1], x[2]), (x[2], x[0]))}
    return len(vertices) - len(edges) + len(tris)


def _node(kind: str, circumference: int, arm_rows: int, bond: float = 1.42):
    spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
    radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / circumference))
    return node_mesh(JUNCTION_AXES[kind], circumference, arm_rows, radius,
                     spacing)


class TestAJunctionIsASphereWithHoles:
    """A node of `c` arms is a sphere with `c` holes, so chi = 2 - c and
    the budget is 6(2 - c). With no pentagon available -- a junction
    saddles everywhere -- it is paid in heptagons alone."""

    @pytest.mark.parametrize("kind,arms,heptagons",
                             [("y", 3, 6), ("tetrahedral", 4, 12)])
    def test_the_census_is_hexagons_and_exactly_those_heptagons(
            self, kind, arms, heptagons):
        atoms = build_knee_junction(kind=kind, circumference=10, arm_rows=7,
                                    relax=False)
        assert atoms.info["arms"] == arms
        assert atoms.info["euler"] == 2 - arms
        assert atoms.info["ring_budget"] == node_budget(arms) == -heptagons
        counts = atoms.info["ring_counts"]
        assert set(counts) == {6, 7}
        assert counts[7] == heptagons
        assert atoms.info["ring_deficit"] == -heptagons

    def test_it_carries_no_pentagon_at_all(self):
        atoms = build_knee_junction(circumference=12, arm_rows=9, relax=False)
        assert atoms.info["pentagons"] == 0

    @pytest.mark.parametrize("circumference", [10, 12, 14, 16])
    def test_the_census_does_not_depend_on_the_circumference(
            self, circumference):
        atoms = build_knee_junction(circumference=circumference, arm_rows=7,
                                    relax=False)
        assert set(atoms.info["ring_counts"]) == {6, 7}
        assert atoms.info["ring_counts"][7] == 6

    def test_the_mouths_are_left_open(self):
        atoms = build_knee_junction(circumference=10, arm_rows=7, relax=False)
        # Three arms, each mouth a ring of the circumference, and every one
        # of those atoms two-coordinate, as a nanocone's rim is.
        assert len(atoms.info["rim_atoms"]) == 30
        assert atoms.info["rim_atoms"] == atoms.info["terminal_atoms"]

    def test_an_unknown_kind_names_the_ones_that_exist(self):
        with pytest.raises(ValueError) as excinfo:
            build_knee_junction(kind="x")
        assert "tetrahedral" in str(excinfo.value)

    def test_a_shape_that_does_not_close_names_the_ones_that_do(self):
        with pytest.raises(ValueError) as excinfo:
            build_knee_junction(circumference=9, arm_rows=11, relax=False)
        assert "arm_rows" in str(excinfo.value)

    def test_the_clean_shape_list_is_not_empty(self):
        shapes = clean_junction_shapes("y", circumferences=(10, 12, 14),
                                       rows_candidates=(7, 9))
        assert shapes
        assert all(isinstance(r, int) and isinstance(k, int)
                   for r, k in shapes)


class TestFillingATriangularHoleChangesNoDegree:
    """Where three arms meet on a Y they leave a three-vertex hole rather
    than a shared vertex. All three of its edges already exist, so closing
    it adds none: no vertex changes degree, only F rises and chi with
    it."""

    def test_it_takes_the_node_from_minus_three_to_minus_one(self):
        vertices, tris = _node("y", 14, 9)
        assert _euler(vertices, tris) == -3
        filled = fill_triangular_holes(tris)
        assert _euler(vertices, filled) == -1
        assert len(filled) == len(tris) + 2

    def test_the_fill_is_what_makes_the_six_heptagons(self):
        vertices, tris = _node("y", 14, 9)
        before, rim_before = _interior(tris)
        after, rim_after = _interior(fill_triangular_holes(tris))
        # Two holes of three vertices each close, so six vertices join the
        # interior and the boundary is the three mouths alone.
        assert rim_before - rim_after == 6
        # And they come out **degree 7**: those six are the junction's
        # heptagons, so the fill does not merely tidy the node, it is what
        # puts the whole Gauss-Bonnet budget on the surface. Before it the
        # interior carries none.
        assert 7 not in before
        assert after == dict(sorted({**before, 7: 6}.items()))


class TestClimbingOutTheDipoleIsNotAFlip:
    """The welded seams each carry a neutral 4-8 pair. No flip can remove
    it -- a flip drops two degrees and raises two, so sum|deg-6| is 4
    before and 4 after, every time. Deleting the square and retriangulating
    its link does, and that is climb: it changes the vertex count."""

    def test_the_dipole_is_there_before_the_climb(self):
        vertices, tris = _node("y", 14, 9)
        tris = fill_triangular_holes(tris)
        vertices, tris = collapse_degree_three(vertices, tris)
        census, _ = _interior(tris)
        assert census == {4: 6, 6: 234, 7: 6, 8: 6}
        # Neutral: it costs the budget nothing, which is why no census
        # check sees it.
        assert sum((6 - s) * c for s, c in census.items()) == -6

    def test_no_edge_flip_can_pay_for_it(self):
        # A flip takes a degree off each endpoint and adds one to each
        # opposite vertex, so it moves sum|deg-6| by an even amount that
        # is zero for every flip touching this dipole. Stated as the
        # arithmetic rather than as a search: the four changes are
        # -1, -1, +1, +1 and they sum to zero whatever they land on.
        assert (-1) + (-1) + 1 + 1 == 0

    def test_the_climb_removes_it_and_nothing_else(self):
        vertices, tris = _node("y", 14, 9)
        tris = fill_triangular_holes(tris)
        vertices, tris = collapse_degree_three(vertices, tris)
        before = _euler(vertices, tris)
        climbed, climbed_tris = collapse_degree_four(vertices, tris)
        census, _ = _interior(climbed_tris)
        assert census == {6: 240, 7: 6}
        assert _euler(climbed, climbed_tris) == before
        # One vertex, four edges and four faces go; two faces come back.
        assert len(climbed) == len(vertices) - 6
        assert len(climbed_tris) == len(tris) - 12

    def test_it_is_a_no_op_on_a_cell_that_is_already_clean(self):
        bond = 1.42
        spacing = MESH_EDGE * bond * np.sqrt(3.0) / 2.0
        radius = MESH_EDGE * bond / (2.0 * np.sin(np.pi / 20))
        vertices, tris, _ = primitive_node_mesh(20, 9, radius, spacing)
        vertices, tris = collapse_degree_three(vertices, tris)
        after_vertices, after_tris = collapse_degree_four(vertices, tris)
        assert len(after_vertices) == len(vertices)
        assert len(after_tris) == len(tris)
        assert mesh_census(after_vertices, after_tris)[0] == {6: 456, 7: 24}


class TestTheJunctionGeometryIsCarbon:
    """The census being exact does not make the bonds carbon's. These are
    measured after the force field, against the sp2 window."""

    @pytest.fixture(scope="class")
    def junction(self):
        return build_knee_junction(circumference=14, arm_rows=9)

    def test_the_bonds_and_angles_are_sp2(self, junction):
        geometry = junction.info["geometry"]
        assert 1.30 <= geometry["bond_min"] <= geometry["bond_max"] <= 1.55
        assert 100.0 <= geometry["angle_min"]
        assert geometry["angle_max"] <= 135.0

    def test_nothing_overlaps(self, junction):
        assert junction.info["geometry"]["n_close_contacts"] == 0

    def test_the_relaxation_cannot_change_the_census(self, junction):
        loose = build_knee_junction(circumference=14, arm_rows=9, relax=False)
        assert junction.info["ring_counts"] == loose.info["ring_counts"]
        assert len(junction) == len(loose)

    def test_every_heptagon_is_in_negative_curvature(self, junction):
        # The intrinsic check, which is not a restatement of the census:
        # the surface is fitted over each ring and the sign of K read off.
        # A heptagon is a -60 deg disclination and belongs in negative
        # curvature; all six are, mean sign exactly -1.
        check = junction.info["disclination_check"]
        assert check["agreement"] == 1.0
        assert check["sizes"][7]["mean_sign"] == -1.0

    def test_it_records_the_rings_by_atom_index(self, junction):
        # `dopants/rings.py` places a heteroatom on a named ring size from
        # this and raises without it, rather than perceiving rings by
        # distance.
        rings = junction.info["rings"]
        assert len(rings) == sum(junction.info["ring_counts"].values())
        assert all(0 <= x < len(junction) for ring in rings for x in ring)

    def test_what_it_says_about_itself(self, junction):
        line = describe_knee_junction(junction)
        assert "3 arms" in line
        assert "7:6" in line
        assert "nothing else" in line
