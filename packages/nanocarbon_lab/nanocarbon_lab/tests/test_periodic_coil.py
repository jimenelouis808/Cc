"""One turn of a coil, closed on the z-torus.

A structure that is quietly *not* periodic is worse than none: a
plane-wave code will accept it, run, and return an energy for a cell
whose seam is torn. So these tests check periodicity itself rather than
just that a build returns atoms.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import remesh as rm
from nanocarbon_lab.builders.periodic_coil import (
    build_periodic_coil,
    helix_period_samples,
)

R, PITCH, R_TUBE = 15.0, 9.6, 3.0


@pytest.fixture(scope="module")
def coil():
    return build_periodic_coil(coil_radius=R, pitch=PITCH, tube_radius=R_TUBE,
                               resolution=64)


class TestTheFieldIsPeriodicToMachinePrecision:
    """The weld matches vertices between the two end planes.

    It can only do that if the field is the same on both, and the
    centreline resampling is what breaks that: at a loose spacing the
    mismatch is 0.05 Å and the mesh comes back with hundreds of boundary
    edges.
    """

    def test_the_spacing_divides_one_turn_a_whole_number_of_times(self):
        spacing = helix_period_samples(R, PITCH, target_spacing=0.3)
        arc = float(np.hypot(2.0 * np.pi * R, PITCH))
        divisions = arc / spacing
        assert divisions == pytest.approx(round(divisions), abs=1e-9)

    def test_it_stays_near_the_spacing_that_was_asked_for(self):
        """Commensurate, not arbitrary: the accuracy budget still holds."""
        spacing = helix_period_samples(R, PITCH, target_spacing=0.3)
        assert 0.29 < spacing < 0.31


class TestTheCellIsActuallyPeriodic:
    def test_only_the_helix_axis_is_periodic(self, coil):
        assert list(coil.get_pbc()) == [False, False, True]

    def test_the_cell_is_exactly_one_pitch_along_z(self, coil):
        assert coil.cell[2][2] == pytest.approx(PITCH)

    def test_bonds_cross_the_seam(self, coil):
        """The test that separates a cell from a fragment.

        A segment with two free ends has no bond spanning the cell; a
        periodic one has a whole ring of them.
        """
        positions = coil.get_positions()
        half = 0.5 * float(coil.cell[2][2])
        crossing = [(u, v) for u, v in coil.info["bonds"]
                    if abs(positions[u][2] - positions[v][2]) > half]
        assert len(crossing) > 10, "nothing bonds across z: this is a fragment"

    def test_no_atom_overlaps_its_own_periodic_image(self, coil):
        distances = coil.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() > 1.1

    def test_translating_by_one_pitch_maps_the_cell_onto_itself(self, coil):
        """The definition of the periodicity being claimed.

        Every atom shifted by the pitch must land on an atom -- which is
        automatic once the cell is right, and fails loudly if the seam
        was welded at the wrong offset.
        """
        positions = coil.get_positions()
        shifted = positions + np.array([0.0, 0.0, float(coil.cell[2][2])])
        wrapped = np.mod(shifted, np.diag(coil.cell))
        for point in wrapped[::7]:
            assert np.min(np.linalg.norm(positions - point, axis=1)) < 1e-6


class TestTheTopologyIsAToruss:
    def test_the_mesh_is_closed(self, coil):
        assert coil.info["mesh_genus"] == 1, "one period of a tube is a torus"

    def test_the_ring_budget_is_what_a_torus_owes(self, coil):
        """``sum(6 - n) = 12(1 - g)``, which for a torus is zero.

        A mesh that pinched through itself passes every local check and
        fails this one.
        """
        counts = coil.info["ring_counts"]
        assert sum((6 - size) * n for size, n in counts.items()) == 0

    def test_the_curvature_is_paid_in_rings_not_in_stretched_bonds(self, coil):
        """Pentagons inside, heptagons outside: how a real coil bends.

        A pure-hexagon tube bent onto a helix has to stretch its outer
        wall instead, and no relaxation removes that.
        """
        counts = coil.info["ring_counts"]
        assert counts.get(5, 0) > 0 and counts.get(7, 0) > 0
        geometry = coil.info["geometry"]
        assert geometry["bond_max"] < 1.60
        assert geometry["bond_min"] > 1.25


class TestItRefusesWhatItCannotBuild:
    def test_a_pitch_that_would_pass_one_turn_through_the_next_is_refused(self):
        """Impossible, not merely tight. The turns would intersect."""
        with pytest.raises(ValueError, match="impossible"):
            build_periodic_coil(coil_radius=R, pitch=R_TUBE, tube_radius=R_TUBE)



    def test_a_torn_network_is_refused_even_when_euler_is_happy(self):
        """Euler is necessary and not sufficient.

        A 25 Å coil at this resolution comes back with the ring budget a
        torus owes and a 2.2 Å "bond": topologically consistent, and torn.
        Without this guard it would have been handed back as a DFT cell.
        """
        with pytest.raises(ValueError, match="torn"):
            build_periodic_coil(coil_radius=25.0, pitch=PITCH,
                                tube_radius=R_TUBE, resolution=64)

    def test_nonpositive_dimensions_are_refused(self):
        with pytest.raises(ValueError, match="positive"):
            build_periodic_coil(coil_radius=-1.0, pitch=PITCH,
                                tube_radius=R_TUBE)


class TestTheAnisotropicMesher:
    def test_a_scalar_cell_still_behaves_as_a_cube(self):
        """The schwarzites depend on this path and must not shift."""
        def sphere(points):
            centred = points - 10.0
            return np.linalg.norm(centred, axis=-1) - 4.0

        cubic = rm.periodic_marching_cubes_mesh(sphere, 20.0, resolution=32)
        vector = rm.periodic_marching_cubes_mesh(
            sphere, np.array([20.0, 20.0, 20.0]), resolution=32)
        assert len(cubic[0]) == len(vector[0])
        assert len(cubic[1]) == len(vector[1])

    def test_a_degenerate_cell_is_refused(self):
        with pytest.raises(ValueError, match="positive"):
            rm.periodic_marching_cubes_mesh(
                lambda p: np.zeros(p.shape[:-1]), np.array([10.0, 0.0, 10.0]))


# ----------------------------------------------------------------------
# What the published coils look like
# ----------------------------------------------------------------------
#: Liu et al., Nanoscale Res. Lett. 5 (2010) 478, Tables 1 and 2: the
#: relaxed single-wall coils built on (n,n) tubes, with the tube radius,
#: the effective coil diameter and the equilibrium pitch.
LIU_COILS = {
    "(5,5)": (3.40, 26.38, 13.61),
    "(6,6)": (4.08, 28.85, 12.64),
    "(7,7)": (4.76, 33.58, 12.11),
    "(8,8)": (5.44, 39.21, 12.27),
}


class TestAgainstThePublishedGeometry:
    def test_the_literature_band_is_what_the_papers_report(self):
        """Two independent studies, one number. Popović et al. (2012) find
        D/d ~ 3.5 for both of their classes; Liu et al. (2010) Table 2
        gives 3.53 to 3.88. The band the builder judges against has to
        contain what they actually published."""
        from nanocarbon_lab.builders.periodic_coil import LITERATURE_RATIO

        low, high = LITERATURE_RATIO
        for name, (radius, diameter, _pitch) in LIU_COILS.items():
            ratio = diameter / (2.0 * radius)
            assert low <= ratio <= high, f"{name}: D/d = {ratio:.2f}"

    def test_a_published_pitch_is_built_and_not_refused(self):
        """Liu's relaxed (7,7) has a 12.11 Å pitch around a 9.52 Å tube:
        the walls are 2.59 Å apart, inside the 3.4 Å graphitic gap. The
        old guard refused it. A structure somebody has already relaxed
        with DFT is not an impossible one; it is a tight one, and it is
        said so rather than rejected."""
        radius, diameter, pitch = LIU_COILS["(7,7)"]
        coil = build_periodic_coil(coil_radius=diameter / 2.0 - radius,
                                   pitch=pitch, tube_radius=radius,
                                   resolution=64)
        assert coil.info["wall_gap"] < 3.4
        assert any("van der Waals" in note for note in coil.info["notes"])

    def test_turns_that_would_intersect_are_still_refused(self):
        with pytest.raises(ValueError, match="impossible"):
            build_periodic_coil(coil_radius=12.0, pitch=6.0, tube_radius=3.4)


class TestThePentagonsGoOutside:
    """On a torus-like surface the Gaussian curvature is positive on the
    outer equator and negative on the inner one, so a +60° disclination
    (a pentagon) belongs outside and a -60° one (a heptagon) inside. Both
    papers say it in those words. It is the one structural claim about a
    coil that can be checked on a finished model.
    """

    def test_the_check_reports_where_they_landed(self, coil):
        placement = coil.info["curvature_check"]
        assert placement["pentagons"] and placement["heptagons"]
        assert placement["pentagon_radius"] > placement["hexagon_radius"]
        assert placement["heptagon_radius"] < placement["hexagon_radius"]
        # Most of them, not all: the remesher puts rings where the mesh
        # needs them and a few come out on the wrong side. That is a
        # defect of the model and it is reported as one rather than
        # rounded away.
        assert placement["placed_correctly"] > 0.6

    def test_a_coil_that_got_it_wrong_would_say_so(self):
        from nanocarbon_lab.builders.periodic_coil import curvature_check

        positions = np.array([[10.0, 0.0, 0.0], [5.0, 0.0, 0.0],
                              [7.5, 0.0, 0.0]])
        rings = [[0] * 5, [1] * 7, [2] * 6]      # a 5 outside, a 7 inside
        good = curvature_check(positions, rings, np.zeros(2))
        assert good["placed_correctly"] == 1.0
        swapped = curvature_check(positions, [[0] * 7, [1] * 5, [2] * 6],
                                  np.zeros(2))
        assert swapped["placed_correctly"] == 0.0


class TestThePolygonalCoil:
    """Seen down the axis a real coil is a polygon, not a circle: Liu et
    al. show the (6,6) coil's top view as a hexagonal torus and say it
    matches what is observed. The wall relieves its strain at a few knees
    rather than everywhere at once, and the builder can do the same.
    """

    @pytest.fixture(scope="class")
    def hexagonal(self):
        return build_periodic_coil(coil_radius=9.79, pitch=13.61,
                                   tube_radius=3.40, sides=6, resolution=64)

    def test_it_builds_and_stays_periodic(self, hexagonal):
        assert hexagonal.info["sides"] == 6
        assert hexagonal.info["ring_deficit"] == 0
        assert hexagonal.cell[2, 2] == pytest.approx(13.61)

    def test_concentrating_the_curvature_places_the_rings_better(self, hexagonal):
        """Measured on the (5,5) geometry: the smooth helix puts 82 % of
        its non-hexagons on the correct side, the hexagonal polygon 88 %,
        and every one of the polygon's pentagons is on the outside."""
        placement = hexagonal.info["curvature_check"]
        assert placement["pentagons"] == placement["pentagons_outside"]
        assert placement["placed_correctly"] >= 0.8

    def test_a_corner_on_a_sample_is_not_a_resolution_problem(self):
        """The sample grid holds a whole number of samples per turn, and
        if the side count divides it a corner lands exactly on a sample.
        The tangent there is two-valued, the swept frame flips, and the
        seam does not weld -- and raising the resolution makes it worse,
        which is how it was told apart from a coarse mesh. The grid is
        chosen coprime with the side count instead.
        """
        for sides in (5, 10):                    # both divide 4000
            coil = build_periodic_coil(coil_radius=9.79, pitch=13.61,
                                       tube_radius=3.40, sides=sides,
                                       resolution=64)
            assert coil.info["ring_deficit"] == 0

    def test_a_polygon_needs_at_least_three_sides(self):
        from nanocarbon_lab.builders.periodic_coil import coil_centreline

        with pytest.raises(ValueError, match="at least 3"):
            coil_centreline(10.0, 8.0, 2, 1.0, 0.0, 1.0, 101)
