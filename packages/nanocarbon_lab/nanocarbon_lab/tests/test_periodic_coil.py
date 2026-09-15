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
    def test_a_pitch_that_would_merge_the_turns_is_refused(self):
        with pytest.raises(ValueError, match="merge"):
            build_periodic_coil(coil_radius=R, pitch=4.0, tube_radius=R_TUBE)

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
