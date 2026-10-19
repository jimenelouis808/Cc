"""Tests for the structure builders."""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import (
    build_carbon_foam,
    build_cnt,
    build_graphene,
    build_graphene_supercell,
    build_nanoribbon,
)
from nanocarbon_lab.utils.constants import CC_BOND, HARD_MIN_DISTANCE


class TestCNT:
    def test_zigzag_chirality(self):
        atoms = build_cnt(6, 0, length=10)
        assert atoms.info["chirality"] == "zigzag"
        assert len(atoms) > 0
        # All atoms must be carbons.
        assert set(atoms.get_chemical_symbols()) == {"C"}

    def test_armchair_chirality(self):
        atoms = build_cnt(5, 5, length=8)
        assert atoms.info["chirality"] == "armchair"
        # Periodic along z only.
        assert list(atoms.get_pbc()) == [False, False, True]

    def test_chiral_chirality(self):
        atoms = build_cnt(6, 3, length=5)
        assert atoms.info["chirality"] == "chiral"

    def test_length_at_least_target(self):
        atoms = build_cnt(5, 5, length=15)
        assert atoms.cell[2, 2] >= 15

    def test_vacuum_applied(self):
        atoms = build_cnt(5, 5, length=5, vacuum=12.0)
        r = atoms.info["radius"]
        # Transverse box should be 2r + vacuum.
        assert atoms.cell[0, 0] == pytest.approx(2 * r + 12.0, abs=1e-6)
        assert atoms.cell[1, 1] == pytest.approx(2 * r + 12.0, abs=1e-6)

    def test_invalid_chirality_raises(self):
        with pytest.raises(ValueError):
            build_cnt(3, 5, length=5)
        with pytest.raises(ValueError):
            build_cnt(0, 0, length=5)

    def test_negative_length_raises(self):
        with pytest.raises(ValueError):
            build_cnt(5, 5, length=-1)

    def test_no_atomic_overlap(self):
        atoms = build_cnt(6, 6, length=8)
        dmat = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dmat, np.inf)
        assert dmat.min() > HARD_MIN_DISTANCE


class TestGraphene:
    def test_primitive_has_two_atoms(self):
        atoms = build_graphene()
        assert len(atoms) == 2
        assert list(atoms.get_pbc()) == [True, True, False]

    def test_supercell_size(self):
        atoms = build_graphene_supercell(3, 2)
        assert len(atoms) == 4 * 3 * 2

    def test_bond_length_close_to_nominal(self):
        atoms = build_graphene_supercell(2, 2)
        dmat = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dmat, np.inf)
        min_d = dmat.min()
        assert abs(min_d - CC_BOND) < 0.02


class TestNanoribbon:
    def test_zigzag_ribbon(self):
        atoms = build_nanoribbon(4, 3, edge="zigzag")
        assert len(atoms) > 0
        # z, the axis the repeat unit runs along -- 3 units of 2.46 Å,
        # not the 15 Å of vacuum that y and x carry.
        assert list(atoms.get_pbc()) == [False, False, True]
        assert atoms.cell.lengths()[2] == pytest.approx(3 * 3**0.5 * 1.42,
                                                        rel=1e-3)

    def test_passivation_adds_hydrogens(self):
        plain = build_nanoribbon(4, 3, edge="zigzag", passivate=False)
        passivated = build_nanoribbon(4, 3, edge="zigzag", passivate=True)
        assert "H" not in plain.get_chemical_symbols()
        assert "H" in passivated.get_chemical_symbols()


class TestCarbonFoam:
    def test_reproducible_with_seed(self):
        a = build_carbon_foam(box_size=20, n_flakes=5, flake_radius=3.0, seed=1)
        b = build_carbon_foam(box_size=20, n_flakes=5, flake_radius=3.0, seed=1)
        assert len(a) == len(b)
        np.testing.assert_allclose(a.get_positions(), b.get_positions())

    def test_min_distance_respected(self):
        atoms = build_carbon_foam(box_size=25, n_flakes=5, flake_radius=3.0,
                                  seed=0, min_distance=1.2)
        dmat = atoms.get_all_distances(mic=True)
        np.fill_diagonal(dmat, np.inf)
        assert dmat.min() >= HARD_MIN_DISTANCE

    def test_too_small_box_raises(self):
        with pytest.raises(ValueError):
            build_carbon_foam(box_size=5, n_flakes=3, flake_radius=3.0)


class TestASweptRibbon:
    """Bending a ribbon out of its own plane is an ISOMETRY, and bending
    it in plane is not. That is the whole of this class: the two options
    are not styling choices, they differ by whether any bond changes
    length at all."""

    @staticmethod
    def _bonds(atoms) -> np.ndarray:
        positions = atoms.get_positions()
        pairs = np.asarray(atoms.info["bonds"])
        return np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]],
                              axis=1)

    def test_out_of_plane_costs_no_strain(self):
        """Every atom across the width rides the frame's binormal, which
        a rotation-minimizing frame does not rotate, so each
        cross-section moves rigidly and the deformation is an isometry.

        What is left is 2 parts in 10,000 -- bonds 1.4197-1.4200 Å -- and
        that residual is **not** strain: the sweep interpolates between
        a finite number of frames along the path, so a bond spanning two
        of them picks up a chordal error. It shrinks with
        ``frame_samples`` and has no limit in the geometry, which is the
        opposite of what a strain does. An in-plane bend of the same
        ribbon is a hundred times larger and will not shrink at all."""
        atoms = build_nanoribbon(width=6, length=30, shape="zigzag",
                                 shape_turns=1.0, waviness=1.0,
                                 bend_plane="out_of_plane")
        lengths = self._bonds(atoms)
        assert np.allclose(lengths, CC_BOND, atol=1e-3)
        assert abs(lengths - CC_BOND).max() < 1e-3
        assert atoms.info["path_strain"] == 0.0

    def test_in_plane_spends_the_budget_and_buys_less_bend(self):
        """The comparison that makes the default worth having: same
        ribbon, same path, six times less excursion for bonds that are
        no longer ideal."""
        folded = build_nanoribbon(width=6, length=30, shape="zigzag",
                                  shape_turns=1.0, waviness=1.0,
                                  bend_plane="out_of_plane")
        bent = build_nanoribbon(width=6, length=30, shape="zigzag",
                                shape_turns=1.0, waviness=1.0,
                                bend_plane="in_plane")
        assert bent.info["path_strain"] > 0.0
        assert self._bonds(bent).max() > CC_BOND + 0.05
        assert folded.info["excursion"] > 4.0 * bent.info["excursion"]

    def test_a_swept_ribbon_is_finite_and_its_bonds_are_recomputed(self):
        """A straight ribbon is periodic along its length and its bond
        list names pairs across that seam. A bent one has no period, so
        carrying those bonds over would draw lines the length of the
        cell -- the mistake this package has now made in six places."""
        straight = build_nanoribbon(width=6, length=30)
        swept = build_nanoribbon(width=6, length=30, shape="zigzag",
                                 shape_turns=1.0, waviness=1.0)
        assert list(straight.get_pbc()) == [False, False, True]
        assert list(swept.get_pbc()) == [False, False, False]
        assert straight.info["periodic_axis"] == 2
        assert swept.info["periodic_axis"] is None
        assert self._bonds(swept).max() < 1.8

    def test_straight_leaves_the_periodic_ribbon_alone(self):
        atoms = build_nanoribbon(width=6, length=30, shape="straight")
        assert "swept" not in atoms.info
        assert list(atoms.get_pbc()) == [False, False, True]

    def test_a_fold_that_passes_through_itself_is_refused(self):
        """Nothing else limits an out-of-plane fold -- it costs no
        strain -- so the sheet meeting itself is the only limit there
        is, and it has to be the one that is checked."""
        with pytest.raises(ValueError, match="within"):
            build_nanoribbon(width=6, length=30, shape="zigzag",
                             shape_turns=2.0, waviness=1.0,
                             bend_plane="out_of_plane")

    def test_an_unknown_bend_plane_is_refused(self):
        with pytest.raises(ValueError, match="Unknown bend_plane"):
            build_nanoribbon(width=6, length=30, shape="zigzag",
                             bend_plane="sideways")

    def test_the_excursion_means_the_same_thing_in_both_modes(self):
        """An in-plane bend swings the ribbon inside its own plane, so
        its x span is the bend PLUS the width. Reporting that span would
        make the in-plane number look five times larger than the bend
        it describes."""
        bent = build_nanoribbon(width=6, length=30, shape="zigzag",
                                shape_turns=1.0, waviness=1.0,
                                bend_plane="in_plane")
        width = np.ptp(build_nanoribbon(width=6, length=30
                                        ).get_positions()[:, 0])
        assert bent.info["excursion"] < width

    def test_a_swept_ribbon_is_still_all_hexagons(self):
        atoms = build_nanoribbon(width=6, length=30, shape="zigzag",
                                 shape_turns=1.0, waviness=1.0)
        assert set(atoms.info["ring_counts"]) == {6}
        # An open sheet with two free edges and no period: a disc, so
        # chi is 1 where the straight ribbon's cylinder gives 0.
        assert atoms.info["euler_expected"] == 1
