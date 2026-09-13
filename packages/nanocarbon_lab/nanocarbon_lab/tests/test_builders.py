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
        assert list(atoms.get_pbc()) == [False, True, False]

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
