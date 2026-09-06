"""Tests for the structure builders."""

from __future__ import annotations

import numpy as np
import pytest

from carbonforge.builders import (
    build_cnt,
    build_graphene,
    build_graphene_supercell,
    build_nanocoil,
    build_nanoribbon,
    build_carbon_foam,
)
from carbonforge.utils.constants import CC_BOND, HARD_MIN_DISTANCE


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
        # ASE lays the ribbon in x-z with the axis along z.
        assert list(atoms.get_pbc()) == [False, False, True]

    @pytest.mark.parametrize("edge", ["zigzag", "armchair"])
    def test_periodic_axis_is_the_one_atoms_fill(self, edge):
        """The declared periodic axis must be the one the atoms tile.

        Regression test. The builder used to declare the ribbon periodic
        along y, which is pure vacuum: the band path then ran through empty
        space, the k-mesh sampled the vacuum direction while treating the
        real one as isolated, and the vacuum check looked at the wrong axes.
        Every ribbon export was wrong while looking perfectly well-formed.
        """
        atoms = build_nanoribbon(6, 3, edge=edge)
        positions = atoms.get_positions()
        cell = np.diag(np.array(atoms.cell))
        axis = int(np.argmax(atoms.get_pbc()))
        span = float(np.ptp(positions[:, axis]))
        # Along a periodic axis the atoms nearly fill the cell; along a
        # padded one they occupy far less than half of it.
        assert span > 0.5 * cell[axis], (
            f"eje {axis} declarado periódico pero los átomos solo ocupan "
            f"{span:.2f} de {cell[axis]:.2f} Å"
        )

    def test_ribbon_passes_validation(self):
        """Would have caught the wrong-axis bug: vacuum was checked on z."""
        from carbonforge.validation import run_basic_checks

        report = run_basic_checks(build_nanoribbon(6, 3, edge="zigzag"))
        assert report.ok, report.summary()

    def test_band_path_follows_periodic_axis(self):
        from carbonforge.calculations import suggest_band_path

        atoms = build_nanoribbon(6, 3, edge="zigzag")
        spec = suggest_band_path(atoms)
        axis = int(np.argmax(atoms.get_pbc()))
        assert abs(spec.points[-1][axis]) == pytest.approx(0.5)

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


# Builders are listed here with a small, fast set of arguments so the
# invariants below run over every structure type the package can produce.
ALL_BUILDERS = [
    ("cnt_armchair", lambda: build_cnt(6, 6, length=8)),
    ("cnt_zigzag", lambda: build_cnt(10, 0, length=8)),
    ("cnt_chiral", lambda: build_cnt(6, 3, length=8)),
    ("graphene_primitive", build_graphene),
    ("graphene_supercell", lambda: build_graphene_supercell(3, 3)),
    ("ribbon_zigzag", lambda: build_nanoribbon(6, 3, edge="zigzag")),
    ("ribbon_armchair", lambda: build_nanoribbon(6, 3, edge="armchair")),
    ("ribbon_passivated",
     lambda: build_nanoribbon(6, 3, edge="zigzag", passivate=True)),
    ("nanocoil", lambda: build_nanocoil(n=6, m=6, coil_radius=25.0,
                                        pitch=12.0, n_turns=0.5)),
    ("foam", lambda: build_carbon_foam(box_size=20, n_flakes=4,
                                       flake_radius=3.0, seed=0)),
]


class TestBuilderInvariants:
    """Properties every builder must satisfy, whatever it produces.

    These exist because a nanoribbon shipped for weeks declaring the wrong
    periodic axis. Per-builder tests did not catch it; an invariant applied
    to all of them would have.
    """

    @pytest.mark.parametrize("name,builder", ALL_BUILDERS,
                             ids=[n for n, _ in ALL_BUILDERS])
    def test_passes_validation(self, name, builder):
        from carbonforge.validation import run_basic_checks

        report = run_basic_checks(builder())
        assert report.ok, f"{name}:\n{report.summary()}"

    @pytest.mark.parametrize("name,builder", ALL_BUILDERS,
                             ids=[n for n, _ in ALL_BUILDERS])
    def test_periodic_axes_are_real(self, name, builder):
        """No axis may be declared periodic if the atoms are flat along it."""
        from carbonforge.validation import check_periodicity_coherence

        report = check_periodicity_coherence(builder())
        assert report.ok, f"{name}:\n{report.summary()}"

    @pytest.mark.parametrize("name,builder", ALL_BUILDERS,
                             ids=[n for n, _ in ALL_BUILDERS])
    def test_no_atomic_overlap(self, name, builder):
        atoms = builder()
        distances = atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() > HARD_MIN_DISTANCE, name

    @pytest.mark.parametrize("name,builder", ALL_BUILDERS,
                             ids=[n for n, _ in ALL_BUILDERS])
    def test_records_its_type(self, name, builder):
        assert "structure_type" in builder().info, name

    @pytest.mark.parametrize("name,builder", ALL_BUILDERS,
                             ids=[n for n, _ in ALL_BUILDERS])
    def test_band_path_avoids_vacuum(self, name, builder):
        """Every high-symmetry point must lie in a periodic direction."""
        from carbonforge.calculations import suggest_band_path

        atoms = builder()
        spec = suggest_band_path(atoms)
        pbc = atoms.get_pbc()
        for point in spec.points:
            for axis in range(3):
                if not pbc[axis]:
                    assert abs(point[axis]) < 1e-8, (
                        f"{name}: el camino de bandas se mete por el eje "
                        f"{'xyz'[axis]}, que no es periódico"
                    )
