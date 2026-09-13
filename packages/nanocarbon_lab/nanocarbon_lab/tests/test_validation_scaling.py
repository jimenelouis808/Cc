"""Validation must reach every family, and must scale.

Two faults lived here at once and reinforced each other, so both are
pinned:

1. **Element coverage.** ``COVALENT_RADII`` held only C, N, B, S, P, H
   and O. Every other element fell back to a 1.80 Å cutoff, so a 2.404 Å
   Mo-S bond was not a bond at all: every dichalcogenide validated as
   "isolated atoms" and both exporters refused it. The entire ``tmd``
   package could not reach Quantum ESPRESSO or LAMMPS.

2. **Scaling.** Bond finding and the closest-pair check each built the
   full N x N distance matrix -- 24 s and 79 MB at 3136 atoms, hours and
   gigabytes past ten thousand. Validation runs on the path of every
   export, so that quadratic was in front of every structure produced.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from nanocarbon_lab.builders import build_graphene_supercell
from nanocarbon_lab.tmd import (
    build_tmd_bulk,
    build_tmd_layers,
    build_tmd_nanotube,
)
from nanocarbon_lab.utils.constants import (
    BOND_CUTOFF_OVERRIDE,
    COVALENT_RADII,
    MAX_COORDINATION,
)
from nanocarbon_lab.utils.geometry import guess_bonds
from nanocarbon_lab.validation.checks import run_basic_checks


class TestElementCoverage:
    @pytest.mark.parametrize(
        "element", ["Mo", "W", "S", "Se", "Te", "Nb", "Ta", "Ti", "Zr", "Hf"])
    def test_dichalcogenide_elements_have_radii(self, element):
        assert element in COVALENT_RADII

    def test_a_metal_bonds_to_its_chalcogen(self):
        """The 2.404 Å Mo-S bond has to be inside the cutoff, or the
        whole layer reads as isolated atoms."""
        atoms = build_tmd_layers("MoS2", n_layers=1, nx=2, ny=2)
        assert len(guess_bonds(atoms)) > 0

    def test_a_metal_does_not_bond_to_its_own_lattice_neighbours(self):
        """Two metallic radii overshoot the lattice constant -- Mo + Mo +
        0.30 is 3.38 Å against MoS2's 3.16 -- so without an override every
        metal picks up its six in-plane neighbours and reads as
        12-coordinate."""
        atoms = build_tmd_layers("MoS2", n_layers=1, nx=3, ny=3)
        symbols = np.array(atoms.get_chemical_symbols())
        metal_metal = [
            (i, j) for i, j, _ in guess_bonds(atoms)
            if symbols[i] == "Mo" and symbols[j] == "Mo"
        ]
        assert metal_metal == []

    def test_mixed_metal_pairs_are_covered_too(self):
        """An alloy puts Mo next to W at the same lattice spacing, so the
        override has to span pairs, not just same-element ones."""
        assert ("Mo", "W") in BOND_CUTOFF_OVERRIDE
        assert ("W", "Mo") in BOND_CUTOFF_OVERRIDE

    def test_the_coordination_ceiling_is_per_element(self):
        """A six-coordinate metal is correct; judging it by carbon's
        limit of 4 rejected every structure in the tmd package."""
        assert MAX_COORDINATION["C"] == 4
        assert MAX_COORDINATION["Mo"] >= 6


class TestEveryFamilyValidates:
    @pytest.mark.parametrize(
        ("name", "factory"),
        [
            ("monolayer", lambda: build_tmd_layers("MoS2", n_layers=1,
                                                   nx=3, ny=3)),
            ("bilayer", lambda: build_tmd_layers("MoS2", n_layers=2,
                                                 nx=2, ny=2)),
            ("1T-prime", lambda: build_tmd_layers("MoS2", n_layers=1,
                                                  phase="1T'", nx=2, ny=2)),
            ("bulk", lambda: build_tmd_bulk("MoS2")),
            ("nanotube", lambda: build_tmd_nanotube("MoS2", n=30, m=0)),
        ],
    )
    def test_it_passes_validation(self, name, factory):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            atoms = factory()
        assert not run_basic_checks(atoms).errors, name

    def test_the_1t_prime_dimer_is_allowed(self):
        """1T' is defined by a 2.8 Å metal-metal dimer, so its metals are
        seven-coordinate: six chalcogens and one partner. A ceiling of
        six would reject the phase for being itself."""
        atoms = build_tmd_layers("MoS2", n_layers=1, phase="1T'", nx=2, ny=2)
        assert not run_basic_checks(atoms).errors


class TestScaling:
    def test_bond_finding_is_not_quadratic(self):
        """Timing is a blunt instrument, but the gap here is three orders
        of magnitude: the old matrix took 24 s at 3136 atoms where the
        neighbour list takes 0.2 s, and quadrupling the atoms must not
        quadruple-square the time."""
        small = build_graphene_supercell(10, 10)
        large = build_graphene_supercell(20, 20)
        assert len(large) == 4 * len(small)

        start = time.perf_counter()
        guess_bonds(small)
        small_time = time.perf_counter() - start

        start = time.perf_counter()
        guess_bonds(large)
        large_time = time.perf_counter() - start

        # Linear would be 4x, quadratic 16x. Allow generous slack for a
        # loaded machine and still catch a return to the matrix.
        assert large_time < max(0.05, small_time * 10.0)

    def test_a_large_structure_validates_in_reasonable_time(self):
        """3200 atoms took 24 s through the old path, and the memory for
        the matrix grew as the square on top of that."""
        atoms = build_graphene_supercell(30, 30)
        assert len(atoms) > 3000
        start = time.perf_counter()
        report = run_basic_checks(atoms)
        assert time.perf_counter() - start < 20.0
        assert not report.errors

    def test_the_closest_pair_is_still_found(self):
        """The neighbour list must not miss a genuine overlap just
        because it stopped looking at a cutoff."""
        atoms = build_graphene_supercell(3, 3)
        positions = atoms.get_positions()
        positions[1] = positions[0] + np.array([0.4, 0.0, 0.0])
        atoms.set_positions(positions)
        report = run_basic_checks(atoms)
        assert report.info["min_interatomic_distance"] == pytest.approx(
            0.4, abs=1e-6)
        assert any("apart" in message for message in report.errors)
