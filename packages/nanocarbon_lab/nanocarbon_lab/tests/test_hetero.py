"""Tests for twisted bilayers and van der Waals stacks.

The commensurate cell is the whole point here, so most of these check
counts and angles against closed forms rather than against tolerances:
the cell holds exactly ``m^2 + mn + n^2`` primitive cells per layer, and
if the fill produces any other number the structure is wrong however
plausible it looks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from ase.neighborlist import neighbor_list

from nanocarbon_lab.hetero import (
    MAX_MISMATCH,
    available_layers,
    build_twisted_bilayer,
    build_vdw_stack,
    cells_per_layer,
    commensurate_series,
    get_layer,
    nearest_commensurate,
    twist_angle,
)
from nanocarbon_lab.validation.checks import run_basic_checks


class TestCommensurateAngles:
    def test_the_identity_pair_is_no_twist(self):
        assert twist_angle(1, 1) == pytest.approx(0.0, abs=1e-9)

    def test_the_textbook_angle(self):
        """(2,1) is 21.787 deg, the smallest non-trivial commensurate
        twist and the one every paper on the subject starts from."""
        assert twist_angle(2, 1) == pytest.approx(21.7868, abs=1e-4)

    def test_the_magic_angle(self):
        """(31,30) is 1.0845 deg -- where twisted bilayer graphene's
        moire bands flatten. Worth pinning: it is the reason `max_index`
        has to reach into the thirties to be useful."""
        m, n, angle, cells = nearest_commensurate(1.1, max_index=40)
        assert (m, n) == (31, 30)
        assert angle == pytest.approx(1.0845, abs=1e-3)
        assert cells == 2791

    def test_cells_grow_as_the_angle_shrinks(self):
        _, _, _, wide = nearest_commensurate(21.8, max_index=40)
        _, _, _, tight = nearest_commensurate(2.0, max_index=40)
        assert tight > wide * 100

    def test_the_series_is_sorted_and_coprime(self):
        series = commensurate_series(max_index=12)
        angles = [entry[2] for entry in series]
        assert angles == sorted(angles)
        for m, n, _, _ in series:
            assert math.gcd(m, n) == 1

    def test_cells_matches_the_closed_form(self):
        assert cells_per_layer(31, 30) == 31 * 31 + 31 * 30 + 30 * 30

    def test_bad_indices_are_rejected(self):
        with pytest.raises(ValueError, match="Need m > 0"):
            twist_angle(0, 0)
        with pytest.raises(ValueError, match="Need m > 0"):
            twist_angle(2, 5)


class TestLayers:
    def test_every_advertised_layer_resolves(self):
        for name in available_layers():
            layer = get_layer(name)
            assert layer.a > 0
            assert layer.n_sites >= 2

    def test_an_unknown_layer_names_the_alternatives(self):
        with pytest.raises(KeyError, match="graphene"):
            get_layer("unobtainium")

    def test_the_honeycomb_basis_gives_the_right_bond(self):
        """(1/3, 1/3) in the 60-degree cell, not (1/3, 2/3): the second
        is the 120-degree convention and puts the sites a/3 = 0.82 Å
        apart instead of a/sqrt(3) = 1.42."""
        layer = get_layer("graphene")
        a1 = np.array([layer.a, 0.0])
        a2 = np.array([layer.a * 0.5, layer.a * math.sqrt(3.0) / 2.0])
        _, u, v, _ = layer.basis[1]
        assert np.linalg.norm(u * a1 + v * a2) == pytest.approx(1.42, abs=1e-3)

    def test_a_dichalcogenide_layer_is_a_sandwich(self):
        layer = get_layer("MoS2")
        heights = sorted(site[3] for site in layer.basis)
        assert heights[0] < 0 < heights[2]
        assert layer.thickness == pytest.approx(heights[2] - heights[0])


class TestTwistedBilayer:
    def test_atom_count_is_the_commensurate_one(self):
        atoms = build_twisted_bilayer("graphene", target_angle=21.8)
        info = atoms.info
        assert len(atoms) == info["cells_per_layer"] * 4  # 2 sites x 2 layers
        assert info["n_bottom"] == info["n_top"]

    def test_the_lattice_survives_the_twist(self):
        """Every bond still 1.42 Å: a twist rotates a layer, it does not
        stretch one."""
        atoms = build_twisted_bilayer("graphene", target_angle=13.2)
        _, _, distance = neighbor_list("ijd", atoms, cutoff=1.8)
        assert distance.min() == pytest.approx(1.42, abs=1e-3)
        assert distance.max() == pytest.approx(1.42, abs=1e-3)

    def test_no_atoms_overlap(self):
        for angle in (21.8, 13.2, 7.3):
            atoms = build_twisted_bilayer("graphene", target_angle=angle)
            assert not run_basic_checks(atoms).errors

    def test_the_achieved_angle_is_reported_not_the_request(self):
        atoms = build_twisted_bilayer("graphene", target_angle=7.0)
        assert atoms.info["requested_angle"] == 7.0
        assert atoms.info["twist_angle"] != pytest.approx(7.0, abs=1e-6)

    def test_the_layers_are_a_gap_apart(self):
        gap = 3.6
        atoms = build_twisted_bilayer("graphene", target_angle=21.8, gap=gap)
        z = atoms.get_positions()[:, 2]
        assert np.ptp(z) == pytest.approx(gap, abs=1e-6)

    def test_a_dichalcogenide_twists_too(self):
        atoms = build_twisted_bilayer("MoS2", target_angle=13.2)
        assert len(atoms) == atoms.info["cells_per_layer"] * 6
        _, _, distance = neighbor_list("ijd", atoms, cutoff=2.6)
        assert distance.min() == pytest.approx(2.404, abs=1e-3)

    def test_a_heterobilayer_records_its_strain(self):
        atoms = build_twisted_bilayer("graphene", top_layer="hBN",
                                      target_angle=7.3)
        assert atoms.info["lattice_mismatch"] == pytest.approx(0.0179, abs=1e-3)
        assert atoms.info["imposed_strain"] != 0.0
        assert set(atoms.get_chemical_symbols()) == {"C", "B", "N"}

    def test_too_large_a_mismatch_is_refused(self):
        """Graphene and MoS2 differ by 28%. A common cell exists on paper
        and describes nothing real."""
        with pytest.raises(ValueError, match="over the"):
            build_twisted_bilayer("graphene", top_layer="MoS2",
                                  target_angle=7.3)

    def test_a_non_positive_gap_is_rejected(self):
        with pytest.raises(ValueError, match="gap"):
            build_twisted_bilayer("graphene", target_angle=21.8, gap=0.0)

    @pytest.mark.slow
    def test_the_magic_angle_builds(self):
        """11 164 atoms. Slow only because it is large -- the point is
        that both the fill and the validation scale to it, which they did
        not while either was quadratic."""
        atoms = build_twisted_bilayer("graphene", target_angle=1.1,
                                      max_index=40)
        assert len(atoms) == 11164
        assert atoms.info["moire_period"] == pytest.approx(129.9, rel=0.01)
        assert not run_basic_checks(atoms).errors


class TestVdwStack:
    def test_layers_stack_in_order(self):
        atoms = build_vdw_stack(["graphene", "hBN"])
        z = atoms.get_positions()[:, 2]
        symbols = np.array(atoms.get_chemical_symbols())
        assert z[symbols == "C"].mean() < z[symbols == "B"].mean()

    def test_three_layers(self):
        atoms = build_vdw_stack(["graphene", "hBN", "graphene"], nx=2, ny=2)
        assert len(atoms) == 2 * 3 * 4  # 2 sites x 3 layers x 4 cells
        assert not run_basic_checks(atoms).errors

    def test_a_single_layer_is_not_a_stack(self):
        with pytest.raises(ValueError, match="at least two"):
            build_vdw_stack(["graphene"])

    def test_mismatch_is_bounded(self):
        with pytest.raises(ValueError, match="over the"):
            build_vdw_stack(["graphene", "MoS2"])

    def test_the_bound_is_the_documented_one(self):
        assert MAX_MISMATCH == pytest.approx(0.05)
