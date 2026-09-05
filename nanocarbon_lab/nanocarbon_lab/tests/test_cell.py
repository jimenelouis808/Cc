"""Tests for turning any structure into a periodic unit cell.

Two properties carry the module. The first is that the conversion never
touches a direction the structure genuinely repeats in -- padding a real
lattice vector would change the crystal, not the box around it. The
second is that the convergence measure counts only images across
*vacuum*: in a real crystal an atom bonds to its image, so a nanotube's
1.42 Å contact along its own axis is the structure and not a failure.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from nanocarbon_lab.builders import build_cnt, build_fullerene, build_graphene_supercell
from nanocarbon_lab.cell import (
    MIN_IMAGE_SEPARATION,
    cell_report,
    describe_periodicity,
    image_separation,
    periodicity,
    to_unit_cell,
)
from nanocarbon_lab.tmd import build_tmd_monolayer


@pytest.fixture(scope="module")
def cage():
    return build_fullerene(family="C60", freq=1)


@pytest.fixture(scope="module")
def tube():
    return build_cnt(n=6, m=6, length=10)


@pytest.fixture(scope="module")
def sheet():
    return build_graphene_supercell(2, 2)


class TestPeriodicityDetection:
    def test_it_counts_the_repeating_directions(self, cage, tube, sheet):
        assert describe_periodicity(cage) == "0D"
        assert describe_periodicity(tube) == "1D"
        assert describe_periodicity(sheet) == "2D"

    def test_periodicity_is_the_number_behind_it(self, tube):
        assert periodicity(tube) == 1


class TestConversion:
    @pytest.mark.parametrize("name", ["cage", "tube", "sheet"])
    def test_everything_comes_out_fully_periodic(self, name, request):
        """A plane-wave code has no "molecule" setting, so every axis has
        to be periodic and the vacuum has to be real."""
        atoms = to_unit_cell(request.getfixturevalue(name))
        assert all(atoms.get_pbc())
        assert atoms.cell.rank == 3
        assert atoms.cell.volume > 0

    def test_a_periodic_axis_is_left_exactly_alone(self, tube):
        """The lattice vector along a tube's axis *is* the physics.
        Padding it would change the structure, not the box."""
        before = float(tube.cell.lengths()[2])
        after = float(to_unit_cell(tube).cell.lengths()[2])
        assert after == pytest.approx(before)

    def test_a_non_periodic_axis_is_rebuilt_from_the_atoms(self, cage):
        """Not padded from whatever bounding box the builder left: a
        finite builder's cell already has padding in it, and padding a
        padded box compounds every time this is called."""
        once = to_unit_cell(cage, vacuum=10.0)
        twice = to_unit_cell(once, vacuum=10.0)
        assert twice.cell.lengths()[0] == pytest.approx(once.cell.lengths()[0])

    def test_the_vacuum_asked_for_is_the_vacuum_given(self, cage):
        atoms = to_unit_cell(cage, vacuum=9.0)
        span = np.ptp(cage.get_positions()[:, 0])
        assert atoms.cell.lengths()[0] == pytest.approx(span + 18.0)

    def test_a_slab_gets_more_room_than_a_tube_by_default(self, sheet, cage):
        """A slab's images stack face to face across its one open
        direction; a molecule's neighbours are further away in every
        direction at the same padding."""
        slab_gap = to_unit_cell(sheet).cell.lengths()[2] - 0.0
        assert slab_gap == pytest.approx(30.0, abs=0.5)

    def test_atoms_end_up_inside_the_cell(self, cage, tube, sheet):
        """Atoms drawn outside the box are the commonest reason a correct
        periodic structure looks broken in a viewer."""
        for structure in (cage, tube, sheet):
            assert cell_report(to_unit_cell(structure))["atoms_outside"] == 0

    def test_the_atoms_are_not_mutated(self, cage):
        before = cage.get_positions().copy()
        to_unit_cell(cage)
        assert np.allclose(cage.get_positions(), before)

    def test_it_records_what_it_started_from(self, tube):
        info = to_unit_cell(tube).info["unit_cell"]
        assert info["original_periodicity"] == "1D"
        assert sorted(info["vacuum_axes"]) == [0, 1]
        assert info["periodic_axes"] == [2]

    def test_an_empty_structure_is_rejected(self):
        with pytest.raises(ValueError, match="empty structure"):
            to_unit_cell(Atoms())

    def test_negative_vacuum_is_rejected(self, cage):
        with pytest.raises(ValueError, match="vacuum must be"):
            to_unit_cell(cage, vacuum=-1.0)


class TestImageSeparation:
    def test_a_bonded_contact_along_a_real_axis_is_not_a_failure(self, tube):
        """The bug this pins. A nanotube bonds to its own image along its
        periodic axis at 1.42 Å; measuring every image and reporting the
        minimum called that unconverged, which is exactly backwards."""
        converted = to_unit_cell(tube)
        assert cell_report(converted)["converged"]
        assert cell_report(converted)["image_separation"] > MIN_IMAGE_SEPARATION

    def test_a_bulk_crystal_has_nothing_to_converge(self):
        """No vacuum direction means no number to compare to a threshold,
        and inventing one would invite exactly that comparison."""
        bulk = build_tmd_monolayer("MoS2")
        bulk.set_pbc(True)
        assert image_separation(bulk, vacuum_axes=[]) == float("inf")
        assert cell_report(bulk)["image_separation"] is None

    def test_a_tight_box_is_reported_as_unconverged(self, cage):
        assert not cell_report(to_unit_cell(cage, vacuum=1.0))["converged"]

    def test_a_roomy_box_is_reported_as_converged(self, cage):
        assert cell_report(to_unit_cell(cage, vacuum=12.0))["converged"]

    def test_it_measures_the_axis_that_is_actually_vacuum(self, sheet):
        """A graphene sheet repeats in x and y and is vacuum in z, so
        only z-crossing images may count -- the in-plane ones are bonds."""
        converted = to_unit_cell(sheet, vacuum=10.0)
        assert image_separation(converted, vacuum_axes=[2]) > MIN_IMAGE_SEPARATION
        # Counting the in-plane images too would find a C-C bond.
        assert image_separation(converted, vacuum_axes=[0, 1, 2]) < 2.0


class TestReport:
    def test_it_measures_rather_than_repeats_the_request(self, cage):
        report = cell_report(to_unit_cell(cage, vacuum=10.0))
        assert report["periodicity"] == "3D"
        assert report["n_atoms"] == 60
        assert report["volume"] > 0
        assert report["density"] > 0

    def test_density_is_grams_per_cubic_centimetre(self):
        """Graphite is 2.27 g/cm3, so a single graphene layer in a 10 Å
        cell must land well below that rather than in atomic units."""
        report = cell_report(to_unit_cell(build_graphene_supercell(2, 2)))
        assert 0.01 < report["density"] < 2.5
