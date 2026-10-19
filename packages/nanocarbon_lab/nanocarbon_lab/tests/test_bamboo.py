"""Bamboo tubes: graphitic cups stacked inside a tube.

The structural claim worth testing is what the cups ARE. They are
separate pieces at a van der Waals contact, not a covalent part of the
wall, because a sealed septum is not a manifold surface -- three sheets
meet along its rim.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from nanocarbon_lab.builders.bamboo import CUP_ORIENTATIONS, build_bamboo_cnt


def _pieces(atoms, kind):
    return [index
            for name, low, high in atoms.info["pieces"] if name == kind
            for index in range(low, high)]


class TestRefusals:
    def test_a_narrow_tube_holds_no_cup(self):
        """Measured: a (10,10) has radius 6.78 A, which leaves 3.38 A
        inside a 3.40 A graphitic clearance. Real bamboo tubes are
        multi-wall and tens of nm across, so this is the same kind of
        fact as the nanocoil's radius, not a default to taste."""
        with pytest.raises(ValueError, match="not enough for a cup"):
            build_bamboo_cnt(n=10, m=10, n_compartments=2)

    def test_only_the_clean_apex_is_allowed(self):
        """2 and 3 pentagons put a square or a triangle at the apex --
        sound topology with 1.34 / 1.23 A bonds, which is not carbon."""
        for count in (2, 3):
            with pytest.raises(ValueError, match="apex"):
                build_bamboo_cnt(cup_pentagons=count)

    def test_zero_compartments_is_a_plain_tube(self):
        with pytest.raises(ValueError, match="at least 1 compartment"):
            build_bamboo_cnt(n_compartments=0)

    def test_an_unknown_orientation_names_the_options(self):
        with pytest.raises(ValueError, match="|".join(CUP_ORIENTATIONS)):
            build_bamboo_cnt(orientation="sideways")


@pytest.mark.slow
class TestBuiltBamboo:
    @pytest.fixture(scope="class")
    def tube(self):
        return build_bamboo_cnt(n=16, m=16, n_compartments=3,
                                compartment_length=20.0)

    def test_one_pentagon_per_cup_and_nothing_else(self, tube):
        """The crystalline result: the wall is a rolled lattice so it
        contributes only hexagons, and each cup contributes exactly its
        one apex pentagon. No amorphous wall anywhere."""
        census = tube.info["ring_counts"]
        assert census.get(5) == tube.info["n_cups"]
        assert set(census) == {5, 6}

    def test_bonds_are_graphitic_under_minimum_image(self, tube):
        """The tube is PERIODIC along its axis and its bond list records
        pairs across that seam, so the span has to be taken under the
        minimum image. Assembled as finite it showed a 61.5 A 'bond'
        against a real 1.42 -- the fifth time that fact has bitten."""
        positions = tube.get_positions()
        bonds = np.asarray(tube.info["bonds"])
        delta = positions[bonds[:, 1]] - positions[bonds[:, 0]]
        period = float(tube.cell[2, 2])
        delta[:, 2] -= period * np.round(delta[:, 2] / period)
        lengths = np.linalg.norm(delta, axis=1)
        assert lengths.min() > 1.30
        assert lengths.max() < 1.60

    def test_it_stays_periodic_along_its_axis(self, tube):
        assert tuple(bool(flag) for flag in tube.pbc) == (False, False, True)
        assert tube.cell[2, 2] == pytest.approx(tube.info["axial_period"])

    def test_the_compartment_length_is_an_output(self, tube):
        """The lattice quantises the period, so what was asked and what
        was achieved are both recorded and they differ."""
        assert tube.info["compartment_length_requested"] == 20.0
        achieved = tube.info["compartment_length"]
        assert achieved == pytest.approx(
            tube.info["axial_period"] / tube.info["n_compartments"])

    def test_a_cup_touches_the_wall_and_is_not_bonded_to_it(self, tube):
        """The whole design in one measurement: the closest approach is
        a van der Waals contact, and no bond in the list joins the two
        pieces."""
        positions = tube.get_positions()
        wall = _pieces(tube, "tube")
        cups = _pieces(tube, "cup")
        gap, _ = cKDTree(positions[wall]).query(positions[cups])
        assert gap.min() > 3.0

        wall_set = set(wall)
        for first, second in tube.info["bonds"]:
            assert (first in wall_set) == (second in wall_set), (
                "a bond joins a cup to the wall, which is not what holds "
                "them together")
