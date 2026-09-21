"""Where the disclinations sit, against surfaces whose answer is known.

The coil papers' claim -- pentagons in positive curvature, heptagons in
negative -- generalised off the coil. These tests are mostly calibration:
a sphere, a plane and a flat pentagon-heptagon lattice have answers that
do not depend on this code being right.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nanocarbon_lab.analyse.curvature import disclination_check, local_curvature
from nanocarbon_lab.builders.fullerene import build_fullerene
from nanocarbon_lab.builders.haeckelite import build_haeckelite
from nanocarbon_lab.builders.junction import build_junction


def _quiet(make):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make()


@pytest.fixture(scope="module")
def flat_lattice():
    """R5,7: pentagons and heptagons, and perfectly flat."""
    return _quiet(lambda: build_haeckelite(nx=4, ny=4, pattern="r57"))


@pytest.fixture(scope="module")
def cage():
    return _quiet(lambda: build_fullerene(family="C60"))


@pytest.fixture(scope="module")
def junction():
    return _quiet(lambda: build_junction(kind="Y", tube_radius=8.0,
                                         arm_length=20.0, seed=0))


class TestItIsNotCircular:
    """The objection this module has to answer.

    The patch fitted around a pentagon is dominated by that pentagon, so
    does the fit just recover the ring size under another name? If it
    did, a FLAT lattice of pentagons and heptagons would still report
    them as curved -- and the whole measure would be a census wearing a
    disguise.
    """

    def test_a_flat_lattice_of_disclinations_reads_exactly_zero(
            self, flat_lattice):
        box = np.array([flat_lattice.cell[0][0],
                        flat_lattice.cell[1][1], 0.0])
        check = disclination_check(flat_lattice.get_positions(),
                                   flat_lattice.info["rings"],
                                   flat_lattice.info["bonds"], box)
        assert set(check["sizes"]) == {5, 7}
        for size in (5, 7):
            assert check["sizes"][size]["count"] > 0
            assert check["sizes"][size]["mean_sign"] == pytest.approx(0.0)
            assert check["sizes"][size]["median"] == pytest.approx(0.0)

    def test_so_a_flat_lattice_scores_zero_agreement(self, flat_lattice):
        """Not a failure -- the correct answer. There is no curvature for
        a disclination to be on the right side of."""
        box = np.array([flat_lattice.cell[0][0],
                        flat_lattice.cell[1][1], 0.0])
        check = disclination_check(flat_lattice.get_positions(),
                                   flat_lattice.info["rings"],
                                   flat_lattice.info["bonds"], box)
        assert check["agreement"] == 0.0


class TestTheKnownSurfaces:
    def test_a_sphere_is_positive_everywhere(self, cage):
        curvature = local_curvature(cage.get_positions(), cage.info["bonds"])
        assert np.all(curvature > 0)

    def test_even_its_hexagons(self, cage):
        """C60 is a sphere, so its hexagons are curved too. A measure
        that called them flat would be reading the census."""
        check = disclination_check(cage.get_positions(), cage.info["rings"],
                                   cage.info["bonds"])
        assert check["sizes"][5]["mean_sign"] == pytest.approx(1.0)
        assert check["sizes"][6]["mean_sign"] == pytest.approx(1.0)


class TestTheClaimOnCurvedCarbon:
    def test_a_junction_puts_them_where_the_papers_say(self, junction):
        check = junction.info["disclination_check"]
        assert check["sizes"][5]["mean_sign"] > 0.5
        assert check["sizes"][7]["mean_sign"] < -0.5
        assert check["agreement"] > 0.8

    def test_its_arms_are_cylinders_so_its_hexagons_are_flat(self, junction):
        """A cylinder has zero Gaussian curvature, and most of a
        junction's hexagons are on its arms."""
        assert abs(junction.info["disclination_check"]["sizes"][6]["mean_sign"]) < 0.3

    def test_hexagons_are_not_scored(self, junction):
        """A hexagon is the flat case and has no side to be on, so
        counting it as a pass would inflate every score."""
        check = junction.info["disclination_check"]
        non_hex = sum(row["count"] for size, row in check["sizes"].items()
                      if size != 6)
        assert check["n_scored"] == non_hex


class TestEveryImplicitBuilderRecordsIt:
    """It goes in through ``junction._finish``, so junction, schwarzite,
    network and supernetwork all carry it without each having to ask."""

    def test_the_junction_does(self, junction):
        assert "disclination_check" in junction.info

    def test_the_supernetwork_does(self):
        from nanocarbon_lab.builders.supernetwork import build_supernetwork

        atoms = _quiet(lambda: build_supernetwork(
            "super-square", scale=34.0, tube_radius=5.0, blend=4.0,
            grid_resolution=64))
        check = atoms.info["disclination_check"]
        assert check["agreement"] > 0.7
        assert check["sizes"][7]["mean_sign"] < 0
