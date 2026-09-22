"""sp2/sp3 character, measured from the angle sum at each carbon.

The controls are what make this trustworthy, and the important one is
negative: a flat haeckelite is nothing but pentagons and heptagons and
must still read exactly sp2. Curvature is not hybridisation, and a
measure that confused them would be measuring the ring census twice.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from nanocarbon_lab.analyse.hybridisation import (
    PLANAR_SUM,
    TETRAHEDRAL_SUM,
    describe_hybridisation,
    hybridisation_report,
    sp3_character,
)


def _report(atoms):
    return hybridisation_report(atoms.positions, atoms.info.get("bonds", []),
                                np.asarray(atoms.cell))


class TestTheScale:
    def test_a_planar_carbon_reads_zero(self):
        """Three bonds at 120 deg in a plane: the definition of sp2."""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.42, 0.0, 0.0],
            [-0.71, 1.2297, 0.0],
            [-0.71, -1.2297, 0.0],
        ])
        value = sp3_character(positions, [(0, 1), (0, 2), (0, 3)])[0]
        assert value == pytest.approx(0.0, abs=1e-6)

    def test_a_tetrahedral_carbon_reads_one(self):
        """Three of methane's four bonds, the fourth pointing away."""
        a = 1.0 / math.sqrt(3.0)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [a, a, a], [a, -a, -a], [-a, a, -a],
        ]) * 1.54
        value = sp3_character(positions, [(0, 1), (0, 2), (0, 3)])[0]
        assert value == pytest.approx(1.0, rel=2e-3)

    def test_a_four_coordinate_carbon_is_sp3_without_measuring(self):
        positions = np.zeros((5, 3))
        positions[1:] = np.eye(3).tolist() + [[-1.0, -1.0, -1.0]]
        value = sp3_character(positions, [(0, 1), (0, 2), (0, 3), (0, 4)])[0]
        assert value == 1.0

    def test_an_atom_with_too_few_bonds_is_not_called_sp2(self):
        """nan, not 0. An edge atom has no angle sum, and reporting 0
        would quietly count it as perfectly planar."""
        positions = np.array([[0.0, 0.0, 0.0], [1.42, 0.0, 0.0]])
        assert np.isnan(sp3_character(positions, [(0, 1)])[0])


class TestAgainstKnownStructures:
    """The scale is only worth having if the known cases land where they
    should, and they span the range: flat, curved, and crushed."""

    def test_a_flat_haeckelite_is_exactly_sp2(self):
        """The control that matters. All pentagons and heptagons, zero
        sp3 -- curvature is not hybridisation."""
        from nanocarbon_lab.builders.haeckelite import build_haeckelite

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            atoms = build_haeckelite(nx=4, ny=4, pattern="r57")
        report = _report(atoms)
        assert report["angle_sum_max"] == pytest.approx(PLANAR_SUM, abs=0.5)
        assert report["sp3_fraction"] == 0.0

    def test_a_narrow_tube_is_more_pyramidal_than_a_wide_one(self):
        """Ordering, not absolute values: curvature pyramidalises, and a
        (4,4) is curved more tightly than a (10,10)."""
        from nanocarbon_lab.builders.cnt import build_cnt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            narrow = _report(build_cnt(n=4, m=4, length=15.0))
            wide = _report(build_cnt(n=10, m=10, length=15.0))
        assert narrow["mean_character"] > wide["mean_character"]
        assert wide["mean_character"] < 0.1

    def test_c60_sits_between_graphene_and_diamond(self):
        from nanocarbon_lab.builders.fullerene import build_fullerene

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = _report(build_fullerene(freq=1))
        assert 0.2 < report["mean_character"] < 0.6
        assert TETRAHEDRAL_SUM < report["angle_sum_min"] < PLANAR_SUM

    def test_the_heptanene_embedding_is_past_tetrahedral(self):
        """Why it is refused, as a number rather than an adjective: the
        Klein quartic's cubic-torus embedding bends its carbons further
        than tetrahedral, which no sp2 or sp3 carbon does."""
        from nanocarbon_lab.builders.heptanene import build_heptanene

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = _report(build_heptanene(strict=False))
        assert report["max_character"] > 1.0
        assert report["sp3_fraction"] > 0.5


class TestDescription:
    def test_it_names_the_scale_it_used(self):
        from nanocarbon_lab.builders.cnt import build_cnt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            text = describe_hybridisation(_report(build_cnt(n=5, m=5,
                                                            length=10.0)))
        assert "360" in text and "328.4" in text

    def test_nothing_to_measure_says_so(self):
        assert "nothing" in describe_hybridisation(
            hybridisation_report(np.zeros((2, 3)), [(0, 1)]))
