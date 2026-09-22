"""Topological coordinates: positions recovered from the bonds alone.

The test that matters is the round trip. This package can already build
a toroidal polyhex and a C60 *exactly*, so their coordinates can be
thrown away, rebuilt from the adjacency matrix and compared against what
was discarded -- which is a check with an answer, not a plausibility
argument.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from ase import Atoms

from nanocarbon_lab.analyse.topological import (
    bilobal_pairs,
    bilobal_spectrum,
    is_bilobal,
    spherical_coordinates,
    toroidal_coordinates,
)
from nanocarbon_lab.builders.fullerene import build_fullerene
from nanocarbon_lab.builders.toroid import build_polyhex_toroid
from nanocarbon_lab.utils.geometry import guess_bonds


def _quiet(make):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make()


@pytest.fixture(scope="module")
def ring():
    return _quiet(lambda: build_polyhex_toroid(n=5, m=5, periods=60))


@pytest.fixture(scope="module")
def cage():
    return _quiet(lambda: build_fullerene(family="C60"))


def _recovered_graph(points, bond=1.42):
    """The bond graph the rebuilt coordinates imply, by distance."""
    atoms = Atoms(symbols=["C"] * len(points), positions=points, pbc=False)
    atoms.set_cell(points.max(axis=0) - points.min(axis=0) + 20.0)
    atoms.center()
    return {tuple(sorted((int(i), int(j)))) for i, j, _ in guess_bonds(atoms)}


class TestTheTorusRoundTrip:
    """Four bi-lobal eigenvectors, and the coordinates come back."""

    def test_the_graph_is_recovered_exactly(self, ring):
        """The strongest statement available: place the atoms from the
        adjacency alone, then ask what bonds the *placement* implies.
        They must be the bonds that went in."""
        bonds = [tuple(b) for b in ring.info["bonds"]]
        points = toroidal_coordinates(bonds, len(ring), bond=1.42)
        assert _recovered_graph(points) == {tuple(sorted(b)) for b in bonds}

    def test_the_bond_lengths_match_the_original(self, ring):
        bonds = [tuple(b) for b in ring.info["bonds"]]
        original = ring.get_positions()
        points = toroidal_coordinates(bonds, len(ring), bond=1.42)
        first = np.array([i for i, _ in bonds])
        second = np.array([j for _, j in bonds])
        was = np.linalg.norm(original[second] - original[first], axis=1)
        now = np.linalg.norm(points[second] - points[first], axis=1)
        assert now.mean() == pytest.approx(was.mean(), abs=0.01)
        assert now.std() == pytest.approx(was.std(), abs=0.01)

    def test_it_really_is_a_torus(self, ring):
        """Hollow in the middle: no atom near the axis."""
        bonds = [tuple(b) for b in ring.info["bonds"]]
        points = toroidal_coordinates(bonds, len(ring), bond=1.42)
        points = points - points.mean(axis=0)
        radial = np.hypot(points[:, 0], points[:, 1])
        assert radial.min() > 0.4 * radial.max()

    def test_three_eigenvectors_are_not_enough(self, ring):
        """Graovac et al.'s result, and the reason the torus needs its
        own function: the spherical construction flattens it."""
        bonds = [tuple(b) for b in ring.info["bonds"]]
        flat = spherical_coordinates(bonds, len(ring), bond=1.42)
        spread = flat.max(axis=0) - flat.min(axis=0)
        assert spread.min() / spread.max() < 0.2


class TestTheSphereRoundTrip:
    def test_the_graph_is_recovered_exactly(self, cage):
        bonds = [tuple(b[:2]) for b in cage.info["bonds"]]
        points = spherical_coordinates(bonds, len(cage), bond=1.42)
        assert _recovered_graph(points) == {tuple(sorted(b)) for b in bonds}

    def test_every_atom_lands_on_one_sphere(self, cage):
        bonds = [tuple(b[:2]) for b in cage.info["bonds"]]
        points = spherical_coordinates(bonds, len(cage), bond=1.42)
        points = points - points.mean(axis=0)
        radius = np.linalg.norm(points, axis=1)
        assert radius.std() < 1e-6

    def test_the_radius_matches_the_built_cage(self, cage):
        """3.52 Å either way. Not relaxed -- placed -- so the bonds come
        out less uniform than the builder's, which is the known character
        of the method rather than a fault."""
        bonds = [tuple(b[:2]) for b in cage.info["bonds"]]
        points = spherical_coordinates(bonds, len(cage), bond=1.42)
        original = cage.get_positions() - cage.get_positions().mean(axis=0)
        assert (np.linalg.norm(points, axis=1).mean()
                == pytest.approx(np.linalg.norm(original, axis=1).mean(),
                                 abs=0.05))


class TestTheBilobalTest:
    def test_the_constant_vector_is_not_bilobal(self, cage):
        """One lobe, not two."""
        bonds = [tuple(b[:2]) for b in cage.info["bonds"]]
        assert not is_bilobal(np.ones(len(cage)), bonds, len(cage))

    def test_the_torus_pairs_are_degenerate_and_the_first_two(self, ring):
        """Measured, and the reason the recipe is believable: the two
        highest-eigenvalue degenerate bi-lobal pairs are the ring angle
        and the tube angle, agreeing with each to 1.0000."""
        bonds = [tuple(b) for b in ring.info["bonds"]]
        values, _vectors, flags = bilobal_spectrum(bonds, len(ring), most=40)
        pairs = bilobal_pairs(values, flags)
        assert len(pairs) >= 2
        for first, second in pairs[:2]:
            assert values[first] == pytest.approx(values[second], abs=1e-6)
        # and they are genuinely different harmonics
        assert values[pairs[0][0]] > values[pairs[1][0]]

    def test_a_graph_with_no_pairs_is_refused(self):
        """A path graph has no degenerate pair, so it is not toroidal and
        the method says so rather than returning something."""
        bonds = [(i, i + 1) for i in range(9)]
        with pytest.raises(ValueError, match="not toroidal"):
            toroidal_coordinates(bonds, 10)
