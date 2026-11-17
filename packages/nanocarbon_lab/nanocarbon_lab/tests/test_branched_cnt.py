"""Tests for the branched nanotube: branches fused at mesh level.

One small build is shared by a module-scoped fixture so the file stays
in the fast suite; the combinatorial pieces are tested on their own.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nanocarbon_lab.builders.branched_cnt import (
    build_branched_cnt,
    hexagon_sides,
)


class TestTheHexagon:
    @pytest.mark.parametrize("perimeter", [10, 15, 20, 25])
    def test_it_closes_on_the_triangular_lattice(self, perimeter):
        """Sides (a..f) close iff a+b == d+e and b+c == e+f."""
        a, b, c, d, e, f = hexagon_sides(perimeter)
        assert a + b + c + d + e + f == perimeter
        assert a + b == d + e and b + c == e + f
        assert min(a, b, c, d, e, f) >= 1


@pytest.fixture(scope="module")
def branched():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_branched_cnt(freq=3, n_body_rings=12, branches=1,
                                  branch_freq=2, branch_tilt=30.0,
                                  shape="arc", waviness=0.3, seed=0)


class TestOneBranch:
    def test_the_junction_is_six_heptagons_and_nothing_else(self, branched):
        """Two capped tubes carry 12 + 12; the weld removes 6 + 6 and
        adds six heptagons, so the fused shell is a sphere again."""
        census = branched.info["ring_counts"]
        assert branched.info["n_branches"] == 1
        assert set(census) == {5, 6, 7}
        assert census[7] == 6 and census[5] == 18
        assert sum((6 - n) * count for n, count in census.items()) == 12

    def test_it_is_one_trivalent_sp2_network(self, branched):
        geometry = branched.info["geometry"]
        assert 1.30 < geometry["bond_min"] and geometry["bond_max"] < 1.55
        degree = np.bincount(np.asarray(branched.info["bonds"]).ravel(),
                             minlength=len(branched))
        assert (degree == 3).all()
        assert set(branched.get_chemical_symbols()) == {"C"}

    def test_the_parts_and_normals_cover_every_atom(self, branched):
        part = np.asarray(branched.info["part"])
        normals = np.asarray(branched.info["normals"])
        assert len(part) == len(normals) == len(branched)
        assert set(part) == {0, 1}
        assert (part == 0).sum() > (part == 1).sum() > 30
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
        # Outward: a step along the normal leaves the shell's centroid
        # behind for nearly every atom of a convex-ish tube.
        centre = branched.positions[part == 0].mean(axis=0)
        outward = ((branched.positions - centre) * normals).sum(axis=1)
        assert (outward[part == 0] > 0).mean() > 0.9


class TestTheRest:
    def test_no_branches_is_a_plain_capped_tube(self):
        tube = build_branched_cnt(freq=2, n_body_rings=8, branches=0,
                                  shape="straight", seed=0)
        assert set(tube.info["ring_counts"]) == {5, 6}
        assert tube.info["ring_counts"][5] == 12
        assert tube.info["heptagons"] == 0

    def test_it_is_reproducible_from_the_seed(self):
        kwargs = dict(freq=2, n_body_rings=8, branches=0, shape="arc")
        first = build_branched_cnt(seed=3, **kwargs)
        again = build_branched_cnt(seed=3, **kwargs)
        assert np.allclose(first.positions, again.positions)

    def test_a_branch_wider_than_its_trunk_is_refused(self):
        with pytest.raises(ValueError, match="wider"):
            build_branched_cnt(freq=2, branch_freq=3)
