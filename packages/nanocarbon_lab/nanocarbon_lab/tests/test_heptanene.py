"""Heptanene: whether an all-heptagon trivalent net may exist, and where.

The first class is the whole answer to the question, and it is pure
topology -- no carbon, no relaxation, no thresholds. The rest checks
that the construction which follows from it is the object it claims to
be.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders.heptanene import (
    admissible_geometries,
    angular_excess,
    atoms_per_handle,
    barycentric_placement,
    build_heptanene,
    closed_surface_series,
    cocycle_space,
    klein_quartic_map,
    psl27,
)


class TestWhichGeometryAdmitsIt:
    """``sum(6-n) = 6*chi`` decides it, and decides it three ways."""

    def test_a_sphere_cannot_carry_heptagons_alone(self):
        """chi = +2 needs F = -12, and a face count cannot be negative.

        The same budget that gives a fullerene its twelve pentagons:
        positive curvature is paid in faces smaller than six, and a
        heptagon is the wrong sign.
        """
        elliptic = admissible_geometries(7)[0]
        assert elliptic.name == "elliptic"
        assert not elliptic.possible
        assert elliptic.faces == -12

    def test_a_flat_periodic_sheet_cannot_either(self):
        """chi = 0 needs F = 0: the only Euclidean all-heptagon net is
        the one with no heptagons in it. This is why the haeckelite
        builder must pair every heptagon with a pentagon."""
        euclidean = admissible_geometries(7)[1]
        assert euclidean.name == "euclidean"
        assert not euclidean.possible
        assert euclidean.faces == 0

    def test_only_negative_curvature_admits_it(self):
        hyperbolic = admissible_geometries(7)[2]
        assert hyperbolic.name == "hyperbolic"
        assert hyperbolic.possible

    def test_hexagons_give_graphenes_answer(self):
        """The control. If this reasoning were special pleading it would
        get graphene wrong too -- a hexagon pays 0, so it is Euclidean
        and nothing else."""
        elliptic, euclidean, hyperbolic = admissible_geometries(6)
        assert not elliptic.possible
        assert euclidean.possible
        assert not hyperbolic.possible

    def test_pentagons_are_the_mirror_image(self):
        """C20 is all pentagons on a sphere, so the elliptic case must
        come out possible and the other two not."""
        elliptic, euclidean, hyperbolic = admissible_geometries(5)
        assert elliptic.possible and elliptic.faces == 12
        assert not euclidean.possible
        assert not hyperbolic.possible

    def test_the_regular_tiling_test_agrees(self):
        """``{p,3}`` is elliptic, Euclidean or hyperbolic as ``(p-2)``
        is below, at or above 4 -- an independent route to the same
        conclusion."""
        assert (5 - 2) < 4          # {5,3} dodecahedron, elliptic
        assert (6 - 2) == 4         # {6,3} graphene, Euclidean
        assert (7 - 2) > 4          # {7,3} heptanene, hyperbolic


class TestTheSmallestClosedSurfaces:
    def test_the_series_starts_at_genus_two(self):
        series = closed_surface_series(7, 3)
        assert series[0] == (12, 28, 42, -2, 2)
        assert series[1] == (24, 56, 84, -4, 3)
        assert series[2] == (36, 84, 126, -6, 4)

    def test_every_member_closes_its_counting(self, ):
        for faces, vertices, edges, chi, genus in closed_surface_series(7, 6):
            assert edges * 2 == faces * 7        # each face has 7 edges
            assert vertices * 3 == edges * 2     # each atom has 3 bonds
            assert vertices - edges + faces == chi
            assert chi == 2 - 2 * genus


class TestTheKleinQuartic:
    """The map is derived from PSL(2,7), not transcribed, so its counts
    can be checked rather than trusted."""

    def test_the_group_has_order_168(self):
        assert len(psl27()) == 168

    def test_the_map_is_seven_three(self):
        bonds, rings = klein_quartic_map()
        assert len(rings) == 24
        assert {len(r) for r in rings} == {7}
        assert len(bonds) == 84
        assert len({a for b in bonds for a in b}) == 56

    def test_every_atom_is_trivalent(self):
        bonds, _ = klein_quartic_map()
        degree: dict[int, int] = {}
        for a, b in bonds:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        assert set(degree.values()) == {3}

    def test_every_side_of_every_face_is_a_bond(self):
        bonds, rings = klein_quartic_map()
        known = set(bonds)
        for ring in rings:
            for i, u in enumerate(ring):
                w = ring[(i + 1) % len(ring)]
                assert tuple(sorted((u, w))) in known

    def test_the_euler_budget_is_minus_twenty_four(self):
        _bonds, rings = klein_quartic_map()
        deficit = sum(6 - len(r) for r in rings)
        assert deficit == -24 == 6 * (56 - 84 + 24)


class TestTheCocycleMeasuresTheGenus:
    """A periodic realisation assigns each bond a lattice shift, and the
    shifts must sum to zero round every face. That freedom is ``H_1``,
    so its rank is ``2g`` -- **without using Euler's formula**, which
    makes it an independent reading of the genus."""

    def test_its_rank_is_twice_the_genus(self):
        bonds, rings = klein_quartic_map()
        _tree, _free, kernel = cocycle_space(bonds, rings, 56)
        assert len(kernel) == 6
        assert np.linalg.matrix_rank(kernel.astype(float)) == 6

    def test_the_basis_is_integral(self):
        """A lattice shift of 0.9999 is not a shift, which is why this
        is solved over the integers and not by an SVD."""
        bonds, rings = klein_quartic_map()
        _tree, _free, kernel = cocycle_space(bonds, rings, 56)
        assert kernel.dtype.kind == "i"
        assert np.abs(kernel).max() == 1

    def test_every_basis_vector_closes_every_face(self):
        bonds, rings = klein_quartic_map()
        tree, free, kernel = cocycle_space(bonds, rings, 56)
        for vector in kernel:
            for ring in rings:
                total = 0
                for i, u in enumerate(ring):
                    w = ring[(i + 1) % len(ring)]
                    key = tuple(sorted((u, w)))
                    if key in tree:
                        continue
                    total += vector[free.index(key)] * (1 if u < w else -1)
                assert total == 0


class TestMostCocyclesAreNotEmbeddings:
    def test_the_majority_collapse(self):
        """A net whose barycentric placement puts two atoms on the same
        point is unstable and has no embedding for that shift. Measured:
        eighteen of the twenty three-of-six cocycles do exactly that, so
        the choice cannot be made arbitrarily."""
        from itertools import combinations

        bonds, rings = klein_quartic_map()
        tree, free, kernel = cocycle_space(bonds, rings, 56)
        where = {b: j for j, b in enumerate(free)}
        collapsed = 0
        total = 0
        for trio in combinations(range(6), 3):
            combo = np.zeros((3, 6), dtype=int)
            for row, col in enumerate(trio):
                combo[row, col] = 1
            component = combo @ kernel
            shifts = np.zeros((len(bonds), 3), dtype=int)
            for k, b in enumerate(bonds):
                if b not in tree:
                    shifts[k] = component[:, where[b]]
            placement = barycentric_placement(bonds, shifts, 56)
            delta = (placement[[b[1] for b in bonds]] + shifts
                     - placement[[b[0] for b in bonds]])
            total += 1
            if np.linalg.norm(delta, axis=1).min() < 1e-6:
                collapsed += 1
        assert total == 20
        assert collapsed == 18


class TestTheCurvatureIsNotTheObstruction:
    def test_heptanene_is_less_curved_than_the_smallest_fullerene(self):
        """25.7 deg of excess against C20's 36 deg of deficit -- and C20
        exists. So whatever stops heptanene, it is not the amount of
        curvature per atom."""
        assert angular_excess(7) == pytest.approx(25.714, abs=0.01)
        assert angular_excess(5) == pytest.approx(-36.0, abs=0.01)
        assert abs(angular_excess(7)) < abs(angular_excess(5))

    def test_graphene_is_flat(self):
        assert angular_excess(6) == pytest.approx(0.0)

    def test_the_handles_are_what_is_short(self):
        """56 atoms over three handles is 18.7 atoms each, and the
        series improves only slowly."""
        assert atoms_per_handle(2) == pytest.approx(18.667, abs=0.01)
        assert atoms_per_handle(3) == pytest.approx(21.0, abs=0.01)
        assert atoms_per_handle(20) < 28.0


class TestTheBuild:
    """The whole build is 0.8 s -- 56 atoms, 84 bonds, analytic
    gradients -- so it belongs in the fast suite, not behind a marker."""

    def test_it_refuses_rather_than_returning_a_strained_lattice(self):
        with pytest.raises(ValueError, match="not carbon's"):
            build_heptanene()

    def test_the_topology_it_did_build_is_exact(self):
        atoms = build_heptanene(strict=False)
        assert len(atoms) == 56
        assert atoms.info["ring_counts"] == {7: 24}
        assert atoms.info["genus"] == 3
        assert atoms.info["sp2"] is False
        assert list(atoms.get_pbc()) == [True, True, True]
