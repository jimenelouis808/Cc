"""The curvature-targeted annealing objective.

The rule is discrete Gauss-Bonnet read locally: over a region,
``sum(6 - deg) = (3/pi) * integral K dA``, so one vertex's share is
``3 * K(v) / pi``. What makes that testable rather than decorative is
that summing it over a closed mesh must return ``6 * chi`` -- the same
budget every other builder in this package checks against.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from nanocarbon_lab.builders import implicit as im
from nanocarbon_lab.builders import remesh as rm

BOND = 1.42


def _mesh(kind: str):
    field, extent = im.junction_field(kind, tube_radius=6.0, arm_length=22.0,
                                      blend=4.0)
    mesh = rm.marching_cubes_mesh(field, extent, resolution=70)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rm.isotropic_remesh(
            mesh, field, target_edge=math.sqrt(3.0) * BOND, iterations=25,
            anneal_sweeps=0, rng=np.random.default_rng(0))


class TestCurvatureTargets:
    """What the targets must satisfy to be a curvature measure at all."""

    @pytest.mark.parametrize("kind", ["L", "Y", "X"])
    def test_the_targets_sum_to_the_euler_budget(self, kind):
        """The whole justification. If this drifts the rule is not
        Gauss-Bonnet any more, whatever it is."""
        mesh = _mesh(kind)
        vertices, faces = mesh
        euler = len(vertices) - len(faces) * 3 // 2 + len(faces)
        total = float(rm.vertex_curvature_targets(mesh).sum())
        assert total == pytest.approx(6.0 * euler, rel=0.03)

    def test_a_flat_sheet_asks_for_nothing(self):
        """A degree-5 vertex on a flat sheet has five 72 deg angles
        summing to exactly 2*pi, so its deficit is zero. That is what
        makes the measure geometric rather than a restatement of the
        degree -- and it is the property the census objective lacks."""
        # A hexagonal patch of a flat triangular lattice.
        points, index = [], {}
        for i in range(-6, 7):
            for j in range(-6, 7):
                if abs(i + j) > 6:
                    continue
                index[(i, j)] = len(points)
                points.append([i + 0.5 * j, j * math.sqrt(3.0) / 2.0, 0.0])
        faces = []
        for (i, j), a in index.items():
            b, c = index.get((i + 1, j)), index.get((i, j + 1))
            d = index.get((i - 1, j + 1))
            if b is not None and c is not None:
                faces.append([a, b, c])
            if c is not None and d is not None:
                faces.append([a, c, d])
        mesh = (np.asarray(points, float), np.asarray(faces, int))
        targets = rm.vertex_curvature_targets(mesh, smoothing=0)
        # Interior vertices only: the rim has no closed fan and its
        # deficit is a boundary term, not curvature.
        adjacency = rm._adjacency(mesh[1])
        interior = [v for v in range(len(points))
                    if len(adjacency[v]) == 6
                    and abs(points[v][0]) < 4 and abs(points[v][1]) < 4]
        assert interior, "no interior vertices to test"
        assert max(abs(targets[v]) for v in interior) < 1e-9

    def test_a_sphere_asks_for_twelve(self):
        """Closed, genus 0: the targets must add to 12, which is the
        twelve pentagons every fullerene has."""
        from nanocarbon_lab.builders import fullerene_mesh as fm

        # A closed capsule, subdivided and pushed onto a sphere. The
        # topology gives the 12 whatever the shape; making it round is
        # what checks that the ANGLES are being read, not the degrees.
        mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(n_rings=3), freq=3)
        vertices = np.asarray(mesh[0], float)
        vertices -= vertices.mean(axis=0)
        vertices /= np.linalg.norm(vertices, axis=1)[:, None]
        total = float(rm.vertex_curvature_targets((vertices, mesh[1])).sum())
        assert total == pytest.approx(12.0, rel=0.03)


class TestPlacement:
    """`place_disclinations` moves disclinations; it must not invent them."""

    @pytest.mark.parametrize("kind", ["L", "Y"])
    def test_the_descent_is_monotonic(self, kind):
        """Every round must lower the misfit. A round that does not is
        rolled back, so a non-monotonic history means the incremental
        bookkeeping has drifted from the true measure again."""
        _out, history = rm.place_disclinations(_mesh(kind))
        assert len(history) > 1, "no round improved at all"
        assert all(b < a for a, b in zip(history, history[1:], strict=False)), history

    @pytest.mark.parametrize("kind", ["L", "Y"])
    def test_the_euler_budget_is_untouched(self, kind):
        """Flips cannot change sum(6 - deg), so this is exact, not close."""
        mesh = _mesh(kind)
        out, _ = rm.place_disclinations(mesh)

        def budget(m):
            adjacency = rm._adjacency(m[1])
            return sum(6 - len(adjacency[v]) for v in range(len(m[0])))

        assert budget(out) == budget(mesh)

    @pytest.mark.parametrize("kind", ["L", "Y"])
    def test_it_never_adds_a_disclination(self, kind):
        """Without this guard the descent buys misfit by manufacturing
        5-7 pairs to chase curvature finer than a disclination can
        represent -- the misfit fell 17-36% while the fraction of
        disclinations on the right side of the curvature fell, the two
        measures moving opposite ways."""
        mesh = _mesh(kind)
        out, _ = rm.place_disclinations(mesh)

        def defects(m):
            return sum(1 for ns in rm._adjacency(m[1]).values()
                       if len(ns) != 6)

        assert defects(out) <= defects(mesh)

    def test_positions_are_never_moved(self):
        mesh = _mesh("L")
        out, _ = rm.place_disclinations(mesh)
        assert np.array_equal(out[0], mesh[0])

    def test_degrees_stay_inside_the_clamp(self):
        """Three- and four-membered rings cannot exist in sp2 carbon, so
        the clamp is a hard constraint rather than a preference."""
        out, _ = rm.place_disclinations(_mesh("Y"))
        degrees = [len(ns) for ns in rm._adjacency(out[1]).values()]
        assert min(degrees) >= 5
        assert max(degrees) <= 8

    def test_a_flat_sheet_has_nothing_to_place(self):
        """Zero curvature everywhere means zero target everywhere, so a
        defect-free flat mesh is already at the floor."""
        points, index = [], {}
        for i in range(-5, 6):
            for j in range(-5, 6):
                if abs(i + j) > 5:
                    continue
                index[(i, j)] = len(points)
                points.append([i + 0.5 * j, j * math.sqrt(3.0) / 2.0, 0.0])
        faces = []
        for (i, j), a in index.items():
            b, c = index.get((i + 1, j)), index.get((i, j + 1))
            d = index.get((i - 1, j + 1))
            if b is not None and c is not None:
                faces.append([a, b, c])
            if c is not None and d is not None:
                faces.append([a, c, d])
        mesh = (np.asarray(points, float), np.asarray(faces, int))
        _out, history = rm.place_disclinations(mesh)
        assert len(history) == 1



class TestRefinement:
    """`refine_disclinations` alternates the two passes. Neither alone works.

    The toroid is the demonstration, and the numbers are in the
    function's own docstring: census annealing alone reaches 21/17 and
    scrambles the Gauss-Bonnet split, placement alone holds the split and
    stalls at 47/43, and alternating reaches 17/15 holding it.
    """

    def test_it_changes_no_vertex_and_no_face(self):
        """Flips are the only move, so V and F are invariant and the
        positions are untouched. A toroid built through the full builder
        came out two atoms heavier, which is the dual step downstream,
        not this."""
        mesh = _mesh("Y")
        out, _log = rm.refine_disclinations(mesh, np.random.default_rng(0),
                                            cycles=2)
        assert len(out[0]) == len(mesh[0])
        assert len(out[1]) == len(mesh[1])
        assert np.array_equal(out[0], mesh[0])

    def test_the_euler_budget_survives(self):
        mesh = _mesh("Y")
        out, _log = rm.refine_disclinations(mesh, np.random.default_rng(0),
                                            cycles=2)

        def budget(m):
            adjacency = rm._adjacency(m[1])
            return sum(6 - len(adjacency[v]) for v in range(len(m[0])))

        assert budget(out) == budget(mesh)

    def test_it_removes_disclinations_rather_than_adding(self):
        mesh = _mesh("Y")
        out, _log = rm.refine_disclinations(mesh, np.random.default_rng(0),
                                            cycles=3)

        def defects(m):
            return sum(1 for ns in rm._adjacency(m[1]).values()
                       if len(ns) != 6)

        assert defects(out) < defects(mesh)

    def test_it_reports_a_cycle_log(self):
        """Returned rather than printed, so a caller can show it
        converged instead of assuming it did."""
        _out, log = rm.refine_disclinations(_mesh("L"),
                                            np.random.default_rng(0), cycles=2)
        assert log and all("disclinations" in line for line in log)


class TestTheDunlapBudget:
    """Where Dunlap's twelve actually comes from.

    In ``K dA = cos(phi) dphi dtheta`` the radii cancel, so the integral
    over a torus's outer half is 4*pi for ANY R and r, and the
    disclination budget there is ``3/pi * 4*pi = 12``. The formula tried
    first here, ``4*pi*r/e``, depends on the tube radius and gave 17 or
    27 -- it was never the right quantity.
    """

    @pytest.mark.parametrize("major,minor",
                             [(20.0, 5.0), (30.0, 3.0), (12.0, 4.0)])
    def test_the_outer_half_owes_exactly_twelve(self, major, minor):
        outer = 2.0 * math.pi * (math.sin(math.pi / 2) - math.sin(-math.pi / 2))
        assert 3.0 * outer / math.pi == pytest.approx(12.0)
        # and the whole torus owes nothing, which is genus 1.
        whole = 2.0 * math.pi * (math.sin(3 * math.pi / 2)
                                 - math.sin(-math.pi / 2))
        assert 3.0 * whole / math.pi == pytest.approx(0.0, abs=1e-9)

    def test_the_mesh_measure_agrees_with_the_analytic_one(self):
        """`vertex_curvature_targets` must reproduce the +12, or it is
        not measuring the same thing the rule is derived from."""
        major, minor, rings_round, rings_tube = 20.0, 5.0, 120, 40
        verts, faces = [], []
        for i in range(rings_round):
            theta = 2 * math.pi * i / rings_round
            for k in range(rings_tube):
                phi = 2 * math.pi * k / rings_tube
                rho = major + minor * math.cos(phi)
                verts.append([rho * math.cos(theta), rho * math.sin(theta),
                              minor * math.sin(phi)])
        def index(i, k):
            return (i % rings_round) * rings_tube + (k % rings_tube)
        for i in range(rings_round):
            for k in range(rings_tube):
                a, b = index(i, k), index(i + 1, k)
                c, d = index(i, k + 1), index(i + 1, k + 1)
                faces += [[a, b, c], [b, d, c]]
        mesh = (np.asarray(verts, float), np.asarray(faces, int))
        targets = rm.vertex_curvature_targets(mesh, smoothing=0)
        outer = sum(t for t, v in zip(targets, mesh[0], strict=True)
                    if math.hypot(v[0], v[1]) >= major)
        assert outer == pytest.approx(12.0, rel=0.05)
