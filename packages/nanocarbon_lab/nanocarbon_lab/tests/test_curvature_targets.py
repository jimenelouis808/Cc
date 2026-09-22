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


class TestObjectiveIsSelectable:
    """The objective is opt-in, and refuses a name it does not know."""

    def test_an_unknown_objective_is_refused(self):
        mesh = _mesh("L")
        with pytest.raises(ValueError, match="census.*curvature"):
            rm.anneal_edge_flips(mesh, np.random.default_rng(0), sweeps=1,
                                 objective="whatever")

    def test_the_census_objective_is_unchanged_by_the_new_argument(self):
        """The default must be bit-identical to what shipped before, or
        every structure in the package quietly moves."""
        mesh = _mesh("L")
        a = rm.anneal_edge_flips(mesh, np.random.default_rng(7), sweeps=8)
        b = rm.anneal_edge_flips(mesh, np.random.default_rng(7), sweeps=8,
                                 objective="census")
        assert np.array_equal(a[1], b[1])

    def test_the_curvature_objective_produces_a_valid_mesh(self):
        mesh = _mesh("L")
        out = rm.anneal_edge_flips(mesh, np.random.default_rng(3), sweeps=8,
                                   objective="curvature")
        assert len(out[0]) == len(mesh[0])
        assert len(out[1]) == len(mesh[1])
        degrees = {v: len(ns) for v, ns in rm._adjacency(out[1]).items()}
        assert min(degrees.values()) >= 5
        assert max(degrees.values()) <= 8
