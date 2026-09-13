"""Tests for closed fullerene cages and the onions built from them.

These have a luxury the curved-surface builders do not: the answers are
known. C60 is the truncated icosahedron -- 12 pentagons, 20 hexagons,
radius 3.55 Å -- and the icosahedral series is fixed by the Goldberg
classification, so the tests check named structures rather than ranges.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import build_fullerene, build_nano_onion
from nanocarbon_lab.builders.fullerene import (
    icosahedron_mesh,
    pentakis_dodecahedron_mesh,
)
from nanocarbon_lab.builders.remesh import degree_histogram, mesh_statistics
from nanocarbon_lab.validation.quality import sp2_quality


def _deficit(ring_counts: dict[int, int]) -> int:
    return sum((6 - size) * count for size, count in ring_counts.items())


class TestSeedPolyhedra:
    """Vertex degree is ring size, so the seeds decide the cage."""

    def test_icosahedron_is_twelve_degree_five_vertices(self):
        mesh = icosahedron_mesh()
        assert degree_histogram(mesh) == {5: 12}
        stats = mesh_statistics(mesh)
        assert (stats["vertices"], stats["faces"], stats["euler"]) == (12, 20, 2)

    def test_pentakis_dodecahedron_gives_the_c60_ring_census(self):
        """12 degree-5 + 20 degree-6 vertices is exactly C60's dual."""
        mesh = pentakis_dodecahedron_mesh()
        assert degree_histogram(mesh) == {5: 12, 6: 20}
        stats = mesh_statistics(mesh)
        assert (stats["vertices"], stats["faces"], stats["euler"]) == (32, 60, 2)
        assert stats["boundary_edges"] == 0

    def test_seeds_are_consistently_oriented(self):
        """Mixed winding would make the dual order a ring's atoms into a
        self-crossing polygon, so every triangle must face outward."""
        for mesh in (icosahedron_mesh(), pentakis_dodecahedron_mesh()):
            verts, faces = mesh
            for tri in faces:
                a, b, c = verts[tri]
                assert np.cross(b - a, c - a) @ (a + b + c) > 0


class TestFullereneCages:
    def test_c60_is_the_truncated_icosahedron(self):
        cage = build_fullerene(freq=1, family="C60")
        assert len(cage) == 60
        assert cage.info["ring_counts"] == {5: 12, 6: 20}
        # Literature radius is 3.55 Å.
        assert cage.info["radius"] == pytest.approx(3.55, abs=0.15)

    @pytest.mark.parametrize(
        ("family", "freq", "n_atoms"),
        [
            ("C60", 1, 60), ("C60", 2, 240), ("C60", 3, 540),
            ("C20", 1, 20), ("C20", 2, 80), ("C20", 3, 180),
        ],
    )
    def test_the_goldberg_series_comes_out_at_the_named_sizes(
        self, family, freq, n_atoms
    ):
        """Subdividing a triangulation by f multiplies its faces -- and so
        the dual's atoms -- by f squared."""
        cage = build_fullerene(freq=freq, family=family)
        assert len(cage) == n_atoms
        assert cage.info["formula"] == f"C{n_atoms}"

    @pytest.mark.parametrize("family", ["C60", "C20"])
    @pytest.mark.parametrize("freq", [1, 2])
    def test_every_cage_has_exactly_twelve_pentagons(self, family, freq):
        """Euler's budget for a sphere, and it is derived, not imposed."""
        cage = build_fullerene(freq=freq, family=family)
        assert cage.info["ring_counts"].get(5, 0) == 12
        assert _deficit(cage.info["ring_counts"]) == 12

    @pytest.mark.parametrize("family", ["C60", "C20"])
    def test_cages_relax_to_clean_sp2_geometry(self, family):
        cage = build_fullerene(freq=2, family=family)
        verdict, why = sp2_quality(cage.info["geometry"])
        assert verdict == "clean", why

    def test_radius_grows_linearly_with_frequency(self):
        """It is this ~3.5 Å per step that makes a graphitic onion possible."""
        radii = [build_fullerene(freq=f, family="C60").info["radius"]
                 for f in (1, 2, 3)]
        steps = np.diff(radii)
        assert steps == pytest.approx([3.5, 3.5], abs=0.2)

    def test_class_one_steps_too_finely_for_an_onion(self):
        """The reason the C60 family is the default: C20's ~2 Å steps
        cannot be combined into graphite's 3.4 Å at any freq_step."""
        radii = [build_fullerene(freq=f, family="C20").info["radius"]
                 for f in (1, 2, 3)]
        steps = np.diff(radii)
        assert steps == pytest.approx([2.0, 2.0], abs=0.2)

    def test_all_atoms_are_three_coordinate(self):
        cage = build_fullerene(freq=2, family="C60")
        degree = np.zeros(len(cage), dtype=int)
        for a, b in cage.info["bonds"]:
            degree[a] += 1
            degree[b] += 1
        assert set(np.unique(degree)) == {3}

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            build_fullerene(freq=0)
        with pytest.raises(ValueError, match="class-I"):
            build_fullerene(family="C70")


class TestNanoOnion:
    @pytest.fixture(scope="class")
    def onion(self):
        return build_nano_onion(n_shells=3, family="C60")

    def test_it_is_the_classic_c60_at_c240_at_c540(self, onion):
        assert onion.info["formula"] == "C60@C240@C540"
        assert len(onion) == 60 + 240 + 540

    def test_each_shell_pays_its_own_euler_budget(self, onion):
        """Three closed cages owe 12 pentagons apiece."""
        assert onion.info["ring_counts"].get(5, 0) == 36
        assert _deficit(onion.info["ring_counts"]) == 36

    def test_shells_nest_at_the_graphitic_spacing(self, onion):
        """The point of the C60 family. The covalent force field has no
        dispersion term, so this is measured, never assumed."""
        assert onion.info["shell_spacing"] == pytest.approx(3.5, abs=0.25)
        gap = onion.info["geometry"]["min_wall_separation"]
        assert 3.0 < gap < 3.8

    def test_shells_do_not_interpenetrate(self, onion):
        assert onion.info["geometry"]["n_close_contacts"] == 0
        verdict, why = sp2_quality(onion.info["geometry"])
        assert verdict == "clean", why

    def test_a_single_shell_reports_no_wall_separation(self):
        """With no second cage there is no gap; saying `nan` beats
        reporting an intra-shell distance that means nothing."""
        lone = build_nano_onion(n_shells=1)
        assert np.isnan(lone.info["geometry"]["min_wall_separation"])

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            build_nano_onion(n_shells=0)
        with pytest.raises(ValueError):
            build_nano_onion(freq_step=0)
