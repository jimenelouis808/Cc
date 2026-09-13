"""Tests for curved tubes built through the implicit route.

The point of this builder is that the ring topology is *derived* from the
curvature rather than imposed, so the tests check exactly that: a coiled
tube has to carry more pentagons and heptagons than a straight one of the
same length, its Euler budget still has to come to 12, and -- the part
that motivated the whole module -- its bonds have to stay at graphitic
length instead of being stretched by the bend.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import implicit as im
from nanocarbon_lab.builders import remesh as rm
from nanocarbon_lab.builders.swept import _measure_coil, build_coil, build_swept_tube
from nanocarbon_lab.validation.quality import sp2_quality

pytest.importorskip("skimage", reason="scikit-image provides marching cubes")


def _deficit(ring_counts: dict[int, int]) -> int:
    return sum((6 - size) * count for size, count in ring_counts.items())


class TestTubeAlongPath:
    def test_field_is_a_signed_distance_to_the_tube(self):
        path = np.array([[0.0, 0.0, -10.0], [0.0, 0.0, 10.0]])
        field, lower, upper = im.tube_along_path(path, radius=5.0)
        probes = np.array([
            [0.0, 0.0, 0.0],    # on the axis: -radius
            [5.0, 0.0, 0.0],    # on the surface: 0
            [9.0, 0.0, 0.0],    # 4 A outside
        ])
        assert field(probes) == pytest.approx([-5.0, 0.0, 4.0], abs=0.02)
        # The box must clear the whole capsule, caps included.
        assert lower[2] < -15.0 and upper[2] > 15.0

    def test_box_is_not_forced_to_be_a_cube(self):
        """A flat, wide shape must not be padded into a cube."""
        angle = np.linspace(0, 2 * np.pi, 60)
        ring = np.stack([40 * np.cos(angle), 40 * np.sin(angle),
                         np.zeros_like(angle)], axis=-1)
        _, lower, upper = im.tube_along_path(ring, radius=5.0)
        extent = upper - lower
        assert extent[2] < 0.5 * extent[0]

    def test_degenerate_inputs_raise(self):
        with pytest.raises(ValueError):
            im.tube_along_path(np.zeros((1, 3)), radius=5.0)
        with pytest.raises(ValueError):
            im.tube_along_path(np.array([[0, 0, 0], [0, 0, 1]], float), radius=0.0)
        with pytest.raises(ValueError):
            # Sampling coarser than the tube would not resolve it.
            im.tube_along_path(np.array([[0, 0, 0], [0, 0, 9]], float),
                               radius=1.0, sample_spacing=5.0)


class TestMarchingCubesBox:
    def test_vertices_land_in_absolute_coordinates(self):
        """The box is not centred on the origin, so nor is the surface."""
        centre = np.array([20.0, -10.0, 5.0])

        def sphere(points):
            return np.linalg.norm(points - centre, axis=-1) - 6.0

        mesh = rm.marching_cubes_box(sphere, centre - 10.0, centre + 10.0, spacing=0.5)
        verts = mesh[0]
        assert verts.mean(axis=0) == pytest.approx(centre, abs=0.3)
        radii = np.linalg.norm(verts - centre, axis=1)
        assert radii.min() > 5.7 and radii.max() < 6.3
        assert rm.mesh_statistics(mesh)["boundary_edges"] == 0

    def test_oversized_grid_is_refused_rather_than_attempted(self):
        def sphere(points):
            return np.linalg.norm(points, axis=-1) - 1.0

        with pytest.raises(ValueError, match="exceeds"):
            rm.marching_cubes_box(sphere, -np.ones(3) * 500, np.ones(3) * 500,
                                  spacing=0.05)


@pytest.fixture(scope="module")
def coil():
    """One relaxed coil, shared by the tests below (it takes minutes).

    The dimensions are chosen to be physically comfortable rather than
    merely small. A 26 Å coil around a 5 Å tube builds faster but is tight
    enough to leave a 1.58 Å bond -- a real result, and one the sp2
    verdict correctly calls broken, but not the behaviour these tests are
    about. Turns are above 1.0 because pitch cannot be measured below that
    (see :func:`_measure_coil`).
    """
    return build_coil(coil_radius=34.0, pitch=20.0, turns=1.25, tube_radius=4.5,
                      remesh_iterations=20, anneal_sweeps=40)


@pytest.mark.slow
class TestRelaxedCoil:
    def test_euler_budget_holds(self, coil):
        assert _deficit(coil.info["ring_counts"]) == 12
        assert coil.info["genus"] == 0

    def test_curvature_buys_its_own_pentagons_and_heptagons(self, coil):
        """A coiled tube needs far more than the 12 pentagons that cap it.

        A straight capped tube gets by with exactly 12 pentagons and no
        heptagons. Bending it means compressing the inner wall and
        stretching the outer one, and the honeycomb pays for that with
        extra 5-7 pairs -- which is the whole reason this builder exists.
        """
        counts = coil.info["ring_counts"]
        assert counts.get(5, 0) > 20
        assert counts.get(7, 0) > 8
        # The extras must cancel, or Euler above would not have held.
        assert counts.get(5, 0) - counts.get(7, 0) - 2 * counts.get(8, 0) == 12

    def test_bonds_stay_graphitic_despite_the_curvature(self, coil):
        """The payoff: curvature absorbed by topology, not by bond stretch."""
        verdict, why = sp2_quality(coil.info["geometry"])
        assert verdict in ("clean", "strained"), why
        assert coil.info["geometry"]["n_close_contacts"] == 0

    def test_requested_coil_radius_is_honoured(self, coil):
        """Curvature *is* encoded in the ring sizes, so the radius holds."""
        assert coil.info["achieved_coil_radius"] == pytest.approx(34.0, rel=0.15)

    def test_pitch_is_reported_rather_than_assumed(self, coil):
        """Torsion is *not* encoded, so the pitch is free to move.

        A gentle coil like this one barely does (20 Å requested, 22 Å
        relaxed), but a tighter one springs open much further, which is
        why the achieved value is measured and reported instead of the
        requested one being echoed back.
        """
        assert coil.info["achieved_pitch"] == pytest.approx(20.0, rel=0.35)

    def test_pitch_is_nan_below_one_full_turn(self):
        """One turn visits each azimuthal sector once, so there is no
        second cluster to measure a gap to. Saying so beats the plausible
        wrong number the axial-span fallback used to give (63.9 Å for a
        20 Å pitch)."""
        angle = np.linspace(0.0, 2 * np.pi * 0.8, 500)
        arc = np.stack([30 * np.cos(angle), 30 * np.sin(angle),
                        20.0 * angle / (2 * np.pi)], axis=-1)
        _, pitch = _measure_coil(arc)
        assert np.isnan(pitch)

    def test_all_atoms_are_three_coordinate(self, coil):
        degree = np.zeros(len(coil), dtype=int)
        for a, b in coil.info["bonds"]:
            degree[a] += 1
            degree[b] += 1
        assert set(np.unique(degree)) == {3}

    def test_a_coil_too_tight_to_exist_is_refused(self):
        with pytest.raises(ValueError, match="turns"):
            build_coil(coil_radius=30.0, pitch=8.0, tube_radius=6.0)


class TestCoilMeasurement:
    def test_pitch_is_read_from_azimuthal_sectors(self):
        """Reading pitch by unwrapping the azimuth along z is wrong.

        Atoms at one height wrap right around the tube, so the azimuth is
        multivalued in z and unwrapping accumulates turns the coil does not
        have. The sector method must recover the true pitch of a synthetic
        helix where the unwrapping method does not.
        """
        angle = np.linspace(0.0, 2 * np.pi * 3.0, 4000)
        axis = np.stack([26.0 * np.cos(angle), 26.0 * np.sin(angle),
                         20.0 * angle / (2 * np.pi)], axis=-1)
        # Dress the centreline with a tube of radius 5 so the azimuth really
        # is multivalued in z, as it is for a built structure.
        rng = np.random.default_rng(0)
        offsets = rng.normal(size=(len(axis), 3))
        offsets -= (offsets * np.gradient(axis, axis=0)).sum(1)[:, None] * 0.0
        offsets /= np.linalg.norm(offsets, axis=1)[:, None]
        cloud = axis + 5.0 * offsets

        radius, pitch = _measure_coil(cloud)
        assert radius == pytest.approx(26.0, rel=0.15)
        assert pitch == pytest.approx(20.0, rel=0.15)


@pytest.mark.slow
class TestSweptTube:
    def test_a_straight_path_gives_a_plain_capped_tube(self):
        """With no curvature to pay for, only the 12 cap pentagons appear."""
        path = np.array([[0.0, 0.0, -12.0], [0.0, 0.0, 12.0]])
        atoms = build_swept_tube(path, tube_radius=5.0, remesh_iterations=20,
                                 anneal_sweeps=40)
        counts = atoms.info["ring_counts"]
        assert _deficit(counts) == 12
        # A straight tube needs no heptagons at all beyond stray pairs.
        assert counts.get(7, 0) <= 6
        assert atoms.info["geometry"]["n_close_contacts"] == 0
