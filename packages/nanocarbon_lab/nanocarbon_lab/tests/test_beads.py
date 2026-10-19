"""Graphitic spheres joined by necks.

The whole structure is decided by its graph before any geometry
exists, so most of what is worth checking costs no build at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import implicit as im
from nanocarbon_lab.builders.beads import (
    BEAD_LAYOUTS,
    MIN_SPACING_FRACTION,
    bead_graph,
    build_bead_chain,
)


class TestPerSegmentRadius:
    """A bead is a zero-length capsule, which is why there is one field.

    Generalising ``graph_field`` rather than unioning a second field is
    only safe if the scalar path it already had is untouched, so that
    is asserted exactly rather than approximately -- every supernetwork,
    every cage and every nanotube network goes through it.
    """

    def test_the_scalar_path_is_bit_identical(self):
        rng = np.random.default_rng(0)
        segments = rng.normal(size=(6, 2, 3)) * 8.0
        box = np.array([40.0, 40.0, 40.0])
        points = rng.normal(size=(400, 3)) * 12.0
        scalar = im.graph_field(segments, box, (True, True, False),
                                tube_radius=5.0, blend=3.0)(points)
        per_segment = im.graph_field(segments, box, (True, True, False),
                                     tube_radius=[5.0] * 6,
                                     blend=3.0)(points)
        assert np.array_equal(scalar, per_segment)

    def test_a_zero_length_segment_is_a_sphere(self):
        """Exactly a sphere: the distance is |r| - radius everywhere."""
        sphere = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        field = im.graph_field(sphere, np.array([80.0] * 3),
                               (False, False, False), tube_radius=4.0,
                               blend=3.0)
        probe = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                          [0.0, 6.0, 0.0], [0.0, 0.0, 10.0]])
        assert np.allclose(field(probe), [-4.0, 0.0, 2.0, 6.0])

    def test_a_wrong_length_radius_array_is_refused(self):
        segments = np.zeros((3, 2, 3))
        segments[:, 1, 0] = 1.0
        with pytest.raises(ValueError, match="one per segment"):
            im.graph_field(segments, np.array([40.0] * 3),
                           (False, False, False), tube_radius=[1.0, 2.0])


class TestBeadGraph:
    """``V - E`` fixes the ring budget before anything is meshed."""

    def test_a_chain_is_a_tree_at_every_length(self):
        """So its budget is +12 however many beads it has: a sausage is
        still sphere-like, which is the claim worth pinning."""
        for count in (2, 3, 5, 9):
            vertices, edges = bead_graph("chain", count, 16.8)
            assert len(vertices) - len(edges) == 1

    def test_a_ring_closes(self):
        for count in (3, 6, 10):
            vertices, edges = bead_graph("ring", count, 16.8)
            assert len(vertices) - len(edges) == 0

    def test_a_cross_is_a_tree(self):
        vertices, edges = bead_graph("cross", 2, 16.8, arms=4)
        assert len(vertices) == 9
        assert len(vertices) - len(edges) == 1

    def test_ring_spacing_sets_the_radius(self):
        """n chords of length `spacing` must close the circle."""
        vertices, _ = bead_graph("ring", 6, 16.8)
        step = np.linalg.norm(vertices[1] - vertices[0])
        assert step == pytest.approx(16.8, rel=1e-6)

    def test_unknown_layout_names_the_options(self):
        with pytest.raises(ValueError, match="|".join(BEAD_LAYOUTS)):
            bead_graph("spiral", 3, 16.8)


class TestRefusals:
    """Each refusal is a shape that would come back sound and wrong."""

    def test_a_neck_as_wide_as_the_bead_is_refused(self):
        with pytest.raises(ValueError, match="lozenge"):
            build_bead_chain(bead_radius=7.0, neck_radius=6.5)

    def test_intersecting_beads_are_refused(self):
        """Closer than twice the radius the spheres intersect, so there
        is no waist and the result is a lumpy tube. The first version
        defaulted BELOW this floor and the render showed exactly that."""
        with pytest.raises(ValueError, match="INTERSECT"):
            build_bead_chain(bead_radius=7.0, neck_radius=3.0,
                             spacing=1.5 * 7.0)

    def test_the_floor_is_where_spheres_touch(self):
        assert MIN_SPACING_FRACTION == 2.0

    def test_a_ring_whose_hole_closes_is_refused(self):
        """Measured: a hole of +1.4 A closes as a SPHERE and returns a
        sound structure that is not the ring asked for -- which is why
        this is caught up front rather than left to the census."""
        # Spacing 14.0 clears the intersect floor (2 x 7.0) so that
        # guard does not fire first, and still leaves a 2.9 A hole --
        # under the blend, so the beads merge across the middle.
        with pytest.raises(ValueError, match="closes as a sphere"):
            build_bead_chain(layout="ring", count=4, bead_radius=7.0,
                             neck_radius=3.0, spacing=14.0)

    def test_too_few_beads(self):
        with pytest.raises(ValueError, match="at least 2"):
            build_bead_chain(count=1)


@pytest.mark.slow
class TestBuiltStructures:
    """The builds themselves, which take seconds each."""

    def test_a_chain_meets_its_graph_budget(self):
        atoms = build_bead_chain(layout="chain", count=3, bead_radius=7.0,
                                 neck_radius=3.0)
        census = atoms.info["ring_counts"]
        measured = sum((6 - size) * n for size, n in census.items())
        assert measured == atoms.info["ring_budget"] == 12
        assert atoms.info["geometry"]["n_close_contacts"] == 0

    def test_a_ring_comes_out_toroidal(self):
        atoms = build_bead_chain(layout="ring", count=6, bead_radius=7.0,
                                 neck_radius=3.0)
        census = atoms.info["ring_counts"]
        measured = sum((6 - size) * n for size, n in census.items())
        assert measured == atoms.info["ring_budget"] == 0
        assert atoms.info["euler"] == 0
