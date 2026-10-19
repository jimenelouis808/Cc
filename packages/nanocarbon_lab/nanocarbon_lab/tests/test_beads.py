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
    MAX_ZIGZAG_ANGLE,
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


class TestTheZigzagLayout:
    """A bead chain may corner where a swept tube may not, and the
    reason is the representation rather than a looser limit: this is an
    implicit field, so a corner is a smooth union of two necks instead
    of a bent lattice with strain to pay. What limits it is instead the
    beads either side of a corner meeting -- the same intersection the
    straight chain's spacing floor is about, reached without the spacing
    having changed at all."""

    def test_it_is_a_layout(self):
        assert "zigzag" in BEAD_LAYOUTS

    def test_every_edge_keeps_the_full_spacing(self):
        """The step keeps its length through the turn, so the neck
        geometry is the same on every edge as in a straight chain. If a
        corner shortened an edge, the neck there would be a dimple and
        the spacing guard would be checking the wrong number."""
        vertices, edges = bead_graph("zigzag", 10, 16.8,
                                     zigzag_angle=45.0, zigzag_leg=3)
        lengths = np.linalg.norm(
            vertices[[a for a, _ in edges]] - vertices[[b for _, b in edges]],
            axis=1)
        assert np.allclose(lengths, 16.8)

    def test_it_is_a_tree_so_the_budget_is_still_twelve(self):
        vertices, edges = bead_graph("zigzag", 7, 16.8)
        assert 12 * (len(vertices) - len(edges)) == 12

    def test_the_legs_turn_where_asked(self):
        vertices, _ = bead_graph("zigzag", 9, 16.8, zigzag_angle=45.0,
                                 zigzag_leg=2)
        lateral = vertices[:, 0]
        turns = int((np.diff(np.sign(np.diff(lateral))) != 0).sum())
        # Eight steps at two per leg is four legs, so three turns.
        assert turns == 3

    def test_the_across_corner_distance_is_what_the_geometry_says(self):
        """A turn swings the direction by twice the leg angle, so the
        beads either side of a corner sit 2 * spacing * cos(angle)
        apart -- the closed form the builder's guard uses."""
        for angle in (30.0, 45.0, 60.0):
            vertices, edges = bead_graph("zigzag", 9, 16.8,
                                         zigzag_angle=angle, zigzag_leg=2)
            distances = np.linalg.norm(vertices[:, None] - vertices[None, :],
                                       axis=-1)
            np.fill_diagonal(distances, np.inf)
            for first, second in edges:
                distances[first, second] = np.inf
                distances[second, first] = np.inf
            expected = 2.0 * 16.8 * np.cos(np.radians(angle))
            assert distances.min() == pytest.approx(expected, rel=1e-6)

    def test_a_sharp_turn_is_refused_before_anything_is_meshed(self):
        with pytest.raises(ValueError, match="either side of a corner"):
            build_bead_chain(layout="zigzag", count=4, bead_radius=7.0,
                             neck_radius=3.0, zigzag_angle=80.0)

    def test_a_degenerate_angle_is_refused_by_the_graph(self):
        for angle in (0.0, MAX_ZIGZAG_ANGLE, 90.0):
            with pytest.raises(ValueError, match="zigzag_angle"):
                bead_graph("zigzag", 5, 16.8, zigzag_angle=angle)

    def test_a_zero_length_leg_is_refused(self):
        with pytest.raises(ValueError, match="zigzag_leg"):
            bead_graph("zigzag", 5, 16.8, zigzag_leg=0)

    def test_a_straight_chain_is_asked_for_by_name(self):
        """Angle 0 is a straight chain, and the message says so rather
        than returning one under the zigzag's name."""
        with pytest.raises(ValueError, match="chain"):
            bead_graph("zigzag", 5, 16.8, zigzag_angle=0.0)

    @pytest.mark.slow
    def test_a_zigzag_chain_folds_far_harder_than_a_swept_tube_can(self):
        """The claim the Z path rests on, measured against its
        alternative. A capped tube swept onto a Z path is capped by the
        strain budget at a few per cent of its length; this is a field,
        so it corners for free."""
        atoms = build_bead_chain(layout="zigzag", count=5, bead_radius=6.0,
                                 neck_radius=2.6, zigzag_angle=45.0,
                                 zigzag_leg=1, seed=0)
        census = atoms.info["ring_counts"]
        measured = sum((6 - size) * n for size, n in census.items())
        assert measured == atoms.info["ring_budget"] == 12
        assert atoms.info["geometry"]["n_close_contacts"] == 0
        positions = atoms.get_positions()
        across = float(np.ptp(positions[:, 0]) / np.ptp(positions[:, 2]))
        # Measured 0.43. A swept tube at the default budget reaches 0.08
        # at one period and 0.018 at two, so the margin is not marginal.
        assert across > 0.25
