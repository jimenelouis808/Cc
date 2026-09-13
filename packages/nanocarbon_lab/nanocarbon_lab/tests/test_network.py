"""Tests for periodic 3D networks of interconnected nanotubes.

The whole argument for building these implicitly rather than gluing
junctions together is that the ring statistics come out *derived*. So
the assertions that matter are topological: the Euler budget against the
net's own genus, straight sections that are all hexagon, and odd rings
appearing only where the curvature is.

Two things here are cheap to get wrong and expensive to notice. The
field has to be genuinely periodic, or the marching-cubes weld finds
nothing to join and the cell comes out torn. And it has to evaluate in
bounded memory: the point-by-strut array is 3.9 GB for a diamond cell on
a 72^3 grid, which had the process killed outright rather than raising.
Both are pinned here without building anything.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import implicit as im
from nanocarbon_lab.builders.network import (
    STRUT_FRACTION,
    build_nanotube_network,
    minimum_cell,
)
from nanocarbon_lab.cell import cell_report
from nanocarbon_lab.validation import run_basic_checks


class TestNetworkField:
    @pytest.mark.parametrize("kind", ["cubic", "diamond"])
    def test_opposite_faces_of_the_cell_agree(self, kind):
        """Exactly, not approximately. The periodic marching cubes weld
        joins the two faces, so any mismatch tears the surface."""
        cell = 60.0
        field, _ = im.network_field(kind, cell=cell, tube_radius=6.0, blend=5.0)
        rng = np.random.default_rng(0)
        face = rng.uniform(0.0, cell, size=(300, 3))
        for axis in range(3):
            low, high = face.copy(), face.copy()
            low[:, axis] = 0.0
            high[:, axis] = cell
            assert np.allclose(field(low), field(high), atol=1e-9)

    @pytest.mark.parametrize("kind", ["cubic", "diamond"])
    def test_it_is_a_distance_field(self, kind):
        """|grad| must be about 1, or the remesher's surface projection
        walks the wrong distance and the mesh drifts off the surface."""
        field, _ = im.network_field(kind, cell=60.0, tube_radius=6.0, blend=5.0)
        points = np.random.default_rng(1).uniform(0.0, 60.0, size=(200, 3))
        magnitude = np.linalg.norm(im.gradient(field, points), axis=1)
        assert 0.75 < float(np.median(magnitude)) <= 1.05

    def test_the_solid_does_not_fill_the_cell(self):
        """An exponential soft-min over all 27 images subtracted
        blend*log(n) everywhere at once, which inflated the tubes until
        the whole cell read as inside. Only the nearest few may blend."""
        field, _ = im.network_field("diamond", cell=70.0, tube_radius=6.0,
                                    blend=5.0)
        points = np.random.default_rng(2).uniform(0.0, 70.0, size=(500, 3))
        inside = float((field(points) < 0).mean())
        assert 0.05 < inside < 0.6

    def test_evaluation_stays_within_a_memory_budget(self):
        """A whole marching-cubes grid's worth of points at once. Before
        chunking this allocated 3.9 GB for a diamond cell and the process
        was killed -- no traceback, no failure message."""
        import resource

        field, _ = im.network_field("diamond", cell=70.0, tube_radius=6.0,
                                    blend=5.0)
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        field(np.random.default_rng(3).uniform(0.0, 70.0, size=(60000, 3)))
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KiB on Linux; 1.5 GB of growth would mean the
        # chunking is not in effect.
        assert (after - before) < 1.5e6

    def test_an_unknown_net_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown network"):
            im.network_field("kagome", cell=40.0)


class TestNetworkGeometry:
    def test_the_segments_span_the_cell(self):
        """A cubic strut runs a full edge; a diamond one is the
        tetrahedral quarter-diagonal."""
        for kind in ("cubic", "diamond"):
            segments = im.network_segments(kind, cell=40.0)
            lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
            assert np.allclose(lengths, STRUT_FRACTION[kind] * 40.0)

    def test_the_minimum_cell_leaves_a_tube(self):
        """At the floor exactly, the free tube length is what the floor
        was defined to leave -- one tube radius."""
        radius, blend = 6.0, 5.0
        for kind in ("cubic", "diamond"):
            floor = minimum_cell(kind, radius, blend)
            free = STRUT_FRACTION[kind] * floor - 2.0 * (radius + blend)
            assert free == pytest.approx(radius)

    def test_diamond_needs_a_bigger_cell_than_cubic(self):
        """Its struts are the quarter-diagonal, so the same cell edge
        buys 0.43 of the tube length."""
        assert minimum_cell("diamond", 6.0, 5.0) > minimum_cell("cubic", 6.0, 5.0)

    def test_a_cell_below_the_floor_is_refused_with_the_number(self):
        with pytest.raises(ValueError, match=r"too small.*cell >= \d+"):
            build_nanotube_network("cubic", cell=20.0, tube_radius=6.0)

    def test_an_unknown_net_is_refused(self):
        with pytest.raises(ValueError, match="Unknown network"):
            minimum_cell("honeycomb", 6.0, 5.0)


@pytest.mark.slow
class TestBuiltNetwork:
    @pytest.fixture(scope="class")
    def cubic(self):
        return build_nanotube_network("cubic", cell=40.0, tube_radius=6.0)

    def test_it_is_a_periodic_cubic_cell(self, cubic):
        assert all(cubic.get_pbc())
        lengths = cubic.cell.lengths()
        assert lengths[0] == pytest.approx(lengths[1])
        assert lengths[1] == pytest.approx(lengths[2])

    def test_the_ring_budget_matches_the_genus(self, cubic):
        """The whole point of deriving the topology: nobody puts the
        heptagons at the nodes, and the count is fixed by Euler."""
        deficit = sum((6 - size) * count
                      for size, count in cubic.info["ring_counts"].items())
        assert deficit == 6 * cubic.info["euler"]

    def test_a_cubic_net_has_genus_three(self, cubic):
        """Three independent handles on the 3-torus, one per axis --
        the same genus as the Schwarz P surface, and for the same reason."""
        assert cubic.info["genus"] == 3

    def test_the_walls_are_mostly_hexagons(self, cubic):
        counts = cubic.info["ring_counts"]
        assert counts[6] / sum(counts.values()) > 0.7

    def test_the_odd_rings_are_there_at_all(self, cubic):
        """A network needs negative curvature at its nodes, and on a
        hexagonal net that is heptagons. A cell without them would mean
        the nodes had been meshed as something other than saddles."""
        assert cubic.info["ring_counts"].get(7, 0) > 0

    def test_the_geometry_is_sp2_and_not_overlapping(self, cubic):
        geometry = cubic.info["geometry"]
        assert 1.30 < geometry["bond_min"] < 1.45
        assert 1.40 < geometry["bond_max"] < 1.60
        assert geometry["n_close_contacts"] == 0

    def test_it_validates_and_is_already_a_unit_cell(self, cubic):
        assert not run_basic_checks(cubic).errors
        report = cell_report(cubic)
        assert report["periodicity"] == "3D"
        # Periodic in every direction, so there is no vacuum to converge.
        assert report["image_separation"] is None

    def test_it_records_the_net_it_was_built_from(self, cubic):
        info = cubic.info
        assert info["structure_type"] == "nanotube_network"
        assert info["network_kind"] == "cubic"
        assert info["node_coordination"] == 6
        assert info["n_nodes"] == 1
