"""Tests for validation and topology modules."""

from __future__ import annotations

import numpy as np

from nanocarbon_lab.builders import build_cnt, build_graphene_supercell
from nanocarbon_lab.topology import (
    build_bond_graph,
    connected_components,
    coordination_numbers,
    ring_statistics,
)
from nanocarbon_lab.validation import (
    check_cell_consistency,
    check_minimum_distances,
    check_vacuum,
    run_basic_checks,
)


class TestTopology:
    def test_bond_graph_has_nodes(self):
        gr = build_graphene_supercell(2, 2)
        g = build_bond_graph(gr)
        assert g.number_of_nodes() == len(gr)
        assert g.number_of_edges() > 0

    def test_graphene_is_mostly_3_coordinated(self):
        gr = build_graphene_supercell(3, 3)
        coord = coordination_numbers(gr)
        # With PBC every bulk C should be 3-coordinated.
        assert int(np.mean(coord == 3) * 100) >= 90

    def test_graphene_is_single_component(self):
        gr = build_graphene_supercell(3, 3)
        comps = connected_components(gr)
        assert len(comps) == 1

    def test_ring_statistics_hexagons_dominate(self):
        gr = build_graphene_supercell(3, 3)
        stats = ring_statistics(gr, max_ring=8)
        assert stats[6] > 0
        # Hexagons should dominate over smaller rings in pristine graphene.
        assert stats[6] >= stats[5]


class TestValidation:
    def test_pristine_graphene_passes(self):
        gr = build_graphene_supercell(3, 3)
        rep = run_basic_checks(gr)
        assert rep.ok, rep.summary()

    def test_pristine_cnt_passes(self):
        cnt = build_cnt(6, 6, length=8)
        rep = run_basic_checks(cnt)
        assert rep.ok, rep.summary()

    def test_overlap_detected(self):
        gr = build_graphene_supercell(2, 2)
        pos = gr.get_positions()
        pos[1] = pos[0] + np.array([0.3, 0.0, 0.0])  # overlap
        gr.set_positions(pos)
        rep = check_minimum_distances(gr)
        assert not rep.ok

    def test_vacuum_insufficient_flagged(self):
        cnt = build_cnt(5, 5, length=5, vacuum=1.0)
        rep = check_vacuum(cnt, min_vacuum=5.0)
        assert not rep.ok

    def test_cell_consistency_passes(self):
        gr = build_graphene_supercell(2, 2)
        assert check_cell_consistency(gr).ok
