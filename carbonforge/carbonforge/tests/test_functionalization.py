"""Tests for functional groups and nitrogen configurations."""

from __future__ import annotations

import numpy as np
import pytest

from carbonforge.builders import (
    build_cnt,
    build_graphene_supercell,
    build_nanoribbon,
)
from carbonforge.functionalization import (
    GROUPS,
    NITROGEN_GROUPS,
    coverage,
    describe_groups,
    find_bridge_sites,
    find_sites,
    functionalize,
    functionalize_bridges,
    functionalize_random,
    get_group,
    make_graphitic_n,
    make_pyridinic_n,
    make_pyridinic_n_oxide,
    make_pyrrolic_like,
    nitrogen_report,
    passivate_edges,
)
from carbonforge.topology import build_bond_graph, coordination_numbers
from carbonforge.utils.constants import HARD_MIN_DISTANCE
from carbonforge.validation import run_basic_checks


def _ribbon():
    return build_nanoribbon(6, 3, edge="zigzag")


class TestGroupGeometry:
    """Every group's internal geometry must be chemically sane."""

    @pytest.mark.parametrize("key", sorted(GROUPS))
    def test_internal_distances_are_physical(self, key):
        group = get_group(key)
        if len(group) < 2:
            return
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                distance = np.linalg.norm(group.positions[i] - group.positions[j])
                assert distance > HARD_MIN_DISTANCE, (
                    f"{key}: átomos {i}-{j} a {distance:.3f} Å"
                )

    @pytest.mark.parametrize("key", sorted(GROUPS))
    def test_group_points_away_from_the_sheet(self, key):
        """No atom may fold back below the anchor carbon.

        Regression test for the nitro group, whose oxygens were parameterised
        with the O-N-O angle but transformed as if it were a C-N-O angle. They
        ended up 0.9 Å above the anchor and 1.1 Å sideways — sitting on top of
        the neighbouring ring carbons, which the bond graph then read as
        four-coordinate oxygen.
        """
        group = get_group(key)
        z_values = group.positions[:, 2]
        assert z_values.min() > 0.3, (
            f"{key}: algún átomo queda a z={z_values.min():.2f}, "
            "demasiado cerca del plano del anclaje"
        )

    @pytest.mark.parametrize("key", sorted(GROUPS))
    def test_first_atom_sits_at_a_bond_length(self, key):
        """The atom bonded to the anchor must be at a plausible bond length."""
        group = get_group(key)
        distance = float(np.linalg.norm(group.positions[0]))
        assert 0.9 < distance < 2.0, f"{key}: primer átomo a {distance:.3f} Å"

    def test_describe_lists_every_group(self):
        text = describe_groups()
        for key in GROUPS:
            assert key in text

    def test_unknown_group_lists_options(self):
        with pytest.raises(ValueError, match="Disponibles"):
            get_group("XYZ")


class TestSites:
    def test_ribbon_has_edge_and_basal_sites(self):
        ribbon = _ribbon()
        assert find_sites(ribbon, "edge")
        assert find_sites(ribbon, "basal")

    def test_periodic_graphene_has_no_edges(self):
        """A fully periodic sheet has no under-coordinated carbon."""
        assert find_sites(build_graphene_supercell(3, 3), "edge") == []

    def test_edge_direction_points_outward(self):
        """The dangling bond must point away from the ribbon, not into it."""
        ribbon = _ribbon()
        centre = ribbon.get_positions().mean(axis=0)
        for site in find_sites(ribbon, "edge"):
            outward = site.origin - centre
            outward_in_plane = np.array([outward[0], 0.0, outward[2]])
            if np.linalg.norm(outward_in_plane) < 1e-6:
                continue
            outward_in_plane /= np.linalg.norm(outward_in_plane)
            direction_in_plane = np.array(
                [site.direction[0], 0.0, site.direction[2]]
            )
            if np.linalg.norm(direction_in_plane) < 1e-6:
                continue
            direction_in_plane /= np.linalg.norm(direction_in_plane)
            assert np.dot(outward_in_plane, direction_in_plane) > -0.2

    def test_basal_normal_is_perpendicular_to_the_sheet(self):
        """The ribbon lies in x-z, so its normal must be along y."""
        for site in find_sites(_ribbon(), "basal"):
            assert abs(abs(site.direction[1]) - 1.0) < 0.05

    def test_cnt_basal_normals_point_outward(self):
        """On a tube the normal must point away from the axis."""
        tube = build_cnt(6, 6, length=8)
        centre = tube.get_positions().mean(axis=0)
        for site in find_sites(tube, "basal")[:20]:
            radial = site.origin - centre
            radial[2] = 0.0
            if np.linalg.norm(radial) < 1e-6:
                continue
            radial /= np.linalg.norm(radial)
            assert np.dot(radial, site.direction) > 0.5

    def test_bridge_sites_found_on_graphene(self):
        assert find_bridge_sites(build_graphene_supercell(3, 3))


class TestAttaching:
    @pytest.mark.parametrize(
        "key", ["H", "OH", "NH2", "NO2", "CN", "COOH", "CHO", "CONH2", "SH", "CH3"]
    )
    def test_edge_attachment_gives_valid_structure(self, key):
        out = functionalize_random(_ribbon(), key, n_groups=2, seed=0)
        report = run_basic_checks(out)
        assert report.ok, f"{key}:\n{report.summary()}"

    def test_carbonyl_rejected_on_basal_site(self):
        """=O consumes two valences, so it cannot go on a 3-coordinate C."""
        with pytest.raises(ValueError, match="dos valencias"):
            functionalize_random(
                _ribbon(), "O", n_groups=1, site_kind="basal", seed=0
            )

    def test_epoxide_needs_the_bridge_helper(self):
        with pytest.raises(ValueError, match="puente"):
            functionalize_random(_ribbon(), "epoxy", n_groups=1, seed=0)

    def test_atom_count_grows_by_group_size(self):
        ribbon = _ribbon()
        out = functionalize_random(ribbon, "COOH", n_groups=2, seed=0)
        assert len(out) == len(ribbon) + 2 * len(get_group("COOH"))

    def test_reproducible_with_seed(self):
        a = functionalize_random(_ribbon(), "NH2", n_groups=3, seed=7)
        b = functionalize_random(_ribbon(), "NH2", n_groups=3, seed=7)
        assert a.get_chemical_symbols() == b.get_chemical_symbols()
        np.testing.assert_allclose(a.get_positions(), b.get_positions())

    def test_overlap_is_caught(self):
        """Crowding groups onto neighbours must fail loudly, not silently."""
        with pytest.raises(ValueError, match="mínimo físico"):
            functionalize_random(
                _ribbon(), "COOH", n_groups=6, seed=0, min_separation=0.0
            )

    def test_vacuum_is_restored_after_attaching(self):
        """Groups protrude into the padding; it must be grown back."""
        out = functionalize_random(_ribbon(), "COOH", n_groups=2, seed=0)
        positions = out.get_positions()
        cell = np.array(out.cell)
        for axis, periodic in enumerate(out.get_pbc()):
            if periodic:
                continue
            span = np.ptp(positions[:, axis])
            assert cell[axis, axis] - span >= 11.9

    def test_too_many_groups_rejected(self):
        with pytest.raises(ValueError, match="solo hay"):
            functionalize_random(_ribbon(), "H", n_groups=999, seed=0)

    def test_no_edge_sites_gives_clear_message(self):
        with pytest.raises(ValueError, match="sin bordes"):
            functionalize_random(
                build_graphene_supercell(3, 3), "OH", n_groups=1, seed=0
            )

    def test_explicit_indices(self):
        ribbon = _ribbon()
        index = find_sites(ribbon, "edge")[0].index
        out = functionalize(ribbon, "H", indices=[index])
        assert len(out) == len(ribbon) + 1

    def test_invalid_index_rejected(self):
        with pytest.raises(ValueError, match="no es un sitio"):
            functionalize(_ribbon(), "H", indices=[9999])

    def test_passivation_saturates_every_edge(self):
        ribbon = _ribbon()
        n_edges = len(find_sites(ribbon, "edge"))
        out = passivate_edges(ribbon)
        hydrogens = sum(1 for s in out.get_chemical_symbols() if s == "H")
        assert hydrogens == n_edges

    def test_coverage_summary(self):
        out = functionalize_random(_ribbon(), "NH2", n_groups=2, seed=0)
        info = coverage(out)
        assert info["n_groups"] == 2
        assert info["groups"]["NH2"] == 2


class TestBasalAndBridges:
    def test_basal_attachment_on_graphene(self):
        out = functionalize_random(
            build_graphene_supercell(4, 4), "OH", n_groups=2,
            site_kind="basal", seed=0,
        )
        assert "O" in out.get_chemical_symbols()

    def test_epoxide_bridges(self):
        sheet = build_graphene_supercell(4, 4)
        out = functionalize_bridges(sheet, n_groups=2, seed=0)
        assert len(out) == len(sheet) + 2

    def test_epoxide_needs_room(self):
        with pytest.raises(ValueError, match="Solo caben"):
            functionalize_bridges(
                build_graphene_supercell(2, 2), n_groups=50, seed=0
            )


class TestNitrogenConfigurations:
    def test_graphitic_keeps_three_bonds(self):
        out = make_graphitic_n(build_graphene_supercell(4, 4), n_sites=2, seed=0)
        graph = build_bond_graph(out)
        symbols = out.get_chemical_symbols()
        nitrogens = [i for i, s in enumerate(symbols) if s == "N"]
        assert len(nitrogens) == 2
        for index in nitrogens:
            assert graph.degree[index] == 3

    def test_graphitic_preserves_atom_count(self):
        sheet = build_graphene_supercell(4, 4)
        out = make_graphitic_n(sheet, n_sites=3, seed=0)
        assert len(out) == len(sheet)

    def test_pyridinic_removes_a_carbon_and_lowers_coordination(self):
        sheet = build_graphene_supercell(5, 5)
        out = make_pyridinic_n(sheet, n_defects=1, n_per_vacancy=1, seed=0)
        assert len(out) == len(sheet) - 1
        graph = build_bond_graph(out)
        symbols = out.get_chemical_symbols()
        nitrogens = [i for i, s in enumerate(symbols) if s == "N"]
        assert nitrogens
        # Pyridinic N sits on a vacancy rim, so it is two-coordinate.
        assert all(graph.degree[i] == 2 for i in nitrogens)

    def test_pyridinic_n3_variant(self):
        out = make_pyridinic_n(
            build_graphene_supercell(5, 5), n_defects=1, n_per_vacancy=3, seed=0
        )
        assert sum(1 for s in out.get_chemical_symbols() if s == "N") == 3

    def test_pyridinic_rejects_bad_count(self):
        with pytest.raises(ValueError, match="n_per_vacancy"):
            make_pyridinic_n(build_graphene_supercell(4, 4), n_per_vacancy=5)

    def test_pyrrolic_precursor_is_labelled_honestly(self):
        """It is a precursor until relaxed; the metadata must say so."""
        out = make_pyrrolic_like(build_graphene_supercell(5, 5), seed=0)
        entry = out.info["nitrogen_configurations"][-1]
        assert entry["type"] == "pyrrolic_precursor"
        assert "warning" in entry
        assert "N" in out.get_chemical_symbols()
        assert "H" in out.get_chemical_symbols()

    def test_n_oxide_adds_oxygen_on_nitrogen(self):
        out = make_pyridinic_n_oxide(build_graphene_supercell(5, 5), seed=0)
        symbols = out.get_chemical_symbols()
        assert "N" in symbols and "O" in symbols
        graph = build_bond_graph(out)
        oxygens = [i for i, s in enumerate(symbols) if s == "O"]
        assert any(
            any(symbols[j] == "N" for j in graph.neighbors(i)) for i in oxygens
        )

    def test_report_without_nitrogen(self):
        assert "no contiene nitrógeno" in nitrogen_report(
            build_graphene_supercell(2, 2)
        )

    def test_report_counts_and_interprets(self):
        out = make_graphitic_n(build_graphene_supercell(4, 4), n_sites=2, seed=0)
        text = nitrogen_report(out)
        assert "grafítico" in text
        assert "% at." in text

    def test_report_surfaces_precursor_warning(self):
        out = make_pyrrolic_like(build_graphene_supercell(5, 5), seed=0)
        assert "⚠️" in nitrogen_report(out)

    def test_nitrogen_group_list_is_coherent(self):
        for key in NITROGEN_GROUPS:
            assert "N" in get_group(key).symbols
