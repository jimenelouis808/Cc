"""Tests for controlled placement of dopants and functional groups.

The regression at the heart of this file: a dopant and a group used to land
on neighbouring carbons (each modification chose its sites blind to the
others), and the -OH hydrogen folded back onto the lattice.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from carbonforge.builders import (
    build_finite_nanoribbon,
    build_graphene_supercell,
    build_nanoribbon,
)
from carbonforge.defects import introduce_vacancies
from carbonforge.defects.stone_wales import stone_wales_defect
from carbonforge.dopants import dope_at_sites, dope_random
from carbonforge.functionalization import (
    find_sites,
    functionalize_at_sites,
    functionalize_random,
    make_graphitic_n,
)
from carbonforge.gui.params import apply_functionalization, apply_modifiers
from carbonforge.placement import (
    REGIONS,
    candidate_sites,
    choose_sites,
    classify_sites,
    occupied_atoms,
    parse_indices,
    plane_normal,
)
from carbonforge.topology import build_bond_graph
from carbonforge.utils.constants import HARD_MIN_DISTANCE


def _tags(atoms):
    return Counter(t for s in classify_sites(atoms).values() for t in s.tags)


@pytest.fixture(scope="module")
def flake():
    return build_finite_nanoribbon(5, 6, edge="armchair")


@pytest.fixture(scope="module")
def sheet():
    return build_graphene_supercell(6, 6)


class TestClassification:
    def test_passivated_edge_is_edge_not_basal(self):
        ribbon = build_nanoribbon(6, 4, edge="zigzag", passivate=True)
        tags = _tags(ribbon)
        assert tags["terminated_edge"] == 8 and tags["edge_zigzag"] == 8
        assert tags["basal"] == 40

    def test_periodic_ribbon_has_no_false_rings(self):
        # Cycles that wrap the periodic axis are not rings.
        tags = _tags(build_nanoribbon(6, 4, edge="zigzag"))
        assert tags["octagon"] == 0 and tags["near_defect"] == 0

    def test_finite_flake_edges(self, flake):
        tags = _tags(flake)
        assert tags["edge_armchair"] > 0 and tags["edge_zigzag"] > 0 and tags["corner"] > 0

    def test_vacancy_rim(self, sheet):
        tags = _tags(introduce_vacancies(sheet, sites=[20]))
        assert tags["vacancy_rim"] == 3 and tags["edge"] == 0

    def test_stone_wales_rings(self, sheet):
        tags = _tags(stone_wales_defect(sheet, seed=0))
        # 2 pentagons (10 atoms) and 2 heptagons sharing the rotated bond (12).
        assert tags["pentagon"] == 10 and tags["heptagon"] == 12
        assert tags["vacancy_rim"] == 0 and tags["near_defect"] > 0

    def test_regions_are_known(self, flake):
        for region in REGIONS:
            candidate_sites(flake, region)
        with pytest.raises(ValueError, match="Región desconocida"):
            candidate_sites(flake, "centro")

    def test_corners_excluded_unless_asked(self, flake):
        info = classify_sites(flake)
        zigzag = candidate_sites(flake, "edge_zigzag", info=info)
        assert not any("corner" in info[i].tags for i in zigzag)
        with_corners = candidate_sites(flake, "edge_zigzag", info=info, include_corners=True)
        assert len(with_corners) > len(zigzag)


class TestStoneWalesFix:
    @pytest.mark.parametrize("seed", range(5))
    def test_no_collapse(self, sheet, seed):
        """Both atoms of the rotated bond used to collapse onto its midpoint."""
        out = stone_wales_defect(sheet, seed=seed)
        d = out.get_all_distances(mic=True)
        np.fill_diagonal(d, np.inf)
        assert d.min() > 1.3


class TestVacancies:
    def test_only_carbons_removed(self):
        ribbon = build_nanoribbon(6, 4, edge="zigzag", passivate=True)
        for seed in range(10):
            out = introduce_vacancies(ribbon, n_defects=2, seed=seed)
            assert Counter(out.get_chemical_symbols())["H"] == 8

    def test_explicit_hydrogen_refused(self):
        ribbon = build_nanoribbon(6, 4, edge="zigzag", passivate=True)
        hydrogen = ribbon.get_chemical_symbols().index("H")
        with pytest.raises(ValueError, match="only carbons"):
            introduce_vacancies(ribbon, sites=[hydrogen])


class TestChoose:
    def test_separation_and_exclusion(self, flake):
        doped = dope_at_sites(flake, "B", count=1, region="basal", seed=0)
        avoid = occupied_atoms(doped)
        picked = choose_sites(doped, candidate_sites(doped, "basal"), 4, seed=1,
                              min_separation=3.0, avoid=avoid, avoid_radius=2.6)
        d = doped.get_all_distances(mic=True)
        assert all(d[i, j] >= 3.0 for i in picked for j in picked if i != j)
        assert all(d[i, a] >= 2.6 for i in picked for a in avoid)

    def test_reproducible(self, flake):
        c = candidate_sites(flake, "basal")
        assert choose_sites(flake, c, 3, seed=5) == choose_sites(flake, c, 3, seed=5)

    def test_impossible_request_says_how_many_fit(self, flake):
        with pytest.raises(ValueError, match="sitio"):
            choose_sites(flake, candidate_sites(flake, "edge_zigzag"), 999, seed=0)

    def test_parse_indices(self):
        assert parse_indices("3, 7 12-14") == [3, 7, 12, 13, 14]
        assert parse_indices("") == []


class TestDoping:
    @pytest.mark.parametrize("region", ["basal", "edge_zigzag", "edge_armchair", "edge"])
    def test_region_respected(self, flake, region):
        out = dope_at_sites(flake, "N", count=2, region=region, seed=3)
        info = classify_sites(out)
        for i in out.info["dopants"][-1]["indices"]:
            assert out[i].symbol == "N"
            assert region in info[i].tags

    def test_edge_nitrogen_is_pyridinic(self, flake):
        out = dope_at_sites(flake, "N", count=1, region="edge_zigzag", seed=0)
        n = out.info["dopants"][-1]["indices"][0]
        neighbours = [out[j].symbol for j in build_bond_graph(out).neighbors(n)]
        assert sorted(neighbours) == ["C", "C"]            # no N-H
        boron = dope_at_sites(flake, "B", count=1, region="edge_zigzag", seed=0)
        b = boron.info["dopants"][-1]["indices"][0]
        assert "H" in [boron[j].symbol for j in build_bond_graph(boron).neighbors(b)]

    def test_explicit_indices(self, flake):
        target = candidate_sites(flake, "basal")[0]
        out = dope_at_sites(flake, "S", indices=[target])
        assert out[target].symbol == "S" and out.info["doping_region"] == "índices"
        with pytest.raises(ValueError, match="no son carbonos"):
            dope_at_sites(flake, "S", indices=[flake.get_chemical_symbols().index("H")])

    def test_defect_sites(self, sheet):
        defective = stone_wales_defect(sheet, seed=0)
        out = dope_at_sites(defective, "N", count=1, region="pentagon", seed=0)
        n = out.info["dopants"][-1]["indices"][0]
        assert "pentagon" in classify_sites(out)[n].tags

    def test_graphitic_n_never_on_a_passivated_edge(self):
        ribbon = build_nanoribbon(6, 4, edge="zigzag", passivate=True)
        for seed in range(15):
            out = make_graphitic_n(ribbon, n_sites=2, seed=seed)
            for i in out.info["nitrogen_configurations"][-1]["indices"]:
                assert all(out[j].symbol != "H" for j in build_bond_graph(out).neighbors(i))


class TestGroups:
    def test_group_replaces_edge_hydrogen(self, flake):
        out = functionalize_at_sites(flake, "NH2", count=2, region="edge_armchair", seed=0)
        assert Counter(out.get_chemical_symbols())["N"] == 2
        # Two H removed, four H added by the two -NH2.
        n_h = Counter(flake.get_chemical_symbols())["H"]
        assert Counter(out.get_chemical_symbols())["H"] == n_h + 2

    def test_groups_keep_clear_of_dopants(self, flake):
        doped = dope_at_sites(flake, "N", count=2, region="basal", seed=1)
        for seed in range(10):
            out = functionalize_at_sites(doped, "OH", count=3, region="basal", seed=seed)
            d = out.get_all_distances(mic=True)
            nitrogens = [i for i, s in enumerate(out.get_chemical_symbols()) if s == "N"]
            anchors = [r["anchor"] for r in out.info["functionalization"]]
            assert all(d[a, n] >= 2.6 for a in anchors for n in nitrogens)

    def test_avoid_can_be_switched_off(self, flake):
        doped = dope_at_sites(flake, "N", count=1, region="basal", seed=0)
        n = doped.info["dopants"][-1]["indices"][0]
        neighbour = next(j for j in build_bond_graph(doped).neighbors(n)
                         if doped[j].symbol == "C" and "basal" in classify_sites(doped)[j].tags)
        out = functionalize_at_sites(doped, "OH", indices=[neighbour])
        assert out.info["functionalization"][-1]["anchor"] == neighbour

    def test_face(self, flake):
        normal = plane_normal(flake, "+")
        up = functionalize_at_sites(flake, "OH", count=2, region="basal", seed=0, face="+")
        down = functionalize_at_sites(flake, "OH", count=2, region="basal", seed=0, face="-")
        centre = flake.positions.mean(axis=0)

        def side(atoms):
            oxygens = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "O"]
            offset = atoms.positions[oxygens] - atoms.positions[:len(flake)].mean(0)
            return np.sign(offset @ normal)

        assert np.all(side(up) > 0) and np.all(side(down) < 0)
        assert centre is not None

    def test_vacancy_rim_only_when_asked(self, sheet):
        holed = introduce_vacancies(sheet, sites=[20])
        with pytest.raises(ValueError, match="No hay sitios 'edge'"):
            functionalize_at_sites(holed, "OH", count=1, region="edge")
        out = functionalize_at_sites(holed, "OH", count=1, region="vacancy_rim", seed=0)
        assert out.info["functionalization"][-1]["region"] == "vacancy_rim"

    def test_carbonyl_needs_an_edge(self, sheet):
        with pytest.raises(ValueError):
            functionalize_at_sites(sheet, "O", count=1, region="basal")

    def test_find_sites_skips_terminated_carbons(self):
        ribbon = build_nanoribbon(6, 4, edge="zigzag", passivate=True)
        assert find_sites(ribbon, kind="edge") == []
        assert all(ribbon[s.index].symbol == "C" for s in find_sites(ribbon, kind="basal"))
        assert len(find_sites(ribbon, kind="basal")) == 40


class TestGuiFlow:
    """The exact path the window takes, where the user saw the problem."""

    def _build(self, seed, site, extra=None):
        ribbon = build_nanoribbon(6, 4, edge="zigzag")
        ribbon = apply_modifiers(ribbon, {"dopant": "N", "dopant_concentration": 0.05,
                                          "vacancies": 0, "seed": seed})
        values = {"group": "OH", "group_count": 2, "group_site": site, "seed": seed}
        values.update(extra or {})
        return apply_functionalization(ribbon, values)

    @pytest.mark.parametrize("site", ["edge", "basal"])
    def test_no_group_next_to_a_dopant(self, site):
        for seed in range(12):
            try:
                out = self._build(seed, site)
            except ValueError:
                continue                          # "only N sites fit": an honest refusal
            graph = build_bond_graph(out)
            symbols = out.get_chemical_symbols()
            for record in out.info["functionalization"]:
                near = set(graph.neighbors(record["anchor"]))
                assert not any(symbols[j] == "N" and j in near for j in near)
            d = out.get_all_distances(mic=True)
            np.fill_diagonal(d, np.inf)
            assert d.min() >= HARD_MIN_DISTANCE

    def test_region_and_indices_from_the_form(self):
        ribbon = build_nanoribbon(6, 4, edge="zigzag", passivate=True)
        doped = apply_modifiers(ribbon, {"dopant": "B", "dopant_count": 2,
                                         "dopant_region": "basal", "seed": 0})
        assert Counter(doped.get_chemical_symbols())["B"] == 2
        target = candidate_sites(doped, "basal")[-1]
        out = apply_functionalization(doped, {"group": "OH", "group_indices": str(target),
                                              "group_face": "-", "seed": 0})
        assert out.info["functionalization"][-1]["anchor"] == target

    def test_random_doping_unchanged_by_default(self):
        ribbon = build_nanoribbon(6, 4, edge="zigzag")
        a = apply_modifiers(ribbon, {"dopant": "N", "dopant_concentration": 0.1, "seed": 4})
        b = dope_random(ribbon, "N", 0.1, seed=4)
        assert a.get_chemical_symbols() == b.get_chemical_symbols()


def test_functionalize_random_now_replaces_edge_hydrogen():
    ribbon = build_nanoribbon(6, 4, edge="zigzag", passivate=True)
    out = functionalize_random(ribbon, "OH", n_groups=2, seed=0)
    assert Counter(out.get_chemical_symbols())["O"] == 2
