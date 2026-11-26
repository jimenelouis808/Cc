"""Goldberg-Coxeter superstructures: flat patches, defects at map vertices."""

import numpy as np
import pytest

from nanocarbon_lab.analyse.curvature import disclination_check
from nanocarbon_lab.builders import goldberg as gc
from nanocarbon_lab.validation.quality import sp2_quality


def _degrees(tris):
    count = {}
    for tri in tris:
        for v in tri:
            count[int(v)] = count.get(int(v), 0) + 1
    return count


class TestOperations:
    def test_subdivide_keeps_coarse_degrees_and_adds_only_sixes(self):
        coarse, box = gc.primitive_octagon_map()
        before = _degrees(coarse[1])
        vertices, tris = gc.subdivide(coarse, 3, box)
        after = _degrees(tris)
        assert len(tris) == 9 * len(coarse[1])
        for v, d in before.items():
            assert after[v] == d
        assert {d for v, d in after.items() if v >= len(coarse[0])} == {6}

    def test_root_three_triples_faces_and_keeps_degrees(self):
        coarse, box = gc.primitive_octagon_map()
        vertices, tris = gc.root_three(coarse, box)
        assert len(tris) == 3 * len(coarse[1])
        assert gc.degree_census(tris) == {6: 8 + 48, 8: 12}

    def test_chiral_indices_are_refused(self):
        coarse, box = gc.primitive_octagon_map()
        with pytest.raises(ValueError, match="chiral"):
            gc.goldberg_coxeter(coarse, 2, 1, box)

    def test_triangulation_number(self):
        assert gc.triangulation_number(1, 1) == 3
        assert gc.triangulation_number(3, 0) == 9


class TestMaps:
    def test_muoctahedron_is_genus_three(self):
        (vertices, tris), _ = gc.primitive_octagon_map()
        edges = {tuple(sorted((int(a), int(b)))) for t in tris
                 for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        assert len(vertices) - len(edges) + len(tris) == -4
        assert gc.degree_census(tris) == {6: 8, 8: 12}

    def test_square_layer_folds_to_genus_two(self):
        coarse, box = gc.square_octagon_map()
        fine = gc.subdivide(coarse, 2, box)
        vertices, tris = gc.fold_mesh(fine, box, box / np.array([2, 2, 1]))
        edges = {tuple(sorted((int(a), int(b)))) for t in tris
                 for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        assert len(vertices) - len(edges) + len(tris) == -2


class TestBuild:
    @pytest.mark.parametrize("m, n, atoms", [(2, 0, 192), (1, 1, 144),
                                             (3, 0, 432)])
    def test_primitive_is_twelve_octagons_and_clean(self, m, n, atoms):
        built = gc.build_goldberg("primitive", m, n)
        assert len(built) == atoms
        census = built.info["ring_counts"]
        assert set(census) == {6, 8} and census[8] == 12
        assert built.info["ring_deficit"] == -24
        verdict, _ = sp2_quality(built.info["geometry"])
        assert str(verdict) == "clean"

    def test_super_square_is_four_poles_eight_saddles_and_clean(self):
        built = gc.build_goldberg("super-square", 2, 0)
        assert built.info["ring_counts"] == {5: 4, 6: 90, 8: 8}
        assert tuple(built.pbc) == (True, True, False)
        verdict, _ = sp2_quality(built.info["geometry"])
        assert str(verdict) == "clean"

    def test_super_square_needs_a_refined_map(self):
        with pytest.raises(ValueError, match="m >= 2"):
            gc.build_goldberg("super-square", 1, 0)

    @pytest.mark.parametrize("kind", ["primitive", "super-square"])
    def test_every_disclination_on_its_own_side(self, kind):
        built = gc.build_goldberg(kind, 3, 0)
        box = np.where(built.pbc, built.cell.lengths(), 0.0)
        check = disclination_check(built.positions, built.info["rings"],
                                   built.info["bonds"], box=box)
        assert check["agreement"] == 1.0

    def test_unknown_kind_names_the_choices(self):
        with pytest.raises(ValueError, match="primitive"):
            gc.build_goldberg("hexagonal-nonsense")


class TestGeneralRoute:
    def test_refine_is_identity_at_one(self):
        coarse, box = gc.primitive_octagon_map()
        assert gc.refine_remeshed(coarse, 1, lambda p: p[:, 0], box) is coarse

    def test_refine_subdivides_and_lands_on_the_field(self):
        coarse, box = gc.primitive_octagon_map()
        k = 2.0 * np.pi / box[0]

        def field(p):
            return np.cos(k * p[:, 0]) + np.cos(k * p[:, 1]) + \
                np.cos(k * p[:, 2])
        vertices, tris = gc.refine_remeshed(coarse, 2, field, box)
        assert len(tris) == 4 * len(coarse[1])
        assert np.abs(field(vertices)).max() < 1e-6

    def test_the_meshed_builders_take_the_option(self):
        import inspect

        from nanocarbon_lab.builders.junction import build_schwarzite
        from nanocarbon_lab.builders.supernetwork import build_supernetwork

        for builder in (build_schwarzite, build_supernetwork):
            assert "goldberg" in inspect.signature(builder).parameters


class TestDiamondFromC168:
    def test_c168_rebuilds_from_the_papers_table(self):
        positions, pairs, box = gc.c168_atoms()
        assert len(positions) == 672
        lengths = []
        for i, j in pairs:
            d = positions[j] - positions[i]
            d -= box * np.round(d / box)
            lengths.append(np.linalg.norm(d))
        assert 1.38 < min(lengths) and max(lengths) < 1.46

    def test_the_klein_map_is_96_heptagon_vertices(self):
        (vertices, tris), _ = gc.diamond_heptagon_map()
        assert gc.degree_census(tris) == {7: 96}
        assert len(tris) == 224

    def test_class_two_one_is_c168_itself(self):
        built = gc.build_goldberg("diamond", 1, 1)
        assert len(built) == 672
        assert built.info["ring_counts"] == {6: 224, 7: 96}
        assert abs(built.cell[0][0] - gc.C168_LATTICE) < 0.05
        verdict, _ = sp2_quality(built.info["geometry"])
        assert str(verdict) == "clean"

    @pytest.mark.parametrize("m, atoms", [(1, 224), (2, 896)])
    def test_class_one_is_clean_and_all_heptagons_are_placed(self, m, atoms):
        built = gc.build_goldberg("diamond", m, 0)
        assert len(built) == atoms
        assert built.info["ring_counts"][7] == 96
        assert set(built.info["ring_counts"]) <= {6, 7}
        verdict, _ = sp2_quality(built.info["geometry"])
        assert str(verdict) == "clean"
        assert built.info["disclinations_placed"] == 1.0

    def test_the_count_estimate_is_exact(self):
        assert gc.goldberg_atom_count("diamond", 1, 1) == 672
        assert gc.goldberg_atom_count("diamond", 2, 0) == 896
