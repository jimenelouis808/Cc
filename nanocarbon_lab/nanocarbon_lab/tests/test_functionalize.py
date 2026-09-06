"""Surface functionalisation: the group grammar and the grafting.

The tests are grouped by the claim they defend rather than by function,
because every one of them corresponds to something that was wrong first
and would be silently wrong again.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import build_cnt, build_fullerene, build_graphene
from nanocarbon_lab.functionalize.attach import (
    INNER_BLOCKED_DEPTH,
    PYRAMIDAL_SUM,
    _components,
    bond_pairs,
    candidate_sites,
    describe_functionalization,
    enclosure,
    functionalize,
    inner_face_blocked,
    is_enclosing,
    preserve_vacuum,
    sublattice_parity,
    surface_normals,
)
from nanocarbon_lab.functionalize.groups import (
    GROUPS,
    VALENCE,
    bond_length,
    build_bridging_positions,
    build_positions,
    describe,
    get_group,
    substitute,
    viable_swaps,
)
from nanocarbon_lab.tmd import build_tmd_monolayer
from nanocarbon_lab.utils.constants import COVALENT_RADII, MAX_COORDINATION
from nanocarbon_lab.utils.metadata import keep_indices, remap_after_removal
from nanocarbon_lab.validation import run_basic_checks

ORIGIN = np.zeros(3)
UP = np.array([0.0, 0.0, 1.0])


def _angle(first: np.ndarray, vertex: np.ndarray, second: np.ndarray) -> float:
    """Bond angle at ``vertex``, in degrees."""
    left, right = first - vertex, second - vertex
    cosine = np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


class TestGroupGeometry:
    """The built geometry must match the literature, not just be self-consistent."""

    # (group, bonded pair as (index_or_-1, index), expected Å, tolerance)
    BONDS = [
        ("hydroxyl", (-1, 0), 1.43, 0.02),   # C-O in an alcohol
        ("hydroxyl", (0, 1), 0.97, 0.02),    # O-H
        ("thiol", (-1, 0), 1.82, 0.02),      # C-S
        ("thiol", (0, 1), 1.34, 0.03),       # S-H
        ("amine", (-1, 0), 1.47, 0.02),      # C-N in aniline
        ("amine", (0, 1), 1.01, 0.02),       # N-H
        ("methyl", (-1, 0), 1.53, 0.02),     # C-C sp3
        ("carbonyl", (-1, 0), 1.23, 0.02),   # C=O
        ("carboxyl", (0, 1), 1.23, 0.02),    # C=O of the acid
        ("carboxyl", (0, 2), 1.34, 0.03),    # conjugated C-OH, not 1.43
        ("fluorine", (-1, 0), 1.35, 0.03),   # C-F
    ]

    @pytest.mark.parametrize("name,pair,expected,tolerance", BONDS)
    def test_bond_lengths_match_the_literature(self, name, pair, expected,
                                               tolerance):
        group = get_group(name)
        _, positions = build_positions(group, ORIGIN, UP, "C")
        parent, child = pair
        start = ORIGIN if parent < 0 else positions[parent]
        measured = float(np.linalg.norm(positions[child] - start))
        assert measured == pytest.approx(expected, abs=tolerance), (
            f"{name} bond {pair} came out {measured:.3f} Å against "
            f"{expected} Å"
        )

    @pytest.mark.parametrize("name,expected", [
        ("hydroxyl", 108.5), ("thiol", 97.0), ("amine", 112.0),
        ("methyl", 109.5), ("carboxyl", 121.0), ("nitro", 127.0),
    ])
    def test_the_stored_angle_is_the_bond_angle(self, name, expected):
        """The registry holds angles as chemistry quotes them.

        The Z-matrix works from the parent bond's continuation, so a
        109.5 degree centre is a 70.5 degree deflection. Storing the
        deflection would make every group definition carry a supplement
        nobody would recognise, and the conversion belongs in one place.
        """
        group = get_group(name)
        _, positions = build_positions(group, ORIGIN, UP, "C")
        child = next(index for index, atom in enumerate(group.atoms)
                     if atom.parent == 0)
        measured = _angle(ORIGIN, positions[0], positions[child])
        assert measured == pytest.approx(expected, abs=0.5)

    def test_no_group_overlaps_itself(self):
        for name, group in GROUPS.items():
            if group.bridging:
                continue
            _, positions = build_positions(group, ORIGIN, UP, "C")
            points = np.vstack([ORIGIN, positions])
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    separation = float(np.linalg.norm(points[i] - points[j]))
                    assert separation > 0.9, f"{name}: atoms {i},{j} at {separation:.2f} Å"

    def test_the_root_goes_along_the_normal(self):
        for name, group in GROUPS.items():
            if group.bridging:
                continue
            _, positions = build_positions(group, ORIGIN, UP, "C")
            direction = positions[0] / np.linalg.norm(positions[0])
            assert np.dot(direction, UP) > 0.99, name

    def test_twist_rotates_the_group_without_deforming_it(self):
        """A single bond to the surface is a free rotation.

        The whole group must turn rigidly: every internal distance is
        unchanged and the root, which lies on the axis, does not move.
        """
        group = get_group("carboxyl")
        _, straight = build_positions(group, ORIGIN, UP, "C")
        _, turned = build_positions(group, ORIGIN, UP, "C", twist=90.0)
        assert np.allclose(straight[0], turned[0], atol=1e-9)
        for i in range(len(straight)):
            for j in range(i + 1, len(straight)):
                assert np.linalg.norm(straight[i] - straight[j]) == pytest.approx(
                    float(np.linalg.norm(turned[i] - turned[j])), abs=1e-9)
        assert not np.allclose(straight[1], turned[1], atol=1e-3)


class TestSubstitution:
    """Swapping an element must rebuild the lengths, not carry them over."""

    @pytest.mark.parametrize("target,expected", [
        ("S", 1.82), ("Se", 1.97), ("Te", 2.15),
    ])
    def test_the_new_bond_is_the_new_element_s_own(self, target, expected):
        """This is the property the whole Z-matrix design exists for.

        Stored as Cartesians, a hydroxyl with its oxygen renamed to
        sulphur would leave that sulphur at oxygen's 1.42 Å instead of
        its own 1.81 -- a 0.4 Å error in the one bond that defines the
        group.
        """
        swapped = substitute(get_group("hydroxyl"), {"O": target})
        _, positions = build_positions(swapped, ORIGIN, UP, "C")
        measured = float(np.linalg.norm(positions[0]))
        assert measured == pytest.approx(expected, abs=0.03)
        assert measured != pytest.approx(1.42, abs=0.05)

    def test_substituting_oxygen_for_sulphur_reproduces_the_shipped_thiol(self):
        made = substitute(get_group("hydroxyl"), {"O": "S"})
        shipped = get_group("thiol")
        _, from_swap = build_positions(made, ORIGIN, UP, "C")
        _, from_registry = build_positions(shipped, ORIGIN, UP, "C")
        # The bond lengths must agree exactly; the H sits at each
        # element's own angle, so only the lengths are compared.
        assert np.linalg.norm(from_swap[0]) == pytest.approx(
            float(np.linalg.norm(from_registry[0])), abs=1e-9)
        assert np.linalg.norm(from_swap[1] - from_swap[0]) == pytest.approx(
            float(np.linalg.norm(from_registry[1] - from_registry[0])), abs=1e-9)

    def test_a_cross_valence_swap_is_refused_with_the_reason(self):
        with pytest.raises(ValueError, match="bonds"):
            substitute(get_group("hydroxyl"), {"O": "N"})

    def test_an_unknown_element_is_refused(self):
        with pytest.raises(ValueError, match="Unknown element"):
            substitute(get_group("hydroxyl"), {"O": "Xx"})

    def test_viable_swaps_are_same_valence_and_have_radii(self):
        for element in VALENCE:
            for other in viable_swaps(element):
                assert VALENCE[other] == VALENCE[element]
                assert other in COVALENT_RADII
                assert other != element

    def test_every_viable_swap_actually_builds(self):
        """The suggestion in the error message must be actionable."""
        for group in GROUPS.values():
            for element in group.elements():
                for other in viable_swaps(element):
                    swapped = substitute(group, {element: other})
                    if group.bridging:
                        build_bridging_positions(
                            swapped, ORIGIN, np.array([1.42, 0.0, 0.0]), UP)
                    else:
                        build_positions(swapped, ORIGIN, UP, "C")


class TestSurfaceNormals:
    """Which way is out, on every shape the framework builds."""

    def test_a_tube_s_normals_are_radial(self):
        cnt = build_cnt(6, 6, length=12.0)
        normals = surface_normals(cnt)
        positions = cnt.get_positions()
        axis = positions[:, :2].mean(axis=0)
        radial = positions[:, :2] - axis
        radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        projection = np.einsum("ij,ij->i", normals[:, :2], radial)
        assert projection.min() > 0.99

    def test_a_cage_s_normals_point_away_from_its_centre(self):
        cage = build_fullerene(freq=1, family="C60")
        normals = surface_normals(cage)
        positions = cage.get_positions()
        outward = positions - positions.mean(axis=0)
        outward /= np.linalg.norm(outward, axis=1, keepdims=True)
        assert np.einsum("ij,ij->i", normals, outward).min() > 0.9

    def test_a_sheet_s_normals_are_all_the_same_way(self):
        sheet = build_graphene().repeat((4, 4, 1))
        normals = surface_normals(sheet)
        assert np.allclose(np.abs(normals[:, 2]), 1.0, atol=1e-6)
        assert (normals[:, 2] > 0).all() or (normals[:, 2] < 0).all()

    def test_inner_and_outer_are_opposite(self):
        cnt = build_cnt(6, 6, length=12.0)
        assert np.allclose(surface_normals(cnt, face="outer"),
                           -surface_normals(cnt, face="inner"), atol=1e-9)

    def test_a_pyramidal_site_does_not_use_the_plane_normal(self):
        """An MX2 chalcogen's three bonds are not coplanar.

        The plane-normal branch has no plane to find there and returns
        whichever direction the three bond vectors happen to vary least
        along, which refused every group on every MoS2 surface. The
        leaning rule is the correct one for such a site.
        """
        slab = build_tmd_monolayer("MoS2").repeat((3, 3, 1))
        normals = surface_normals(slab)
        positions = slab.get_positions()
        symbols = slab.get_chemical_symbols()
        middle = positions[:, 2].mean()
        for index, symbol in enumerate(symbols):
            if symbol != "S":
                continue
            expected = 1.0 if positions[index, 2] > middle else -1.0
            assert normals[index, 2] == pytest.approx(expected, abs=0.02)

    def test_the_pyramidal_threshold_separates_the_real_cases(self):
        """The constant is measured, so the measurements must hold."""
        def leaning(atoms, index):
            # Minimum image: on a periodic sheet a neighbour across the
            # seam is stored a whole cell away, and a raw offset would
            # make the flattest surface there is look pyramidal.
            pairs = bond_pairs(atoms)
            here = [int(j) for i, j in pairs if i == index]
            here += [int(i) for i, j in pairs if j == index]
            offsets = np.array([atoms.get_distance(index, other, mic=True,
                                                   vector=True)
                                for other in here])
            units = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
            return float(np.linalg.norm(units.sum(axis=0)))

        sheet = build_graphene().repeat((3, 3, 1))
        assert leaning(sheet, 0) < 0.05

        slab = build_tmd_monolayer("MoS2").repeat((3, 3, 1))
        sulphur = slab.get_chemical_symbols().index("S")
        assert leaning(slab, sulphur) > PYRAMIDAL_SUM

    def test_enclosure_tells_a_tube_from_a_sheet(self):
        cnt = build_cnt(6, 6, length=12.0)
        sheet = build_graphene().repeat((4, 4, 1))
        assert is_enclosing(cnt)
        assert not is_enclosing(sheet)
        assert enclosure(sheet, surface_normals(sheet),
                         list(range(len(sheet)))) == pytest.approx(0.0, abs=0.1)

    def test_a_multi_wall_tube_orients_each_shell_separately(self):
        """Each closed surface has its own outside.

        Orienting them together would put every group on the inner tube
        inside its own wall. The shells are disjoint components of the
        bond graph, so each gets its own flip -- which is why the
        divergence test is applied per component and not once globally.
        """
        from nanocarbon_lab.builders import build_multiwall_cnt

        mwnt = build_multiwall_cnt(n_shells=2, inner_freq=3, n_body_rings=6)
        normals = surface_normals(mwnt)
        positions = mwnt.get_positions()

        # These are capped tubes, so at a cap the normal runs along the
        # axis rather than radially; the property that holds everywhere
        # is that it points away from that shell's own centre.
        pairs = bond_pairs(mwnt)
        neighbours = [[] for _ in range(len(mwnt))]
        for first, second in pairs:
            neighbours[int(first)].append(int(second))
            neighbours[int(second)].append(int(first))
        shells = _components(neighbours, len(mwnt))
        assert len(shells) == 2, "the two walls must be separate components"

        for members in shells:
            centre = positions[members].mean(axis=0)
            outward = positions[members] - centre
            outward /= np.linalg.norm(outward, axis=1, keepdims=True)
            projection = np.einsum("ij,ij->i", normals[members], outward)
            # Strictly outward is the claim. Not *near* 1: on an
            # elongated capsule the ray from the centroid to a shoulder
            # atom meets the surface at a shallow angle, so 0.27 there
            # is the shape, not a misorientation.
            assert projection.min() > 0.05
            assert projection.mean() > 0.6

    def test_the_inner_wall_is_not_turned_inside_out(self):
        """The failure this guards is specific and silent.

        A single global flip satisfies the outer shell and reverses the
        inner one, so every group grafted onto the inner tube would be
        placed inside its own wall.
        """
        from nanocarbon_lab.builders import build_multiwall_cnt

        mwnt = build_multiwall_cnt(n_shells=2, inner_freq=3, n_body_rings=6)
        normals = surface_normals(mwnt)
        positions = mwnt.get_positions()
        radius = np.linalg.norm(positions[:, :2]
                                - positions[:, :2].mean(axis=0), axis=1)
        inner = radius < np.mean(mwnt.info["shell_radii"])
        # Every atom's normal carries it further from the common axis or
        # along it, never towards it, on both shells at once.
        for mask in (inner, ~inner):
            moved = positions[mask] + 0.5 * normals[mask]
            grew = (np.linalg.norm(moved[:, :2]
                                   - positions[:, :2].mean(axis=0), axis=1)
                    >= radius[mask] - 1e-9)
            assert grew.mean() > 0.95


class TestSiteSelection:
    def test_a_dichalcogenide_offers_only_its_chalcogens(self):
        """The metal is buried between two chalcogen planes.

        A group grafted onto it would be threaded through the surface,
        so this is enforced rather than left to the steric test.
        """
        slab = build_tmd_monolayer("MoS2").repeat((3, 3, 1))
        symbols = slab.get_chemical_symbols()
        sites = candidate_sites(slab)
        assert sites
        assert {symbols[i] for i in sites} == {"S"}

    def test_edge_selection_finds_an_open_end_and_nothing_on_a_closed_cage(self):
        cage = build_fullerene(freq=1, family="C60")
        assert candidate_sites(cage, where="edge") == []

    def test_ring_selection_needs_recorded_rings(self):
        sheet = build_graphene().repeat((3, 3, 1))
        with pytest.raises(ValueError, match="ring metadata"):
            candidate_sites(sheet, where="defect")

    def test_ring_selection_reads_the_recorded_rings(self):
        cage = build_fullerene(freq=1, family="C60")
        pentagons = candidate_sites(cage, where="ring:5")
        assert len(pentagons) == 60  # every C60 carbon is in one pentagon
        assert candidate_sites(cage, where="defect") == pentagons

    def test_an_unknown_selection_is_refused(self):
        sheet = build_graphene().repeat((3, 3, 1))
        with pytest.raises(ValueError, match="Unknown site selection"):
            candidate_sites(sheet, where="everywhere")


class TestGrafting:
    def test_groups_land_outside_the_wall(self):
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, "hydroxyl", coverage=0.1, seed=1)
        positions = grafted.get_positions()
        axis = cnt.get_positions()[:, :2].mean(axis=0)
        wall = np.linalg.norm(cnt.get_positions()[:, :2] - axis, axis=1).max()
        added = np.linalg.norm(positions[len(cnt):, :2] - axis, axis=1)
        assert added.min() > wall

    def test_the_anchor_bond_has_the_right_length(self):
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, "hydroxyl", coverage=0.1, seed=1)
        record = grafted.info["functionalization"]
        expected = bond_length("C", "O")
        for offset, site in enumerate(record["sites"]):
            root = len(cnt) + 2 * offset
            assert grafted.get_distance(site, root, mic=True) == pytest.approx(
                expected, abs=1e-6)

    def test_existing_atoms_keep_their_indices(self):
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, "hydroxyl", coverage=0.1, seed=1)
        assert np.allclose(grafted.get_positions()[:len(cnt)],
                           cnt.get_positions())
        assert (grafted.get_chemical_symbols()[:len(cnt)]
                == cnt.get_chemical_symbols())

    def test_the_same_seed_gives_the_same_structure(self):
        cnt = build_cnt(6, 6, length=12.0)
        first = functionalize(cnt, "hydroxyl", coverage=0.2, seed=7)
        second = functionalize(cnt, "hydroxyl", coverage=0.2, seed=7)
        assert np.allclose(first.get_positions(), second.get_positions())
        assert (first.info["functionalization"]["sites"]
                == second.info["functionalization"]["sites"])

    def test_a_different_seed_gives_a_different_pattern(self):
        cnt = build_cnt(6, 6, length=12.0)
        first = functionalize(cnt, "hydroxyl", coverage=0.2, seed=1)
        second = functionalize(cnt, "hydroxyl", coverage=0.2, seed=2)
        assert (first.info["functionalization"]["sites"]
                != second.info["functionalization"]["sites"])

    def test_count_and_coverage_are_mutually_exclusive(self):
        cnt = build_cnt(6, 6, length=12.0)
        with pytest.raises(ValueError, match="exactly one"):
            functionalize(cnt, "hydroxyl")
        with pytest.raises(ValueError, match="exactly one"):
            functionalize(cnt, "hydroxyl", coverage=0.1, count=3)

    def test_count_is_honoured_exactly_when_it_fits(self):
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, "hydroxyl", count=5, seed=1)
        assert grafted.info["functionalization"]["n_grafted"] == 5

    def test_the_recorded_bond_graph_is_extended(self):
        """Otherwise the grafted atoms read as isolated.

        `coordination_numbers` prefers the recorded graph, so a builder
        that records one must keep it complete or validation judges the
        new atoms against a graph they are not in.
        """
        cage = build_fullerene(freq=1, family="C60")
        assert "bonds" in cage.info
        grafted = functionalize(cage, "hydroxyl", coverage=0.2, seed=1)
        bonds = np.asarray(grafted.info["bonds"])
        assert bonds.max() == len(grafted) - 1
        from nanocarbon_lab.topology import coordination_numbers
        assert coordination_numbers(grafted).min() >= 1


class TestStericHonesty:
    """Achieved coverage is a measurement, and it must be a real one."""

    def test_fluorographene_reaches_full_stoichiometry(self):
        """CF is a real material, so full coverage must be reachable.

        It only is in the chair conformation -- alternating by
        sublattice. Alternating by shuffle order puts two fluorines on
        adjacent carbons on the same face, 1.42 Å apart, and coverage
        stalls at 42%.
        """
        sheet = build_graphene().repeat((5, 5, 1))
        grafted = functionalize(sheet, "fluorine", coverage=1.0,
                                face="both", seed=1)
        record = grafted.info["functionalization"]
        assert record["coverage"] == pytest.approx(1.0)
        assert record["refused_steric"] == 0
        assert grafted.get_chemical_formula() == "C50F50"

    def test_a_bulky_group_reaches_less_coverage_than_a_small_one(self):
        sheet = build_graphene().repeat((5, 5, 1))
        reached = {}
        for name in ("fluorine", "hydroxyl", "methyl"):
            grafted = functionalize(sheet, name, coverage=1.0,
                                    face="both", seed=1)
            reached[name] = grafted.info["functionalization"]["coverage"]
        assert reached["fluorine"] > reached["hydroxyl"] > reached["methyl"]

    def test_a_shortfall_is_reported_rather_than_hidden(self):
        sheet = build_graphene().repeat((5, 5, 1))
        grafted = functionalize(sheet, "methyl", coverage=1.0, seed=1)
        record = grafted.info["functionalization"]
        assert record["n_grafted"] < record["n_requested"]
        assert record["refused_steric"] > 0
        assert "did not fit" in describe_functionalization(grafted)

    def test_no_grafted_atom_overlaps_anything(self):
        cases = [
            functionalize(build_cnt(6, 6, length=12.0), "hydroxyl",
                          coverage=0.3, seed=1),
            functionalize(build_graphene().repeat((5, 5, 1)), "fluorine",
                          coverage=1.0, face="both", seed=1),
            functionalize(build_tmd_monolayer("MoS2").repeat((3, 3, 1)),
                          "thiol", coverage=0.3, seed=1),
        ]
        for atoms in cases:
            distances = atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, np.inf)
            assert distances.min() > 0.9

    def test_min_separation_is_honoured(self):
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, "hydroxyl", coverage=0.5,
                                min_separation=4.0, seed=1)
        sites = grafted.info["functionalization"]["sites"]
        for first in sites:
            for second in sites:
                if first != second:
                    assert cnt.get_distance(first, second, mic=True) >= 4.0

    def test_the_periodic_seam_is_not_a_free_pass(self):
        """A group near a cell face has its own image on the other side.

        Ignore it and the cell looks fine until a neighbouring image is
        drawn. The exemption for bonded neighbours has to reach across
        the seam too, or every group along it is refused instead.
        """
        sheet = build_graphene().repeat((5, 5, 1))
        grafted = functionalize(sheet, "fluorine", coverage=1.0,
                                face="both", seed=3)
        distances = grafted.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() > 1.2


class TestFaces:
    def test_a_sheet_can_be_grafted_on_one_face_or_both(self):
        sheet = build_graphene().repeat((5, 5, 1))
        middle = sheet.get_positions()[:, 2].mean()

        one = functionalize(sheet, "hydroxyl", coverage=0.2,
                            face="outer", seed=1)
        heights = one.get_positions()[len(sheet):, 2] - middle
        assert (heights > 0).all()

        both = functionalize(sheet, "hydroxyl", coverage=0.4,
                             face="both", seed=1)
        heights = both.get_positions()[len(sheet):, 2] - middle
        assert (heights > 0).any() and (heights < 0).any()

    def test_a_sandwich_has_only_one_free_face_per_chalcogen(self):
        """An MX2 bond graph is bipartite between metal and chalcogen.

        So every chalcogen takes the same colour, and alternating by
        sublattice aimed every single group into the sandwich: coverage
        came out at exactly zero, blamed on sterics, which was true and
        useless.
        """
        slab = build_tmd_monolayer("MoS2").repeat((3, 3, 1))
        normals = surface_normals(slab)
        blocked = inner_face_blocked(slab, normals)
        assert blocked.all()

        parity = sublattice_parity(slab)
        sulphurs = [i for i, s in enumerate(slab.get_chemical_symbols())
                    if s == "S"]
        assert len({parity[i] for i in sulphurs}) == 1

        outer = functionalize(slab, "thiol", coverage=0.25,
                              face="outer", seed=4)
        both = functionalize(slab, "thiol", coverage=0.25,
                             face="both", seed=4)
        assert outer.info["functionalization"]["n_grafted"] > 0
        assert (both.info["functionalization"]["n_grafted"]
                == outer.info["functionalization"]["n_grafted"])

    def test_a_single_sheet_has_two_free_faces(self):
        sheet = build_graphene().repeat((4, 4, 1))
        assert not inner_face_blocked(sheet, surface_normals(sheet)).any()

    def test_the_blocked_depth_separates_a_sandwich_from_a_cage(self):
        """A fullerene's cavity is real; an MX2's interior is not."""
        cage = build_fullerene(freq=1, family="C60")
        assert not inner_face_blocked(cage, surface_normals(cage)).any()
        assert INNER_BLOCKED_DEPTH > 0.3  # above a cage's pyramidalisation


class TestBridging:
    def test_an_epoxide_sits_symmetrically_over_a_bond(self):
        sheet = build_graphene().repeat((5, 5, 1))
        grafted = functionalize(sheet, "epoxide", coverage=0.3, seed=2)
        record = grafted.info["functionalization"]
        assert record["bridging"]
        expected = bond_length("C", "O")
        for offset, (first, second) in enumerate(record["bridges"]):
            oxygen = len(sheet) + offset
            left = grafted.get_distance(first, oxygen, mic=True)
            right = grafted.get_distance(second, oxygen, mic=True)
            assert left == pytest.approx(expected, abs=1e-6)
            assert right == pytest.approx(expected, abs=1e-6)

    def test_no_two_bridges_share_an_atom(self):
        sheet = build_graphene().repeat((5, 5, 1))
        grafted = functionalize(sheet, "epoxide", coverage=1.0, seed=2)
        record = grafted.info["functionalization"]
        used = [index for pair in record["bridges"] for index in pair]
        assert len(used) == len(set(used))

    def test_full_coverage_is_bounded_by_the_matching_not_by_sterics(self):
        """Each epoxide consumes two carbons, so at most half can carry one."""
        sheet = build_graphene().repeat((5, 5, 1))
        grafted = functionalize(sheet, "epoxide", coverage=1.0, seed=2)
        record = grafted.info["functionalization"]
        assert record["n_grafted"] <= len(sheet) // 2
        assert record["refused_occupied"] > 0
        assert "no two can do" in describe_functionalization(grafted)

    def test_a_bridge_too_wide_to_span_is_refused(self):
        with pytest.raises(ValueError, match="cannot bridge"):
            build_bridging_positions(
                get_group("epoxide"), ORIGIN, np.array([5.0, 0.0, 0.0]), UP)

    def test_build_positions_refuses_a_bridging_group(self):
        with pytest.raises(ValueError, match="bridges two"):
            build_positions(get_group("epoxide"), ORIGIN, UP)

    def test_a_terminal_group_is_refused_by_the_bridging_builder(self):
        with pytest.raises(ValueError, match="single atom"):
            build_bridging_positions(
                get_group("hydroxyl"), ORIGIN, np.array([1.42, 0.0, 0.0]), UP)


class TestValidationAndMetadata:
    """A functionalised structure must be exportable and stay honest."""

    @pytest.mark.parametrize("name", [
        "hydroxyl", "carboxyl", "carbonyl", "amine", "thiol",
        "methyl", "nitro", "aldehyde", "fluorine", "epoxide",
    ])
    def test_every_group_passes_validation_on_a_tube(self, name):
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, name, coverage=0.08, seed=1)
        report = run_basic_checks(grafted)
        assert report.ok, report.summary()
        assert not report.warnings, report.summary()

    def test_a_monovalent_atom_is_not_dangling(self):
        """Coordination 1 is a full valence for H and the halogens.

        The ceiling in `check_coordination` was always per element; this
        test was not, so a correct CF monolayer produced 50 warnings.
        """
        sheet = build_graphene().repeat((5, 5, 1))
        grafted = functionalize(sheet, "fluorine", coverage=1.0,
                                face="both", seed=1)
        report = run_basic_checks(grafted)
        assert not any("dangling" in w for w in report.warnings)

    def test_a_double_bonded_terminal_atom_is_not_dangling(self):
        """A carbonyl oxygen has one neighbour and a full valence."""
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, "carbonyl", coverage=0.05, seed=1)
        assert grafted.info["terminal_atoms"]
        report = run_basic_checks(grafted)
        assert not any("dangling" in w for w in report.warnings)

    def test_an_o_h_bond_is_not_reported_as_too_short(self):
        """0.970 Å is the literature O-H length exactly.

        Judged against carbon's sp2 window it looked suspicious on every
        hydroxylated structure; judged against its own covalent radii it
        is a textbook bond.
        """
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, "hydroxyl", coverage=0.1, seed=1)
        report = run_basic_checks(grafted)
        assert not any("shorter than any bond" in w for w in report.warnings)

    def test_the_vacuum_a_structure_was_built_with_is_preserved(self):
        """Groups stick out, and a finite cell is a bounding box.

        Left alone, a (6,6) tube built with 12 Å of vacuum came back
        with 8.55 Å once hydroxylated and validation refused it.
        """
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, "hydroxyl", coverage=0.2, seed=1)
        for axis in range(3):
            if cnt.pbc[axis]:
                assert np.allclose(grafted.cell[axis], cnt.cell[axis])
                continue
            before = cnt.get_positions()[:, axis]
            after = grafted.get_positions()[:, axis]
            gap_before = cnt.cell[axis][axis] - (before.max() - before.min())
            gap_after = grafted.cell[axis][axis] - (after.max() - after.min())
            assert gap_after == pytest.approx(gap_before, abs=1e-6)

    def test_a_periodic_axis_is_never_repadded(self):
        sheet = build_graphene().repeat((4, 4, 1))
        grafted = functionalize(sheet, "hydroxyl", coverage=0.2, seed=1)
        for axis in range(2):
            assert np.allclose(grafted.cell[axis], sheet.cell[axis])

    def test_preserve_vacuum_is_idempotent(self):
        cnt = build_cnt(6, 6, length=12.0)
        grafted = functionalize(cnt, "hydroxyl", coverage=0.2, seed=1)
        cell_once = np.array(grafted.cell)
        preserve_vacuum(grafted, grafted)
        assert np.allclose(np.array(grafted.cell), cell_once)

    def test_removing_atoms_renumbers_the_recorded_sites(self):
        """The rule every deletion path in this repo has to follow.

        The sites are atom indices nested inside `info`, so
        `INDEX_LIST_KEYS` cannot reach them and they need their own
        handling -- otherwise a vacancy leaves "site 412" pointing at a
        different atom, exactly the silent corruption `utils/metadata`
        exists to prevent.
        """
        cage = build_fullerene(freq=1, family="C60")
        grafted = functionalize(cage, "hydroxyl", coverage=0.2, seed=1)
        record = grafted.info["functionalization"]
        original = [int(i) for i in record["sites"]]

        removed = [0, 1, 2]
        keep = keep_indices(len(grafted), removed)
        cut = grafted[keep]
        cut.info = remap_after_removal(grafted.info, keep)

        sites = cut.info["functionalization"]["sites"]
        assert all(0 <= i < len(cut) for i in sites)
        assert cut.info["functionalization"]["n_grafted"] == len(sites)
        # An untouched site must still name the same atom.
        survivor = next(i for i in original if i not in removed)
        assert keep.index(survivor) in sites

    def test_removing_atoms_renumbers_the_terminal_list(self):
        """A carbonyl oxygen's index moves with everything after it."""
        cage = build_fullerene(freq=1, family="C60")
        grafted = functionalize(cage, "carboxyl", coverage=0.1, seed=1)
        assert grafted.info["terminal_atoms"]
        keep = keep_indices(len(grafted), [0, 1, 2])
        info = remap_after_removal(grafted.info, keep)
        assert info["terminal_atoms"]
        assert all(0 <= i < len(keep) for i in info["terminal_atoms"])

    def test_removing_a_bridge_end_drops_the_bridge(self):
        sheet = build_graphene().repeat((5, 5, 1))
        grafted = functionalize(sheet, "epoxide", coverage=0.3, seed=2)
        first, second = grafted.info["functionalization"]["bridges"][0]
        keep = keep_indices(len(grafted), [first])
        info = remap_after_removal(grafted.info, keep)
        remaining = info["functionalization"]["bridges"]
        assert len(remaining) == len(
            grafted.info["functionalization"]["bridges"]) - 1
        assert all(0 <= a < len(keep) and 0 <= b < len(keep)
                   for a, b in remaining)
        assert info["functionalization"]["n_grafted"] == len(remaining)


class TestRegistry:
    def test_every_group_has_a_note_and_a_formula(self):
        for name, group in GROUPS.items():
            assert group.name == name
            assert group.note
            assert group.formula
            assert group.n_atoms >= 1
            assert group.site_hybridisation in ("sp2", "sp3")

    def test_every_group_element_has_a_radius_and_a_ceiling(self):
        """The rule `dopants/chemistry.py` already carries.

        An element with no radius falls back to 1.80 Å and its bonds are
        not bonds; one with no coordination ceiling falls back to 6, so a
        monovalent iodine is never flagged when it is over-bonded.
        """
        reachable = {element for group in GROUPS.values()
                     for symbol in group.elements()
                     for element in (symbol, *viable_swaps(symbol))}
        for element in reachable:
            assert element in COVALENT_RADII, element
            assert element in MAX_COORDINATION, element

    def test_the_z_matrix_orders_parents_before_children(self):
        for name, group in GROUPS.items():
            for index, atom in enumerate(group.atoms):
                assert atom.parent < index, name
            assert group.atoms[0].parent == -1, name

    def test_an_unknown_group_lists_what_is_available(self):
        with pytest.raises(ValueError, match="Available"):
            get_group("hydroxide")

    def test_describe_mentions_the_anchor_bond(self):
        assert "1.42" in describe(get_group("hydroxyl"), "C")
        assert "bridging" in describe(get_group("epoxide"), "C")
