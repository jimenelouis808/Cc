"""Analysing structures the framework did not build.

The strongest test available is that perception must reproduce what the
builders recorded, on structures where the recorded answer is known to be
right. So most of these strip `atoms.info` and check the analysis against
the metadata that was thrown away.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms, io

from nanocarbon_lab.analyse import (
    analyse,
    describe_shape,
    format_report,
    is_surface_net,
    perceive_rings,
    read_structure,
    ring_report,
    trace_faces,
)
from nanocarbon_lab.builders import (
    build_bundle,
    build_capped_cnt,
    build_cnt,
    build_fullerene,
    build_graphene,
    build_junction,
    build_multiwall_cnt,
    build_nanoribbon,
    build_schwarzite,
)
from nanocarbon_lab.functionalize import functionalize
from nanocarbon_lab.tmd import (
    build_tmd_bulk,
    build_tmd_layers,
    build_tmd_monolayer,
    build_tmd_nanotube,
    build_tmd_ribbon,
)
from nanocarbon_lab.utils.geometry import guess_bonds


def foreign(atoms: Atoms) -> Atoms:
    """The same structure as it would arrive from someone else's code.

    Every trace of what built it removed, so nothing can be read where it
    was meant to be perceived.
    """
    stripped = atoms.copy()
    stripped.info = {}
    return stripped


def inferred_pairs(atoms: Atoms) -> np.ndarray:
    return np.asarray([(i, j) for i, j, _ in guess_bonds(atoms)], dtype=int)


class TestRingPerception:
    """Perception must reproduce the rings the builder recorded."""

    KNOWN = [
        ("C60", lambda: build_fullerene(freq=1, family="C60")),
        ("C240", lambda: build_fullerene(freq=2, family="C60")),
        ("capped tube", lambda: build_capped_cnt(n_body_rings=6, freq=3)),
        ("Y junction", lambda: build_junction(kind="Y", tube_radius=6.0,
                                              arm_length=20.0)),
        ("Schwarz P", lambda: build_schwarzite(kind="primitive", cell=36.0)),
    ]

    @pytest.mark.parametrize("name,make", KNOWN, ids=[n for n, _ in KNOWN])
    def test_it_reproduces_the_recorded_census(self, name, make):
        built = make()
        recorded = dict(sorted(built.info["ring_counts"].items()))
        bare = foreign(built)
        report = ring_report(bare, inferred_pairs(bare))
        assert report["method"] == "faces"
        assert report["counts"] == recorded, name

    @pytest.mark.parametrize("name,make", KNOWN, ids=[n for n, _ in KNOWN])
    def test_the_face_count_satisfies_euler(self, name, make):
        """``F = E - V + 2 - 2g``, exactly.

        This is the property face tracing has and cycle perception does
        not: the faces are the faces, so their number is fixed by the
        topology rather than by how long the search ran.
        """
        built = make()
        genus = built.info.get("genus", 0)
        bare = foreign(built)
        pairs = inferred_pairs(bare)
        faces, boundary = trace_faces(bare, pairs)
        assert boundary == 0, "a closed surface has no boundary walk"
        assert len(faces) == len(pairs) - len(bare) + 2 - 2 * genus

    def test_shortest_path_rings_undercount_a_tiled_surface(self):
        """The reason face tracing exists, stated as a test.

        A heptagon every one of whose bonds also borders a hexagon is
        never the *smallest* ring through any bond, so a shortest-path
        search never emits it. Both methods answer their own question
        correctly; only one answers "what tiles this surface".
        """
        built = build_junction(kind="Y", tube_radius=6.0, arm_length=20.0)
        truth = built.info["ring_counts"]
        bare = foreign(built)
        pairs = inferred_pairs(bare)

        faces, _ = trace_faces(bare, pairs)
        by_faces: dict[int, int] = {}
        for face in faces:
            by_faces[len(face)] = by_faces.get(len(face), 0) + 1

        by_paths: dict[int, int] = {}
        for ring in perceive_rings(bare, pairs):
            by_paths[len(ring)] = by_paths.get(len(ring), 0) + 1

        assert by_faces[7] == truth[7]
        assert by_paths[7] < truth[7]

    def test_graphene_and_a_tube_are_all_hexagons(self):
        for atoms in (build_graphene().repeat((4, 4, 1)),
                      build_cnt(6, 6, length=12.0)):
            bare = foreign(atoms)
            report = ring_report(bare, inferred_pairs(bare))
            assert set(report["counts"]) == {6}
            assert report["euler_deficit"] == 0

    def test_a_cell_too_small_for_a_graph_is_refused(self):
        """A 1x1 graphene cell has two atoms and nine bonds between them.

        A simple graph holds one edge per pair, so any census on it is
        meaningless. Saying so beats quoting a number.
        """
        bare = foreign(build_graphene())
        report = ring_report(bare, inferred_pairs(bare))
        assert not report["reliable"]
        assert report["method"] == "none"
        assert report["counts"] == {}
        assert "more than one periodic image" in report["caveat"]

    def test_a_dichalcogenide_falls_back_and_says_so(self):
        """No trivalent surface, so there are no faces to trace."""
        slab = foreign(build_tmd_monolayer("MoS2").repeat((3, 3, 1)))
        report = ring_report(slab, inferred_pairs(slab))
        assert report["method"] == "shortest-path"
        assert "shortest-path" in report["caveat"]

    def test_is_surface_net_separates_the_two_families(self):
        assert is_surface_net(*(lambda a: (a, inferred_pairs(a)))(
            foreign(build_cnt(6, 6, length=12.0))))
        slab = foreign(build_tmd_monolayer("MoS2").repeat((3, 3, 1)))
        assert not is_surface_net(slab, inferred_pairs(slab))


class TestShape:
    """Dimensionality from geometry, since a foreign file rarely says."""

    CASES = [
        ("cage", lambda: build_fullerene(freq=1, family="C60"), 0, "cage"),
        ("tube", lambda: build_cnt(6, 6, length=12.0), 1, "tube"),
        ("capped tube", lambda: build_capped_cnt(n_body_rings=6, freq=3),
         1, "tube"),
        ("sheet", lambda: build_graphene().repeat((4, 4, 1)), 2, "sheet"),
        ("flake", lambda: build_nanoribbon(width=4, length=6), 2, "flake"),
        ("junction", lambda: build_junction(kind="Y", tube_radius=6.0,
                                            arm_length=20.0),
         0, "branched shell"),
        ("bulk", lambda: build_schwarzite(kind="primitive", cell=36.0),
         3, "bulk"),
        ("MWCNT", lambda: build_multiwall_cnt(n_shells=2, inner_freq=3,
                                              n_body_rings=6), 1, "tube"),
        ("bundle", lambda: build_bundle(n_rings_across=1, n_body_rings=6,
                                        freq=3), 1, "tube"),
        ("MX2 sheet", lambda: build_tmd_monolayer("MoS2"), 2, "sheet"),
        ("MX2 slab", lambda: build_tmd_layers("MoS2", n_layers=2), 2, "slab"),
        ("MX2 bulk", lambda: build_tmd_bulk("MoS2"), 3, "bulk"),
        ("MX2 tube", lambda: build_tmd_nanotube("MoS2", n=30, m=0), 1, "tube"),
        ("MX2 ribbon", lambda: build_tmd_ribbon("MoS2", width=3, length=4),
         1, "ribbon"),
    ]

    @pytest.mark.parametrize("name,make,dimension,shape", CASES,
                             ids=[case[0] for case in CASES])
    def test_shape_is_recovered_from_geometry(self, name, make, dimension,
                                              shape):
        result = describe_shape(foreign(make()))
        assert result["dimensionality"] == dimension, result["reason"]
        assert result["shape"] == shape, result["reason"]

    def test_a_slab_written_as_fully_periodic_is_still_2d(self):
        """Every plane-wave code writes a slab as `pbc=(True, True, True)`.

        Trusting that would call it bulk and then quote a density for it,
        which would be a density of the padding.
        """
        slab = foreign(build_tmd_monolayer("MoS2").repeat((3, 3, 1)))
        slab.pbc = (True, True, True)
        result = describe_shape(slab)
        assert result["dimensionality"] == 2
        assert result["vacuum_axes"] == [2]

    def test_a_short_fat_tube_is_not_a_chain(self):
        """Its longest principal axis is a *diameter*, not its axis.

        A one-cell MoS2 nanotube is 33 Å across and 4.6 Å long, and
        measuring hollowness about the longest principal axis called a
        textbook nanotube solid.
        """
        tube = foreign(build_tmd_nanotube("MoS2", n=30, m=0))
        assert describe_shape(tube)["shape"] == "tube"

    def test_stacked_layers_are_a_slab_not_two_sheets(self):
        """Both are true; the slab accounts for every atom."""
        result = describe_shape(foreign(build_tmd_layers("MoS2", n_layers=2)))
        assert result["shape"] == "slab"
        assert result["n_components"] == 2

    def test_nested_tubes_are_tubes_not_a_solid(self):
        """Here the union *is* wrong: the annulus between the walls is
        empty space, not material."""
        result = describe_shape(foreign(
            build_multiwall_cnt(n_shells=2, inner_freq=3, n_body_rings=6)))
        assert result["shape"] == "tube"
        assert result["n_components"] == 2
        assert result["component_shapes"] == ["tube", "tube"]


class TestProvenance:
    """What was recorded, measured and inferred must never be conflated."""

    def test_a_foreign_file_records_nothing(self):
        result = analyse(foreign(build_capped_cnt(n_body_rings=6, freq=3)))
        assert not result["recorded"]["has_ring_list"]
        assert not result["recorded"]["has_bond_graph"]
        assert result["inferred"]["rings"]["counts"]
        assert "nothing" in format_report(result)

    def test_our_own_file_is_read_rather_than_perceived(self):
        built = build_capped_cnt(n_body_rings=6, freq=3)
        result = analyse(built)
        assert result["recorded"]["ring_counts"]
        assert result["inferred"]["rings_are_recorded"]
        assert "the file records them" in format_report(result)

    def test_the_bond_assumption_is_reported(self):
        """Every inferred number rests on it, so it cannot be implicit."""
        result = analyse(foreign(build_cnt(6, 6, length=12.0)), tolerance=0.25)
        assert result["inferred"]["bond_tolerance"] == 0.25
        assert "0.25" in format_report(result)

    def test_bond_lengths_are_split_by_element_pair(self):
        """A mixed structure has no single bond length worth quoting."""
        grafted = functionalize(build_cnt(6, 6, length=12.0), "hydroxyl",
                                coverage=0.2, seed=1)
        result = analyse(foreign(grafted))
        by_pair = result["measured"]["bonds"]["by_pair"]
        assert set(by_pair) == {"C-C", "C-O", "H-O"}
        assert by_pair["C-C"]["mean"] == pytest.approx(1.42, abs=0.02)
        assert by_pair["H-O"]["mean"] == pytest.approx(0.97, abs=0.01)

    def test_density_only_for_a_real_bulk(self):
        """A slab's cell is mostly vacuum, so its density is the padding's."""
        bulk = analyse(foreign(build_schwarzite(kind="primitive", cell=36.0)))
        assert bulk["measured"]["density_g_cm3"] is not None
        sheet = analyse(foreign(build_graphene().repeat((4, 4, 1))))
        assert sheet["measured"]["density_g_cm3"] is None


class TestVerdict:
    def test_a_dichalcogenide_is_judged_by_the_mx2_model(self):
        """Carbon's "five or more is unphysical" rejects every MoS2 cell."""
        result = analyse(foreign(build_tmd_monolayer("MoS2").repeat((3, 3, 1))))
        assert result["verdict"]["model"] == "dichalcogenide"
        assert result["verdict"]["verdict"] == "clean"

    def test_carbon_is_judged_by_the_sp2_model(self):
        result = analyse(foreign(build_capped_cnt(n_body_rings=6, freq=3)))
        assert result["verdict"]["model"] == "sp2 carbon"
        assert result["verdict"]["verdict"] == "clean"
        assert "caveat" in result["verdict"]

    def test_a_broken_structure_is_reported_as_broken(self):
        atoms = build_cnt(6, 6, length=12.0)
        positions = atoms.get_positions()
        positions[0] += np.array([0.0, 0.0, 0.6])   # stretch its bonds
        atoms.set_positions(positions)
        result = analyse(foreign(atoms))
        assert result["verdict"]["verdict"] in ("strained", "broken")

    def test_something_that_is_neither_says_so(self):
        water = Atoms("H2O", positions=[[0.0, 0.76, -0.48],
                                        [0.0, -0.76, -0.48],
                                        [0.0, 0.0, 0.12]])
        result = analyse(water)
        assert result["verdict"]["model"] == "none"
        assert result["verdict"]["verdict"] == "unknown"


class TestReadingFiles:
    def test_it_reads_what_the_package_writes(self, tmp_path):
        from nanocarbon_lab.exports.xyz import write_render_bundle

        built = build_capped_cnt(n_body_rings=6, freq=3)
        xyz, _ = write_render_bundle(built, tmp_path / "tube")
        atoms = read_structure(xyz)
        assert len(atoms) == len(built)
        result = analyse(atoms)
        assert result["inferred"]["rings"]["counts"] == dict(
            sorted(built.info["ring_counts"].items()))

    def test_a_cif_round_trip_keeps_the_verdict(self, tmp_path):
        built = build_tmd_layers("MoS2", n_layers=2)
        target = tmp_path / "mos2.cif"
        io.write(str(target), foreign(built))
        result = analyse(read_structure(target))
        assert result["verdict"]["model"] == "dichalcogenide"
        assert result["inferred"]["shape"]["shape"] == "slab"

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(ValueError, match="No such file"):
            read_structure(tmp_path / "absent.xyz")

    def test_an_unreadable_file_names_the_reason(self, tmp_path):
        target = tmp_path / "broken.xyz"
        target.write_text("this is not a structure\n")
        with pytest.raises(ValueError, match="Could not read"):
            read_structure(target)


class TestGraftedStructuresStayHonest:
    """Grafting and analysis meet here, and both had bugs the other found."""

    def test_graphene_oxide_is_chemically_sound(self):
        """Two grafts, and the second must not land on the first.

        A hydroxyl offered the oxygen of an epoxide already on the sheet
        took it, and the two oxygens came out 1.32 Å apart -- a peroxide
        bridge nobody asked for, on a structure whose every other number
        looked right.
        """
        sheet = build_graphene().repeat((5, 5, 1))
        oxide = functionalize(sheet, "epoxide", coverage=0.25, seed=1)
        oxide = functionalize(oxide, "hydroxyl", coverage=0.2,
                              face="both", seed=2)

        result = analyse(foreign(oxide))
        assert result["validation"]["ok"], result["validation"]["errors"]
        assert not result["validation"]["warnings"]
        assert "O-O" not in result["measured"]["bonds"]["by_pair"]
        carbon = result["measured"]["coordination"]["per_element"]["C"]
        assert max(carbon) <= 4

    def test_a_grafted_site_is_not_offered_twice(self):
        from nanocarbon_lab.functionalize.attach import candidate_sites

        sheet = build_graphene().repeat((5, 5, 1))
        oxide = functionalize(sheet, "epoxide", coverage=0.3, seed=1)
        symbols = oxide.get_chemical_symbols()
        grafted = set(oxide.info["grafted_atoms"])
        for site in candidate_sites(oxide):
            assert site not in grafted
            assert symbols[site] == "C"

    def test_an_atom_at_full_valence_is_not_offered(self):
        """A carbon already carrying an epoxide has no bond left."""
        from nanocarbon_lab.functionalize.attach import candidate_sites
        from nanocarbon_lab.topology import coordination_numbers

        sheet = build_graphene().repeat((5, 5, 1))
        oxide = functionalize(sheet, "epoxide", coverage=0.3, seed=1)
        numbers = coordination_numbers(oxide)
        for site in candidate_sites(oxide):
            assert numbers[site] < 4


class TestCoordinationAcrossImages:
    def test_a_minimal_cell_does_not_undercount(self):
        """A networkx Graph holds one edge per *pair*.

        In a 1x1 MoS2 cell an atom reaches the same neighbour through
        several images, so the collapsed degree gave its metals 2 instead
        of 6 and validation warned about dangling chalcogens in a perfect
        crystal.
        """
        from nanocarbon_lab.topology import coordination_numbers

        cell = build_tmd_monolayer("MoS2")
        numbers = coordination_numbers(foreign(cell))
        symbols = cell.get_chemical_symbols()
        for index, symbol in enumerate(symbols):
            assert numbers[index] == (6 if symbol == "Mo" else 3), symbol

    def test_a_repeated_cell_agrees_with_the_minimal_one(self):
        from nanocarbon_lab.topology import coordination_numbers

        small = coordination_numbers(foreign(build_tmd_monolayer("MoS2")))
        large = coordination_numbers(
            foreign(build_tmd_monolayer("MoS2").repeat((3, 3, 1))))
        assert sorted(set(small)) == sorted(set(large))
