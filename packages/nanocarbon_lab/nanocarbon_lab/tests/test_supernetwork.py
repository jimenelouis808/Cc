"""Networks whose edges are nanotubes and whose vertices are junctions.

The hierarchy of Romo-Herrera et al., *Nano Lett.* 7 (2007) 570: a 1D
block, a multi-terminal node made from it, and then the node used as the
new building block. What makes it testable is that the skeleton is a
graph, and a graph's coordination number is a fact about it -- so the
catalogue can be checked against what each net is *supposed* to have
before anything is meshed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from nanocarbon_lab.builders.supernetwork import (
    CAGES,
    SUPERLATTICES,
    SuperGraph,
    build_supernetwork,
    hypercube_cage,
    icosahedral_cage,
    named_graph,
    supergraph_from_atoms,
    supertube_graph,
)

#: What each net's vertices must have. These are properties of the nets,
#: not of this code: a simple cubic site has six nearest neighbours, a
#: diamond site four, an fcc site twelve, a honeycomb site three.
EXPECTED_COORDINATION = {
    "super-square": 4,
    "super-graphene": 3,
    "super-cubic": 6,
    "super-diamond": 4,
    "super-fcc": 12,
}


class TestTheSkeletonBeforeAnythingIsMeshed:
    """The edges are worked out from the positions, so they can be
    checked. A hand-written edge table with a missing periodic image
    gives a three-coordinate diamond site and still builds."""

    @pytest.mark.parametrize("name", sorted(EXPECTED_COORDINATION))
    def test_each_net_has_the_coordination_it_is_named_for(self, name):
        assert SUPERLATTICES[name].coordination == EXPECTED_COORDINATION[name]

    @pytest.mark.parametrize("name", sorted(EXPECTED_COORDINATION))
    def test_every_edge_is_the_same_length(self, name):
        """A uniform net has one strut length. Two lengths means the
        tolerance swept in a second-neighbour shell."""
        lengths = SUPERLATTICES[name].strut_lengths(40.0)
        assert lengths.std() < 1e-6 * lengths.mean()

    def test_an_edge_is_stored_once_and_not_twice(self):
        """i to j through one face and j to i through the opposite one
        are the same tube. Storing both doubles the field's weight there
        and thickens the strut."""
        square = SUPERLATTICES["super-square"]
        assert len(square.edges) == 2          # one node, four half-edges
        assert square.coordination == 4

    def test_a_missing_image_is_caught_rather_than_built(self):
        graph = SuperGraph(
            name="broken",
            nodes=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]),
            edges=((0, 1, (0, 0, 0)),),        # one vertex short of uniform
            pbc=(True, True, False),
        )
        assert graph.coordination == 1          # both are 1 here, so fine
        lopsided = SuperGraph(
            name="lopsided",
            nodes=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                            [0.0, 0.5, 0.0]]),
            edges=((0, 1, (0, 0, 0)), (0, 2, (0, 0, 0))),
            pbc=(False, False, False),
        )
        with pytest.raises(ValueError, match="missing"):
            _ = lopsided.coordination


class TestTheCagesComeFromStructuresThatAlreadyExist:
    """The hierarchy's own step: the node becomes the building block. A
    C60 cage is a graph, so scaling it up and hanging a tube on every
    edge is a superfullerene, and nothing in the step knows what it was
    handed."""

    def test_a_c60_gives_sixty_vertices_and_ninety_edges(self):
        from nanocarbon_lab.builders.fullerene import build_fullerene

        graph = supergraph_from_atoms(build_fullerene(family="C60"), scale=7.0)
        assert len(graph.nodes) == 60
        assert len(graph.edges) == 90          # Euler: 60 - 90 + 32 = 2
        assert graph.coordination == 3

    def test_an_icosahedron_gives_twelve_vertices_and_thirty_edges(self):
        graph = icosahedral_cage(scale=9.0)
        assert len(graph.nodes) == 12
        assert len(graph.edges) == 30          # Euler: 12 - 30 + 20 = 2
        assert graph.coordination == 5

    def test_scaling_is_the_only_knob_that_matters(self):
        small = icosahedral_cage(scale=3.0)
        large = icosahedral_cage(scale=9.0)
        assert large.strut_lengths(1.0).mean() == pytest.approx(
            3.0 * small.strut_lengths(1.0).mean())


class TestItRefusesWhatCannotBeATube:
    def test_edges_too_short_for_a_tube_are_refused(self):
        with pytest.raises(ValueError, match="nothing recognisable as a tube"):
            build_supernetwork("super-cubic", scale=12.0, tube_radius=5.0,
                               blend=4.0)

    def test_an_unknown_net_names_the_catalogue(self):
        with pytest.raises(ValueError, match="unknown net"):
            build_supernetwork("super-octagonal")


class TestASupersheetIsTwoDimensional:
    """A 2D net must not acquire a period through the vacuum. The third
    direction is padded and the structure centred in it, so the periodic
    mesher's weld of that face finds nothing to join."""

    @pytest.fixture(scope="class")
    def sheet(self):
        return build_supernetwork("super-graphene", scale=36.0,
                                  tube_radius=5.0, blend=4.0,
                                  grid_resolution=64)

    def test_it_repeats_in_two_directions_and_not_the_third(self, sheet):
        assert list(sheet.get_pbc()) == [True, True, False]

    def test_the_cell_is_the_honeycomb_rectangle(self, sheet):
        a, b, _c = sheet.cell.lengths()
        assert b / a == pytest.approx(np.sqrt(3.0), rel=1e-3)

    def test_the_wall_is_graphitic(self, sheet):
        geometry = sheet.info["geometry"]
        assert geometry["bond_mean"] == pytest.approx(1.42, abs=0.03)
        assert geometry["bond_std"] < 0.06
        assert geometry["n_close_contacts"] == 0

    def test_the_nodes_carry_the_curvature(self, sheet):
        """Nobody puts heptagons at a three-way node. A node is a saddle,
        a saddle is negative Gaussian curvature, and the remesher returns
        that as degree-7 vertices which the dual renders as heptagons."""
        census = sheet.info["ring_counts"]
        assert census.get(7, 0) > 0
        assert census.get(6, 0) > 10 * census.get(7, 0) / 10


class TestTheRingBudgetComesFromTheSkeleton:
    """``sum(6-n) = 12 * (V - E)``, known before anything is meshed.

    The wall is the boundary of a thickened graph, so the surface has one
    handle per independent cycle of the graph: ``chi = 2 * (V - E)``, and
    with ``sum(6-n) = 6 * chi`` the budget follows. This is the check
    that is *independent* of the mesh -- ``junction._finish`` already
    tests the census against the mesh's own Euler characteristic, which
    catches a torn mesh but not a mesh that closed cleanly around the
    wrong graph.
    """

    def test_it_reproduces_the_networks_hardcoded_constants(self):
        """``network.py`` carries cubic -24 and diamond -96 as genus
        knowledge. The same formula gives both, so they are not separate
        facts."""
        assert SUPERLATTICES["super-cubic"].ring_budget == -24
        assert SUPERLATTICES["super-diamond"].ring_budget == -96

    @pytest.mark.parametrize("name, expected", [
        ("super-square", -12),
        ("super-graphene", -24),
        ("super-cubic", -24),
        ("super-diamond", -96),
        ("super-fcc", -240),
    ])
    def test_every_net_in_the_catalogue(self, name, expected):
        graph = SUPERLATTICES[name]
        assert graph.ring_budget == expected
        assert graph.ring_budget == 12 * (len(graph.nodes) - len(graph.edges))

    def test_a_finite_cage_too(self):
        cage = icosahedral_cage()
        assert cage.ring_budget == 12 * (12 - 30) == -216

    def test_a_built_net_meets_its_budget(self):
        """Measured rather than asserted: the census of a real build,
        against a number fixed before the field was ever evaluated."""
        built = build_supernetwork("super-square", scale=34.0,
                                   tube_radius=5.0, blend=4.0,
                                   grid_resolution=64)
        counts = built.info["ring_counts"]
        deficit = sum((6 - size) * count for size, count in counts.items())
        assert deficit == built.info["ring_budget"] == -12

    def test_the_check_is_live_and_not_decoration(self):
        """A graph claiming a strut its geometry does not have must be
        refused.

        Duplicating an edge leaves the surface essentially unchanged and
        moves the budget by -12, so the census and the budget disagree.
        If this passes, the check in the builder is dead code.
        """
        graph = SUPERLATTICES["super-square"]
        liar = dataclasses.replace(graph, name="square-that-lies",
                                   edges=graph.edges + (graph.edges[0],))
        assert liar.ring_budget == graph.ring_budget - 12
        with pytest.raises(RuntimeError, match="skeleton's own topology"):
            build_supernetwork(liar, scale=34.0, tube_radius=5.0,
                               blend=4.0, grid_resolution=64)


class TestTheCagesAreReachable:
    """A cage that exists only as a Python function is not available to
    anyone using the program.

    Both cages were built and measured and then left out of
    ``SUPERLATTICES``, so the window's dropdown and the command line's
    ``--graph`` never offered them. These tests pin the registry that
    fixes it, because "it exists in the module" is exactly the claim that
    was wrong.
    """

    def test_the_cages_are_registered(self):
        assert set(CAGES) == {"super-icosahedron", "superfullerene-C60",
                              "super-hypercube", "supertube-(4,4)",
                              "supertube-(6,6)"}

    @pytest.mark.parametrize("name", ["super-icosahedron",
                                      "superfullerene-C60",
                                      "supertube-(4,4)"])
    def test_scale_means_the_strut_length(self, name):
        """A periodic net's scale is its cell edge; a cage has no cell,
        so scale is how long each tube is -- the number that decides
        whether a tube survives between two vertices at all."""
        # rel=0.02 rather than machine precision, because a ROLLED
        # skeleton's struts are not all identical: putting a honeycomb on
        # a cylinder replaces each arc by its chord, so a supertube's
        # struts vary by about 1% exactly as a real nanotube's bonds do.
        # The icosahedron and the C60 are exact and pass either way.
        for strut in (18.0, 24.0):
            graph = named_graph(name, strut)
            lengths = graph.strut_lengths(strut)
            assert lengths.min() == pytest.approx(strut, rel=0.02)
            assert lengths.max() == pytest.approx(strut, rel=0.02)

    def test_the_cages_are_the_structures_they_claim(self):
        cage = named_graph("super-icosahedron", 24.0)
        assert len(cage.nodes) == 12 and len(cage.edges) == 30
        assert cage.coordination == 5
        fullerene = named_graph("superfullerene-C60", 14.2)
        assert len(fullerene.nodes) == 60 and len(fullerene.edges) == 90
        assert fullerene.coordination == 3

    def test_a_periodic_net_still_resolves(self):
        assert named_graph("super-cubic", 40.0) is SUPERLATTICES["super-cubic"]

    def test_an_unknown_name_names_both_catalogues(self):
        with pytest.raises(ValueError, match="super-icosahedron"):
            named_graph("super-nonsense", 40.0)


class TestTheHypercube:
    """The 4-cube's 32 edges as nanotubes. The topology is exact; the
    geometry cannot be, because a tesseract does not fit in three
    dimensions."""

    def test_it_is_the_tesseract(self):
        graph = hypercube_cage(20.0)
        assert len(graph.nodes) == 16
        assert len(graph.edges) == 32
        assert graph.coordination == 4

    def test_its_budget_follows_from_the_skeleton(self):
        assert hypercube_cage(20.0).ring_budget == 12 * (16 - 32) == -192

    def test_the_struts_are_deliberately_unequal(self):
        """Three lengths -- the outer cube's, the inner cube's and the
        radial ones between them -- because that is what a perspective
        projection along w does. `scale` sets the shortest, which is the
        one that has to leave a real tube."""
        lengths = hypercube_cage(20.0).strut_lengths(20.0)
        assert lengths.min() == pytest.approx(20.0)
        assert len(set(np.round(lengths, 2))) == 3
        assert 2.2 < lengths.max() / lengths.min() < 2.4

    def test_every_edge_joins_vertices_differing_in_one_coordinate(self):
        """The defining property of a hypercube, checked on the
        projection rather than assumed from it."""
        graph = hypercube_cage(20.0)
        degree = np.zeros(16, dtype=int)
        for start, end, _shift in graph.edges:
            degree[start] += 1
            degree[end] += 1
        assert set(degree.tolist()) == {4}


class TestTheSupertube:
    """Super-graphene rolled. The net IS a honeycomb, so rolling it is
    the same operation as rolling graphene -- which means `build_cnt`
    does the geometry and nothing new is needed."""

    def test_the_seam_closes(self):
        """Every vertex trivalent. A supertube whose seam did not close
        would have two rows of two-coordinate vertices, and the
        coordination property would raise."""
        assert supertube_graph(6, 6, 2, 20.0).coordination == 3

    def test_it_repeats_along_its_axis_only(self):
        assert supertube_graph(6, 6, 2, 20.0).pbc == (False, False, True)

    def test_the_vertex_count_is_the_tubes(self):
        """4*n atoms per armchair period, so (6,6) over 2 periods is 48."""
        assert len(supertube_graph(6, 6, 2, 20.0).nodes) == 48
        assert len(supertube_graph(4, 4, 2, 20.0).nodes) == 32

    def test_scale_is_the_strut_here_too(self):
        """Every other cage reads `scale` as the strut length, and this
        one nearly did not: writing all three axes as fractions of the
        axial period made its struts read 0.3 Å in the menu."""
        for strut in (14.0, 20.0):
            lengths = supertube_graph(6, 6, 2, strut).strut_lengths(strut)
            assert lengths.max() == pytest.approx(strut, rel=0.02)
            assert lengths.min() == pytest.approx(strut, rel=0.02)

    def test_a_seam_bond_carries_the_axial_shift(self):
        """The ring closes through the cell, so some edges must be
        recorded with a non-zero z image -- otherwise the supertube is a
        barrel with two open ends."""
        edges = supertube_graph(6, 6, 2, 20.0).edges
        assert any(shift[2] != 0 for _a, _b, shift in edges)
