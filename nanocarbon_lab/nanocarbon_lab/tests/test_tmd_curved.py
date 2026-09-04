"""Tests for MX2 on a triply periodic minimal surface.

The interesting failures here are topological rather than numerical, so
most of these assert invariants that hold for any cell: the ring budget
against the surface's own genus, the M/X alternation, and the fact that
what alternation is left over is a homology obstruction rather than a bug.

Builds are ~20 s each, so the ones that need a whole cell are marked
slow; the parity and colouring machinery is tested directly, which is
fast and is where the reasoning actually lives.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import fullerene_mesh as fm
from nanocarbon_lab.builders import implicit as im
from nanocarbon_lab.builders import remesh as rm
from nanocarbon_lab.tmd.curved import (
    MIN_SCHWARZITE_CELL,
    build_tmd_schwarzite,
    odd_vertices,
    repair_parity_by_flipping,
    repair_parity_by_splitting,
    schwarzite_quality,
    two_colour,
)
from nanocarbon_lab.tmd.quality import geometry_report


@pytest.fixture(scope="module")
def small_mesh():
    """A remeshed Schwarz P cell, before any parity work."""
    field, _ = im.schwarzite_field("primitive", cell=30.0, thickness=0.0)
    mesh = rm.periodic_marching_cubes_mesh(field, 30.0, resolution=64)
    mesh = rm.isotropic_remesh(
        mesh, field, target_edge=3.16, iterations=25, box=30.0,
        anneal_sweeps=0, rng=np.random.default_rng(0),
    )
    return mesh, field, 30.0


class TestTwoColouring:
    def test_an_even_cycle_is_bipartite(self):
        bonds = [(0, 1), (1, 2), (2, 3), (3, 0)]
        _, frustrated = two_colour(4, bonds)
        assert frustrated == 0

    def test_an_odd_cycle_is_not(self):
        """One bond must stay homoelemental, and no colouring beats that."""
        bonds = [(0, 1), (1, 2), (2, 0)]
        _, frustrated = two_colour(3, bonds)
        assert frustrated == 1

    def test_it_finds_the_minimum_on_two_odd_cycles(self):
        # Two disjoint triangles: exactly one bad bond each, no more.
        bonds = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)]
        _, frustrated = two_colour(6, bonds)
        assert frustrated == 2


class TestParityRepair:
    def test_the_remesh_leaves_odd_rings(self, small_mesh):
        """Pentagons and heptagons -- fine in carbon, impossible in MX2."""
        mesh, _, _ = small_mesh
        assert odd_vertices(mesh)

    def test_splitting_reaches_exactly_zero_odd(self, small_mesh):
        """A split toggles the two vertices opposite the edge and nothing
        else, which is the weight-two move that can annihilate a pair."""
        mesh, field, cell = small_mesh
        repaired, splits = repair_parity_by_splitting(mesh, field, cell, 3.16)
        assert splits > 0
        assert odd_vertices(repaired) == set()

    def test_splitting_preserves_the_surface(self, small_mesh):
        """Euler characteristic and the closed-manifold property are what
        a bad face rewrite would break first."""
        mesh, field, cell = small_mesh
        before = rm.mesh_statistics(mesh)
        repaired, _ = repair_parity_by_splitting(mesh, field, cell, 3.16)
        after = rm.mesh_statistics(repaired)
        assert after["euler"] == before["euler"]
        assert after["boundary_edges"] == 0

    def test_splitting_keeps_the_ring_budget(self, small_mesh):
        """sum(6 - n) is fixed by the genus, whatever the repair does."""
        mesh, field, cell = small_mesh
        repaired, _ = repair_parity_by_splitting(mesh, field, cell, 3.16)
        rings = rm.degree_histogram(repaired)
        deficit = sum((6 - k) * v for k, v in rings.items())
        assert deficit == 6 * rm.mesh_statistics(repaired)["euler"]

    def test_flipping_reduces_odd_without_adding_vertices(self, small_mesh):
        """Flips only rewire, which is why their geometry survives where
        splitting's does not -- but a flip toggles four degrees at once,
        so it cannot annihilate the last pair and will not reach zero."""
        mesh, _, _ = small_mesh
        before = len(odd_vertices(mesh))
        flipped, odd = repair_parity_by_flipping(mesh, sweeps=200, seed=0)
        assert len(flipped[0]) == len(mesh[0])
        assert odd < before

    def test_flipping_preserves_the_surface(self, small_mesh):
        mesh, _, _ = small_mesh
        flipped, _ = repair_parity_by_flipping(mesh, sweeps=200, seed=0)
        stats = rm.mesh_statistics(flipped)
        assert stats["euler"] == rm.mesh_statistics(mesh)["euler"]
        assert stats["boundary_edges"] == 0


class TestRelaxerExtensions:
    """The two `relax_shell` options this builder needed."""

    def test_excluding_13_pairs_lets_them_collapse(self):
        """The measured failure: with no angle term and 1-3 pairs excluded
        from the repulsion, two ligands of the same centre have nothing at
        all holding them apart. Three atoms bonded to one centre is the
        smallest case that shows it.
        """
        # A centre at the origin with three ligands, deliberately crowded.
        positions = np.array([
            [0.0, 0.0, 0.0],
            [2.4, 0.0, 0.0],
            [2.3, 0.3, 0.0],
            [2.3, -0.3, 0.0],
        ])
        bonds = {(0, 1), (0, 2), (0, 3)}
        kwargs = dict(equilibrium=2.404, k_bond=40.0, k_angle=0.0,
                      k_repel=60.0, repel_cutoff=3.0, repel_skin=1.0,
                      outer_cycles=3, max_iterations=2000)
        excluded = fm.relax_shell(positions.copy(), bonds, exclude_13=True,
                                  **kwargs)
        included = fm.relax_shell(positions.copy(), bonds, exclude_13=False,
                                  **kwargs)

        def closest_ligand_gap(p):
            return min(np.linalg.norm(p[i] - p[j])
                       for i, j in ((1, 2), (1, 3), (2, 3)))

        assert closest_ligand_gap(included) > closest_ligand_gap(excluded)
        assert closest_ligand_gap(included) > 2.0

    def test_per_bond_equilibrium_is_honoured(self):
        """A binary net has a few homoelemental defect bonds that are not
        the M-X length; forcing them to it warps everything nearby."""
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                              [4.0, 0.0, 0.0]])
        targets = {(0, 1): 2.404, (1, 2): 2.80}
        out = fm.relax_shell(positions.copy(), set(targets),
                             equilibrium=targets, k_bond=40.0, k_angle=0.0,
                             k_repel=0.0, outer_cycles=2, max_iterations=2000)
        assert np.linalg.norm(out[1] - out[0]) == pytest.approx(2.404, abs=1e-3)
        assert np.linalg.norm(out[2] - out[1]) == pytest.approx(2.80, abs=1e-3)

    def test_a_missing_per_bond_length_is_rejected(self):
        positions = np.zeros((3, 3))
        with pytest.raises(ValueError, match="missing a length"):
            fm.relax_shell(positions, {(0, 1), (1, 2)},
                           equilibrium={(0, 1): 2.4})

    def test_carbon_still_excludes_13_by_default(self):
        """The default must not change: sp2 carbon's angle term owns those
        pairs, and repelling them as well would fight it."""
        import inspect

        signature = inspect.signature(fm.relax_shell)
        assert signature.parameters["exclude_13"].default is True


class TestValidation:
    def test_an_unknown_parity_is_rejected(self):
        with pytest.raises(ValueError, match="parity must be"):
            build_tmd_schwarzite(parity="anneal")

    def test_a_cell_below_the_floor_is_rejected(self):
        floor = MIN_SCHWARZITE_CELL["primitive"]
        with pytest.raises(ValueError, match="too small"):
            build_tmd_schwarzite(cell=floor - 5.0)

    def test_the_floor_exceeds_the_carbon_one(self):
        """The MX2 sandwich is ~3.1 Å thick where graphene is one atom, so
        a channel a carbon net tiles happily is too narrow here."""
        from nanocarbon_lab.builders.junction import (
            MIN_SCHWARZITE_CELL as CARBON_FLOOR,
        )

        for kind, floor in MIN_SCHWARZITE_CELL.items():
            assert floor >= CARBON_FLOOR.get(kind, 22.0) or floor >= 30.0


@pytest.mark.slow
class TestBuild:
    @pytest.mark.parametrize("parity", ["none", "flip", "split"])
    def test_the_ring_budget_matches_the_genus(self, parity):
        """sum(6 - n) = 6*chi, never the sphere's hardcoded +12: a
        schwarzite is legitimately, strongly negative."""
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                     parity=parity)
        assert atoms.info["ring_deficit"] == 6 * atoms.info["euler"]
        assert atoms.info["genus"] == 3

    @pytest.mark.parametrize("parity", ["none", "flip", "split"])
    def test_no_atoms_overlap(self, parity):
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                     parity=parity)
        assert geometry_report(atoms)["n_close_contacts"] == 0

    @pytest.mark.parametrize("parity", ["none", "flip", "split"])
    def test_coordination_never_exceeds_the_chemistry(self, parity):
        """Six for the metal, three for the chalcogen, from the bond graph
        -- a distance cutoff reads this wrong on a saddle."""
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                     parity=parity)
        assert atoms.info["graph_metal_coordination"][1] <= 6
        assert atoms.info["graph_chalcogen_coordination"][1] <= 3

    def test_splitting_removes_every_odd_ring(self):
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                     parity="split")
        assert atoms.info["odd_rings"] == 0
        assert all(size % 2 == 0 for size in atoms.info["ring_counts"])

    def test_even_rings_also_fix_the_stoichiometry(self):
        """X/M = 2 needs the two sublattices to be the same size, and an
        even-degree triangulation is what balances them. Left unrepaired
        the colour classes drift apart -- 1.89 at this cell, 5% off."""
        ragged = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                      parity="none")
        even = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                    parity="split")
        assert even.info["stoichiometry"] == pytest.approx(2.0, abs=0.01)
        assert abs(even.info["stoichiometry"] - 2.0) < abs(
            ragged.info["stoichiometry"] - 2.0)

    def test_better_parity_costs_geometry(self):
        """The trade the `parity` option exists to expose. Both halves are
        asserted because a change that improved one silently at the other's
        expense would otherwise look like a win."""
        built = {p: build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                         parity=p)
                 for p in ("none", "split")}
        assert (built["split"].info["antiphase_fraction"]
                < built["none"].info["antiphase_fraction"])
        assert (built["split"].info["bond_deviation_p95"]
                > built["none"].info["bond_deviation_p95"])

    def test_a_perfectly_alternating_cell_is_reachable(self):
        """Even degrees are only a *sphere* result: at genus 3 there are
        six more Z/2 classes, and even-degree meshes exist on both sides
        of them. So the obstruction is a property of the triangulation,
        not of the surface, and it can come out zero -- this cell is
        perfectly bipartite and exactly MX2, which is the whole point of
        the split repair.
        """
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                     parity="split")
        assert atoms.info["odd_rings"] == 0
        assert atoms.info["antiphase_bonds"] == 0
        assert atoms.info["stoichiometry"] == pytest.approx(2.0, abs=1e-9)

    def test_but_it_is_not_guaranteed(self):
        """At other cells the same repair lands on the other side of the
        homology class and a few bonds stay homoelemental. Worth pinning:
        it is why the count is reported rather than assumed to be zero."""
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=36.0,
                                     parity="split")
        assert atoms.info["odd_rings"] == 0
        assert atoms.info["antiphase_bonds"] > 0

    def test_it_is_periodic_in_three_dimensions(self):
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                     parity="none")
        assert all(atoms.get_pbc())
        assert atoms.cell.lengths() == pytest.approx([30.0] * 3)

    def test_the_verdict_names_the_boundary(self):
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                     parity="none")
        verdict, why = schwarzite_quality(atoms)
        assert verdict in ("clean", "strained", "broken")
        assert "homoelemental" in why

    def test_the_verdict_does_not_invent_a_boundary_when_there_is_none(self):
        """With zero homoelemental bonds there is no inversion domain, and
        saying '0.0% of bonds are an inversion-domain boundary' is
        nonsense — the cell alternates perfectly."""
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=30.0,
                                     parity="split")
        assert atoms.info["antiphase_bonds"] == 0
        _, why = schwarzite_quality(atoms)
        assert "0.0% of bonds are homoelemental" not in why
        assert "alternates perfectly" in why

    def test_a_torn_grid_is_retried_not_returned(self):
        """Schwarz P at 42 Å tears at resolution 64 -- neighbouring sites
        21 Å apart -- and is clean at 72. Whether a given neck falls
        between sample points is an artefact of the grid, so the builder
        retries on a shifted one; before it did, this cell came back with
        a 300% bond."""
        atoms = build_tmd_schwarzite("MoS2", "primitive", cell=42.0,
                                     parity="none", grid_resolution=64)
        assert atoms.info["bond_deviation_max"] < 0.5

    def test_another_surface_and_material_build(self):
        atoms = build_tmd_schwarzite("WS2", "gyroid", cell=34.0,
                                     parity="none")
        assert atoms.info["genus"] == 5
        assert atoms.info["ring_deficit"] == 6 * atoms.info["euler"]
        assert set(atoms.get_chemical_symbols()) == {"W", "S"}
