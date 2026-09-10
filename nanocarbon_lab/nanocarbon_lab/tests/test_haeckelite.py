"""Designing 2D carbon allotropes by patterning Stone-Wales rotations.

Two invariants run through everything here. A 2D periodic sheet is a
torus, so ``sum(6 - n)`` over its rings is exactly zero, and an edge flip
pays zero -- so every test that checks the budget is really checking that
the topology came from the mesh rather than from bare rewiring. And the
geometry comes from the *other* side: graphene's own dual with each
rotated dimer turned 90 deg in the plane, because a flat triangulation
carrying pentagons and heptagons can never have equal edges.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nanocarbon_lab.analyse.rings import trace_faces
from nanocarbon_lab.builders import fullerene_mesh as fm
from nanocarbon_lab.builders.haeckelite import (
    ANGLE_CEILING,
    ANGLE_FLOOR,
    BOND_CEILING,
    BOND_FLOOR,
    PATTERNS,
    apply_flips,
    build_haeckelite,
    describe_haeckelite,
    flippable,
    mesh_edges,
    select_flips,
    strain_score,
    triangular_torus_mesh,
)
from nanocarbon_lab.validation import run_basic_checks
from nanocarbon_lab.validation.quality import (
    HAECKELITE_ANGLE_RANGE,
    HAECKELITE_BOND_RANGE,
    sp2_quality,
)

BOND = 1.42
SPACING = BOND * math.sqrt(3.0)


def degrees_of(triangles: np.ndarray, n_vertices: int) -> np.ndarray:
    table = np.zeros(n_vertices, dtype=int)
    for a, b in mesh_edges(triangles):
        table[a] += 1
        table[b] += 1
    return table


class TestTriangularTorusMesh:
    """The seed must be a real triangulation of a torus."""

    @pytest.mark.parametrize("m,n", [(3, 2), (3, 4), (5, 3)])
    def test_it_has_the_counts_a_torus_forces(self, m, n):
        """``V - E + F = 0``, and every triangle has three edges.

        For a triangulation that means ``E = 3V`` and ``F = 2V``; the dual
        then has ``F`` atoms, which must equal graphene's ``4*m*n``.
        """
        vertices, triangles, _ = triangular_torus_mesh(m, n, SPACING)
        edges = mesh_edges(triangles)
        assert len(vertices) == 2 * m * n
        assert len(triangles) == 4 * m * n
        assert len(edges) == 3 * len(vertices)
        assert len(vertices) - len(edges) + len(triangles) == 0

    @pytest.mark.parametrize("m,n", [(3, 2), (4, 3)])
    def test_every_vertex_has_six_neighbours(self, m, n):
        vertices, triangles, _ = triangular_torus_mesh(m, n, SPACING)
        assert set(degrees_of(triangles, len(vertices)).tolist()) == {6}

    @pytest.mark.parametrize("m,n", [(3, 2), (3, 5)])
    def test_every_triangle_is_equilateral(self, m, n):
        vertices, triangles, box = triangular_torus_mesh(m, n, SPACING)
        for triangle in triangles:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                delta = fm.minimum_image(
                    vertices[triangle[b]] - vertices[triangle[a]], box)
                assert float(np.linalg.norm(delta)) == pytest.approx(
                    SPACING, abs=1e-9)

    def test_every_edge_borders_exactly_two_triangles(self):
        """The definition of a closed surface, and what `edge_flip` needs."""
        _, triangles, _ = triangular_torus_mesh(3, 3, SPACING)
        for first, second in mesh_edges(triangles):
            incident = [t for t in triangles if first in t and second in t]
            assert len(incident) == 2

    def test_a_cell_too_narrow_in_x_is_refused(self):
        """At m = 2 a vertex's two x-neighbours are the same vertex."""
        with pytest.raises(ValueError, match="at least 3"):
            triangular_torus_mesh(2, 3, SPACING)

    def test_n_equals_two_is_allowed_because_y_is_offset(self):
        """The asymmetry is real: the two sublattices are offset in y.

        Pinning it here because the obvious guard -- refuse both below the
        same number -- is wrong in one direction and was written that way
        first.
        """
        vertices, triangles, _ = triangular_torus_mesh(3, 2, SPACING)
        assert set(degrees_of(triangles, len(vertices)).tolist()) == {6}
        for first, second in mesh_edges(triangles):
            incident = [t for t in triangles if first in t and second in t]
            assert len(incident) == 2


class TestFlips:
    def test_a_flip_keeps_the_torus_counts(self):
        vertices, triangles, box = triangular_torus_mesh(4, 4, SPACING)
        first, second = mesh_edges(triangles)[0]
        _, flipped = fm.edge_flip((vertices, triangles), first, second)
        assert len(flipped) == len(triangles)
        assert len(mesh_edges(flipped)) == len(mesh_edges(triangles))

    def test_a_flip_pays_exactly_zero(self):
        """Two degrees down, two up. This is why no pattern can break Euler."""
        vertices, triangles, _ = triangular_torus_mesh(4, 4, SPACING)
        first, second = mesh_edges(triangles)[0]
        _, flipped = fm.edge_flip((vertices, triangles), first, second)
        before = degrees_of(triangles, len(vertices))
        after = degrees_of(flipped, len(vertices))
        assert int(np.sum(6 - before)) == 0
        assert int(np.sum(6 - after)) == 0
        changed = np.flatnonzero(before != after)
        assert len(changed) == 4
        assert sorted((after - before)[changed].tolist()) == [-1, -1, 1, 1]

    def test_flipping_onto_an_existing_edge_becomes_refused(self):
        """It would give the mesh a doubled edge.

        The dual of that is two rings sharing two bonds, which is not a
        surface -- and `dual_honeycomb` would build it without complaint.
        In the perfect lattice every flip is legal; the illegal ones appear
        once earlier flips have brought two opposite vertices together.
        """
        vertices, triangles, box = triangular_torus_mesh(4, 4, SPACING)
        assert all(flippable(triangles, a, b) for a, b in mesh_edges(triangles))
        for a, b in mesh_edges(triangles):
            if flippable(triangles, a, b):
                _, triangles = fm.edge_flip((vertices, triangles), a, b)
        assert not all(flippable(triangles, a, b)
                       for a, b in mesh_edges(triangles))


class TestPatterns:
    @pytest.mark.parametrize("pattern", [p for p in PATTERNS if p != "none"])
    def test_the_selection_is_vertex_disjoint(self, pattern):
        """Two flips sharing a vertex would each be chosen against a mesh
        the other had already changed."""
        vertices, triangles, box = triangular_torus_mesh(5, 5, SPACING)
        flips = select_flips(vertices, triangles, box, 5, 5, pattern=pattern,
                             density=0.4, seed=0)
        seen: set[int] = set()
        for first, second in flips:
            assert first not in seen and second not in seen
            seen.update((first, second))

    def test_none_selects_nothing(self):
        vertices, triangles, box = triangular_torus_mesh(4, 4, SPACING)
        assert select_flips(vertices, triangles, box, 4, 4,
                            pattern="none") == []

    def test_the_random_pattern_is_reproducible(self):
        vertices, triangles, box = triangular_torus_mesh(5, 5, SPACING)
        args = dict(pattern="random", density=0.3)
        first = select_flips(vertices, triangles, box, 5, 5, seed=7, **args)
        again = select_flips(vertices, triangles, box, 5, 5, seed=7, **args)
        other = select_flips(vertices, triangles, box, 5, 5, seed=8, **args)
        assert first == again
        assert first != other

    def test_an_unknown_pattern_lists_what_exists(self):
        vertices, triangles, box = triangular_torus_mesh(3, 3, SPACING)
        with pytest.raises(ValueError, match="Unknown pattern"):
            select_flips(vertices, triangles, box, 3, 3, pattern="spiral")


class TestApplyFlips:
    """The two rules that decide which requested flips actually happen.

    Both were found by building lattices that broke them, and both are
    invisible in the Euler budget -- which is why they need their own
    tests rather than being covered by the census.
    """

    def _setup(self, m=5, n=5):
        vertices, triangles, mesh_box = triangular_torus_mesh(m, n, SPACING)
        positions, bonds, _ = fm.dual_honeycomb((vertices, triangles),
                                               box=mesh_box)
        return vertices, triangles, mesh_box, positions, bonds

    def test_every_flip_lands_on_four_degree_six_vertices(self):
        """Otherwise the degree changes stack and the census grows squares
        and nonagons, which is sound topology and not a haeckelite."""
        vertices, triangles, mesh_box, positions, bonds = self._setup()
        flips = select_flips(vertices, triangles, mesh_box, 5, 5,
                             pattern="r57")
        final, labels, dimers, refused = apply_flips(
            vertices, triangles, mesh_box, flips, bonds, len(positions))
        _, _, rings = fm.dual_honeycomb((vertices, final), box=mesh_box)
        sizes = {len(ring) for ring in rings}
        assert sizes <= {5, 6, 7}, sizes
        assert dimers
        assert refused > 0, "a dense pattern must have candidates refused"

    def test_no_two_rotated_dimers_are_bonded(self):
        """Two rotations sharing a bond each turn an atom the other needs,
        and the pair lands 3.4 Å apart."""
        vertices, triangles, mesh_box, positions, bonds = self._setup()
        flips = select_flips(vertices, triangles, mesh_box, 5, 5,
                             pattern="r57")
        _, _, dimers, _ = apply_flips(vertices, triangles, mesh_box, flips,
                                      bonds, len(positions))
        rotated = {atom for dimer in dimers for atom in dimer}
        assert len(rotated) == 2 * len(dimers), "a dimer shares an atom"
        own = {tuple(sorted(dimer)) for dimer in dimers}
        for first, second in bonds:
            if tuple(sorted((first, second))) in own:
                continue      # a dimer's own bond: the two atoms it rotates
            assert not (first in rotated and second in rotated), (
                f"atoms {first} and {second} are bonded and both rotated")

    def test_the_budget_survives_every_applied_flip(self):
        vertices, triangles, mesh_box, positions, bonds = self._setup()
        flips = select_flips(vertices, triangles, mesh_box, 5, 5,
                             pattern="random", density=0.5, seed=3)
        final, _, _, _ = apply_flips(vertices, triangles, mesh_box, flips,
                                     bonds, len(positions))
        _, _, rings = fm.dual_honeycomb((vertices, final), box=mesh_box)
        assert sum(6 - len(ring) for ring in rings) == 0


class TestBuild:
    """Every lattice the engine produces must be a real material."""

    CASES = [
        ("none", {}),
        ("r57", {}),
        ("stripes", {"period": 2}),
        ("sparse", {"period": 2}),
        ("random", {"density": 0.15}),
    ]

    @pytest.mark.parametrize("pattern,extra", CASES,
                             ids=[c[0] for c in CASES])
    def test_the_euler_budget_is_exactly_zero(self, pattern, extra):
        """A torus owes nothing, and a flip pays nothing."""
        atoms = build_haeckelite(nx=4, ny=4, pattern=pattern, seed=1, **extra)
        assert atoms.info["euler"] == 0
        counts = atoms.info["ring_counts"]
        assert sum((6 - size) * count for size, count in counts.items()) == 0

    @pytest.mark.parametrize("pattern,extra", CASES,
                             ids=[c[0] for c in CASES])
    def test_the_geometry_is_physical(self, pattern, extra):
        """Right topology and impossible geometry is the combination this
        framework exists to avoid, so the bonds are checked, not assumed.

        The window is the builder's own gate, which a build that fails it
        raises on -- so this test is really pinning that the gate is
        reached rather than bypassed.
        """
        atoms = build_haeckelite(nx=4, ny=4, pattern=pattern, seed=1, **extra)
        geometry = atoms.info["geometry"]
        assert BOND_FLOOR <= geometry["bond_min"], geometry
        assert geometry["bond_max"] <= BOND_CEILING, geometry
        assert ANGLE_FLOOR <= geometry["angle_min"], geometry
        assert geometry["angle_max"] <= ANGLE_CEILING, geometry
        assert geometry["n_close_contacts"] == 0, geometry

    @pytest.mark.parametrize("pattern,extra", CASES,
                             ids=[c[0] for c in CASES])
    def test_it_passes_validation(self, pattern, extra):
        atoms = build_haeckelite(nx=4, ny=4, pattern=pattern, seed=1, **extra)
        report = run_basic_checks(atoms)
        assert report.ok, report.summary()

    @pytest.mark.parametrize("pattern,extra", CASES,
                             ids=[c[0] for c in CASES])
    def test_the_sheet_embeds_the_mesh_it_came_from(self, pattern, extra):
        """The rings are read off the mesh, where they are exact.

        Tracing the finished sheet's own faces has to agree with them, or
        the ring list beside the coordinates does not describe them.
        """
        atoms = build_haeckelite(nx=4, ny=4, pattern=pattern, seed=1, **extra)
        pairs = np.asarray(atoms.info["bonds"], dtype=int)
        faces, boundary = trace_faces(atoms, pairs, max_size=12)
        traced: dict[int, int] = {}
        for face in faces:
            traced[len(face)] = traced.get(len(face), 0) + 1
        assert boundary == 0
        assert traced == atoms.info["ring_counts"]

    @pytest.mark.parametrize("pattern,extra", CASES,
                             ids=[c[0] for c in CASES])
    def test_the_area_per_atom_stays_graphene_s(self, pattern, extra):
        """The cell search's own guardrail.

        Given out-of-plane freedom the search bought low bond strain by
        crumpling the sheet into a smaller footprint -- a 4x4 R5,7 came
        back at 1.51 Å^2 per atom against graphene's 2.619. Pentagons and
        heptagons change the area a little; they do not halve it.
        """
        atoms = build_haeckelite(nx=4, ny=4, pattern=pattern, seed=1, **extra)
        area = atoms.info["cell_a"] * atoms.info["cell_b"] / len(atoms)
        assert 2.45 <= area <= 2.95, area

    @pytest.mark.parametrize("pattern,extra", CASES,
                             ids=[c[0] for c in CASES])
    def test_the_sheet_is_flat(self, pattern, extra):
        """Deliberately: the force field has no flexural term, so any
        buckling amplitude it produced would be its artefact."""
        atoms = build_haeckelite(nx=4, ny=4, pattern=pattern, seed=1, **extra)
        z = atoms.positions[:, 2]
        assert float(z.max() - z.min()) < 1e-6

    def test_no_pattern_changes_the_atom_count(self):
        """A flip moves bonds, never atoms."""
        counts = {build_haeckelite(nx=4, ny=4, pattern=pattern, seed=1,
                                   density=0.2).get_global_number_of_atoms()
                  for pattern in PATTERNS}
        assert counts == {4 * 4 * 4}

    def test_no_pattern_gives_a_three_or_four_membered_ring(self):
        """Those are not carbon chemistry at any density."""
        for pattern, extra in self.CASES:
            atoms = build_haeckelite(nx=4, ny=4, pattern=pattern, seed=2,
                                     **extra)
            assert min(atoms.info["ring_counts"]) >= 5, pattern
            assert max(atoms.info["ring_counts"]) <= 7, pattern

    def test_every_atom_is_three_coordinate(self):
        from nanocarbon_lab.topology import coordination_numbers

        atoms = build_haeckelite(nx=4, ny=4, pattern="r57", seed=1)
        assert set(coordination_numbers(atoms).tolist()) == {3}

    def test_the_vacuum_is_a_gap_not_the_cell_length(self):
        """Setting the cell length *to* the vacuum quietly failed the 10 Å
        guardrail as soon as the sheet had any thickness."""
        atoms = build_haeckelite(nx=4, ny=4, pattern="r57", seed=1,
                                 vacuum=14.0)
        span = float(atoms.positions[:, 2].max() - atoms.positions[:, 2].min())
        assert float(atoms.cell[2, 2]) - span == pytest.approx(14.0, abs=1e-6)

    def test_refused_flips_are_reported_not_hidden(self):
        """A pattern is a request; most of a dense one is turned down, and
        the census is the only honest account of what was built."""
        atoms = build_haeckelite(nx=4, ny=4, pattern="r57", seed=1)
        assert atoms.info["n_flips"] > 0
        assert atoms.info["n_flips_refused"] > atoms.info["n_flips"]


class TestGrapheneIsTheBaseline:
    """`none` must reproduce graphene exactly, or nothing else is trustworthy."""

    def test_it_is_all_hexagons_at_the_ideal_bond(self):
        atoms = build_haeckelite(nx=4, ny=4, pattern="none")
        assert atoms.info["ring_counts"] == {6: 32}
        geometry = atoms.info["geometry"]
        assert geometry["bond_min"] == pytest.approx(BOND, abs=1e-5)
        assert geometry["bond_max"] == pytest.approx(BOND, abs=1e-5)
        assert geometry["angle_mean"] == pytest.approx(120.0, abs=1e-4)

    def test_the_cell_search_finds_graphene_s_own_cell(self):
        """The search validates itself here: it must not move."""
        atoms = build_haeckelite(nx=4, ny=4, pattern="none")
        assert atoms.info["cell_a"] == pytest.approx(4 * math.sqrt(3) * BOND,
                                                     abs=1e-6)
        assert atoms.info["cell_b"] == pytest.approx(4 * 3.0 * BOND, abs=1e-6)
        assert atoms.info["strain_score"] == pytest.approx(0.0, abs=1e-4)

    def test_it_is_judged_clean(self):
        atoms = build_haeckelite(nx=4, ny=4, pattern="none")
        verdict, _ = sp2_quality(atoms.info["geometry"],
                                 atoms.info["quality_family"])
        assert verdict == "clean"


class TestASingleDefect:
    """The calibration point: one Stone-Wales rotation in a large cell.

    Its relaxed geometry is published -- 1.32-1.48 Å -- so this is the one
    case where the engine can be checked against the literature rather
    than against itself.
    """

    def test_it_reproduces_the_published_5_7_7_5_geometry(self):
        atoms = build_haeckelite(nx=6, ny=6, pattern="sparse", period=6,
                                 seed=1)
        assert atoms.info["ring_counts"] == {5: 2, 6: 68, 7: 2}
        geometry = atoms.info["geometry"]
        assert 1.30 <= geometry["bond_min"] <= 1.34, geometry
        assert 1.45 <= geometry["bond_max"] <= 1.50, geometry
        assert geometry["n_close_contacts"] == 0

    def test_a_lone_defect_barely_moves_the_cell(self):
        """Two pentagons and two heptagons in 72 rings is a perturbation."""
        defected = build_haeckelite(nx=6, ny=6, pattern="sparse", period=6,
                                    seed=1)
        pristine = build_haeckelite(nx=6, ny=6, pattern="none")
        for key in ("cell_a", "cell_b"):
            ratio = defected.info[key] / pristine.info[key]
            assert 0.95 < ratio < 1.05, (key, ratio)


class TestDensePatterns:
    """The dense limit, and the one that broke every shortcut."""

    def test_r57_is_mostly_pentagons_and_heptagons(self):
        atoms = build_haeckelite(nx=6, ny=6, pattern="r57", seed=1)
        counts = atoms.info["ring_counts"]
        assert set(counts) == {5, 6, 7}
        assert counts[5] == counts[7], "a flip makes them in pairs"
        assert atoms.info["non_hexagonal_fraction"] > 0.6

    def test_a_bigger_cell_reaches_a_denser_lattice(self):
        """The rules are local, so a larger cell packs more of them in."""
        small = build_haeckelite(nx=4, ny=4, pattern="r57", seed=1)
        large = build_haeckelite(nx=6, ny=6, pattern="r57", seed=1)
        assert (large.info["non_hexagonal_fraction"]
                > small.info["non_hexagonal_fraction"])

    def test_its_verdict_is_reported_against_its_own_window(self):
        atoms = build_haeckelite(nx=6, ny=6, pattern="r57", seed=1)
        verdict, reason = sp2_quality(atoms.info["geometry"],
                                      atoms.info["quality_family"])
        assert verdict in ("clean", "strained"), reason
        assert reason

    def test_the_default_sp2_window_would_misjudge_it(self):
        """Pinning the bug this fixed: a heptagon's interior angle is
        128.6 deg before any strain, so the hexagonal window called every
        sound haeckelite BROKEN."""
        atoms = build_haeckelite(nx=6, ny=6, pattern="r57", seed=1)
        assert sp2_quality(atoms.info["geometry"])[0] == "broken"
        assert sp2_quality(atoms.info["geometry"], "haeckelite")[0] != "broken"


class TestTheGeometryGate:
    """The gate exists; the point of these tests is that it never fires.

    Without the selection rules, dense patterns came back at 1.225-1.663 Å
    -- a converged minimum of the force field and not a material. With
    them, every lattice the engine will build passes. So the gate is a
    guard rather than the mechanism, and what is worth pinning is that the
    *rules* are what keep the geometry sound.
    """

    MATRIX = [
        (nx, pattern, extra)
        for nx in (3, 5, 6)
        for pattern, extra in (
            ("r57", {}),
            ("stripes", {"period": 1}),
            ("sparse", {"period": 1}),
            ("random", {"density": 1.0}),
        )
    ]

    @pytest.mark.parametrize("nx,pattern,extra", MATRIX,
                             ids=[f"{n}-{p}-{tuple(e.values())}"
                                  for n, p, e in MATRIX])
    def test_what_the_rules_admit_always_passes_the_gate(self, nx, pattern,
                                                         extra):
        """Asked for the densest pattern it can draw, at every cell size,
        the engine must not need the gate to save it."""
        atoms = build_haeckelite(nx=nx, ny=nx, pattern=pattern, seed=1,
                                 **extra)
        geometry = atoms.info["geometry"]
        assert geometry["n_close_contacts"] == 0
        assert BOND_FLOOR <= geometry["bond_min"]
        assert geometry["bond_max"] <= BOND_CEILING
        assert set(atoms.info["ring_counts"]) <= {5, 6, 7}
        assert atoms.info["euler"] == 0

    def test_the_window_the_gate_uses_matches_the_verdict_s(self):
        """Two windows that disagree would let the builder return a sheet
        its own quality verdict calls broken."""
        bond_lo, bond_hi = HAECKELITE_BOND_RANGE
        angle_lo, angle_hi = HAECKELITE_ANGLE_RANGE
        assert BOND_FLOOR <= bond_lo
        assert bond_hi <= BOND_CEILING
        assert ANGLE_FLOOR <= angle_lo
        assert angle_hi <= ANGLE_CEILING


class TestStrainScore:
    def test_it_needs_the_angle_term(self):
        """Bond strain alone cannot find the cell.

        Squeeze a sheet and it buckles rather than compressing its bonds,
        so the bonds stay at 1.42 Å while the cell collapses. Buckling
        costs angle energy, and only that term sees it.
        """
        vertices, triangles, box = triangular_torus_mesh(4, 4, SPACING)
        positions, bonds, _ = fm.dual_honeycomb((vertices, triangles), box=box)
        flat_box = np.array([box[0], box[1], 0.0])

        ideal = strain_score(positions, bonds, flat_box, BOND)
        squeezed = positions * np.array([0.9, 0.9, 1.0])
        pinched = strain_score(squeezed, bonds,
                               flat_box * np.array([0.9, 0.9, 1.0]), BOND)
        assert ideal == pytest.approx(0.0, abs=1e-6)
        assert pinched > ideal

    def test_an_overlap_vetoes_a_candidate_outright(self):
        """A folded sheet can have excellent bonds and angles everywhere --
        they are local, and folding is not."""
        vertices, triangles, box = triangular_torus_mesh(4, 4, SPACING)
        positions, bonds, _ = fm.dual_honeycomb((vertices, triangles), box=box)
        flat_box = np.array([box[0], box[1], 0.0])
        collapsed = positions.copy()
        collapsed[1] = collapsed[0] + np.array([0.1, 0.0, 0.0])
        assert strain_score(collapsed, bonds, flat_box, BOND) == float("inf")


class TestDescribe:
    def test_it_reports_the_census_and_the_budget(self):
        atoms = build_haeckelite(nx=4, ny=4, pattern="stripes", period=2,
                                 seed=1)
        text = describe_haeckelite(atoms)
        assert "stripes" in text
        assert "sum(6-n) = +0" in text
        assert "non-hexagonal" in text
