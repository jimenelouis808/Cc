"""Tests for implicit-surface junctions and schwarzite fragments.

The invariant that matters throughout is Euler's: for a closed carbon
shell, ``sum(6 - ring_size)`` equals ``6 * chi``. It is 12 for anything
sphere-like -- a capped tube, or a junction with any number of arms --
and goes sharply negative once the surface grows handles. Nothing in the
builder prescribes ring counts, so these tests check that the topology
that falls out of the geometry is actually consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import build_junction, build_schwarzite
from nanocarbon_lab.builders import implicit as im
from nanocarbon_lab.builders import remesh as rm

pytest.importorskip("skimage", reason="scikit-image provides marching cubes")


def _deficit(ring_counts: dict[int, int]) -> int:
    return sum((6 - size) * count for size, count in ring_counts.items())


class TestImplicitFields:
    def test_capsule_distance_is_signed_and_correct(self):
        field = im.capsule((0, 0, -5), (0, 0, 5), radius=3.0)
        probes = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0], [0, 0, 10]], dtype=float)
        values = field(probes)
        assert values[0] == pytest.approx(-3.0)  # centre, 3 Å inside
        assert values[1] == pytest.approx(0.0)   # on the surface
        assert values[2] == pytest.approx(3.0)   # 3 Å outside
        assert values[3] == pytest.approx(2.0)   # beyond the rounded cap

    def test_smooth_union_is_bounded_by_the_hard_union(self):
        a = im.capsule((-10, 0, 0), (10, 0, 0), 3.0)
        b = im.capsule((0, -10, 0), (0, 10, 0), 3.0)
        blended = im.smooth_union(a, b, blend=2.0)
        probes = np.random.default_rng(0).uniform(-12, 12, size=(200, 3))
        hard = np.minimum(a(probes), b(probes))
        # Smoothing only ever adds material, never removes it.
        assert np.all(blended(probes) <= hard + 1e-9)

    def test_clip_binds_outside_the_ball(self):
        inner = im.capsule((-50, 0, 0), (50, 0, 0), 3.0)
        clipped = im.intersect_with_ball(inner, radius=10.0, softness=1.0)
        far = np.array([[40.0, 0.0, 0.0]])
        # Far outside the ball the clip must dominate, or it does not clip
        # at all and the surface escapes the sampling box.
        assert clipped(far)[0] > 0

    def test_normalisation_puts_a_trig_field_into_distance_units(self):
        raw, _ = im.schwarzite_field("primitive", cell=20.0)
        normalised = im.normalize_to_distance(raw)
        probes = np.random.default_rng(1).uniform(-10, 10, size=(200, 3))
        # The zero set is unchanged in sign, but values become O(Å).
        assert np.all(np.sign(raw(probes)) == np.sign(normalised(probes)))
        assert np.abs(normalised(probes)).max() < np.abs(raw(probes)).max() * 20

    def test_unknown_kinds_raise(self):
        with pytest.raises(ValueError):
            im.junction_field("Z")
        with pytest.raises(ValueError):
            field, _ = im.schwarzite_field("pretzel")
            field(np.zeros((1, 3)))


@pytest.fixture(scope="module")
def remeshed():
    """Raw and remeshed Y-junction meshes, shared across the remesh tests."""
    field, extent = im.junction_field("Y", tube_radius=6.0, arm_length=18.0)
    mesh = rm.marching_cubes_mesh(field, extent, resolution=56)
    return field, mesh, rm.isotropic_remesh(mesh, field, target_edge=2.46)


class TestRemesher:
    def test_marching_cubes_gives_a_closed_manifold(self, remeshed):
        _, mesh, _ = remeshed
        stats = rm.mesh_statistics(mesh)
        assert stats["boundary_edges"] == 0
        assert stats["euler"] == 2  # capped junction is sphere-like

    def test_remesh_preserves_topology(self, remeshed):
        _, mesh, out = remeshed
        assert rm.mesh_statistics(out)["euler"] == rm.mesh_statistics(mesh)["euler"]
        assert rm.mesh_statistics(out)["boundary_edges"] == 0

    def test_remesh_removes_impossible_ring_sizes(self, remeshed):
        _, mesh, out = remeshed
        before = rm.degree_histogram(mesh)
        after = rm.degree_histogram(out)
        # Vertex degree becomes ring size, so degrees below 5 would be
        # three- and four-membered carbon rings.
        assert min(before) < 5, "raw marching cubes should have low degrees to fix"
        assert min(after) >= 5
        assert max(after) <= 8
        # ...and the bulk should end up hexagonal.
        assert after[6] / sum(after.values()) > 0.7

    def test_remesh_rejects_nonpositive_target(self, remeshed):
        field, mesh, _ = remeshed
        with pytest.raises(ValueError):
            rm.isotropic_remesh(mesh, field, target_edge=0.0)


class TestJunctions:
    @pytest.mark.parametrize("kind", ["L", "T", "Y", "X"])
    def test_junction_topology_and_geometry(self, kind):
        atoms = build_junction(kind, arm_length=18.0, grid_resolution=56)
        info = atoms.info
        assert info["genus"] == 0  # capped, however many arms
        assert _deficit(info["ring_counts"]) == 12
        assert set(info["ring_counts"]) <= {5, 6, 7, 8}

        g = info["geometry"]
        assert g["n_close_contacts"] == 0
        assert 1.25 < g["bond_min"] <= g["bond_max"] < 1.60
        assert 95.0 < g["angle_min"] <= g["angle_max"] < 140.0

    def test_branches_carry_heptagons(self):
        """A branch is a saddle, and saddles need negative curvature.

        Nothing tells the builder this -- heptagons emerge from the
        remeshed geometry -- so it is worth asserting that a branched
        structure really does grow them, and more of them than a plain
        elbow with a gentler neck.
        """
        elbow = build_junction("L", arm_length=18.0, grid_resolution=56)
        tee = build_junction("X", arm_length=18.0, grid_resolution=56)
        assert elbow.info["ring_counts"].get(7, 0) > 0
        assert tee.info["ring_counts"].get(7, 0) > elbow.info["ring_counts"].get(7, 0)

    def test_all_atoms_three_coordinate(self):
        atoms = build_junction("Y", arm_length=18.0, grid_resolution=56)
        degree = np.zeros(len(atoms), dtype=int)
        for a, b in atoms.info["bonds"]:
            degree[a] += 1
            degree[b] += 1
        assert np.all(degree == 3)

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            build_junction("Y", tube_radius=0.0)
        with pytest.raises(ValueError):
            build_junction("Y", arm_length=-1.0)


class TestSchwarzites:
    # Genus of one period of each triply periodic minimal surface. These are
    # the textbook values, and getting them out of the mesh is the check that
    # the periodic weld really closed the cell on the 3-torus rather than
    # leaving a torn slab.
    EXPECTED_GENUS = {"primitive": 3, "gyroid": 5, "diamond": 9}

    @pytest.mark.parametrize("kind", ["primitive", "gyroid"])
    def test_unit_cell_has_the_textbook_genus_and_ring_budget(self, kind):
        atoms = build_schwarzite(kind, cell=36.0)
        info = atoms.info
        assert info["genus"] == self.EXPECTED_GENUS[kind]
        assert info["euler"] == 2 - 2 * info["genus"]
        assert _deficit(info["ring_counts"]) == 6 * info["euler"]

    @pytest.mark.parametrize("kind", ["primitive", "gyroid"])
    def test_cell_is_periodic(self, kind):
        atoms = build_schwarzite(kind, cell=36.0)
        assert all(atoms.get_pbc())
        length = float(atoms.cell[0][0])
        assert length > 0
        assert np.allclose(np.array(atoms.cell), np.eye(3) * length)
        # Atoms live inside the cell, so the network tiles by translation.
        positions = atoms.get_positions()
        assert positions.min() >= -1e-6
        assert positions.max() <= length + 1e-6

    def test_negative_curvature_favours_heptagons(self):
        counts = build_schwarzite("primitive", cell=36.0).info["ring_counts"]
        # Opposite of a fullerene: saddles everywhere, so rings larger than
        # six must outnumber the pentagons.
        assert counts.get(7, 0) + counts.get(8, 0) > counts.get(5, 0)

    def test_geometry_is_physical_across_the_seam(self):
        atoms = build_schwarzite("primitive", cell=36.0)
        g = atoms.info["geometry"]
        # Measured minimum-image: without that a bond wrapping the cell would
        # read as a ~26 Å "bond" and this would fail loudly.
        assert g["n_close_contacts"] == 0
        assert 1.20 < g["bond_min"] <= g["bond_max"] < 1.70
        assert g["bond_mean"] == pytest.approx(1.42, abs=0.02)

    def test_cell_too_small_for_the_surface_is_rejected(self):
        # Schwarz D packs nine handles into the cell, so its necks pinch
        # first; the builder should say so rather than emit a torn network.
        with pytest.raises(ValueError, match="too small"):
            build_schwarzite("diamond", cell=30.0)

    def test_nonpositive_cell_rejected(self):
        with pytest.raises(ValueError):
            build_schwarzite("primitive", cell=0.0)

    def test_annealing_is_off_by_default_because_it_hurts_here(self):
        """The 5-7 pairs are the mechanism, not the defect.

        On a junction, annealing away stray pentagon-heptagon pairs is an
        improvement. On a triply periodic *minimal* surface it is not: the
        surface saddles everywhere and those pairs are how a hexagonal net
        covers that Gaussian curvature. Remove them and the remaining
        lattice has to stretch instead. Measured across every surface and
        cell size tried, so the default must stay 0.
        """
        import inspect

        assert (
            inspect.signature(build_schwarzite).parameters["anneal_sweeps"].default
            == 0
        )

        as_grown = build_schwarzite("primitive", cell=30.0, anneal_sweeps=0)
        annealed = build_schwarzite("primitive", cell=30.0, anneal_sweeps=80)

        def strays(atoms):
            counts = atoms.info["ring_counts"]
            return min(counts.get(5, 0), counts.get(7, 0))

        # Annealing does what it says to the topology...
        assert strays(annealed) < strays(as_grown)
        # ...and the geometry pays for it.
        assert (annealed.info["geometry"]["bond_max"]
                > as_grown.info["geometry"]["bond_max"])

    def test_the_minimum_cell_is_where_geometry_stops_being_broken(self):
        """The floor is set by bond quality, not merely by "it built".

        Below it the cell still meets the tear gate and still relaxes to
        bonds outside the sp2 range, which is the failure the earlier,
        lower limits let through.
        """
        from nanocarbon_lab.builders.junction import MIN_SCHWARZITE_CELL
        from nanocarbon_lab.validation.quality import sp2_quality

        for kind in ("primitive", "gyroid"):
            atoms = build_schwarzite(kind, cell=MIN_SCHWARZITE_CELL[kind])
            verdict, why = sp2_quality(atoms.info["geometry"])
            assert verdict != "broken", f"{kind} at its minimum cell: {why}"
