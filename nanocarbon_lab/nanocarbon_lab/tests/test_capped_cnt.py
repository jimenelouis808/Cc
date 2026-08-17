"""Tests for the capped/defected CNT ("elongated fullerene") builder."""

from __future__ import annotations

import numpy as np
import pytest

from nanocarbon_lab.builders import build_capped_cnt
from nanocarbon_lab.builders import fullerene_mesh as fm
from nanocarbon_lab.exports.xyz import write_render_bundle
from nanocarbon_lab.utils.constants import HARD_MIN_DISTANCE


def _ring_deficit(ring_counts: dict[int, int]) -> int:
    """sum((6 - size) * count) -- must equal 12 for any closed shell."""
    return sum((6 - size) * count for size, count in ring_counts.items())


class TestFullereneMeshPrimitives:
    def test_seed_mesh_euler(self):
        verts, faces = fm.seed_capsule_mesh(6)
        edges = {
            tuple(sorted((int(a), int(b))))
            for f in faces
            for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))
        }
        assert len(verts) - len(edges) + len(faces) == 2

    def test_seed_mesh_degree_distribution(self):
        # Exactly 12 degree-5 vertices (2 poles + their adjacent rings),
        # regardless of n_rings -- this is what guarantees 6 pentagons/cap.
        for n_rings in (2, 3, 5, 9):
            _, faces = fm.seed_capsule_mesh(n_rings)
            nbrs = fm.mesh_adjacency(faces)
            degrees = [len(v) for v in nbrs.values()]
            assert degrees.count(5) == 12
            assert all(d in (5, 6) for d in degrees)

    def test_seed_mesh_rejects_too_few_rings(self):
        with pytest.raises(ValueError):
            fm.seed_capsule_mesh(1)

    def test_dual_of_plain_capsule_is_12_pentagons_rest_hexagons(self):
        mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(5), 2)
        _, _, rings = fm.dual_honeycomb(mesh)
        counts = fm.ring_size_histogram(rings)
        assert counts.get(5) == 12
        assert set(counts) <= {5, 6}
        assert _ring_deficit(counts) == 12

    def test_every_atom_is_3_coordinate(self):
        mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(6), 3)
        positions, bonds, _ = fm.dual_honeycomb(mesh)
        degree = np.zeros(len(positions), dtype=int)
        for a, b in bonds:
            degree[a] += 1
            degree[b] += 1
        assert np.all(degree == 3)

    def test_edge_flip_gives_stone_wales_5775(self):
        mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(6), 3)
        rng = np.random.default_rng(0)
        edge = fm.pick_interior_edge(mesh[1], rng)
        assert edge is not None
        before = fm.ring_size_histogram(fm.dual_honeycomb(mesh)[2])
        after = fm.ring_size_histogram(fm.dual_honeycomb(fm.edge_flip(mesh, *edge))[2])
        assert after.get(5, 0) - before.get(5, 0) == 2
        assert after.get(7, 0) - before.get(7, 0) == 2
        assert after.get(6, 0) - before.get(6, 0) == -4
        assert _ring_deficit(after) == 12

    def test_edge_flip_rejects_non_interior_edge(self):
        mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(6), 3)
        verts, faces = mesh
        nbrs = fm.mesh_adjacency(faces)
        # The two poles (index 0 and the last vertex) sit at opposite ends
        # of the mesh and are never adjacent for a body of this length.
        u, v = 0, len(verts) - 1
        assert v not in nbrs[u]
        with pytest.raises(ValueError):
            fm.edge_flip(mesh, u, v)

    def test_contract_edge_gives_divacancy_585(self):
        mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(6), 3)
        rng = np.random.default_rng(0)
        edge = fm.pick_interior_edge(mesh[1], rng)
        assert edge is not None
        before = fm.ring_size_histogram(fm.dual_honeycomb(mesh)[2])
        new_mesh, remap = fm.contract_edge(mesh, *edge)
        after = fm.ring_size_histogram(fm.dual_honeycomb(new_mesh)[2])
        assert after.get(5, 0) - before.get(5, 0) == 2
        assert after.get(8, 0) - before.get(8, 0) == 1
        assert _ring_deficit(after) == 12
        # contraction removes exactly one mesh vertex (one atom-triangle).
        assert edge[1] not in remap

    def test_capsule_project_shape(self):
        mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(8), 3)
        positions, _, _ = fm.dual_honeycomb(mesh)
        proj = fm.capsule_project(positions, radius=5.0, half_length=3.0)
        z = proj[:, 2]
        rho = np.linalg.norm(proj[:, :2], axis=1)
        body = np.abs(z) <= 3.0
        assert np.allclose(rho[body], 5.0, atol=1e-6)
        assert np.all(rho[~body] <= 5.0 + 1e-6)

    def test_relax_shell_preserves_topology_and_converges(self):
        mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(6), 3)
        positions, bonds, _ = fm.dual_honeycomb(mesh)
        proj = fm.capsule_project(positions, radius=5.0, half_length=2.0)
        bl0 = np.array([np.linalg.norm(proj[a] - proj[b]) for a, b in bonds])
        proj *= 1.42 / bl0.mean()
        relaxed = fm.relax_shell(proj, bonds, steps=500)
        bl = np.array([np.linalg.norm(relaxed[a] - relaxed[b]) for a, b in bonds])
        assert bl.mean() == pytest.approx(1.42, abs=0.05)
        assert bl.min() > 1.0
        assert bl.max() < 1.9


class TestBuildCappedCNT:
    def test_plain_cap_is_topologically_perfect_fullerene(self):
        atoms = build_capped_cnt(n_body_rings=6, freq=2, radius=5.0, seed=0)
        counts = atoms.info["ring_counts"]
        assert set(counts) == {5, 6}  # only pentagons (caps) and hexagons
        assert counts[5] == 12
        assert _ring_deficit(counts) == 12

    def test_all_carbon_and_finite(self):
        atoms = build_capped_cnt(n_body_rings=6, freq=2, radius=5.0, seed=0)
        assert set(atoms.get_chemical_symbols()) == {"C"}
        assert not any(atoms.get_pbc())

    def test_bond_lengths_are_physical(self):
        atoms = build_capped_cnt(n_body_rings=6, freq=2, radius=5.0, seed=0)
        bl = atoms.info["bond_length"]
        assert 1.2 < bl["min"] <= bl["mean"] <= bl["max"] < 1.8
        # No atom pair anywhere in the structure should be closer than the
        # hard overlap threshold (real bonded pairs are ~1.4, so this also
        # implicitly checks there is no self-intersection of the shell).
        positions = atoms.get_positions()
        from scipy.spatial import cKDTree

        tree = cKDTree(positions)
        d, _ = tree.query(positions, k=2)
        assert d[:, 1].min() > HARD_MIN_DISTANCE

    def test_larger_freq_gives_more_atoms(self):
        small = build_capped_cnt(n_body_rings=6, freq=2, radius=5.0, seed=0)
        big = build_capped_cnt(n_body_rings=6, freq=4, radius=5.0, seed=0)
        assert len(big) > len(small)

    def test_stone_wales_defect_ring_delta(self):
        base = build_capped_cnt(n_body_rings=10, freq=3, radius=6.0, seed=0)
        defected = build_capped_cnt(
            n_body_rings=10, freq=3, radius=6.0,
            defects=[{"type": "stone_wales", "count": 2}], seed=0,
        )
        base_counts, def_counts = base.info["ring_counts"], defected.info["ring_counts"]
        assert def_counts.get(5, 0) - base_counts.get(5, 0) == 4
        assert def_counts.get(7, 0) - base_counts.get(7, 0) == 4
        assert _ring_deficit(def_counts) == 12

    def test_divacancy_defect_ring_delta(self):
        base = build_capped_cnt(n_body_rings=10, freq=3, radius=6.0, seed=0)
        defected = build_capped_cnt(
            n_body_rings=10, freq=3, radius=6.0,
            defects=[{"type": "divacancy", "count": 2}], seed=0,
        )
        base_counts, def_counts = base.info["ring_counts"], defected.info["ring_counts"]
        assert def_counts.get(5, 0) - base_counts.get(5, 0) == 4
        assert def_counts.get(8, 0) - base_counts.get(8, 0) == 2
        assert _ring_deficit(def_counts) == 12

    def test_mixed_defects_compose(self):
        atoms = build_capped_cnt(
            n_body_rings=12, freq=3, radius=6.5,
            defects=[
                {"type": "stone_wales", "count": 1},
                {"type": "divacancy", "count": 1},
            ],
            seed=3,
        )
        counts = atoms.info["ring_counts"]
        assert counts.get(7, 0) == 2
        assert counts.get(8, 0) == 1
        assert counts.get(5, 0) == 12 + 2 + 2
        assert _ring_deficit(counts) == 12
        assert len(atoms.info["defect_log"]) == 2

    def test_bend_angle_produces_finite_non_nan_structure(self):
        atoms = build_capped_cnt(
            n_body_rings=10, freq=3, radius=6.0, bend_angle=0.6, seed=0
        )
        positions = atoms.get_positions()
        assert np.isfinite(positions).all()
        assert atoms.info["ring_counts"][5] == 12

    def test_reproducible_with_seed(self):
        a = build_capped_cnt(
            n_body_rings=10, freq=3, radius=6.0,
            defects=[{"type": "stone_wales", "count": 2}], seed=11,
        )
        b = build_capped_cnt(
            n_body_rings=10, freq=3, radius=6.0,
            defects=[{"type": "stone_wales", "count": 2}], seed=11,
        )
        assert a.info["defect_log"] == b.info["defect_log"]
        assert np.allclose(a.get_positions(), b.get_positions())

    def test_invalid_parameters_raise(self):
        with pytest.raises(ValueError):
            build_capped_cnt(n_body_rings=1)
        with pytest.raises(ValueError):
            build_capped_cnt(freq=0)
        with pytest.raises(ValueError):
            build_capped_cnt(radius=0)

    def test_unknown_defect_type_raises(self):
        with pytest.raises(ValueError):
            build_capped_cnt(defects=[{"type": "not_a_real_defect"}])

    def test_render_bundle_round_trip(self, tmp_path):
        atoms = build_capped_cnt(
            n_body_rings=8, freq=2, radius=5.0,
            defects=[{"type": "divacancy", "count": 1}], seed=1,
        )
        xyz_path, json_path = write_render_bundle(atoms, tmp_path / "demo")
        assert xyz_path.exists() and json_path.exists()

        xyz_lines = xyz_path.read_text().splitlines()
        assert int(xyz_lines[0]) == len(atoms)
        assert len(xyz_lines) == len(atoms) + 2

        import json

        bundle = json.loads(json_path.read_text())
        assert bundle["n_atoms"] == len(atoms)
        assert len(bundle["bonds"]) == len(atoms.info["bonds"])
        assert len(bundle["ring_sizes_per_atom"]) == len(atoms)
        # Every atom on a closed 3-coordinate honeycomb sits on exactly 3 rings.
        assert all(len(sizes) == 3 for sizes in bundle["ring_sizes_per_atom"])
