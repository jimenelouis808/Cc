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

    def test_relax_shell_reaches_sp2_geometry(self):
        mesh = fm.smooth_mesh_on_capsule(
            fm.subdivide_mesh(fm.seed_capsule_mesh(6), 3),
            radius=1.0, half_length=2.5,
        )
        positions, bonds, _ = fm.dual_honeycomb(mesh)
        bl0 = np.array([np.linalg.norm(positions[a] - positions[b]) for a, b in bonds])
        positions = positions * (1.42 / bl0.mean())
        relaxed = fm.relax_shell(positions, bonds)
        bl = np.array([np.linalg.norm(relaxed[a] - relaxed[b]) for a, b in bonds])
        # A real valence force field converges tightly, not just "roughly".
        assert bl.mean() == pytest.approx(1.42, abs=0.01)
        assert bl.std() < 0.02
        assert bl.min() > 1.30 and bl.max() < 1.55

    def test_smoothing_preserves_topology(self):
        mesh = fm.subdivide_mesh(fm.seed_capsule_mesh(6), 3)
        before = fm.ring_size_histogram(fm.dual_honeycomb(mesh)[2])
        smoothed = fm.smooth_mesh_on_capsule(mesh, radius=1.0, half_length=2.5)
        after = fm.ring_size_histogram(fm.dual_honeycomb(smoothed)[2])
        assert before == after
        assert np.array_equal(mesh[1], smoothed[1])

    def test_radius_freq_round_trip(self):
        for freq in (2, 3, 4, 5, 6):
            radius = fm.radius_for_freq(freq)
            assert fm.freq_for_radius(radius) == freq
        # radius grows linearly with freq: ~1.96 A per unit at bond=1.42
        assert fm.radius_for_freq(4) == pytest.approx(4 * 1.9563, abs=0.04)

    def test_freq_for_radius_rejects_nonpositive(self):
        with pytest.raises(ValueError):
            fm.freq_for_radius(0.0)


class TestBuildCappedCNT:
    def test_plain_cap_is_topologically_perfect_fullerene(self):
        atoms = build_capped_cnt(n_body_rings=6, freq=2, seed=0)
        counts = atoms.info["ring_counts"]
        assert set(counts) == {5, 6}  # only pentagons (caps) and hexagons
        assert counts[5] == 12
        assert _ring_deficit(counts) == 12

    def test_all_carbon_and_finite(self):
        atoms = build_capped_cnt(n_body_rings=6, freq=2, seed=0)
        assert set(atoms.get_chemical_symbols()) == {"C"}
        assert not any(atoms.get_pbc())

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(n_body_rings=6, freq=2),
            dict(n_body_rings=10, freq=3),
            dict(n_body_rings=8, freq=4),
            dict(n_body_rings=10, freq=3, bend_angle=0.5),
            dict(
                n_body_rings=12, freq=3,
                defects=[
                    {"type": "stone_wales", "count": 2},
                    {"type": "divacancy", "count": 1},
                ],
            ),
        ],
    )
    def test_geometry_is_realistic_sp2(self, kwargs):
        """Structures must be physically realistic, not merely valid topology.

        Real graphitic carbon: 1.42 A bonds, 120 deg angles (108 deg inside
        a pentagon), and no non-bonded pair closer than the ~2.46 A 1-3
        distance. An earlier fixed-step relaxer satisfied the topology
        assertions above while producing 66-164 deg angles and dozens of
        sub-2 A contacts, so quality is asserted explicitly here.
        """
        atoms = build_capped_cnt(seed=0, **kwargs)
        g = atoms.info["geometry"]
        assert 1.30 < g["bond_min"] <= g["bond_max"] < 1.55
        assert g["bond_mean"] == pytest.approx(1.42, abs=0.02)
        assert g["bond_std"] < 0.03
        # 100 deg is comfortably below a pentagon's 108 deg interior angle.
        assert 100.0 < g["angle_min"] <= g["angle_max"] < 135.0
        assert g["angle_std"] < 4.0
        assert g["n_close_contacts"] == 0

        positions = atoms.get_positions()
        from scipy.spatial import cKDTree

        d, _ = cKDTree(positions).query(positions, k=2)
        assert d[:, 1].min() > HARD_MIN_DISTANCE

    def test_radius_matches_lattice_prediction(self):
        for freq in (2, 3, 4):
            atoms = build_capped_cnt(n_body_rings=8, freq=freq, seed=0)
            assert atoms.info["radius"] == pytest.approx(
                fm.radius_for_freq(freq), abs=0.15
            )

    def test_target_radius_selects_freq(self):
        atoms = build_capped_cnt(n_body_rings=8, target_radius=7.8, seed=0)
        assert atoms.info["freq"] == 4
        assert atoms.info["radius"] == pytest.approx(7.8, abs=0.5)

    def test_bend_is_retained(self):
        straight = build_capped_cnt(n_body_rings=10, freq=3, seed=0)
        bent = build_capped_cnt(n_body_rings=10, freq=3, bend_angle=0.6, seed=0)
        # A bent tube is more compact along its longest axis than a straight
        # one of the same atom count.
        assert len(bent) == len(straight)
        span = lambda a: (a.get_positions().max(axis=0)
                          - a.get_positions().min(axis=0)).max()
        assert span(bent) < span(straight)

    def test_larger_freq_gives_more_atoms(self):
        small = build_capped_cnt(n_body_rings=6, freq=2, seed=0)
        big = build_capped_cnt(n_body_rings=6, freq=4, seed=0)
        assert len(big) > len(small)

    def test_stone_wales_defect_ring_delta(self):
        base = build_capped_cnt(n_body_rings=10, freq=3, seed=0)
        defected = build_capped_cnt(
            n_body_rings=10, freq=3,
            defects=[{"type": "stone_wales", "count": 2}], seed=0,
        )
        base_counts, def_counts = base.info["ring_counts"], defected.info["ring_counts"]
        assert def_counts.get(5, 0) - base_counts.get(5, 0) == 4
        assert def_counts.get(7, 0) - base_counts.get(7, 0) == 4
        assert _ring_deficit(def_counts) == 12

    def test_divacancy_defect_ring_delta(self):
        base = build_capped_cnt(n_body_rings=10, freq=3, seed=0)
        defected = build_capped_cnt(
            n_body_rings=10, freq=3,
            defects=[{"type": "divacancy", "count": 2}], seed=0,
        )
        base_counts, def_counts = base.info["ring_counts"], defected.info["ring_counts"]
        assert def_counts.get(5, 0) - base_counts.get(5, 0) == 4
        assert def_counts.get(8, 0) - base_counts.get(8, 0) == 2
        assert _ring_deficit(def_counts) == 12

    def test_mixed_defects_compose(self):
        atoms = build_capped_cnt(
            n_body_rings=12, freq=3,
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
            n_body_rings=10, freq=3, bend_angle=0.6, seed=0
        )
        positions = atoms.get_positions()
        assert np.isfinite(positions).all()
        assert atoms.info["ring_counts"][5] == 12

    def test_reproducible_with_seed(self):
        a = build_capped_cnt(
            n_body_rings=10, freq=3,
            defects=[{"type": "stone_wales", "count": 2}], seed=11,
        )
        b = build_capped_cnt(
            n_body_rings=10, freq=3,
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
            build_capped_cnt(bond=0)
        with pytest.raises(ValueError):
            build_capped_cnt(bend_angle=2.0)  # beyond the elastic-bend model

    def test_unknown_defect_type_raises(self):
        with pytest.raises(ValueError):
            build_capped_cnt(defects=[{"type": "not_a_real_defect"}])

    def test_render_bundle_round_trip(self, tmp_path):
        atoms = build_capped_cnt(
            n_body_rings=8, freq=2,
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


class TestCentrelineShapes:
    """Sweeping the tube along a 3D path must not wreck its geometry."""

    def test_rotation_minimizing_frame_is_orthonormal_and_continuous(self):
        from nanocarbon_lab.builders import centerline as cl

        rng = np.random.default_rng(0)
        control = cl.random_control_points(12, 1.0, rng)
        sample, total = cl.arclength_sampler(control)
        positions, tangents = sample(np.linspace(0.0, total, 500))
        normals = cl.rotation_minimizing_frames(positions, tangents)

        assert np.abs(np.sum(tangents * normals, axis=1)).max() < 1e-10
        assert np.abs(np.linalg.norm(normals, axis=1) - 1.0).max() < 1e-10
        # A Frenet frame flips ~180 deg at each inflection point, which would
        # show up as a step of ~2 between consecutive unit normals.
        assert np.linalg.norm(np.diff(normals, axis=0), axis=1).max() < 0.5

    def test_strain_budget_flattens_an_over_curved_path(self):
        from nanocarbon_lab.builders import centerline as cl

        rng = np.random.default_rng(3)
        control = cl.random_control_points(16, 1.4, rng)
        fitted, strain = cl.fit_to_strain_budget(
            control, total_length=170.0, tube_radius=7.85, max_strain=0.08
        )
        assert strain <= 0.08 + 1e-6
        assert len(fitted) == len(control)

    def test_gentle_path_is_left_alone(self):
        from nanocarbon_lab.builders import centerline as cl

        control = cl.shape_control_points("arc", np.random.default_rng(0), amplitude=0.1)
        _, strain = cl.fit_to_strain_budget(
            control, total_length=200.0, tube_radius=4.0, max_strain=0.20
        )
        assert strain < 0.20

    def test_unknown_shape_raises(self):
        from nanocarbon_lab.builders import centerline as cl

        with pytest.raises(ValueError):
            cl.shape_control_points("pretzel", np.random.default_rng(0))

    @pytest.mark.parametrize("shape", ["arc", "s_curve", "helix", "random"])
    def test_swept_shapes_keep_sp2_geometry(self, shape):
        atoms = build_capped_cnt(
            n_body_rings=14, freq=3, shape=shape, waviness=1.0, seed=5,
        )
        g = atoms.info["geometry"]
        assert atoms.info["shape"] == shape
        assert atoms.info["path_strain"] <= 0.08 + 1e-6
        assert g["n_close_contacts"] == 0
        assert 1.30 < g["bond_min"] <= g["bond_max"] < 1.55
        assert 100.0 < g["angle_min"] <= g["angle_max"] < 135.0
        # Topology is untouched by bending.
        assert atoms.info["ring_counts"][5] == 12

    def test_shape_actually_bends_the_tube(self):
        straight = build_capped_cnt(n_body_rings=16, freq=2, shape="straight", seed=5)
        wavy = build_capped_cnt(
            n_body_rings=16, freq=2, shape="random", waviness=1.0, seed=5,
        )
        assert len(wavy) == len(straight)
        span = lambda a: (a.get_positions().max(axis=0)
                          - a.get_positions().min(axis=0)).max()
        # A meandering tube of the same atom count is more compact end to end.
        assert span(wavy) < span(straight)
        assert wavy.info["path_strain"] > 0.0

    def test_waviness_out_of_range_raises(self):
        with pytest.raises(ValueError):
            build_capped_cnt(n_body_rings=8, freq=2, shape="random", waviness=1.5)

    def test_excessive_strain_budget_warns(self):
        with pytest.warns(UserWarning, match="no longer physically"):
            build_capped_cnt(
                n_body_rings=10, freq=2, shape="random",
                waviness=1.0, max_strain=0.30, seed=1,
            )
