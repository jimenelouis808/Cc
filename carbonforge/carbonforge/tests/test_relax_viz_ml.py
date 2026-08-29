"""Tests for the bonus modules: relax, viz, ML dataset."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from carbonforge.builders import (
    build_graphene_supercell,
    build_cnt,
    build_carbon_foam,
)
from carbonforge.relax import harmonic_pre_relax, relax_with_calculator
from carbonforge.workflows import (
    BatchJob,
    compute_features,
    write_ml_dataset,
)


class TestHarmonicRelax:
    def test_does_not_change_already_relaxed(self):
        gr = build_graphene_supercell(3, 3)
        initial = gr.get_positions().copy()
        harmonic_pre_relax(gr, steps=50)
        # Relaxation of an already good structure must not blow up atoms.
        delta = np.linalg.norm(gr.get_positions() - initial, axis=1).max()
        assert delta < 0.1

    def test_improves_noisy_structure(self):
        gr = build_graphene_supercell(3, 3)
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.05, size=gr.get_positions().shape)
        gr.set_positions(gr.get_positions() + noise)
        from carbonforge.utils.geometry import guess_bonds
        before = np.array([d for _, _, d in guess_bonds(gr)])
        harmonic_pre_relax(gr, steps=500)
        after = np.array([d for _, _, d in guess_bonds(gr)])
        # Bond-length dispersion should shrink toward the equilibrium value.
        assert after.std() < before.std()

    def test_records_metadata(self):
        gr = build_graphene_supercell(2, 2)
        harmonic_pre_relax(gr, steps=10)
        assert "harmonic_relax" in gr.info

    def test_relax_with_calculator_rejects_unknown_algo(self):
        gr = build_graphene_supercell(2, 2)

        class _DummyCalc:
            def calculate(self, *args, **kwargs):
                pass

        with pytest.raises(ValueError):
            relax_with_calculator(gr, _DummyCalc(), algorithm="unknown")


class TestVizSmoke:
    def test_save_png(self, tmp_path):
        pytest.importorskip("matplotlib")
        from carbonforge.viz import save_structure_png

        atoms = build_cnt(5, 5, length=6)
        path = save_structure_png(atoms, tmp_path / "cnt.png")
        assert path.exists() and path.stat().st_size > 0


class TestMLDataset:
    def test_compute_features_basic(self):
        gr = build_graphene_supercell(2, 2)
        feats = compute_features(gr)
        assert feats["n_atoms"] == len(gr)
        assert feats["count_C"] == len(gr)
        assert feats["rings_6"] > 0

    def test_write_ml_dataset(self, tmp_path):
        jobs = [
            BatchJob(name="cnt_5_5",
                     builder=partial(build_cnt, 5, 5, length=6)),
            BatchJob(name="gr_3x3",
                     builder=partial(build_graphene_supercell, 3, 3)),
        ]
        meta = write_ml_dataset(jobs, tmp_path)
        assert meta.exists()
        csv_path = tmp_path / "features.csv"
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "n_atoms" in content
        assert "cnt_5_5" in content
        assert (tmp_path / "structures" / "cnt_5_5.xyz").exists()
