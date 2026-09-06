"""Tests for the batch workflow module."""

from __future__ import annotations

import json

import pytest
from functools import partial

from carbonforge.builders import build_cnt, build_graphene_supercell
from carbonforge.dopants import dope_random
from carbonforge.workflows import BatchJob, write_dataset, batch_cnt_sweep


def test_single_job(tmp_path):
    jobs = [BatchJob(name="cnt_6_6", builder=partial(build_cnt, 6, 6, length=6),
                     export="qe")]
    meta = write_dataset(jobs, tmp_path)
    assert meta.exists()
    data = json.loads(meta.read_text())
    assert len(data) == 1
    assert data[0]["name"] == "cnt_6_6"
    assert data[0]["validation_ok"]


def test_sweep_generates_all_jobs(tmp_path):
    jobs = batch_cnt_sweep(
        chiralities=[(5, 5), (6, 0)],
        lengths=[6.0],
        dopant="N",
        dopant_concentrations=[0.0, 0.1],
        vacancies=[0, 1],
        seed=0,
        export="qe",
    )
    assert len(jobs) == 2 * 1 * 2 * 2


def test_job_with_doping(tmp_path):
    jobs = [
        BatchJob(
            name="gr_doped",
            builder=partial(build_graphene_supercell, 3, 3),
            post=[partial(dope_random, element="N", concentration=0.1, seed=0)],
            export="lammps",
        )
    ]
    meta_path = write_dataset(jobs, tmp_path)
    meta = json.loads(meta_path.read_text())
    assert "N" in meta[0]["formula"]


class TestGeneralSweep:
    """The sweep helper must work for any builder, not just nanotubes."""

    def test_ribbon_sweep_covers_the_grid(self):
        from carbonforge.builders import build_nanoribbon
        from carbonforge.workflows import batch_structure_sweep

        jobs = batch_structure_sweep(
            build_nanoribbon,
            {"width": [4, 6], "edge": ["zigzag", "armchair"], "length": [3]},
            name_prefix="gnr",
        )
        assert len(jobs) == 4
        assert all(job.name.startswith("gnr_") for job in jobs)

    def test_graphene_sweep_builds_valid_structures(self, tmp_path):
        from carbonforge.builders import build_graphene_supercell
        from carbonforge.workflows import batch_structure_sweep, write_dataset

        jobs = batch_structure_sweep(
            build_graphene_supercell, {"nx": [2, 3], "ny": [2]},
            name_prefix="gr",
        )
        meta = json.loads(write_dataset(jobs, tmp_path).read_text())
        assert len(meta) == 2
        assert all(entry["validation_ok"] for entry in meta)

    def test_post_factory_receives_params_and_seed(self):
        from carbonforge.builders import build_nanoribbon
        from carbonforge.workflows import batch_structure_sweep

        seen = []

        def factory(params, seed):
            seen.append((params, seed))
            return []

        batch_structure_sweep(
            build_nanoribbon, {"width": [4, 6], "length": [3]},
            post_factory=factory, seed=100,
        )
        assert len(seen) == 2
        # Each job gets its own seed, so the sweep stays reproducible.
        assert {s for _, s in seen} == {100, 101}

    def test_functionalised_sweep_end_to_end(self, tmp_path):
        from functools import partial

        from carbonforge.builders import build_nanoribbon
        from carbonforge.functionalization import functionalize_random
        from carbonforge.workflows import batch_structure_sweep, write_dataset

        jobs = batch_structure_sweep(
            build_nanoribbon,
            {"width": [6], "edge": ["zigzag", "armchair"], "length": [3]},
            name_prefix="gnr",
            post_factory=lambda params, seed: [
                partial(functionalize_random, group_key="NH2",
                        n_groups=2, seed=seed)
            ],
        )
        meta = json.loads(write_dataset(jobs, tmp_path).read_text())
        assert len(meta) == 2
        for entry in meta:
            assert "N" in entry["formula"]
            assert entry["validation_ok"]

    def test_empty_grid_rejected(self):
        from carbonforge.builders import build_graphene_supercell
        from carbonforge.workflows import batch_structure_sweep

        with pytest.raises(ValueError):
            batch_structure_sweep(build_graphene_supercell, {})
