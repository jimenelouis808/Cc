"""Tests for the batch workflow module."""

from __future__ import annotations

import json
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
