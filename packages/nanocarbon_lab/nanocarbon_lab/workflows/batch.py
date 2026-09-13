"""Batch generation of nanocarbon datasets.

A :class:`BatchJob` describes how to build a single structure (builder +
dopants + defects + validation). :func:`write_dataset` iterates over a list
of jobs, exports them to QE and/or LAMMPS, and writes a JSON metadata
file describing the whole dataset.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ase import Atoms

from ..builders import build_cnt
from ..defects import introduce_vacancies
from ..dopants import dope_random
from ..exports.lammps import write_lammps
from ..exports.qe import write_qe_input
from ..validation.checks import run_basic_checks

ExportFormat = Literal["qe", "lammps", "both"]


@dataclass
class BatchJob:
    """One element of a batch run.

    The ``builder`` is a zero-argument callable returning an :class:`ase.Atoms`
    (typically a ``functools.partial`` over one of the builders). Post-processing
    hooks are applied in order.
    """

    name: str
    builder: Callable[[], Atoms]
    post: list[Callable[[Atoms], Atoms]] = field(default_factory=list)
    export: ExportFormat = "qe"
    qe_calculation: str = "scf"
    force_export: bool = False


def _build_and_process(job: BatchJob) -> Atoms:
    atoms = job.builder()
    for step in job.post:
        atoms = step(atoms)
    return atoms


def write_dataset(
    jobs: Iterable[BatchJob],
    root: str | Path,
    metadata_filename: str = "dataset.json",
) -> Path:
    """Run every job in ``jobs``, export it, and write aggregate metadata.

    Parameters
    ----------
    jobs
        Iterable of :class:`BatchJob`.
    root
        Root output directory; one subfolder per job is created.
    metadata_filename
        Name of the aggregate JSON file (in ``root``).

    Returns
    -------
    pathlib.Path
        Path to the metadata file.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for job in jobs:
        job_dir = root / job.name
        job_dir.mkdir(parents=True, exist_ok=True)
        atoms = _build_and_process(job)
        report = run_basic_checks(atoms)
        entry: dict[str, object] = {
            "name": job.name,
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
            "pbc": list(map(bool, atoms.get_pbc())),
            "cell": atoms.cell.tolist(),
            "validation_ok": report.ok,
            "validation_errors": report.errors,
            "validation_warnings": report.warnings,
            "validation_info": report.info,
            "atoms_info": _serialise_info(atoms.info),
            "exports": [],
        }
        if job.export in ("qe", "both"):
            qe_path = write_qe_input(
                atoms, job_dir / "qe", force=job.force_export
            )
            entry["exports"].append({"format": "qe", "path": str(qe_path)})
        if job.export in ("lammps", "both"):
            data, inp = write_lammps(
                atoms, job_dir / "lammps", force=job.force_export
            )
            entry["exports"].append(
                {"format": "lammps", "data": str(data), "input": str(inp)}
            )
        manifest.append(entry)

    meta_path = root / metadata_filename
    meta_path.write_text(json.dumps(manifest, indent=2, default=str))
    return meta_path


def _serialise_info(info: dict) -> dict:
    """Return a JSON-safe copy of ``atoms.info`` (drops numpy etc.)."""
    out: dict = {}
    for k, v in info.items():
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


def batch_cnt_sweep(
    chiralities: Iterable[tuple[int, int]],
    lengths: Iterable[float],
    dopant: str | None = None,
    dopant_concentrations: Iterable[float] = (0.0,),
    vacancies: Iterable[int] = (0,),
    seed: int = 0,
    export: ExportFormat = "qe",
) -> list[BatchJob]:
    """Cartesian-product sweep of CNT geometries, dopants and vacancies.

    Parameters
    ----------
    chiralities
        Iterable of ``(n, m)`` tuples.
    lengths
        Iterable of CNT lengths in Å.
    dopant
        If given, random doping of the listed concentrations is applied.
    dopant_concentrations
        Iterable of dopant fractions (only used if ``dopant`` is not None).
    vacancies
        Iterable of vacancy counts (0 means pristine).
    seed
        Base seed; each job gets a unique derived seed for reproducibility.
    export
        Target export format for all jobs.

    Returns
    -------
    list[BatchJob]
        Jobs suitable for :func:`write_dataset`.
    """
    from functools import partial

    jobs: list[BatchJob] = []
    job_idx = 0
    concentrations = list(dopant_concentrations) if dopant else [0.0]

    for n, m in chiralities:
        for length in lengths:
            for conc in concentrations:
                for n_vac in vacancies:
                    job_seed = seed + job_idx
                    name = f"cnt_{n}_{m}_L{length:g}_dop{conc:g}_vac{n_vac}"
                    post: list[Callable[[Atoms], Atoms]] = []
                    if dopant and conc > 0:
                        post.append(partial(dope_random, element=dopant,
                                            concentration=conc, seed=job_seed))
                    if n_vac > 0:
                        post.append(partial(introduce_vacancies,
                                            n_defects=n_vac, seed=job_seed))
                    jobs.append(
                        BatchJob(
                            name=name,
                            builder=partial(build_cnt, n=n, m=m, length=length),
                            post=post,
                            export=export,
                        )
                    )
                    job_idx += 1
    return jobs
