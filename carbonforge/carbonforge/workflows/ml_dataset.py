"""ML dataset generation.

Given a set of :class:`~carbonforge.workflows.batch.BatchJob`, produce a
flat, ML-friendly dataset:

* one ``.xyz`` file per structure (extended XYZ with ``atoms.info`` comments),
* a ``features.csv`` table with one row per structure and columns describing
  composition, topology and geometry,
* a ``manifest.json`` mirroring :func:`write_dataset`.

The feature extractor is minimal but extensible — the point is to provide a
baseline that produces usable inputs for e.g. GPR / MLP surrogates, ready to
augment with SOAP / ACE / M3GNet descriptors by the caller.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from ase import Atoms
from ase.io import write as ase_write

from ..topology.graph import (
    coordination_numbers,
    connected_components,
    ring_statistics,
)
from ..validation.checks import run_basic_checks
from .batch import BatchJob, _build_and_process, _serialise_info


def compute_features(atoms: Atoms) -> dict[str, float | int | str]:
    """Extract a structural fingerprint from an :class:`ase.Atoms`.

    Returns a flat dictionary suitable for a CSV row.
    """
    symbols = atoms.get_chemical_symbols()
    comp = Counter(symbols)
    n = len(atoms)
    coord = coordination_numbers(atoms)
    rings = ring_statistics(atoms, max_ring=8)
    comps = connected_components(atoms)

    volume = float(abs(np.linalg.det(atoms.cell))) if all(atoms.cell.any(axis=1)) else 0.0
    density = (
        float(sum(atoms.get_masses())) / volume * 1.66054 if volume > 0 else float("nan")
    )

    out: dict[str, float | int | str] = {
        "n_atoms": n,
        "formula": atoms.get_chemical_formula(),
        "structure_type": str(atoms.info.get("structure_type", "unknown")),
        "dimensionality": int(sum(atoms.get_pbc())),
        "volume_A3": volume,
        "density_g_cm3": density,
        "n_components": len(comps),
        "largest_component_size": len(comps[0]) if comps else 0,
        "mean_coordination": float(coord.mean()) if n else 0.0,
        "frac_coord_3": float(np.mean(coord == 3)) if n else 0.0,
        "frac_coord_2": float(np.mean(coord == 2)) if n else 0.0,
    }
    for element in ("C", "N", "B", "S", "P", "H"):
        out[f"count_{element}"] = int(comp.get(element, 0))
        out[f"frac_{element}"] = float(comp.get(element, 0)) / n if n else 0.0
    for size, count in rings.items():
        out[f"rings_{size}"] = int(count)
    return out


def write_ml_dataset(
    jobs: Iterable[BatchJob],
    root: str | Path,
    feature_extractor: Callable[[Atoms], dict] | None = None,
    write_xyz: bool = True,
) -> Path:
    """Build every job, dump XYZ files and a features CSV + manifest.

    Parameters
    ----------
    jobs
        Iterable of :class:`BatchJob`.
    root
        Output directory (created if missing). Per-structure XYZ files go
        into ``root/structures/`` and the summary files into ``root``.
    feature_extractor
        Callable ``Atoms -> dict`` overriding :func:`compute_features`.
    write_xyz
        If ``True`` (default), dump each structure as extended XYZ.

    Returns
    -------
    pathlib.Path
        Path to the written ``manifest.json``.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    xyz_dir = root / "structures"
    if write_xyz:
        xyz_dir.mkdir(parents=True, exist_ok=True)

    extractor = feature_extractor or compute_features
    manifest: list[dict] = []
    feature_rows: list[dict] = []

    for job in jobs:
        atoms = _build_and_process(job)
        report = run_basic_checks(atoms)

        entry: dict[str, object] = {
            "name": job.name,
            "validation_ok": report.ok,
            "validation_errors": report.errors,
            "validation_warnings": report.warnings,
            "atoms_info": _serialise_info(atoms.info),
        }

        features = {"name": job.name, **extractor(atoms)}
        feature_rows.append(features)

        if write_xyz:
            xyz_path = xyz_dir / f"{job.name}.xyz"
            ase_write(xyz_path, atoms, format="extxyz")
            entry["xyz"] = str(xyz_path)

        manifest.append(entry)

    # Write features.csv with the union of all keys.
    if feature_rows:
        fields: list[str] = []
        seen: set[str] = set()
        for row in feature_rows:
            for k in row:
                if k not in seen:
                    fields.append(k)
                    seen.add(k)
        csv_path = root / "features.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in feature_rows:
                w.writerow({k: row.get(k, "") for k in fields})

    meta_path = root / "manifest.json"
    meta_path.write_text(json.dumps(manifest, indent=2, default=str))
    return meta_path
