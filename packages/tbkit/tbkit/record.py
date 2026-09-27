"""Reproducible runs: one JSON file holds everything needed to redo a result.

A **record** stores the structure (as extxyz text), the full parameter set
(every law written out, with the sources of the file it came from), the
settings (k points, smearing, tolerances, charge, U...), the code version
and git commit, the date, and the results. A **simulation file** is the
input side: ``tbkit run simulation.json`` reads it, runs the task, and
writes the record next to it.

Simulation file::

    {
      "structure": "zgnr.extxyz",
      "model": "pi_huckel",                    # or "xu_carbon", {"file": ...},
                                               # {"skf": dir, "orbitals": {...}}
      "task": "hubbard",                       # levels, dos, bands, hubbard, relax, phonons
      "settings": {"U": 2.7, "kmesh": 48, "kT": 0.005},
      "output": "zgnr_hubbard.record.json"
    }
"""

from __future__ import annotations

import io
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from ase import Atoms
from ase.io import read, write

from . import __version__
from .params import TBModel, load_parameters, model_from_dict, model_to_dict, pi_model


def git_commit() -> Optional[str]:
    """The commit of the checkout tbkit runs from, if it is one."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def structure_text(atoms: Atoms) -> str:
    buffer = io.StringIO()
    write(buffer, atoms, format="extxyz")
    return buffer.getvalue()


def make_record(atoms: Atoms, model: TBModel, task: str, settings: dict,
                results: dict) -> dict:
    import ase
    import scipy

    return _jsonable({
        "tbkit": __version__,
        "commit": git_commit(),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "ase": ase.__version__},
        "task": task,
        "settings": settings,
        "model": model_to_dict(model),
        "structure": structure_text(atoms),
        "results": results,
    })


def save_record(path: str | Path, record: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def model_from_spec(spec, base: Path) -> TBModel:
    """A model from the ``model`` entry of a simulation file."""
    if isinstance(spec, str):
        return pi_model() if spec == "pi_huckel" else load_parameters(spec)
    if "file" in spec:
        return load_parameters(base / spec["file"])
    if "pi" in spec:
        return pi_model(**spec["pi"])
    if "skf" in spec:
        from .skf import load_skf_set

        return load_skf_set(base / spec["skf"],
                            {el: tuple(o) for el, o in spec["orbitals"].items()})
    if "parameters" in spec:                 # a model written by model_to_dict (a record)
        return model_from_dict(spec["parameters"])
    raise ValueError(f"Modelo no reconocido en el archivo de simulación: {spec!r}.")


def run_simulation(path: str | Path) -> dict:
    """Run a simulation file; returns (and saves) its record."""
    from . import tasks

    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    atoms = read(base / config["structure"])
    model = model_from_spec(config["model"], base)
    task = config["task"]
    settings = dict(config.get("settings", {}))
    if task not in tasks.TASKS:
        raise ValueError(f"Tarea desconocida {task!r}. Opciones: {', '.join(tasks.TASKS)}.")
    results, final_atoms = tasks.TASKS[task](atoms, model, **settings)
    record = make_record(final_atoms, model, task, settings, results)
    output = base / config.get("output", f"{path.stem}.record.json")
    save_record(output, record)
    record["_path"] = str(output)
    return record


def replay(record_path: str | Path) -> dict:
    """Run a record's task again from the record alone; returns the new results."""
    from . import tasks

    record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    atoms = read(io.StringIO(record["structure"]), format="extxyz")
    model = model_from_dict(record["model"])
    results, _ = tasks.TASKS[record["task"]](atoms, model, **record["settings"])
    return _jsonable(results)
