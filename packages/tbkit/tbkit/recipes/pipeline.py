"""A recipe of your own: one structure, one model and a list of tasks run in order.

Made by the window's «Nueva receta…» (or by hand) as a JSON file::

    {
      "name": "coil_N_relajada",
      "structure": "coil_N.extxyz",              # relative to this file
      "model": "chn",                            # a window model name or a parameter file
      "steps": [
        {"task": "relax",   "settings": {"fmax": 0.02, "kmesh": 2}},
        {"task": "levels",  "settings": {"kmesh": 4}},
        {"task": "dos",     "settings": {"sigma": 0.05}},
        {"task": "phonons", "settings": {"kmesh": 4}}
      ]
    }

and run with::

    python -m tbkit.recipes.pipeline RECETA.json          # or from the Recetas tab

Tasks are those of ``tbkit run`` (:data:`tbkit.tasks.TASKS`: levels, dos, bands, hubbard,
relax, phonons, raman). Each step writes a reproducible record
(``NN_task.record.json``: structure, full parameters, settings, results, version) in the
folder ``RECETA/`` next to the file, and a step that changes the structure (relax) hands
its result to the next. Resumable: a step whose record exists is not run again, and its
structure is taken from that record. The recipe and its steps count in ``tbkit estado``.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path


def load(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("structure", "model", "steps"):
        if key not in data:
            raise ValueError(f"{path}: falta «{key}» en la receta.")
    from ..tasks import TASKS

    for k, step in enumerate(data["steps"]):
        if step.get("task") not in TASKS:
            raise ValueError(f"Paso {k + 1}: tarea {step.get('task')!r} desconocida "
                             f"(hay: {', '.join(TASKS)}).")
    return data


def workdir_of(path: Path) -> Path:
    return Path(path).with_suffix("")


def run(path: Path, first: int = 1, last: int | None = None) -> list[dict]:
    """Run the steps ``first``..``last`` (1-based) of the recipe file; returns a summary row
    per step."""
    from ase.io import read

    from ..gui.actions import load_model
    from ..progress import Progress
    from ..record import make_record, save_record
    from ..structure_io import read_structure
    from ..tasks import TASKS

    path = Path(path)
    data = load(path)
    base = path.parent
    model_spec = data["model"]
    model = load_model(str(base / model_spec) if str(model_spec).endswith(".json")
                       else model_spec)
    atoms = read_structure(base / data["structure"])
    folder = workdir_of(path)
    folder.mkdir(parents=True, exist_ok=True)
    steps = data["steps"]
    last = len(steps) if last is None else last
    rows = []
    bar = Progress(len(steps), data.get("name", path.stem), status=folder / "progreso.json")
    for k, step in enumerate(steps, start=1):
        record_path = folder / f"{k:02d}_{step['task']}.record.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            atoms = read(io.StringIO(record["structure"]), format="extxyz")
            rows.append({"step": k, "task": step["task"], "record": record_path.name,
                         "status": "hecho antes"})
            bar.step(note=f"{k}: {step['task']} (ya hecho)")
            continue
        if k < first or k > last:
            rows.append({"step": k, "task": step["task"], "status": "omitido"})
            continue
        settings = dict(step.get("settings", {}))
        print(f"paso {k}/{len(steps)}: {step['task']} {settings}", flush=True)
        results, atoms_after = TASKS[step["task"]](atoms, model, **settings)
        record = make_record(atoms_after, model, step["task"], settings, results)
        save_record(record_path, record)
        atoms = atoms_after
        atoms.write(folder / f"{k:02d}_{step['task']}.extxyz", format="extxyz")
        rows.append({"step": k, "task": step["task"], "record": record_path.name,
                     "status": "hecho"})
        bar.step(note=f"{k}: {step['task']}")
    bar.close()
    (folder / "resumen.json").write_text(json.dumps({"recipe": str(path), "steps": rows},
                                                    indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    return rows


def write(path: Path, name: str, atoms, model: str, steps: list[dict]) -> Path:
    """Write a recipe file and its structure next to it (``NAME.extxyz``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    structure = path.with_name(f"{path.stem}_estructura.extxyz")
    atoms.write(structure, format="extxyz")
    data = {"name": name, "structure": structure.name, "model": model, "steps": steps}
    load_check = json.dumps(data, indent=1, ensure_ascii=False)
    path.write_text(load_check, encoding="utf-8")
    load(path)                                  # refuse a recipe that would not run
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("receta", type=Path,
                        help="archivo JSON de la receta (lo escribe «Nueva receta…»)")
    parser.add_argument("--desde", type=int, default=1,
                        help="primer paso a correr (1 = el primero)")
    parser.add_argument("--hasta", type=int, default=None,
                        help="último paso a correr (vacío: hasta el final)")
    args = parser.parse_args(argv)
    for row in run(args.receta, args.desde, args.hasta):
        print(f"  {row['step']:2d} {row['task']:8s} {row['status']}"
              + (f" → {row['record']}" if row.get("record") else ""))


if __name__ == "__main__":
    main()
