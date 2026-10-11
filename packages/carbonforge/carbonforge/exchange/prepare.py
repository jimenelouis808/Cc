"""Make a ``nanocarbon-dft/1`` request folder runnable, and say how far it got.

``prepare`` checks the request (schema, structure, tasks, what the engine needs), copies the
driver as ``run_dft.py`` and, if asked, writes a SLURM script. The folder is then a job of
:class:`carbonforge.jobs.queue.JobQueue` (``ExchangeAdapter``) or runs by hand anywhere the
engine is installed.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

SCHEMA = "nanocarbon-dft/1"
CODES = ("gpaw", "qe", "dftb+", "emt")
KINDS = ("punto_simple", "relajar", "desplazamientos", "hessiana", "dos")
DRIVER = Path(__file__).with_name("driver.py")


def read_request(folder: str | Path) -> dict:
    """The request of ``folder``, checked; ``ValueError`` says what is wrong."""
    folder = Path(folder)
    path = folder / "solicitud.json"
    if not path.exists():
        raise ValueError(f"{folder}: falta solicitud.json")
    request = json.loads(path.read_text(encoding="utf-8"))
    if request.get("esquema") != SCHEMA:
        raise ValueError(f"esquema {request.get('esquema')!r}; carbonforge lee {SCHEMA!r}")
    if request.get("codigo") not in CODES:
        raise ValueError(f"código {request.get('codigo')!r}: {', '.join(CODES)}")
    if not (folder / request.get("estructura", "")).exists():
        raise ValueError(f"falta la estructura {request.get('estructura')!r}")
    ids = set()
    for task in request.get("tareas", []):
        if task.get("tipo") not in KINDS:
            raise ValueError(f"tarea {task.get('id')!r}: tipo {task.get('tipo')!r} "
                             f"({', '.join(KINDS)})")
        if not task.get("id") or task["id"] in ids:
            raise ValueError(f"cada tarea necesita un id único (repetido o vacío: {task.get('id')!r})")
        source = task.get("sobre", "entrada")
        if source not in (None, "entrada") and source not in ids:
            raise ValueError(f"tarea {task['id']}: «sobre» {source!r} debe ser una tarea anterior")
        if task["tipo"] == "desplazamientos" and not (folder / task.get("vectores", "")).exists():
            raise ValueError(f"tarea {task['id']}: falta {task.get('vectores')!r}")
        ids.add(task["id"])
    if not ids:
        raise ValueError("la solicitud no tiene tareas")
    settings = request.get("ajustes", {})
    if request["codigo"] == "qe" and not settings.get("pseudos"):
        raise ValueError("Quantum ESPRESSO necesita ajustes.pseudos ({elemento: archivo .UPF})")
    if request["codigo"] == "dftb+" and not settings.get("skf_dir"):
        raise ValueError("DFTB+ necesita ajustes.skf_dir (carpeta con los A-B.skf)")
    return request


def slurm_script(request: dict, partition: str | None = None, hours: int = 24) -> str:
    """A minimal SLURM script: one node, the request's MPI processes."""
    resources = request.get("recursos", {})
    mpi = int(resources.get("mpi", 1))
    memory = resources.get("memoria_gb")
    launcher = "gpaw python" if request["codigo"] == "gpaw" else "python"
    lines = ["#!/bin/bash", f"#SBATCH --job-name={request.get('nombre', 'dft')[:30]}",
             "#SBATCH --nodes=1", f"#SBATCH --ntasks={mpi}", "#SBATCH --cpus-per-task=1",
             f"#SBATCH --time={hours}:00:00"]
    if memory:
        lines.append(f"#SBATCH --mem={int(memory)}G")
    if partition:
        lines.append(f"#SBATCH --partition={partition}")
    lines += ["", "export OMP_NUM_THREADS=1", "cd \"$SLURM_SUBMIT_DIR\"",
              "# Relanzar con el mismo script retoma donde quedó (tareas/).",
              f"srun -n {mpi} {launcher} run_dft.py" if mpi > 1 else f"{launcher} run_dft.py"]
    return "\n".join(lines) + "\n"


def prepare(folder: str | Path, slurm: bool = False, partition: str | None = None,
            hours: int = 24) -> Path:
    """Check the request and make the folder runnable; returns the ``run_dft.py`` path."""
    folder = Path(folder)
    request = read_request(folder)
    target = folder / "run_dft.py"
    shutil.copyfile(DRIVER, target)
    if slurm:
        (folder / "trabajo.slurm").write_text(slurm_script(request, partition, hours),
                                              encoding="utf-8")
    return target


def status(folder: str | Path) -> dict:
    """``{"state", "done", "total", "line", "tasks_done"}`` from the folder's files."""
    folder = Path(folder)
    out = {"state": "sin preparar", "done": 0, "total": 0, "line": "", "tasks_done": []}
    if (folder / "run_dft.py").exists():
        out["state"] = "preparado"
    result = folder / "resultado.json"
    if result.exists():
        data = json.loads(result.read_text(encoding="utf-8"))
        out["tasks_done"] = list(data.get("tareas", {}))
        out["state"] = "terminado" if data.get("terminado") else "en curso"
    progress = folder / "progreso.json"
    if progress.exists() and out["state"] != "terminado":
        p = json.loads(progress.read_text(encoding="utf-8"))
        out.update(done=p.get("done", 0), total=p.get("total", 0),
                   line=f"{p.get('label', '')}: {p.get('done', 0)}/{p.get('total', 0)}"
                        + (f" · {p['note']}" if p.get("note") else ""))
    return out
