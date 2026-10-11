"""run_dft.py — runs a ``nanocarbon-dft/1`` request (docs/INTERCAMBIO_DFT.md of the workspace).

This file is copied into the request folder by ``carbonforge dft preparar`` and runs there::

    python run_dft.py                       # one process
    mpiexec -n 4 gpaw python run_dft.py     # GPAW in parallel (or: mpiexec -n 4 python run_dft.py)

It imports only ASE, numpy and the engine (GPAW, Quantum ESPRESSO through ASE, DFTB+ through
ASE), so it runs wherever the engine is installed, a cluster included, without carbonforge or
tbkit. Every calculation leaves its own file under ``tareas/`` and is skipped when it exists:
a killed run is relaunched with the same command and goes on. Units: eV, Å, e·Å.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA = "nanocarbon-dft/1"
HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- MPI-safe output

def _rank() -> int:
    try:
        from ase.parallel import world
        return world.rank
    except ImportError:
        return 0


def _barrier() -> None:
    try:
        from ase.parallel import world
    except ImportError:
        return
    world.barrier()


def _write_json(path: Path, data) -> None:
    """Atomic, and only on rank 0."""
    if _rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    _barrier()


def _log(text: str) -> None:
    if _rank() == 0:
        print(time.strftime("%H:%M:%S"), text, flush=True)


class Progress:
    """The same status file as ``tbkit.progress`` (``tbkit estado`` reads it)."""

    def __init__(self, total: int, label: str, done: int = 0):
        self.total, self.label, self.done = total, label, done
        self.started = time.time()
        self.last = self.started
        self.steps: list[float] = []
        self.note = ""
        self.write()

    def step(self, note: str = "") -> None:
        now = time.time()
        self.steps = (self.steps + [now - self.last])[-20:]
        self.last = now
        self.done += 1
        self.note = note
        self.write()

    def write(self) -> None:
        rate = sum(self.steps) / len(self.steps) if self.steps else None
        _write_json(HERE / "progreso.json", {
            "label": self.label, "done": self.done, "total": self.total,
            "started": self.started, "updated": time.time(), "pid": os.getpid(),
            "seconds_per_step": rate,
            "remaining_seconds": None if rate is None else rate * max(0, self.total - self.done),
            "note": self.note})


# --------------------------------------------------------------------------- engines

def calculator(code: str, settings: dict, label: str):
    """A fresh ASE calculator for ``code`` with the request's settings."""
    kpts = tuple(settings.get("kpts", (1, 1, 1)))
    width = float(settings.get("ensanchamiento_ev", 0.05))
    if code == "emt":                                   # tests only: no DFT
        from ase.calculators.emt import EMT
        return EMT()
    if code == "gpaw":
        from gpaw import GPAW, PW, FermiDirac
        mode = settings.get("modo", "lcao")
        mode = PW(float(settings.get("ecut_ev", 500.0))) if mode == "pw" else mode
        kwargs = {"mode": mode, "xc": settings.get("xc", "PBE"), "kpts": kpts,
                  "occupations": FermiDirac(width), "txt": str(HERE / "tareas" / f"{label}.txt"),
                  "spinpol": bool(settings.get("espin", False)),
                  "charge": float(settings.get("carga", 0.0)), "symmetry": "off"}
        if settings.get("modo", "lcao") == "lcao":
            kwargs["basis"] = settings.get("base", "dzp")
        if settings.get("h"):
            kwargs["h"] = float(settings["h"])
        return GPAW(**kwargs)
    if code == "qe":
        from ase.calculators.espresso import Espresso, EspressoProfile
        command = os.environ.get("ASE_ESPRESSO_COMMAND", "pw.x")
        profile = EspressoProfile(command=command,
                                  pseudo_dir=settings.get("pseudo_dir", os.environ.get(
                                      "ESPRESSO_PSEUDO", str(HERE))))
        input_data = {"control": {"tprnfor": True, "tstress": False},
                      "system": {"ecutwfc": float(settings.get("ecut_ry", 40.0)),
                                 "occupations": "smearing", "smearing": "fd",
                                 "degauss": width / 13.605693122994,
                                 "tot_charge": float(settings.get("carga", 0.0)),
                                 "nspin": 2 if settings.get("espin") else 1}}
        if settings.get("ecutrho_ry"):
            input_data["system"]["ecutrho"] = float(settings["ecutrho_ry"])
        return Espresso(profile=profile, pseudopotentials=settings["pseudos"], kpts=kpts,
                        input_data=input_data, directory=str(HERE / "tareas" / f"qe_{label}"))
    if code == "dftb+":
        # Filling: DFTB+'s default (the request's ensanchamiento_ev is not passed: the ASE
        # keyword syntax for it was not checked against a DFTB+ run). Unverified: say so.
        from ase.calculators.dftb import Dftb
        kwargs = {"Hamiltonian_SCC": "Yes" if settings.get("scc", True) else "No",
                  "Hamiltonian_MaxAngularMomentum_": ""}
        for element, shell in settings.get("momento_angular", {}).items():
            kwargs[f"Hamiltonian_MaxAngularMomentum_{element}"] = f'"{shell}"'
        return Dftb(kpts=kpts, slako_dir=settings.get("skf_dir"),
                    directory=str(HERE / "tareas" / f"dftb_{label}"), **kwargs)
    raise ValueError(f"código desconocido {code!r}: gpaw, qe, dftb+ (o emt para pruebas)")


def evaluate(atoms, code: str, settings: dict, label: str) -> dict:
    """Energy, forces and what else the engine gives, for one geometry."""
    atoms = atoms.copy()
    atoms.calc = calculator(code, settings, label)
    out = {"energia": float(atoms.get_potential_energy()),
           "fuerzas": atoms.get_forces().tolist()}
    try:
        out["dipolo"] = [float(x) for x in atoms.get_dipole_moment()]
    except Exception:  # noqa: BLE001  (not every engine has it)
        out["dipolo"] = None
    calc = atoms.calc
    try:
        out["fermi"] = float(calc.get_fermi_level())
        weights = np.asarray(calc.get_k_point_weights())
        spins = 2 if settings.get("espin") else 1
        out["autovalores"] = [[list(map(float, calc.get_eigenvalues(kpt=k, spin=s)))
                               for k in range(len(weights))] for s in range(spins)]
        out["pesos_k"] = weights.tolist()
    except Exception as error:  # noqa: BLE001  (EMT and some engines have no levels)
        out["autovalores_motivo"] = f"el motor no dio niveles: {type(error).__name__}"
    return out


# --------------------------------------------------------------------------- tasks

def _structure(request: dict, results: dict):
    from ase.io import read
    return read(HERE / request["estructura"])


def _start(task: dict, request: dict, results: dict):
    from ase.io import read
    source = task.get("sobre", "entrada")
    if source in (None, "entrada"):
        return _structure(request, results)
    if source not in results or "estructura" not in results[source]:
        raise ValueError(f"tarea {task['id']}: «sobre» {source!r} no es una relajación previa")
    return read(HERE / results[source]["estructura"])


def task_single(task, request, results):
    atoms = _start(task, request, results)
    path = HERE / "tareas" / task["id"] / "0.json"
    if not path.exists():
        _write_json(path, evaluate(atoms, request["codigo"], request["ajustes"], task["id"]))
    return json.loads(path.read_text())


def task_relax(task, request, results):
    from ase.io import Trajectory, read, write
    from ase.optimize import BFGS

    folder = HERE / "tareas" / task["id"]
    done = folder / "final.json"
    if done.exists():
        return json.loads(done.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    atoms = _start(task, request, results)
    trajectory = folder / "relax.traj"
    if trajectory.exists() and trajectory.stat().st_size > 0:
        try:
            atoms.positions = read(trajectory, index=-1).positions
            _log(f"{task['id']}: retomada desde la trayectoria")
        except Exception as error:  # noqa: BLE001  (a trajectory cut mid-write)
            _log(f"{task['id']}: trayectoria ilegible ({error}); empieza de nuevo")
    atoms.calc = calculator(request["codigo"], request["ajustes"], task["id"])
    steps = int(task.get("pasos", 300))
    bar = Progress(steps, f"{request.get('nombre', 'dft')}: relajación {task['id']}")
    with Trajectory(str(trajectory), "a", atoms) as traj:
        optimizer = BFGS(atoms, restart=str(folder / "bfgs.json"), logfile=None)
        optimizer.attach(traj.write, interval=1)
        optimizer.attach(lambda: bar.step(
            f"|F|máx {np.linalg.norm(atoms.get_forces(), axis=1).max():.3f} eV/Å"), interval=1)
        converged = bool(optimizer.run(fmax=float(task.get("fmax", 0.05)), steps=steps))
    final = folder / "final.extxyz"
    if _rank() == 0:
        write(final, atoms)
    _barrier()
    out = evaluate(atoms, request["codigo"], request["ajustes"], task["id"] + "_final")
    out.update(estructura=str(final.relative_to(HERE)), convergida=converged,
               pasos=optimizer.get_number_of_steps())
    _write_json(done, out)
    return out


def task_displacements(task, request, results):
    atoms = _start(task, request, results)
    vectors = np.load(HERE / task["vectores"])
    if vectors.shape[1:] != (len(atoms), 3):
        raise ValueError(f"vectores {vectors.shape} no corresponden a {len(atoms)} átomos")
    amplitude = float(task.get("amplitud", 0.01))
    folder = HERE / "tareas" / task["id"]
    jobs = [(i, sign) for i in range(len(vectors)) for sign in (1, -1)]
    left = [j for j in jobs if not (folder / f"{j[0]}_{'p' if j[1] > 0 else 'm'}.json").exists()]
    bar = Progress(len(jobs), f"{request.get('nombre', 'dft')}: {task['id']}",
                   done=len(jobs) - len(left))
    scales = []
    for i, vector in enumerate(vectors):
        largest = float(np.linalg.norm(vector, axis=1).max()) or 1.0
        scales.append(amplitude / largest)
    for i, sign in left:
        moved = atoms.copy()
        moved.positions += sign * scales[i] * vectors[i]
        out = evaluate(moved, request["codigo"], request["ajustes"], f"{task['id']}_{i}")
        _write_json(folder / f"{i}_{'p' if sign > 0 else 'm'}.json",
                    {"vector": i, "signo": sign, **out})
        bar.step(f"vector {i} {'+' if sign > 0 else '−'}")
    rows = [json.loads((folder / f"{i}_{'p' if s > 0 else 'm'}.json").read_text())
            for i, s in jobs]
    return {"amplitud": amplitude, "escalas": scales, "desplazamientos": rows}


def task_hessian(task, request, results):
    atoms = _start(task, request, results)
    delta = float(task.get("delta", 0.01))
    chosen = task.get("atomos") or list(range(len(atoms)))
    folder = HERE / "tareas" / task["id"]
    jobs = [(a, x, s) for a in chosen for x in range(3) for s in (1, -1)]

    def name(a, x, s):
        return folder / f"{a}_{'xyz'[x]}{'p' if s > 0 else 'm'}.json"

    left = [j for j in jobs if not name(*j).exists()]
    bar = Progress(len(jobs), f"{request.get('nombre', 'dft')}: hessiana {task['id']}",
                   done=len(jobs) - len(left))
    for a, x, s in left:
        moved = atoms.copy()
        moved.positions[a, x] += s * delta
        out = evaluate(moved, request["codigo"], request["ajustes"], f"{task['id']}_{a}{x}")
        _write_json(name(a, x, s), {"fuerzas": out["fuerzas"], "energia": out["energia"]})
        bar.step(f"átomo {a} {'xyz'[x]}{'+' if s > 0 else '−'}")
    rows = []
    for a in chosen:
        for x in range(3):
            plus = np.array(json.loads(name(a, x, 1).read_text())["fuerzas"])
            minus = np.array(json.loads(name(a, x, -1).read_text())["fuerzas"])
            rows.append(-(plus - minus).ravel() / (2 * delta))
    matrix = HERE / "tareas" / f"{task['id']}_hessiana.npy"
    if _rank() == 0:
        np.save(matrix, np.array(rows))
    _barrier()
    return {"matriz": str(matrix.relative_to(HERE)), "atomos": chosen, "delta": delta}


def task_dos(task, request, results):
    base = task.get("de", "sp")
    if base not in results or "autovalores" not in results[base]:
        sp = task_single({"id": task["id"] + "_scf", "sobre": task.get("sobre", "entrada")},
                         request, results)
    else:
        sp = results[base]
    if "autovalores" not in sp:
        raise ValueError("el motor no da autovalores: no hay DOS")
    sigma = float(task.get("sigma_ev", 0.1))
    grid = np.linspace(-15.0, 10.0, 2501)
    total = np.zeros_like(grid)
    weights = np.asarray(sp["pesos_k"])
    levels = []
    for spin in sp["autovalores"]:
        for w, row in zip(weights, spin, strict=True):
            e = np.asarray(row) - sp["fermi"]
            levels += list(e)
            total += w * np.exp(-((grid[:, None] - e[None, :]) / sigma) ** 2 / 2).sum(axis=1)
    total *= (2 if len(sp["autovalores"]) == 1 else 1) / (sigma * np.sqrt(2 * np.pi))
    levels = np.asarray(levels)
    below, above = levels[levels <= 0], levels[levels > 0]
    gap = float(above.min() - below.max()) if len(below) and len(above) else None
    return {"energia_menos_fermi": grid.tolist(), "total": total.tolist(), "gap": gap,
            "sigma_ev": sigma}


TASKS = {"punto_simple": task_single, "relajar": task_relax,
         "desplazamientos": task_displacements, "hessiana": task_hessian, "dos": task_dos}


def engine_version(code: str) -> str | None:
    try:
        if code == "gpaw":
            import gpaw
            return gpaw.__version__
    except ImportError:
        return None
    return None


def main() -> int:
    import ase

    request = json.loads((HERE / "solicitud.json").read_text(encoding="utf-8"))
    if request.get("esquema") != SCHEMA:
        raise SystemExit(f"esquema {request.get('esquema')!r}; este conductor lee {SCHEMA!r}")
    used = HERE / "tareas" / "solicitud_usada.json"
    if used.exists() and json.loads(used.read_text()) != request:
        raise SystemExit("solicitud.json cambió después de empezar: los resultados de tareas/ "
                         "son de otra solicitud. Crea otra carpeta.")
    _write_json(used, request)
    out = {"esquema": SCHEMA, "codigo": request["codigo"],
           "version_codigo": engine_version(request["codigo"]), "ase": ase.__version__,
           "unidades": {"energia": "eV", "fuerza": "eV/Å", "distancia": "Å", "dipolo": "e·Å"},
           "terminado": False, "tareas": {}}
    for task in request["tareas"]:
        kind = task.get("tipo")
        if kind not in TASKS:
            raise SystemExit(f"tarea {task.get('id')}: tipo {kind!r} desconocido "
                             f"({', '.join(TASKS)})")
        _log(f"tarea {task['id']} ({kind})")
        out["tareas"][task["id"]] = TASKS[kind](task, request, out["tareas"])
        _write_json(HERE / "resultado.json", out)
    out["terminado"] = True
    _write_json(HERE / "resultado.json", out)
    _log("terminado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
