"""Everything the vibspec window does, without the window.

Same split as :mod:`carbonforge.gui`: this module is pure Python and fully
tested on a machine with no display; :mod:`carbonforge.vibspec.gui.app` only
lays out widgets and calls it. Nothing physical is decided here either --
building, checking, preparing, running and analysing are all
:mod:`carbonforge.vibspec.core`; this layer turns form values into calls and
results into things a window can show.

The job runner
--------------
A DFT calculation never runs in the GUI process. Each job is a
**subprocess** running the calculation's own ``run.py`` (optionally under
``mpiexec``), for the reason :mod:`nanocarbon_lab`'s GUI learned the hard way:
a thread cannot be interrupted mid-computation, and a calculation that cannot
be cancelled freezes the window for hours. A subprocess can be terminated,
and because ``run.py`` is restartable, cancelling loses at most the
displacement in progress. The window polls :meth:`JobQueue.poll` from its
event loop; the job's state comes from the process and from ``record.json``,
never from guessing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.io import read

from ...builders.nanoribbon import DEFAULT_VACUUM_PER_SIDE, build_finite_nanoribbon
from ...gui.params import ParamSpec, collect_values
from ...validation.checks import ValidationReport
from ..core import (
    PRESETS,
    CalcRecord,
    CalcSpec,
    SpinAdvice,
    apply_preset,
    check_structure,
    gpaw_available,
    prepare,
    suggest_spin,
)

# --------------------------------------------------------------------------
# Forms
# --------------------------------------------------------------------------

AUTO = "auto"

BUILDER_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("edge", "Borde de la cinta", "choice", "armchair",
              choices=("armchair", "zigzag"),
              help="Tipo de los bordes largos. Los extremos son del otro tipo."),
    ParamSpec("width", "Ancho (líneas de dímeros)", "int", 5, minimum=2, maximum=20),
    ParamSpec("length", "Largo (celdas)", "int", 3, minimum=1, maximum=20),
    ParamSpec("vacuum_per_side", "Vacío por lado (Å)", "float", DEFAULT_VACUUM_PER_SIDE,
              minimum=6.0, maximum=20.0,
              help="Mínimo 6 Å; en modo PW, 8 Å."),
    ParamSpec("preset", "Funcionalización", "choice", "pristine", choices=tuple(PRESETS)),
    ParamSpec("site", "Sitio", "choice", AUTO,
              choices=(AUTO, "middle", "center", "near_edge"),
              help="auto: centro de un borde largo (grupos de borde) o centro de la "
                   "cinta (interiores). Para un átomo concreto, usa el índice."),
    ParamSpec("site_index", "Índice de átomo (opcional)", "int", -1, minimum=-1,
              help="-1 para usar el sitio de arriba."),
    ParamSpec("site_edge", "Tipo de borde del sitio", "choice", AUTO,
              choices=(AUTO, "armchair", "zigzag")),
)

_DEFAULT_SPEC = CalcSpec()

CALC_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("xc", "Funcional", "choice", _DEFAULT_SPEC.xc,
              choices=("PBE", "RPBE", "PBEsol", "BLYP", "LDA")),
    ParamSpec("mode", "Modo GPAW", "choice", _DEFAULT_SPEC.mode, choices=("lcao", "fd", "pw"),
              help="lcao para barrer, fd para confirmar; pw con más vacío."),
    ParamSpec("basis", "Base LCAO", "choice", _DEFAULT_SPEC.basis,
              choices=("dzp", "szp", "sz")),
    ParamSpec("h", "Rejilla h (Å)", "float", _DEFAULT_SPEC.h, minimum=0.1, maximum=0.3),
    ParamSpec("ecut", "Corte PW (eV)", "float", _DEFAULT_SPEC.ecut, minimum=200, maximum=1500),
    ParamSpec("spinpol", "Espín", "choice", AUTO, choices=(AUTO, "sí", "no"),
              help="auto: lo decide la estructura (electrones impares, bordes zigzag)."),
    ParamSpec("charge", "Carga", "int", _DEFAULT_SPEC.charge, minimum=-4, maximum=4),
    ParamSpec("fmax", "fmax relajación (eV/Å)", "float", _DEFAULT_SPEC.fmax,
              minimum=1e-4, maximum=0.2),
    ParamSpec("delta", "Desplazamiento (Å)", "float", _DEFAULT_SPEC.delta,
              minimum=1e-4, maximum=0.1),
    ParamSpec("nfree", "Desplazamientos por coordenada", "choice", "2", choices=("2", "4")),
    ParamSpec("ir_method", "Constantes de fuerza", "choice", _DEFAULT_SPEC.ir_method,
              choices=("frederiksen", "standard")),
    ParamSpec("scale_factor", "Factor de escala", "float", _DEFAULT_SPEC.scale_factor,
              minimum=0.5, maximum=1.5),
)

RUN_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("nprocs", "Procesos MPI", "int", 1, minimum=1, maximum=256,
              help="1 = en serie. Más de 1 necesita mpiexec y GPAW compilado con MPI."),
)


def defaults(specs: Sequence[ParamSpec]) -> dict[str, Any]:
    """The default raw value of every field, as a form starts."""
    return {spec.key: spec.default for spec in specs}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class ModelResult:
    """A built structure and what the checks say about it."""

    atoms: Atoms
    report: ValidationReport
    spin: SpinAdvice

    def summary(self) -> str:
        preset = self.atoms.info.get("vibspec_preset", {}).get("key", "pristine")
        lines = [f"{self.atoms.get_chemical_formula()} — {len(self.atoms)} átomos — "
                 f"preset: {preset}",
                 f"Espín recomendado: {'sí' if self.spin.spinpol else 'no'}"]
        # Plain-text markers: they render with any Tk font.
        lines += [f"ERROR: {e}" for e in self.report.errors]
        lines += [f"AVISO: {w}" for w in self.report.warnings]
        return "\n".join(lines)


def build_model(raw: dict[str, Any]) -> ModelResult:
    """Build the ribbon, apply the preset and check it, from raw form values."""
    values = collect_values(BUILDER_PARAMS, raw)
    atoms = build_finite_nanoribbon(
        values["width"], values["length"], edge=values["edge"],
        vacuum_per_side=values["vacuum_per_side"],
    )
    if values["site_index"] >= 0:
        position: Any = values["site_index"]
    else:
        position = None if values["site"] == AUTO else values["site"]
    edge = None if values["site_edge"] == AUTO else values["site_edge"]
    atoms = apply_preset(atoms, values["preset"], position=position, edge=edge)
    return ModelResult(atoms, check_structure(atoms), suggest_spin(atoms))


def spec_from_form(raw: dict[str, Any]) -> CalcSpec:
    """A :class:`CalcSpec` from raw form values (bounds checked, Spanish errors)."""
    values = collect_values(CALC_PARAMS, raw)
    values["spinpol"] = {AUTO: None, "sí": True, "no": False}[values["spinpol"]]
    values["nfree"] = int(values["nfree"])
    return CalcSpec(**values)


def job_name(atoms: Atoms) -> str:
    """A readable, filesystem-safe default name: ``amine_armchair_5x3``."""
    info = atoms.info
    preset = info.get("vibspec_preset", {}).get("key", "modelo")
    name = f"{preset}_{info.get('edge', 'gnr')}_{info.get('width', '')}x{info.get('length', '')}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def prepare_job(atoms: Atoms, spec: CalcSpec, root: Path, name: Optional[str] = None,
                force: bool = False) -> Path:
    """Prepare a calculation directory under ``root``; returns its path."""
    directory = Path(root) / (name or job_name(atoms))
    prepare(atoms, spec, directory, force=force)
    return directory


# --------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------

#: Job states as the window shows them.
QUEUED, RUNNING, DONE, ERROR, CANCELLED = "en cola", "corriendo", "terminado", "error", "cancelado"

CommandBuilder = Callable[[Path, int], list[str]]


def default_command(directory: Path, nprocs: int = 1) -> list[str]:
    """The command that runs a prepared calculation.

    ``python run.py`` in serial; ``mpiexec -n N python run.py`` in parallel,
    which is how GPAW runs under MPI since version 22.
    """
    script = str(Path(directory) / "run.py")
    if nprocs > 1:
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            raise RuntimeError("No se encontró mpiexec: instala OpenMPI o usa 1 proceso.")
        return [mpiexec, "-n", str(nprocs), sys.executable, script]
    return [sys.executable, script]


def can_run_locally() -> tuple[bool, str]:
    """Whether this machine can run calculations, and why not if it cannot."""
    if gpaw_available():
        return True, ""
    return False, ("GPAW no está instalado aquí (lo normal en Windows). Prepara los "
                   "cálculos y córrelos en Ubuntu con run.py; luego ábrelos en Resultados.")


def progress(directory: Path) -> str:
    """One line on how far a calculation has got, read from its files."""
    directory = Path(directory)
    try:
        record = CalcRecord.load(directory)
    except (FileNotFoundError, ValueError):
        return "sin record.json"
    status = record.status
    if status == "relaxing":
        log = directory / "relax.log"
        steps = max(0, len(log.read_text(encoding="utf-8").splitlines()) - 1) if log.exists() else 0
        return f"relajando: paso {steps}"
    if status == "vibrations":
        per_atom = 6 if int(record.spec.get("nfree", 2)) == 2 else 12
        total = per_atom * record.n_atoms + 1
        cache = directory / "ir"
        done = len(list(cache.glob("cache.*.json"))) if cache.exists() else 0
        return f"vibraciones: {done}/{total} desplazamientos"
    return {"prepared": "preparado", "relaxed": "relajado", "done": "terminado",
            "error": f"error: {record.history[-1]['message'] if record.history else ''}"
            }.get(status, status)


@dataclass
class Job:
    """One calculation directory and the process running it, if any."""

    directory: Path
    nprocs: int = 1
    state: str = QUEUED
    process: Optional[subprocess.Popen] = None
    returncode: Optional[int] = None
    command: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def log_path(self) -> Path:
        return self.directory / "job.log"


class JobQueue:
    """Runs prepared calculations as subprocesses, a few at a time.

    Parameters
    ----------
    max_parallel
        How many jobs may run at once. One is right for a desktop: two GPAW
        runs on the same cores are slower than one after the other.
    command
        ``command(directory, nprocs) -> argv``. Tests pass a stand-in.
    """

    def __init__(self, max_parallel: int = 1, command: CommandBuilder = default_command):
        self.max_parallel = max_parallel
        self.command = command
        self.jobs: list[Job] = []

    def submit(self, directory: Path, nprocs: int = 1) -> Job:
        """Queue a prepared calculation. Refuses one already queued or running."""
        directory = Path(directory)
        CalcRecord.load(directory)          # must be a calculation directory
        for job in self.jobs:
            if job.directory == directory and job.state in (QUEUED, RUNNING):
                raise ValueError(f"{directory.name} ya está {job.state}.")
        job = Job(directory=directory, nprocs=nprocs)
        self.jobs.append(job)
        return job

    def _start(self, job: Job) -> None:
        job.command = self.command(job.directory, job.nprocs)
        handle = job.log_path.open("ab")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        job.process = subprocess.Popen(
            job.command, cwd=job.directory, stdout=handle, stderr=subprocess.STDOUT, env=env,
        )
        handle.close()                      # the child has its own descriptor
        job.state = RUNNING

    def poll(self) -> list[Job]:
        """Advance the queue; return the jobs whose state changed."""
        changed = []
        for job in self.jobs:
            if job.state != RUNNING or job.process is None:
                continue
            code = job.process.poll()
            if code is None:
                continue
            job.returncode = code
            try:
                status = CalcRecord.load(job.directory).status
            except (FileNotFoundError, ValueError):
                status = "error"
            job.state = DONE if code == 0 and status == "done" else ERROR
            changed.append(job)
        running = sum(job.state == RUNNING for job in self.jobs)
        for job in self.jobs:
            if running >= self.max_parallel:
                break
            if job.state == QUEUED:
                try:
                    self._start(job)
                except (OSError, RuntimeError) as exc:
                    job.state = ERROR
                    with job.log_path.open("a", encoding="utf-8") as handle:
                        handle.write(f"No se pudo lanzar: {exc}\n")
                else:
                    running += 1
                changed.append(job)
        return changed

    def cancel(self, job: Job, timeout: float = 10.0) -> None:
        """Stop a job. Its directory stays valid: run.py resumes where it stopped."""
        if job.state == RUNNING and job.process is not None:
            job.process.terminate()
            try:
                job.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                job.process.kill()
                job.process.wait()
        if job.state in (QUEUED, RUNNING):
            job.state = CANCELLED

    def active(self) -> bool:
        return any(job.state in (QUEUED, RUNNING) for job in self.jobs)

    def shutdown(self) -> None:
        """Cancel everything; called when the window closes."""
        for job in self.jobs:
            self.cancel(job)


def tail(path: Path, lines: int = 40) -> str:
    """The last ``lines`` lines of a text file, or an empty string."""
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def job_log(job: Job, lines: int = 40) -> str:
    """What the window shows for a job: its output, then the relaxation log."""
    parts = [f"$ {' '.join(job.command)}" if job.command else "", tail(job.log_path, lines)]
    relax = tail(job.directory / "relax.log", 12)
    if relax:
        parts += ["--- relax.log ---", relax]
    return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------
# Normal modes
# --------------------------------------------------------------------------

def load_modes(directory: Path) -> tuple[Atoms, dict[str, np.ndarray], CalcRecord]:
    """The relaxed structure, the stored modes and the record of a finished run."""
    directory = Path(directory)
    record = CalcRecord.load(directory)
    if record.status != "done":
        raise ValueError(f"{directory.name}: el cálculo no ha terminado ({record.status}).")
    atoms = read(directory / record.files["relaxed"])
    with np.load(directory / record.files["modes"]) as data:
        modes = {key: data[key] for key in data.files}
    return atoms, modes, record


def mode_at(record: CalcRecord, wavenumber: float, scale_factor: float = 1.0,
            window_cm1: float = 40.0) -> Optional[int]:
    """The mode behind the band clicked at ``wavenumber`` (scaled axis).

    The strongest IR mode within ``window_cm1``; if none is that close, the
    nearest. Returns an index into ``modes.npz``, or ``None`` with no modes.
    """
    frequencies = np.asarray(record.results.get("frequencies_cm1", []), dtype=float)
    if frequencies.size == 0:
        return None
    intensities = np.asarray(record.results["ir_intensity"], dtype=float)
    indices = record.results["mode_indices"]
    distance = np.abs(frequencies * scale_factor - wavenumber)
    near = np.flatnonzero(distance <= window_cm1)
    best = near[np.argmax(intensities[near])] if near.size else int(np.argmin(distance))
    return int(indices[best])


def mode_vector(modes: dict[str, np.ndarray], mode_index: int) -> np.ndarray:
    """Cartesian displacement of one mode, scaled so the largest is 1 Å."""
    vector = np.asarray(modes["modes"][mode_index], dtype=float)
    largest = float(np.linalg.norm(vector, axis=1).max())
    return vector / largest if largest > 0 else vector


def mode_frames(atoms: Atoms, vector: np.ndarray, n_frames: int = 24,
                amplitude: float = 0.35) -> list[np.ndarray]:
    """Positions for one period of the mode, largest excursion ``amplitude`` Å."""
    base = atoms.get_positions()
    phases = np.sin(2 * np.pi * np.arange(n_frames) / n_frames)
    return [base + amplitude * phase * vector for phase in phases]


def view_angles(atoms: Atoms) -> tuple[float, float]:
    """Matplotlib ``(elev, azim)`` that looks straight down on a planar molecule.

    The view direction is the principal axis of least spread -- the plane
    normal of a flake, whichever way it lies -- so a mode in the plane is
    seen face-on instead of edge-on.
    """
    positions = atoms.get_positions() - atoms.get_positions().mean(axis=0)
    _, _, vt = np.linalg.svd(positions, full_matrices=False)
    normal = vt[-1] if len(vt) == 3 else np.array([0.0, 0.0, 1.0])
    x, y, z = normal / np.linalg.norm(normal)
    elevation = float(np.degrees(np.arcsin(np.clip(z, -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(y, x)))
    return elevation, azimuth


def mode_character(atoms: Atoms, vector: np.ndarray, top: int = 4) -> str:
    """Which atoms carry the motion, as a line for the band table.

    Shares are of the mass-weighted kinetic energy, the usual measure of
    how much of a mode belongs to each atom; an N-H stretch shows up as
    mostly H with some N, whatever the rest of the ribbon does.
    """
    weights = atoms.get_masses() * (np.asarray(vector) ** 2).sum(axis=1)
    total = weights.sum()
    if total <= 0:
        return "sin desplazamiento"
    share = weights / total
    symbols = atoms.get_chemical_symbols()
    order = np.argsort(share)[::-1][:top]
    by_element: dict[str, float] = {}
    for symbol, value in zip(symbols, share, strict=True):
        by_element[symbol] = by_element.get(symbol, 0.0) + float(value)
    elements = ", ".join(f"{k} {v:.0%}" for k, v in sorted(by_element.items(),
                                                            key=lambda kv: -kv[1]) if v >= 0.05)
    atoms_text = ", ".join(f"{symbols[i]}{i} {share[i]:.0%}" for i in order)
    return f"{elements}  |  {atoms_text}"
