"""One queue for every calculation the window launches.

A directory is a job. What runs it, and how to tell whether it finished,
depends on what kind of directory it is; an :class:`Adapter` answers that:

* :class:`VibspecAdapter` -- vibspec's GPAW directories (``record.json`` +
  ``run.py``), finished when the record says ``done``.
* :class:`StepsAdapter` -- QE / SIESTA / LAMMPS export directories
  (``job.json``), run by :mod:`carbonforge.jobs.run`, finished when the
  manifest says ``done``.

Jobs are subprocesses, never threads: a thread cannot be cancelled, and a DFT
run must be. A job's state comes from the process AND the directory's record:
exit code 0 without a finished record is an error.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .manifest import MANIFEST, JobManifest

#: Job states as the window shows them.
QUEUED, RUNNING, DONE, ERROR, CANCELLED = "en cola", "corriendo", "terminado", "error", "cancelado"

CommandBuilder = Callable[[Path, int], list[str]]


# --------------------------------------------------------------------------
# Kinds of directory
# --------------------------------------------------------------------------

class Adapter:
    """How to run one kind of calculation directory."""

    engine = ""

    def matches(self, directory: Path) -> bool:
        raise NotImplementedError

    def check(self, directory: Path) -> None:
        """Raise if the directory cannot be queued."""

    def command(self, directory: Path, nprocs: int) -> list[str]:
        raise NotImplementedError

    def succeeded(self, directory: Path) -> bool:
        raise NotImplementedError

    def progress(self, directory: Path) -> str:
        raise NotImplementedError

    def available(self, directory: Path) -> tuple[bool, str]:
        """Whether this machine has what the job needs, and why not."""
        return True, ""


def gpaw_command(directory: Path, nprocs: int = 1) -> list[str]:
    """``python run.py``, under ``mpiexec -n N`` in parallel (GPAW >= 22)."""
    script = str(Path(directory) / "run.py")
    if nprocs > 1:
        mpiexec = shutil.which("mpiexec") or shutil.which("mpirun")
        if mpiexec is None:
            raise RuntimeError("No se encontró mpiexec: instala OpenMPI o usa 1 proceso.")
        return [mpiexec, "-n", str(nprocs), sys.executable, script]
    return [sys.executable, script]


class VibspecAdapter(Adapter):
    engine = "gpaw"

    def matches(self, directory: Path) -> bool:
        return (directory / "record.json").exists()

    def check(self, directory: Path) -> None:
        from ..vibspec.core.record import CalcRecord

        CalcRecord.load(directory)

    def command(self, directory: Path, nprocs: int) -> list[str]:
        return gpaw_command(directory, nprocs)

    def succeeded(self, directory: Path) -> bool:
        from ..vibspec.core.record import CalcRecord

        try:
            return CalcRecord.load(directory).status == "done"
        except (FileNotFoundError, ValueError):
            return False

    def progress(self, directory: Path) -> str:
        from ..vibspec.core.record import CalcRecord

        try:
            record = CalcRecord.load(directory)
        except (FileNotFoundError, ValueError):
            return "sin record.json"
        status = record.status
        if status == "relaxing":
            log = directory / "relax.log"
            lines = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
            return f"relajando: paso {max(0, len(lines) - 1)}"
        if status == "vibrations":
            per_atom = 6 if int(record.spec.get("nfree", 2)) == 2 else 12
            total = per_atom * record.n_atoms + 1
            cache = directory / "ir"
            done = len(list(cache.glob("cache.*.json"))) if cache.exists() else 0
            return f"vibraciones: {done}/{total} desplazamientos"
        return {"prepared": "preparado", "relaxed": "relajado", "done": "terminado",
                "error": f"error: {record.history[-1]['message'] if record.history else ''}"
                }.get(status, status)

    def available(self, directory: Path) -> tuple[bool, str]:
        from ..vibspec.core import gpaw_available

        if gpaw_available():
            return True, ""
        return False, ("GPAW no está instalado aquí (lo normal en Windows). Córrelo en "
                       f"Ubuntu: cd {directory} && python run.py")


class StepsAdapter(Adapter):
    engine = "pasos"

    def matches(self, directory: Path) -> bool:
        return (directory / MANIFEST).exists()

    def check(self, directory: Path) -> None:
        JobManifest.load(directory)

    def command(self, directory: Path, nprocs: int) -> list[str]:
        return [sys.executable, "-m", "carbonforge.jobs.run", str(directory),
                "--nprocs", str(nprocs)]

    def succeeded(self, directory: Path) -> bool:
        try:
            return JobManifest.load(directory).status == "done"
        except (FileNotFoundError, ValueError):
            return False

    def progress(self, directory: Path) -> str:
        try:
            return JobManifest.load(directory).progress()
        except (FileNotFoundError, ValueError):
            return f"sin {MANIFEST}"

    def available(self, directory: Path) -> tuple[bool, str]:
        from .run import find_program

        manifest = JobManifest.load(directory)
        missing = sorted({s.program for s in manifest.steps
                          if not s.done and s.program != "carbonforge"
                          and find_program(s.program) is None})
        if not missing:
            return True, ""
        return False, (f"Falta {', '.join(missing)} en esta máquina. Córrelo donde esté "
                       f"instalado: python -m carbonforge.jobs.run {directory}")


ADAPTERS: tuple[Adapter, ...] = (VibspecAdapter(), StepsAdapter())


def adapter_for(directory: Path) -> Adapter:
    """The adapter for ``directory``; ``FileNotFoundError`` if none applies."""
    directory = Path(directory)
    for adapter in ADAPTERS:
        if adapter.matches(directory):
            return adapter
    raise FileNotFoundError(
        f"{directory.name}: no es un directorio de cálculo (falta record.json de vibspec "
        f"o {MANIFEST} de una exportación)."
    )


def engine_of(directory: Path) -> str:
    """``gpaw``, ``qe``, ``siesta`` or ``lammps``, for display."""
    adapter = adapter_for(directory)
    if isinstance(adapter, StepsAdapter):
        return JobManifest.load(directory).engine
    return adapter.engine


# --------------------------------------------------------------------------
# The queue
# --------------------------------------------------------------------------

@dataclass
class Job:
    """One calculation directory and the process running it, if any."""

    directory: Path
    nprocs: int = 1
    state: str = QUEUED
    process: Optional[subprocess.Popen] = None
    returncode: Optional[int] = None
    command: list[str] = field(default_factory=list)
    adapter: Optional[Adapter] = None

    @property
    def name(self) -> str:
        return self.directory.name

    @property
    def engine(self) -> str:
        try:
            return engine_of(self.directory)
        except (FileNotFoundError, ValueError):
            return "?"

    @property
    def log_path(self) -> Path:
        return self.directory / "job.log"

    def progress(self) -> str:
        return self.adapter.progress(self.directory) if self.adapter else ""


class JobQueue:
    """Runs calculation directories as subprocesses, a few at a time.

    Parameters
    ----------
    max_parallel
        How many jobs may run at once. One is right for a desktop: two DFT
        runs on the same cores are slower than one after the other.
    command
        ``command(directory, nprocs) -> argv`` for every job, replacing the
        adapter's. Tests pass a stand-in.
    """

    def __init__(self, max_parallel: int = 1, command: Optional[CommandBuilder] = None):
        self.max_parallel = max_parallel
        self.command = command
        self.jobs: list[Job] = []

    def submit(self, directory: Path, nprocs: int = 1) -> Job:
        """Queue a calculation directory. Refuses one already queued or running."""
        directory = Path(directory)
        adapter = adapter_for(directory)
        adapter.check(directory)
        for job in self.jobs:
            if job.directory == directory and job.state in (QUEUED, RUNNING):
                raise ValueError(f"{directory.name} ya está {job.state}.")
        job = Job(directory=directory, nprocs=nprocs, adapter=adapter)
        self.jobs.append(job)
        return job

    def _start(self, job: Job) -> None:
        builder = self.command or job.adapter.command
        job.command = builder(job.directory, job.nprocs)
        handle = job.log_path.open("ab")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # Own process group on POSIX, so cancelling also stops pw.x & co.
        extra = {"start_new_session": True} if os.name == "posix" else {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        job.process = subprocess.Popen(job.command, cwd=job.directory, stdout=handle,
                                       stderr=subprocess.STDOUT, env=env, **extra)
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
            job.state = DONE if code == 0 and job.adapter.succeeded(job.directory) else ERROR
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
        """Stop a job and whatever it launched. The directory stays valid:
        running it again resumes where it stopped."""
        if job.state == RUNNING and job.process is not None and job.process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(job.process.pid, signal.SIGTERM)
                else:
                    job.process.terminate()
            except (ProcessLookupError, PermissionError):
                job.process.terminate()
            try:
                job.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(job.process.pid, signal.SIGKILL)
                else:
                    job.process.kill()
                job.process.wait()
        if job.state in (QUEUED, RUNNING):
            job.state = CANCELLED

    def remove_finished(self) -> None:
        self.jobs = [job for job in self.jobs if job.state in (QUEUED, RUNNING)]

    def active(self) -> bool:
        return any(job.state in (QUEUED, RUNNING) for job in self.jobs)

    def shutdown(self) -> None:
        """Cancel everything; called when the window closes."""
        for job in self.jobs:
            self.cancel(job)


def result_of(directory: Path) -> Optional[tuple[str, Path]]:
    """What a finished directory produced that the window can show.

    Returns ``(kind, path)`` with kind ``"ir_gpaw"`` (vibspec), ``"bands"``
    (QE ``bands.dat.gnu`` or SIESTA ``*.bands``), ``"spectrum"`` (QE
    ``dynmat.out``), ``"pdos"`` (``projwfc.x`` files) or ``"dos"`` (QE
    ``dos.dat``); None when there is nothing
    to show (a LAMMPS run, a plain scf).
    """
    directory = Path(directory)
    if (directory / "record.json").exists():
        return "ir_gpaw", directory
    for kind, pattern in (("bands", "bands.dat.gnu"), ("bands", "*.bands"),
                          ("spectrum", "dynmat.out"), ("pdos", "*pdos_tot"),
                          ("dos", "*.dos"), ("dos", "dos.dat")):
        found = sorted(directory.glob(pattern))
        if found:
            return kind, found[0]
    return None


def tail(path: Path, lines: int = 40) -> str:
    """The last ``lines`` lines of a text file, or an empty string."""
    path = Path(path)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(text[-lines:])


def job_log(job: Job, lines: int = 40) -> str:
    """What the window shows for a job: its command, its output, and the
    relaxation log (vibspec) when there is one."""
    parts = [f"$ {' '.join(job.command)}" if job.command else "", tail(job.log_path, lines)]
    relax = tail(job.directory / "relax.log", 12)
    if relax:
        parts += ["--- relax.log ---", relax]
    return "\n".join(p for p in parts if p)
