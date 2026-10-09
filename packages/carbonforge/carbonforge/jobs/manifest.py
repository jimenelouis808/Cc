"""``job.json``: what a calculation directory runs, in order, and how far it got.

carbonforge writes QE, SIESTA and LAMMPS inputs; running them is a chain of
programs that must go in order and share files (pw.x → ph.x → dynmat.x). The
manifest names that chain once, so the same directory runs identically from
the window's queue, from a terminal (``python -m carbonforge.jobs.run DIR``)
or on a cluster, and keeps a history of what happened.

The steps are inferred from the files the exporters write
(:func:`manifest_for_directory`): an export directory is its own description,
and nothing in the writers has to change to keep the two in sync. vibspec's
GPAW directories have their own ``record.json`` and are not described here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

MANIFEST = "job.json"

#: Manifest states. ``running`` with a dead process means it was interrupted.
STATES = ("prepared", "running", "done", "error", "cancelled")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Step:
    """One program run.

    Attributes
    ----------
    name
        Short label (``"scf"``, ``"ph"``).
    program
        Executable name looked up on ``PATH`` (``pw.x``, ``siesta``, ``lmp``).
    input
        Input file in the directory.
    output
        Where the program's standard output goes.
    parallel
        Whether it runs under MPI when more than one process is asked for.
        Post-processing tools (``bands.x``, ``dynmat.x``) are serial.
    stdin
        Feed the input on standard input (SIESTA) instead of ``-in FILE``.
    args
        Extra arguments. For ``program="carbonforge"`` they are a carbonforge
        command run with this Python (``update-geometry pw.relax.out ...``).
    pools
        k-point pools (``-nk``) for pw.x / ph.x; used only when the number of
        processes is a multiple of it.
    done
        Set by the runner when the step finished with exit code 0.
    """

    name: str
    program: str
    input: str
    output: str
    parallel: bool = True
    stdin: bool = False
    args: list[str] = field(default_factory=list)
    pools: int = 1
    done: bool = False


@dataclass
class JobManifest:
    """The chain of steps of one directory, and its history."""

    engine: str
    title: str
    steps: list[Step]
    status: str = "prepared"
    history: list[dict] = field(default_factory=list)
    created: str = field(default_factory=_now)
    version: int = 1

    def set_status(self, status: str, message: str = "") -> None:
        if status not in STATES:
            raise ValueError(f"Estado desconocido: '{status}'. Válidos: {STATES}.")
        self.status = status
        self.history.append({"time": _now(), "status": status, "message": message})

    @property
    def next_step(self) -> Optional[Step]:
        return next((step for step in self.steps if not step.done), None)

    def progress(self) -> str:
        done = sum(step.done for step in self.steps)
        total = len(self.steps)
        if self.status == "running":
            step = self.next_step
            text = f"paso {done + 1}/{total}: {step.program if step else ''}"
            # pw.x, ph.x or LAMMPS cannot tell how far they are; how long the current
            # step has been running can be said (since the last state change).
            if self.history:
                from datetime import datetime, timezone

                from ..utils.eta import duration

                since = datetime.fromisoformat(self.history[-1]["time"])
                text += f" · lleva {duration((datetime.now(timezone.utc) - since).total_seconds())}"
            return text
        if self.status == "error":
            message = self.history[-1]["message"] if self.history else ""
            return f"error: {message}"
        return {"prepared": f"preparado ({total} pasos)" if not done
                else f"interrumpido tras {done}/{total}",
                "done": "terminado", "cancelled": f"cancelado tras {done}/{total}"
                }[self.status]

    def reset(self) -> None:
        """Forget which steps finished: the next run starts from the first."""
        for step in self.steps:
            step.done = False
        self.set_status("prepared", "reiniciado")

    def save(self, directory: Path) -> Path:
        path = Path(directory) / MANIFEST
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path) -> "JobManifest":
        path = Path(directory) / MANIFEST
        if not path.exists():
            raise FileNotFoundError(f"{Path(directory).name}: no hay {MANIFEST}.")
        data = json.loads(path.read_text(encoding="utf-8"))
        data["steps"] = [Step(**step) for step in data["steps"]]
        return cls(**data)


def _qe_chain(directory: Path) -> Optional[tuple[str, list[Step]]]:
    has = lambda name: (directory / name).exists()  # noqa: E731
    if has("pw.scf.in") and has("pw.bands.in"):
        return "QE bandas", [
            Step("scf", "pw.x", "pw.scf.in", "pw.scf.out"),
            Step("bands", "pw.x", "pw.bands.in", "pw.bands.out"),
            Step("bands.x", "bands.x", "bands.in", "bands.out", parallel=False),
        ]
    if has("pw.scf.in") and has("pw.nscf.in") and has("dos.in"):
        steps = [
            Step("scf", "pw.x", "pw.scf.in", "pw.scf.out"),
            Step("nscf", "pw.x", "pw.nscf.in", "pw.nscf.out"),
            Step("dos.x", "dos.x", "dos.in", "dos.out", parallel=False),
        ]
        if has("projwfc.in"):
            steps.append(Step("projwfc.x", "projwfc.x", "projwfc.in", "projwfc.out",
                              parallel=False))
        return "QE densidad de estados", steps
    if has("pw.in") and has("ph.in") and has("dynmat.in"):
        return "QE fonones / IR / Raman", [
            Step("scf", "pw.x", "pw.in", "pw.out"),
            Step("ph", "ph.x", "ph.in", "ph.out"),
            # Never dynmat.out as filout: that name is this step's stdout.
            Step("dynmat.x", "dynmat.x", "dynmat.in", "dynmat.out", parallel=False),
        ]
    if has("pw.in"):
        return "QE pw.x", [Step("pw", "pw.x", "pw.in", "pw.out")]
    if has("pw.scf.in"):
        return "QE scf", [Step("scf", "pw.x", "pw.scf.in", "pw.scf.out")]
    return None


def _qe_steps(directory: Path, pools: int) -> Optional[tuple[str, list[Step]]]:
    """The QE chain, preceded by a relaxation when a recipe asked for one.

    A recipe project (``workflows.pipeline.write_preset_project``) relaxes
    first and rewrites the property inputs with the relaxed geometry
    (``carbonforge update-geometry``) before computing the property on it.
    """
    chain = _qe_chain(directory)
    steps: list[Step] = []
    title = chain[0] if chain else "QE relajación"
    if (directory / "pw.relax.in").exists():
        steps.append(Step("relax", "pw.x", "pw.relax.in", "pw.relax.out"))
        if chain is not None:
            steps.append(Step("geometría", "carbonforge", "", "update-geometry.out",
                              parallel=False,
                              args=["update-geometry", "pw.relax.out", "--apply-to", "."]))
            title = f"{title} tras relajar"
    if chain is not None:
        steps += chain[1]
    if not steps:
        return None
    for step in steps:
        if step.program in ("pw.x", "ph.x"):
            step.pools = pools
    return title, steps


def manifest_for_directory(directory: str | Path, pools: int = 1) -> JobManifest:
    """Describe an export directory (``qe/``, ``siesta/``, ``lammps/``, or a
    recipe project) as steps. ``pools`` sets pw.x/ph.x ``-nk``.

    Raises
    ------
    ValueError
        If the directory holds no input carbonforge knows how to run.
    """
    directory = Path(directory)
    chain = _qe_steps(directory, pools)
    if chain is not None:
        title, steps = chain
        return JobManifest("qe", title, steps)
    fdf = sorted(directory.glob("*.fdf"))
    if fdf:
        stem = fdf[0].stem
        return JobManifest("siesta", "SIESTA", [
            Step("siesta", "siesta", fdf[0].name, f"{stem}.out", stdin=True)])
    if (directory / "in.lammps").exists():
        return JobManifest("lammps", "LAMMPS", [
            Step("lammps", "lmp", "in.lammps", "lammps.out")])
    raise ValueError(
        f"{directory.name}: no hay entradas de QE (pw.in...), SIESTA (*.fdf) ni LAMMPS "
        "(in.lammps) que carbonforge sepa correr."
    )


def write_manifest(directory: str | Path, pools: int = 1) -> Path:
    """Infer and save ``job.json`` for an export directory; returns its path."""
    return manifest_for_directory(directory, pools=pools).save(Path(directory))
