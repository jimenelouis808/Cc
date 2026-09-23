"""One calculation as one reproducible unit on disk.

A calculation lives in its own directory::

    runs/amine/
        record.json     what was asked, what ran, what came out, with versions
        initial.xyz     the structure as prepared (with its provenance in info)
        relaxed.xyz     the structure the vibrations were computed on
        run.py          the script to run it: ``mpiexec -n 4 gpaw python run.py``
        modes.npz       all frequencies and mode vectors, for animation
        relax.traj, relax.log, gpaw.txt, ir/   raw output of ASE and GPAW

Paths inside ``record.json`` are relative to the directory, so the directory
can be prepared on Windows, run on Ubuntu or a cluster, and brought back.

The record's ``status`` follows the job states a GUI shows:
``prepared`` (queued), ``relaxing`` / ``relaxed`` / ``vibrations`` (running),
``done`` and ``error``. Every change is appended to ``history`` with a
timestamp, so a failed run says where it failed.

:func:`index_records` puts every record under a directory into an ASE
database, one row per calculation: that is the searchable catalogue, and the
piece an electronic lab notebook can later connect to.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import ase
import numpy as np

import carbonforge

RECORD_FILE = "record.json"
SCHEMA_VERSION = 1

#: Job states, in the order a successful run goes through them.
STATES = ("prepared", "relaxing", "relaxed", "vibrations", "done", "error")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> Optional[str]:
    """Commit of the carbonforge checkout, when it is one."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(carbonforge.__file__).parent,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def code_versions() -> dict[str, Optional[str]]:
    """Versions of everything that can change the numbers."""
    from .engines import gpaw_version

    return {
        "carbonforge": carbonforge.__version__,
        "carbonforge_git": _git_commit(),
        "ase": ase.__version__,
        "numpy": np.__version__,
        "gpaw": gpaw_version(),
        "python": platform.python_version(),
    }


def _jsonable(value: Any) -> Any:
    """Make numpy scalars and arrays in ``atoms.info`` storable as JSON."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class CalcRecord:
    """Everything about one calculation. Stored as ``record.json``.

    ``versions`` are those of the machine that *ran* the calculation (set
    again by :func:`~carbonforge.vibspec.core.workflow.run`), since that is
    where the numbers come from; ``prepared_with`` keeps the carbonforge
    version that prepared it.
    """

    name: str
    spec: dict[str, Any]
    formula: str
    n_atoms: int
    preset: Optional[dict[str, Any]] = None
    status: str = "prepared"
    history: list[dict[str, str]] = field(default_factory=list)
    versions: dict[str, Optional[str]] = field(default_factory=code_versions)
    files: dict[str, str] = field(default_factory=dict)
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    relax: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    schema: int = SCHEMA_VERSION
    created: str = field(default_factory=_now)

    def set_status(self, status: str, message: str = "") -> None:
        """Move to ``status`` and log it."""
        if status not in STATES:
            raise ValueError(f"Estado desconocido: '{status}'. Válidos: {STATES}.")
        self.status = status
        self.history.append({"time": _now(), "status": status, "message": message})

    def save(self, directory: Path) -> Path:
        path = Path(directory) / RECORD_FILE
        path.write_text(
            json.dumps(_jsonable(self.__dict__), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: Path) -> "CalcRecord":
        path = Path(directory) / RECORD_FILE
        if not path.exists():
            raise FileNotFoundError(f"No hay {RECORD_FILE} en {directory}.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"{path}: esquema {data.get('schema')}, este carbonforge lee el "
                f"{SCHEMA_VERSION}."
            )
        return cls(**data)

    def summary(self) -> str:
        """Short human-readable report."""
        lines = [
            f"{self.name}: {self.formula} ({self.n_atoms} átomos) — {self.status}",
            f"  {self.spec.get('xc')} / {self.spec.get('mode')}"
            + (f" / {self.spec.get('basis')}" if self.spec.get("mode") == "lcao" else "")
            + (f", ecut={self.spec.get('ecut')} eV" if self.spec.get("mode") == "pw"
               else f", h={self.spec.get('h')} Å")
            + f", fmax={self.spec.get('fmax')} eV/Å",
        ]
        if self.preset:
            lines.append(f"  preset: {self.preset.get('key')}")
        if self.relax:
            lines.append(
                f"  relajación: {self.relax.get('steps')} pasos, fuerza residual "
                f"{self.relax.get('fmax'):.4f} eV/Å"
            )
        if self.results:
            lines.append(
                f"  {len(self.results.get('frequencies_cm1', []))} modos internos, "
                f"{self.results.get('n_rigid')} de sólido rígido descartados"
            )
        for warning in self.results.get("warnings", []):
            lines.append(f"  ⚠️  {warning}")
        if self.error:
            lines.append(f"  error: {self.error.strip().splitlines()[-1]}")
        return "\n".join(lines)


def find_records(root: Path) -> list[Path]:
    """Every calculation directory under ``root``."""
    return sorted(p.parent for p in Path(root).rglob(RECORD_FILE))


def index_records(root: Path, db_path: Path) -> int:
    """Put every record under ``root`` into the ASE database ``db_path``.

    One row per calculation, keyed by its directory relative to ``root``.
    Re-indexing replaces the row, so the database follows the directories
    rather than accumulating stale copies. Returns the number indexed.
    """
    from ase.db import connect
    from ase.io import read

    root = Path(root)
    db = connect(str(db_path))
    count = 0
    for directory in find_records(root):
        record = CalcRecord.load(directory)
        key = directory.relative_to(root).as_posix()
        structure = record.files.get("relaxed") or record.files.get("initial")
        atoms = read(directory / structure)
        atoms.info = {}
        pairs: dict[str, Any] = {
            "run": key,
            "status": record.status,
            "xc": str(record.spec.get("xc")),
            "mode": str(record.spec.get("mode")),
            "preset": (record.preset or {}).get("key") or "",
        }
        if record.relax.get("fmax") is not None:
            pairs["relaxed_fmax"] = float(record.relax["fmax"])
        for row in db.select(run=key):
            del db[row.id]
        db.write(atoms, key_value_pairs=pairs, data={"record": _jsonable(record.__dict__)})
        count += 1
    return count
