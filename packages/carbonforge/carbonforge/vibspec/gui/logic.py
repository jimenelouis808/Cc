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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.io import read

from ...builders.nanoribbon import DEFAULT_VACUUM_PER_SIDE, build_finite_nanoribbon
# The job queue is carbonforge's (carbonforge.jobs), shared with QE, SIESTA
# and LAMMPS jobs; re-exported here so vibspec's API does not change.
from ...jobs.queue import (  # noqa: F401
    CANCELLED,
    DONE,
    ERROR,
    QUEUED,
    RUNNING,
    CommandBuilder,
    Job,
    JobQueue,
    VibspecAdapter,
    job_log,
    tail,
)
from ...jobs.queue import gpaw_command as default_command  # noqa: F401
from ...gui.params import ParamSpec, collect_values
# Mode helpers are shared with QE's modes on the Resultados page.
from ...results.modes import (  # noqa: F401
    mode_character,
    mode_frames,
    normalised,
    view_angles,
)
from ...validation.checks import ValidationReport
from ..core import (
    PRESETS,
    CalcRecord,
    CalcSpec,
    SpinAdvice,
    apply_preset,
    check_structure,
    gpaw_available,
    list_library,
    load_atoms,
    load_structure,
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
    """A built or loaded structure and what the checks say about it."""

    atoms: Atoms
    report: ValidationReport
    spin: SpinAdvice
    import_report: str = ""

    def summary(self) -> str:
        preset = self.atoms.info.get("vibspec_preset", {}).get("key", "pristine")
        lines = [f"{self.atoms.get_chemical_formula()} — {len(self.atoms)} átomos — "
                 f"preset: {preset}",
                 f"Espín recomendado: {'sí' if self.spin.spinpol else 'no'}"]
        # Plain-text markers: they render with any Tk font.
        lines += [f"ERROR: {e}" for e in self.report.errors]
        lines += [f"AVISO: {w}" for w in self.report.warnings]
        if self.import_report:
            lines += ["", "--- Archivo cargado ---", self.import_report]
        return "\n".join(lines)


def build_model(raw: dict[str, Any], source: Optional[Path] = None,
                atoms: Optional[Atoms] = None, label: str = "estructura actual") -> ModelResult:
    """Build (or load) the ribbon, apply the preset and check it.

    Parameters
    ----------
    raw
        Raw form values (:data:`BUILDER_PARAMS`).
    source
        A structure file to start from instead of building a ribbon. Its
        atoms and groups are kept; the form's edge, width and length are
        ignored, its vacuum and preset are applied (``pristine`` adds
        nothing).
    atoms, label
        A structure from another part of carbonforge (the builder's current
        structure), treated like a file: same checks and refusals. Takes
        precedence over ``source``; the caller's atoms are not modified.
    """
    values = collect_values(BUILDER_PARAMS, raw)
    import_report = ""
    if atoms is not None:
        atoms, import_report = load_atoms(atoms, values["vacuum_per_side"], label=label)
    elif source is not None:
        atoms, import_report = load_structure(source, vacuum_per_side=values["vacuum_per_side"])
    else:
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
    return ModelResult(atoms, check_structure(atoms), suggest_spin(atoms), import_report)


def library(directory: Path) -> list[Path]:
    """The structure files in the user's library folder."""
    return list_library(directory)


def save_to_library(atoms: Atoms, directory: Path, name: str) -> Path:
    """Save a model into the library folder as extended XYZ; returns the path.

    Extended XYZ keeps the model's provenance (preset, site, source file) in
    the comment line, so reloading it later restores more than coordinates.
    Refuses to overwrite: a library entry is something you chose to keep.
    """
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "modelo"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}.xyz"
    if path.exists():
        raise FileExistsError(f"Ya hay un modelo llamado {path.name} en la biblioteca.")
    atoms.write(path, format="extxyz")
    return path


def spec_from_form(raw: dict[str, Any], advanced: Optional[dict[str, Any]] = None) -> CalcSpec:
    """A :class:`CalcSpec` from raw form values (bounds checked, Spanish errors).

    ``advanced`` holds extra GPAW keywords as raw text (``{"maxiter": "500"}``),
    checked against the GPAW catalogue; an invalid one raises ``ValueError``.
    """
    from ...codes import load_catalog

    values = collect_values(CALC_PARAMS, raw)
    values["spinpol"] = {AUTO: None, "sí": True, "no": False}[values["spinpol"]]
    values["nfree"] = int(values["nfree"])
    if advanced:
        typed, report = load_catalog("gpaw").check(advanced)
        if report.errors:
            raise ValueError("Parámetros avanzados de GPAW:\n" + "\n".join(report.errors))
        values["extra"] = typed
    return CalcSpec(**values)


def job_name(atoms: Atoms) -> str:
    """A readable, filesystem-safe default name: ``amine_armchair_5x3``.

    A loaded structure is named after its file: ``mi_cinta_amine``.
    """
    info = atoms.info
    preset = info.get("vibspec_preset", {}).get("key", "modelo")
    if "source_file" in info:
        stem = Path(info["source_file"]).stem
        name = stem if preset == "pristine" else f"{stem}_{preset}"
    else:
        name = (f"{preset}_{info.get('edge', 'gnr')}_"
                f"{info.get('width', '')}x{info.get('length', '')}")
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

def can_run_locally() -> tuple[bool, str]:
    """Whether this machine can run calculations, and why not if it cannot."""
    if gpaw_available():
        return True, ""
    return False, ("GPAW no está instalado aquí (lo normal en Windows). Prepara los "
                   "cálculos y córrelos en Ubuntu con run.py; luego ábrelos en Resultados.")


def progress(directory: Path) -> str:
    """One line on how far a calculation has got, read from its files."""
    return VibspecAdapter().progress(Path(directory))


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
    return normalised(modes["modes"][mode_index])
