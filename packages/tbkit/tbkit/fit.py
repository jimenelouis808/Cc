"""Fit tight-binding parameters to reference levels (DFT, experiment).

The reference is a set of one-electron levels per structure: GPAW's
(:func:`read_gpaw_eigenvalues` reads them from a ``gpaw.txt``, e.g. of a
vibspec calculation), or any array. A DFT calculation has many more levels
than a π model, so the comparison uses a window around the gap: the
``n_below`` highest occupied and ``n_above`` lowest empty levels of each,
aligned at their own mid-gap (absolute DFT energies have an arbitrary
zero; ``Reference.align = "none"`` compares absolute energies instead). Pick the window so it holds only states the model can describe (the
π levels for a π model: check the DFT orbitals).

``fit`` minimises the weighted residuals with ``scipy.optimize.least_squares``
over whatever parameters ``build(x) -> TBModel`` exposes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from ase import Atoms
from scipy.optimize import least_squares

from .hamiltonian import System
from .params import TBModel
from .solver import solve


@dataclass
class Reference:
    """Levels of one structure to fit against."""

    atoms: Atoms
    levels: np.ndarray                 # eV, all levels (any zero)
    n_occupied: int                    # how many of them are occupied (per spin channel)
    n_below: int = 3
    n_above: int = 3
    weight: float = 1.0
    label: str = ""
    #: ``"midgap"``: compare levels relative to each side's own mid-gap (DFT
    #: absolute energies have an arbitrary zero), which leaves on-site
    #: energies determined only up to such a shift and can admit mirror
    #: solutions. ``"none"``: absolute, when reference and model share a zero.
    align: str = "midgap"

    def window(self) -> np.ndarray:
        levels = np.sort(np.asarray(self.levels, dtype=float))
        homo, lumo = levels[self.n_occupied - 1], levels[self.n_occupied]
        mid = 0.5 * (homo + lumo) if self.align == "midgap" else 0.0
        below = levels[self.n_occupied - self.n_below:self.n_occupied]
        above = levels[self.n_occupied:self.n_occupied + self.n_above]
        if len(below) < self.n_below or len(above) < self.n_above:
            raise ValueError(f"{self.label}: la referencia no tiene {self.n_below} niveles "
                             f"ocupados y {self.n_above} vacíos alrededor del gap.")
        return np.concatenate([below, above]) - mid


def model_window(model: TBModel, reference: Reference) -> np.ndarray:
    """The model's levels in the same window, aligned the same way."""
    system = System.build(reference.atoms, model)
    solution = solve(system, kT=1e-4)
    levels = np.sort(solution.energies[0, 0])
    occupied = int(round(system.electrons / 2))
    probe = Reference(reference.atoms, levels, occupied, reference.n_below, reference.n_above,
                      label=f"modelo ({reference.label})", align=reference.align)
    return probe.window()


@dataclass
class FitResult:
    x: np.ndarray
    names: list[str]
    model: TBModel
    rms: dict[str, float]
    cost: float
    success: bool
    message: str
    residuals: np.ndarray = field(repr=False, default=None)

    def summary(self) -> str:
        params = ", ".join(f"{n} = {v:.4f}" for n, v in zip(self.names, self.x, strict=True))
        errors = ", ".join(f"{k}: {v * 1000:.1f} meV" for k, v in self.rms.items())
        return f"Ajuste {'ok' if self.success else 'FALLIDO'}: {params}. RMS: {errors}"


def fit(build: Callable[[np.ndarray], TBModel], x0: Sequence[float],
        references: Sequence[Reference], names: Optional[Sequence[str]] = None,
        bounds=(-np.inf, np.inf)) -> FitResult:
    """Fit ``build(x)`` to the references; returns the best parameters and model."""
    references = list(references)
    targets = [ref.window() for ref in references]

    def residuals(x):
        model = build(np.asarray(x))
        return np.concatenate([np.sqrt(ref.weight) * (model_window(model, ref) - target)
                               for ref, target in zip(references, targets, strict=True)])

    result = least_squares(residuals, np.asarray(x0, dtype=float), bounds=bounds)
    model = build(result.x)
    rms = {}
    for i, (ref, target) in enumerate(zip(references, targets, strict=True)):
        diff = model_window(model, ref) - target
        rms[ref.label or f"ref{i}"] = float(np.sqrt(np.mean(diff ** 2)))
    return FitResult(result.x, list(names or [f"x{i}" for i in range(len(x0))]), model, rms,
                     float(result.cost), bool(result.success), str(result.message),
                     result.fun)


# --------------------------------------------------------------------------
# GPAW output
# --------------------------------------------------------------------------

@dataclass
class GPAWLevels:
    energies: np.ndarray               # [spin, band], eV
    occupations: np.ndarray            # [spin, band]
    fermi: Optional[float]

    def n_occupied(self, spin: int = 0) -> int:
        full = 2.0 if self.energies.shape[0] == 1 else 1.0
        return int(np.sum(self.occupations[spin] > 0.5 * full))


def read_gpaw_eigenvalues(path: str | Path) -> GPAWLevels:
    """The last eigenvalue table of a GPAW text log (``gpaw.txt``).

    Handles the spin-paired table (``Band Eigenvalues Occupancy``) and the
    spin-polarised one (two pairs of columns, up then down). Γ-point /
    finite systems: for several k points only the first table printed
    after the last SCF is read.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    starts = [i for i, ln in enumerate(lines)
              if re.match(r"^\s*Band\s+Eigenvalues\s+Occupancy", ln)]
    if not starts:
        raise ValueError(f"{Path(path).name}: no hay tabla 'Band Eigenvalues Occupancy'.")
    rows = []
    for line in lines[starts[-1] + 1:]:
        parts = line.split()
        if not parts or not parts[0].isdigit():
            break
        rows.append([float(x) for x in parts[1:]])
    data = np.array(rows)
    if data.shape[1] >= 4:
        energies = np.array([data[:, 0], data[:, 2]])
        occupations = np.array([data[:, 1], data[:, 3]])
    else:
        energies, occupations = data[None, :, 0], data[None, :, 1]
    fermi = None
    for line in reversed(lines):
        match = re.search(r"Fermi levels?:\s*([-\d.]+)", line)
        if match:
            fermi = float(match.group(1))
            break
    return GPAWLevels(energies, occupations, fermi)
