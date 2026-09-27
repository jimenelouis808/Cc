"""Non-resonant Raman activities from polarizability derivatives.

The Raman activity of a mode is set by how the molecule's polarizability
tensor changes along it. With the modes already known from the IR step
(``modes.npz`` holds, per mode, the Cartesian displacement per unit normal
coordinate, ``L = e / sqrt(m)``), this module:

1. displaces every atom by ``±delta`` along x, y and z (6N geometries,
   central differences) and computes the polarizability ``alpha`` (3x3, Å³)
   of each, caching every result on disk so an interrupted run resumes;
2. builds ``d alpha / d x`` and projects it on each mode,
   ``d alpha / d Q_k = sum_ai (d alpha / d x_ai) L_k[a, i]``;
3. reduces each derivative to the two rotational invariants, the mean
   ``a'`` and the anisotropy ``g'^2``, giving the activity
   ``S = 45 a'^2 + 7 g'^2`` (Å⁴/amu) and the depolarization ratio
   ``rho = 3 g'^2 / (45 a'^2 + 4 g'^2)``.

The activity is what quantum chemistry codes report. A measured intensity
also carries the (ν_laser − ν)⁴ scattering factor and the Bose occupation;
those are applied at analysis time
(:func:`carbonforge.results.spectra.broaden`), never stored.

Where the polarizability comes from is a choice with consequences:

``"field"``
    DFT: the dipole response to a small uniform electric field, ±E along
    each axis (six SCF runs per geometry, 36N in all). The physically
    meaningful option, and the expensive one. Needs a finite model with
    zero boundary conditions (LCAO/FD), like the IR.
``"bond"``
    The Lippincott-Stuttman bond-polarizability model (via ASE): each bond
    contributes a parallel and a perpendicular polarizability that depends
    on its length and elements. Seconds, not days -- but empirical: it knows
    nothing of conjugation, charge transfer or dopant electronic effects,
    so relative intensities are only indicative. Good for a first look at
    which modes are Raman-active; not for assigning intensities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from ase import Atoms

#: ``polarizability(atoms) -> (3, 3)`` in Å³.
PolarizabilityFunction = Callable[[Atoms], np.ndarray]

RAMAN_ACTIVITY_UNIT = "Å⁴/amu"

#: Elements the Lippincott-Stuttman parameters cover (ASE's table).
BOND_MODEL_ELEMENTS = frozenset({"H", "Be", "B", "C", "N", "O", "Al", "Si", "P", "S"})

#: 1 / (4 pi eps0) in eV Å / e²: turns e Å² / V into Å³.
_COULOMB = 14.399645


def bond_polarizability(atoms: Atoms) -> np.ndarray:
    """Lippincott-Stuttman bond-polarizability model, Å³ (empirical)."""
    from ase.calculators.bond_polarizability import BondPolarizability
    from ase.units import Bohr, Ha

    missing = sorted(set(atoms.get_chemical_symbols()) - BOND_MODEL_ELEMENTS)
    if missing:
        raise ValueError(f"El modelo de enlaces no tiene parámetros para {', '.join(missing)}.")
    return np.asarray(BondPolarizability()(atoms), dtype=float) * Bohr * Ha


def field_polarizability(atoms: Atoms, field_v_per_a: float = 0.05) -> np.ndarray:
    """``alpha_ij = d mu_i / d E_j`` by central differences in a uniform field.

    ``atoms.calc`` must accept a uniform external field through
    :func:`carbonforge.vibspec.core.engines.set_uniform_field` (GPAW). The
    field is switched off again before returning.
    """
    from .engines import set_uniform_field

    alpha = np.zeros((3, 3))
    try:
        for j in range(3):
            dipoles = []
            for sign in (1.0, -1.0):
                set_uniform_field(atoms.calc, sign * field_v_per_a, j)
                dipoles.append(np.asarray(atoms.get_dipole_moment(), dtype=float))
            # mu in e Å, field in V/Å -> e Å² / V; times 1/(4 pi eps0) -> Å³.
            alpha[:, j] = (dipoles[0] - dipoles[1]) / (2.0 * field_v_per_a) * _COULOMB
    finally:
        set_uniform_field(atoms.calc, 0.0, 0)
    return 0.5 * (alpha + alpha.T)     # symmetric in exact arithmetic


def _cache_path(cache: Path, atom: int, axis: int, sign: str) -> Path:
    return cache / f"alpha.{atom}{'xyz'[axis]}{sign}.json"


def displacement_count(n_atoms: int) -> int:
    """Geometries the Raman step computes (central differences)."""
    return 6 * n_atoms


def polarizability_derivatives(
    atoms: Atoms,
    polarizability: PolarizabilityFunction,
    cache: Path,
    delta: float = 0.01,
    save: bool = True,
) -> np.ndarray:
    """``d alpha / d x`` for every atom and axis, shape ``(N, 3, 3, 3)``, Å².

    Each displaced geometry's polarizability is read from ``cache`` when
    present, computed and stored otherwise (``save`` is False on MPI ranks
    other than the master). ``atoms`` is left at its original positions.
    """
    cache = Path(cache)
    if save:
        cache.mkdir(parents=True, exist_ok=True)
    base = atoms.get_positions().copy()
    derivative = np.zeros((len(atoms), 3, 3, 3))
    try:
        for atom in range(len(atoms)):
            for axis in range(3):
                pair = []
                for sign, step in (("+", delta), ("-", -delta)):
                    path = _cache_path(cache, atom, axis, sign)
                    if path.exists():
                        pair.append(np.array(json.loads(path.read_text())["alpha"]))
                        continue
                    positions = base.copy()
                    positions[atom, axis] += step
                    atoms.set_positions(positions)
                    alpha = np.asarray(polarizability(atoms), dtype=float)
                    if save:
                        path.write_text(json.dumps({"alpha": alpha.tolist(), "delta": delta}))
                    pair.append(alpha)
                derivative[atom, axis] = (pair[0] - pair[1]) / (2.0 * delta)
    finally:
        atoms.set_positions(base)
    return derivative


def invariants(dalpha_dq: np.ndarray) -> tuple[float, float]:
    """Mean ``a'`` and anisotropy ``g'^2`` of a polarizability derivative."""
    t = 0.5 * (np.asarray(dalpha_dq) + np.asarray(dalpha_dq).T)
    mean = float(np.trace(t) / 3.0)
    gamma2 = 0.5 * ((t[0, 0] - t[1, 1]) ** 2 + (t[1, 1] - t[2, 2]) ** 2
                    + (t[2, 2] - t[0, 0]) ** 2
                    + 6.0 * (t[0, 1] ** 2 + t[1, 2] ** 2 + t[2, 0] ** 2))
    return mean, float(gamma2)


def raman_activities(
    derivative: np.ndarray, modes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Activities (Å⁴/amu) and depolarization ratios for every mode.

    Parameters
    ----------
    derivative
        ``d alpha / d x``, ``(N, 3, 3, 3)`` (atom, displacement axis, tensor).
    modes
        ``(n_modes, N, 3)``: Cartesian displacement per unit normal
        coordinate (ASE's ``e / sqrt(m)``, as stored in ``modes.npz``).
    """
    activities, ratios = [], []
    for mode in np.asarray(modes, dtype=float):
        dalpha_dq = np.einsum("aijk,ai->jk", derivative, mode)
        mean, gamma2 = invariants(dalpha_dq)
        activity = 45.0 * mean ** 2 + 7.0 * gamma2
        denominator = 45.0 * mean ** 2 + 4.0 * gamma2
        activities.append(activity)
        ratios.append(3.0 * gamma2 / denominator if denominator > 1e-12 else 0.0)
    return np.array(activities), np.array(ratios)


def raman_progress(cache: Path, n_atoms: int) -> Optional[str]:
    """``"raman: k/6N polarizabilidades"`` from the cache, or None."""
    cache = Path(cache)
    if not cache.exists():
        return None
    done = len(list(cache.glob("alpha.*.json")))
    return f"raman: {done}/{displacement_count(n_atoms)} polarizabilidades"
