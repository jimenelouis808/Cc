"""Core-level (XPS) binding-energy shifts by the equivalent-core (Z+1) approximation.

A tight-binding model has no core orbitals, but the valence of an atom with a 1s hole
behaves like that of the next element: a C 1s hole is, for the valence electrons, an N
nucleus with one electron less (Jolly & Hendrickson's equivalent cores). The core
ionisation energy of atom A is then, up to a constant of the element,

    E_B(A) ≈ E[structure with A → Z+1, charge +1] − E[structure]  (+ const)

both total energies of the model, with the self-consistent charges relaxing around the
substituted atom: the final-state screening a ΔSCF calculation has, and that an
initial-state (charge or potential) estimate misses. Only differences are meaningful, so
the result is a shift against a reference site of the same element computed the same way
(CH4 for C 1s, NH3 for N 1s, or any site of one's own).

Needs a set with the Z+1 element and charge self-consistency (``TBModel.scc``): C 1s with
xu_chn or xu_chno (C → N), N 1s with xu_chno (N → O). Validated against GPAW ΔSCF with
full core-hole setups in ``recipes/xps_validation``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from ase import Atoms

#: The equivalent core of each element (Z + 1).
Z_PLUS_ONE = {"H": "He", "B": "C", "C": "N", "N": "O", "O": "F", "P": "S", "S": "Cl"}


@dataclass
class CoreLevel:
    index: int
    element: str
    energy_eV: float              # E(Z+1, +1) − E(ground), model total energies
    shift_eV: Optional[float]     # against the reference, when one is given


def _energy(atoms: Atoms, model, charge: float, kT: float, kpts=None) -> float:
    from .hamiltonian import System
    from .scc import energy_and_forces, self_consistent

    system = System.build(atoms, model)
    weights = None
    if system.periodic and kpts is not None:
        from .kpoints import mesh

        kpts, weights = mesh(system, kpts)
    result = self_consistent(system, charge=charge, kT=kT, tol=1e-9, max_iter=1500,
                             kpts=kpts if system.periodic else None, weights=weights)
    if not result.converged:
        raise RuntimeError("SCC sin converger: " + result.summary())
    return float(energy_and_forces(result, need_forces=False)[0])


def ionisation_energy(atoms: Atoms, model, index: int, kT: float = 0.01, kpts=None,
                      ground: Optional[float] = None) -> float:
    """``E[Z+1 at index, charge +1] − E[ground]`` (eV) with the model."""
    if not model.scc:
        raise ValueError(f"El modelo '{model.name}' no tiene cargas autoconsistentes: el "
                         "núcleo equivalente necesita que la valencia se relaje (SCC).")
    element = atoms[index].symbol
    equivalent = Z_PLUS_ONE.get(element)
    if equivalent is None or equivalent not in model.orbitals:
        raise ValueError(f"El modelo '{model.name}' no tiene {equivalent or '?'}, el núcleo "
                         f"equivalente de {element}: no puede dar {element} 1s.")
    if ground is None:
        ground = _energy(atoms, model, 0.0, kT, kpts)
    ion = atoms.copy()
    ion.symbols[index] = equivalent
    return _energy(ion, model, 1.0, kT, kpts) - ground


def xps_scale(set_name: str, element: str) -> float:
    """λ of ``parameters/xps_scale.json`` for a set and element (fitted to GPAW ΔSCF on half
    the sites of ``recipes/xps_validation``, tested on the rest)."""
    from .params import read_parameter_file

    return float(read_parameter_file("xps_scale")["scales"][set_name][element]["value"])


def core_levels(atoms: Atoms, model, indices: Sequence[int], kT: float = 0.01, kpts=None,
                reference: Optional[tuple[Atoms, int]] = None,
                scale: float = 1.0) -> list[CoreLevel]:
    """Z+1 ionisation energies of ``indices`` and, with ``reference`` (structure, index),
    their shifts against it (same element only), times ``scale`` (1: the raw model shift;
    :func:`xps_scale` for the one calibrated against GPAW)."""
    ground = _energy(atoms, model, 0.0, kT, kpts)
    ref_energy = ref_element = None
    if reference is not None:
        ref_atoms, ref_index = reference
        ref_element = ref_atoms[ref_index].symbol
        ref_energy = ionisation_energy(ref_atoms, model, ref_index, kT)
    out = []
    for i in indices:
        e = ionisation_energy(atoms, model, i, kT, kpts, ground)
        shift = scale * (e - ref_energy) if ref_energy is not None and \
            atoms[i].symbol == ref_element else None
        out.append(CoreLevel(int(i), atoms[i].symbol, float(e), shift))
    return out


def spectrum(levels: Sequence[CoreLevel], fwhm: float = 0.8,
             grid: Optional[np.ndarray] = None, key: str = "shift_eV"):
    """A Gaussian-broadened spectrum of the shifts (or energies): ``(grid, intensity)``."""
    values = np.array([getattr(lv, key) for lv in levels], dtype=float)
    if grid is None:
        grid = np.linspace(values.min() - 3 * fwhm, values.max() + 3 * fwhm, 600)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    intensity = np.exp(-0.5 * ((grid[:, None] - values[None]) / sigma) ** 2).sum(axis=1)
    return grid, intensity
