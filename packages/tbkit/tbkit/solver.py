"""Diagonalise, fill with electrons, and keep everything the analysis needs.

``solve`` returns a :class:`Solution` holding, for every spin channel and k
point, the eigenvalues, the eigenvectors and the occupations, plus the
Fermi level. Occupations follow a Fermi-Dirac distribution of width
``kT`` (eV); a small width keeps degenerate levels of a molecule equally
filled instead of breaking the symmetry arbitrarily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

from .hamiltonian import System
from .kpoints import gamma, mesh


@dataclass
class Solution:
    """Eigenstates of a system.

    Shapes: ``energies[spin, k, band]``, ``vectors[spin, k, orbital, band]``,
    ``occupations[spin, k, band]`` (0..1 per spin channel, or 0..2 when
    ``nspin == 1``), ``weights[k]``.
    """

    system: System
    kpts: np.ndarray
    weights: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray
    occupations: np.ndarray
    fermi: float
    electrons: float
    kT: float
    overlaps: Optional[np.ndarray] = None          # S(k) per k, or None
    info: dict = field(default_factory=dict)

    @property
    def nspin(self) -> int:
        return self.energies.shape[0]

    def homo_lumo(self) -> tuple[float, float]:
        """Highest (mostly) occupied and lowest (mostly) empty level, over k and spin."""
        full = 2.0 if self.nspin == 1 else 1.0
        occupied = self.energies[self.occupations > 0.5 * full]
        empty = self.energies[self.occupations <= 0.5 * full]
        return float(occupied.max()), float(empty.min())

    def gap(self) -> float:
        homo, lumo = self.homo_lumo()
        return max(0.0, lumo - homo)

    def band_energy(self) -> float:
        return float(np.einsum("k,skb,skb->", self.weights, self.occupations, self.energies))


def _fermi_dirac(e: np.ndarray, mu: float, kT: float) -> np.ndarray:
    x = np.clip((e - mu) / kT, -200, 200)
    return 1.0 / (np.exp(x) + 1.0)


def fermi_level(energies: np.ndarray, weights: np.ndarray, electrons: float, kT: float,
                spin_degeneracy: float) -> float:
    """Chemical potential giving ``electrons`` (energies[spin, k, band])."""
    def count(mu):
        return float(spin_degeneracy * np.einsum(
            "k,skb->", weights, _fermi_dirac(energies, mu, kT))) - electrons
    low, high = energies.min() - 10 * kT - 1.0, energies.max() + 10 * kT + 1.0
    capacity = spin_degeneracy * energies.shape[0] * energies.shape[2]
    if not 0 <= electrons <= capacity:
        raise ValueError(f"{electrons} electrones no caben en {capacity:g} estados.")
    return brentq(count, low, high, xtol=1e-12)


def diagonalise(system: System, kpts: np.ndarray,
                extra_onsite: Optional[np.ndarray] = None):
    """Eigenvalues, eigenvectors and overlaps at every k; one spin channel."""
    energies, vectors, overlaps = [], [], []
    for k in kpts:
        h, s = system.hamiltonian(k, extra_onsite)
        if s is not None:
            smallest = float(np.linalg.eigvalsh(s).min())
            if smallest <= 1e-8:
                raise ValueError(
                    f"S(k={tuple(np.round(k, 4))}) no es definida positiva (autovalor mínimo "
                    f"{smallest:.2e}): átomos demasiado cerca o solapamientos fuera del rango "
                    "de validez de los parámetros.")
        e, c = eigh(h, s) if s is not None else eigh(h)
        energies.append(e)
        vectors.append(c)
        overlaps.append(s)
    return (np.array(energies), np.array(vectors),
            None if system.model.orthogonal else np.array(overlaps))


def solve(system: System, kpts=None, weights=None, charge: float = 0.0, kT: float = 0.01,
          spin_potentials: Optional[tuple[np.ndarray, np.ndarray]] = None,
          fixed_fermi: Optional[float] = None) -> Solution:
    """Solve a system.

    Parameters
    ----------
    system
        From :meth:`System.build`.
    kpts, weights
        Fractional k points and weights. Default: Γ for a finite system, a
        12-point-per-axis mesh for a periodic one.
    charge
        Net charge in units of e (positive = electrons removed).
    kT
        Fermi-Dirac width, eV.
    spin_potentials
        Two on-site potentials (per orbital), for spin up and down: gives a
        spin-polarised solution (``nspin = 2``). Used by the Hubbard model.
    fixed_fermi
        Fill up to this chemical potential instead of fixing the electron
        count (grand canonical: magnetisation against Fermi level).
    """
    if kpts is None:
        kpts, weights = mesh(system.atoms) if system.periodic else gamma()
    kpts = np.atleast_2d(np.asarray(kpts, dtype=float))
    weights = np.full(len(kpts), 1.0 / len(kpts)) if weights is None else np.asarray(weights)
    channels = [None] if spin_potentials is None else list(spin_potentials)
    results = [diagonalise(system, kpts, extra) for extra in channels]
    energies = np.array([r[0] for r in results])
    vectors = np.array([r[1] for r in results])
    overlaps = results[0][2]
    degeneracy = 2.0 if spin_potentials is None else 1.0
    electrons = system.electrons - charge
    if fixed_fermi is None:
        mu = fermi_level(energies, weights, electrons, kT, degeneracy)
    else:
        mu = float(fixed_fermi)
    occupations = degeneracy * _fermi_dirac(energies, mu, kT)
    electrons = float(np.einsum("k,skb->", weights, occupations))
    return Solution(system, kpts, weights, energies, vectors, occupations, mu, electrons, kT,
                    overlaps)
