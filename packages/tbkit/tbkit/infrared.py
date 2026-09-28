"""Infrared intensities from the tight-binding dipole moment.

The dipole of a finite system in the model is the expectation value of the
same position operator the polarizability uses (:mod:`tbkit.optics`,
:mod:`tbkit.dipoles`):

    μ = Σ_A Q_A (R_A - R_c) - Σ_A p_A        (e·Å)

with Q_A the net Mulliken charges (+ = electrons lost; self-consistent when
the model is SCC) and p_A = Σ_{μν∈A} ρ_μν D_μν the intra-atomic s-p dipoles
of the electrons (zero for a model without ``onsite_dipole``). IR and Raman
therefore come from one consistent response.

Born effective charges ``Z*_{a,ij} = ∂μ_j/∂x_{a,i}`` are central differences
of μ; a mode's IR intensity is ``|∂μ/∂Q_k|² = |Σ_ai Z*_{a,i·} L_k[a,i]|²``,
reported in km/mol (42.2561 km/mol per (D/Å)²/amu, as in ``ase.vibrations``).
Two exact checks are tested: the Born charges of a neutral molecule sum to
zero (moving the whole molecule does not change μ), and symmetry-forbidden
modes have zero intensity.

Scope: finite systems (no Berry-phase polarization for crystals). The
charges are Mulliken charges of a minimal-basis model: intensities are
indicative, relative ones more than absolute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ase import Atoms

from .hamiltonian import System
from .params import TBModel

#: km/mol per (D/Å)²/amu (ase.vibrations.infrared) and D per e·Å.
KM_PER_MOL = 42.2561
DEBYE_PER_EA = 4.80320


def dipole_moment(system: System, kT: float = 0.01, onsite_dipoles: bool = True,
                  tol: float = 1e-10) -> np.ndarray:
    """μ (e·Å) of a finite system in its ground state (SCC when the model is)."""
    from .analysis import atomic_charges
    from .dipoles import atomic_dipoles, dipole_matrices, has_dipoles
    from .kpoints import gamma
    from .scc import self_consistent
    from .solver import solve

    if system.periodic:
        raise ValueError("Momento dipolar: solo sistemas finitos.")
    if system.model.scc:
        result = self_consistent(system, kT=kT, tol=tol)
        if not result.converged:
            raise RuntimeError("SCC sin converger: " + result.summary())
        solution, charges = result.solution, result.charges
    else:
        solution = solve(system, *gamma(), kT=kT)
        charges = atomic_charges(solution)
    positions = system.atoms.get_positions()
    centre = positions.mean(axis=0)
    mu = sum(q * (positions[a] - centre) for a, q in charges.items())
    if onsite_dipoles and has_dipoles(system.model):
        c = np.real(solution.vectors[0, 0])
        density = (c * solution.occupations[0, 0]) @ c.T
        mu = mu - atomic_dipoles(system, density, dipole_matrices(system)).sum(axis=0)
    return np.asarray(mu, dtype=float)


def born_charges(atoms: Atoms, model: TBModel, delta: float = 0.005, kT: float = 0.01,
                 onsite_dipoles: bool = True) -> np.ndarray:
    """Z*[a, i, j] = ∂μ_j/∂x_{a,i} (e), central differences."""
    base = atoms.get_positions()
    probe = atoms.copy()
    probe.calc = None
    out = np.zeros((len(atoms), 3, 3))
    for a in range(len(atoms)):
        for i in range(3):
            moments = []
            for step in (delta, -delta):
                positions = base.copy()
                positions[a, i] += step
                probe.set_positions(positions)
                moments.append(dipole_moment(System.build(probe, model), kT, onsite_dipoles))
            out[a, i] = (moments[0] - moments[1]) / (2 * delta)
    return out


@dataclass
class IRResult:
    frequencies: np.ndarray            # internal modes, cm⁻¹
    intensities: np.ndarray            # km/mol
    derivatives: np.ndarray            # (n, 3) ∂μ/∂Q, e/√amu
    born: np.ndarray                   # (N, 3, 3) e
    dipole: np.ndarray                 # μ at the structure, e·Å
    warnings: list[str] = field(default_factory=list)

    def groups(self, tolerance: float = 1.0, threshold: float = 1e-3) -> list[dict]:
        """Degenerate sets with non-negligible intensity (summed)."""
        order = np.argsort(self.frequencies)
        sets: list[list[int]] = []
        for index in order:
            if sets and self.frequencies[index] - self.frequencies[sets[-1][-1]] <= tolerance:
                sets[-1].append(int(index))
            else:
                sets.append([int(index)])
        strongest = max(self.intensities.max(), 1e-30)
        out = []
        for members in sets:
            total = float(self.intensities[members].sum())
            if total >= threshold * strongest:
                out.append({"frequency_cm1": float(np.mean(self.frequencies[members])),
                            "degeneracy": len(members), "intensity_km_mol": total})
        return out

    def summary(self) -> str:
        lines = [f"IR (dipolo del modelo); μ = {np.linalg.norm(self.dipole) * DEBYE_PER_EA:.3f} D",
                 f"{'cm⁻¹':>9} {'deg':>4} {'km/mol':>10}"]
        for g in self.groups():
            lines.append(f"{g['frequency_cm1']:9.1f} {g['degeneracy']:4d} "
                         f"{g['intensity_km_mol']:10.3g}")
        lines += [f"AVISO: {w}" for w in self.warnings]
        return "\n".join(lines)


def infrared(atoms: Atoms, model: TBModel, kT: float = 0.01, delta: float = 0.005,
             phonon_delta: float = 0.005, phonons: Optional[tuple] = None,
             onsite_dipoles: bool = True) -> IRResult:
    """IR intensities of a finite, relaxed structure under ``model``.

    ``phonons=(frequencies, L)`` takes modes from elsewhere (QE, GPAW); the
    model then gives only the dipole (and needs no repulsive term).
    """
    from .raman import internal_modes, phonons_for

    if atoms.get_pbc().any():
        raise ValueError("IR: solo sistemas finitos (sin polarización de Berry).")
    warnings: list[str] = []
    atoms = atoms.copy()
    frequencies, modes = phonons_for(atoms, model, 12, kT, phonon_delta, phonons, warnings)
    internal = internal_modes(atoms, frequencies, warnings)
    total_charge = 0.0
    z = born_charges(atoms, model, delta, kT, onsite_dipoles)
    residual = np.abs(z.sum(axis=0) - total_charge * np.eye(3)).max()
    if residual > 1e-3:
        warnings.append(f"Regla de la suma de las cargas de Born incumplida en {residual:.1e} e "
                        "(convergencia SCC o paso de diferencias).")
    derivatives = np.einsum("aij,mai->mj", z, modes[internal])
    intensities = np.sum(derivatives ** 2, axis=1) * DEBYE_PER_EA ** 2 * KM_PER_MOL
    mu = dipole_moment(System.build(atoms, model), kT, onsite_dipoles)
    return IRResult(frequencies[internal], intensities, derivatives, z, mu, warnings)


def ir_spectrum(result: IRResult, grid: Optional[np.ndarray] = None, fwhm: float = 10.0
                ) -> tuple[np.ndarray, np.ndarray]:
    """Lorentzian-broadened absorption (km/mol per cm⁻¹)."""
    grid = np.arange(0.0, max(result.frequencies.max(), 100.0) + 200, 1.0) if grid is None \
        else np.asarray(grid, float)
    half = fwhm / 2
    shape = half / np.pi / ((grid[:, None] - result.frequencies[None, :]) ** 2 + half ** 2)
    return grid, shape @ result.intensities
