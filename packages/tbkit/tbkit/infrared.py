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


# --------------------------------------------------------------------------
# Wires and slabs: the dipole along the open (non-periodic) directions
# --------------------------------------------------------------------------

def open_dipole(atoms: Atoms, calc) -> np.ndarray:
    """μ (e·Å) along the non-periodic axes of a wire or slab, NaN along periodic ones.

    ``calc`` is a :class:`tbkit.calculator.TBCalculator` that has just computed
    ``atoms``: its Mulliken charges (self-consistent with an SCC model) and, if the
    model has them, the intra-atomic s-p dipoles from the density summed over k.
    Along an open axis the position is single-valued (no atom is wrapped there), so
    ``Σ Q_A x_A`` is the dipole of the cell; along a periodic axis it is not (it
    would need the Berry phase), and is left out. In a gapless model the axial
    response is metallic anyway; the transverse one stays finite."""
    from .dipoles import atomic_dipoles, has_dipoles

    pbc = np.asarray(atoms.get_pbc(), bool)
    charges = calc.results["charges"]
    positions = atoms.get_positions()
    mu = (charges[:, None] * (positions - positions.mean(axis=0))).sum(axis=0)
    solution = calc.last_solution
    if has_dipoles(calc.model):
        density = np.zeros(solution.vectors.shape[2:3] * 2)
        for s in range(solution.nspin):
            for k, w in enumerate(solution.weights):
                c = solution.vectors[s, k]
                density = density + w * np.real((c * solution.occupations[s, k]) @ c.conj().T)
        mu = mu - atomic_dipoles(solution.system, density).sum(axis=0)
    return np.where(pbc, np.nan, mu)


def born_rows(atoms: Atoms, make_calc, folder, indices=None, delta: float = 0.005,
              part: tuple = (0, 1)) -> np.ndarray | None:
    """Born charges ``Z*[a, i, j] = ∂μ_j/∂x_{a,i}`` (e) of ``indices`` for a periodic
    or finite structure, j along the open axes (NaN along periodic ones), with the
    dipole of each ±δ displacement cached in ``folder`` (``z_A_C_S.npy``): resumable,
    and ``part=(r, n)`` shares the work as :func:`tbkit.modes.hessian_rows` does.
    Returns (len(indices), 3, 3), or None from a part that stops early."""
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(atoms))) if indices is None else [int(i) for i in indices]
    base = atoms.get_positions()
    calc = None                      # one per process: the SCC starts from the last charges
    jobs = [(a, c, s) for a in indices for c in range(3) for s in (1, -1)]
    for k, (a, c, sign) in enumerate(jobs):
        if k % part[1] != part[0]:
            continue
        path = folder / f"z_{a:04d}_{c}_{'p' if sign > 0 else 'm'}.npy"
        if path.exists():
            continue
        probe = atoms.copy()
        positions = base.copy()
        positions[a, c] += sign * delta
        probe.set_positions(positions)
        calc = calc or make_calc()
        probe.calc = calc
        probe.get_potential_energy()
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, np.concatenate([open_dipole(probe, calc), calc.results["charges"]]))
        tmp.replace(path)
    out = np.zeros((len(indices), 3, 3))
    for r, a in enumerate(indices):
        for c in range(3):
            pair = [folder / f"z_{a:04d}_{c}_{t}.npy" for t in ("p", "m")]
            if not all(x.exists() for x in pair):
                return None
            out[r, c] = (np.load(pair[0])[:3] - np.load(pair[1])[:3]) / (2 * delta)
    return out


def mode_intensities(born: np.ndarray, modes: np.ndarray, sum_rule: bool = True
                     ) -> tuple[np.ndarray, dict]:
    """IR intensities (km/mol) of ``modes`` (n, N, 3; L as in ``Vibrations.modes``)
    from Born charges (N, 3, 3), summed over the axes where Z* is finite. With
    ``sum_rule`` the residual ``Σ_a Z*_a`` (zero for a neutral cell: a rigid
    translation does not change μ) is removed evenly from every atom first; its
    size is returned as a check of the finite differences (or of an embedding)."""
    axes = [j for j in range(3) if np.isfinite(born[:, :, j]).all()]
    z = born[:, :, axes]
    residual = z.sum(axis=0)
    if sum_rule:
        z = z - residual[None] / len(z)
    derivatives = np.einsum("aij,mai->mj", z, modes)
    intensities = np.sum(derivatives ** 2, axis=1) * DEBYE_PER_EA ** 2 * KM_PER_MOL
    return intensities, {"axes": axes, "sum_rule_residual_e": float(np.abs(residual).max())}
