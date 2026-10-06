"""The calculations a simulation file (or the CLI) can ask for, as plain functions.

Each takes ``(atoms, model, **settings)`` and returns ``(results, atoms)``:
a JSON-ready dict and the final structure (changed only by ``relax``).
Settings not given take the defaults written here, and the record stores the
settings as given, so a replay uses the same values.
"""

from __future__ import annotations

import os
import tempfile

from typing import Optional

import numpy as np
from ase import Atoms

from .analysis import atomic_charges, bands, dos, spin_moments
from .hamiltonian import System
from .kpoints import band_path, gamma, mesh
from .params import TBModel
from .solver import solve


def _kpts(system: System, kmesh: int):
    return mesh(system.atoms, kmesh) if system.periodic else gamma()


def levels(atoms: Atoms, model: TBModel, kmesh: int = 24, kT: float = 0.01,
           charge: float = 0.0, scc: Optional[bool] = None):
    system = System.build(atoms, model)
    if model.scc if scc is None else scc:
        from .scc import self_consistent

        result = self_consistent(system, charge=charge, kT=kT)
        solution, charges = result.solution, result.charges
        extra = {"scc_iterations": result.iterations, "scc_converged": result.converged}
    else:
        solution = solve(system, *_kpts(system, kmesh), charge=charge, kT=kT)
        charges, extra = atomic_charges(solution), {}
    homo, lumo = solution.homo_lumo()
    return {"homo": homo, "lumo": lumo, "gap": solution.gap(), "fermi": solution.fermi,
            "electrons": solution.electrons, "charges": charges,
            "levels": np.sort(solution.energies.ravel()), **extra}, atoms


def density_of_states(atoms: Atoms, model: TBModel, kmesh: int = 24, kT: float = 0.01,
                      sigma: float = 0.05, points: int = 2000):
    system = System.build(atoms, model)
    solution = solve(system, *_kpts(system, kmesh), kT=kT)
    grid = np.linspace(solution.energies.min() - 1, solution.energies.max() + 1, points)
    grid, total = dos(solution, grid, sigma)
    return {"energy_minus_fermi": grid - solution.fermi, "dos": total,
            "fermi": solution.fermi}, atoms


def band_structure(atoms: Atoms, model: TBModel, path: str | None = None, npoints: int = 200):
    system = System.build(atoms, model)
    if not system.periodic:
        raise ValueError("Bandas: la estructura no es periódica.")
    bp = band_path(atoms, path, npoints)
    x, special_x, labels = bp.get_linear_kpoint_axis()
    return {"k": x, "special_k": special_x, "labels": labels,
            "energies": bands(system, bp)[0]}, atoms


def hubbard(atoms: Atoms, model: TBModel, U: float | None = None, kmesh: int = 24,
            kT: float = 0.005, guess: str = "antiferro", field: float = 0.0,
            charge: float = 0.0, mixing: float = 0.4, tol: float = 1e-7):
    from .hubbard import magnetization_vs_energy, mean_field

    system = System.build(atoms, model)
    k, w = _kpts(system, kmesh)
    result = mean_field(system, U=U, kpts=k, weights=w, kT=kT, guess=guess, field=field,
                        charge=charge, mixing=mixing, tol=tol)
    energy, m, dm = magnetization_vs_energy(result, sigma=0.05)
    return {"magnetization": result.magnetization, "moments": spin_moments(result.solution),
            "energy": result.energy, "converged": result.converged,
            "iterations": result.iterations, "fermi": result.solution.fermi,
            "m_of_E": {"energy_minus_fermi": energy[::10], "m": m[::10], "dm_dE": dm[::10]}
            }, atoms


def relax(atoms: Atoms, model: TBModel, kmesh: int = 8, kT: float = 0.02, fmax: float = 0.01,
          steps: int = 500, scc: Optional[bool] = None):
    """BFGS relaxation of positions (cell fixed) with the model's forces."""
    from ase.optimize import BFGS

    from .calculator import TBCalculator

    atoms = atoms.copy()
    atoms.calc = TBCalculator(model, kpts=kmesh, kT=kT, scc=scc)
    optimizer = BFGS(atoms, logfile=None)
    converged = bool(optimizer.run(fmax=fmax, steps=steps))
    forces = atoms.get_forces()
    return {"converged": converged, "steps": optimizer.get_number_of_steps(),
            "energy": atoms.get_potential_energy(),
            "max_force": float(np.linalg.norm(forces, axis=1).max())}, atoms


def phonons(atoms: Atoms, model: TBModel, kmesh: int = 12, kT: float = 0.02,
            delta: float = 0.005, scc: Optional[bool] = None):
    """Γ-point phonons (finite differences of the forces, via ase.vibrations).

    For a periodic cell these are the Γ modes of the crystal (the ones Raman
    and IR probe); for a molecule, its normal modes. Relax first: at a
    non-equilibrium geometry the frequencies are not the harmonic ones.
    """
    from ase.vibrations import Vibrations

    from .calculator import TBCalculator

    atoms = atoms.copy()
    atoms.calc = TBCalculator(model, kpts=kmesh, kT=kT, scc=scc)
    residual = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    from .modes import GAPLESS_WARNING, gapless_periodic

    gapless = gapless_periodic(atoms)
    with tempfile.TemporaryDirectory() as directory:
        vibrations = Vibrations(atoms, name=os.path.join(directory, "vib"), delta=delta)
        vibrations.run()
        data = vibrations.get_vibrations()
        energies, modes = data.get_energies_and_modes(all_atoms=True)
    from ase.units import invcm

    frequencies = np.where(np.abs(energies.imag) > np.abs(energies.real),
                           -np.abs(energies.imag), np.abs(energies.real)) / invcm
    out = {"frequencies_cm1": frequencies, "residual_force": residual, "modes": modes,
           "warnings": [GAPLESS_WARNING] if gapless else []}
    scale = frequency_scale(model)
    if scale is not None:
        # Against GPAW, from the set's own molecules (recipes/frequency_scaling.py);
        # the raw frequencies stay what the model gives.
        out["frequency_scale"] = scale
        out["frequencies_scaled_cm1"] = scale * frequencies
    return out, atoms


def kmesh_convergence(atoms: Atoms, model: TBModel, meshes=(12, 24, 36, 48), kT: float = 0.02,
                      top: int = 6, tolerance: float = 2.0, scc: Optional[bool] = None) -> dict:
    """Γ frequencies against the k mesh, densest last; stops once the ``top`` highest
    modes move less than ``tolerance`` cm⁻¹ between two meshes.

    Needed for gapless crystals (see ``modes.GAPLESS_WARNING``): a coarse mesh can
    soften the modes that couple to the Fermi surface by over 100 cm⁻¹."""
    from .modes import vibrations

    rows, previous, converged = [], None, None
    for n in meshes:
        frequencies = np.sort(vibrations(atoms, model, kmesh=n, kT=kT, scc=scc).frequencies)
        highest = frequencies[-top:]
        change = None if previous is None else float(np.abs(highest - previous).max())
        rows.append({"kmesh": n, "highest_cm1": highest.tolist(), "max_change_cm1": change})
        if change is not None and change < tolerance:
            converged = rows[-2]["kmesh"]
            break
        previous = highest
    return {"rows": rows, "converged_kmesh": converged, "kT": kT, "tolerance_cm1": tolerance}


def frequency_scale(model: TBModel) -> float | None:
    """The set's frequency scale factor (``frequency_scale`` in its file), or None."""
    entry = model.metadata.get("parameters", {}).get("frequency_scale")
    return float(entry["value"]) if entry else None


def raman_task(atoms: Atoms, model: TBModel, kmesh: int = 12, kT: float = 0.01,
               delta: float = 0.01, screening: str = "auto"):
    """Non-resonant Raman (Γ modes, activities, depolarization); see tbkit.raman."""
    from .raman import raman

    result = raman(atoms, model, kmesh=kmesh, kT=kT, delta=delta, screening=screening)
    return {"method": result.method, "alpha": result.alpha, "groups": result.groups(),
            "frequencies_cm1": result.frequencies, "activities": result.activities,
            "depolarization": result.depolarization, "warnings": result.warnings}, atoms


TASKS = {"levels": levels, "dos": density_of_states, "bands": band_structure,
         "hubbard": hubbard, "relax": relax, "phonons": phonons, "raman": raman_task}
