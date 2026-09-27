"""The calculations a simulation file (or the CLI) can ask for, as plain functions.

Each takes ``(atoms, model, **settings)`` and returns ``(results, atoms)``:
a JSON-ready dict and the final structure (changed only by ``relax``).
Settings not given take the defaults written here, and the record stores the
settings as given, so a replay uses the same values.
"""

from __future__ import annotations

import os
import tempfile

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
           charge: float = 0.0, scc: bool = False):
    system = System.build(atoms, model)
    if scc:
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
          steps: int = 500, scc: bool = False):
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
            delta: float = 0.005, scc: bool = False):
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
    with tempfile.TemporaryDirectory() as directory:
        vibrations = Vibrations(atoms, name=os.path.join(directory, "vib"), delta=delta)
        vibrations.run()
        data = vibrations.get_vibrations()
        energies, modes = data.get_energies_and_modes(all_atoms=True)
    from ase.units import invcm

    frequencies = np.where(np.abs(energies.imag) > np.abs(energies.real),
                           -np.abs(energies.imag), np.abs(energies.real)) / invcm
    return {"frequencies_cm1": frequencies, "residual_force": residual,
            "modes": modes}, atoms


TASKS = {"levels": levels, "dos": density_of_states, "bands": band_structure,
         "hubbard": hubbard, "relax": relax, "phonons": phonons}
