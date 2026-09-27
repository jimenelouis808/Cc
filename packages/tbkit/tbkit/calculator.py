"""tbkit as an ASE calculator: relax with ASE's optimisers, vibrate with ase.vibrations.

    from tbkit.calculator import TBCalculator
    atoms.calc = TBCalculator(xu_carbon(), kpts=8)
    BFGS(atoms).run(fmax=0.01)

Properties: ``energy`` and ``free_energy`` (both the Mermin free energy the
forces derive from), ``forces``, and for finite systems ``dipole`` (from
Mulliken charges, e·Å) and ``charges``. The model must have a repulsive
term. With ``scc=True`` charges are self-consistent (finite systems only).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .analysis import atomic_charges
from .hamiltonian import System
from .kpoints import gamma, mesh
from .params import TBModel
from .solver import solve


class TBCalculator(Calculator):
    implemented_properties = ["energy", "free_energy", "forces", "dipole", "charges"]

    def __init__(self, model: TBModel, kpts: Optional[int | tuple] = None, kT: float = 0.02,
                 charge: float = 0.0, scc: bool = False, **kwargs):
        super().__init__(**kwargs)
        if model.repulsive is None:
            raise ValueError(f"El modelo '{model.name}' no tiene parte repulsiva: no puede "
                             "dar energías totales ni fuerzas.")
        self.model = model
        self.kpts = kpts
        self.kT = kT
        self.charge = charge
        self.scc = scc
        self.last_solution = None

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from . import forces as band
        from . import scc as scc_module

        system = System.build(self.atoms, self.model)
        need_forces = "forces" in properties
        if self.scc:
            # Forces need tightly converged charges (their error scales with it).
            result = scc_module.self_consistent(system, charge=self.charge, kT=self.kT,
                                                tol=1e-10)
            if not result.converged:
                raise RuntimeError("SCC sin converger: " + result.summary())
            energy, forces, parts = scc_module.energy_and_forces(result, need_forces)
            solution = result.solution
            charges_by_atom = result.charges
        else:
            if system.periodic:
                kpts, weights = mesh(self.atoms, self.kpts or 8)
            else:
                kpts, weights = gamma()
            solution = solve(system, kpts, weights, charge=self.charge, kT=self.kT)
            energy, forces, parts = band.energy_and_forces(solution, need_forces)
            charges_by_atom = atomic_charges(solution)
        self.last_solution = solution
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        if forces is not None:
            self.results["forces"] = forces
        charges = np.zeros(len(self.atoms))
        for atom, q in charges_by_atom.items():
            charges[atom] = q
        self.results["charges"] = charges
        if not system.periodic:
            positions = self.atoms.get_positions()
            self.results["dipole"] = (charges[:, None] * (positions - positions.mean(0))).sum(0)
        self.results["parts"] = parts
