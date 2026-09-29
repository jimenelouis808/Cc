"""What the GUI computes, as plain functions without Qt.

Every page of the window calls one of these in a worker thread and draws what
comes back (arrays, dicts, tables). Keeping them free of Qt means they are
tested like the rest of tbkit, and the window only lays out and draws.

The ground state is computed once per (structure, model, charge, SCC) and
shared: levels, DOS, PDOS, charges and orbitals all come from the same
solution, self-consistent when the model says so (``TBModel.scc``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from ..analysis import atomic_charges, bond_orders, dos, orbital_on_grid, pdos
from ..hamiltonian import System
from ..kpoints import band_path, gamma, mesh
from ..params import TBModel, load_parameters, pi_model, xu_carbon
from ..solver import Solution, solve

#: Built-in models the window offers, name -> (label, loader).
MODELS = {
    "pi": ("π (Hückel, t = −2,7 eV)", lambda: pi_model(t=-2.7)),
    "sp3": ("Xu: carbono sp³ con repulsión", xu_carbon),
    "chn": ("C/H/N (xu_chn, SCC)", lambda: load_parameters("xu_chn")),
    "chno": ("C/H/N/O (xu_chno, SCC)", lambda: load_parameters("xu_chno")),
    "chnob": ("C/H/N/O + B (xu_chnob, SCC)", lambda: load_parameters("xu_chnob")),
    "chnos": ("C/H/N/O + S (xu_chnos, SCC)", lambda: load_parameters("xu_chnos")),
    "chnop": ("C/H/N/O + P (xu_chnop, SCC)", lambda: load_parameters("xu_chnop")),
}


def load_model(name_or_path: str) -> TBModel:
    """A built-in model by name, or any parameter file by path."""
    if name_or_path in MODELS:
        return MODELS[name_or_path][1]()
    return load_parameters(name_or_path)


def suggest_model(atoms: Atoms) -> str:
    """The smallest built-in set that covers the elements of ``atoms``."""
    elements = set(atoms.get_chemical_symbols())
    for name, covers in (("sp3", {"C"}), ("chn", {"C", "H", "N"}),
                         ("chno", {"C", "H", "N", "O"}), ("chnob", {"C", "H", "N", "O", "B"}),
                         ("chnos", {"C", "H", "N", "O", "S"}),
                         ("chnop", {"C", "H", "N", "O", "P"})):
        if elements <= covers:
            return name
    return "pi"


def check_model(atoms: Atoms, model: TBModel, scc: Optional[bool] = None) -> list[str]:
    """Reasons the model cannot treat ``atoms`` (empty list: it can).

    ``scc`` overrides the model's choice, as the window's selector does: a set
    fitted with SCC may be run without it on purpose (a periodic structure),
    never by default."""
    problems = []
    missing = sorted(set(atoms.get_chemical_symbols()) - set(model.orbitals))
    if missing:
        problems.append(f"El modelo no tiene parámetros para: {', '.join(missing)}.")
    if (model.scc if scc is None else scc) and any(atoms.get_pbc()):
        problems.append("SCC y estructura periódica: SCC necesita un sistema finito (no hay "
                        "suma de Ewald). Pon SCC en «no» a sabiendas (el modelo se ajustó "
                        "con SCC) o usa un fragmento finito.")
    return problems


def structure_summary(atoms: Atoms) -> dict:
    pbc = atoms.get_pbc()
    return {"formula": atoms.get_chemical_formula(), "atoms": len(atoms),
            "periodic": "no" if not pbc.any() else "".join("xyz"[i] for i in range(3) if pbc[i]),
            "elements": sorted(set(atoms.get_chemical_symbols()))}


# --------------------------------------------------------------------------
# Ground state
# --------------------------------------------------------------------------

@dataclass
class GroundState:
    solution: Solution
    charges: dict[int, float]
    scc: bool
    iterations: int = 0
    converged: bool = True
    info: dict = field(default_factory=dict)


def ground_state(atoms: Atoms, model: TBModel, charge: float = 0.0, kT: float = 0.01,
                 kmesh: int = 24, scc: Optional[bool] = None) -> GroundState:
    """Solve once: SCC when the model (or ``scc``) asks for it, else one diagonalisation."""
    problems = check_model(atoms, model, scc)
    if problems:
        raise ValueError(" ".join(problems))
    system = System.build(atoms, model)
    use_scc = model.scc if scc is None else scc
    if use_scc:
        from ..scc import self_consistent

        result = self_consistent(system, charge=charge, kT=kT)
        state = GroundState(result.solution, result.charges, True, result.iterations,
                            result.converged)
    else:
        kpts = mesh(atoms, kmesh) if system.periodic else gamma()
        solution = solve(system, *kpts, charge=charge, kT=kT)
        state = GroundState(solution, atomic_charges(solution), False)
    solution = state.solution
    homo, lumo = solution.homo_lumo()
    state.info = {"homo": homo, "lumo": lumo, "gap": solution.gap(), "fermi": solution.fermi,
                  "electrons": solution.electrons, "orbitals": solution.system.basis.size
                  if hasattr(solution.system.basis, "size") else None}
    return state


def levels_table(state: GroundState, around: int = 10) -> list[tuple[int, float, float, str]]:
    """``(index, energy, occupation, label)`` for the levels around the gap (Γ, spin 0)."""
    solution = state.solution
    energies = solution.energies[0, 0]
    occupations = solution.occupations[0, 0]           # 2 per level when spin-paired
    homo = int(np.flatnonzero(occupations > 1e-3).max()) if (occupations > 1e-3).any() else -1
    low, high = max(0, homo - around + 1), min(len(energies), homo + around + 1)
    rows = []
    for i in range(low, high):
        label = "HOMO" if i == homo else "LUMO" if i == homo + 1 else \
            f"HOMO−{homo - i}" if i < homo else f"LUMO+{i - homo - 1}"
        rows.append((i, float(energies[i]), float(occupations[i]), label))
    return rows


def dos_curves(state: GroundState, sigma: float = 0.1, by: Optional[str] = "element",
               window: tuple[float, float] = (-15.0, 10.0), points: int = 1500) -> dict:
    """Total DOS and (optionally) PDOS against E − E_F."""
    solution = state.solution
    grid = np.linspace(solution.fermi + window[0], solution.fermi + window[1], points)
    _, total = dos(solution, grid, sigma)
    out = {"energy": grid - solution.fermi, "total": total, "projected": {}}
    if by:
        _, out["projected"] = pdos(solution, by, grid, sigma)
    return out


def band_curves(atoms: Atoms, model: TBModel, path: Optional[str] = None,
                npoints: int = 200, kT: float = 0.01, kmesh: int = 24) -> dict:
    """Bands along a path of a periodic structure, relative to E_F of a mesh."""
    from ..analysis import bands

    system = System.build(atoms, model)
    if not system.periodic:
        raise ValueError("Las bandas necesitan una estructura periódica.")
    bandpath = band_path(atoms, path, npoints)
    energies = bands(system, bandpath)
    fermi = solve(system, *mesh(atoms, kmesh), kT=kT).fermi
    x, xticks, labels = bandpath.get_linear_kpoint_axis()
    return {"x": x, "energies": energies[0] - fermi, "ticks": xticks, "labels": list(labels)}


def orbital_grid(state: GroundState, band: int, spacing: float = 0.25) -> dict:
    """One eigenstate on a grid (for an isosurface)."""
    origin, steps, values = orbital_on_grid(state.solution, band, spacing=spacing)
    return {"origin": [float(v) for v in origin], "spacing": [float(v) for v in np.diag(steps)],
            "values": values,
            "energy": float(state.solution.energies[0, 0, band])}


def bonds_of(atoms: Atoms, factor: float = 1.2) -> list[tuple[int, int]]:
    """Bonds to draw: closer than ``factor`` times the covalent radii."""
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    radii = covalent_radii[atoms.numbers]
    ii, jj, dd = neighbor_list("ijd", atoms, factor * 2 * radii.max(), self_interaction=False)
    return [(int(i), int(j)) for i, j, d in zip(ii, jj, dd, strict=True)
            if i < j and d < factor * (radii[i] + radii[j])]


def bond_order_table(state: GroundState, threshold: float = 0.1):
    return sorted(bond_orders(state.solution, threshold).items())


def read_structure(path: str | Path) -> Atoms:
    from ase.io import read

    atoms = read(str(path))
    if not any(atoms.get_pbc()) and atoms.cell.rank == 0:
        atoms.pbc = False
    return atoms
