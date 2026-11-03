"""Core-level (XPS) binding-energy shifts: a core hole inside the charge SCC.

A tight-binding model has no core orbitals, but what a core hole does to the valence can
be put into the self-consistent charges, which is what a ΔSCF calculation measures:

* ``method="hole"`` (default): the ionised atom A carries a +1 point charge that every
  valence electron feels, through the potential ``V_B = -γ^h_AB`` on each atom B: on A
  itself the core-valence Coulomb constant J of its core level (``parameters/core_holes.json``,
  ``recipes/core_constants``: ``∂ε_valence/∂n_core`` of the free atom, computed, not
  fitted), elsewhere the Klopman-Ohno γ of the SCC with J in place of A's U (Ewald-summed in
  a periodic system). The valence count is unchanged; the SCC relaxes the charges around
  the hole (final-state screening), and

      E_B(A) ≈ E[with the hole] − E[ground] = Σ_B V_B Δq_B + relaxation  (+ const of A),

  with Δq the electrons gained by each atom. Any core level with a J: C, N, O, B 1s, P and
  S 2p, Se 3d.
* ``method="z+1"``: the equivalent core (Jolly & Hendrickson): A becomes the next element
  and the structure loses one electron. Needs the Z+1 element in the set, and its
  parameters as a cation: measured to work for C 1s with xu_chno, not for N 1s.

Only differences are meaningful (each method has an element constant): shifts against a
reference site of the same element, computed the same way. Validated against GPAW ΔSCF with
full core-hole setups in ``recipes/xps_validation``; a calibration factor per element, set
and method is shipped only where the model correlates with GPAW (``xps_calibration``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from ase import Atoms

#: The equivalent core of each element (Z + 1).
Z_PLUS_ONE = {"H": "He", "B": "C", "C": "N", "N": "O", "O": "F", "P": "S", "S": "Cl",
              "Se": "Br"}
METHODS = ("hole", "z+1")


@dataclass
class CoreLevel:
    index: int
    element: str
    energy_eV: float              # E(with hole) − E(ground), model total energies
    shift_eV: Optional[float]     # against the reference, when one is given
    method: str = "hole"


def core_hole_constant(element: str) -> tuple[str, float]:
    """(core level, J in eV) of ``parameters/core_holes.json``."""
    from .params import read_parameter_file

    table = read_parameter_file("core_holes")["core_holes"]
    if element not in table:
        raise ValueError(f"No hay constante de hueco de núcleo para {element}: corre "
                         "python -m tbkit.recipes.core_constants con ese elemento.")
    entry = table[element]
    return entry["level"], float(entry["J"]["value"])


def hole_potential(system, index: int, U=None) -> np.ndarray:
    """``V_B = -γ^h_AB`` (eV per electron) on every model atom for a hole on ``index``."""
    from .scc import COULOMB, _u_atom

    atoms = system.basis.atoms
    a = atoms.index(index)
    _, j = core_hole_constant(system.atoms[index].symbol)
    u = _u_atom(system, U).astype(float).copy()
    u[a] = j
    positions = system.atoms.get_positions()[atoms]
    if system.periodic:
        from .ewald import periodic_gamma

        gamma, _ = periodic_gamma(positions, np.asarray(system.atoms.get_cell()), u)
        row = gamma[a]
    else:
        r = np.linalg.norm(positions - positions[a], axis=1)
        ubar = 0.5 * (u[a] + u)
        row = COULOMB / np.sqrt(r ** 2 + (COULOMB / ubar) ** 2)
    return -row


def _scc(system, kT, kpts, charge=0.0, external=None):
    from .scc import energy_and_forces, self_consistent

    weights = None
    if system.periodic and kpts is not None:
        from .kpoints import mesh

        kpts, weights = mesh(system, kpts)
    result = self_consistent(system, charge=charge, kT=kT, tol=1e-9, max_iter=1500,
                             kpts=kpts if system.periodic else None, weights=weights,
                             external_potential=external)
    if not result.converged:
        raise RuntimeError("SCC sin converger: " + result.summary())
    return float(energy_and_forces(result, need_forces=False)[0])


def _check(model):
    if not model.scc:
        raise ValueError(f"El modelo '{model.name}' no tiene cargas autoconsistentes: el "
                         "hueco de núcleo necesita que la valencia se relaje (SCC).")


def ground_energy(atoms: Atoms, model, kT: float = 0.01, kpts=None) -> float:
    from .hamiltonian import System

    _check(model)
    return _scc(System.build(atoms, model), kT, kpts)


def ionisation_parts(atoms: Atoms, model, index: int, kT: float = 0.01, kpts=None,
                     ground: Optional[float] = None) -> tuple[float, float]:
    """The hole method split: (initial state ``Σ V_B Δq_B`` with the ground-state charges,
    relaxation = the rest of ``E[hole] − E[ground]``)."""
    from .hamiltonian import System
    from .scc import self_consistent

    total = ionisation_energy(atoms, model, index, kT, kpts, ground, "hole")
    system = System.build(atoms, model)
    weights = None
    if system.periodic and kpts is not None:
        from .kpoints import mesh

        kpts, weights = mesh(system, kpts)
    result = self_consistent(system, kT=kT, tol=1e-9, max_iter=1500,
                             kpts=kpts if system.periodic else None, weights=weights)
    initial = float(hole_potential(system, index) @ result.dq)
    return initial, total - initial


def ionisation_energy(atoms: Atoms, model, index: int, kT: float = 0.01, kpts=None,
                      ground: Optional[float] = None, method: str = "hole") -> float:
    """``E[core hole on index] − E[ground]`` (eV) with the model, by ``method``."""
    from .hamiltonian import System
    from .hubbard import _neutral

    _check(model)
    if method not in METHODS:
        raise ValueError(f"Método desconocido: {method!r} ({', '.join(METHODS)}).")
    if ground is None:
        ground = ground_energy(atoms, model, kT, kpts)
    element = atoms[index].symbol
    if method == "z+1":
        equivalent = Z_PLUS_ONE.get(element)
        if equivalent is None or equivalent not in model.orbitals:
            raise ValueError(f"El modelo '{model.name}' no tiene {equivalent or '?'}, el "
                             f"núcleo equivalente de {element}: usa method='hole'.")
        ion = atoms.copy()
        ion.symbols[index] = equivalent
        return _scc(System.build(ion, model), kT, kpts, charge=1.0) - ground
    system = System.build(atoms, model)
    potential = hole_potential(system, index)
    n0 = _neutral(system)
    q0 = np.array([n0[system.basis.of_atom(a).start:system.basis.of_atom(a).stop].sum()
                   for a in system.basis.atoms])
    # the SCC energy keeps Σ V_B q_B (all valence electrons); the hole's energy is the
    # interaction with what each atom gained, Σ V_B Δq_B
    return _scc(system, kT, kpts, external=potential) - float(potential @ q0) - ground


def xps_calibration(set_name: str, element: str, method: str = "hole") -> dict:
    """The calibration of ``parameters/xps_scale.json`` for a method, set and element:
    ``{"initial": a, "relaxation": b}`` (hole) or ``{"scale": λ}`` (z+1), fitted to GPAW
    ΔSCF on half the sites of ``recipes/xps_validation`` and tested on the rest."""
    from .params import read_parameter_file

    scales = read_parameter_file("xps_scale")["scales"]
    try:
        entry = scales[method][set_name][element]
    except KeyError:
        raise KeyError(f"Sin calibración XPS para {element} con {set_name} ({method}): el "
                       "modelo no correlaciona con GPAW o no se validó.") from None
    return {k: float(v["value"]) for k, v in entry.items()}


def _shift(parts, ref_parts, calibration, scale):
    if calibration is None:
        return scale * (sum(parts) - sum(ref_parts))
    if "scale" in calibration:
        return calibration["scale"] * (sum(parts) - sum(ref_parts))
    return (calibration["initial"] * (parts[0] - ref_parts[0])
            + calibration["relaxation"] * (parts[1] - ref_parts[1]))


def core_levels(atoms: Atoms, model, indices: Sequence[int], kT: float = 0.01, kpts=None,
                reference: Optional[tuple[Atoms, int]] = None, scale: float = 1.0,
                method: str = "hole", calibration: Optional[dict] = None) -> list[CoreLevel]:
    """Ionisation energies of ``indices`` and, with ``reference`` (structure, index), their
    shifts against it (same element only): raw times ``scale`` (default 1), or with a
    ``calibration`` of :func:`xps_calibration` (hole: initial-state and relaxation parts
    weighted; z+1: one factor)."""
    split = calibration is not None and method == "hole" and "initial" in calibration

    def parts(a, i, ground=None):
        if split:
            return ionisation_parts(a, model, i, kT, kpts if a is atoms else None, ground)
        return (ionisation_energy(a, model, i, kT, kpts if a is atoms else None, ground,
                                  method), 0.0)

    ground = ground_energy(atoms, model, kT, kpts)
    ref_parts = ref_element = None
    if reference is not None:
        ref_atoms, ref_index = reference
        ref_element = ref_atoms[ref_index].symbol
        ref_parts = parts(ref_atoms, ref_index)
    out = []
    for i in indices:
        mine = parts(atoms, i, ground)
        shift = _shift(mine, ref_parts, calibration, scale) \
            if ref_parts is not None and atoms[i].symbol == ref_element else None
        out.append(CoreLevel(int(i), atoms[i].symbol, float(sum(mine)), shift, method))
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
