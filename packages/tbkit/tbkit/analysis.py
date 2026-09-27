"""What a solution contains: densities, charges, bonds, spectra, orbitals.

* :func:`density_matrix` -- ``P = sum_n f_n c_n c_n^†`` (per spin, per k).
* :func:`populations` -- Mulliken (``PS``) or Löwdin (``S^½ P S^½``) orbital
  populations, summed per atom by :func:`atomic_charges`.
* :func:`bond_orders` -- Mayer bond orders (Wiberg for an orthogonal model);
  :func:`coulson_bond_orders` for a π model.
* :func:`dos` / :func:`pdos` -- Gaussian-broadened densities of states, the
  projected one split by Mulliken weights per atom, element or orbital.
* :func:`bands` -- eigenvalues along a path.
* :func:`orbital_on_grid` / :func:`write_cube` -- a molecular orbital built
  from Slater-type functions on a real-space grid, for isosurfaces.

Mulliken populations of a non-orthogonal model depend on the basis and can
leave an orbital slightly negative; Löwdin's are always within 0..2 per
orbital. Neither is an observable: they are bookkeeping for trends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.units import Bohr

from .solver import Solution


def density_matrix(solution: Solution, spin: Optional[int] = None,
                   k_index: int = 0) -> np.ndarray:
    """``P(k)`` for one k point; summed over spins unless ``spin`` is given."""
    spins = range(solution.nspin) if spin is None else [spin]
    n = solution.vectors.shape[2]
    p = np.zeros((n, n), dtype=solution.vectors.dtype)
    for s in spins:
        c = solution.vectors[s, k_index]
        p += (c * solution.occupations[s, k_index]) @ c.conj().T
    return p


def _weights(solution: Solution, s: int, k: int, method: str) -> np.ndarray:
    """Per orbital and band, how much of band n sits on orbital mu."""
    c = solution.vectors[s, k]
    if solution.overlaps is None:
        return np.abs(c) ** 2
    overlap = solution.overlaps[k]
    if method == "mulliken":
        return np.real(c.conj() * (overlap @ c))
    from scipy.linalg import sqrtm

    root = sqrtm(overlap)
    return np.abs(root @ c) ** 2


def populations(solution: Solution, method: str = "mulliken") -> np.ndarray:
    """Electron population of every orbital, ``[spin, orbital]``."""
    if method not in ("mulliken", "lowdin"):
        raise ValueError("method: 'mulliken' o 'lowdin'.")
    out = np.zeros((solution.nspin, solution.vectors.shape[2]))
    for s in range(solution.nspin):
        for k, w in enumerate(solution.weights):
            out[s] += w * _weights(solution, s, k, method) @ solution.occupations[s, k]
    return out


def atomic_charges(solution: Solution, method: str = "mulliken") -> dict[int, float]:
    """Net charge per atom in the model, e (positive = electrons lost)."""
    system = solution.system
    pops = populations(solution, method).sum(axis=0)
    coordination = system.coordination()
    charges = {}
    for atom in system.basis.atoms:
        rng = system.basis.of_atom(atom)
        neutral = system.model.electrons_of(system.atoms[atom].symbol, coordination[atom])
        charges[atom] = float(neutral - pops[rng.start:rng.stop].sum())
    return charges


def spin_moments(solution: Solution, method: str = "mulliken") -> dict[int, float]:
    """Local moment per atom, ``n↑ - n↓`` in Bohr magnetons (0 without spin)."""
    system = solution.system
    if solution.nspin == 1:
        return {atom: 0.0 for atom in system.basis.atoms}
    pops = populations(solution, method)
    return {atom: float((pops[0] - pops[1])[system.basis.of_atom(atom)].sum())
            for atom in system.basis.atoms}


def bond_orders(solution: Solution, threshold: float = 0.05) -> dict[tuple[int, int], float]:
    """Mayer bond orders between atoms (finite systems, Γ only).

    ``B_AB = 2 sum_sigma sum_{mu in A, nu in B} (P^σ S)_{mu nu} (P^σ S)_{nu mu}``,
    Wiberg's index when S = 1. Only the orbitals of the model count: in a π
    model it is the square of Coulson's π bond order (benzene: 0.444 =
    (2/3)²); see :func:`coulson_bond_orders` for the latter.
    """
    if solution.system.periodic:
        raise ValueError("Órdenes de enlace de Mayer: solo sistemas finitos (Γ).")
    basis = solution.system.basis
    n = basis.size
    overlap = solution.overlaps[0] if solution.overlaps is not None else np.eye(n)
    if solution.nspin == 1:
        half = 0.5 * np.real(density_matrix(solution)) @ overlap
        channels = [half, half]
    else:
        channels = [np.real(density_matrix(solution, spin=s)) @ overlap for s in range(2)]
    atoms = basis.atoms
    out = {}
    for x, a in enumerate(atoms):
        ra = basis.of_atom(a)
        for b in atoms[x + 1:]:
            rb = basis.of_atom(b)
            value = 2.0 * sum(float(np.sum(ps[ra.start:ra.stop, rb.start:rb.stop]
                                           * ps[rb.start:rb.stop, ra.start:ra.stop].T))
                              for ps in channels)
            if value >= threshold:
                out[(a, b)] = value
    return out


def coulson_bond_orders(solution: Solution, threshold: float = 0.05
                        ) -> dict[tuple[int, int], float]:
    """Coulson π bond orders ``p_ab = P_ab`` of an orthogonal π model (Γ).

    Benzene 2/3, graphene ≈ 0.525, ethylene 1. Only defined for one π orbital per
    atom and S = 1.
    """
    system = solution.system
    if system.periodic or not system.model.orthogonal or any(
            len(system.basis.of_atom(a)) != 1 for a in system.basis.atoms):
        raise ValueError("Orden de enlace de Coulson: modelo π ortogonal y sistema finito.")
    p = np.real(density_matrix(solution))
    out = {}
    for bond in system.bonds:              # bonded pairs only (both directions listed)
        a, b = sorted((bond.i, bond.j))
        value = float(p[system.basis.first[a], system.basis.first[b]])
        if abs(value) >= threshold:
            out[(a, b)] = value
    return out


def _gauss(grid: np.ndarray, centres: np.ndarray, weights: np.ndarray, sigma: float):
    diff = (grid[:, None] - centres[None, :]) / sigma
    return (np.exp(-0.5 * diff ** 2) @ weights) / (sigma * np.sqrt(2 * np.pi))


def dos(solution: Solution, grid: Optional[np.ndarray] = None, sigma: float = 0.1,
        spin: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Density of states, states/eV (per unit cell for a periodic system).

    Summed over spins (both channels, or twice the single one) unless
    ``spin`` selects one channel.
    """
    if grid is None:
        grid = np.linspace(solution.energies.min() - 1, solution.energies.max() + 1, 2000)
    degeneracy = 2.0 if solution.nspin == 1 else 1.0
    spins = range(solution.nspin) if spin is None else [spin]
    total = np.zeros_like(grid)
    for s in spins:
        e = solution.energies[s].ravel()
        w = np.repeat(solution.weights, solution.energies.shape[2])
        total += (degeneracy if spin is None else 1.0) * _gauss(grid, e, w, sigma)
    return grid, total


def pdos(solution: Solution, by: str = "element", grid: Optional[np.ndarray] = None,
         sigma: float = 0.1, spin: Optional[int] = None) -> tuple[np.ndarray, dict]:
    """Projected DOS by ``"element"``, ``"atom"`` or ``"orbital"`` (Mulliken weights).

    The projections add up to :func:`dos` (Mulliken weights of each state sum
    to 1), unlike a projection on atomic orbitals in a plane-wave code.
    """
    if grid is None:
        grid = np.linspace(solution.energies.min() - 1, solution.energies.max() + 1, 2000)
    basis = solution.system.basis
    labels = {"element": [o.element for o in basis.orbitals],
              "atom": [f"{o.element}{o.atom}" for o in basis.orbitals],
              "orbital": [f"{o.element}-{o.name}" for o in basis.orbitals]}[by]
    degeneracy = 2.0 if solution.nspin == 1 else 1.0
    spins = range(solution.nspin) if spin is None else [spin]
    out: dict[str, np.ndarray] = {label: np.zeros_like(grid) for label in dict.fromkeys(labels)}
    for s in spins:
        for k, wk in enumerate(solution.weights):
            weights = _weights(solution, s, k, "mulliken")
            for mu, label in enumerate(labels):
                out[label] += (degeneracy if spin is None else 1.0) * _gauss(
                    grid, solution.energies[s, k], wk * weights[mu], sigma)
    return grid, out


def bands(system, path, spin_potentials=None) -> np.ndarray:
    """Eigenvalues along ``path`` (ASE BandPath or fractional k array): ``[spin, k, band]``."""
    from .solver import diagonalise

    kpts = path.kpts if hasattr(path, "kpts") else np.asarray(path)
    channels = [None] if spin_potentials is None else list(spin_potentials)
    return np.array([diagonalise(system, kpts, extra)[0] for extra in channels])


# --------------------------------------------------------------------------
# Orbitals in real space
# --------------------------------------------------------------------------

#: Single-zeta Slater exponents, 1/Bohr (Slater's rules; H uses 1.24, the
#: value optimised for molecules by Hehre, Stewart and Pople 1969). Only for
#: drawing orbitals: the model itself never evaluates a wavefunction.
SLATER_ZETA = {"H": 1.24, "B": 1.30, "C": 1.625, "N": 1.95, "O": 2.275}
_PRINCIPAL = {"H": 1, "B": 2, "C": 2, "N": 2, "O": 2}


def _sto(orbital: str, element: str, dx, dy, dz, r, normal=None) -> np.ndarray:
    zeta = SLATER_ZETA[element] / Bohr            # 1/Å
    n = _PRINCIPAL[element]
    norm = (2 * zeta) ** n * np.sqrt(2 * zeta / _factorial(2 * n))
    radial = norm * r ** (n - 1) * np.exp(-zeta * r)
    if orbital == "s":
        return radial / np.sqrt(4 * np.pi)
    if orbital == "pi":
        nx, ny, nz = normal
        component = nx * dx + ny * dy + nz * dz
    else:
        component = {"px": dx, "py": dy, "pz": dz}[orbital]
    with np.errstate(invalid="ignore", divide="ignore"):
        angular = np.where(r > 1e-12, component / r, 0.0)
    return radial * np.sqrt(3 / (4 * np.pi)) * angular


def _factorial(n: int) -> int:
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def local_normals(atoms: Atoms, cutoff: float = 1.8) -> np.ndarray:
    """Unit surface normal at every atom, from its neighbours (π orbital axis).

    The normal of the plane through an atom and its neighbours; with fewer
    than two neighbours, z. Signs are made consistent: +z for a flat sheet,
    outward from the centroid for a curved one.
    """
    from ase.neighborlist import neighbor_list

    i, d = neighbor_list("iD", atoms, cutoff)
    positions = atoms.get_positions()
    centre = positions.mean(axis=0)
    flat = np.ptp(positions[:, 2]) < 0.3
    normals = np.tile([0.0, 0.0, 1.0], (len(atoms), 1))
    for atom in range(len(atoms)):
        vectors = d[i == atom]
        if len(vectors) >= 2:
            _, _, vt = np.linalg.svd(np.vstack([vectors, np.zeros(3)]))
            normal = vt[-1]
            reference = np.array([0, 0, 1.0]) if flat else positions[atom] - centre
            if normal @ reference < 0:
                normal = -normal
            normals[atom] = normal
    return normals


def orbital_on_grid(solution: Solution, band: int, spin: int = 0, k_index: int = 0,
                    spacing: float = 0.2, padding: float = 3.0):
    """Real part of one eigenstate on a grid; returns ``(origin, axes, values)``.

    The grid box encloses the atoms plus ``padding`` Å; ``axes`` are the three
    step vectors (Å). A π orbital points along the local surface normal,
    taken from each atom's neighbours (its sign is chosen consistently on a
    flat or gently curved sheet: along +z, or outwards from the centre).
    """
    atoms = solution.system.atoms
    basis = solution.system.basis
    c = solution.vectors[spin, k_index, :, band]
    positions = atoms.get_positions()
    low = positions.min(axis=0) - padding
    high = positions.max(axis=0) + padding
    counts = np.maximum(2, np.ceil((high - low) / spacing).astype(int))
    axes = [np.linspace(low[i], high[i], counts[i]) for i in range(3)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    values = np.zeros_like(x)
    normals = local_normals(atoms)
    for coefficient, orbital in zip(c, basis.orbitals, strict=True):
        if abs(coefficient) < 1e-8:
            continue
        dx, dy, dz = (x - positions[orbital.atom, 0], y - positions[orbital.atom, 1],
                      z - positions[orbital.atom, 2])
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        values += np.real(coefficient) * _sto(orbital.name, orbital.element, dx, dy, dz, r,
                                              normals[orbital.atom])
    steps = np.diag((high - low) / (counts - 1))
    return low, steps, values


def write_cube(path: str | Path, atoms: Atoms, origin, steps, values,
               comment: str = "tbkit orbital") -> Path:
    """Write a Gaussian cube file (VESTA, VMD, Avogadro, pyvista read it)."""
    path = Path(path)
    lines = [comment, "valores reales del orbital (unidades arbitrarias de amplitud)"]
    to_bohr = 1.0 / Bohr
    lines.append(f"{len(atoms):5d} " + " ".join(f"{v * to_bohr:12.6f}" for v in origin))
    for count, step in zip(values.shape, steps, strict=True):
        lines.append(f"{count:5d} " + " ".join(f"{v * to_bohr:12.6f}" for v in step))
    for number, position in zip(atoms.numbers, atoms.get_positions(), strict=True):
        lines.append(f"{number:5d} {float(number):12.6f} "
                     + " ".join(f"{v * to_bohr:12.6f}" for v in position))
    flat = values.reshape(values.shape[0] * values.shape[1], values.shape[2])
    for row in flat:
        for start in range(0, len(row), 6):
            lines.append(" ".join(f"{v:13.5e}" for v in row[start:start + 6]))
    path.write_text("\n".join(lines) + "\n")
    return path
