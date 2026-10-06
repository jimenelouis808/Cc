"""Vibrational modes: what moves, where, and why a frequency shifted.

A frequency alone does not say what a mode is. This module keeps the force
constants (the Hessian) with the modes, so that each mode can be described
and re-examined:

* :func:`participation`: the share of each atom, element or group in a mode,
  ``P_X(ν) = Σ_{i∈X} |e_i(ν)|²`` with e the orthonormal mass-weighted
  eigenvector: 1 for a mode entirely on X. A dopant or group mode is local
  when its P is large compared with its share of the atoms.
* :func:`cylindrical_character`: radial, tangential and axial shares of a
  mode about an axis (nanotubes: the RBM is radial, G tangential).
* :func:`vibrational_dos`: the density of vibrational states, total or
  projected (by element or any atom set), with Gaussian broadening. It is
  not a Raman or IR spectrum: every mode counts, active or not.
* :meth:`Vibrations.with_masses`: the same force constants with other
  masses. Giving a dopant the mass of carbon separates the mass effect on a
  frequency (ω ∝ √(k/m)) from the chemical one (a changed k).

Modes are obtained by central differences of the forces (``ase.vibrations``);
L = e/√m is the Cartesian displacement per unit normal coordinate.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.units import invcm

from .params import TBModel


@dataclass
class Vibrations:
    """Γ modes of a structure with the Hessian they come from."""

    atoms: Atoms
    hessian: np.ndarray                # (3N, 3N) eV/Å²
    masses: np.ndarray                 # amu
    frequencies: np.ndarray            # cm⁻¹, ascending, imaginary as negative
    eigenvectors: np.ndarray           # (3N, N, 3) orthonormal, mass-weighted e
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_hessian(cls, atoms: Atoms, hessian: np.ndarray, masses=None,
                     source: str = "") -> "Vibrations":
        masses = atoms.get_masses() if masses is None else np.asarray(masses, dtype=float)
        weights = np.repeat(masses ** -0.5, 3)
        hessian = 0.5 * (hessian + hessian.T)
        omega2, vectors = np.linalg.eigh(weights[:, None] * hessian * weights[None, :])
        from ase import units

        conversion = units._hbar * units.m / np.sqrt(units._e * units._amu)
        energies = conversion * np.sqrt(np.abs(omega2)) * np.sign(omega2)
        return cls(atoms.copy(), hessian, masses, energies / invcm,
                   vectors.T.reshape(len(omega2), len(atoms), 3), source)

    @property
    def modes(self) -> np.ndarray:
        """L = e/√m (Å per unit normal coordinate), (3N, N, 3)."""
        return self.eigenvectors / np.sqrt(self.masses)[None, :, None]

    def with_masses(self, masses: dict[int, float] | Sequence[float]) -> "Vibrations":
        """Same force constants, other masses (``{atom: mass}`` or a full list)."""
        new = self.masses.copy()
        if isinstance(masses, dict):
            for index, mass in masses.items():
                new[index] = float(mass)
        else:
            new = np.asarray(masses, dtype=float)
        return Vibrations.from_hessian(self.atoms, self.hessian, new,
                                       self.source + " (masas cambiadas)")

    def save(self, directory: str | Path) -> Path:
        """Write ``modes.npz`` (frequencies, eigenvectors, L, Hessian, masses) and the
        structure, never overwriting an existing file."""
        from ase.io import write

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "modes.npz"
        if target.exists():
            raise FileExistsError(f"{target} ya existe: los resultados no se sobrescriben.")
        np.savez(target, frequencies_cm1=self.frequencies, eigenvectors=self.eigenvectors,
                 modes=self.modes, hessian_ev_per_a2=self.hessian, masses_amu=self.masses,
                 source=self.source)
        write(directory / "structure.extxyz", self.atoms)
        return target


#: Measured on graphene with Xu (Γ G mode, kT = 0.05 eV): 12 k per axis gives
#: 1572 cm⁻¹, 24 -> 1656, 48 -> 1672, 72 -> 1674; at kT = 0.025 eV, 12 k gives 1455.
GAPLESS_WARNING = (
    "Sin gap: las frecuencias de los modos que acoplan con los estados en E_F (anomalías "
    "de Kohn: G del grafeno, tubos metálicos) dependen mucho de la malla k. En grafeno con "
    "Xu y kT = 0,05 eV, G vale 1572 cm⁻¹ con 12 k por eje, 1656 con 24 y 1674 convergida "
    "(≥ 48). Converge la malla (tasks.kmesh_convergence) antes de usar estas frecuencias.")


def gapless_periodic(atoms: Atoms) -> bool:
    """True for a periodic structure whose last TB solution has no gap."""
    solution = getattr(atoms.calc, "last_solution", None)
    return bool(atoms.get_pbc().any() and solution is not None and solution.gap() <= 0)


def vibrations(atoms: Atoms, model: TBModel, kmesh: int = 12, kT: float = 0.02,
               delta: float = 0.005, scc: Optional[bool] = None) -> Vibrations:
    """Γ modes of ``atoms`` (relax it first) under ``model``, keeping the Hessian."""
    from ase.vibrations import Vibrations as AseVibrations

    from .calculator import TBCalculator

    atoms = atoms.copy()
    atoms.calc = TBCalculator(model, kpts=kmesh, kT=kT, scc=scc)
    residual = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    gapless = gapless_periodic(atoms)
    with tempfile.TemporaryDirectory() as directory:
        ase_vib = AseVibrations(atoms, name=os.path.join(directory, "vib"), delta=delta)
        ase_vib.run()
        hessian = ase_vib.get_vibrations().get_hessian_2d()
    result = Vibrations.from_hessian(atoms, hessian, source=model.name)
    if residual > 0.05:
        result.warnings.append(f"Fuerza residual {residual:.3f} eV/Å: geometría sin relajar.")
    if gapless:
        result.warnings.append(GAPLESS_WARNING)
    return result


def hessian_rows(atoms: Atoms, make_calc, folder, indices=None, delta: float = 0.01,
                 part: tuple = (0, 1)) -> np.ndarray | None:
    """Finite-difference Hessian rows (eV/Å²) of ``indices`` (default: every atom), with
    the forces of each ±δ displacement cached in ``folder`` (``d_A_C_S.npy``): a rerun
    resumes and ``part=(r, n)`` lets n processes share the displacements. Returns the
    (3·len(indices), 3N) rows once all are there (None from a part that stops early).
    Rows are ``-∂F/∂x`` by central differences, not yet symmetrised."""
    from pathlib import Path

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(atoms))) if indices is None else [int(i) for i in indices]
    jobs = [(a, c, s) for a in indices for c in range(3) for s in (1, -1)]
    base = atoms.get_positions()
    calc = None                      # one per process: an SCC calculator starts warm
    for k, (a, c, sign) in enumerate(jobs):
        if k % part[1] != part[0]:
            continue
        path = folder / f"d_{a:04d}_{c}_{'p' if sign > 0 else 'm'}.npy"
        if path.exists():
            continue
        probe = atoms.copy()
        positions = base.copy()
        positions[a, c] += sign * delta
        probe.set_positions(positions)
        calc = calc or make_calc()
        probe.calc = calc
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, probe.get_forces())
        tmp.replace(path)
    rows = np.zeros((3 * len(indices), 3 * len(atoms)))
    for r, a in enumerate(indices):
        for c in range(3):
            pair = [folder / f"d_{a:04d}_{c}_{t}.npy" for t in ("p", "m")]
            if not all(x.exists() for x in pair):
                return None
            rows[3 * r + c] = -(np.load(pair[0]) - np.load(pair[1])).ravel() / (2 * delta)
    return rows


def embedded_hessian(reference: np.ndarray, n_host: int, n_atoms: int, region,
                     region_rows: np.ndarray) -> np.ndarray:
    """A Hessian for a locally modified structure: the reference (host) Hessian
    everywhere except the rows and columns of ``region`` (which must contain every
    atom ≥ ``n_host``), taken from ``region_rows`` (``hessian_rows`` of the new
    structure). Pairs inside the region are averaged with their transpose.

    The approximation is that force constants between atoms far from the change
    stay as in the host; check it by the Rayleigh quotient of a few resulting modes
    with the full model (``sites.projected_frequency``)."""
    region = [int(i) for i in region]
    missing = set(range(n_host, n_atoms)) - set(region)
    if missing:
        raise ValueError(f"Los átomos añadidos {sorted(missing)} deben estar en la región.")
    hessian = np.zeros((3 * n_atoms, 3 * n_atoms))
    hessian[:3 * n_host, :3 * n_host] = reference
    cols = np.concatenate([np.arange(3 * a, 3 * a + 3) for a in region])
    hessian[cols, :] = region_rows
    hessian[:, cols] = region_rows.T
    inner = np.ix_(cols, cols)
    block = region_rows[:, cols]
    hessian[inner] = 0.5 * (block + block.T)
    return 0.5 * (hessian + hessian.T)


# --------------------------------------------------------------------------
# Describing a mode
# --------------------------------------------------------------------------

def participation(vib: Vibrations, groups: dict[str, Iterable[int]] | None = None
                  ) -> dict[str, np.ndarray]:
    """``P_X(ν)`` for each group (default: by element), shape (3N,) each.

    Shares of the mass-weighted eigenvector: they add up to 1 over any
    partition of the atoms.
    """
    weights = np.sum(np.abs(vib.eigenvectors) ** 2, axis=2)          # (3N, N)
    if groups is None:
        symbols = np.array(vib.atoms.get_chemical_symbols())
        groups = {el: np.flatnonzero(symbols == el) for el in dict.fromkeys(symbols)}
    return {name: weights[:, list(indices)].sum(axis=1) for name, indices in groups.items()}


def localisation(vib: Vibrations, atoms: Iterable[int]) -> np.ndarray:
    """``P_X(ν) / (N_X/N)``: how much more a mode sits on X than a uniform mode would.

    1 for a delocalised mode, N/N_X at most (entirely on X).
    """
    indices = list(atoms)
    share = participation(vib, {"X": indices})["X"]
    return share / (len(indices) / len(vib.atoms))


def cylindrical_character(vib: Vibrations, axis: Optional[np.ndarray] = None,
                          centre: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """Radial, tangential and axial shares of every mode (each (3N,), summing to 1).

    ``axis`` defaults to the periodic direction for a 1D-periodic cell, else
    the principal axis of largest spread; ``centre`` to the centroid.
    """
    positions = vib.atoms.get_positions()
    centre = positions.mean(axis=0) if centre is None else np.asarray(centre, float)
    if axis is None:
        pbc = vib.atoms.get_pbc()
        if pbc.sum() == 1:
            axis = vib.atoms.cell[int(np.flatnonzero(pbc)[0])]
        else:
            _, _, vt = np.linalg.svd(positions - centre, full_matrices=False)
            axis = vt[0]
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    relative = positions - centre
    radial = relative - np.outer(relative @ axis, axis)
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = np.divide(radial, norms, out=np.zeros_like(radial), where=norms > 1e-8)
    tangential = np.cross(axis, radial)
    e = vib.eigenvectors
    out = {"radial": np.sum(np.einsum("mnx,nx->mn", e, radial) ** 2, axis=1),
           "tangential": np.sum(np.einsum("mnx,nx->mn", e, tangential) ** 2, axis=1),
           "axial": np.sum(np.einsum("mnx,x->mn", e, axis) ** 2, axis=1)}
    return out


def breathing_overlap(vib: Vibrations, axis: Optional[np.ndarray] = None,
                      centre: Optional[np.ndarray] = None) -> np.ndarray:
    """``|⟨e|b⟩|²`` with b every atom moving radially outwards in phase (mass-weighted).

    1 for a pure radial breathing mode (the RBM of a tube, the breathing of a
    ring about its normal).
    """
    positions = vib.atoms.get_positions()
    centre = positions.mean(axis=0) if centre is None else np.asarray(centre, float)
    pbc = vib.atoms.get_pbc()
    if axis is None:
        axis = vib.atoms.cell[int(np.flatnonzero(pbc)[0])] if pbc.sum() == 1 else \
            np.linalg.svd(positions - centre)[2][0]
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    relative = positions - centre
    radial = relative - np.outer(relative @ axis, axis)
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    pattern = radial * np.sqrt(vib.masses)[:, None]
    pattern /= np.linalg.norm(pattern)
    return np.einsum("mnx,nx->m", vib.eigenvectors, pattern) ** 2


def optical_character(vib: Vibrations, cutoff: float = 1.8) -> np.ndarray:
    """How much each atom moves against its bonded neighbours, in [-1, 1].

    ``-Σ_bonds e_i·e_j / Σ_bonds (|e_i|² + |e_j|²)/2``: 1 when every atom moves
    opposite to all its neighbours (graphene's Γ optical modes, from which
    the G modes of tubes come), about 0 or negative for zone-boundary and
    acoustic-like patterns.
    """
    from ase.neighborlist import neighbor_list

    i, j = neighbor_list("ij", vib.atoms, cutoff)
    keep = i < j
    i, j = i[keep], j[keep]
    e = vib.eigenvectors
    dot = np.einsum("mbx,mbx->mb", e[:, i], e[:, j]).sum(axis=1)
    norm = 0.5 * (np.sum(e[:, i] ** 2, axis=(1, 2)) + np.sum(e[:, j] ** 2, axis=(1, 2)))
    return -dot / np.where(norm > 1e-12, norm, 1.0)


def rotational_symmetry(vib: Vibrations, order: int, axis: Optional[np.ndarray] = None,
                        centre: Optional[np.ndarray] = None, tolerance: float = 0.05
                        ) -> np.ndarray:
    """``⟨e|C e⟩`` for every mode, C the rotation by 2π/order about ``axis``.

    +1 for a mode invariant under the rotation (A symmetry: the RBM, the
    totally symmetric G modes of a nanotube), -1 for one that changes sign,
    in between for degenerate (E) modes. Rotated atoms are matched to atoms
    (modulo the lattice vector along a periodic axis); raises if the
    structure does not have the symmetry.
    """
    positions = vib.atoms.get_positions()
    centre = positions.mean(axis=0) if centre is None else np.asarray(centre, float)
    pbc = vib.atoms.get_pbc()
    period = vib.atoms.cell[int(np.flatnonzero(pbc)[0])] if pbc.sum() == 1 else None
    if axis is None:
        axis = period if period is not None else np.linalg.svd(positions - centre)[2][0]
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    angle = 2 * np.pi / order
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)
    rotated = (positions - centre) @ rotation.T + centre
    image = np.zeros(len(positions), dtype=int)
    for i, r in enumerate(rotated):
        diff = positions - r
        if period is not None:
            length = np.linalg.norm(period)
            along = diff @ period / length ** 2
            diff = diff - np.outer(np.round(along), period)
        distance = np.linalg.norm(diff, axis=1)
        j = int(np.argmin(distance))
        if distance[j] > tolerance:
            raise ValueError(f"La estructura no tiene simetría C{order} alrededor de ese eje.")
        image[i] = j
    e = vib.eigenvectors
    moved = np.zeros_like(e)
    moved[:, image, :] = np.einsum("xy,mny->mnx", rotation, e)
    return np.einsum("mnx,mnx->m", e, moved)


def vibrational_dos(vib: Vibrations, grid: Optional[np.ndarray] = None, sigma: float = 10.0,
                    groups: dict[str, Iterable[int]] | None = None,
                    skip_rigid: bool = True) -> dict[str, np.ndarray]:
    """``g(ω) = Σ_ν w_X(ν) G_σ(ω - ω_ν)``: total and projected (states per cm⁻¹).

    ``groups`` as in :func:`participation` (default by element); the total
    is the sum of the projections. Gaussian of width ``sigma`` cm⁻¹. The
    lowest 6 (finite) or 3 (periodic) modes are dropped with ``skip_rigid``.
    """
    frequencies = vib.frequencies
    keep = np.ones(len(frequencies), dtype=bool)
    if skip_rigid:
        n_rigid = 3 if vib.atoms.get_pbc().any() else 6
        keep[np.argsort(np.abs(frequencies))[:n_rigid]] = False
    grid = np.arange(0.0, max(frequencies.max(), 100.0) + 5 * sigma, 1.0) if grid is None \
        else np.asarray(grid, float)
    gauss = np.exp(-0.5 * ((grid[:, None] - frequencies[None, keep]) / sigma) ** 2) \
        / (np.sqrt(2 * np.pi) * sigma)
    shares = participation(vib, groups)
    out = {name: gauss @ share[keep] for name, share in shares.items()}
    out["total"] = sum(out.values())
    out["grid"] = grid
    return out


def mass_versus_chemistry(vib: Vibrations, atoms: Iterable[int], reference_mass: float = 12.011
                          ) -> dict[str, np.ndarray]:
    """Frequencies with the given atoms' mass set to ``reference_mass`` (carbon).

    Returns ``{"actual", "reference_mass"}``: the difference between the two
    is the mass effect; what remains between the reference-mass spectrum and
    the pristine one is chemical (changed force constants). Modes are matched
    by index after sorting, which is exact when the order does not change.
    """
    swapped = vib.with_masses({int(i): reference_mass for i in atoms})
    return {"actual": vib.frequencies, "reference_mass": swapped.frequencies}


def describe(vib: Vibrations, index: int, groups: dict[str, Iterable[int]] | None = None,
             cylinder: bool = False) -> str:
    """One line per mode: frequency, participation by group, character."""
    shares = participation(vib, groups)
    parts = ", ".join(f"{name} {share[index]:.0%}" for name, share in shares.items()
                      if share[index] >= 0.05)
    text = f"{vib.frequencies[index]:8.1f} cm⁻¹  {parts}"
    if cylinder:
        c = cylindrical_character(vib)
        text += f"  | radial {c['radial'][index]:.0%}, tangencial {c['tangential'][index]:.0%}, " \
                f"axial {c['axial'][index]:.0%}"
    return text
