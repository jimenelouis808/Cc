"""Sites of a structure (rings, heteroatoms, their neighbours) and modes seen from them.

Two uses, both independent of the model:

* **Groups of atoms** to read modes by: atoms on 5-, 6-, 7-membered rings,
  each heteroatom element, the carbons bonded to a heteroatom, hydrogens.
  ``tbkit.modes.participation(vib, site_groups(atoms))`` then says how much of
  each mode sits on each group (shares of the mass-weighted eigenvector; they
  add up to 1 over a partition, and these groups overlap, so read them one by one).
* **The frequency of a given mode under another calculator**:
  ``projected_frequency`` displaces along the mode and measures the restoring
  force, ω² = u·K·u / u·M·u with K·u from forces. Any ASE calculator: GPAW,
  another TB set. It is the Rayleigh quotient: exact for an eigenvector of that
  calculator, an upper bound otherwise (a TB mode that mixes stiffer motions
  under DFT comes out higher). Two force calls per mode (one with ``forces0``),
  instead of a full Hessian (6N calls).

Rings are the simple cycles of the bond graph up to ``largest`` atoms. In an
sp2 net whose faces have at most 7 atoms these are exactly the faces (two faces
sharing a bond already make a cycle of 8 or more).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from ase import Atoms

#: The ring sizes reported as groups.
RING_SIZES = (3, 4, 5, 6, 7, 8)


def _neighbours(atoms: Atoms, cutoff: Optional[float], mult: float):
    from ase.neighborlist import natural_cutoffs, neighbor_list

    radius = float(cutoff) if cutoff is not None else natural_cutoffs(atoms, mult=mult)
    return neighbor_list("ijS", atoms, radius)


def bond_graph(atoms: Atoms, cutoff: Optional[float] = None, mult: float = 1.2) -> set:
    """Pairs ``(i, j)``, ``i < j``, closer than ``cutoff`` (Å), or than ``mult`` × the sum
    of covalent radii when ``cutoff`` is None (1.82 Å for C-C)."""
    i, j, _ = _neighbours(atoms, cutoff, mult)
    return {(int(a), int(b)) for a, b in zip(i, j, strict=True) if a < b}


def ring_cycles(atoms: Atoms, largest: int = 7, cutoff: Optional[float] = None
                ) -> list[tuple[tuple, np.ndarray]]:
    """Every simple cycle of at most ``largest`` atoms that closes in space, as
    ``(atoms in path order, lattice offsets of each, (n, 3) integers)``.

    Lattice offsets are followed along the walk: a path that returns to its first
    atom in another cell wraps the boundary and is not a ring (a 3×3 graphene cell
    has such 6-cycles). Two rings are the same only if their atoms *and* offsets
    agree up to a lattice translation: in a √3×√3 graphene cell the three hexagons
    are made of the same six atoms in different images."""
    i, j, shifts = _neighbours(atoms, cutoff, 1.2)
    adjacency = {k: [] for k in range(len(atoms))}
    for a, b, shift in zip(i, j, shifts, strict=True):
        adjacency[int(a)].append((int(b), tuple(int(x) for x in shift)))
    found = {}

    def walk(path, offsets):
        u, offset = path[-1], offsets[-1]
        for v, shift in adjacency[u]:
            total = (offset[0] + shift[0], offset[1] + shift[1], offset[2] + shift[2])
            if v == path[0] and len(path) >= 3:
                if total == (0, 0, 0):
                    key = frozenset(zip(path, offsets))
                    found.setdefault(key, (tuple(path), np.array(offsets)))
            elif v > path[0] and len(path) < largest and \
                    (v, total) not in set(zip(path, offsets)):
                walk(path + [v], offsets + [total])

    for start in range(len(atoms)):
        walk([start], [(0, 0, 0)])
    return sorted(found.values(), key=lambda c: (len(c[0]), sorted(c[0])))


def rings(atoms: Atoms, largest: int = 7, cutoff: Optional[float] = None) -> list[tuple]:
    """Every ring (see :func:`ring_cycles`) as a sorted tuple of atom indices; in a
    small periodic cell two rings may share the same indices."""
    return [tuple(sorted(path)) for path, _ in ring_cycles(atoms, largest, cutoff)]


def ring_census(atoms: Atoms, largest: int = 7, cutoff: Optional[float] = None) -> dict:
    counts: dict[int, int] = {}
    for ring in rings(atoms, largest, cutoff):
        counts[len(ring)] = counts.get(len(ring), 0) + 1
    return dict(sorted(counts.items()))


def ring_atoms(atoms: Atoms, sizes=(5, 7), largest: int = 7,
               cutoff: Optional[float] = None) -> dict:
    out = {s: set() for s in sizes}
    for ring in rings(atoms, largest, cutoff):
        if len(ring) in out:
            out[len(ring)].update(ring)
    return {s: sorted(v) for s, v in out.items()}


def site_groups(atoms: Atoms, largest: int = 7, cutoff: Optional[float] = None
                ) -> dict[str, list[int]]:
    """Named groups of atoms, only the non-empty ones, in a fixed order."""
    symbols = np.array(atoms.get_chemical_symbols())
    groups: dict[str, list[int]] = {}
    on = ring_atoms(atoms, tuple(s for s in RING_SIZES if s <= largest), largest, cutoff)
    for size, members in on.items():
        if members:
            groups[f"anillos de {size}"] = members
    hetero = [el for el in dict.fromkeys(symbols) if el not in ("C", "H")]
    for element in hetero:
        groups[element] = [int(i) for i in np.flatnonzero(symbols == element)]
    if hetero:
        heavy = set(np.flatnonzero(np.isin(symbols, hetero)).tolist())
        near = set()
        for a, b in bond_graph(atoms, cutoff):
            if a in heavy and symbols[b] == "C":
                near.add(b)
            if b in heavy and symbols[a] == "C":
                near.add(a)
        if near:
            groups["C vecinos de heteroátomo"] = sorted(int(i) for i in near)
    if "H" in symbols:
        groups["H"] = [int(i) for i in np.flatnonzero(symbols == "H")]
    return groups


def projected_frequency(atoms: Atoms, mode: np.ndarray, calculator: Callable[[], object],
                        max_disp: float = 0.02, forces0: Optional[np.ndarray] = None,
                        masses: Optional[np.ndarray] = None) -> dict:
    """Frequency (cm⁻¹) of the Cartesian displacement pattern ``mode`` (N, 3) under the
    calculator ``calculator()`` makes (a new one per force call).

    Central difference (two force calls) unless ``forces0`` (forces at ``atoms``)
    is given: then one call, F(x + s u) - F(x). The step s moves the most
    displaced atom by ``max_disp`` Å. A negative result means negative curvature."""
    from ase.units import _amu, _e, _hbar, invcm

    u = np.asarray(mode, dtype=float).reshape(len(atoms), 3)
    masses = atoms.get_masses() if masses is None else np.asarray(masses, dtype=float)
    step = max_disp / float(np.linalg.norm(u, axis=1).max())

    def forces_at(sign):
        probe = atoms.copy()
        probe.positions += sign * step * u
        probe.calc = calculator()
        return probe.get_forces()

    if forces0 is None:
        k_u = -(forces_at(+1) - forces_at(-1)) / (2 * step)
        calls = 2
    else:
        k_u = -(forces_at(+1) - np.asarray(forces0)) / step
        calls = 1
    curvature = float(np.sum(u * k_u))                      # eV/Å² per unit u
    inertia = float(np.sum(masses[:, None] * u ** 2))        # amu per unit u
    scale = _hbar * 1e10 / np.sqrt(_e * _amu)                # eV for k in eV/Å², m in amu
    omega = scale * np.sqrt(abs(curvature) / inertia) / invcm
    return {"frequency_cm1": float(np.sign(curvature) * omega), "step_A": step,
            "force_calls": calls}


def ring_breathing(atoms: Atoms, modes: np.ndarray, masses: Optional[np.ndarray] = None,
                   largest: int = 7, cutoff: Optional[float] = None) -> dict:
    """How much each mode is a breathing of the rings: the character of the D band.

    For ring R and mode k, ``b_R = Σ_{i∈R} e_i·r̂_i / √n_R`` with e the mass-weighted
    displacement and r̂_i the in-plane radial direction from the ring's centroid
    (plane fitted to the ring, so curved nets work). The breathing character is
    ``B_k = Σ_R b_R² / Σ_R Σ_{i∈R} |e_i|²``: 1 for one isolated ring breathing,
    0 for a mode that translates every ring rigidly (graphene's G, E2g). The D band
    of graphitic carbon comes from the K-point A1' mode, a Kekulé pattern of
    breathing rings (Castiglioni, Tommasini et al.); in a structure without
    translational order the modes that carry it are found by B, not by frequency.

    Returns ``{"B": (n_modes,), "rings": [...], "amplitudes": (n_modes, n_rings)}``.
    """
    modes = np.asarray(modes, dtype=float)
    masses = atoms.get_masses() if masses is None else np.asarray(masses, dtype=float)
    e = modes * np.sqrt(masses)[None, :, None]                     # L = e/√m -> e
    e /= np.linalg.norm(e.reshape(len(e), -1), axis=1)[:, None, None]
    faces = ring_cycles(atoms, largest, cutoff)
    positions = atoms.get_positions()
    cell = np.asarray(atoms.get_cell())
    amplitudes = np.zeros((len(e), len(faces)))
    norm = np.zeros(len(e))
    for r, (path, offsets) in enumerate(faces):
        idx = list(path)
        pts = positions[idx] + offsets @ cell
        centre = pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts - centre)
        normal = vt[2]
        radial = pts - centre
        radial -= np.outer(radial @ normal, normal)
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        amplitudes[:, r] = np.einsum("kij,ij->k", e[:, idx], radial) / np.sqrt(len(idx))
        norm += np.sum(e[:, idx] ** 2, axis=(1, 2))
    breathing = np.sum(amplitudes ** 2, axis=1) / np.where(norm > 0, norm, 1.0)
    return {"B": breathing, "rings": [tuple(sorted(f[0])) for f in faces],
            "amplitudes": amplitudes}
