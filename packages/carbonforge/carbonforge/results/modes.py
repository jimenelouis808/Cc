"""Normal modes: read them, find the one behind a peak, and animate it.

``dynmat.x`` writes the eigen-displacements of every mode to an animated
XCrySDen file (``filout = 'dynmat.axsf'`` in carbonforge's inputs); vibspec
stores its own in ``modes.npz``. Both end up as the same thing -- a structure
and one displacement vector per mode -- and the helpers here serve both:
the mode nearest a clicked wavenumber, the share of motion per element, and
the frames of one oscillation.

Like the other readers in :mod:`carbonforge.results`, the ``.axsf`` parser is
tested against synthetic files written in the documented format, not yet
against output from a real QE installation.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols

_KEYWORDS = re.compile(r"^(ANIMSTEPS|CRYSTAL|SLAB|POLYMER|MOLECULE|PRIMVEC|CONVVEC|"
                       r"PRIMCOORD|ATOMS)\b")


def _symbol(token: str) -> str:
    return chemical_symbols[int(token)] if token.isdigit() else token


def read_axsf_modes(path: str | Path) -> tuple[Atoms, list[np.ndarray]]:
    """Read an animated ``.axsf`` (``dynmat.x`` ``filout``) into modes.

    Returns
    -------
    (atoms, vectors)
        The equilibrium structure (with the cell when the file has one) and,
        per mode in file order -- the order of the ``dynmat.x`` table -- an
        ``(n_atoms, 3)`` displacement array.

    Raises
    ------
    ValueError
        If no mode blocks are found.
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    cell: Optional[np.ndarray] = None
    blocks: list[list[list[str]]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("PRIMVEC"):
            cell = np.array([[float(x) for x in lines[i + k].split()[:3]] for k in (1, 2, 3)])
            i += 4
            continue
        if line.startswith(("PRIMCOORD", "ATOMS")):
            i += 1
            count = None
            if line.startswith("PRIMCOORD") and i < len(lines):
                count = int(lines[i].split()[0])
                i += 1
            rows = []
            while i < len(lines) and lines[i].strip() and not _KEYWORDS.match(lines[i].strip()) \
                    and (count is None or len(rows) < count):
                rows.append(lines[i].split())
                i += 1
            blocks.append(rows)
            continue
        i += 1
    if not blocks:
        raise ValueError(f"{Path(path).name}: no hay bloques PRIMCOORD/ATOMS con modos.")
    first = blocks[0]
    atoms = Atoms([_symbol(row[0]) for row in first],
                  positions=[[float(x) for x in row[1:4]] for row in first])
    if cell is not None:
        atoms.cell = cell
        atoms.pbc = True
    vectors = []
    for rows in blocks:
        if len(rows) != len(first):
            raise ValueError(f"{Path(path).name}: los bloques no tienen el mismo número de átomos.")
        vectors.append(np.array([[float(x) for x in row[4:7]] if len(row) >= 7 else [0.0] * 3
                                 for row in rows]))
    return atoms, vectors


def normalised(vector: np.ndarray) -> np.ndarray:
    """The displacement scaled so that the largest atomic excursion is 1 Å."""
    vector = np.asarray(vector, dtype=float)
    largest = float(np.linalg.norm(vector, axis=1).max()) if vector.size else 0.0
    return vector / largest if largest > 0 else vector


def nearest_mode(frequencies, activities, wavenumber: float,
                 window_cm1: float = 40.0) -> Optional[int]:
    """Index of the mode behind a band clicked at ``wavenumber``.

    The most active mode within ``window_cm1``; if none is that close, the
    nearest one. ``activities`` may be None (then only distance counts).
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.size == 0:
        return None
    distance = np.abs(frequencies - wavenumber)
    near = np.flatnonzero(distance <= window_cm1)
    if activities is not None and near.size:
        weights = np.nan_to_num(np.asarray(activities, dtype=float))
        return int(near[np.argmax(weights[near])])
    return int(np.argmin(distance))


def mode_frames(atoms: Atoms, vector: np.ndarray, n_frames: int = 24,
                amplitude: float = 0.35) -> list[np.ndarray]:
    """Positions for one period of the mode, largest excursion ``amplitude`` Å."""
    base = atoms.get_positions()
    phases = np.sin(2 * np.pi * np.arange(n_frames) / n_frames)
    return [base + amplitude * phase * np.asarray(vector) for phase in phases]


def view_angles(atoms: Atoms) -> tuple[float, float]:
    """Matplotlib ``(elev, azim)`` that looks straight down on a planar molecule.

    The view direction is the principal axis of least spread -- the plane
    normal of a flake, whichever way it lies -- so a mode in the plane is
    seen face-on instead of edge-on.
    """
    positions = atoms.get_positions() - atoms.get_positions().mean(axis=0)
    _, _, vt = np.linalg.svd(positions, full_matrices=False)
    normal = vt[-1] if len(vt) == 3 else np.array([0.0, 0.0, 1.0])
    x, y, z = normal / np.linalg.norm(normal)
    return float(np.degrees(np.arcsin(np.clip(z, -1, 1)))), float(np.degrees(np.arctan2(y, x)))


def mode_character(atoms: Atoms, vector: np.ndarray, top: int = 4) -> str:
    """Which atoms carry the motion, as one line.

    Shares are of the mass-weighted kinetic energy, the usual measure of how
    much of a mode belongs to each atom; an N-H stretch shows up as mostly H
    with some N, whatever the rest of the ribbon does.
    """
    weights = atoms.get_masses() * (np.asarray(vector) ** 2).sum(axis=1)
    total = weights.sum()
    if total <= 0:
        return "sin desplazamiento"
    share = weights / total
    symbols = atoms.get_chemical_symbols()
    order = np.argsort(share)[::-1][:top]
    by_element: dict[str, float] = {}
    for symbol, value in zip(symbols, share, strict=True):
        by_element[symbol] = by_element.get(symbol, 0.0) + float(value)
    elements = ", ".join(f"{k} {v:.0%}" for k, v in sorted(by_element.items(),
                                                            key=lambda kv: -kv[1]) if v >= 0.05)
    atoms_text = ", ".join(f"{symbols[i]}{i} {share[i]:.0%}" for i in order)
    return f"{elements}  |  {atoms_text}"
