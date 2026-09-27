"""Γ phonons from Quantum ESPRESSO, for Raman intensities from the TB model.

``dynmat.x`` (``filout``, ``fileig``) and ``matdyn.x`` (``flvec``, ``fleig``)
print the modes in one layout::

     q =       0.0000      0.0000      0.0000
     **************************************************************
     freq (    1) =       1.234567 [THz] =      41.180000 [cm-1]
     ( 0.123  0.000   -0.456  0.000    0.000  0.000 )
     ...one row per atom: real and imaginary parts of x, y, z
     **************************************************************

Some of these files hold eigenvectors of the dynamical matrix (orthonormal),
others normalised displacements (eigenvectors divided by √m, then normalised;
not orthogonal). :func:`normal_mode_vectors` turns either into what Raman
needs, the Cartesian displacement per unit normal coordinate ``L = e/√m``
with ``e`` normalised: from a displacement ``u``, ``e ∝ u √m``. So it does
not matter which file is given, as long as the masses are the ones used in
the phonon run (ASE's standard masses otherwise: isotopes change them).

Only Γ is read (q = 0: the modes first-order Raman sees). Atom order must
match the structure's; the file has no positions, so that is checked only
through the atom count and, for mode symmetry, is the user's responsibility.
Like the rest of tbkit, the parser is tested against files written here in
the documented layout, not yet against output of a real QE installation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FREQ = re.compile(r"freq\s*\(\s*(\d+)\s*\)\s*=\s*([-+\d.EeDd]+)\s*\[THz\]\s*=\s*"
                   r"([-+\d.EeDd]+)\s*\[cm-1\]")
_Q = re.compile(r"^\s*q\s*=\s*([-+\d.Ee]+)\s+([-+\d.Ee]+)\s+([-+\d.Ee]+)")


@dataclass
class QEModes:
    """Γ modes as read: frequencies (cm⁻¹, negative = imaginary) and vectors."""

    frequencies: np.ndarray            # (n_modes,)
    vectors: np.ndarray                # (n_modes, n_atoms, 3), complex, as printed
    source: str


def _float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def read_qe_modes(path: str | Path) -> QEModes:
    """Read the q = 0 modes of a ``dynmat.x``/``matdyn.x`` mode file."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    blocks: list[tuple[np.ndarray, list[tuple[float, list[list[float]]]]]] = []
    current_q = np.zeros(3)
    modes: list[tuple[float, list[list[float]]]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        q = _Q.match(line)
        if q:
            if modes:
                blocks.append((current_q, modes))
            current_q = np.array([_float(v) for v in q.groups()])
            modes = []
            i += 1
            continue
        freq = _FREQ.search(line)
        if freq:
            rows = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("("):
                values = [_float(v) for v in lines[i].replace("(", " ").replace(")", " ").split()]
                if len(values) < 6:
                    raise ValueError(f"{path.name}: fila de modo ilegible: {lines[i]!r}")
                rows.append(values[:6])
                i += 1
            modes.append((_float(freq.group(3)), rows))
            continue
        i += 1
    if modes:
        blocks.append((current_q, modes))
    gamma = [m for q, m in blocks if np.allclose(q, 0.0, atol=1e-6)]
    if not gamma:
        raise ValueError(f"{path.name}: no hay modos en q = 0 (Γ).")
    modes = gamma[0]
    n_atoms = {len(rows) for _, rows in modes}
    if len(n_atoms) != 1:
        raise ValueError(f"{path.name}: los modos no tienen todos el mismo número de átomos.")
    frequencies = np.array([f for f, _ in modes])
    vectors = np.array([[[row[0] + 1j * row[1], row[2] + 1j * row[3], row[4] + 1j * row[5]]
                         for row in rows] for _, rows in modes])
    return QEModes(frequencies, vectors, path.name)


def degenerate_sets(frequencies: np.ndarray, tolerance: float = 0.5) -> list[list[int]]:
    """Indices of modes whose frequencies agree within ``tolerance`` cm⁻¹."""
    order = np.argsort(frequencies)
    sets: list[list[int]] = []
    for index in order:
        if sets and abs(frequencies[index] - frequencies[sets[-1][-1]]) <= tolerance:
            sets[-1].append(int(index))
        else:
            sets.append([int(index)])
    return sets


def real_basis(e: np.ndarray, frequencies: np.ndarray, tolerance: float = 0.5) -> np.ndarray:
    """Real orthonormal mass-weighted modes from complex ones, per degenerate set.

    A Γ mode is real up to a global phase, but inside a degenerate set QE may
    print complex combinations ((x + iy)/√2). The real and imaginary parts of
    the set span its real subspace; the leading singular vectors give an
    orthonormal real basis of it. The Raman activity summed over the set does
    not depend on which basis is used.
    """
    out = np.zeros(e.shape)
    for members in degenerate_sets(frequencies, tolerance):
        flat = e[members].reshape(len(members), -1)
        stacked = np.vstack([flat.real, flat.imag])
        u, sigma, vt = np.linalg.svd(stacked, full_matrices=False)
        if len(sigma) > len(members) and sigma[len(members)] > 1e-3 * sigma[0]:
            raise ValueError("Modos en Γ que no son reales en su subespacio degenerado: "
                             "¿es de verdad q = 0?")
        basis = vt[:len(members)]
        for k, index in enumerate(members):
            out[index] = basis[k].reshape(e.shape[1:])
    return out


def normal_mode_vectors(vectors: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """``L = e/√m`` (Å per unit normal coordinate) from displacement patterns ``u``.

    ``e ∝ u √m``, normalised, then divided by √m: independent of how ``u``
    was normalised.
    """
    sqrt_m = np.sqrt(np.asarray(masses, dtype=float))[None, :, None]
    e = np.asarray(vectors) * sqrt_m
    norms = np.linalg.norm(e.reshape(len(e), -1), axis=1)
    return e / np.where(norms > 0, norms, 1.0)[:, None, None] / sqrt_m


def is_orthonormal(vectors: np.ndarray, tol: float = 1e-3) -> bool:
    """Whether the modes, flattened, form an orthonormal set (eigenvectors do)."""
    flat = np.asarray(vectors).reshape(len(vectors), -1)
    return bool(np.allclose(flat.conj() @ flat.T, np.eye(len(flat)), atol=tol))


def modes_for_raman(modes: QEModes, masses: np.ndarray, kind: str = "auto") -> np.ndarray:
    """``L`` for every mode, given what the file holds.

    ``kind="displacements"`` (``filout`` of dynmat.x, ``flvec`` of matdyn.x),
    ``"eigenvectors"`` (``fileig``, ``fleig``: mass-weighted and orthonormal,
    so ``L = e/√m`` directly), or ``"auto"``: eigenvectors when the set is
    orthonormal, displacements otherwise (with equal masses both readings give
    the same ``L``).
    """
    masses = np.asarray(masses, dtype=float)
    if modes.vectors.shape[1] != len(masses):
        raise ValueError(f"{modes.source}: {modes.vectors.shape[1]} átomos en los modos y "
                         f"{len(masses)} en la estructura.")
    if kind == "auto":
        kind = "eigenvectors" if is_orthonormal(modes.vectors) else "displacements"
    sqrt_m = np.sqrt(masses)[None, :, None]
    if kind == "displacements":
        e = normal_mode_vectors(modes.vectors, masses) * sqrt_m
    elif kind == "eigenvectors":
        flat = modes.vectors.reshape(len(modes.vectors), -1)
        e = (flat / np.linalg.norm(flat, axis=1)[:, None]).reshape(modes.vectors.shape)
    else:
        raise ValueError(f"kind debe ser 'displacements' o 'eigenvectors', no {kind!r}.")
    return real_basis(e, modes.frequencies) / sqrt_m


def write_qe_modes(path: str | Path, frequencies_cm1, vectors, kind: str = "displacements",
                   masses=None) -> Path:
    """Write modes in the dynmat.x layout (tests, and exporting tbkit's own modes).

    ``vectors`` are ``L = e/√m``; they are written as normalised displacements
    (``kind="displacements"``) or as eigenvectors ``e`` (needs ``masses``).
    """
    vectors = np.asarray(vectors, dtype=float)
    if kind == "eigenvectors":
        vectors = vectors * np.sqrt(np.asarray(masses, dtype=float))[None, :, None]
    lines = ["     diagonalizing the dynamical matrix ...", "",
             " q =       0.0000      0.0000      0.0000",
             " " + "*" * 74]
    for k, (freq, vec) in enumerate(zip(frequencies_cm1, vectors, strict=True), start=1):
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm > 0 else vec
        lines.append(f"     freq ({k:5d}) = {freq / 33.35641:14.6f} [THz] = {freq:14.6f} [cm-1]")
        for x, y, z in vec:
            lines.append(f" ( {x:10.6f} {0.0:10.6f} {y:10.6f} {0.0:10.6f} {z:10.6f} {0.0:10.6f} )")
    lines.append(" " + "*" * 74)
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
