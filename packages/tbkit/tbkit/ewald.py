"""Periodic γ for self-consistent charges: Ewald sum plus a short-range part.

For a periodic system the SCC energy ``½ Σ_AB Δq_A γ_AB Δq_B`` needs γ summed
over every image of B,

    γ_AB = Σ_L γ_KO(|R_B + L - R_A|),     γ_KO(r) = C / sqrt(r² + a²),

with ``C = e²/(4πε0)`` and ``a = C/Ū``. That sum converges only
conditionally (γ_KO ~ C/r), so it is split as in Elstner et al., Phys. Rev.
B 58, 7260 (1998):

* ``C/r`` summed by Ewald (real and reciprocal parts; a uniform neutralising
  background for a charged cell, as in plane-wave DFT);
* its asymptote ``-C a²/(2 r³)`` by the Ewald sum for 1/r³ (real part with
  Γ(3/2, α²r²), reciprocal part with E1). Σ_L 1/r³ diverges in three
  dimensions only through its G = 0 term, a constant: it cancels for a
  neutral cell of one element and is dropped otherwise (a convention, like
  the neutralising background; it is what summing by image shells gives);
* the rest ``γ_KO - C/r + C a²/(2 r³)``, which falls as r⁻⁵, in real space,
  switched off smoothly between ``r_short - 5`` and ``r_short`` Å (value and
  slope continuous, so forces stay derivatives of the energy). Summing the
  whole r⁻³ tail that way instead oscillated by 10⁻² eV between 40 and 60 Å.

``L = 0, A = B`` is the on-site term, ``γ_AA = U_A``. Directions that are not
periodic still need a cell: they are treated as periodic with the vacuum the
cell gives them (a supercell, as in plane-wave DFT), so the vacuum must be
large enough for images not to interact (tested against the finite result).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

#: e² / (4 π ε0), eV Å.
COULOMB = 14.399645


def _lattice_vectors(cell: np.ndarray, radius: float) -> np.ndarray:
    """Every lattice vector L = n·cell with |L + d| possibly below ``radius`` for
    any d inside the cell (a generous box of integer triples)."""
    inverse_norms = np.linalg.norm(np.linalg.inv(cell), axis=0)   # 1 / plane spacings
    reach = np.ceil(radius * inverse_norms).astype(int) + 1
    n = np.stack(np.meshgrid(*[np.arange(-m, m + 1) for m in reach], indexing="ij"),
                 axis=-1).reshape(-1, 3)
    return n @ cell


def ewald_coulomb(positions: np.ndarray, cell: np.ndarray, tolerance: float = 1e-10
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Φ_AB = Σ'_L 1/|R_B + L - R_A| (1/Å; the prime drops L = 0 when A = B), with a
    neutralising background, and its gradient ``∂Φ_AB/∂R_A`` (N, N, 3).

    Only charge-neutral combinations ``Σ_AB q_A q_B Φ_AB`` are physical (the
    background absorbs the rest); the Madelung constant of NaCl is its test.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    volume = abs(np.linalg.det(cell))
    n = len(positions)
    alpha = np.sqrt(np.pi) * (max(n, 1) / volume ** 2) ** (1 / 6)
    log_tol = np.sqrt(-np.log(tolerance))
    r_cut = log_tol / alpha
    g_cut = 2 * alpha * log_tol
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    d = positions[None, :, :] - positions[:, None, :]            # R_B - R_A
    phi = np.zeros((n, n))
    grad = np.zeros((n, n, 3))
    # Real space.
    for lattice in _lattice_vectors(cell, r_cut):
        r_vec = d + lattice
        r = np.linalg.norm(r_vec, axis=-1)
        mask = (r > 1e-12) & (r < r_cut)
        if not mask.any():
            continue
        rm = r[mask]
        phi[mask] += erfc(alpha * rm) / rm
        dphi_dr = -(erfc(alpha * rm) / rm ** 2
                    + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha * rm) ** 2) / rm)
        grad[mask] -= (dphi_dr / rm)[:, None] * r_vec[mask]      # ∂/∂R_A = -∂/∂r
    # Reciprocal space.
    for g in _lattice_vectors(reciprocal, g_cut):
        g2 = float(g @ g)
        if g2 < 1e-20 or g2 > g_cut ** 2:
            continue
        factor = 4 * np.pi / volume * np.exp(-g2 / (4 * alpha ** 2)) / g2
        phase = d @ g
        phi += factor * np.cos(phase)
        grad += factor * np.sin(phase)[..., None] * g           # -∂cos/∂r · (-1)
    # Self term and background.
    phi -= np.pi / (alpha ** 2 * volume)
    phi[np.diag_indices(n)] -= 2 * alpha / np.sqrt(np.pi)
    return phi, grad


def ewald_inverse_cube(positions: np.ndarray, cell: np.ndarray, tolerance: float = 1e-10
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Φ3_AB = Σ'_L 1/|R_B + L - R_A|³ (1/Å³) without its divergent G = 0 constant,
    and ``∂Φ3_AB/∂R_A``."""
    from scipy.special import exp1

    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    volume = abs(np.linalg.det(cell))
    n = len(positions)
    alpha = np.sqrt(np.pi) * (max(n, 1) / volume ** 2) ** (1 / 6)
    log_tol = np.sqrt(-np.log(tolerance))
    r_cut = log_tol / alpha
    g_cut = 2 * alpha * log_tol
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    d = positions[None, :, :] - positions[:, None, :]
    phi = np.zeros((n, n))
    grad = np.zeros((n, n, 3))
    for lattice in _lattice_vectors(cell, r_cut):
        r_vec = d + lattice
        r = np.linalg.norm(r_vec, axis=-1)
        mask = (r > 1e-12) & (r < r_cut)
        if not mask.any():
            continue
        rm = r[mask]
        x = alpha * rm
        numerator = erfc(x) + 2 * x / np.sqrt(np.pi) * np.exp(-x ** 2)
        phi[mask] += numerator / rm ** 3
        dnum = -4 * alpha ** 3 * rm ** 2 / np.sqrt(np.pi) * np.exp(-x ** 2)
        dphi_dr = dnum / rm ** 3 - 3 * numerator / rm ** 4
        grad[mask] -= (dphi_dr / rm)[:, None] * r_vec[mask]
    for g in _lattice_vectors(reciprocal, g_cut):
        g2 = float(g @ g)
        if g2 < 1e-20 or g2 > g_cut ** 2:
            continue
        factor = 2 * np.pi / volume * exp1(g2 / (4 * alpha ** 2))
        phase = d @ g
        phi += factor * np.cos(phase)
        grad += factor * np.sin(phase)[..., None] * g
    phi[np.diag_indices(n)] -= 4 * alpha ** 3 / (3 * np.sqrt(np.pi))
    return phi, grad


def _taper(r: np.ndarray, r_short: float, width: float = 5.0):
    x = np.clip((r - (r_short - width)) / width, 0.0, 1.0)
    return 1 - 3 * x ** 2 + 2 * x ** 3, (-6 * x + 6 * x ** 2) / width


def periodic_gamma(positions: np.ndarray, cell: np.ndarray, u_atom: np.ndarray,
                   r_short: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """γ_AB summed over images (eV) and ``∂γ_AB/∂R_A`` (eV/Å), see the module notes."""
    positions = np.asarray(positions, dtype=float)
    u_atom = np.asarray(u_atom, dtype=float)
    phi, dphi = ewald_coulomb(positions, cell)
    n = len(positions)
    a = COULOMB / (0.5 * (u_atom[:, None] + u_atom[None, :]))
    phi3, dphi3 = ewald_inverse_cube(positions, cell)
    tail = -0.5 * COULOMB * a ** 2
    gamma = COULOMB * phi + tail * phi3
    grad = COULOMB * dphi + tail[..., None] * dphi3
    d = positions[None, :, :] - positions[:, None, :]
    for lattice in _lattice_vectors(np.asarray(cell, dtype=float), r_short):
        r_vec = d + lattice
        r = np.linalg.norm(r_vec, axis=-1)
        mask = (r > 1e-12) & (r < r_short)
        if not mask.any():
            continue
        rm, am = r[mask], a[mask]
        ko = COULOMB / np.sqrt(rm ** 2 + am ** 2)
        short = ko - COULOMB / rm + 0.5 * COULOMB * am ** 2 / rm ** 3
        dshort = (-rm * ko ** 3 / COULOMB ** 2 + COULOMB / rm ** 2
                  - 1.5 * COULOMB * am ** 2 / rm ** 4)
        s, ds = _taper(rm, r_short)
        gamma[mask] += short * s
        derivative = dshort * s + short * ds
        grad[mask] -= (derivative / rm)[:, None] * r_vec[mask]
    gamma[np.diag_indices(n)] += u_atom           # on-site, L = 0
    return gamma, grad
