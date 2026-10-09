"""First- and second-order resonant Raman for any periodic tbkit system, by perturbation theory.

:mod:`tbkit.graphene` computes graphene's G and double-resonant 2D band with
the analytic two-band π model. This module does the same calculation for any
periodic structure a tbkit model describes — a nanocoil, a doped supercell —
with all its bands, following the same third- and fourth-order perturbation
theory (Thomsen and Reich, Phys. Rev. Lett. 85, 5214 (2000); Venezuela,
Lazzeri and Mauri, Phys. Rev. B 84, 035433 (2011)).

**Momentum by supercell.** Phonons of momentum q ≠ 0 enter as Γ modes of a
supercell in which q is commensurate (zone folding): a supercell of N periods
along an axis holds q = 2πm/(N a) along it. The electron states are those of
the supercell at its own k points, so momentum conservation is exact and
needs no bookkeeping: an amplitude between states of different primitive
momentum simply vanishes. The second-order (two-phonon) process of a pair of
supercell modes (μ, μ') is the 2D (overtone) or the D+D' (combination) of the
primitive crystal; a supercell that contains a defect gives, at first order,
the defect-activated D band (the defect provides the elastic scattering).

**The amplitudes.** With ``P^a_{nm} = ⟨n|∂_{k_a} H|m⟩`` (velocity gauge, the
optical vertex), ``g^μ_{nm}`` the electron-phonon matrix element of one
quantum of mode μ, and the electron-hole pair (c, v) as the intermediate
state, one phonon vertex acts on a pair amplitude X[c, v] as

    G_μ(X) = g^μ_cc X − X g^μ_vv        (the electron scatters, or the hole)

and each intermediate pair carries ``D(c, v; ω) = ħω_L − ω − (ε_c − ε_v) + iγ``.
Then, summed over k with the mesh weights,

    A^(1)_μ    = Σ_k Σ_cv P^s*_cv · G_μ(P^i/D(·;0)) / D(·; ω_μ)
    A^(2)_μμ'  = Σ_k Σ_cv P^s*_cv · G_μ'(G_μ(P^i/D(·;0)) / D(·;ω_μ)) / D(·; ω_μ + ω_μ')
                 + (μ ↔ μ')

The vertex form contains, without enumerating them, the four time orderings
of the double-resonance literature (ee, hh, eh, he) and all intra- and
interband scatterings. For graphene's two π bands it is exactly
``graphene.g_band`` and ``graphene._four_processes`` (tested).

**Electron-phonon matrix elements** are frozen-phonon finite differences of
the model's H(k) and S(k) along each mode: ``g = c†(∂H − ε̄ ∂S)c`` with ε̄ the
mean of the two energies. The self-consistent charge shifts of the ground
state are kept fixed (unscreened coupling); that is stated, not hidden.

What it is not: excitonic effects, the Kohn-anomaly renormalisation beyond
the model's own phonons, and energies of resonance that are optical: they are
the model's (small TB gaps), as everywhere in tbkit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from scipy.linalg import eigh

from .graphene import CM1_TO_EV, zero_point
from .hamiltonian import System
from .optics import _bloch_ii


@dataclass
class Mode:
    """A Γ phonon of the (super)cell: frequency (cm⁻¹) and mass-weighted eigenvector
    ``e`` (n_atoms, 3), normalised over the cell (Σ|e|² = 1)."""

    frequency: float
    vector: np.ndarray


def real_modes(frequencies: np.ndarray, vectors: np.ndarray, tol: float = 0.5) -> list[Mode]:
    """Real Γ eigenvectors (a frozen phonon is a real displacement) from complex ones:
    within each set degenerate to ``tol`` cm⁻¹, an orthonormal real basis of the span of
    the real and imaginary parts."""
    frequencies = np.asarray(frequencies, float)
    vectors = np.asarray(vectors)
    shape = vectors.shape[1:]
    order = np.argsort(frequencies)
    out, start = [], 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and frequencies[order[stop]] - frequencies[order[start]] < tol:
            stop += 1
        group = order[start:stop]
        flat = vectors[group].reshape(len(group), -1)
        u, _, _ = np.linalg.svd(np.concatenate([flat.real, flat.imag]).T, full_matrices=False)
        basis = u[:, :len(group)].T
        for f, b in zip(frequencies[group], basis, strict=True):
            out.append(Mode(float(f), b.reshape(shape)))
        start = stop
    return out


@dataclass
class KData:
    """Everything the amplitudes need at one k point (states inside the window)."""

    weight: float
    ev: np.ndarray                     # (n_v,) valence energies
    ec: np.ndarray                     # (n_c,)
    velocity: np.ndarray               # (2, n_c, n_v) P^a_cv for the polarisations
    g_cc: np.ndarray                   # (n_modes, n_c, n_c)
    g_vv: np.ndarray                   # (n_modes, n_v, n_v)


def _hk(system: System, k, shift):
    h, s, dh, ds = _bloch_ii(system, k)
    if shift is not None:
        if s is None:
            h = h + np.diag(shift)
        else:
            sym = 0.5 * (shift[:, None] + shift[None, :])
            h = h + s * sym
            dh = np.array([d + dsa * sym for d, dsa in zip(dh, ds, strict=True)])
    return h, s, dh, ds


def _hk_fixed_phase(system: System, k, shift, phase: np.ndarray):
    """H(k), S(k) of a displaced system in the positions gauge of the *undisplaced* one:
    lattice-shift Bloch sums (no position in the phase), then the equilibrium phases
    ``e^{ik·r_a⁰}``. Taking the positions of the displaced atoms into the phase would add
    a change of basis to ∂H/∂u (a term ∝ k), which is not electron-phonon coupling."""
    h, s = system.hamiltonian(k)
    h = h.astype(complex)
    if shift is not None:
        h = h + (np.diag(shift) if s is None else s * 0.5 * (shift[:, None] + shift[None, :]))
    h = phase.conj()[:, None] * h * phase[None, :]
    if s is not None:
        s = phase.conj()[:, None] * s.astype(complex) * phase[None, :]
    return h, s


def electron_phonon(atoms: Atoms, model, modes: list[Mode], kpts: np.ndarray,
                    weights: np.ndarray, window_ev: float, fermi: float | None = None,
                    polarizations=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    step: float = 0.01, shift: np.ndarray | None = None) -> list[KData]:
    """States, optical and electron-phonon matrix elements on ``kpts`` (fractional).

    Bands are occupied by index at T = 0 (an even electron count per cell is required);
    only states within ``window_ev`` of the Fermi level enter (``fermi``; by default the
    middle between the highest occupied and lowest empty level of the k set). ``step`` (Å) is the largest atomic displacement
    of the finite difference along each mode. ``shift``: fixed orbital potentials (the
    SCC shifts of the ground state), or None.
    """
    system = System.build(atoms, model)
    masses = atoms.get_masses()
    displaced = []
    for mode in modes:
        if np.iscomplexobj(mode.vector) and np.abs(np.imag(mode.vector)).max() > 1e-10:
            raise ValueError("Un fonón congelado es un desplazamiento real: usa real_modes().")
        u = np.real(mode.vector) / np.sqrt(masses)[:, None]                 # Å per (amu^½·Å) of Q
        scale = step / np.abs(u).max()
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions += scale * u
        minus.positions -= scale * u
        zp = float(zero_point(mode.frequency))                     # one quantum, Å·√amu
        displaced.append((System.build(plus, model), System.build(minus, model),
                          zp / (2 * scale)))
    base = [_hk(system, k, shift) for k in kpts]
    orbital_atom = np.concatenate([[a] * (sl.stop - sl.start) for a, sl in
                                   enumerate(system.basis.of_atom(i) for i in range(len(atoms)))])
    orbital_pos = atoms.positions[orbital_atom]
    states = [eigh(h, s) if s is not None else eigh(h) for h, s, _, _ in base]
    electrons = round(system.electrons)
    if electrons % 2:
        raise ValueError(f"{electrons} electrones por celda: una banda semillena (un metal); "
                         "duplica la celda o usa un número par de electrones.")
    n_occ = electrons // 2
    if fermi is None:
        fermi = 0.5 * (max(e[n_occ - 1] for e, _ in states) + min(e[n_occ] for e, _ in states))
    out = []
    pols = np.asarray(polarizations, dtype=float)
    for k, w, (h, s, dh, ds), (e, c) in zip(kpts, weights, base, states, strict=True):
        index = np.arange(len(e))           # occupied by band index (T = 0): a Dirac
        v_idx = np.flatnonzero((index < n_occ) & (e > fermi - window_ev))   # point at E_F
        c_idx = np.flatnonzero((index >= n_occ) & (e < fermi + window_ev))  # stays split
        cv, cc = c[:, v_idx], c[:, c_idx]
        velocity = []
        for p in pols:
            d = np.tensordot(p, dh, axes=1)
            if ds is not None:
                dsp = np.tensordot(p, ds, axes=1)
                velocity.append(cc.conj().T @ d @ cv
                                - (cc.conj().T @ dsp @ cv) * e[v_idx][None, :])
            else:
                velocity.append(cc.conj().T @ d @ cv)
        g_cc, g_vv = [], []
        for sp, sm, factor in displaced:
            phase = np.exp(1j * (orbital_pos @ system.kpoint_cartesian(k)))
            hp, spp = _hk_fixed_phase(sp, k, shift, phase)
            hm, smm = _hk_fixed_phase(sm, k, shift, phase)
            dhq = (hp - hm) * factor
            dsq = None if spp is None else (spp - smm) * factor
            g_cc.append(_project(cc, dhq, dsq, e[c_idx]))
            g_vv.append(_project(cv, dhq, dsq, e[v_idx]))
        out.append(KData(float(w), e[v_idx], e[c_idx], np.array(velocity),
                         np.array(g_cc), np.array(g_vv)))
    return out


def _project(c, dh, ds, energies):
    g = c.conj().T @ dh @ c
    if ds is not None:
        g = g - 0.5 * (energies[:, None] + energies[None, :]) * (c.conj().T @ ds @ c)
    return g


def _pairs(kd: KData, laser: float, omega: float, gamma: float) -> np.ndarray:
    return laser - omega - (kd.ec[:, None] - kd.ev[None, :]) + 1j * gamma


def _vertex(kd: KData, mode: int, x: np.ndarray) -> np.ndarray:
    return kd.g_cc[mode] @ x - x @ kd.g_vv[mode]


def first_order(data: list[KData], modes: list[Mode], laser_ev: float,
                gamma: float = 0.1) -> np.ndarray:
    """Raman tensors A[μ, s, i] (scattered, incident polarisation) of one-phonon scattering."""
    n = len(modes)
    npol = data[0].velocity.shape[0]
    out = np.zeros((n, npol, npol), dtype=complex)
    for kd in data:
        d0 = _pairs(kd, laser_ev, 0.0, gamma)
        for mu, mode in enumerate(modes):
            d1 = _pairs(kd, laser_ev, mode.frequency * CM1_TO_EV, gamma)
            for i in range(npol):
                y = _vertex(kd, mu, kd.velocity[i] / d0) / d1
                for s in range(npol):
                    out[mu, s, i] += kd.weight * np.sum(np.conj(kd.velocity[s]) * y)
    return out


def second_order(data: list[KData], modes: list[Mode], pairs: list[tuple[int, int]],
                 laser_ev: float, gamma: float = 0.1) -> np.ndarray:
    """Tensors A[p, s, i] of two-phonon scattering for each pair (μ, μ') in ``pairs``,
    both emission orders summed."""
    npol = data[0].velocity.shape[0]
    out = np.zeros((len(pairs), npol, npol), dtype=complex)
    for kd in data:
        d0 = _pairs(kd, laser_ev, 0.0, gamma)
        cache = {}
        for p, (a, b) in enumerate(pairs):
            wa, wb = modes[a].frequency * CM1_TO_EV, modes[b].frequency * CM1_TO_EV
            d2 = _pairs(kd, laser_ev, wa + wb, gamma)
            for i in range(npol):
                x = kd.velocity[i] / d0
                total = 0.0
                for first, second, w1 in ((a, b, wa), (b, a, wb)):
                    key = (i, first)
                    if key not in cache:
                        cache[key] = _vertex(kd, first, x) / _pairs(kd, laser_ev, w1, gamma)
                    total = total + _vertex(kd, second, cache[key]) / d2
                for s in range(npol):
                    out[p, s, i] += kd.weight * np.sum(np.conj(kd.velocity[s]) * total)
    return out


def two_phonon_intensity(amplitudes: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Σ_si |A|² per pair, to normalised two-phonon states: an overtone (μ, μ) reaches
    |2_μ⟩, whose matrix element is A/√2 (both orders are the same path counted twice)."""
    power = np.sum(np.abs(amplitudes) ** 2, axis=(1, 2))
    return np.array([power[p] / 2 if a == b else power[p] for p, (a, b) in enumerate(pairs)])
