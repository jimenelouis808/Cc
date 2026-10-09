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


# --------------------------------------------------------------------------
# Phonons of any momentum q (not only Γ of a supercell)
# --------------------------------------------------------------------------

@dataclass
class Bands:
    """States of one k point of a regular mesh, with the optical elements."""

    k: np.ndarray                      # fractional
    ev: np.ndarray
    ec: np.ndarray
    cv: np.ndarray                     # (n_orb, n_v) coefficients, positions gauge
    cc: np.ndarray                     # (n_orb, n_c)
    velocity: np.ndarray               # (n_pol, n_c, n_v)


def _orbital_positions(system: System) -> np.ndarray:
    atom = np.concatenate([[a] * len(system.basis.of_atom(a)) for a in range(len(system.atoms))])
    return system.atoms.positions[atom]


def mesh_bands(system: System, mesh: tuple[int, int, int], window_ev: float,
               polarizations, shift: np.ndarray | None = None,
               fermi: float | None = None) -> tuple[list[Bands], float]:
    """States inside ``window_ev`` of E_F on a Γ-centred ``mesh`` (index order i, j, l)."""
    n1, n2, n3 = mesh
    kpts = np.array([[i / n1, j / n2, l / n3] for i in range(n1) for j in range(n2)
                     for l in range(n3)])
    base = [_hk(system, k, shift) for k in kpts]
    states = [eigh(h, s) if s is not None else eigh(h) for h, s, _, _ in base]
    electrons = round(system.electrons)
    if electrons % 2:
        raise ValueError(f"{electrons} electrones por celda: una banda semillena.")
    n_occ = electrons // 2
    if fermi is None:
        fermi = 0.5 * (max(e[n_occ - 1] for e, _ in states) + min(e[n_occ] for e, _ in states))
    pols = np.asarray(polarizations, dtype=float)
    out = []
    for k, (h, s, dh, ds), (e, c) in zip(kpts, base, states, strict=True):
        index = np.arange(len(e))
        v_idx = np.flatnonzero((index < n_occ) & (e > fermi - window_ev))
        c_idx = np.flatnonzero((index >= n_occ) & (e < fermi + window_ev))
        cv, cc = c[:, v_idx], c[:, c_idx]
        vel = []
        for p in pols:
            d = np.tensordot(p, dh, axes=1)
            m = cc.conj().T @ d @ cv
            if ds is not None:
                m = m - (cc.conj().T @ np.tensordot(p, ds, axes=1) @ cv) * e[v_idx][None, :]
            vel.append(m)
        out.append(Bands(k, e[v_idx], e[c_idx], cv, cc, np.array(vel)))
    return out, float(fermi)


class Coupling:
    """⟨k'| ∂H |k⟩ for a displacement field ``u_a e^{2πi q·R}`` (lattice convention: atom a
    of cell R moves by u_a times the phase), in the positions gauge of the Bloch states:

        ΔH_ij(k', k) = Σ_bonds e^{-ik'·r_i} [∂h/∂d · (u_j e^{2πi q·R} − u_i)] e^{ik·(r_j+R)}

    with k' = k + q (any representative). The bond derivatives are the ones the forces
    use (:func:`tbkit.forces._block_derivatives`); the overlap changes the same way, and
    the fixed SCC shifts enter through ½ΔS(V_i + V_j)."""

    def __init__(self, system: System, shift: np.ndarray | None = None):
        from .forces import _block_derivatives

        self.system = system
        self.shift = shift
        model = system.model
        self.dh = [_block_derivatives(system, b, model.hopping) for b in system.bonds]
        self.ds = (None if model.orthogonal else
                   [_block_derivatives(system, b, model.overlap) for b in system.bonds])
        self.slices = [system.basis.of_atom(a) for a in range(len(system.atoms))]

    def matrices(self, k_to, k_from, q, u: np.ndarray):
        system = self.system
        n = system.basis.size
        kt = system.kpoint_cartesian(k_to)
        kf = system.kpoint_cartesian(k_from)
        pos = system.atoms.positions
        dh = np.zeros((n, n), dtype=complex)
        ds = None if self.ds is None else np.zeros((n, n), dtype=complex)
        q = np.asarray(q, float)
        for p, bond in enumerate(system.bonds):
            i, j = bond.i, bond.j
            change = u[j] * np.exp(2j * np.pi * float(q @ bond.shift)) - u[i]       # (3,)
            phase = np.exp(-1j * float(kt @ pos[i])) * np.exp(1j * float(kf @ (pos[i] + bond.vector)))
            ri, rj = self.slices[i], self.slices[j]
            dh[ri.start:ri.stop, rj.start:rj.stop] += np.tensordot(change, self.dh[p], axes=1) * phase
            if ds is not None:
                ds[ri.start:ri.stop, rj.start:rj.stop] += np.tensordot(change, self.ds[p], axes=1) * phase
        if ds is not None and self.shift is not None:
            dh = dh + ds * 0.5 * (self.shift[:, None] + self.shift[None, :])
        return dh, ds

    def element(self, to: Bands, frm: Bands, q, u, block: str) -> np.ndarray:
        """g between the conduction ('cc') or valence ('vv') states of two k points."""
        dh, ds = self.matrices(to.k, frm.k, q, u)
        a, ea = (to.cc, to.ec) if block == "cc" else (to.cv, to.ev)
        b, eb = (frm.cc, frm.ec) if block == "cc" else (frm.cv, frm.ev)
        g = a.conj().T @ dh @ b
        if ds is not None:
            g = g - 0.5 * (ea[:, None] + eb[None, :]) * (a.conj().T @ ds @ b)
        return g


def overtone_q(bands: list[Bands], mesh: tuple[int, int, int], coupling: Coupling,
               q_index: tuple[int, int, int], frequency: float, e_q: np.ndarray,
               masses: np.ndarray, laser_ev: float, gamma: float = 0.1) -> np.ndarray:
    """Amplitude tensor A[s, i] of emitting the phonons (q, ν) and (−q, ν), both orders,
    summed over the k mesh (mean): graphene's ``_four_processes`` for any band structure.

    ``e_q`` is the lattice-convention eigenvector of D(q) (n_atoms, 3); the mode −q is
    its complex conjugate. ``q_index`` is q in mesh steps."""
    n1, n2, n3 = mesh
    q = np.array(q_index, float) / np.array(mesh, float)
    w = frequency * CM1_TO_EV
    u = e_q / np.sqrt(masses)[:, None] * float(zero_point(frequency))

    def index(i, j, l):
        return ((i % n1) * n2 + (j % n2)) * n3 + (l % n3)

    npol = bands[0].velocity.shape[0]
    total = np.zeros((npol, npol), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            for l in range(n3):
                total += _overtone_at(bands, index, (i, j, l), q_index, q, u, w, coupling,
                                      laser_ev, gamma)
    return total / len(bands)


def _overtone_at(bands, index, ijl, qi, q, u, w, coupling, laser, gamma):
    i, j, l = ijl
    k = bands[index(i, j, l)]
    kp = bands[index(i + qi[0], j + qi[1], l + qi[2])]
    km = bands[index(i - qi[0], j - qi[1], l - qi[2])]

    def pairs(b, omega):
        return laser - omega - (b.ec[:, None] - b.ev[None, :]) + 1j * gamma

    def pairs2(bc, bv, omega):
        return laser - omega - (bc.ec[:, None] - bv.ev[None, :]) + 1j * gamma

    npol = k.velocity.shape[0]
    out = np.zeros((npol, npol), dtype=complex)
    uc = np.conj(u)
    for first, second in ((u, uc, ), (uc, u)):
        sign = 1.0 if first is u else -1.0
        qq = sign * q
        a, b = (kp, km) if sign > 0 else (km, kp)        # k + qq, k − qq
        # first vertex: transfer +qq; second: −qq
        g1_e = coupling.element(a, k, qq, first, "cc")          # c: k -> k+qq
        g1_h = coupling.element(k, b, qq, first, "vv")          # v: k-qq -> k
        g2_e_back = coupling.element(k, a, -qq, second, "cc")   # c: k+qq -> k
        g2_h_fwd = coupling.element(k, a, -qq, second, "vv")    # v: k+qq -> k
        g2_e_fwd = coupling.element(b, k, -qq, second, "cc")    # c: k -> k-qq
        g2_h_back = coupling.element(b, k, -qq, second, "vv")   # v: k -> k-qq
        for ii in range(npol):
            x = k.velocity[ii] / pairs(k, 0.0)                        # (c_k, v_k)
            xe = (g1_e @ x) / pairs2(a, k, w)                         # (c_{k+q}, v_k)
            xh = -(x @ g1_h) / pairs2(k, b, w)                        # (c_k, v_{k-q})
            same = (g2_e_back @ xe - xh @ g2_h_back) / pairs(k, 2 * w)        # ee + hh
            eh = -(xe @ g2_h_fwd) / pairs(a, 2 * w)                    # (c_{k+q}, v_{k+q})
            he = (g2_e_fwd @ xh) / pairs(b, 2 * w)                     # (c_{k-q}, v_{k-q})
            for s in range(npol):
                out[s, ii] += (np.sum(np.conj(k.velocity[s]) * same)
                               + np.sum(np.conj(a.velocity[s]) * eh)
                               + np.sum(np.conj(b.velocity[s]) * he))
    return out
