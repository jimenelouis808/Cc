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
    the fixed SCC shifts enter through ½ΔS(V_i + V_j). Bonds are grouped by block shape
    and summed with array operations (a coil has thousands)."""

    def __init__(self, system: System, shift: np.ndarray | None = None):
        from .forces import _block_derivatives

        self.system = system
        self.shift = shift
        model = system.model
        slices = [system.basis.of_atom(a) for a in range(len(system.atoms))]
        groups: dict = {}
        for b in system.bonds:
            ri, rj = slices[b.i], slices[b.j]
            groups.setdefault((len(ri), len(rj)), []).append(b)
        self.groups = []
        for (ni, nj), bonds in groups.items():
            dh = np.array([_block_derivatives(system, b, model.hopping) for b in bonds])
            ds = (None if model.orthogonal else
                  np.array([_block_derivatives(system, b, model.overlap) for b in bonds]))
            i = np.array([b.i for b in bonds])
            j = np.array([b.j for b in bonds])
            rows = np.array([slices[a].start for a in i])[:, None, None] + np.arange(ni)[None, :, None]
            cols = np.array([slices[a].start for a in j])[:, None, None] + np.arange(nj)[None, None, :]
            self.groups.append({"i": i, "j": j, "dh": dh, "ds": ds,
                                "shift": np.array([b.shift for b in bonds], float),
                                "vector": np.array([b.vector for b in bonds]),
                                "rows": np.broadcast_to(rows, (len(bonds), ni, nj)),
                                "cols": np.broadcast_to(cols, (len(bonds), ni, nj))})

    def matrices(self, k_to, k_from, q, u: np.ndarray):
        system = self.system
        n = system.basis.size
        kt = system.kpoint_cartesian(k_to)
        kf = system.kpoint_cartesian(k_from)
        pos = system.atoms.positions
        q = np.asarray(q, float)
        dh = np.zeros((n, n), dtype=complex)
        ds = None
        for g in self.groups:
            change = u[g["j"]] * np.exp(2j * np.pi * (g["shift"] @ q))[:, None] - u[g["i"]]
            phase = np.exp(-1j * (pos[g["i"]] @ kt) + 1j * ((pos[g["i"]] + g["vector"]) @ kf))
            blocks = np.einsum("pa,paxy->pxy", change, g["dh"]) * phase[:, None, None]
            np.add.at(dh, (g["rows"], g["cols"]), blocks)
            if g["ds"] is not None:
                ds = np.zeros((n, n), dtype=complex) if ds is None else ds
                blocks = np.einsum("pa,paxy->pxy", change, g["ds"]) * phase[:, None, None]
                np.add.at(ds, (g["rows"], g["cols"]), blocks)
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


class PhononVertex:
    """One quantum of the mode with eigenvector ``e_q`` (lattice convention) at +q."""

    def __init__(self, coupling: Coupling, e_q: np.ndarray, masses: np.ndarray,
                 frequency: float, conjugate: bool = False):
        u = e_q / np.sqrt(masses)[:, None] * float(zero_point(frequency))
        self.u = np.conj(u) if conjugate else u
        self.coupling = coupling
        self.energy = frequency * CM1_TO_EV

    def __call__(self, to: Bands, frm: Bands, q, block: str) -> np.ndarray:
        return self.coupling.element(to, frm, q, self.u, block)


class DefectVertex:
    """Elastic scattering by one defect in an otherwise periodic crystal: ΔH (and ΔS)
    between the cell with the defect and the pristine one, as real-space blocks
    ``{(i, j, shift): (dH, dS)}`` on the pristine atoms. Its matrix element between
    Bloch states is ``Σ e^{-ik'·r_i} ΔH e^{ik·(r_j+R)}`` (per defect; the intensity of a
    defect-activated band scales with the defect density)."""

    energy = 0.0

    def __init__(self, system: System, blocks: dict):
        self.system = system
        self.blocks = blocks
        self.slices = [system.basis.of_atom(a) for a in range(len(system.atoms))]

    def __call__(self, to: Bands, frm: Bands, q, block: str) -> np.ndarray:
        system = self.system
        n = system.basis.size
        kt, kf = system.kpoint_cartesian(to.k), system.kpoint_cartesian(frm.k)
        pos, cell = system.atoms.positions, system.atoms.cell.array
        dh = np.zeros((n, n), dtype=complex)
        ds = None
        for (i, j, shift), (bh, bs) in self.blocks.items():
            rj = pos[j] + np.asarray(shift, float) @ cell
            phase = np.exp(-1j * float(kt @ pos[i]) + 1j * float(kf @ rj))
            si, sj = self.slices[i], self.slices[j]
            dh[si.start:si.stop, sj.start:sj.stop] += bh * phase
            if bs is not None:
                ds = np.zeros((n, n), dtype=complex) if ds is None else ds
                ds[si.start:si.stop, sj.start:sj.stop] += bs * phase
        a, ea = (to.cc, to.ec) if block == "cc" else (to.cv, to.ev)
        b, eb = (frm.cc, frm.ec) if block == "cc" else (frm.cv, frm.ev)
        g = a.conj().T @ dh @ b
        if ds is not None:
            g = g - 0.5 * (ea[:, None] + eb[None, :]) * (a.conj().T @ ds @ b)
        return g


def two_vertices_q(bands: list[Bands], mesh: tuple[int, int, int], q_index, first, second,
                   laser_ev: float, gamma: float = 0.1) -> np.ndarray:
    """Amplitude A[s, i] of two scatterings, ``first`` transferring +q and ``second`` −q,
    in both time orders, summed over the k mesh (mean). Each vertex is called as
    ``vertex(to, from, q, block)`` and has an ``energy`` (eV) it leaves behind (a phonon's,
    or 0 for a defect). Two phonons (q, ν), (−q, ν) give the 2D-type overtone; a phonon and
    a defect give the defect-activated D-type band."""
    n1, n2, n3 = mesh
    q = np.array(q_index, float) / np.array(mesh, float)

    def index(i, j, l):
        return ((i % n1) * n2 + (j % n2)) * n3 + (l % n3)

    npol = bands[0].velocity.shape[0]
    total = np.zeros((npol, npol), dtype=complex)
    for i in range(n1):
        for j in range(n2):
            for l in range(n3):
                k = bands[index(i, j, l)]
                kp = bands[index(i + q_index[0], j + q_index[1], l + q_index[2])]
                km = bands[index(i - q_index[0], j - q_index[1], l - q_index[2])]
                total += _ordered(k, kp, km, q, first, second, laser_ev, gamma)
                total += _ordered(k, km, kp, -q, second, first, laser_ev, gamma)
    return total / len(bands)


def _ordered(k, a, b, q, v1, v2, laser, gamma):
    """One time order: v1 transfers +q (k -> a = k+q for electrons), then v2 transfers −q."""
    def pairs(bc, bv, omega):
        return laser - omega - (bc.ec[:, None] - bv.ev[None, :]) + 1j * gamma

    w1, w12 = v1.energy, v1.energy + v2.energy
    g1_e = v1(a, k, q, "cc")              # c: k -> k+q
    g1_h = v1(k, b, q, "vv")              # v: k-q -> k
    g2_e_back = v2(k, a, -q, "cc")        # c: k+q -> k
    g2_h_fwd = v2(k, a, -q, "vv")         # v: k+q -> k
    g2_e_fwd = v2(b, k, -q, "cc")         # c: k -> k-q
    g2_h_back = v2(b, k, -q, "vv")        # v: k -> k-q
    npol = k.velocity.shape[0]
    out = np.zeros((npol, npol), dtype=complex)
    for ii in range(npol):
        x = k.velocity[ii] / pairs(k, k, 0.0)
        xe = (g1_e @ x) / pairs(a, k, w1)                     # (c_{k+q}, v_k)
        xh = -(x @ g1_h) / pairs(k, b, w1)                    # (c_k, v_{k-q})
        same = (g2_e_back @ xe - xh @ g2_h_back) / pairs(k, k, w12)       # ee + hh
        eh = -(xe @ g2_h_fwd) / pairs(a, a, w12)              # (c_{k+q}, v_{k+q})
        he = (g2_e_fwd @ xh) / pairs(b, b, w12)               # (c_{k-q}, v_{k-q})
        for s in range(npol):
            out[s, ii] += (np.sum(np.conj(k.velocity[s]) * same)
                           + np.sum(np.conj(a.velocity[s]) * eh)
                           + np.sum(np.conj(b.velocity[s]) * he))
    return out


def overtone_q(bands: list[Bands], mesh: tuple[int, int, int], coupling: Coupling,
               q_index: tuple[int, int, int], frequency: float, e_q: np.ndarray,
               masses: np.ndarray, laser_ev: float, gamma: float = 0.1) -> np.ndarray:
    """Amplitude tensor A[s, i] of emitting the phonons (q, ν) and (−q, ν), both orders,
    summed over the k mesh (mean): graphene's ``_four_processes`` for any band structure.
    ``e_q`` is the lattice-convention eigenvector of D(q) (n_atoms, 3); the mode −q is its
    complex conjugate. ``q_index`` is q in mesh steps."""
    first = PhononVertex(coupling, e_q, masses, frequency)
    second = PhononVertex(coupling, e_q, masses, frequency, conjugate=True)
    return two_vertices_q(bands, mesh, q_index, first, second, laser_ev, gamma)


def one_vertex(bands: list[Bands], vertex, laser_ev: float, gamma: float = 0.1) -> np.ndarray:
    """First-order amplitude A[s, i] of a q = 0 scattering on the mesh ``bands`` (mean):
    the Γ phonon's Raman tensor of :func:`first_order`, from the vertex objects."""
    npol = bands[0].velocity.shape[0]
    out = np.zeros((npol, npol), dtype=complex)
    zero = np.zeros(3)
    for k in bands:
        d0 = laser_ev - (k.ec[:, None] - k.ev[None, :]) + 1j * gamma
        d1 = d0 - vertex.energy
        g_cc, g_vv = vertex(k, k, zero, "cc"), vertex(k, k, zero, "vv")
        for i in range(npol):
            x = k.velocity[i] / d0
            y = (g_cc @ x - x @ g_vv) / d1
            for s in range(npol):
                out[s, i] += np.sum(np.conj(k.velocity[s]) * y)
    return out / len(bands)


def phonon_pairs_q(bands: list[Bands], mesh: tuple[int, int, int], coupling: Coupling,
                   q_index, frequencies: np.ndarray, vectors: np.ndarray, masses: np.ndarray,
                   laser_ev: float, gamma: float = 0.1) -> np.ndarray:
    """Σ_si |A|² for every pair of phonons (q, ν), (−q, ν') of the given branches.

    In a crystal with many atoms per cell one band of graphene (its TO at K, the 2D
    mode) spreads over many branches, so the two phonons of a second-order process
    need not belong to the same branch: this returns the matrix W[ν, ν'] of all pairs,
    each amplitude with both time orders (the first at +q, the second at −q, then the
    reverse), so W[ν, ν] is :func:`overtone_q`. The first-vertex half of the chain is
    computed once per branch and contracted with every second branch at once."""
    n1, n2, n3 = mesh
    q = np.array(q_index, float) / np.array(mesh, float)
    vertices = [PhononVertex(coupling, e, masses, f) for f, e in zip(frequencies, vectors, strict=True)]
    conj = [PhononVertex(coupling, e, masses, f, conjugate=True)
            for f, e in zip(frequencies, vectors, strict=True)]
    w = np.asarray(frequencies, float) * CM1_TO_EV

    def index(i, j, l):
        return ((i % n1) * n2 + (j % n2)) * n3 + (l % n3)

    total = None
    for i in range(n1):
        for j in range(n2):
            for l in range(n3):
                k = bands[index(i, j, l)]
                kp = bands[index(i + q_index[0], j + q_index[1], l + q_index[2])]
                km = bands[index(i - q_index[0], j - q_index[1], l - q_index[2])]
                amp = _pairs_ordered(k, kp, km, q, vertices, conj, w, laser_ev, gamma)
                amp += np.transpose(_pairs_ordered(k, km, kp, -q, conj, vertices, w, laser_ev,
                                                   gamma), (1, 0, 2, 3))
                total = amp if total is None else total + amp
    return np.sum(np.abs(total / len(bands)) ** 2, axis=(2, 3))


def _pairs_ordered(k, a, b, q, first, second, w, laser, gamma):
    """A[ν, ν', s, i] of one time order: ``first[ν]`` transfers +q, then ``second[ν']`` −q."""
    def pairs(bc, bv, omega):
        return laser - np.asarray(omega)[..., None, None] - (bc.ec[:, None] - bv.ev[None, :]) \
            + 1j * gamma

    g1_e = np.array([v(a, k, q, "cc") for v in first])
    g1_h = np.array([v(k, b, q, "vv") for v in first])
    g2_e_back = np.array([v(k, a, -q, "cc") for v in second])
    g2_h_fwd = np.array([v(k, a, -q, "vv") for v in second])
    g2_e_fwd = np.array([v(b, k, -q, "cc") for v in second])
    g2_h_back = np.array([v(b, k, -q, "vv") for v in second])
    total_w = w[:, None] + w[None, :]
    d_same, d_eh, d_he = pairs(k, k, total_w), pairs(a, a, total_w), pairs(b, b, total_w)
    npol = k.velocity.shape[0]
    n = len(w)
    out = np.zeros((n, n, npol, npol), dtype=complex)
    for ii in range(npol):
        x = k.velocity[ii] / (laser - (k.ec[:, None] - k.ev[None, :]) + 1j * gamma)
        xe = np.einsum("ncd,dv->ncv", g1_e, x) / pairs(a, k, w)            # (ν, c_{k+q}, v_k)
        xh = -np.einsum("cv,nvu->ncu", x, g1_h) / pairs(k, b, w)           # (ν, c_k, v_{k-q})
        same = (np.einsum("mcd,ndv->nmcv", g2_e_back, xe)
                - np.einsum("ncv,mvu->nmcu", xh, g2_h_back)) / d_same
        eh = -np.einsum("ncv,mvu->nmcu", xe, g2_h_fwd) / d_eh
        he = np.einsum("mcd,ndv->nmcv", g2_e_fwd, xh) / d_he
        for s in range(npol):
            out[:, :, s, ii] = (np.einsum("cv,nmcv->nm", np.conj(k.velocity[s]), same)
                                + np.einsum("cv,nmcv->nm", np.conj(a.velocity[s]), eh)
                                + np.einsum("cv,nmcv->nm", np.conj(b.velocity[s]), he))
    return out


def _pairs_ordered_fast(k, a, b, q, first, second, w, laser, gamma, order: int = 7):
    """:func:`_pairs_ordered` with the last denominator expanded about the mean pair
    energy: 1/(D₀ − δ) = Σ_p δ^p / D₀^{p+1}, δ = (ω_ν − ω̄) + (ω_ν' − ω̄). Each power
    factorises into a branch-by-branch product, so the cost is linear in the number of
    pairs instead of quadratic in the chain; with |δ| ≤ 0.03 eV and |D₀| ≥ γ = 0.1 eV,
    seven terms leave ~1e-4 relative error (tested against the exact form)."""
    from math import comb

    def pairs(bc, bv, omega):
        return laser - np.asarray(omega)[..., None, None] - (bc.ec[:, None] - bv.ev[None, :]) \
            + 1j * gamma

    g1_e = np.array([v(a, k, q, "cc") for v in first])
    g1_h = np.array([v(k, b, q, "vv") for v in first])
    g2_e_back = np.array([v(k, a, -q, "cc") for v in second])
    g2_h_fwd = np.array([v(k, a, -q, "vv") for v in second])
    g2_e_fwd = np.array([v(b, k, -q, "cc") for v in second])
    g2_h_back = np.array([v(b, k, -q, "vv") for v in second])
    mean = float(np.mean(w))
    dw = w - mean
    d0 = {key: pairs(x, x, 2 * mean) for key, x in (("k", k), ("a", a), ("b", b))}
    npol = k.velocity.shape[0]
    n = len(w)
    out = np.zeros((n, n, npol, npol), dtype=complex)
    powers = np.array([dw ** p for p in range(order)])                 # (p, n)
    for ii in range(npol):
        x = k.velocity[ii] / (laser - (k.ec[:, None] - k.ev[None, :]) + 1j * gamma)
        xe = np.einsum("ncd,dv->ncv", g1_e, x) / pairs(a, k, w)            # (n, c_{k+q}, v_k)
        xh = -np.einsum("cv,nvu->ncu", x, g1_h) / pairs(k, b, w)           # (n, c_k, v_{k-q})
        for s in range(npol):
            terms = np.zeros((order, order, n, n), dtype=complex)        # [a, b]: δ_n^a δ_m^b
            for p in range(order):
                yk = np.conj(k.velocity[s]) / d0["k"] ** (p + 1)
                ya = np.conj(a.velocity[s]) / d0["a"] ** (p + 1)
                yb = np.conj(b.velocity[s]) / d0["b"] ** (p + 1)
                # same: Σ yk ∘ (G2eb[m] xe[n] − xh[n] G2hb[m])
                t = np.einsum("ndv,mdv->nm", xe, np.einsum("cv,mcd->mdv", yk, g2_e_back))
                t -= np.einsum("ncu,mcu->nm", xh, np.einsum("cv,muv->mcu", yk, g2_h_back))
                # eh: −Σ ya ∘ (xe[n] G2hf[m]);  he: Σ yb ∘ (G2ef[m] xh[n])
                t -= np.einsum("ncv,mcv->nm", xe, np.einsum("cu,mvu->mcv", ya, g2_h_fwd))
                t += np.einsum("ndv,mdv->nm", xh, np.einsum("cv,mcd->mdv", yb, g2_e_fwd))
                for a_pow in range(p + 1):
                    terms[a_pow, p - a_pow] += comb(p, a_pow) * t
            out[:, :, s, ii] = np.einsum("an,bm,abnm->nm", powers, powers, terms)
    return out


def phonon_pairs_q_fast(bands: list[Bands], mesh: tuple[int, int, int], coupling: Coupling,
                        q_index, frequencies: np.ndarray, vectors: np.ndarray,
                        masses: np.ndarray, laser_ev: float, gamma: float = 0.1,
                        order: int = 7) -> np.ndarray:
    """:func:`phonon_pairs_q` with the factorised last denominator (for many branches)."""
    n1, n2, n3 = mesh
    q = np.array(q_index, float) / np.array(mesh, float)
    vertices = [PhononVertex(coupling, e, masses, f) for f, e in zip(frequencies, vectors, strict=True)]
    conj = [PhononVertex(coupling, e, masses, f, conjugate=True)
            for f, e in zip(frequencies, vectors, strict=True)]
    w = np.asarray(frequencies, float) * CM1_TO_EV

    def index(i, j, l):
        return ((i % n1) * n2 + (j % n2)) * n3 + (l % n3)

    total = None
    for i in range(n1):
        for j in range(n2):
            for l in range(n3):
                k = bands[index(i, j, l)]
                kp = bands[index(i + q_index[0], j + q_index[1], l + q_index[2])]
                km = bands[index(i - q_index[0], j - q_index[1], l - q_index[2])]
                amp = _pairs_ordered_fast(k, kp, km, q, vertices, conj, w, laser_ev, gamma, order)
                amp += np.transpose(_pairs_ordered_fast(k, km, kp, -q, conj, vertices, w,
                                                        laser_ev, gamma, order), (1, 0, 2, 3))
                total = amp if total is None else total + amp
    return np.sum(np.abs(total / len(bands)) ** 2, axis=(2, 3))
