"""Graphene Raman by perturbation theory: the G band and the double-resonant 2D band.

Graphene's Raman is resonant at any laser energy, and its second-order bands
come from the double-resonance process (Thomsen and Reich, Phys. Rev. Lett.
85, 5214 (2000)): a photon creates an electron-hole pair near the Dirac point
K, a phonon of momentum q ≈ K scatters the electron (or the hole) to the
other valley, a second phonon of momentum -q brings it back, and the pair
recombines. Following Venezuela, Lazzeri and Mauri (Phys. Rev. B 84, 035433
(2011)), the amplitude sums over electron momenta k all the time orderings
in which one carrier emits both phonons (ee, hh) or each carrier emits one
(eh, he), with lifetime-broadened energy denominators.

Ingredients, each from a tbkit model with its stated source:

* **electrons**: the π model (``parameters/pi_huckel.json``): nearest-neighbour
  hopping t(d) = t exp(-β(d/a - 1)); bands ±|f(k)|;
* **light**: in-plane dipole matrix elements ``⟨c|e·∂_k H|v⟩`` (velocity gauge);
* **electron-phonon coupling**: the change of t(d) with the bond length,
  ``t'(a) = -β t/a``, projected on each phonon (analytic, below);
* **phonons**: force constants of Xu's sp³ carbon model in a supercell,
  Fourier-interpolated to any q (:class:`GraphenePhonons`), or any other
  force constants in the same form.

Conventions. Bloch sums include the atomic positions (``e^{ik·(R+τ)}``); a
phonon (q, ν) displaces atom s of cell R by ``e_s e^{iq·(R+τ_s)}/√M_s`` per
unit normal coordinate, with ``D(q) e = ω² e`` and
``D_st(q) = Σ_R Φ_st(R) e^{iq·(R+τ_t-τ_s)}/√(M_s M_t)``. With these, the
matrix element between a state at k and one at k+q is

    g_AB(k+q, k) = Σ_δ t'(a) e^{ik·δ} δ̂·(u_B e^{iq·δ} - u_A)
    g_BA(k+q, k) = Σ_δ t'(a) e^{-i(k+q)·δ} δ̂·(u_B e^{iq·δ} - u_A)

(δ the three A→B bond vectors, u_s = e_s/√M_s), which the tests check against
the change of H under a frozen phonon in a commensurate supercell.

What it is not: excitonic effects, electron-electron renormalisation of the
Fermi velocity and of the Kohn anomalies (which set the 2D dispersion in real
graphene; Xu's force constants have their own, weaker ones), and defects
(no D band here).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np
from ase import Atoms

from .params import read_parameter_file

#: ħ c in eV·cm (for converting phonon energies): 1 cm⁻¹ = 1.239842e-4 eV.
CM1_TO_EV = 1.239841984e-4


def graphene_cell(a_cc: float = 1.42, vacuum: float = 15.0) -> Atoms:
    """Primitive graphene cell: A at (0, 0, c/2), B at (a_cc, 0, c/2).

    The sheet sits in the middle of the vacuum (grid codes need it away from
    the non-periodic boundary); bond vectors, not positions, enter the models.
    """
    a = a_cc * np.sqrt(3)
    cell = [[1.5 * a_cc, a / 2, 0.0], [1.5 * a_cc, -a / 2, 0.0], [0.0, 0.0, vacuum]]
    z = vacuum / 2
    return Atoms("C2", positions=[[0.0, 0.0, z], [a_cc, 0.0, z]], cell=cell,
                 pbc=[True, True, False])


# --------------------------------------------------------------------------
# Phonons in the whole Brillouin zone
# --------------------------------------------------------------------------

@dataclass
class GraphenePhonons:
    """Force constants Φ_st(R) and the dynamical matrix at any q.

    Built from finite displacements of the two atoms of one cell inside an
    ``n x n`` supercell (:meth:`from_model`). Each pair (s, t, R) is assigned
    to its minimum-image lattice vectors in the supercell, shared equally
    among equidistant images, so that the interpolation is exact at the q
    commensurate with the supercell and smooth in between.
    """

    atoms: Atoms                                  # primitive cell
    vectors: np.ndarray                           # (n_R, 3) lattice vectors R (Å)
    constants: np.ndarray                         # (n_R, 2, 3, 2, 3) Φ_{sα,tβ}(R), eV/Å²
    source: str

    @classmethod
    def from_model(cls, model, n: int = 8, kmesh: int = 3, kT: float = 0.05,
                   delta: float = 0.01, a_cc: Optional[float] = None) -> "GraphenePhonons":
        """Force constants of ``model`` (a tbkit model with repulsion; Xu by default).

        ``a_cc`` defaults to the model's own equilibrium bond (found by
        minimising the energy of the primitive cell).
        """
        from .calculator import TBCalculator

        if a_cc is None:
            a_cc = equilibrium_bond(model, kmesh=kmesh * n, kT=kT)
        primitive = graphene_cell(a_cc)
        supercell = primitive.repeat((n, n, 1))
        base = supercell.get_positions().copy()
        calc = TBCalculator(model, kpts=(kmesh, kmesh, 1), kT=kT)
        forces = np.zeros((2, 3, len(supercell), 3))
        for s in range(2):                     # atoms 0, 1 of the supercell: cell (0, 0)
            for alpha in range(3):
                pair = []
                for step in (delta, -delta):
                    moved = supercell.copy()
                    positions = base.copy()
                    positions[s, alpha] += step
                    moved.set_positions(positions)
                    moved.calc = calc
                    pair.append(moved.get_forces())
                    calc.reset()
                forces[s, alpha] = (pair[0] - pair[1]) / (2 * delta)
        return cls._from_supercell_forces(primitive, n, forces,
                                          f"{model.name}, supercelda {n}x{n}, kT = {kT} eV")

    @classmethod
    def from_calculator(cls, make_calculator, a_cc: float, n: int = 6, delta: float = 0.01,
                        source: str = "", workers: int = 1) -> "GraphenePhonons":
        """Force constants from any ASE calculator (``make_calculator()`` returns a new one).

        The six displaced supercells are independent: ``workers`` runs them
        in parallel processes (the factory must then be picklable).
        """
        primitive = graphene_cell(a_cc)
        supercell = primitive.repeat((n, n, 1))
        jobs = [(supercell, s, alpha, sign * delta, make_calculator)
                for s in range(2) for alpha in range(3) for sign in (1, -1)]
        if workers > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(workers) as pool:
                results = list(pool.map(_displaced_forces, jobs))
        else:
            results = [_displaced_forces(job) for job in jobs]
        forces = np.zeros((2, 3, len(supercell), 3))
        for (_, s, alpha, step, _), f in zip(jobs, results, strict=True):
            forces[s, alpha] += f * np.sign(step) / (2 * delta)
        return cls._from_supercell_forces(primitive, n, forces, source or f"supercelda {n}x{n}")

    def to_dict(self) -> dict:
        """JSON-ready force constants (for storing DFT phonons with their source)."""
        return {"a_cc": float(np.linalg.norm(self.atoms.positions[1] - self.atoms.positions[0])),
                "vacuum": float(self.atoms.cell[2, 2]), "source": self.source,
                "vectors": np.round(self.vectors, 8).tolist(),
                "constants": np.round(self.constants, 8).tolist(), "unit": "eV/Å^2"}

    @classmethod
    def from_dict(cls, data: dict) -> "GraphenePhonons":
        return cls(graphene_cell(data["a_cc"], data.get("vacuum", 15.0)),
                   np.array(data["vectors"]), np.array(data["constants"]), data["source"])

    @classmethod
    def _from_supercell_forces(cls, primitive: Atoms, n: int, dforces: np.ndarray,
                               source: str) -> "GraphenePhonons":
        cell = primitive.cell.array
        tau = primitive.get_positions()
        entries: dict[tuple[int, int], np.ndarray] = {}
        for i1 in range(n):
            for i2 in range(n):
                for t in range(2):
                    index = 2 * (i1 * n + i2) + t        # ASE repeat order
                    phi = -dforces[:, :, index, :]       # Φ_{sα,tβ}(R): (2, 3, 3)
                    # all images R + n·(m1 a1 + m2 a2); keep the shortest ones
                    candidates = []
                    for m1 in (-1, 0, 1):
                        for m2 in (-1, 0, 1):
                            r = (i1 + n * m1) * cell[0] + (i2 + n * m2) * cell[1]
                            for s in range(2):
                                candidates.append((s, (i1 + n * m1, i2 + n * m2), r,
                                                   np.linalg.norm(r + tau[t] - tau[s])))
                    for s in range(2):
                        mine = [c for c in candidates if c[0] == s]
                        shortest = min(c[3] for c in mine)
                        images = [c for c in mine if c[3] < shortest + 1e-6]
                        for _, key, _, _ in images:
                            block = entries.setdefault(key, np.zeros((2, 3, 2, 3)))
                            block[s, :, t, :] += phi[s] / len(images)
        keys = sorted(entries)
        vectors = np.array([k[0] * cell[0] + k[1] * cell[1] for k in keys])
        constants = np.array([entries[k] for k in keys])
        return cls(primitive, vectors, constants, source)

    def with_acoustic_sum_rule(self) -> "GraphenePhonons":
        """Force constants with Σ_{t,R} Φ_{sα,tβ}(R) = 0 imposed (a rigid translation
        costs nothing): the violation, from grid noise in DFT forces, is removed
        from each atom's self term, then Φ is re-symmetrised."""
        constants = self.constants.copy()
        zero = int(np.argmin(np.linalg.norm(self.vectors, axis=1)))
        for s_atom in range(2):
            total = constants[:, s_atom].sum(axis=(0, 2))                 # (3, 3) over t, R
            constants[zero, s_atom, :, s_atom, :] -= total
        block = constants[zero]
        constants[zero] = 0.5 * (block + block.transpose(2, 3, 0, 1))
        return GraphenePhonons(self.atoms, self.vectors, constants,
                               self.source + "; regla de la suma acústica impuesta")

    @cached_property
    def _masses(self) -> np.ndarray:
        return self.atoms.get_masses()

    def dynamical_matrix(self, q: np.ndarray) -> np.ndarray:
        """``D(q)`` (6 x 6), q Cartesian in 1/Å (2π included)."""
        tau = self.atoms.get_positions()
        phase = np.exp(1j * (self.vectors @ q))                           # e^{iq·R}
        d = np.einsum("r,rsatb->satb", phase, self.constants)
        d = d * np.exp(1j * (tau @ q))[None, None, :, None]              # e^{iq·τ_t}
        d = d * np.exp(-1j * (tau @ q))[:, None, None, None]             # e^{-iq·τ_s}
        m = np.sqrt(self._masses)
        d = d / (m[:, None, None, None] * m[None, None, :, None])
        d = d.reshape(6, 6)
        return 0.5 * (d + d.conj().T)

    def modes(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Frequencies (cm⁻¹, ascending; imaginary as negative) and eigenvectors e (6, 2, 3)."""
        from ase.units import _amu, _e, _hbar

        w2, e = np.linalg.eigh(self.dynamical_matrix(np.asarray(q, dtype=float)))
        # eV/Å²/amu -> (rad/s)²; ħω in eV; to cm⁻¹
        omega = np.sqrt(np.abs(w2) * _e / 1e-20 / _amu) * np.sign(w2)
        energy_ev = omega * _hbar / _e
        return energy_ev / CM1_TO_EV, e.T.reshape(6, 2, 3)

    def reciprocal(self) -> np.ndarray:
        """Reciprocal vectors b1, b2 (rows, 1/Å, 2π included)."""
        return 2 * np.pi * self.atoms.cell.reciprocal()[:2]

    def special_points(self) -> dict[str, np.ndarray]:
        b = self.reciprocal()
        return {"Γ": np.zeros(3), "M": 0.5 * b[0], "K": (2 * b[0] + b[1]) / 3}


def _displaced_forces(job):
    supercell, s, alpha, step, make_calculator = job
    moved = supercell.copy()
    positions = moved.get_positions()
    positions[s, alpha] += step
    moved.set_positions(positions)
    moved.calc = make_calculator()
    return np.array(moved.get_forces())


def equilibrium_bond(model, kmesh: int = 24, kT: float = 0.05) -> float:
    """C-C bond of flat graphene that minimises the model's energy (Å)."""
    from scipy.optimize import minimize_scalar

    from .calculator import TBCalculator

    def energy(a):
        cell = graphene_cell(a)
        cell.calc = TBCalculator(model, kpts=(kmesh, kmesh, 1), kT=kT)
        return cell.get_potential_energy()

    return float(minimize_scalar(energy, bounds=(1.36, 1.48), method="bounded",
                                 options={"xatol": 1e-4}).x)


# --------------------------------------------------------------------------
# π electrons, light and electron-phonon coupling
# --------------------------------------------------------------------------

@dataclass
class PiElectrons:
    """Nearest-neighbour π bands of graphene with t(d) = t0 exp(-β(d/a0 - 1)).

    ``a_cc`` is the bond of the structure (the phonons' own equilibrium), at
    which the hopping is ``t = t(a_cc)`` and its slope ``t'(a_cc) = -β t/a0``.
    """

    t0: float
    beta: float
    a0: float
    a_cc: float

    @classmethod
    def from_parameters(cls, a_cc: Optional[float] = None) -> "PiElectrons":
        data = read_parameter_file("pi_huckel")
        a0 = float(data["a_cc"]["value"])
        return cls(float(data["t"]["value"]), float(data["strain_beta"]["value"]), a0,
                   a0 if a_cc is None else float(a_cc))

    @property
    def t(self) -> float:
        return self.t0 * np.exp(-self.beta * (self.a_cc / self.a0 - 1.0))

    @property
    def deltas(self) -> np.ndarray:
        """The three A→B bond vectors (Å) of :func:`graphene_cell`."""
        a = self.a_cc
        return np.array([[a, 0.0, 0.0], [-a / 2, a * np.sqrt(3) / 2, 0.0],
                         [-a / 2, -a * np.sqrt(3) / 2, 0.0]])

    @property
    def dt(self) -> float:
        """t'(a_cc) = -β t / a0 (eV/Å)."""
        return -self.beta * self.t / self.a0

    def f(self, k: np.ndarray) -> np.ndarray:
        """``f(k) = t Σ_δ e^{ik·δ}`` for k of shape (..., 3)."""
        return self.t * np.exp(1j * (k @ self.deltas.T)).sum(axis=-1)

    def states(self, k: np.ndarray):
        """Energies (v, c) and eigenvectors (…, 2 sublattices, 2 bands) at k."""
        f = self.f(k)
        phase = np.where(np.abs(f) > 1e-12, f / np.abs(f), 1.0)
        # H = [[0, f], [f*, 0]]: ε = ∓|f|, ψ_v = (1, -φ*)/√2, ψ_c = (1, φ*)/√2
        vec = np.zeros(k.shape[:-1] + (2, 2), dtype=complex)
        vec[..., 0, 0] = vec[..., 0, 1] = 1 / np.sqrt(2)
        vec[..., 1, 0] = -np.conj(phase) / np.sqrt(2)
        vec[..., 1, 1] = np.conj(phase) / np.sqrt(2)
        energies = np.stack([-np.abs(f), np.abs(f)], axis=-1)
        return energies, vec

    def velocity(self, k: np.ndarray, polarization: np.ndarray) -> np.ndarray:
        """``e·∂_k H`` (…, 2, 2) in the sublattice basis."""
        grad = self.t * (1j * (polarization @ self.deltas.T)) * np.exp(1j * (k @ self.deltas.T))
        off = grad.sum(axis=-1)
        out = np.zeros(k.shape[:-1] + (2, 2), dtype=complex)
        out[..., 0, 1] = off
        out[..., 1, 0] = np.conj(off)
        return out

    def coupling(self, k: np.ndarray, q: np.ndarray, u: np.ndarray) -> np.ndarray:
        """``g(k+q, k)`` (…, 2, 2) in the sublattice basis for displacements u (2, 3).

        ``u`` = e_s/√M_s of one phonon (q, ν); rows index the sublattice at
        k+q, columns the one at k.
        """
        bonds = self.deltas
        unit = bonds / self.a_cc
        # δ̂·(u_B e^{iq·δ} - u_A) for each bond
        projection = (unit @ u[1]) * np.exp(1j * (bonds @ q)) - unit @ u[0]      # (3,)
        g_ab = self.dt * (np.exp(1j * (k @ bonds.T)) * projection).sum(axis=-1)
        g_ba = self.dt * (np.exp(-1j * ((k + q) @ bonds.T)) * projection).sum(axis=-1)
        out = np.zeros(k.shape[:-1] + (2, 2), dtype=complex)
        out[..., 0, 1] = g_ab
        out[..., 1, 0] = g_ba
        return out


# --------------------------------------------------------------------------
# Raman amplitudes
# --------------------------------------------------------------------------

def zero_point(energy_cm1: np.ndarray) -> np.ndarray:
    """√(ħ/2ω) in Å·√amu: the displacement of one phonon quantum (per e/√M)."""
    from ase.units import _amu, _e, _hbar

    energy_j = np.abs(np.asarray(energy_cm1)) * CM1_TO_EV * _e
    with np.errstate(divide="ignore"):
        return np.where(energy_j > 0, np.sqrt(_hbar ** 2 / (2 * energy_j * _amu)) * 1e10, 0.0)


def _band_elements(electrons: PiElectrons, k1, k2, g_sub, bands):
    """⟨b1, k1| g |b2, k2⟩ for (b1, b2) in ``bands`` (0 = v, 1 = c); g in sublattices."""
    _, v1 = electrons.states(k1)
    _, v2 = electrons.states(k2)
    b1, b2 = bands
    return np.einsum("...s,...st,...t->...", np.conj(v1[..., :, b1]), g_sub, v2[..., :, b2])


def _optical(electrons: PiElectrons, k, polarizations) -> list[np.ndarray]:
    """⟨c,k| e·∂_k H |v,k⟩ for each polarization."""
    _, vec = electrons.states(k)
    out = []
    for e in polarizations:
        v = electrons.velocity(k, np.asarray(e, float))
        out.append(np.einsum("...s,...st,...t->...", np.conj(vec[..., :, 1]), v, vec[..., :, 0]))
    return out


_POLARIZATIONS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def _in_plane(modes_vectors: np.ndarray) -> np.ndarray:
    """Indices of the in-plane branches (the only ones a flat π model couples)."""
    weight = np.sum(np.abs(modes_vectors[:, :, 2]) ** 2, axis=1)
    return np.flatnonzero(weight < 0.5)


def g_band(electrons: PiElectrons, phonons: GraphenePhonons, laser_ev: float,
           gamma: float = 0.1, nk: int = 480) -> dict:
    """The first-order G band at ``laser_ev``: frequency and intensities.

    Third order: photon absorption at k, electron (or hole) emits the Γ
    phonon, recombination; summed over the whole zone (the G amplitude gets
    contributions far from resonance too) on an ``nk x nk`` mesh. Returns
    ``{"frequency", "intensity" (all polarizations), "parallel" (xx),
    "cross" (xy)}``; for the E2g pair parallel = cross.
    """
    frequencies, vectors = phonons.modes(np.zeros(3))
    optical = [i for i in _in_plane(vectors) if frequencies[i] > 500]
    b = phonons.reciprocal()
    # Γ-centred: a shifted mesh breaks the hexagonal symmetry
    fractions = np.stack(np.meshgrid(np.arange(nk), np.arange(nk), indexing="ij"), -1)
    k = (fractions.reshape(-1, 2) / nk) @ b
    energies, _ = electrons.states(k)
    de = energies[:, 1] - energies[:, 0]
    optics = _optical(electrons, k, _POLARIZATIONS)
    masses = phonons.atoms.get_masses()
    tensor = np.zeros((len(optical), 2, 2), dtype=complex)
    for index, nu in enumerate(optical):
        omega = frequencies[nu] * CM1_TO_EV
        u = vectors[nu] / np.sqrt(masses)[:, None] * zero_point(frequencies[nu])
        g = electrons.coupling(k, np.zeros(3), u)
        g_cc = _band_elements(electrons, k, k, g, (1, 1))
        g_vv = _band_elements(electrons, k, k, g, (0, 0))
        denominator = (laser_ev - de + 1j * gamma) * (laser_ev - omega - de + 1j * gamma)
        for s_index, m_s in enumerate(optics):
            for i_index, m_i in enumerate(optics):
                tensor[index, s_index, i_index] = np.mean(
                    np.conj(m_s) * (g_cc - g_vv) * m_i / denominator)
    power = np.abs(tensor) ** 2
    return {"frequency": float(np.mean(frequencies[optical])), "tensors": tensor,
            "intensity": float(power.sum()), "parallel": float(power[:, 0, 0].sum()),
            "cross": float(power[:, 0, 1].sum())}


def _dirac_points(phonons: GraphenePhonons) -> list[np.ndarray]:
    """K and K' (Cartesian, 1/Å)."""
    b = phonons.reciprocal()
    return [(2 * b[0] + b[1]) / 3, (b[0] + 2 * b[1]) / 3]


def double_resonance(electrons: PiElectrons, phonons: GraphenePhonons, laser_ev: float,
                     gamma: float = 0.1, dk: float = 0.006, dq: float = 0.02,
                     window_ev: float = 1.5, q_radius: float = 0.7,
                     workers: int = 1) -> dict:
    """Second-order (two-phonon) Raman intensities by double resonance.

    Returns ``{"shifts": ω_ν(q) + ω_ν(q) per term (cm⁻¹), "weights": |A|²,
    "q": the phonon momenta, "branch": ν}`` for the overtones ν = ν' of the
    in-plane branches, with q near K and K' (the 2D band, intervalley) and
    near Γ (the 2D' band, intravalley), within ``q_radius`` (1/Å) on a grid of
    step ``dq``. Electron momenta k run over annuli around K and K' where the
    pair energy is within ``window_ev`` of the laser, step ``dk`` (1/Å).
    """
    dirac = _dirac_points(phonons)
    kmax = (laser_ev + window_ev) / (3 * abs(electrons.t) * electrons.a_cc)
    kmin = max(0.0, (laser_ev - window_ev) / (3 * abs(electrons.t) * electrons.a_cc))
    grid = np.arange(-kmax, kmax + dk / 2, dk)
    kx, ky = np.meshgrid(grid, grid, indexing="ij")
    radius = np.hypot(kx, ky)
    ring = (radius <= kmax) & (radius >= kmin)
    offsets = np.stack([kx[ring], ky[ring], np.zeros(ring.sum())], axis=-1)
    k_all = np.concatenate([centre + offsets for centre in dirac])
    area = abs(np.linalg.det(phonons.atoms.cell.array[:2, :2]))
    k_weight = dk ** 2 * area / (2 * np.pi) ** 2
    # q points: discs around Γ, K and K'
    qgrid = np.arange(-q_radius, q_radius + dq / 2, dq)
    qx, qy = np.meshgrid(qgrid, qgrid, indexing="ij")
    disc = np.hypot(qx, qy) <= q_radius
    q_offsets = np.stack([qx[disc], qy[disc], np.zeros(disc.sum())], axis=-1)
    q_all = np.concatenate([centre + q_offsets for centre in [np.zeros(3)] + dirac])
    q_weight = dq ** 2 * area / (2 * np.pi) ** 2
    args = [(electrons, phonons, laser_ev, gamma, k_all, k_weight, chunk)
            for chunk in np.array_split(q_all, max(1, workers * 4))]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(_double_resonance_chunk, args))
    else:
        parts = [_double_resonance_chunk(a) for a in args]
    shifts = np.concatenate([p[0] for p in parts])
    weights = np.concatenate([p[1] for p in parts]) * q_weight
    qs = np.concatenate([p[2] for p in parts])
    branch = np.concatenate([p[3] for p in parts])
    return {"shifts": shifts, "weights": weights, "q": qs, "branch": branch,
            "laser_ev": laser_ev, "gamma": gamma}


def _double_resonance_chunk(args):
    electrons, phonons, laser_ev, gamma, k, k_weight, qs = args
    masses = np.sqrt(phonons.atoms.get_masses())[:, None]
    e_k, _ = electrons.states(k)
    optics_k = _optical(electrons, k, _POLARIZATIONS)
    out_shift, out_weight, out_q, out_branch = [], [], [], []
    for q in qs:
        frequencies, vectors = phonons.modes(q)
        for nu in _in_plane(vectors):
            if frequencies[nu] < 50:
                continue
            omega = frequencies[nu] * CM1_TO_EV
            u1 = vectors[nu] / masses * zero_point(frequencies[nu])          # transfer +q
            u2 = np.conj(u1)                                                  # mode (ν, -q)
            amplitude = np.zeros((2, 2), dtype=complex)
            for sign in (1.0, -1.0):          # both orders of the two emissions
                qq = sign * q
                first, second = (u1, u2) if sign > 0 else (u2, u1)
                amplitude += _four_processes(electrons, k, e_k, optics_k, qq, first, second,
                                             omega, omega, laser_ev, gamma) * k_weight
            out_shift.append(2 * frequencies[nu])
            out_weight.append(float(np.sum(np.abs(amplitude) ** 2)))
            out_q.append(q)
            out_branch.append(nu)
    return (np.array(out_shift), np.array(out_weight), np.array(out_q).reshape(-1, 3),
            np.array(out_branch, dtype=int))


def _four_processes(electrons, k, e_k, optics_k, q, u_first, u_second, w1, w2, laser, gamma):
    """Amplitude tensor A[s, i] (scattered, incident polarization) summed over k.

    The first vertex transfers +q to the electrons (coupling built from
    ``u_first``), the second -q (from ``u_second``).
    """
    kq, kmq = k + q, k - q
    e_kq, _ = electrons.states(kq)
    e_kmq, _ = electrons.states(kmq)
    # couplings, sublattice basis: g(k_to, k_from) with transfer = k_to - k_from
    g1_k = electrons.coupling(k, q, u_first)            # k -> k+q
    g1_kmq = electrons.coupling(kmq, q, u_first)        # k-q -> k
    g2_kq = electrons.coupling(kq, -q, u_second)        # k+q -> k
    g2_k = electrons.coupling(k, -q, u_second)          # k -> k-q
    c, v = 1, 0
    ee = _band_elements(electrons, k, kq, g2_kq, (c, c)) * _band_elements(electrons, kq, k, g1_k, (c, c))
    hh = _band_elements(electrons, kmq, k, g2_k, (v, v)) * _band_elements(electrons, k, kmq, g1_kmq, (v, v))
    eh_first = _band_elements(electrons, kq, k, g1_k, (c, c))
    eh_second = -_band_elements(electrons, k, kq, g2_kq, (v, v))
    he_first = -_band_elements(electrons, k, kmq, g1_kmq, (v, v))
    he_second = _band_elements(electrons, kmq, k, g2_k, (c, c))
    d1 = laser - (e_k[:, 1] - e_k[:, 0]) + 1j * gamma
    d_ee2 = laser - w1 - (e_kq[:, 1] - e_k[:, 0]) + 1j * gamma
    d_hh2 = laser - w1 - (e_k[:, 1] - e_kmq[:, 0]) + 1j * gamma
    d3_k = laser - w1 - w2 - (e_k[:, 1] - e_k[:, 0]) + 1j * gamma
    d3_kq = laser - w1 - w2 - (e_kq[:, 1] - e_kq[:, 0]) + 1j * gamma
    d3_kmq = laser - w1 - w2 - (e_kmq[:, 1] - e_kmq[:, 0]) + 1j * gamma
    optics_kq = _optical(electrons, kq, _POLARIZATIONS)
    optics_kmq = _optical(electrons, kmq, _POLARIZATIONS)
    same = (ee / d_ee2 + hh / d_hh2) / (d1 * d3_k)          # recombination at k
    eh = eh_first * eh_second / (d1 * d_ee2 * d3_kq)         # recombination at k+q
    he = he_first * he_second / (d1 * d_hh2 * d3_kmq)        # recombination at k-q
    amplitude = np.zeros((2, 2), dtype=complex)
    for s in range(2):
        for i in range(2):
            m_i = optics_k[i]
            amplitude[s, i] = np.sum(m_i * (np.conj(optics_k[s]) * same
                                            + np.conj(optics_kq[s]) * eh
                                            + np.conj(optics_kmq[s]) * he))
    return amplitude


def second_order_spectrum(result: dict, grid: Optional[np.ndarray] = None,
                          fwhm: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    """Lorentzian-broadened I(ω) of :func:`double_resonance` (arbitrary units)."""
    shifts, weights = result["shifts"], result["weights"]
    if grid is None:
        grid = np.arange(2000.0, 3600.0, 1.0)
    half = fwhm / 2
    intensity = np.zeros_like(grid)
    for start in range(0, len(shifts), 2000):
        s, w = shifts[start:start + 2000], weights[start:start + 2000]
        intensity += (w[None, :] * half / np.pi
                      / ((grid[:, None] - s[None, :]) ** 2 + half ** 2)).sum(axis=1)
    return grid, intensity


def load_phonons(which: str = "gpaw") -> GraphenePhonons:
    """Force constants: ``"gpaw"`` (stored DFT, PBE), ``"xu"`` (computed now) or a JSON path.

    Files are read as stored and the acoustic sum rule is imposed on reading.
    """
    import json
    from pathlib import Path

    from .params import PARAMETER_DIR, xu_carbon

    if which == "xu":
        return GraphenePhonons.from_model(xu_carbon(), n=6, kmesh=4, kT=0.05)
    path = PARAMETER_DIR / "references" / "gpaw_graphene_phonons.json" if which == "gpaw" \
        else Path(which)
    phonons = GraphenePhonons.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
    # DFT forces carry grid noise that breaks the acoustic sum rule (≈ -180 cm⁻¹
    # "acoustic" modes at Γ for the stored GPAW set); restore it.
    return phonons.with_acoustic_sum_rule()


def graphene_raman(lasers_ev, phonons: Optional[GraphenePhonons] = None, gamma: float = 0.1,
                   dk: float = 0.01, dq: float = 0.03, workers: int = 1,
                   fwhm: float = 10.0) -> list[dict]:
    """G, 2D and 2D' of pristine graphene at each laser energy.

    Per laser: the G frequency and intensity, the 2D and 2D' peak positions
    and integrated intensities (same units as G, so I(2D)/I(G) is meaningful
    within the model), and the second-order spectrum.
    """
    phonons = load_phonons("gpaw") if phonons is None else phonons
    a_cc = float(np.linalg.norm(phonons.atoms.positions[1] - phonons.atoms.positions[0]))
    electrons = PiElectrons.from_parameters(a_cc=a_cc)
    results = []
    for laser in np.atleast_1d(np.asarray(lasers_ev, dtype=float)):
        g = g_band(electrons, phonons, laser, gamma=gamma)
        dr = double_resonance(electrons, phonons, laser, gamma=gamma, dk=dk, dq=dq,
                              workers=workers)
        grid, intensity = second_order_spectrum(dr, np.arange(2000.0, 3700.0, 1.0), fwhm)
        entry = {"laser_ev": float(laser), "g_frequency": g["frequency"],
                 "g_intensity": g["intensity"], "grid": grid, "spectrum": intensity}
        g_freq = g["frequency"]
        for name, low, high in (("2D", 2 * g_freq - 700, 2 * g_freq - 100),
                                ("2D'", 2 * g_freq - 100, 2 * g_freq + 250)):
            mask = (grid >= low) & (grid <= high)
            entry[f"{name}_position"] = float(grid[mask][np.argmax(intensity[mask])])
            entry[f"{name}_intensity"] = float(intensity[mask].sum())
        results.append(entry)
    return results
