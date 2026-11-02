"""Spin-polarised self-consistent charges: spin-DFTB on any s/p (or π) model.

Each shell l of atom A carries a Mulliken magnetisation ``m_Al = n_Al↑ - n_Al↓``; the
energy gains (Köhler, Seifert & Frauenheim, Chem. Phys. 309, 23 (2005))

    E_spin = ½ Σ_A Σ_ll' W_A,ll' m_Al m_Al'

with W the spin constants of the free atom (``parameters/spin_constants.json``, computed by
``recipes/spin_constants``, never fitted; negative: parallel spins on one atom lower the
energy). Spin σ = ±1 then sees, on every orbital μ of shell l of atom A,

    V_μ^σ = Σ_B γ_AB Δq_B  +  σ Σ_l' W_A,ll' m_Al',     H^σ_μν += ½ S_μν (V_μ^σ + V_ν^σ),

the first term the charge SCC of :mod:`tbkit.scc` (on when the set was made with it,
``TBModel.scc``, or when asked), the second the spin term. The energy is evaluated as a
functional of the output density (as in :func:`tbkit.scc.self_consistent`):

    E = Σ_σ Σ f ε − V_in·q_out + ½ Δq γ Δq − s_in·m_out + ½ m W m.

Forces need no new term: W is on-site and constant, so the only geometry dependence is
H0, S (through Mulliken) and γ, which the SCC forces already carry, with ρ and the
energy-weighted density summed over both spins.

The magnetisation is free (one Fermi level) or fixed (``magnetization``, two Fermi levels:
singlet against triplet, ferro against antiferro of a given total). The starting pattern
(``guess``) decides which minimum is found; compare energies.

Single-orbital limit: a π model with W = -U/2 and no charge SCC is the mean-field Hubbard
model of :mod:`tbkit.hubbard` whenever the charges stay neutral (half filling on a
bipartite lattice), so Lieb's theorem holds for it too (tested). Like any mean field, the
moments are an order parameter, not a correlated ground state.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Optional

import numpy as np

from .analysis import populations
from .hamiltonian import System
from .hubbard import _neutral
from .params import angular, read_parameter_file
from .scc import anderson
from .solver import Solution, _fermi_dirac, fermi_level

SHELLS = ("s", "p")


@dataclass
class SpinResult:
    solution: Solution
    charges: dict[int, float]          # net charge per atom, e (+ = electrons lost)
    moments: dict[int, float]          # n↑ - n↓ per atom, μB
    magnetization: float               # total N↑ - N↓, μB
    energy: float                      # eV, without -TS and repulsion
    iterations: int
    converged: bool
    history: list[float] = dataclass_field(default_factory=list)
    dq: np.ndarray = dataclass_field(default=None, repr=False)
    m: np.ndarray = dataclass_field(default=None, repr=False)          # per shell
    gamma: Optional[np.ndarray] = dataclass_field(default=None, repr=False)
    shifts: np.ndarray = dataclass_field(default=None, repr=False)     # [2, orbitals]
    U: object = None                   # the U of the charge SCC (for the forces' γ)

    def summary(self) -> str:
        state = "convergido" if self.converged else "SIN CONVERGER"
        top = max(self.moments.items(), key=lambda kv: abs(kv[1]))
        return (f"SCC con espín ({state}, {self.iterations} iteraciones): E = "
                f"{self.energy:.6f} eV; M = {self.magnetization:+.4f} μB; momento máximo "
                f"{top[1]:+.4f} μB en el átomo {top[0]}")


@dataclass
class Shells:
    """The shells of a system's model atoms and the block-diagonal W between them."""

    atom: np.ndarray        # model-atom index (position in basis.atoms) of each shell
    of_orbital: np.ndarray  # shell of each orbital
    w: np.ndarray           # (n_shell, n_shell), eV


def spin_table(model) -> dict:
    """W per element as ``{element: {"ss": .., "sp": .., "pp": ..}}`` (eV): the model's own
    ``spin_constants`` if its file has them, else ``parameters/spin_constants.json``."""
    data = model.metadata.get("parameters", {}).get("spin_constants")
    if data is None:
        data = read_parameter_file("spin_constants")["spin_constants"]
    return {el: {k: float(v["value"] if isinstance(v, dict) else v) for k, v in t.items()}
            for el, t in data.items()}


def spin_scale() -> float:
    """λ of ``parameters/spin_scale.json``: the one number fitted to GPAW (the NM − AFM
    energy of a zigzag ribbon, ``recipes/spin_validation fit``)."""
    return float(read_parameter_file("spin_scale")["scale"]["value"])


def scaled_table(model, scale: Optional[float] = None) -> dict:
    """The atomic W times λ (default :func:`spin_scale`)."""
    lam = spin_scale() if scale is None else float(scale)
    return {el: {k: lam * x for k, x in t.items()} for el, t in spin_table(model).items()}


def shells_of(system: System, W=None) -> Shells:
    """``W``: None (the atomic table), ``"scaled"`` (the atomic table × λ, fitted to GPAW:
    :func:`spin_scale`), a per-element table, or a number (every pair of shells of every
    atom: the single-orbital limit, W = -U/2 for a Hubbard U)."""
    if isinstance(W, str):
        if W != "scaled":
            raise ValueError(f"W desconocido: {W!r} (None, 'scaled', tabla o número).")
        W = scaled_table(system.model)
    table = None if np.isscalar(W) else (spin_table(system.model) if W is None else W)
    atoms = system.basis.atoms
    owner, kinds = [], []
    of_orbital = np.zeros(system.basis.size, dtype=int)
    for index, atom in enumerate(atoms):
        rng = system.basis.of_atom(atom)
        present = []
        for mu in range(rng.start, rng.stop):
            shell = angular(system.basis.orbitals[mu].name)
            if shell not in present:
                present.append(shell)
            of_orbital[mu] = len(owner) + present.index(shell)
        owner += [index] * len(present)
        kinds += [(system.atoms[atom].symbol, shell) for shell in present]
    w = np.zeros((len(owner), len(owner)))
    for a in range(len(owner)):
        for b in range(len(owner)):
            if owner[a] != owner[b]:
                continue
            if table is None:
                w[a, b] = float(W)
                continue
            element, la = kinds[a]
            lb = kinds[b][1]
            key = "".join(sorted((la, lb), key=SHELLS.index))
            if element not in table or key not in table[element]:
                raise ValueError(f"Falta la constante de espín W_{key} de {element}: corre "
                                 "python -m tbkit.recipes.spin_constants con ese elemento.")
            w[a, b] = table[element][key]
    return Shells(np.array(owner), of_orbital, w)


def _bond_sublattices(system: System) -> Optional[np.ndarray]:
    """±1 per model atom from the bond graph (closer than 1.85 Å), None if not bipartite."""
    from collections import deque

    from .neighbors import neighbor_list

    atoms = system.basis.atoms
    index = {a: i for i, a in enumerate(atoms)}
    ii, jj = neighbor_list("ij", system.atoms, 1.85)
    neighbours = {a: set() for a in atoms}
    for i, j in zip(ii, jj, strict=True):
        if i in index and j in index:
            neighbours[int(i)].add(int(j))
    colour: dict[int, int] = {}
    for start in atoms:
        if start in colour:
            continue
        colour[start] = 1
        queue = deque([start])
        while queue:
            a = queue.popleft()
            for b in neighbours[a]:
                if b not in colour:
                    colour[b] = -colour[a]
                    queue.append(b)
                elif colour[b] == colour[a]:
                    return None
    return np.array([colour[a] for a in atoms])


def initial_moments(system: System, shells: Shells, guess="ferro", amplitude: float = 0.5,
                    seed: int = 0) -> np.ndarray:
    """Starting m per shell: ``ferro``, ``antiferro`` (bond-graph sublattices), ``random``,
    ``none``, or explicit moments per model atom (μB, put on its last shell)."""
    n_atoms = len(system.basis.atoms)
    if isinstance(guess, str):
        if guess == "ferro":
            per_atom = np.full(n_atoms, amplitude)
        elif guess == "antiferro":
            colours = _bond_sublattices(system)
            if colours is None:
                raise ValueError("La red de enlaces no es bipartita: usa guess='ferro', "
                                 "'random' o momentos explícitos.")
            per_atom = amplitude * colours
        elif guess == "random":
            per_atom = amplitude * np.random.default_rng(seed).uniform(-1, 1, n_atoms)
        elif guess == "none":
            per_atom = np.zeros(n_atoms)
        else:
            raise ValueError(f"guess desconocido: {guess!r}.")
    else:
        per_atom = np.asarray(guess, dtype=float)
    m = np.zeros(len(shells.atom))
    for a in range(n_atoms):
        last = np.flatnonzero(shells.atom == a)[-1]
        m[last] = per_atom[a]
    return m


def solve_spin(system: System, v_up: np.ndarray, v_down: np.ndarray, charge: float,
               kT: float, kpts=None, weights=None,
               magnetization: Optional[float] = None) -> Solution:
    """Both spin channels with their orbital potentials (``½ S (V_μ + V_ν)``), one Fermi
    level, or two when the total ``magnetization`` (N↑ - N↓) is fixed."""
    from scipy.linalg import eigh

    if kpts is None:
        kpts, weights = np.zeros((1, 3)), np.ones(1)
    kpts = np.atleast_2d(np.asarray(kpts, dtype=float))
    weights = np.full(len(kpts), 1.0 / len(kpts)) if weights is None else np.asarray(weights)
    energies, vectors, overlaps = [[], []], [[], []], []
    for k in kpts:
        h, s = system.hamiltonian(k)
        for spin, v in enumerate((v_up, v_down)):
            if s is None:
                e, c = eigh(h + np.diag(v))
            else:
                e, c = eigh(h + 0.5 * s * (v[:, None] + v[None, :]), s)
            energies[spin].append(e)
            vectors[spin].append(c)
        overlaps.append(s)
    energies = np.array(energies)
    vectors = np.array(vectors)
    electrons = system.electrons - charge
    info = {}
    if magnetization is None:
        mu = fermi_level(energies, weights, electrons, kT, 1.0)
        occupations = _fermi_dirac(energies, mu, kT)
    else:
        counts = (0.5 * (electrons + magnetization), 0.5 * (electrons - magnetization))
        levels = [fermi_level(energies[s:s + 1], weights, counts[s], kT, 1.0) for s in (0, 1)]
        occupations = np.array([_fermi_dirac(energies[s], levels[s], kT) for s in (0, 1)])
        mu = 0.5 * (levels[0] + levels[1])
        info = {"fermi_up": levels[0], "fermi_down": levels[1]}
    solution = Solution(system, kpts, weights, energies, vectors, occupations, mu, electrons,
                        kT, None if system.model.orthogonal else np.array(overlaps))
    solution.info.update(info)
    return solution


def spin_polarised(system: System, W=None, U=None, charges: Optional[bool] = None,
                   charge: float = 0.0, magnetization: Optional[float] = None,
                   guess="ferro", amplitude: float = 0.5, kT: float = 0.01,
                   mixing: float = 0.3, tol: float = 1e-9, max_iter: int = 1000,
                   history_length: int = 6, kpts=None, weights=None,
                   seed: int = 0) -> SpinResult:
    """Spin-polarised SCC (see the module docstring).

    Parameters
    ----------
    W
        Spin constants: None (the atomic table), ``"scaled"`` (× λ fitted to GPAW),
        ``{element: {"ss", "sp", "pp"}}`` or a number.
    U, charges
        Charge SCC with the model's (or these) Hubbard U: on by default when the set was
        made with it (``model.scc``).
    magnetization
        Fixed N↑ - N↓ (μB), or None to let it settle.
    guess, amplitude
        Starting moments (:func:`initial_moments`).
    tol
        On the largest change of an atomic charge or a shell moment between input and
        output; forces need it tight.
    """
    if system.periodic and kpts is None:
        from .kpoints import mesh

        kpts, weights = mesh(system)
    if not system.periodic:
        kpts, weights = None, None
    charges = system.model.scc if charges is None else charges
    shells = shells_of(system, W)
    atoms = system.basis.atoms
    owner = np.array([atoms.index(o.atom) for o in system.basis.orbitals])
    n0 = _neutral(system)
    neutral_atom = np.array([n0[system.basis.of_atom(a).start:system.basis.of_atom(a).stop].sum()
                             for a in atoms])
    gamma = None
    if charges:
        from .scc import gamma_matrix

        gamma = gamma_matrix(system, U)
    n_atoms = len(atoms)
    x = np.r_[np.zeros(n_atoms), initial_moments(system, shells, guess, amplitude, seed)]
    inputs, residuals, history = [], [], []
    converged = False

    def potentials(x):
        dq, m = x[:n_atoms], x[n_atoms:]
        v_charge = (gamma @ dq)[owner] if gamma is not None else np.zeros(len(owner))
        v_spin = (shells.w @ m)[shells.of_orbital]
        return v_charge, v_spin

    for iteration in range(1, max_iter + 1):
        v_charge, v_spin = potentials(x)
        solution = solve_spin(system, v_charge + v_spin, v_charge - v_spin, charge, kT, kpts,
                              weights, magnetization)
        pops = populations(solution)
        total = pops.sum(axis=0)
        dq_out = np.bincount(owner, weights=total, minlength=n_atoms) - neutral_atom
        m_out = np.bincount(shells.of_orbital, weights=pops[0] - pops[1],
                            minlength=len(shells.atom))
        out = np.r_[dq_out if gamma is not None else np.zeros(n_atoms), m_out]
        residual = out - x
        change = float(np.abs(residual).max())
        history.append(change)
        if change < tol:
            converged = True
            break
        inputs = (inputs + [x])[-history_length:]
        residuals = (residuals + [residual])[-history_length:]
        x = anderson(inputs, residuals, mixing)
    dq_in, m_in = x[:n_atoms], x[n_atoms:]
    s_in = shells.w @ m_in
    energy = solution.band_energy() - float(s_in @ m_out) + 0.5 * float(m_out @ shells.w @ m_out)
    if gamma is not None:
        shift_in = gamma @ dq_in
        energy += (-float(shift_in @ (neutral_atom + dq_out))
                   + 0.5 * float(dq_out @ gamma @ dq_out))
    v_charge, v_spin = potentials(x)
    per_atom_m = np.bincount(shells.atom, weights=m_out, minlength=n_atoms)
    return SpinResult(
        solution=solution,
        charges={a: float(-q) for a, q in zip(atoms, dq_out, strict=True)},
        moments={a: float(m) for a, m in zip(atoms, per_atom_m, strict=True)},
        magnetization=float(m_out.sum()), energy=float(energy), iterations=iteration,
        converged=converged, history=history, dq=dq_out, m=m_out, gamma=gamma,
        shifts=np.array([v_charge + v_spin, v_charge - v_spin]), U=U)


def energy_and_forces(result: SpinResult, need_forces: bool = True):
    """Free energy and forces of a converged spin-polarised solution.

    ``∂E/∂R = Tr[ρ ∂H0] - Σ_σ Tr[(W_σ - ½ ρ_σ∘(V^σ_μ + V^σ_ν)) ∂S] + ½ Σ Δq_A Δq_B ∂γ_AB
    + ∂E_rep``; the spin term adds nothing explicit (W is on-site)."""
    from .forces import band_forces, entropy_term

    solution = result.solution
    system = solution.system
    if system.model.repulsive is None:
        raise ValueError(f"El modelo '{system.model.name}' no tiene parte repulsiva.")
    ts = entropy_term(solution)
    e_rep, f_rep = system.model.repulsive.energy_and_forces(system.atoms)
    energy = result.energy - ts + e_rep
    parts = {"spin_scc": result.energy, "entropy_TS": ts, "repulsive": e_rep}
    if not need_forces:
        return energy, None, parts
    weighted = None
    if not system.model.orthogonal:
        weighted = []
        for k in range(len(solution.kpts)):
            total = 0
            for spin in (0, 1):
                c = solution.vectors[spin, k]
                f = solution.occupations[spin, k]
                rho = (c * f) @ c.conj().T
                w = (c * (f * solution.energies[spin, k])) @ c.conj().T
                v = result.shifts[spin]
                total = total + w - 0.5 * rho * (v[:, None] + v[None, :])
            weighted.append(total)
    forces = band_forces(solution, energy_weighted=weighted) + f_rep
    if result.gamma is not None:
        forces += _gamma_forces(system, result.dq, result.gamma, result.U)
    return energy, forces, parts


def _gamma_forces(system: System, dq: np.ndarray, gamma: np.ndarray, U=None) -> np.ndarray:
    """-½ Σ Δq_A Δq_B ∂γ_AB/∂R, as in :func:`tbkit.scc.energy_and_forces`."""
    from .scc import COULOMB, _periodic_gamma

    forces = np.zeros((len(system.atoms), 3))
    atoms = system.basis.atoms
    if system.periodic:
        _, dgamma = _periodic_gamma(system, U)
        gradient = np.einsum("a,b,abx->ax", dq, dq, dgamma)
    else:
        positions = system.atoms.get_positions()[atoms]
        vectors = positions[:, None] - positions[None, :]
        r = np.linalg.norm(vectors, axis=-1)
        dgamma = -r * gamma ** 3 / COULOMB ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = np.where(r[..., None] > 0, vectors / r[..., None], 0.0)
        gradient = np.einsum("ab,abx->ax", np.outer(dq, dq) * dgamma, unit)
    for index, atom in enumerate(atoms):
        forces[atom] -= gradient[index]
    return forces
