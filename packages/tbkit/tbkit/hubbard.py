"""Mean-field Hubbard model: spin polarisation of π systems.

For each spin σ the one-electron Hamiltonian gets the on-site term

    H_σ = H_0 + sum_mu U_mu (n_{mu,-σ} - n0_mu / 2) - σ h

where ``n_{mu,σ}`` are the orbital populations of the other spin (solved
self-consistently), ``n0`` the neutral population and ``h`` a Zeeman energy
(``h = μ_B B``; σ = +1 for up). With U ≈ |t| this is the standard model of
the magnetic edges of zigzag graphene ribbons and flakes (Fernández-Rossier
& Palacios 2007; Yazyev, Rep. Prog. Phys. 73, 056501 (2010)): ferromagnetic
along each edge, antiparallel between edges. For a bipartite lattice the
ground-state total spin follows Lieb's theorem, S = |N_A - N_B| / 2, which
the tests check.

What mean field cannot do, and should not be read as doing: it breaks spin
symmetry to describe correlations. The local moments are the mean-field
order parameter; a true ground state of a finite flake with M = 0 is a
singlet with correlated, not static, moments.

Magnetisation, three ways (decision 4 of docs/PLAN_SUITE.md):

* :func:`magnetization_vs_energy` -- ``m(E) = ∫^E [ρ↑ - ρ↓] dE'`` and its
  derivative ``dm/dE = ρ↑ - ρ↓``, from a converged solution;
* :func:`magnetization_vs_field` -- M against the Zeeman energy h, and the
  susceptibility dM/dh;
* :func:`magnetization_vs_doping` -- M against added charge (or, with
  ``fermi_levels``, against a fixed chemical potential).

M is in Bohr magnetons, ``N↑ - N↓`` (g = 2).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .analysis import dos, populations, spin_moments
from .hamiltonian import System
from .solver import Solution, solve

#: Bohr magneton, eV/T: h = MU_B * B for g = 2 and S = 1/2.
MU_B = 5.7883818060e-5


def tesla_to_ev(field_tesla: float) -> float:
    """Zeeman energy ``μ_B B`` of a field in tesla, eV (1 T ≈ 58 μeV)."""
    return MU_B * field_tesla


@dataclass
class HubbardResult:
    """A converged (or not) mean-field solution."""

    solution: Solution
    populations: np.ndarray            # [spin, orbital]
    energy: float                      # mean-field total energy, eV
    iterations: int
    converged: bool
    U: np.ndarray                      # per orbital, eV
    field: float
    history: list[float] = field(default_factory=list)

    @property
    def magnetization(self) -> float:
        """Total ``N↑ - N↓``, μ_B."""
        return float(self.populations[0].sum() - self.populations[1].sum())

    @property
    def moments(self) -> dict[int, float]:
        """Local moment per atom, μ_B."""
        return spin_moments(self.solution)

    def summary(self) -> str:
        moments = np.array(list(self.moments.values()))
        state = "convergido" if self.converged else "SIN CONVERGER"
        return (f"Hubbard campo medio ({state}, {self.iterations} iteraciones): "
                f"M = {self.magnetization:+.4f} μB, |m| máx = {np.abs(moments).max():.4f} μB, "
                f"E = {self.energy:.6f} eV, E_F = {self.solution.fermi:.4f} eV")


def sublattices(system: System) -> Optional[np.ndarray]:
    """+1/-1 per orbital-bearing atom if the bond graph is bipartite, else None."""
    neighbours: dict[int, set[int]] = {a: set() for a in system.basis.atoms}
    for bond in system.bonds:
        neighbours[bond.i].add(bond.j)
    colour: dict[int, int] = {}
    for start in system.basis.atoms:
        if start in colour:
            continue
        colour[start] = 1
        queue = deque([start])
        while queue:
            atom = queue.popleft()
            for other in neighbours[atom]:
                if other not in colour:
                    colour[other] = -colour[atom]
                    queue.append(other)
                elif colour[other] == colour[atom]:
                    return None
    return np.array([colour[a] for a in system.basis.atoms])


def _orbital_u(system: System, U) -> np.ndarray:
    if U is None:
        table = system.model.hubbard_u
        return np.array([table.get(o.element, 0.0) for o in system.basis.orbitals])
    if isinstance(U, dict):
        return np.array([U.get(o.element, 0.0) for o in system.basis.orbitals])
    return np.full(system.basis.size, float(U))


def _neutral(system: System) -> np.ndarray:
    coordination = system.coordination()
    out = np.zeros(system.basis.size)
    for atom in system.basis.atoms:
        rng = system.basis.of_atom(atom)
        out[rng.start:rng.stop] = system.model.electrons_of(
            system.atoms[atom].symbol, coordination[atom]) / len(rng)
    return out


def initial_populations(system: System, guess="antiferro", amplitude: float = 0.3,
                        seed: int = 0) -> np.ndarray:
    """Starting spin populations: ``antiferro`` (sublattice), ``ferro``, ``random``,
    ``paramagnetic``, or an explicit ``[2, n_orbitals]`` array."""
    n0 = _neutral(system)
    if not isinstance(guess, str):
        return np.asarray(guess, dtype=float)
    pattern = np.zeros(system.basis.size)
    if guess == "antiferro":
        colours = sublattices(system)
        if colours is None:
            raise ValueError("La red no es bipartita: usa guess='random' o 'ferro'.")
        for colour, atom in zip(colours, system.basis.atoms, strict=True):
            pattern[system.basis.of_atom(atom)] = colour
    elif guess == "ferro":
        pattern[:] = 1.0
    elif guess == "random":
        pattern = np.random.default_rng(seed).uniform(-1, 1, system.basis.size)
    elif guess != "paramagnetic":
        raise ValueError(f"guess desconocido: {guess!r}.")
    up = np.clip(n0 / 2 + amplitude * pattern * np.minimum(n0, 2 - n0) / 2, 0, 1)
    return np.array([up, n0 - up])


def mean_field(system: System, U=None, charge: float = 0.0, kpts=None, weights=None,
               kT: float = 0.005, field: float = 0.0, fixed_fermi: Optional[float] = None,
               guess="antiferro", mixing: float = 0.4, tol: float = 1e-7,
               max_iter: int = 2000) -> HubbardResult:
    """Solve the mean-field Hubbard model self-consistently.

    Parameters
    ----------
    system
        Usually a π model; ``U`` defaults to the model's ``hubbard_u``.
    U
        Scalar (eV), per-element dict, or None.
    charge, fixed_fermi
        Electron count (canonical) or chemical potential (grand canonical).
    field
        Zeeman energy h, eV (:func:`tesla_to_ev`).
    guess
        Starting spin pattern (:func:`initial_populations`). The result can
        depend on it: compare ``antiferro`` and ``ferro`` energies when in
        doubt.
    mixing, tol, max_iter
        Linear mixing of populations and the convergence threshold on them.
    """
    u = _orbital_u(system, U)
    n0 = _neutral(system)
    pops = initial_populations(system, guess)
    history: list[float] = []
    solution = None
    converged = False
    for iteration in range(1, max_iter + 1):
        potentials = (u * (pops[1] - n0 / 2) - field, u * (pops[0] - n0 / 2) + field)
        solution = solve(system, kpts, weights, charge=charge, kT=kT,
                         spin_potentials=potentials, fixed_fermi=fixed_fermi)
        new = populations(solution)
        change = float(np.abs(new - pops).max())
        history.append(change)
        pops = (1 - mixing) * pops + mixing * new
        if change < tol:
            pops = new
            converged = True
            break
    # Band energy double-counts the interaction: with V_σ = U (n_-σ - n0/2),
    # E = sum f ε - U n↑ n↓ + U n0²/4 equals E_0 + U (n↑ - n0/2)(n↓ - n0/2).
    energy = solution.band_energy() - float(np.sum(u * pops[0] * pops[1])) \
        + float(np.sum(u * n0 ** 2 / 4))
    return HubbardResult(solution, pops, energy, iteration, converged, u, field, history)


# --------------------------------------------------------------------------
# Magnetisation, three ways
# --------------------------------------------------------------------------

def magnetization_vs_energy(result: HubbardResult, grid: Optional[np.ndarray] = None,
                            sigma: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(E, m(E), dm/dE)``: the spin imbalance of all states below E.

    ``dm/dE = ρ↑(E) - ρ↓(E)`` (Gaussian width ``sigma``) and
    ``m(E) = ∫_{-∞}^{E} dm/dE'``. At E = E_F it equals the magnetisation
    (to within the smearing); above E_F it shows what doping would do.
    Energies relative to the Fermi level.
    """
    solution = result.solution
    if grid is None:
        grid = np.linspace(solution.energies.min() - 1.0, solution.energies.max() + 1.0, 4000)
    absolute = grid + solution.fermi
    _, up = dos(solution, absolute, sigma, spin=0)
    _, down = dos(solution, absolute, sigma, spin=1)
    derivative = up - down
    from scipy.integrate import cumulative_trapezoid

    m = cumulative_trapezoid(derivative, absolute, initial=0.0)
    return grid, m, derivative


def _sweep(system: System, values: Sequence[float], key: str, **kwargs):
    results = []
    guess = kwargs.pop("guess", "antiferro")
    for value in values:
        result = mean_field(system, guess=guess, **{key: value}, **kwargs)
        results.append(result)
        guess = result.populations          # follow the branch: hysteresis is real
    return results


def magnetization_vs_field(system: System, fields: Sequence[float], **kwargs):
    """``(h, M(h), χ = dM/dh, results)`` along a sweep of Zeeman energies (eV).

    Each point starts from the previous solution, so a sweep follows one
    branch; sweep up and down to see hysteresis.
    """
    fields = np.asarray(fields, dtype=float)
    results = _sweep(system, fields, "field", **kwargs)
    m = np.array([r.magnetization for r in results])
    chi = np.gradient(m, fields) if len(fields) > 1 else np.full(1, np.nan)
    return fields, m, chi, results


def magnetization_vs_doping(system: System, charges: Optional[Sequence[float]] = None,
                            fermi_levels: Optional[Sequence[float]] = None, **kwargs):
    """``(x, M(x), results)`` against added charge (e) or a fixed chemical potential (eV).

    Give ``charges`` (canonical: positive removes electrons) or
    ``fermi_levels`` (grand canonical), not both.
    """
    if (charges is None) == (fermi_levels is None):
        raise ValueError("Da charges o fermi_levels (uno de los dos).")
    if charges is not None:
        values = np.asarray(charges, dtype=float)
        results = _sweep(system, values, "charge", **kwargs)
    else:
        values = np.asarray(fermi_levels, dtype=float)
        results = _sweep(system, values, "fixed_fermi", **kwargs)
    return values, np.array([r.magnetization for r in results]), results
