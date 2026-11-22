"""Grand-canonical adsorption of a gas MIXTURE in a rigid carbon framework.

The molecules are the three-site linear models of
:mod:`carbonforge.adsorption.co2` (TraPPE CO2 and N2), the wall and the
Ewald sum are its :class:`~carbonforge.adsorption.co2.CO2System`. What
this module adds is the reservoir and the moves a mixture needs:

* **The reservoir is a Peng-Robinson mixture.** Each species enters the
  insertion criterion through its own ``beta f_i = phi_i y_i P / kT``,
  with the fugacity coefficient from the mixture's equation of state and
  van der Waals one-fluid mixing (``PR_KIJ``). At flue-gas conditions
  (0.1 MPa, 298 K) every phi is within a fraction of a per cent of 1, so
  the reservoir is not where a result can go wrong; at high pressure it
  matters and is still only as good as Peng-Robinson.
* **Identity swaps.** A minority species fills slowly by insertion
  alone, and in a narrow pore insertions of the bulkier species are
  rarely accepted. A swap turns one molecule into the other species at
  the same centre and orientation, so the composition equilibrates
  through moves that do not need a hole. Acceptance is
  ``(f_new N_old) / (f_old (N_new + 1)) exp(-beta dU)``.
* **Selectivity is the adsorbed ratio over the gas ratio**,
  ``(x_1/x_2) / (y_1/y_2)``, from absolute amounts. At 0.1 MPa the bulk
  gas in the pore is a fraction of a per cent of what is adsorbed, so
  absolute and excess selectivities coincide to the error bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

from .co2 import (AVOGADRO, BOLTZMANN, CO2, CUTOFF, N2, PR_KIJ, CO2System,
                  Species, place, random_rotation)

R_SI = 8.314462618


def peng_robinson_mixture(temperature: float, pressure_mpa: float,
                          species: tuple[Species, ...],
                          fractions: tuple[float, ...]
                          ) -> tuple[float, np.ndarray]:
    """Vapour compressibility Z and fugacity coefficients of a mixture.

    van der Waals one-fluid mixing, ``a_ij = sqrt(a_i a_j)(1 - k_ij)``
    with ``k_ij`` from :data:`PR_KIJ` (0 when absent). The largest real
    root of the cubic is taken: the reservoir is a gas. Returns
    ``(Z, phi)`` with ``phi`` in the order of ``species``.
    """
    y = np.asarray(fractions, float)
    if not np.isclose(y.sum(), 1.0):
        raise ValueError(f"mole fractions sum to {y.sum():.4f}, not 1")
    t, p = temperature, pressure_mpa * 1e6
    a = np.empty(len(species))
    b = np.empty(len(species))
    for i, sp in enumerate(species):
        tc, pc = sp.tc, sp.pc_mpa * 1e6
        kappa = 0.37464 + 1.54226 * sp.omega - 0.26992 * sp.omega ** 2
        alpha = (1 + kappa * (1 - np.sqrt(t / tc))) ** 2
        a[i] = 0.45724 * (R_SI * tc) ** 2 / pc * alpha
        b[i] = 0.07780 * R_SI * tc / pc
    k = np.zeros((len(species), len(species)))
    for i, si in enumerate(species):
        for j, sj in enumerate(species):
            if i != j:
                k[i, j] = PR_KIJ.get(frozenset((si.name, sj.name)), 0.0)
    aij = np.sqrt(np.outer(a, a)) * (1 - k)
    am = float(y @ aij @ y)
    bm = float(y @ b)
    big_a = am * p / (R_SI * t) ** 2
    big_b = bm * p / (R_SI * t)
    roots = np.roots([1.0, -(1 - big_b), big_a - 3 * big_b ** 2 - 2 * big_b,
                      -(big_a * big_b - big_b ** 2 - big_b ** 3)])
    real = roots[np.abs(roots.imag) < 1e-10].real
    z = float(real[real > big_b].max())
    s2 = np.sqrt(2.0)
    log_term = np.log((z + (1 + s2) * big_b) / (z + (1 - s2) * big_b))
    ln_phi = (b / bm * (z - 1) - np.log(z - big_b)
              - big_a / (2 * s2 * big_b)
              * (2 * (aij @ y) / am - b / bm) * log_term)
    return z, np.exp(ln_phi)


@dataclass
class MixtureResult:
    """One state point of a mixture."""

    temperature: float
    pressure_mpa: float
    species: list[str]
    gas_fractions: list[float]
    mean_counts: list[float]
    count_errors: list[float]
    mol_per_kg: list[float]          # absolute, per kg of framework
    adsorbed_fractions: list[float]
    selectivity: float               # species 0 over species 1
    selectivity_error: float
    fugacity_coefficients: list[float]
    acceptance: dict[str, float]
    energy_drift_k: float
    warnings: list[str] = field(default_factory=list)


def run_mixture_gcmc(
    framework: Atoms | None,
    temperature: float,
    pressure_mpa: float,
    fractions: tuple[float, ...] = (0.15, 0.85),
    species: tuple[Species, ...] = (CO2, N2),
    equilibration: int = 100_000,
    production: int = 200_000,
    seed: int = 0,
    edges: np.ndarray | None = None,
    blocks: int = 10,
    cutoff: float = CUTOFF,
    wall: dict | None = None,
) -> MixtureResult:
    """Grand-canonical mixture in a rigid framework.

    Moves, a fifth each: insertion and deletion (of a species chosen
    uniformly), identity swap, translation (up to 1 A), rotation. The
    incremental energy is checked against a full recomputation at the
    end, as in :func:`~carbonforge.adsorption.co2.run_co2_gcmc`.

    Parameters
    ----------
    framework
        Orthorhombic periodic carbon, or ``None`` for an empty box.
    temperature, pressure_mpa
        State point of the reservoir.
    fractions
        Gas-phase mole fractions, in the order of ``species``. The
        default is a dry flue gas, 15 % CO2 in N2.
    species
        Two or more :class:`~carbonforge.adsorption.co2.Species`. The
        selectivity reported is of the first over the second.
    wall
        :func:`~carbonforge.adsorption.co2.wall_parameters` of the
        framework for a typed, charged wall; ``None`` is neutral Steele.

    Returns
    -------
    MixtureResult
    """
    rng = np.random.default_rng(seed)
    if framework is not None:
        cell = np.asarray(framework.cell, dtype=float)
        box = np.diag(cell).copy()
        if not np.allclose(cell - np.diag(box), 0.0):
            raise ValueError("the framework cell must be orthorhombic.")
    else:
        if edges is None:
            raise ValueError("an empty box needs `edges`.")
        box = np.asarray(edges, dtype=float)
    species = tuple(species)
    ns = len(species)
    if ns < 2 or len(fractions) != ns:
        raise ValueError("a mixture needs two or more species and one "
                         "fraction per species")
    system = CO2System(framework, box, cutoff, wall, species=species)
    beta = 1.0 / temperature
    _z, phi = peng_robinson_mixture(temperature, pressure_mpa, species,
                                    fractions)
    zeta = (phi * np.asarray(fractions) * pressure_mpa * 1e6
            / (BOLTZMANN * temperature) * 1e-30)
    volume = system.volume
    names = ("insert", "delete", "swap", "move", "rotate")
    tried = dict.fromkeys(names, 0)
    accepted = dict.fromkeys(names, 0)
    running = 0.0
    counts = np.zeros(ns, dtype=int)
    samples: list[np.ndarray] = []

    def of_kind(k: int) -> list[int]:
        return [i for i, kk in enumerate(system.kinds) if kk == k]

    for step in range(equilibration + production):
        choice = rng.random()
        if choice < 0.2:
            tried["insert"] += 1
            k = int(rng.integers(ns))
            sites = place(species[k], rng.random(3) * box,
                          random_rotation(rng))
            energy, delta = system.insertion(sites, k)
            if rng.random() < zeta[k] * volume / (counts[k] + 1) * np.exp(
                    -beta * min(energy, 1e6)):
                system.add(sites, k, delta)
                counts[k] += 1
                running += energy
                accepted["insert"] += 1
        elif choice < 0.4:
            tried["delete"] += 1
            k = int(rng.integers(ns))
            if counts[k]:
                index = int(rng.choice(of_kind(k)))
                energy, delta = system.deletion(index)
                if rng.random() < counts[k] / (zeta[k] * volume) * np.exp(
                        -beta * energy):
                    system.remove(index, delta)
                    counts[k] -= 1
                    running += energy
                    accepted["delete"] += 1
        elif choice < 0.6:
            tried["swap"] += 1
            old_k = int(rng.integers(ns))
            if counts[old_k]:
                new_k = int(rng.choice([k for k in range(ns) if k != old_k]))
                index = int(rng.choice(of_kind(old_k)))
                old = system.molecules[index]
                # Same centre, fresh orientation: uniform, so the move
                # is symmetric and its reverse draws the same way.
                trial = place(species[new_k], old[1], random_rotation(rng))
                e_out, d_out = system.deletion(index)
                system.remove(index, d_out)
                e_in, d_in = system.insertion(trial, new_k)
                energy = e_out + e_in
                ratio = (zeta[new_k] * counts[old_k]
                         / (zeta[old_k] * (counts[new_k] + 1)))
                if rng.random() < ratio * np.exp(-beta * min(energy, 1e6)):
                    system.add(trial, new_k, d_in)
                    counts[old_k] -= 1
                    counts[new_k] += 1
                    running += energy
                    accepted["swap"] += 1
                else:
                    # Put the original back where it was in the list.
                    system.molecules.insert(index, old)
                    system.kinds.insert(index, old_k)
                    system.ewald.structure -= d_out
        else:
            kind = "move" if choice < 0.8 else "rotate"
            tried[kind] += 1
            n = len(system.molecules)
            if n:
                index = int(rng.integers(n))
                old = system.molecules[index]
                if kind == "move":
                    shift = (rng.random(3) - 0.5) * 2.0
                    trial = old + shift
                    trial = trial - box * np.floor(trial[1] / box)
                else:
                    trial = place(species[system.kinds[index]], old[1],
                                  random_rotation(rng))
                energy, delta = system.displacement(index, trial)
                if energy < 0 or rng.random() < np.exp(-beta * energy):
                    system.molecules[index] = trial
                    system.ewald.structure += delta
                    running += energy
                    accepted[kind] += 1
        if step >= equilibration:
            samples.append(counts.copy())

    drift = abs(running - system.total_energy())
    sampled = np.asarray(samples, dtype=float)            # (steps, ns)
    split = np.array_split(sampled, blocks)
    block_means = np.array([b.mean(axis=0) for b in split])
    means = sampled.mean(axis=0)
    errors = block_means.std(axis=0, ddof=1) / np.sqrt(blocks)
    y = np.asarray(fractions, float)
    total = means.sum()
    x = means / total if total else np.zeros(ns)
    # Selectivity per block, so its error is the blocks' scatter.
    with np.errstate(divide="ignore", invalid="ignore"):
        block_sel = (block_means[:, 0] / block_means[:, 1]) / (y[0] / y[1])
        sel = float((means[0] / means[1]) / (y[0] / y[1])) \
            if means[1] else float("inf")
    finite = block_sel[np.isfinite(block_sel)]
    sel_err = (float(finite.std(ddof=1) / np.sqrt(len(finite)))
               if len(finite) > 1 else float("nan"))
    warnings: list[str] = []
    for k, sp in enumerate(species):
        if errors[k] > 0.05 * max(means[k], 1.0):
            warnings.append(f"{sp.name}: block error "
                            f"{100 * errors[k] / max(means[k], 1):.0f} % of "
                            "the mean: too short to quote.")
    if means[1] < 2.0:
        warnings.append(f"{species[1].name} averages {means[1]:.2f} "
                        "molecules: the selectivity rests on too few.")
    if drift > 1e-6 * max(1.0, abs(running)) + 1e-3:
        warnings.append(f"incremental energy drifted {drift:.3g} K from a "
                        "full recomputation.")
    per_kg = (1.0 / (system.carbon_mass_g * 1e-3 * AVOGADRO)
              if system.carbon_mass_g else 0.0)
    return MixtureResult(
        temperature=temperature, pressure_mpa=pressure_mpa,
        species=[sp.name for sp in species], gas_fractions=list(map(float, y)),
        mean_counts=list(map(float, means)),
        count_errors=list(map(float, errors)),
        mol_per_kg=[float(m * per_kg) for m in means],
        adsorbed_fractions=list(map(float, x)),
        selectivity=sel, selectivity_error=sel_err,
        fugacity_coefficients=list(map(float, phi)),
        acceptance={k: accepted[k] / tried[k] if tried[k] else 0.0
                    for k in tried},
        energy_drift_k=drift, warnings=warnings)
