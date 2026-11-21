"""CO2 in a rigid carbon framework: a three-site molecule with charges.

A single Lennard-Jones sphere describes H2 well and CO2 badly: CO2 is
linear, 3.5 A long, and carries a quadrupole that is a large part of
how two molecules attract and of how it packs in a pore. This module
is therefore separate from :mod:`carbonforge.adsorption.gcmc` rather
than a new entry in its table of adsorbates.

The model
---------
* **CO2 is TraPPE** (Potoff & Siepmann, *AIChE J.* **47** (2001) 1676):
  rigid and linear, C=O 1.16 A; C sigma 2.80 A, epsilon/k 27.0 K,
  charge +0.70 e; O sigma 3.05 A, epsilon/k 79.0 K, charge -0.35 e. It
  was fitted to the vapour-liquid coexistence curve, which is why the
  empty-box test below can hold it to the real gas.
* **The wall is Steele's graphite carbon**, sigma 3.40 A, epsilon/k
  28.0 K, combined with Lorentz-Berthelot. An undoped carbon framework
  is NEUTRAL here, so the electrostatics act between CO2 molecules only.
  A functionalised or doped wall carries charges and needs them added
  to this framework term; that is not done here and the docstring of
  :func:`run_co2_gcmc` says so.
* **Electrostatics by Ewald summation**, with the structure factor kept
  and updated incrementally, so a move costs one evaluation over the
  k-vectors rather than a full sum. Intramolecular pairs are excluded,
  which is what makes a molecule's self-energy a constant to add on
  insertion and remove on deletion.
* **Classical, no Feynman-Hibbs**: CO2 at 273-298 K is far from the
  quantum regime that makes the correction necessary for H2 at 77 K.
* **No tail corrections** on the Lennard-Jones terms beyond the cutoff.
  At 12 A they are a fraction of a per cent of a pore's adsorption
  energy; the empty-box test is where their absence would show.

The reservoir is Peng-Robinson CO2. Its fugacity is what a GCMC needs,
and below about 5 MPa at room temperature it is within a few per cent
of the reference equation of state -- the same order as TraPPE's own
error, which is why a better reservoir would not buy a better result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from scipy.special import erf, erfc
from scipy.spatial import cKDTree

#: e^2 / (4 pi eps0 k_B), in K A. 1/(4 pi eps0) e^2/A is 14.399645 eV,
#: and k_B is 8.617333e-5 eV/K.
COULOMB_K_A: float = 14.399645 / 8.617333262e-5

AVOGADRO: float = 6.02214076e23
BOLTZMANN: float = 1.380649e-23        # J/K
MOLAR_MASS_CO2: float = 44.0095        # g/mol
MASS_C: float = 12.011

#: TraPPE CO2. Site order is O, C, O.
BOND_CO: float = 1.16
SITE_SIGMA = np.array([3.05, 2.80, 3.05])
SITE_EPSILON = np.array([79.0, 27.0, 79.0])
SITE_CHARGE = np.array([-0.35, 0.70, -0.35])

#: Steele graphite carbon.
WALL_SIGMA: float = 3.40
WALL_EPSILON: float = 28.0

CUTOFF: float = 12.0

#: CO2 critical constants and acentric factor, for Peng-Robinson.
CO2_TC: float = 304.13
CO2_PC_MPA: float = 7.3773
CO2_OMEGA: float = 0.22394


# --------------------------------------------------------------------------
# The reservoir
# --------------------------------------------------------------------------

def peng_robinson(temperature: float, pressure_mpa: float) -> tuple[float, float]:
    """``(Z, phi)`` of CO2 gas: compressibility and fugacity coefficient.

    The vapour root (the largest real Z). Refused in the two-phase or
    liquid region, where the gas root is not the stable one and a
    reservoir there would be a different fluid.
    """
    if temperature < CO2_TC:
        # Saturation pressure from the Wagner-type correlation would be
        # better; PR's own vapour pressure is within a few per cent and
        # is the consistent bound for PR's own roots.
        psat = _pr_saturation_pressure(temperature)
        if pressure_mpa >= psat:
            raise ValueError(
                f"{pressure_mpa} MPa is at or above CO2's saturation "
                f"pressure at {temperature} K ({psat:.2f} MPa by "
                "Peng-Robinson): the reservoir would be liquid.")
    r = 8.314462618
    tc, pc = CO2_TC, CO2_PC_MPA * 1e6
    kappa = 0.37464 + 1.54226 * CO2_OMEGA - 0.26992 * CO2_OMEGA ** 2
    alpha = (1.0 + kappa * (1.0 - np.sqrt(temperature / tc))) ** 2
    a = 0.45724 * r ** 2 * tc ** 2 / pc * alpha
    b = 0.07780 * r * tc / pc
    p = pressure_mpa * 1e6
    big_a = a * p / (r * temperature) ** 2
    big_b = b * p / (r * temperature)
    roots = np.roots([1.0, -(1.0 - big_b), big_a - 3 * big_b ** 2 - 2 * big_b,
                      -(big_a * big_b - big_b ** 2 - big_b ** 3)])
    real = roots[np.abs(roots.imag) < 1e-10].real
    z = float(real.max())
    s2 = np.sqrt(2.0)
    ln_phi = (z - 1.0 - np.log(z - big_b)
              - big_a / (2 * s2 * big_b)
              * np.log((z + (1 + s2) * big_b) / (z + (1 - s2) * big_b)))
    return z, float(np.exp(ln_phi))


def _pr_saturation_pressure(temperature: float) -> float:
    """PR vapour pressure in MPa, by equal fugacities on both roots."""
    low, high = 1e-3, CO2_PC_MPA
    for _ in range(80):
        p = 0.5 * (low + high)
        r = 8.314462618
        tc, pc = CO2_TC, CO2_PC_MPA * 1e6
        kappa = 0.37464 + 1.54226 * CO2_OMEGA - 0.26992 * CO2_OMEGA ** 2
        alpha = (1.0 + kappa * (1.0 - np.sqrt(temperature / tc))) ** 2
        big_a = 0.45724 * r ** 2 * tc ** 2 / pc * alpha * p * 1e6 \
            / (r * temperature) ** 2
        big_b = 0.07780 * r * tc / pc * p * 1e6 / (r * temperature)
        roots = np.roots([1.0, -(1.0 - big_b),
                          big_a - 3 * big_b ** 2 - 2 * big_b,
                          -(big_a * big_b - big_b ** 2 - big_b ** 3)])
        real = np.sort(roots[np.abs(roots.imag) < 1e-10].real)
        real = real[real > big_b]
        if len(real) < 2:
            # One root: above saturation if it is liquid-like, else below.
            if real[0] < 0.3:
                high = p
            else:
                low = p
            continue
        s2 = np.sqrt(2.0)

        def ln_phi(z: float) -> float:
            return (z - 1.0 - np.log(z - big_b) - big_a / (2 * s2 * big_b)
                    * np.log((z + (1 + s2) * big_b) / (z + (1 - s2) * big_b)))

        if ln_phi(real[0]) < ln_phi(real[-1]):
            high = p
        else:
            low = p
    return 0.5 * (low + high)


def gas_density(temperature: float, pressure_mpa: float) -> float:
    """Bulk CO2 number density in molecules per A^3 (Peng-Robinson)."""
    z, _phi = peng_robinson(temperature, pressure_mpa)
    return pressure_mpa * 1e6 / (z * BOLTZMANN * temperature) * 1e-30


def activity(temperature: float, pressure_mpa: float) -> float:
    """beta f in molecules per A^3: what the insertion criterion needs."""
    _z, phi = peng_robinson(temperature, pressure_mpa)
    return phi * pressure_mpa * 1e6 / (BOLTZMANN * temperature) * 1e-30


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def molecule(centre: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Positions of O, C, O for a molecule centred at `centre`."""
    axis = axis / np.linalg.norm(axis)
    return np.array([centre - BOND_CO * axis, centre, centre + BOND_CO * axis])


def random_axis(rng: np.random.Generator) -> np.ndarray:
    """A direction uniform on the sphere."""
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _wrap(delta: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return delta - edges * np.round(delta / edges)


def _lj(r: np.ndarray, sigma: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    s6 = (sigma / r) ** 6
    return 4.0 * epsilon * (s6 * s6 - s6)


# --------------------------------------------------------------------------
# Ewald
# --------------------------------------------------------------------------

@dataclass
class Ewald:
    """Ewald summation in an orthorhombic box, with an updatable S(k).

    Only the half-space of k-vectors is stored; the sum over it is
    doubled. Energies are in kelvin.
    """

    edges: np.ndarray
    cutoff: float = CUTOFF
    precision: float = 1e-5
    alpha: float = field(init=False)
    kvecs: np.ndarray = field(init=False)
    weights: np.ndarray = field(init=False)
    structure: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        tail = np.sqrt(-np.log(self.precision))
        self.alpha = tail / self.cutoff
        k_max = 2.0 * self.alpha * tail
        n_max = np.ceil(k_max * self.edges / (2 * np.pi)).astype(int)
        grid = np.array(np.meshgrid(*[np.arange(-n, n + 1) for n in n_max],
                                    indexing="ij")).reshape(3, -1).T
        # Half-space: first nonzero component positive.
        keep = ((grid[:, 0] > 0)
                | ((grid[:, 0] == 0) & (grid[:, 1] > 0))
                | ((grid[:, 0] == 0) & (grid[:, 1] == 0) & (grid[:, 2] > 0)))
        k = 2 * np.pi * grid[keep] / self.edges
        k2 = np.sum(k * k, axis=1)
        inside = k2 <= k_max ** 2
        self.kvecs = k[inside]
        k2 = k2[inside]
        volume = float(np.prod(self.edges))
        # Doubled for the half space.
        self.weights = (COULOMB_K_A * 4.0 * np.pi / volume
                        * np.exp(-k2 / (4 * self.alpha ** 2)) / k2)
        self.structure = np.zeros(len(self.kvecs), dtype=complex)

    def terms(self, sites: np.ndarray, charges: np.ndarray) -> np.ndarray:
        """This set of charges' contribution to S(k)."""
        return np.exp(1j * (sites @ self.kvecs.T)).T @ charges

    def change(self, delta: np.ndarray) -> float:
        """Reciprocal energy change if S(k) became S(k) + delta."""
        new = self.structure + delta
        return float(np.sum(self.weights * (np.abs(new) ** 2
                                            - np.abs(self.structure) ** 2)))

    def reciprocal(self) -> float:
        return float(np.sum(self.weights * np.abs(self.structure) ** 2))

    def molecule_constant(self) -> float:
        """Self and intramolecular corrections for ONE molecule, in K.

        The same for every molecule (rigid, identical), so an insertion
        adds it and a deletion removes it.
        """
        sites = molecule(np.zeros(3), np.array([0.0, 0.0, 1.0]))
        q = SITE_CHARGE
        self_term = -COULOMB_K_A * self.alpha / np.sqrt(np.pi) * np.sum(q * q)
        intra = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                r = np.linalg.norm(sites[i] - sites[j])
                intra -= COULOMB_K_A * q[i] * q[j] * erf(self.alpha * r) / r
        return float(self_term + intra)


# --------------------------------------------------------------------------
# The system
# --------------------------------------------------------------------------

class CO2System:
    """Molecules in a rigid neutral carbon framework, with Ewald."""

    def __init__(self, framework: Atoms | None, edges: np.ndarray,
                 cutoff: float = CUTOFF):
        edges = np.asarray(edges, dtype=float)
        if edges.min() < 2.0 * cutoff:
            raise ValueError(
                f"the shortest cell edge is {edges.min():.2f} A: the minimum "
                f"image needs at least twice the cutoff ({2 * cutoff:.2f} A).")
        self.edges = edges
        self.volume = float(np.prod(edges))
        self.cutoff = cutoff
        self.ewald = Ewald(edges, cutoff)
        self.constant = self.ewald.molecule_constant()
        if framework is not None and len(framework):
            positions = np.asarray(framework.get_positions(), float) % edges
            self.wall = positions
            self.wall_tree = cKDTree(positions, boxsize=edges)
            self.carbon_mass_g = len(framework) * MASS_C / AVOGADRO
        else:
            self.wall = None
            self.wall_tree = None
            self.carbon_mass_g = 0.0
        self.wall_sigma = 0.5 * (SITE_SIGMA + WALL_SIGMA)
        self.wall_epsilon = np.sqrt(SITE_EPSILON * WALL_EPSILON)
        sig = 0.5 * (SITE_SIGMA[:, None] + SITE_SIGMA[None, :])
        eps = np.sqrt(SITE_EPSILON[:, None] * SITE_EPSILON[None, :])
        self.pair_sigma, self.pair_epsilon = sig, eps
        self.pair_charge = SITE_CHARGE[:, None] * SITE_CHARGE[None, :]
        self.molecules: list[np.ndarray] = []      # each (3, 3)

    # -- energies ---------------------------------------------------------

    def wall_energy(self, sites: np.ndarray) -> float:
        if self.wall_tree is None:
            return 0.0
        total = 0.0
        for s in range(3):
            near = self.wall_tree.query_ball_point(sites[s] % self.edges,
                                                   self.cutoff)
            if not near:
                continue
            r = np.linalg.norm(_wrap(self.wall[near] - sites[s], self.edges),
                               axis=1)
            total += float(np.sum(_lj(np.maximum(r, 0.5), self.wall_sigma[s],
                                      self.wall_epsilon[s])))
        return total

    def fluid_energy(self, sites: np.ndarray, skip: int | None = None) -> float:
        """LJ + real-space Coulomb of one molecule with all the others."""
        if not self.molecules:
            return 0.0
        others = np.asarray(self.molecules)
        if skip is not None:
            others = np.delete(others, skip, axis=0)
            if not len(others):
                return 0.0
        # Molecule cutoff on centres, generous by the molecule's length.
        centre = sites[1]
        dc = np.linalg.norm(_wrap(others[:, 1] - centre, self.edges), axis=1)
        others = others[dc < self.cutoff + 2 * BOND_CO + 0.5]
        if not len(others):
            return 0.0
        delta = _wrap(others[:, None, :, :] - sites[None, :, None, :],
                      self.edges)               # (m, 3 mine, 3 theirs, 3)
        r = np.maximum(np.linalg.norm(delta, axis=-1), 0.5)
        inside = r < self.cutoff
        lj = _lj(r, self.pair_sigma[None], self.pair_epsilon[None])
        coulomb = (COULOMB_K_A * self.pair_charge[None]
                   * erfc(self.ewald.alpha * r) / r)
        return float(np.sum((lj + coulomb) * inside))

    def total_energy(self) -> float:
        """Every term from scratch. For checking the incremental moves."""
        total = 0.0
        saved = self.molecules
        for i, sites in enumerate(saved):
            total += self.wall_energy(sites)
            self.molecules = saved[:i]
            total += self.fluid_energy(sites)
        self.molecules = saved
        structure = np.zeros_like(self.ewald.structure)
        for sites in saved:
            structure += self.ewald.terms(sites, SITE_CHARGE)
        total += float(np.sum(self.ewald.weights * np.abs(structure) ** 2))
        total += len(saved) * self.constant
        return total

    # -- moves ------------------------------------------------------------

    def insertion(self, sites: np.ndarray) -> tuple[float, np.ndarray]:
        delta = self.ewald.terms(sites, SITE_CHARGE)
        energy = (self.wall_energy(sites) + self.fluid_energy(sites)
                  + self.ewald.change(delta) + self.constant)
        return energy, delta

    def deletion(self, index: int) -> tuple[float, np.ndarray]:
        sites = self.molecules[index]
        delta = -self.ewald.terms(sites, SITE_CHARGE)
        energy = -(self.wall_energy(sites) + self.fluid_energy(sites, index)
                   + self.constant) + self.ewald.change(delta)
        return energy, delta

    def displacement(self, index: int, trial: np.ndarray) -> tuple[float, np.ndarray]:
        old = self.molecules[index]
        delta = (self.ewald.terms(trial, SITE_CHARGE)
                 - self.ewald.terms(old, SITE_CHARGE))
        energy = (self.wall_energy(trial) + self.fluid_energy(trial, index)
                  - self.wall_energy(old) - self.fluid_energy(old, index)
                  + self.ewald.change(delta))
        return energy, delta


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------

@dataclass
class CO2Result:
    """One state point."""

    temperature: float
    pressure_mpa: float
    mean_count: float
    count_error: float
    mol_per_kg: float            # absolute, per kg of framework carbon
    excess_mol_per_kg: float     # beyond bulk gas in the accessible volume
    bulk_count: float            # bulk gas molecules the same volume holds
    acceptance: dict[str, float]
    energy_drift_k: float        # incremental vs recomputed, at the end
    warnings: list[str] = field(default_factory=list)


def run_co2_gcmc(
    framework: Atoms | None,
    temperature: float,
    pressure_mpa: float,
    equilibration: int = 100_000,
    production: int = 200_000,
    seed: int = 0,
    edges: np.ndarray | None = None,
    void_fraction: float | None = None,
    blocks: int = 10,
    cutoff: float = CUTOFF,
) -> CO2Result:
    """Grand-canonical CO2 in a rigid NEUTRAL carbon framework.

    Moves, each a quarter: insertion (random centre and orientation),
    deletion, translation (up to 1 A), rotation (a fresh random axis
    about the same centre). The incremental energy is checked against a
    full recomputation at the end and the difference is reported, since
    an Ewald update that drifts would bias every average silently.

    Parameters
    ----------
    framework
        Orthorhombic periodic carbon, or ``None`` for the empty box
        (with ``edges``). Its atoms are uncharged: a doped or
        functionalised wall needs framework charges this does not add.
    temperature, pressure_mpa
        State point. The reservoir is Peng-Robinson CO2 and is refused
        at or above saturation.
    equilibration, production
        Move counts; only production is averaged.
    seed
        For reproducibility.
    edges
        Box edges in A when there is no framework.
    void_fraction
        Accessible fraction for the excess. ``None`` uses the whole
        cell and overstates the bulk correction, as for H2.

    Returns
    -------
    CO2Result
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
    system = CO2System(framework, box, cutoff)
    beta = 1.0 / temperature
    zeta = activity(temperature, pressure_mpa)
    volume = system.volume
    tried = dict.fromkeys(("insert", "delete", "move", "rotate"), 0)
    accepted = dict.fromkeys(tried, 0)
    running = 0.0
    counts: list[int] = []
    for step in range(equilibration + production):
        choice = rng.random()
        n = len(system.molecules)
        if choice < 0.25:
            tried["insert"] += 1
            sites = molecule(rng.random(3) * box, random_axis(rng))
            energy, delta = system.insertion(sites)
            if rng.random() < zeta * volume / (n + 1) * np.exp(
                    -beta * min(energy, 1e6)):
                system.molecules.append(sites)
                system.ewald.structure += delta
                running += energy
                accepted["insert"] += 1
        elif choice < 0.5:
            tried["delete"] += 1
            if n:
                index = int(rng.integers(n))
                energy, delta = system.deletion(index)
                if rng.random() < n / (zeta * volume) * np.exp(
                        -beta * energy):
                    system.molecules.pop(index)
                    system.ewald.structure += delta
                    running += energy
                    accepted["delete"] += 1
        else:
            kind = "move" if choice < 0.75 else "rotate"
            tried[kind] += 1
            if n:
                index = int(rng.integers(n))
                old = system.molecules[index]
                centre = old[1]
                if kind == "move":
                    centre = (centre + (rng.random(3) - 0.5) * 2.0) % box
                    axis = old[2] - old[1]
                else:
                    axis = random_axis(rng)
                trial = molecule(centre, axis)
                energy, delta = system.displacement(index, trial)
                if energy < 0 or rng.random() < np.exp(-beta * energy):
                    system.molecules[index] = trial
                    system.ewald.structure += delta
                    running += energy
                    accepted[kind] += 1
        if step >= equilibration:
            counts.append(len(system.molecules))

    drift = abs(running - system.total_energy())
    sampled = np.asarray(counts, dtype=float)
    means = np.array([b.mean() for b in np.array_split(sampled, blocks)])
    mean = float(sampled.mean())
    error = float(means.std(ddof=1) / np.sqrt(blocks))
    accessible = 1.0 if void_fraction is None else float(void_fraction)
    bulk = gas_density(temperature, pressure_mpa) * volume * accessible
    warnings: list[str] = []
    if void_fraction is None and framework is not None:
        warnings.append("no void fraction: the excess uses the whole cell "
                        "and overstates the bulk correction.")
    if error > 0.05 * max(mean, 1.0):
        warnings.append(f"block error {100 * error / max(mean, 1):.0f} % "
                        "of the mean: too short to quote.")
    if drift > 1e-6 * max(1.0, abs(running)) + 1e-3:
        warnings.append(f"incremental energy drifted {drift:.3g} K from a "
                        "full recomputation: the Ewald update is wrong.")
    per_kg = (1.0 / (system.carbon_mass_g * 1e-3 * AVOGADRO)
              if system.carbon_mass_g else 0.0)
    return CO2Result(
        temperature=temperature, pressure_mpa=pressure_mpa,
        mean_count=mean, count_error=error,
        mol_per_kg=mean * per_kg, excess_mol_per_kg=(mean - bulk) * per_kg,
        bulk_count=bulk,
        acceptance={k: accepted[k] / tried[k] if tried[k] else 0.0
                    for k in tried},
        energy_drift_k=drift, warnings=warnings)


def henry_co2(framework: Atoms, temperature: float, samples: int = 200_000,
              seed: int = 0, cutoff: float = CUTOFF) -> tuple[float, float]:
    """Widom estimate of CO2's Henry constant and zero-coverage q_st.

    Random centre and orientation; only the wall term (one molecule
    alone has no fluid or Ewald energy beyond its constant, which
    cancels). Returns ``(K_H in mol/kg/MPa, q_st in kJ/mol)`` on the
    same basis as :mod:`carbonforge.adsorption.henry`.
    """
    rng = np.random.default_rng(seed)
    cell = np.diag(np.asarray(framework.cell, dtype=float))
    system = CO2System(framework, cell, cutoff)
    energies = np.empty(samples)
    for i in range(samples):
        energies[i] = system.wall_energy(
            molecule(rng.random(3) * cell, random_axis(rng)))
    energies = np.minimum(energies, 5e4)
    weight = np.exp(-energies / temperature)
    mean_weight = float(weight.mean())
    # <N>/P = V <w> / (k_B T); per kg of carbon.
    per_pa = system.volume * 1e-30 * mean_weight / (BOLTZMANN * temperature)
    k_h = per_pa * 1e6 / AVOGADRO / (system.carbon_mass_g * 1e-3)
    mean_u = float(np.sum(weight * energies) / np.sum(weight))
    q_st = (-mean_u + temperature) * 8.314462618e-3
    return k_h, q_st
