"""Grand-canonical Monte Carlo of hydrogen in a rigid carbon framework.

A prototype, written to be **checked against published isotherms before
it is believed**, which is why :mod:`carbonforge.adsorption.bundle`
reproduces one paper's adsorbent and :mod:`carbonforge.adsorption.bulk`
reproduces its reservoir. What the three together can do is run the
case Levesque *et al.* ran and see whether the same numbers come out.

**What a grand-canonical simulation is.** The cell exchanges molecules
with a reservoir at fixed temperature and fugacity, so the number of
adsorbed molecules fluctuates and its average is the loading. Three
moves, picked at random: insert a molecule at a random point, delete
one at random, displace one. Their acceptance probabilities are the
usual ones, with the reservoir entering only through the activity
``z = βf`` that :meth:`~carbonforge.adsorption.bulk.VirialGas.activity`
provides.

**Equilibration is separated from production and is not optional.**
Starting from an empty cell, the first thousands of moves are the cell
filling up, and averaging over them drags every number toward zero. The
split is an argument, the counts are reported, and the production
average carries its own standard error so that a run too short to mean
anything says so.

**What this computes is ABSOLUTE adsorption**, every molecule inside
the cell. An experiment measures **excess**, what is there beyond what
the same volume would hold as bulk gas at the same conditions. The two
differ by ``ρ_bulk × V_accessible`` and the difference is not small
where the bulk is dense: at 77 K and 20 MPa the bulk alone is
0.059 g cm⁻³. Comparing a simulated absolute isotherm with a measured
excess one is the most common way this literature gets compared wrongly,
so :class:`GCMCResult` carries both and makes the caller say which.

**What is left out.** H2 is one Lennard-Jones site: no quadrupole (see
:mod:`carbonforge.adsorption.potentials`). The framework is rigid. The
neighbour search is a tree query per trial move rather than a
pre-tabulated energy grid, which is simpler and exact and makes this
perhaps five times slower than a production code — fine for a
validation run, the first thing to change for a campaign.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

from .bulk import VirialGas, fit_virial
from .potentials import H2_C, H2_H2, MASS_C, MASS_H2, feynman_hibbs

#: Pair cutoff in Å for both interactions. Beyond 12 Å the Lennard-Jones
#: energy of this well depth is under 0.01 K, which is a thousandth of
#: the thermal energy even at 77 K.
CUTOFF: float = 12.0

#: Avogadro's number, for the weight percentage. The atomic masses
#: come from :mod:`~carbonforge.adsorption.potentials`, which is where
#: the pair parameters that use them live; a second copy here is a
#: second thing to get wrong.
AVOGADRO: float = 6.02214076e23


@dataclass
class GCMCResult:
    """What one state point produced, with the caveats attached.

    Attributes
    ----------
    temperature, pressure_mpa
        The state point.
    mean_count, count_error
        Average number of molecules in the cell during production, and
        the standard error of that average from block means. A run
        whose error is a large fraction of the mean has not converged,
        whatever the mean says.
    absolute_density_g_cm3
        Every molecule in the cell, over the cell volume. This is what
        a simulation gives directly, and what the reference tables call
        ``rho_o``.
    excess_density_g_cm3
        Absolute minus what the same volume would hold as bulk gas.
        This is the quantity an experiment measures. Negative values
        are physical at high pressure and are not clipped.
    weight_percent
        ``100 m(H2) / (m(H2) + m(C))`` on the **absolute** basis: the
        hydrogen as a fraction of the loaded material's total mass.
        This is the definition the storage targets use, and the one to
        quote unless a comparison demands otherwise. See
        :attr:`weight_percent_per_carbon` for why a comparison can.
    acceptance
        Accepted fraction per move type. An insertion rate near zero at
        high loading is expected; near zero at low loading means the
        cell is too full of framework for the molecule to fit, which is
        a result about the material.
    """

    temperature: float
    pressure_mpa: float
    mean_count: float
    count_error: float
    absolute_density_g_cm3: float
    excess_density_g_cm3: float
    weight_percent: float
    bulk_density_g_cm3: float
    carbon_mass_g: float = 0.0
    """Mass of the framework in the cell, in grams. Carried so that
    :attr:`weight_percent_per_carbon` needs nothing else."""
    void_fraction: float = 1.0
    """What the excess was computed with. 1.0 means the whole cell, so
    the excess is an upper bound on the correction."""
    snapshots: list[np.ndarray] = field(default_factory=list)
    """Molecule positions (Å) at evenly spaced points in production, one
    ``(m, 3)`` array each. Empty unless ``snapshots`` was asked for. The
    rows are **molecule centres**, one per H2, not atoms — this model
    has one Lennard-Jones site per molecule and there is no second
    hydrogen to place."""
    acceptance: dict[str, float] = field(default_factory=dict)
    equilibration_steps: int = 0
    production_steps: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def weight_percent_per_carbon(self) -> float:
        """``100 m(H2) / m(C))``: hydrogen per mass of *adsorbent*.

        The other convention, and not a pedantic distinction — the two
        differ by a factor ``1 + m(H2)/m(C)``, which is 0.3 % at 1 wt%
        but **5 %** at 5 wt%, so they diverge precisely where cryogenic
        adsorption gets interesting.

        **This is not the convention the reference papers use**, and
        the history of that sentence is worth keeping. Kowalczyk et
        al. (Phys. Chem. Chem. Phys. 9 (2007) 1786) write the formula
        out as their eqn 5 — ``N_H2 m_H2 / (N_H2 m_H2 + N_C m_C)``,
        the total-mass basis — and cite Levesque for it. So
        :attr:`weight_percent` is the one to quote.

        It was briefly concluded here that Levesque's column was
        per-carbon, on the strength of the arithmetic alone: converting
        his ``rho_o`` with the reproduced cell reproduces his 77 K
        ``wt%`` column to 1.4 % on this basis and 3.1 % on the other,
        and lands exactly on his printed 3.2 and 5.2 at the two ends.
        That fit is real and it is not evidence of a convention. With
        the formula in hand the reading is simply that **his two
        columns do not agree with each other at 77 K**, by about 3 %,
        and that the gap happens to look like this basis.

        Which is the argument for validating against ``rho_o`` and
        nothing else: five decimals, no convention, and no second
        column to disagree with. Both weight percentages are still
        returned, because the two differ by ``1 + m(H2)/m(C)`` — 0.3 %
        at 1 wt% but **5 %** at 5 wt% — so anyone comparing numbers
        across papers has to be able to say which they mean.
        """
        hydrogen = self.mean_count * MASS_H2 / AVOGADRO
        return 100.0 * hydrogen / self.carbon_mass_g

    def summary(self) -> str:
        """One line per state point, for a run log."""
        return (
            f"{self.temperature:6.1f} K {self.pressure_mpa:6.2f} MPa  "
            f"N = {self.mean_count:8.1f} +- {self.count_error:5.1f}  "
            f"abs {self.absolute_density_g_cm3:.5f} g/cm3  "
            f"exc {self.excess_density_g_cm3:+.5f}  "
            f"{self.weight_percent:5.2f} wt% total, "
            f"{self.weight_percent_per_carbon:5.2f} wt% per C"
        )


def _minimum_image(delta: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Wrap displacements into the orthorhombic cell. Vectorised."""
    return delta - edges * np.round(delta / edges)


class _Framework:
    """The rigid adsorbent, and the energy a molecule feels from it.

    A tree with ``boxsize`` so that the query is already periodic, built
    once. The alternative — tabulating the energy on a grid — is faster
    per move and introduces an interpolation error on the steepest part
    of the repulsive wall, which is exactly where an insertion is
    decided. For a validation run, exact and slower is the right trade.
    """

    def __init__(self, atoms: Atoms, temperature: float,
                 cutoff: float = CUTOFF):
        edges = np.diag(np.asarray(atoms.cell, dtype=float))
        if not np.allclose(np.asarray(atoms.cell) - np.diag(edges), 0.0):
            raise ValueError(
                "the framework cell must be orthorhombic: the periodic "
                "neighbour search (cKDTree with boxsize) cannot do a "
                "triclinic one, and the minimum image would be wrong."
            )
        # The minimum image is only the nearest image when the cutoff
        # reaches less than half the cell. Past that, `_minimum_image`
        # still returns one displacement per pair while the true sum has
        # two or more images of that neighbour inside the cutoff, so the
        # energy is wrong by whatever those contribute -- silently, and
        # on the steep part of the wall where an insertion is decided.
        # Refused rather than warned for the same reason the triclinic
        # cell is: the result would look entirely plausible.
        if edges.min() < 2.0 * cutoff:
            raise ValueError(
                f"the shortest cell edge is {edges.min():.2f} Å and the pair "
                f"cutoff is {cutoff:.2f} Å. The minimum image needs an edge "
                f"of at least twice the cutoff ({2 * cutoff:.2f} Å), or a "
                "molecule interacts with two images of the same neighbour "
                "and the energy is wrong without looking it. Build a larger "
                "supercell, or pass a smaller `cutoff`."
            )
        positions = np.asarray(atoms.get_positions(), dtype=float) % edges
        self.edges = edges
        self.volume = float(np.prod(edges))
        self.positions = positions
        self.tree = cKDTree(positions, boxsize=edges)
        self.cutoff = cutoff
        self.temperature = temperature
        self.carbon_mass_g = len(atoms) * MASS_C / AVOGADRO

    def energy(self, point: np.ndarray) -> float:
        """Framework-molecule energy in kelvin at one position."""
        near = self.tree.query_ball_point(point % self.edges, self.cutoff)
        if not near:
            return 0.0
        delta = _minimum_image(self.positions[near] - point, self.edges)
        r = np.linalg.norm(delta, axis=1)
        r = np.maximum(r, 1e-6)
        return float(np.sum(feynman_hibbs(r, H2_C, self.temperature)))


def run_gcmc(
    framework: Atoms,
    temperature: float,
    pressure_mpa: float,
    equilibration: int = 200_000,
    production: int = 400_000,
    max_displacement: float = 1.0,
    seed: int = 0,
    gas: VirialGas | None = None,
    blocks: int = 10,
    cutoff: float = CUTOFF,
    void_fraction: float | None = None,
    snapshots: int = 0,
) -> GCMCResult:
    """One state point: insert, delete and displace until it settles.

    Parameters
    ----------
    framework
        Rigid adsorbent in an orthorhombic periodic cell.
    temperature
        In kelvin. It enters three ways: the Boltzmann factor, the
        reservoir's fugacity, and the Feynman-Hibbs correction, which
        is why a potential cached at one temperature is wrong at
        another.
    pressure_mpa
        Reservoir pressure. Converted to a fugacity through ``gas``.
    equilibration, production
        Move counts. The first are discarded; only the second are
        averaged. Both are reported in the result.
    max_displacement
        Maximum trial displacement in Å for a translation move.
    seed
        Required for reproducibility; this is a stochastic method and
        the package's rule is that every one of them takes a seed.
    gas
        The reservoir's equation of state. Defaults to a virial fitted
        to the published bulk densities at this temperature, which only
        exist for 77, 150 and 293 K.
    blocks
        How many blocks the production run is split into for the
        standard error. Fewer than five makes the error itself noise.
    cutoff
        Pair cutoff in Å.
    void_fraction
        Fraction of the cell the gas can occupy, for converting the
        absolute loading to an excess one. This is what
        :func:`~carbonforge.adsorption.porosity.analyse_porosity`
        computes, and its ``helium_void_fraction`` is the member of
        the family the experimental literature means. Left at ``None``
        the whole cell is used, which **overstates** the correction
        because the framework's own volume is counted as accessible —
        an upper bound on the excess correction, and the result says
        which was used.
    snapshots
        How many configurations to keep, spread evenly through
        production. 0 keeps none, which is the default because the
        averages need none and a caller who does not want pictures
        should not pay for them. They are what a paper's "snapshot of
        adsorbed hydrogen" figure is made from, and they are taken
        **during production only**: a configuration from the
        equilibration is a cell still filling up, which looks like a
        result and is not one.

    Returns
    -------
    GCMCResult
        Loading, both bases, acceptance rates and warnings.
    """
    if production < blocks:
        raise ValueError("production must be at least `blocks` moves long")
    rng = np.random.default_rng(seed)
    gas = gas if gas is not None else fit_virial(temperature)
    solid = _Framework(framework, temperature, cutoff=cutoff)
    edges = solid.edges
    volume = solid.volume
    beta = 1.0 / temperature
    # Activity in inverse cubic angstrom: the reservoir enters here and
    # nowhere else.
    activity = gas.activity(pressure_mpa) * 1e-30

    occupied: list[np.ndarray] = []
    accepted = {"insert": 0, "delete": 0, "move": 0}
    tried = {"insert": 0, "delete": 0, "move": 0}

    def fluid_energy(point: np.ndarray, skip: int | None = None) -> float:
        if not occupied:
            return 0.0
        others = np.asarray(occupied)
        if skip is not None:
            others = np.delete(others, skip, axis=0)
            if not len(others):
                return 0.0
        delta = _minimum_image(others - point, edges)
        r = np.linalg.norm(delta, axis=1)
        inside = r < cutoff
        if not np.any(inside):
            return 0.0
        return float(np.sum(feynman_hibbs(
            np.maximum(r[inside], 1e-6), H2_H2, temperature)))

    counts: list[int] = []
    kept: list[np.ndarray] = []
    # Evenly through production, and the LAST one is the final state
    # rather than something short of it -- a figure captioned "at
    # equilibrium" should be the most equilibrated configuration the
    # run reached, not an arbitrary earlier one.
    snapshot_at = ([equilibration + (production * (i + 1)) // snapshots - 1
                    for i in range(snapshots)] if snapshots > 0 else [])
    total = equilibration + production
    for step in range(total):
        choice = rng.random()
        if choice < 1.0 / 3.0:
            tried["insert"] += 1
            point = rng.random(3) * edges
            change = solid.energy(point) + fluid_energy(point)
            n = len(occupied)
            if rng.random() < min(1.0, activity * volume / (n + 1)
                                  * np.exp(-beta * change)):
                occupied.append(point)
                accepted["insert"] += 1
        elif choice < 2.0 / 3.0:
            tried["delete"] += 1
            if occupied:
                index = int(rng.integers(len(occupied)))
                point = occupied[index]
                change = solid.energy(point) + fluid_energy(point, skip=index)
                n = len(occupied)
                if rng.random() < min(1.0, n / (activity * volume)
                                      * np.exp(beta * change)):
                    occupied.pop(index)
                    accepted["delete"] += 1
        else:
            tried["move"] += 1
            if occupied:
                index = int(rng.integers(len(occupied)))
                point = occupied[index]
                trial = (point + (rng.random(3) - 0.5)
                         * 2.0 * max_displacement) % edges
                before = solid.energy(point) + fluid_energy(point, skip=index)
                after = solid.energy(trial) + fluid_energy(trial, skip=index)
                if rng.random() < min(1.0, np.exp(-beta * (after - before))):
                    occupied[index] = trial
                    accepted["move"] += 1
        if step >= equilibration:
            counts.append(len(occupied))
            if snapshot_at and step >= snapshot_at[0]:
                snapshot_at.pop(0)
                kept.append(np.asarray(occupied, dtype=float).reshape(-1, 3))

    sampled = np.asarray(counts, dtype=float)
    block_means = np.array([b.mean() for b in np.array_split(sampled, blocks)])
    mean_count = float(sampled.mean())
    error = float(block_means.std(ddof=1) / np.sqrt(blocks))

    hydrogen_g = mean_count * MASS_H2 / AVOGADRO
    volume_cm3 = volume * 1e-24
    absolute = hydrogen_g / volume_cm3
    bulk = gas.mass_density(pressure_mpa)

    accessible = 1.0 if void_fraction is None else float(void_fraction)
    warnings: list[str] = []
    if void_fraction is None:
        warnings.append(
            "no void fraction was given, so the excess uses the WHOLE cell "
            "volume and overstates the correction. Pass the helium void "
            "fraction from analyse_porosity for the quantity an experiment "
            "measures.")
    if error > 0.05 * max(mean_count, 1.0):
        warnings.append(
            f"the block standard error is {100 * error / max(mean_count, 1):.0f} % "
            "of the mean: this run is too short to quote.")
    drift = block_means[-1] - block_means[0]
    if abs(drift) > 3.0 * error and abs(drift) > 0.02 * max(mean_count, 1.0):
        warnings.append(
            f"the loading drifted by {drift:+.1f} molecules from the first "
            "production block to the last, so equilibration had not "
            "finished when averaging began.")

    return GCMCResult(
        temperature=temperature,
        pressure_mpa=pressure_mpa,
        mean_count=mean_count,
        count_error=error,
        absolute_density_g_cm3=absolute,
        excess_density_g_cm3=absolute - bulk * accessible,
        weight_percent=100.0 * hydrogen_g / (hydrogen_g + solid.carbon_mass_g),
        bulk_density_g_cm3=bulk,
        carbon_mass_g=solid.carbon_mass_g,
        void_fraction=accessible,
        snapshots=kept,
        acceptance={k: (accepted[k] / tried[k] if tried[k] else 0.0)
                    for k in accepted},
        equilibration_steps=equilibration,
        production_steps=production,
        warnings=warnings,
    )
