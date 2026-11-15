"""The zero-coverage limit: what fills first, and at what pressure.

An isotherm answers "how much", and a folder of them costs days. The
question this module answers is the one in front of it -- **at what
pressure does a framework fill** -- and that is set by the initial
slope of the isotherm, which is a Widom average over an empty cell and
costs seconds.

The physics, with nothing fitted. At low enough pressure the adsorbed
molecules do not see each other, so the loading is one molecule's
Boltzmann average over the cell:

    <N> = a(P) * V * <exp(-beta U)>

where ``a(P)`` is the reservoir's activity ``beta f`` and ``U`` the
framework potential at a point. That activity is
:meth:`~carbonforge.adsorption.bulk.VirialGas.activity`, the same
quantity the grand canonical run uses and the only place a reservoir
enters either -- so this is the low-pressure limit **of that
simulation**, not a second model of the same thing. Likewise the
potential is the Feynman-Hibbs-corrected H2-C pair the run uses: at
77 K the correction is about a fifth of the well, and a classical
Henry constant would be too large by about that much.

Three numbers come out of the one sample, and they answer different
questions:

* `henry_coefficient`, mol/kg/MPa -- the initial slope. Twice the
  slope is twice the uptake at any pressure low enough to be linear,
  so this alone ranks adsorbents at low pressure.
* `isosteric_heat`, kJ/mol -- the binding energy a molecule feels in
  an empty framework, ``-<U exp(-beta U)>/<exp(-beta U)> + RT``. It is
  what a calorimeter measures extrapolated to zero coverage, and it is
  the number a design is judged by: hydrogen physisorbed on carbon
  sits at a few kJ/mol, because that is what a van der Waals well on
  carbon is worth. (A slit of this code's own measuring reads 5.4,
  against 4.4 for the ~450 K single-wall well plus RT -- consistent,
  and both far below the tens of kJ/mol an open metal site in a
  framework reaches. Raising it is a chemistry problem, not a
  geometry one, which is the honest limit of everything measured in
  this folder.)
* `half_filling_pressure`, MPa -- where a Langmuir isotherm of the
  same slope and the same saturation would be half full. **That is a
  model**, stated as one: it needs a saturation loading from somewhere
  else, and a real isotherm is not Langmuir. It is here because it is
  the number a designer actually wants, and refusing to produce it
  does not make anyone's estimate better.

What this does NOT answer: anything at a loading where molecules
interact, which is everything above a few per cent of saturation. The
slope says which framework fills first; only an isotherm says which
holds more.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from scipy.spatial import cKDTree

from .bulk import VirialGas, fit_virial
from .gcmc import AVOGADRO, CUTOFF
from .porosity import _edges, _framework_energy
from .potentials import MASS_C, MASS_H2

#: Gas constant in kJ/(mol K), for the isosteric heat.
GAS_CONSTANT_KJ: float = 8.314462618e-3

#: Above this relative standard error the average is not converged and
#: the report says so. `exp(-beta U)` is dominated by the deepest few
#: points, so a cell that is mostly vacuum needs many samples.
ERROR_LIMIT: float = 0.05

#: A heat this high at zero coverage is not a van der Waals well on
#: carbon at all, whatever the geometry, so it means the sample is
#: wrong rather than the structure remarkable -- most often a cell
#: whose atoms overlap, where the probe sits inside a wall. It is a
#: sanity bound set an order of magnitude clear of anything carbon
#: does, not a threshold anybody should tune against.
IMPLAUSIBLE_HEAT_KJ: float = 40.0


@dataclass
class HenryReport:
    """The zero-coverage limit of one framework at one temperature.

    Attributes
    ----------
    temperature
        In kelvin.
    henry_coefficient
        Initial slope of the absolute isotherm, mol/(kg MPa).
    henry_error
        Its standard error, same units, from the Widom average's own
        spread. Sampling only: it says nothing about the potential.
    isosteric_heat
        At zero coverage, kJ/mol, positive for an attractive framework.
    boltzmann_average
        ``<exp(-beta U)>`` itself, dimensionless. Reported because it
        is the measured quantity and everything else is derived from
        it.
    mean_energy
        The Boltzmann-weighted mean of ``U``, in kelvin -- the well a
        molecule actually sits in, as against the unweighted mean over
        the volume that `PorosityReport.mean_energy` gives.
    samples
        Insertion attempts.
    warnings
        Everything the caller should know before quoting a number.
    """

    temperature: float
    henry_coefficient: float
    henry_error: float
    isosteric_heat: float
    boltzmann_average: float
    mean_energy: float
    samples: int
    warnings: list[str] = field(default_factory=list)

    def loading_at(self, pressure_mpa: float) -> float:
        """Absolute loading in mol/kg, in the linear limit.

        Linear by construction, so it over-predicts as soon as the
        framework starts to fill: it is the tangent at the origin, not
        an isotherm. Use it to compare frameworks at a pressure where
        both are nearly empty, and an isotherm above that.
        """
        return self.henry_coefficient * pressure_mpa

    def weight_percent_at(self, pressure_mpa: float) -> float:
        """The same, as ``100 m(H2)/(m(H2)+m(C))``.

        Total mass, which is the convention
        :class:`~carbonforge.adsorption.gcmc.GCMCResult` quotes and the
        one the experimental literature means.
        """
        hydrogen_per_kg = self.loading_at(pressure_mpa) * MASS_H2 * 1e-3
        return 100.0 * hydrogen_per_kg / (1.0 + hydrogen_per_kg)

    def half_filling_pressure(self, saturation_mol_kg: float) -> float:
        """Langmuir half-filling pressure in MPa, given a saturation.

        ``n = n_sat K P / (1 + K P)`` has initial slope ``n_sat K``, so
        ``K = slope / n_sat`` and half filling is at ``1/K``. The
        Langmuir form is an assumption and a real isotherm is not one;
        this is an estimate to rank designs by, not a prediction.
        """
        if saturation_mol_kg <= 0.0 or self.henry_coefficient <= 0.0:
            raise ValueError(
                "a half-filling pressure needs a positive saturation "
                "loading and a positive Henry coefficient")
        return saturation_mol_kg / self.henry_coefficient

    def describe(self) -> str:
        """One line, with the units spelled out."""
        return (f"{self.temperature:6.1f} K  "
                f"K_H = {self.henry_coefficient:9.3f} "
                f"+- {self.henry_error:.3f} mol/kg/MPa  "
                f"q_st = {self.isosteric_heat:5.2f} kJ/mol  "
                f"<exp(-bU)> = {self.boltzmann_average:.4g}")


def henry_constant(
    atoms: Atoms,
    temperature: float = 77.0,
    samples: int = 200_000,
    cutoff: float = CUTOFF,
    gas: VirialGas | None = None,
    seed: int = 0,
) -> HenryReport:
    """Measure the zero-coverage limit by Widom insertion.

    Parameters
    ----------
    atoms
        The framework, in an orthorhombic periodic cell. Triclinic is
        refused for the reason every neighbour search here refuses it.
    temperature
        In kelvin. It enters twice -- the Boltzmann factor and the
        Feynman-Hibbs correction to the potential -- so a result cached
        at one temperature is wrong at another.
    samples
        Insertion attempts, uniform over the cell. The error falls as
        ``1/sqrt(samples)``, but slowly in a cell that is mostly
        vacuum: `exp(-beta U)` is dominated by the deepest points, and
        the report warns when its own error says so.
    cutoff
        Pair cutoff in Å. Must be at most half the shortest cell edge,
        for the reason the GCMC refuses it otherwise: above that the
        minimum image is not the only image inside the cutoff.
    gas
        The reservoir. Defaults to the virial fitted at `temperature`;
        it is used only to convert a slope into pressure units, and the
        Henry coefficient is a zero-pressure limit where fugacity and
        pressure coincide.
    seed
        For the insertion points.

    Returns
    -------
    HenryReport
        With its warnings. The function does not raise on a poorly
        converged average -- it reports it, as everything else here
        does.
    """
    edges = _edges(atoms)
    if np.min(edges) < 2.0 * cutoff:
        raise ValueError(
            f"the shortest cell edge is {np.min(edges):.1f} A against a "
            f"{cutoff:.1f} A cutoff. Below twice the cutoff the minimum "
            "image is not the only image within range, so the energy is "
            "wrong on the steep part of the wall -- which is exactly "
            "where an insertion is decided. Replicate the cell.")
    if not len(atoms):
        raise ValueError("an empty cell has no framework to adsorb on")

    warnings: list[str] = []
    gas = gas if gas is not None else fit_virial(temperature)

    rng = np.random.default_rng(seed)
    points = rng.random((samples, 3)) * edges
    tree = cKDTree(np.mod(atoms.get_positions(), edges), boxsize=edges)

    # The same potential, with the same quantum correction, that the
    # Monte Carlo would see. Overlaps come back as inf and weigh zero.
    energy = _framework_energy(points, tree, edges, temperature,
                               cutoff=cutoff)
    beta = 1.0 / temperature
    weight = np.exp(-beta * np.clip(energy, -1e4, 1e4))
    weight[~np.isfinite(energy)] = 0.0

    average = float(weight.mean())
    spread = float(weight.std(ddof=1)) / np.sqrt(samples) if samples > 1 else 0.0

    # <U> weighted by the same factor: the well a molecule sits in.
    finite = np.isfinite(energy)
    weighted = float(np.sum(weight[finite] * energy[finite]))
    mean_energy = weighted / (average * samples) if average > 0.0 else 0.0

    # Loading per unit pressure. <N> = a(P) V <exp(-bU)> with the
    # activity in inverse cubic angstrom, exactly as the GCMC uses it,
    # so this is that simulation's own low-pressure limit.
    volume = float(np.prod(edges))
    carbon_kg = len(atoms) * MASS_C / AVOGADRO * 1e-3
    # One MPa, then divided out: the activity is linear in fugacity and
    # fugacity is linear in pressure as pressure goes to zero, so the
    # ratio is the limit and not a finite difference.
    reference_mpa = 1e-4
    molecules = gas.activity(reference_mpa) * 1e-30 * volume * average
    coefficient = molecules / AVOGADRO / carbon_kg / reference_mpa
    error = coefficient * (spread / average) if average > 0.0 else 0.0

    # q_st = -<U>_Boltzmann + RT, in kJ/mol. The kelvin-to-kJ/mol
    # conversion is the gas constant, the energies being per molecule
    # in units of k_B.
    heat = (-mean_energy + temperature) * GAS_CONSTANT_KJ

    if average <= 0.0:
        warnings.append(
            "every insertion overlapped the framework: this cell is "
            "closed to hydrogen, and its Henry coefficient is zero "
            "rather than small.")
    elif spread > ERROR_LIMIT * average:
        warnings.append(
            f"the Widom average's own standard error is "
            f"{100 * spread / average:.0f} % of it; raise `samples`. "
            "`exp(-beta U)` is dominated by the deepest points, so a "
            "cell that is mostly vacuum converges slowly.")
    if heat > IMPLAUSIBLE_HEAT_KJ:
        warnings.append(
            f"a zero-coverage heat of {heat:.0f} kJ/mol is an order of "
            "magnitude past any van der Waals well on carbon, so the "
            "sample is wrong rather than the structure remarkable. "
            "Check that the cell is the one intended and that its "
            "atoms do not overlap -- a probe inside a wall reads like "
            "this.")
    if mean_energy >= 0.0 and average > 0.0:
        warnings.append(
            "the Boltzmann-weighted mean energy is not negative, so "
            "this framework does not bind hydrogen at all at this "
            "temperature.")

    return HenryReport(
        temperature=float(temperature),
        henry_coefficient=float(coefficient),
        henry_error=float(error),
        isosteric_heat=float(heat),
        boltzmann_average=average,
        mean_energy=float(mean_energy),
        samples=int(samples),
        warnings=warnings,
    )
