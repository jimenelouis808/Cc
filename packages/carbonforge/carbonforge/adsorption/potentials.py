"""Hydrogen in a carbon framework: the pair potentials, with the quantum fix.

The model is the one Levesque, Gicquel, Lamari Darkrim and Beyaz Kayiran
use in *J. Phys.: Condens. Matter* **14** (2002) 9285, and that Kowalczyk,
Holyst, Terrones and Terrones carry over to the schwarzites in *Phys.
Chem. Chem. Phys.* **9** (2007) 1786. Both are in this repository's
reading for a reason: between them they fix every parameter here and
leave numbers to reproduce.

**H2 is one site.** A single Lennard-Jones centre at the molecular
centre of mass. Levesque adds a quadrupole as three point charges
(``q = 0.4829 e`` on the protons, ``-2q`` at the centre) and that is
**not implemented here** — see :func:`lennard_jones` for what the
omission costs and when it starts to matter.

**The quantum correction is not optional at 77 K.** Hydrogen is light
enough that its de Broglie wavelength is a real fraction of σ, and
treating it classically over-counts how close two molecules get. The
rigorous route is path-integral Monte Carlo; the Feynman-Hibbs effective
potential is the cheap one, exact to order ħ², and it is what both
papers use:

    v_FH(r) = v(r) + ħ²/(24 μ k_B T) · [v''(r) + (2/r) v'(r)]

For a Lennard-Jones ``v`` that bracket has a closed form, so this costs
two more power terms and no derivatives at runtime:

    v'' + 2v'/r = 4ε [ 132 σ¹²/r¹⁴ − 30 σ⁶/r⁸ ]

Levesque gives the size of what it buys: the quantum effects reduce the
adsorbed amount by about **4 % at 293 K and 20 % at 77 K**. A 77 K
isotherm computed classically is therefore not slightly off, it is a
fifth too high.

**Units.** Energies in kelvin (ε/k_B), lengths in ångström, which is how
both papers quote their parameters and what keeps β = 1/T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: ``ħ² / k_B`` in the units this module works in — amu Å² K — so the
#: Feynman-Hibbs prefactor needs no unit bookkeeping at the call site.
#:
#: Worked through rather than quoted, because the first version of this
#: constant was wrong by a factor of three and a quantum correction
#: three times too small is invisible in every output it touches:
#:
#:     ħ²      = (1.054571817e-34 J s)² = 1.1121217e-68 J² s²
#:     ħ²/k_B  = 1.1121217e-68 / 1.380649e-23 = 8.055095e-46 kg m² K
#:     in amu  = × 6.0221408e26
#:     in Å²   = × 1e20
#:             = 48.5089 amu Å² K
#:
#: What caught it was :func:`quantum_fraction` against the number
#: Levesque prints — "≈ 0.07 at T = 70 K" — which the wrong constant
#: returned as 0.026. That check is a test.
HBAR_SQUARED_OVER_KB: float = 48.5089

#: Mass of H2 and of C, in atomic mass units.
MASS_H2: float = 2.01588
MASS_C: float = 12.011


@dataclass(frozen=True)
class PairParameters:
    """A Lennard-Jones pair, and the reduced mass its FH term uses.

    Parameters
    ----------
    epsilon
        Well depth as ε/k_B, in kelvin.
    sigma
        Collision diameter in ångström.
    reduced_mass
        Reduced mass of the pair in amu, for the Feynman-Hibbs term. For
        a fluid molecule against a **fixed** framework atom this is the
        molecule's own mass, not the two-body reduced mass: the wall
        does not recoil.
    source
        Where the numbers come from. Carried so a reproduced isotherm
        can say which parameter set produced it.
    """

    epsilon: float
    sigma: float
    reduced_mass: float
    source: str = ""


#: H2-H2, Levesque 2002 §2: ε/k = 36.7 K, σ = 2.958 Å. The reduced mass
#: of a pair of H2 molecules is half of one, which is what Levesque
#: states ("m_r: reduced mass of a pair of hydrogen molecules").
H2_H2 = PairParameters(
    epsilon=36.7, sigma=2.958, reduced_mass=MASS_H2 / 2.0,
    source="Levesque et al., J. Phys.: Condens. Matter 14 (2002) 9285, §2",
)

#: H2-C, same source: ε/k = 32.05 K, σ = 3.18 Å. The framework is rigid,
#: so the reduced mass is the molecule's.
H2_C = PairParameters(
    epsilon=32.05, sigma=3.18, reduced_mass=MASS_H2,
    source="Levesque et al., J. Phys.: Condens. Matter 14 (2002) 9285, §2",
)


def lennard_jones(r: np.ndarray, pair: PairParameters) -> np.ndarray:
    """Classical Lennard-Jones energy in kelvin.

    No quadrupole. H2's quadrupole moment is small and its effect on a
    bulk isotherm is a few per cent at room temperature, but it grows as
    the temperature falls and the molecules order: if a 77 K isotherm
    computed here comes out low against the reference, the quadrupole is
    the first thing to add, not the last.

    Parameters
    ----------
    r
        Distances in ångström. Values at or below zero are not guarded
        here; the callers clamp.
    pair
        The parameters to use.
    """
    sr6 = (pair.sigma / r) ** 6
    return 4.0 * pair.epsilon * (sr6 * sr6 - sr6)


def feynman_hibbs(
    r: np.ndarray, pair: PairParameters, temperature: float
) -> np.ndarray:
    """Lennard-Jones plus the ħ² quantum correction, in kelvin.

    The correction is **temperature dependent**, which is the thing to
    keep in mind when caching: an energy grid built at 293 K is wrong at
    77 K by more than the temperature change alone, because the
    potential itself changed. :class:`~carbonforge.adsorption.gcmc.Grid`
    records the temperature it was built at for that reason.

    Parameters
    ----------
    r
        Distances in ångström.
    pair
        Lennard-Jones parameters and the reduced mass of the pair.
    temperature
        In kelvin. The correction scales as 1/T, so it vanishes in the
        classical limit, as it should.

    Returns
    -------
    numpy.ndarray
        Energy in kelvin, same shape as ``r``.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    sr2 = (pair.sigma / r) ** 2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    classical = 4.0 * pair.epsilon * (sr12 - sr6)
    prefactor = HBAR_SQUARED_OVER_KB / (
        24.0 * pair.reduced_mass * temperature)
    bracket = 4.0 * pair.epsilon * (132.0 * sr12 - 30.0 * sr6) / (r * r)
    return classical + prefactor * bracket


def quantum_fraction(temperature: float, pair: PairParameters = H2_H2) -> float:
    """Levesque's dimensionless measure of how quantum the fluid is.

    ``ħ²/(m_r σ² k_B T)``, which he quotes as about 0.07 at 70 K. Worth
    printing beside any isotherm: it is the one number that says whether
    the Feynman-Hibbs correction is a tweak or the difference between a
    result and a wrong result.
    """
    return HBAR_SQUARED_OVER_KB / (
        pair.reduced_mass * pair.sigma ** 2 * temperature)
