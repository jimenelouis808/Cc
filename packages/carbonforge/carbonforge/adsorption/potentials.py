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
    critical_temperature
        The molecule's own critical temperature in kelvin, or 0 when it
        is not known or the pair is a cross pair rather than a fluid.
        It is carried because the adsorption code has to know whether
        the reservoir is a gas at all: below it the fluid condenses,
        a pore fills by capillary condensation rather than adsorption,
        and a zero-coverage limit is a tangent to an isotherm whose
        linear stretch ends at the saturation pressure -- which can be
        many orders of magnitude below anything anybody would run.
        That is why 77 K is the temperature for hydrogen (critical at
        33 K, so comfortably a gas) and is **not** a temperature at
        which argon, nitrogen or methane can be compared with it.
    """

    epsilon: float
    sigma: float
    reduced_mass: float
    source: str = ""
    critical_temperature: float = 0.0


#: H2-H2, Levesque 2002 §2: ε/k = 36.7 K, σ = 2.958 Å. The reduced mass
#: of a pair of H2 molecules is half of one, which is what Levesque
#: states ("m_r: reduced mass of a pair of hydrogen molecules").
H2_H2 = PairParameters(
    epsilon=36.7, sigma=2.958, reduced_mass=MASS_H2 / 2.0,
    source="Levesque et al., J. Phys.: Condens. Matter 14 (2002) 9285, §2",
    critical_temperature=32.98,
)

#: H2-C, same source: ε/k = 32.05 K, σ = 3.18 Å. The framework is rigid,
#: so the reduced mass is the molecule's.
H2_C = PairParameters(
    epsilon=32.05, sigma=3.18, reduced_mass=MASS_H2,
    source="Levesque et al., J. Phys.: Condens. Matter 14 (2002) 9285, §2",
)


#: Masses of the other adsorbates offered, in amu.
MASS_AR: float = 39.948
MASS_NE: float = 20.1797
MASS_CH4: float = 16.043
MASS_N2: float = 28.0134


def carbon_self() -> PairParameters:
    """The framework carbon's OWN Lennard-Jones pair, derived not quoted.

    Lorentz-Berthelot says a cross pair is ``sqrt(e1 e2)`` and
    ``(s1 + s2)/2``, so :data:`H2_C` and :data:`H2_H2` between them
    already fix carbon: ``e_C = e_H2C^2 / e_H2`` and
    ``s_C = 2 s_H2C - s_H2``. That returns 27.99 K and 3.402 A, which
    is the graphite parameter set the whole literature uses -- and
    arriving at it this way means a second copy of those numbers
    cannot drift from the two pairs the validated isotherm was
    computed with.

    The reduced mass is carbon's, and is unused: the framework is
    rigid, so every Feynman-Hibbs term here takes the adsorbate's mass
    (the wall does not recoil).
    """
    return PairParameters(
        epsilon=H2_C.epsilon ** 2 / H2_H2.epsilon,
        sigma=2.0 * H2_C.sigma - H2_H2.sigma,
        reduced_mass=MASS_C,
        source="Lorentz-Berthelot, inverted from H2_C and H2_H2",
    )


#: H2's reduced temperature at the 77 K everything in this repository
#: is measured at: 77 / 32.98. Comparing adsorbates means comparing
#: them at the SAME reduced temperature, or the comparison mixes the
#: molecule with the thermodynamic state it is in.
H2_REDUCED_AT_77: float = 77.0 / 32.98


def matched_temperature(adsorbate: PairParameters,
                        reduced: float = H2_REDUCED_AT_77) -> float:
    """The temperature putting `adsorbate` at reduced temperature `reduced`.

    Hydrogen at 77 K sits at ``T/T_c = 2.33``, well clear of
    condensation. Argon, nitrogen and methane at 77 K are all BELOW
    their critical temperatures, so at that temperature they are not
    the same kind of fluid and a Henry coefficient compared across them
    would be reporting the state, not the molecule. This returns the
    temperature at which each is in the state hydrogen is in at 77 K.

    It is the comparison that isolates **size**, which is the design
    question. It is not an application temperature: methane matched to
    hydrogen's 77 K lands at 445 K, and nobody stores methane there.
    """
    if adsorbate.critical_temperature <= 0.0:
        raise ValueError(
            "this pair carries no critical temperature, so there is no "
            "reduced temperature to match. Set `critical_temperature` "
            "on it, or pass the temperature you want directly.")
    return reduced * adsorbate.critical_temperature


def with_carbon(adsorbate: PairParameters) -> PairParameters:
    """The adsorbate-carbon cross pair, by Lorentz-Berthelot.

    The reduced mass is carried straight through from `adsorbate`,
    because against a rigid framework it is the molecule's own mass --
    the wall does not recoil.

    **So the pair handed in must carry the molecular mass, not a
    fluid-fluid reduced mass**, and that is a trap worth stating: a
    self-pair describing two molecules colliding takes half the
    molecular mass, and :data:`H2_H2` is exactly such a pair (Levesque
    defines its ``m_r`` as "the reduced mass of a pair of hydrogen
    molecules"). Handing that one in gives 1.008 amu where the wall
    pair wants 2.016, and the Feynman-Hibbs term goes as ``1/mu``, so
    the correction comes out **twice** what it should -- invisible in
    the shape of every output it touches, which is the same way the
    ``hbar^2/k_B`` constant was once wrong by three.
    :data:`H2_SELF` is the entry to use against a wall, and a test
    pins that ``with_carbon(H2_SELF)`` reproduces :data:`H2_C`
    exactly, masses included.
    """
    carbon = carbon_self()
    return PairParameters(
        epsilon=float(np.sqrt(adsorbate.epsilon * carbon.epsilon)),
        sigma=0.5 * (adsorbate.sigma + carbon.sigma),
        reduced_mass=adsorbate.reduced_mass,
        source=f"Lorentz-Berthelot of ({adsorbate.source}) with carbon",
    )


#: H2 against a WALL: the same well and diameter as :data:`H2_H2` with
#: the molecular mass in place of the fluid-fluid reduced one. See
#: :func:`with_carbon` for why the two cannot be the same object.
H2_SELF = PairParameters(
    epsilon=H2_H2.epsilon, sigma=H2_H2.sigma, reduced_mass=MASS_H2,
    source=H2_H2.source + " (molecular mass, for a rigid wall)",
    critical_temperature=H2_H2.critical_temperature,
)

#: Argon, Ar-Ar. The benchmark adsorbate, and the one case where a
#: single Lennard-Jones centre is not an approximation at all: argon is
#: a sphere. Rowley/standard values, which every simulation of argon on
#: graphite uses.
AR_AR = PairParameters(
    epsilon=119.8, sigma=3.405, reduced_mass=MASS_AR,
    source="standard Ar Lennard-Jones (eps/k 119.8 K, sigma 3.405 A)",
    critical_temperature=150.69,
)

#: Neon, Ne-Ne. Here as the light-but-not-hydrogen case: its well is
#: H2's almost exactly while its mass is ten times larger, so it
#: separates "deep well" from "quantum" in any result that moves.
NE_NE = PairParameters(
    epsilon=36.8, sigma=2.789, reduced_mass=MASS_NE,
    source="standard Ne Lennard-Jones (eps/k 36.8 K, sigma 2.789 A)",
    critical_temperature=44.49,
)

#: Methane, as ONE united-atom site -- the TraPPE-UA value. Methane is
#: nearly spherical and this is the standard coarse-graining of it, so
#: a single centre is a good model rather than a resigned one.
CH4_CH4 = PairParameters(
    epsilon=148.0, sigma=3.73, reduced_mass=MASS_CH4,
    source="TraPPE-UA methane (eps/k 148 K, sigma 3.73 A)",
    critical_temperature=190.56,
)

#: Nitrogen as one site, which is an APPROXIMATION and is offered with
#: that said. N2 is elongated (bond 1.10 A) and carries a quadrupole;
#: a single centre keeps neither, so it misses how the molecule lies
#: against a wall and how it packs in a pore of its own width. Useful
#: for a trend across molecule sizes, not for a number to quote.
N2_N2 = PairParameters(
    epsilon=95.05, sigma=3.698, reduced_mass=MASS_N2,
    source="one-site N2 (eps/k 95.05 K, sigma 3.698 A) -- NO quadrupole, "
           "NO elongation; a trend, not a prediction",
    critical_temperature=126.19,
)

#: The adsorbates this module offers against carbon, by name.
#:
#: **CO2 is deliberately absent.** It is linear and strongly
#: quadrupolar, and its interaction with a carbon wall is dominated by
#: that quadrupole -- a single Lennard-Jones centre does not
#: approximate CO2, it replaces it with a different molecule. Adding a
#: one-site CO2 here would produce numbers that look like the others
#: and mean much less, so it needs a multi-site potential first. Build
#: a :class:`PairParameters` by hand if you want to try one anyway.
ADSORBATES: dict[str, PairParameters] = {
    "H2": H2_SELF,
    "Ne": NE_NE,
    "Ar": AR_AR,
    "CH4": CH4_CH4,
    "N2": N2_N2,
}


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
