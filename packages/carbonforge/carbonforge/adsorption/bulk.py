"""The bulk hydrogen a grand-canonical simulation is in equilibrium with.

A GCMC run needs the **fugacity** of the reservoir, not its pressure.
Hydrogen at 20 MPa is 13 % off ideal at 293 K and 13 % off the other way
at 77 K, so using the pressure in place of the fugacity shifts the whole
isotherm by about that much — comfortably more than the differences
between the materials anyone runs this to compare.

**Where the numbers come from, and why not a reference equation of
state.** The point of this module is to put the simulation in
equilibrium with *the same reservoir the reference calculation used*. A
better equation of state than theirs would make the comparison worse,
not better. So the bulk is defined by a table of measured densities —
by default the one Levesque *et al.* publish alongside the isotherms
this package reproduces — and a two-term virial is fitted to it:

    Z = 1 + B/v + C/v²

**Fitted, and weighted, because their table is rounded.** The published
densities carry five decimals in g cm⁻³, which at the lowest pressure is
*one significant figure*: 0.000 08 g cm⁻³ at 0.1 MPa and 293 K is
±6 %. An unweighted fit lets those points dominate (they have the
largest 1/v leverage) and returns B = 21.6 cm³ mol⁻¹ with a negative
third virial coefficient, which hydrogen does not have at 293 K.
Weighting each point by its own rounding gives B = 17.6 and C = +98,
and reproduces the table to 1.2 % on average — which is the table's own
precision. That is the honest accuracy of this module and it is the
reason a production study should fit to a reference equation of state
(Leachman, or REFPROP) instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Gas constant in J/(mol K) and the molar mass of H2 in kg/mol.
GAS_CONSTANT: float = 8.314462618
MOLAR_MASS_H2: float = 2.01588e-3

#: Half the last printed digit of the reference densities, in g/cm3.
#: The published tables give five decimals, so this is the rounding each
#: entry carries and the weight each one deserves.
REFERENCE_ROUNDING: float = 0.000005

#: Bulk hydrogen density in g/cm3 against pressure in MPa, from the
#: tables of Levesque et al., J. Phys.: Condens. Matter 14 (2002) 9285
#: (their column ``rho_b``, tables 1-3). These are the reservoir states
#: of the isotherms this package reproduces.
LEVESQUE_BULK: dict[float, list[tuple[float, float]]] = {
    293.0: [(0.1, 0.00008), (0.5, 0.00040), (1.0, 0.00080), (3.0, 0.00242),
            (5.0, 0.00398), (10.0, 0.00774), (15.0, 0.01128), (20.0, 0.01461)],
    150.0: [(0.1, 0.00016), (0.5, 0.00079), (1.0, 0.00160), (3.0, 0.00474),
            (5.0, 0.00782), (10.0, 0.01513), (15.0, 0.02179), (20.0, 0.02782)],
    77.0: [(0.1, 0.00031), (0.5, 0.00158), (1.0, 0.00320), (3.0, 0.01013),
           (5.0, 0.01758), (10.0, 0.03641), (15.0, 0.05013), (20.0, 0.05911)],
}


@dataclass(frozen=True)
class VirialGas:
    """A two-term virial equation of state at one temperature.

    Attributes
    ----------
    temperature
        In kelvin.
    b, c
        Second and third virial coefficients in m³/mol and m⁶/mol².
    residual_pct
        Mean absolute error with which this reproduces the densities it
        was fitted to. Carried because it is the accuracy of everything
        downstream, and because a fit whose residual is far above the
        reference's own rounding is a warning that two virial terms do
        not reach that density.
    source
        Where the reference densities came from.
    """

    temperature: float
    b: float
    c: float
    residual_pct: float
    source: str = ""

    def compressibility(self, density_mol_m3: np.ndarray) -> np.ndarray:
        """``Z = 1 + B/v + C/v²`` from the molar density."""
        v = 1.0 / np.asarray(density_mol_m3, dtype=float)
        return 1.0 + self.b / v + self.c / v**2

    def density(self, pressure_mpa: float) -> float:
        """Molar density (mol/m³) at ``pressure_mpa``, solved iteratively.

        The virial is explicit in volume, so the density at a given
        pressure comes from a fixed-point iteration that converges in a
        few steps at these densities.
        """
        pressure = pressure_mpa * 1e6
        rt = GAS_CONSTANT * self.temperature
        v = rt / pressure
        for _ in range(200):
            new = rt * (1.0 + self.b / v + self.c / v**2) / pressure
            if abs(new - v) < 1e-14 * abs(new):
                v = new
                break
            v = new
        return 1.0 / v

    def mass_density(self, pressure_mpa: float) -> float:
        """Bulk density in g/cm³, for comparison with the reference table."""
        return self.density(pressure_mpa) * MOLAR_MASS_H2 * 1e-3

    def fugacity(self, pressure_mpa: float) -> float:
        """Fugacity in Pa.

        From the same virial, so the reservoir's density and its
        fugacity cannot disagree:

            ln φ = 2B/v + (3C − 2B²) / (2v²)
        """
        v = 1.0 / self.density(pressure_mpa)
        ln_phi = 2.0 * self.b / v + (3.0 * self.c - 2.0 * self.b**2) / (
            2.0 * v**2)
        return pressure_mpa * 1e6 * float(np.exp(ln_phi))

    def activity(self, pressure_mpa: float) -> float:
        """``βf`` in m⁻³: what a grand-canonical insertion is weighted by.

        The de Broglie wavelength cancels between insertion and deletion
        in the standard formulation, so this — fugacity over kT — is the
        only reservoir quantity the simulation needs.
        """
        return self.fugacity(pressure_mpa) / (
            1.380649e-23 * self.temperature)


def fit_virial(
    temperature: float,
    reference: list[tuple[float, float]] | None = None,
    rounding: float = REFERENCE_ROUNDING,
    source: str = "Levesque et al. 2002, tables 1-3",
) -> VirialGas:
    """Fit ``Z = 1 + B/v + C/v²`` to measured bulk densities.

    Parameters
    ----------
    temperature
        In kelvin. With ``reference`` omitted it must be one of the
        temperatures in :data:`LEVESQUE_BULK`; there is no interpolation
        between them, because a virial coefficient interpolated from
        three temperatures is a guess wearing a number's clothes.
    reference
        ``[(pressure_MPa, density_g_cm3), ...]`` to fit. Defaults to the
        published table at that temperature.
    rounding
        Half the last printed digit of those densities, in g/cm³. Each
        point is weighted by its own precision; see the module
        docstring for what happens without it.
    source
        Provenance, carried into the result.

    Returns
    -------
    VirialGas
        The fitted equation of state and the error it reproduces its
        own reference with.

    Raises
    ------
    KeyError
        If no reference is given and the temperature is not tabulated.
    ValueError
        If fewer than three points are given — two coefficients fitted
        to two points is an interpolation, not a fit, and its residual
        would be zero and meaningless.
    """
    rows = reference if reference is not None else LEVESQUE_BULK[temperature]
    if len(rows) < 3:
        raise ValueError(
            "fitting two virial coefficients needs at least three reference "
            f"points; {len(rows)} given. With two the residual is zero by "
            "construction and says nothing about the fit."
        )
    pressure = np.array([p for p, _ in rows], dtype=float) * 1e6
    density = np.array([d for _, d in rows], dtype=float)
    molar_volume = MOLAR_MASS_H2 / (density * 1e3)
    z = pressure * molar_volume / (GAS_CONSTANT * temperature)

    weights = density / rounding
    design = np.column_stack(
        [1.0 / molar_volume, 1.0 / molar_volume**2]) * weights[:, None]
    (b, c), *_ = np.linalg.lstsq(design, (z - 1.0) * weights, rcond=None)

    predicted = 1.0 + b / molar_volume + c / molar_volume**2
    residual = float(np.mean(np.abs(predicted - z) / z) * 100.0)
    return VirialGas(temperature=float(temperature), b=float(b), c=float(c),
                     residual_pct=residual, source=source)
