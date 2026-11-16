"""Does another molecule suit a superlattice better than hydrogen does?

    uv run python validacion_rendijas/adsorbatos.py <carpeta> [muestras]

`llenado.py` and `llenado_super.py` found the same thing twice: a
superlattice offers deeper sites than a slit and fewer of them per
gram, so an optimally tuned slit wins the initial slope by about 1.75
with hydrogen. The slit that wins is **narrow** -- 6.6 A between wall
centres, a 3.55 A channel, barely over H2's own diameter -- and that is
the whole reason it wins: both walls are within reach at once.

Which suggests the obvious test. A bigger molecule cannot fit in that
channel, so the slit has to open up to hold it, and once it does its
two walls stop acting together. The superlattices' channels are 18-20 A
and do not care. **So a larger adsorbate should close the gap**, and
the question is whether it closes it enough to change the answer.

**Adsorbates must be compared at the same REDUCED temperature, not the
same temperature**, and this is the trap the whole script is arranged
around. 77 K suits hydrogen because its critical temperature is 33 K,
so it is a gas by a factor of 2.3. Argon, nitrogen and methane at 77 K
are all *below* their critical temperatures -- a pore then fills by
capillary condensation rather than adsorption, and a zero-coverage
slope is the tangent to an isotherm whose linear stretch ends at the
saturation pressure, around 1e-5 Pa for methane. Measured at 77 K
anyway, methane reads 1.6e10 mol/kg/MPa against hydrogen's 714: a
number about the state, not the molecule. `matched_temperature` puts
each one where hydrogen is at 77 K (Ne 104 K, N2 295 K, Ar 352 K,
CH4 445 K), so the only thing differing between two columns is the
size of the molecule.

That is deliberately NOT an application temperature -- nobody stores
methane at 445 K. It is the comparison that isolates geometry, which
is the design question being asked. An engineering answer needs each
adsorbate at its own working temperature, which is a different table.

**Every adsorbate here is a single Lennard-Jones centre**, which is
exact for argon, good for methane (the standard united atom) and an
approximation for nitrogen, whose elongation and quadrupole it drops.
CO2 is absent on purpose: a single centre does not approximate it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from ase.io import read

from carbonforge.adsorption.henry import henry_constant
from carbonforge.adsorption.potentials import (
    ADSORBATES,
    matched_temperature,
    with_carbon,
)

from llenado import slit
from llenado_super import embed, soundness

#: Slit wall separations to scan per adsorbate. Wider than `llenado.py`
#: at the top end, because the optimum is expected to move out with the
#: molecule and a scan that stops before it would report its own edge.
SEPARATIONS = (6.0, 6.4, 6.6, 7.0, 7.4, 7.8, 8.2, 8.6, 9.0, 9.5,
               10.0, 11.0, 12.0, 14.0)

#: Every structure in the folder is measured, and which ones are in
#: the folder is the caller's choice. It was a hardcoded list of three
#: meshed nets, which stopped being the right set the moment the
#: crystalline route turned out to beat them: Schwarz D at (18, 5)
#: reads 12861 mol/kg/MPa for H2 against super-graphene's meshed 4057.
#: Point this at a folder holding the structures the question is about.
#:
#: Super-fcc is worth leaving OUT of such a folder: with hydrogen it
#: loses a factor of 7 because twelve tubes at one node leave a 7.3 A
#: channel against the others' 18-20, a larger molecule can only make
#: that worse, and it is the slowest cell here.

#: Insertion attempts and seeds, as in the rest of the folder.
SAMPLES = 200_000
SEEDS = (0, 1)


def henry(atoms, adsorbate, temperature: float,
          samples: int) -> tuple[float, float, float, list[str]]:
    """``(mean K_H, half the seed spread, q_st, warnings)``.

    The warnings come back and are PRINTED rather than dropped, which
    is a lesson paid for: a bug that made the weighted mean energy
    -4.6e9 kJ/mol at 352 K raised exactly the warning it should have
    ("does not bind at all"), and the first version of this script
    looked only for the subcritical one and threw the rest away. A
    report's warnings are part of its result.
    """
    runs = [henry_constant(atoms, temperature=temperature,
                           adsorbate=adsorbate, samples=samples, seed=seed)
            for seed in SEEDS]
    values = [run.henry_coefficient for run in runs]
    warnings = sorted({w for run in runs for w in run.warnings})
    for warning in warnings:
        if "critical temperature" in warning:
            raise AssertionError(
                "this script compares at matched reduced temperature, "
                f"so no adsorbate should be subcritical: {warning}")
    return (float(np.mean(values)),
            0.5 * (max(values) - min(values)),
            runs[0].isosteric_heat,
            warnings)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1])
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else SAMPLES

    sources = sorted(folder.glob("*.extxyz"))
    if not sources:
        print(f"no hay .extxyz en {folder}")
        return 1
    supers = {}
    for source in sources:
        collapsed, angle_sum = soundness(source)
        if collapsed:
            # A pinched wall is a narrow slit wearing another name, so
            # measuring it here would answer a different question with
            # this table's labels.
            print(f"{source.stem}: saltada, pared colapsada "
                  f"(ángulo mínimo {angle_sum:.1f}°)")
            continue
        big, _repeat, _gap = embed(read(str(source)))
        supers[source.stem] = big
    if not supers:
        print("ninguna estructura sana en la carpeta")
        return 1

    print(f"{samples} inserciones x {len(SEEDS)} semillas, cada adsorbato "
          "a la temperatura que lo pone en el estado del H2 a 77 K\n")

    for name, adsorbate in ADSORBATES.items():
        temperature = matched_temperature(adsorbate)
        cross = with_carbon(adsorbate)
        print(f"### {name}  —  T = {temperature:.0f} K "
              f"(T/Tc = {temperature / adsorbate.critical_temperature:.2f}), "
              f"σ con C = {cross.sigma:.3f} Å, ε = {cross.epsilon:.1f} K",
              flush=True)

        best = None
        scan = []
        for separation in SEPARATIONS:
            coefficient, spread, heat, notes = henry(
                slit(separation), adsorbate, temperature, samples)
            for note in notes:
                print(f"  aviso (rendija {separation:.1f}): {note}")
            scan.append((separation, coefficient, spread, heat))
            if best is None or coefficient > best[1]:
                best = scan[-1]
        if best[0] in (SEPARATIONS[0], SEPARATIONS[-1]):
            print(f"  AVISO: el óptimo de la rendija cae en el borde del "
                  f"barrido (H = {best[0]:.1f} Å). El barrido informa de su "
                  "propio borde, no de un máximo.")
        print(f"  rendija óptima: H = {best[0]:.1f} Å, "
              f"K_H = {best[1]:.3g} ± {best[2]:.3g}, q_st = {best[3]:.2f} "
              f"kJ/mol", flush=True)

        for label, atoms in sorted(supers.items()):
            coefficient, spread, heat, notes = henry(
                atoms, adsorbate, temperature, samples)
            for note in notes:
                print(f"  aviso ({label}): {note}")
            ratio = coefficient / best[1]
            print(f"  {label:15s} K_H = {coefficient:.3g} ± {spread:.3g}, "
                  f"q_st = {heat:5.2f}   "
                  f"{ratio:6.2f}× la mejor rendija", flush=True)
        print(flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
