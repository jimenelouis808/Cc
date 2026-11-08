"""Gas adsorption in a rigid carbon framework.

Three steps, in this order, and the order is the point:

1. :func:`~carbonforge.adsorption.porosity.analyse_porosity` — the
   geometry alone. Seconds, no simulation, and it says whether the
   framework admits the molecule at all and whether that molecule can
   cross the cell. A structure that fails here will adsorb nothing and
   there is no reason to spend hours finding that out.
2. :func:`~carbonforge.adsorption.bulk.fit_virial` — the reservoir. A
   grand-canonical run needs a fugacity, and hydrogen is 13 % off ideal
   at 20 MPa in one direction at 293 K and the other at 77 K.
3. :func:`~carbonforge.adsorption.gcmc.run_gcmc` — the isotherm.

The whole module is pointed at one closed case first:
:func:`~carbonforge.adsorption.bundle.levesque_bundle` rebuilds the
adsorbent of Levesque et al., J. Phys.: Condens. Matter 14 (2002) 9285,
whose published isotherms and bulk densities are the thing to reproduce
before any of this is believed on a new structure. Kowalczyk, Holyst,
Terrones and Terrones (Phys. Chem. Chem. Phys. 9 (2007) 1786) carry the
same method onto the schwarzites and report 3 wt% for the primitive,
diamond and gyroid surfaces at 3.8 MPa and 77 K, which is the next
target after that.

**What the reproduction gave.** Their table 2 (D = 13.3 Å, d = 6 Å,
open tubes, 293 K) against this code at 1.5 × 10⁵ equilibration and
3 × 10⁵ production moves, the absolute density ``rho_o`` in g cm⁻³:

===========  ==========  ==========  =======
P (MPa)      paper       here        diff
===========  ==========  ==========  =======
0.1          0.000 14    0.000 15    +3.7 %
0.5          0.000 73    0.000 72    −1.7 %
1.0          0.001 45    0.001 40    −3.5 %
3.0          0.003 99    0.003 94    −1.3 %
5.0          0.006 14    0.006 12    −0.3 %
10.0         0.010 40    0.010 67    +2.6 %
15.0         0.013 80    0.014 46    +4.8 %
20.0         0.016 47    0.017 42    +5.8 %
===========  ==========  ==========  =======

Eight points inside ±6 %, which is the agreement this prototype is
entitled to: the reservoir is a two-term virial fitted to their own
rounded table and carries 1.2 % of its own (see
:mod:`~carbonforge.adsorption.bulk`), the molecule has no quadrupole,
and the published densities are themselves rounded to five decimals.
The drift that :class:`~carbonforge.adsorption.gcmc.GCMCResult` reports
is −1 % to −2 % of the loading at the upper pressures, so the numbers
here are if anything slight over-estimates of a longer run's.
"""

from .bulk import LEVESQUE_BULK, VirialGas, fit_virial
from .bundle import (LEVESQUE_BUNDLE_ISOTHERMS, levesque_bundle,
                     zigzag_radius, zigzag_tube)
from .gcmc import GCMCResult, run_gcmc
from .porosity import PorosityReport, analyse_porosity
from .potentials import H2_C, H2_H2, feynman_hibbs, quantum_fraction

__all__ = [
    "GCMCResult",
    "LEVESQUE_BULK",
    "LEVESQUE_BUNDLE_ISOTHERMS",
    "H2_C",
    "H2_H2",
    "PorosityReport",
    "VirialGas",
    "analyse_porosity",
    "feynman_hibbs",
    "fit_virial",
    "levesque_bundle",
    "quantum_fraction",
    "run_gcmc",
    "zigzag_radius",
    "zigzag_tube",
]
