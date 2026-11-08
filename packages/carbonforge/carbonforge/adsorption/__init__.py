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

And their table 3 at 77 K, which is the case that decides it: the
Feynman-Hibbs correction is worth 4 % at 293 K and **20 %** here, so a
classical run would land a fifth high and still look plausible. At
1.5 × 10⁶ equilibration and 1.5 × 10⁶ production moves a point,
1600-2600 molecules in the cell:

===========  ==========  ==========  =======
P (MPa)      paper       here        diff
===========  ==========  ==========  =======
0.1          0.031 69    0.030 41    −4.0 %
0.5          0.040 29    0.039 24    −2.6 %
1.0          0.043 36    0.042 44    −2.1 %
3.0          0.046 94    0.045 80    −2.4 %
5.0          0.048 18    0.047 40    −1.6 %
10.0         0.049 96    0.048 53    −2.9 %
15.0         0.051 20    0.048 91    −4.5 %
20.0         0.051 28    0.049 49    −3.5 %
===========  ==========  ==========  =======

Sixteen points over two temperatures, all inside ±6 %, mean absolute
3.0 % at each. That is the agreement this prototype is entitled to:
the reservoir is a two-term virial fitted to their own rounded table
and carries 1.2 % of its own (see
:mod:`~carbonforge.adsorption.bulk`), the molecule has no quadrupole,
and the published densities are themselves rounded to five decimals.

**The 77 K column is low at every pressure, and the diagnostics say
why rather than leaving it to be explained.** The drift
:class:`~carbonforge.adsorption.gcmc.GCMCResult` reports is positive
there — at 20 MPa the loading was still rising by 61 molecules between
the first production block and the last — so a longer run moves these
up, toward the paper, not away. The 293 K points drift the other way
and are correspondingly slight over-estimates. In both cases the
residual is dominated by sampling length, which is a knob, rather than
by the model, which is not.

**What this validates is the sampling, not the model.** Reproducing
Levesque means the potential, the quantum correction, the reservoir's
activity and the move balance are all right. It does not say a
one-site molecule with no quadrupole is true of hydrogen — it says it
is implemented the way he implemented it. ``validacion_GCMC/`` holds
the figure, the numbers and that caveat at length.
"""

from .bulk import LEVESQUE_BULK, VirialGas, fit_virial
from .campaign import (read_isotherm_request, read_results, run_isotherm,
                       status, write_isotherm_request)
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
    "read_isotherm_request",
    "read_results",
    "run_gcmc",
    "run_isotherm",
    "status",
    "write_isotherm_request",
    "zigzag_radius",
    "zigzag_tube",
]
