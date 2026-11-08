"""Relaxation: a geometry cleaner here, a real potential elsewhere.

Three entry points, in the order anyone needs them:

* :func:`relax_structure` — the **valence force field** (bond, a true
  angle term, non-bonded repulsion; L-BFGS on analytic gradients) run on
  a structure that already exists. This is the one to reach for: it
  fixes geometry on tens of thousands of atoms in seconds, and it cannot
  change topology.
* :func:`harmonic_pre_relax` — bond springs alone, no angles, no
  dependencies beyond NumPy. Kept for a first pass over something
  stochastic, and *only* that: springs with no angle term are what let a
  sheet pyramidalise and fold.
* :func:`relax_with_calculator` — hand it to anything with an ASE
  calculator interface when the geometry is not the question and the
  energy is.

The dividing line is worth keeping in mind. A valence force field has
one rest length and one angle target per coordination, so it says
nothing about which of two structures is more stable. For that, the
geometry goes out to a real potential — see
:func:`nanocarbon_lab.exports.relax_request.write_relaxation_request`.
"""

from .optimize import harmonic_pre_relax, relax_with_calculator
from .valence import ANGLE_FOR_COORDINATION, angle_targets, bonds_from_atoms, relax_structure

__all__ = [
    "ANGLE_FOR_COORDINATION",
    "angle_targets",
    "bonds_from_atoms",
    "harmonic_pre_relax",
    "relax_structure",
    "relax_with_calculator",
]
