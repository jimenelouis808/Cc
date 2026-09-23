"""vibspec: IR (and later Raman) spectra of functionalised graphene nanoribbons.

The aim is band assignment: compute the IR spectrum of a nanoribbon carrying
one well-defined functionality (graphitic, pyridinic or pyrrolic N, amine,
nitrile, N-oxide, and the oxygen groups) and compare it with an experimental
FTIR spectrum of functionalised nanotubes.

Models are **finite**, hydrogen-terminated ribbons: the IR intensities come
from finite differences of the dipole moment, which a periodic model does not
have. The flake itself comes from
:func:`carbonforge.builders.build_finite_nanoribbon`; the chemistry from
:mod:`carbonforge.functionalization`, which this subpackage only arranges
into reproducible presets.

Layout
------
:mod:`~carbonforge.vibspec.core`
    Everything that can run from a script or a batch job: presets, site
    selection, physical checks, calculation settings, the prepare / run /
    collect workflow and the on-disk records. No GUI imports, ever, and GPAW
    only inside a running calculation.

Raman is not computed yet. It will reuse the relaxed structure, the record
and :class:`~carbonforge.results.spectra.VibrationalSpectrum` (which already
carries Raman activities); Quantum ESPRESSO's DFPT Raman is already written by
:mod:`carbonforge.exports.qe`.
"""

from .core import PRESETS, apply_preset, check_structure, suggest_spin

__all__ = ["PRESETS", "apply_preset", "check_structure", "suggest_spin"]
