"""X-ray powder diffraction: patterns, phase identification and Rietveld.

The design principle is the one the Raman side already follows: nothing is
computed from a table of "expected peak positions". A phase is a *crystal
structure*, and every peak position and intensity is calculated from it, so
that changing a lattice parameter changes the pattern the way it physically
would. Reference structures are CIF files — the format the Crystallography
Open Database distributes — so adding a phase means dropping its CIF into a
directory, not editing code.

Modules
-------
``symmetry``   symmetry operations, parsing and group closure
``structure``  the ``Crystal`` object: lattice, atoms, d-spacings
``cif``        CIF reader (and writer for the bundled reference set)
``scattering`` atomic form factors
``powder``     structure factors and simulated powder patterns
``pattern``    the measured ``Pattern`` object
``io``         readers for the usual diffractometer exports
``search``     peak finding and phase identification
``rietveld``   full-pattern refinement, manual and automatic
``reference``  the bundled reference library and user CIF directories
"""

from __future__ import annotations

__all__ = [
    "Crystal",
    "Pattern",
    "identify_phases",
    "read_cif",
    "read_pattern",
    "refine",
    "simulate",
]


def __getattr__(name: str):  # pragma: no cover - lazy re-export
    if name == "Crystal":
        from .structure import Crystal

        return Crystal
    if name == "Pattern":
        from .pattern import Pattern

        return Pattern
    if name == "read_cif":
        from .cif import read_cif

        return read_cif
    if name == "read_pattern":
        from .io import read_pattern

        return read_pattern
    if name == "simulate":
        from .powder import simulate

        return simulate
    if name == "identify_phases":
        from .search import identify_phases

        return identify_phases
    if name == "refine":
        from .rietveld import refine

        return refine
    raise AttributeError(name)
