"""X-ray photoelectron spectroscopy: reading, fitting and quantifying.

The fifth instrument of the suite, and the one that answers the question
Raman explicitly cannot: **what the nitrogen is bonded to**. Raman sees that
a carbon lattice is disordered; it cannot separate pyridinic from pyrrolic
from graphitic nitrogen, and the Raman side of this package says so in as
many words. XPS separates them, and this module is what turns that into
numbers with their reasons attached.

What is here:

``spectrum``
    :class:`~ramancarbon.xps.spectrum.XPSSpectrum` — the binding-energy
    scale, the photon energy, and the refusal to guess an anode.
``io``
    PHI ``.spe``, VAMAS (ISO 14976) and text, in and out.
``elements``
    Typed access to the lines, Auger groups and chemical states in
    ``database/data/xps.json``.
``background``
    Shirley, Tougaard and linear, with the endpoints treated as the
    parameter they are.
``lineshapes``
    GL products and sums, and Doniach–Šunjić for metals.
``fitting`` / ``presets``
    Constrained fitting with spin–orbit doublets as one component, and
    models built out of published chemical states.
``calibrate`` / ``survey`` / ``quantify``
    Charge referencing, element identification, and atomic per cent.
"""

from __future__ import annotations

from .background import Background, estimate_background
from .elements import XPSDatabase, load_xps_database
from .fitting import XPSComponent, XPSFitResult, XPSModel, fit_region
from .io import read_spe, read_vamas, read_xps, read_xps_text, write_vamas
from .presets import compare_counts, count_model, free_model, state_model
from .spectrum import XPSError, XPSSpectrum, source_energy

__all__ = [
    "Background",
    "XPSComponent",
    "XPSDatabase",
    "XPSError",
    "XPSFitResult",
    "XPSModel",
    "XPSSpectrum",
    "compare_counts",
    "count_model",
    "estimate_background",
    "fit_region",
    "free_model",
    "load_xps_database",
    "read_spe",
    "read_vamas",
    "read_xps",
    "read_xps_text",
    "source_energy",
    "state_model",
    "write_vamas",
]
