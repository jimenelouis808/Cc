"""ramancarbon — análisis de espectros Raman de nanomateriales de carbono.

Entrada rápida::

    from ramancarbon import read_spectrum, analyse

    spectrum = read_spectrum("muestra.txt", laser_nm=532)
    result = analyse(spectrum)
    print(result.report())

The package is organised so that each layer can be used on its own:

``core``
    The :class:`~ramancarbon.core.spectrum.Spectrum` container, file
    readers, baseline removal, despiking and peak finding.
``models``
    Lineshapes (Lorentzian, Gaussian, pseudo-Voigt, Breit–Wigner–Fano) and
    the bounded least-squares engine that deconvolves the D and G bands.
``database``
    Every literature constant, in editable JSON with its source and a
    confidence flag. Nothing is hardcoded in the analysis code.
``analysis``
    Band assignment, intensity ratios, diameters, shifts against reference
    materials, and the rule-based SWCNT/DWCNT/MWCNT classifier.
``gui`` / ``cli``
    A Tkinter desktop application and a command-line interface.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .analysis.report import AnalysisResult, analyse
from .core.io import read_many, read_spectrum, write_spectrum
from .core.spectrum import Spectrum, stack_average
from .database import load_database

__all__ = [
    "AnalysisResult",
    "Spectrum",
    "__version__",
    "analyse",
    "load_database",
    "read_many",
    "read_spectrum",
    "stack_average",
    "write_spectrum",
]
