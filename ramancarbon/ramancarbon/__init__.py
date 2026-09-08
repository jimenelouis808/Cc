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
    Band assignment, intensity ratios, structural indices, diameters,
    shifts against reference materials, the rule-based
    SWCNT/DWCNT/MWCNT/CNF classifier, measurement-quality checks,
    multi-wavelength combination, parallel batch processing, and a separate
    path for transition-metal dichalcogenides.
``xrd``
    Diffraction: CIF, symmetry, calculated patterns, phase identification,
    Rietveld, Le Bail and Pawley, size-strain analysis and carbon
    microstructure.
``echem``
    Cyclic voltammetry, charge-discharge, impedance with equivalent
    circuits and the distribution of relaxation times, storage mechanism,
    diffusion coefficients and catalysis.
``mapping``
    Raman maps as a cube: per-pixel images and chemometrics (PCA,
    k-means, MCR-ALS).
``plotting``
    A configurable plot engine — scale, line width, colours, markers,
    labels, secondary axes, insets, panels — with journal presets that
    carry the real column widths.
``dataio``
    Format detection from the numbers rather than the extension, one
    reader for everything, export to six text formats, and ``.rcproj``
    project files.
``gui`` / ``cli``
    A Tkinter desktop application and a command-line interface.
``benchmarks``
    Times every operation, so that "optimised" is a number.
"""

from __future__ import annotations

__version__ = "0.5.0"

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
