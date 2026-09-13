"""Reading and plotting the output of a finished calculation.

The rest of the package prepares inputs. This one closes the loop: it parses
what Quantum ESPRESSO and SIESTA actually produced and turns it into a band
diagram or a spectrum.

* :mod:`~carbonforge.results.bands` — ``bands.x`` output (``bands.dat`` and
  the ``.gnu`` variant) and SIESTA ``.bands`` files.
* :mod:`~carbonforge.results.spectra` — the ``dynmat.x`` frequency /
  IR / Raman table, with Lorentzian broadening into a plottable spectrum.

A caveat worth stating plainly: these parsers were written against the
documented file formats and are covered by tests using synthetic fixtures.
They have not been run against output from a real Quantum ESPRESSO or SIESTA
installation, because none is available in the environment this package was
developed in. Treat the first run on your own data as a check of the parser,
not only of the physics — and report anything that looks off.
"""

from .bands import (
    BandStructure,
    draw_bands_on_axes,
    read_qe_bands,
    read_qe_bands_gnu,
    read_siesta_bands,
)
from .dos import (
    DensityOfStates,
    ProjectedDOS,
    draw_dos_on_axes,
    plot_dos,
    read_dos,
    read_pdos,
)
from .spectra import (
    VibrationalSpectrum,
    draw_spectrum_on_axes,
    VibrationalMode,
    broaden,
    read_dynmat,
)

__all__ = [
    "DensityOfStates",
    "ProjectedDOS",
    "read_dos",
    "read_pdos",
    "draw_dos_on_axes",
    "plot_dos",
    "BandStructure",
    "draw_bands_on_axes",
    "draw_spectrum_on_axes",
    "read_qe_bands",
    "read_qe_bands_gnu",
    "read_siesta_bands",
    "VibrationalSpectrum",
    "VibrationalMode",
    "read_dynmat",
    "broaden",
]
