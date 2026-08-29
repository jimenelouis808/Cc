"""Calculation setups layered on top of a structure.

A structure alone does not define a simulation: you also need to say *what*
you want to compute. This package holds those decisions as small, inspectable
dataclasses that the exporters translate into Quantum ESPRESSO or SIESTA
input:

* :mod:`~carbonforge.calculations.kpaths` — high-symmetry band paths,
  dimensionality-aware.
* :mod:`~carbonforge.calculations.spectroscopy` — phonons at Γ plus IR and
  Raman activity (DFPT).
* :mod:`~carbonforge.calculations.spinorbit` — non-collinear magnetism and
  spin-orbit coupling.

Every setup is checked by :mod:`carbonforge.validation.calculations` before
it is written out, because most of these features have hard prerequisites
(norm-conserving pseudopotentials for Raman, a band gap for the dielectric
response, fully-relativistic pseudopotentials for spin-orbit) that are far
cheaper to catch here than after a job dies on a cluster.
"""

from .kpaths import (
    BandPathSpec,
    format_qe_kpath,
    format_siesta_bandlines,
    suggest_band_path,
)
from .spectroscopy import (
    SpectroscopySpec,
    format_dynmat_input,
    format_ph_input,
    format_runner_script,
    infrared_setup,
    phonon_setup,
    raman_setup,
)
from .spinorbit import (
    SpinOrbitSpec,
    heaviest_element,
    qe_system_fields,
    relativistic_pseudo_name,
    soc_is_physically_relevant,
    soc_setup,
)

__all__ = [
    # k-paths
    "BandPathSpec",
    "suggest_band_path",
    "format_qe_kpath",
    "format_siesta_bandlines",
    # spectroscopy
    "SpectroscopySpec",
    "raman_setup",
    "infrared_setup",
    "phonon_setup",
    "format_ph_input",
    "format_dynmat_input",
    "format_runner_script",
    # spin-orbit
    "SpinOrbitSpec",
    "soc_setup",
    "heaviest_element",
    "soc_is_physically_relevant",
    "qe_system_fields",
    "relativistic_pseudo_name",
]
