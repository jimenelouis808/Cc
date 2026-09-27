"""tbkit: parametrisable tight binding for nanocarbons.

Build a :class:`~tbkit.hamiltonian.System` from an ASE structure and a
:class:`~tbkit.params.TBModel`, :func:`~tbkit.solver.solve` it, and read what
you need from :mod:`tbkit.analysis`. Self-consistent charges are in
:mod:`tbkit.scc`, mean-field Hubbard spin (edge magnetism, magnetisation
against energy, field and doping) in :mod:`tbkit.hubbard`, DFTB ``.skf``
parameters in :mod:`tbkit.skf` and fitting to DFT in :mod:`tbkit.fit`.

tbkit imports none of the other packages of the workspace; structures come
in as files (extxyz from carbonforge or nanocarbon_lab).
"""

from .hamiltonian import System
from .params import TBModel, pi_model, xu_carbon
from .solver import Solution, solve

__version__ = "0.1.0"

__all__ = ["Solution", "System", "TBModel", "pi_model", "solve", "xu_carbon"]
