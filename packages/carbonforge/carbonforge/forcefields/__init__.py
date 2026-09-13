"""Classical force fields for charged carbon systems.

The reactive potentials the neutral LAMMPS exporter uses (AIREBO, Tersoff)
carry no charges, which makes them unusable for anything electrostatic — an
electric double layer above all. This package provides the other half: atom
typing from local chemistry, Lennard-Jones parameters and partial charges
with their provenance, electrolyte construction, and EDLC cell assembly.

The partial charges for doped and functionalised carbon are the weak point
and are labelled as such: there is no consensus set, so
:mod:`~carbonforge.forcefields.charges` derives them from a DFT run on the
structure actually being simulated.
"""

from .charges import (
    DerivedCharges,
    apply_derived_charges,
    charge_spread,
    charges_by_type,
    read_lowdin_charges,
)
from .edlc import EDLCCell, build_edlc_cell, check_edlc_setup
from .electrolyte import (
    ElectrolyteBox,
    build_aqueous,
    build_electrolyte,
    build_ionic_liquid,
)
from .parameters import (
    ELECTRODE_PARAMS,
    ELECTROLYTE_PARAMS,
    LJParams,
    describe_provenance,
    get_params,
    mixing_note,
    total_charge,
)
from .typing import ATOM_TYPES, AtomType, TypingResult, assign_types

__all__ = [
    # typing
    "assign_types",
    "TypingResult",
    "AtomType",
    "ATOM_TYPES",
    # parameters
    "LJParams",
    "get_params",
    "total_charge",
    "describe_provenance",
    "mixing_note",
    "ELECTRODE_PARAMS",
    "ELECTROLYTE_PARAMS",
    # electrolyte
    "build_electrolyte",
    "build_aqueous",
    "build_ionic_liquid",
    "ElectrolyteBox",
    # EDLC
    "build_edlc_cell",
    "EDLCCell",
    "check_edlc_setup",
    # charges from DFT
    "read_lowdin_charges",
    "DerivedCharges",
    "charges_by_type",
    "charge_spread",
    "apply_derived_charges",
]
