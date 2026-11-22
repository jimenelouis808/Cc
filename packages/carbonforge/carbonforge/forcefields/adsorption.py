"""A force field for gas adsorption on functionalised carbon walls.

Separate from :data:`~carbonforge.forcefields.parameters.ELECTRODE_PARAMS`
on purpose: the EDLC cells are built on that table and must not move when
an adsorption study corrects a charge. Everything here is the electrode
table plus overrides; nothing in it is edited.

Why it exists. The electrode table charges per atom type, and the generic
typing gives a carboxyl carbon the phenol carbon's type (``C_O``, +0.15 e).
Each COOH then carries -0.53 e net -- a carboxylate, not an acid -- and
in a CO2 Henry sweep that inflated its constant almost fourfold (2108
against 700 with published charges, 6.6 A slit at 298 K). So the groups
are retyped by environment and charged from published sets in which each
group sums to (nearly) zero:

* COOH and phenolic OH: Tenney & Lastoskie, *Environ. Prog.* **25** (2006)
  343, Table 3 (neutral groups on the attaching carbon).
* NH2: Chen et al., *Phys. Chem. Chem. Phys.* (2026), SI Table S1 (DFT,
  NBO-style per-atom charges; the group leaves -0.109 e on the sheet).

* Graphitic N: computed here (``validacion_rendijas/dft_cargas``, UKS
  PBE/def2-SVP on an H-terminated C36NH15 flake, IAO charges, stable to
  0.04 e against a C12NH9 flake). The N is nearly neutral and its three
  carbons slightly positive; the ~+0.25 e left per N is the electron it
  donates to the pi system, spread over the sheet by the neutralisation.
  With these, the CO2 q_st rise on 5 % N (+3.3 kJ/mol) matches a measured
  one (NH3-nitrided carbon, Nanomaterials 9 (2019) 1776: +3.3).

Pyridinic and pyrrolic N keep the electrode table's representative
charges: no source here gives them. These charges
are still model choices, not settled values; for quantitative work derive
them from DFT with :mod:`~carbonforge.forcefields.charges`.
"""

from __future__ import annotations

from ase import Atoms

from ..topology.graph import build_bond_graph
from .parameters import ELECTRODE_PARAMS, LJParams
from .typing import TypingResult, assign_types

_TENNEY = "Tenney & Lastoskie, Environ. Prog. 25 (2006) 343, Table 3"
_CHEN = "Chen et al., Phys. Chem. Chem. Phys. (2026), SI Table S1"
_DFT = ("this repository, validacion_rendijas/dft_cargas "
        "(PBE/def2-SVP, IAO, C36NH15 flake)")
_C = ELECTRODE_PARAMS["C_sp2"]


def _like(base: str, charge: float, source: str) -> LJParams:
    p = ELECTRODE_PARAMS[base]
    return LJParams(epsilon=p.epsilon, sigma=p.sigma, charge=charge,
                    source=source, charge_confidence="literature")


#: Types this force field adds or recharges. LJ is the electrode table's
#: for the matching element; only charges (and names) change.
ADSORPTION_OVERRIDES: dict[str, LJParams] = {
    # Carboxyl: C(=O)OH on a ring carbon.
    "C_carboxyl": LJParams(epsilon=_C.epsilon, sigma=_C.sigma, charge=0.75,
                           source=_TENNEY, charge_confidence="literature"),
    "C_base_carboxyl": LJParams(epsilon=_C.epsilon, sigma=_C.sigma,
                                charge=-0.06, source=_TENNEY,
                                charge_confidence="literature"),
    "O_carboxyl_double": _like("O_carbonyl", -0.50, _TENNEY),
    "O_carboxyl_hydroxyl": _like("O_hydroxyl", -0.55, _TENNEY),
    "H_carboxyl": _like("H_O", 0.36, _TENNEY),
    # Phenolic hydroxyl.
    "C_O": _like("C_O", 0.30, _TENNEY),
    "O_hydroxyl": _like("O_hydroxyl", -0.60, _TENNEY),
    "H_O": _like("H_O", 0.30, _TENNEY),
    # Amine.
    "C_amine": _like("C_N", -0.172, _CHEN),
    "N_amine": _like("N_amine", -0.539, _CHEN),
    "H_N": _like("H_N", 0.301, _CHEN),
    # Graphitic (substitutional) N and the carbons bonded to it.
    "N_graph": _like("N_graph", -0.017, _DFT),
    "C_graphN": _like("C_N", 0.089, _DFT),
}

#: The full table: electrode parameters with the overrides applied.
ADSORPTION_PARAMS: dict[str, LJParams] = {**ELECTRODE_PARAMS,
                                          **ADSORPTION_OVERRIDES}


def assign_adsorption_types(atoms: Atoms) -> TypingResult:
    """Types for :data:`ADSORPTION_PARAMS`: the generic typing, refined.

    A carbon with two oxygen neighbours becomes ``C_carboxyl``; its
    oxygens ``O_carboxyl_double`` (the one without H) and
    ``O_carboxyl_hydroxyl``, that one's H ``H_carboxyl``, and the ring
    carbon it hangs from ``C_base_carboxyl``. A carbon bonded to an
    ``N_amine`` becomes ``C_amine``, so the amine's carbon is not charged
    as the neighbour of a lattice nitrogen, and one bonded to an
    ``N_graph`` becomes ``C_graphN`` (its DFT charge is not the one a
    pyridinic N's neighbour would carry). Everything else is left as
    :func:`~carbonforge.forcefields.typing.assign_types` typed it.

    Returns a :class:`TypingResult` with the refined ``types``.
    """
    typed = assign_types(atoms)
    types = list(typed.types)
    graph = build_bond_graph(atoms)
    symbols = atoms.get_chemical_symbols()

    def neighbours(i: int) -> list[int]:
        return list(graph.neighbors(i)) if i in graph else []

    for i, s in enumerate(symbols):
        if s != "C":
            continue
        oxygens = [j for j in neighbours(i) if symbols[j] == "O"]
        if len(oxygens) == 2:
            types[i] = "C_carboxyl"
            for o in oxygens:
                hydrogens = [h for h in neighbours(o) if symbols[h] == "H"]
                if hydrogens:
                    types[o] = "O_carboxyl_hydroxyl"
                    for h in hydrogens:
                        types[h] = "H_carboxyl"
                else:
                    types[o] = "O_carboxyl_double"
            for c in neighbours(i):
                if symbols[c] == "C":
                    types[c] = "C_base_carboxyl"
        elif any(types[j] == "N_amine" for j in neighbours(i)):
            types[i] = "C_amine"
        elif any(types[j] == "N_graph" for j in neighbours(i)):
            types[i] = "C_graphN"
    typed.types = types
    return typed
