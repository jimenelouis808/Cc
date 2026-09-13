"""Lennard-Jones parameters and partial charges, with their provenance.

Every number here is someone's published fit, and mixing sets that were
parameterised against different water models or different charge schemes is
the commonest way to get a plausible-looking but wrong simulation. So each
entry records where it came from, and :func:`describe_provenance` prints the
lot.

**The partial charges on doped and functionalised carbon deserve a warning
of their own.** Unlike the Lennard-Jones parameters, which come from
established transferable sets, there is no consensus set of charges for
N-doped graphene: published values depend on the charge-partitioning scheme
(Bader, Mulliken, Löwdin, RESP all disagree, sometimes by a factor of two),
on the functional, and on the supercell. The defaults below are
representative of the ranges reported in the literature and are fine for
exploring trends. For anything quantitative, derive them from your own DFT —
:func:`carbonforge.forcefields.charges.charges_from_lowdin` reads them
straight out of a ``projwfc.x`` run on the very structure you are simulating.

Units throughout are LAMMPS ``real``: energies in kcal/mol, distances in Å,
charges in elementary charge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LJParams:
    """Lennard-Jones parameters for one atom type, plus its charge.

    Attributes
    ----------
    epsilon
        Well depth in kcal/mol.
    sigma
        Distance at which the potential crosses zero, in Å.
    charge
        Partial charge in units of e.
    source
        Where the numbers come from.
    charge_confidence
        ``"established"`` for charges from a standard, widely-used model
        (water, simple ions); ``"representative"`` for doped-carbon charges,
        which vary between studies and should be re-derived for quantitative
        work.
    """

    epsilon: float
    sigma: float
    charge: float = 0.0
    source: str = ""
    charge_confidence: str = "established"


#: Electrode atom types. Lennard-Jones from the Steele graphite parameters
#: and OPLS-AA aromatics; charges as discussed in the module docstring.
ELECTRODE_PARAMS: dict[str, LJParams] = {
    "C_sp2": LJParams(
        epsilon=0.0559, sigma=3.40, charge=0.0,
        source="Steele, Surf. Sci. 36, 317 (1973) — el estándar para "
               "superficies de grafito. Carbono prístino sin carga neta.",
    ),
    "C_edge": LJParams(
        epsilon=0.0559, sigma=3.40, charge=-0.05,
        source="LJ de Steele; carga pequeña por la valencia incompleta.",
        charge_confidence="representative",
    ),
    "C_N": LJParams(
        epsilon=0.0559, sigma=3.40, charge=0.15,
        source="LJ de Steele; carga positiva porque el N vecino retira "
               "densidad. Valor representativo, no universal.",
        charge_confidence="representative",
    ),
    "C_O": LJParams(
        epsilon=0.0660, sigma=3.50, charge=0.15,
        source="LJ tipo OPLS-AA para C sp3 unido a O.",
        charge_confidence="representative",
    ),
    "C_sp3": LJParams(
        epsilon=0.0660, sigma=3.50, charge=0.0,
        source="OPLS-AA carbono sp3.",
    ),
    "N_graph": LJParams(
        epsilon=0.1700, sigma=3.25, charge=-0.45,
        source="LJ de tipo OPLS/AMBER para N aromático. La carga negativa "
               "refleja que el N grafítico atrae densidad de sus tres "
               "carbonos vecinos.",
        charge_confidence="representative",
    ),
    "N_pyri": LJParams(
        epsilon=0.1700, sigma=3.25, charge=-0.55,
        source="LJ como el grafítico; más negativo porque el N piridínico "
               "conserva un par libre en el plano.",
        charge_confidence="representative",
    ),
    "N_pyrr": LJParams(
        epsilon=0.1700, sigma=3.25, charge=-0.40,
        source="LJ como el grafítico; menos negativo porque el H comparte "
               "parte de la carga.",
        charge_confidence="representative",
    ),
    "N_amine": LJParams(
        epsilon=0.1700, sigma=3.30, charge=-0.90,
        source="OPLS-AA amina primaria.",
        charge_confidence="representative",
    ),
    "N_nitro": LJParams(
        epsilon=0.1700, sigma=3.25, charge=0.65,
        source="OPLS-AA nitro: el N queda positivo entre dos oxígenos.",
        charge_confidence="representative",
    ),
    "O_hydroxyl": LJParams(
        epsilon=0.1700, sigma=3.12, charge=-0.60,
        source="OPLS-AA alcohol.",
        charge_confidence="representative",
    ),
    "O_epoxy": LJParams(
        epsilon=0.1400, sigma=2.90, charge=-0.40,
        source="OPLS-AA éter; típico en óxido de grafeno.",
        charge_confidence="representative",
    ),
    "O_carbonyl": LJParams(
        epsilon=0.2100, sigma=2.96, charge=-0.50,
        source="OPLS-AA carbonilo.",
        charge_confidence="representative",
    ),
    "O_carboxyl": LJParams(
        epsilon=0.2100, sigma=2.96, charge=-0.45,
        source="OPLS-AA carboxilo (promedio de los dos oxígenos).",
        charge_confidence="representative",
    ),
    "H_C": LJParams(
        epsilon=0.0300, sigma=2.42, charge=0.10,
        source="OPLS-AA H aromático.",
        charge_confidence="representative",
    ),
    "H_O": LJParams(
        epsilon=0.0000, sigma=0.00, charge=0.42,
        source="OPLS-AA H de hidroxilo: sin LJ, solo carga, como en SPC/E.",
        charge_confidence="representative",
    ),
    "H_N": LJParams(
        epsilon=0.0000, sigma=0.00, charge=0.36,
        source="OPLS-AA H de amina.",
        charge_confidence="representative",
    ),
    "S_thiol": LJParams(
        epsilon=0.2500, sigma=3.55, charge=-0.20,
        source="OPLS-AA azufre de tiol.",
        charge_confidence="representative",
    ),
    "B_sub": LJParams(
        epsilon=0.0950, sigma=3.58, charge=0.35,
        source="LJ tipo UFF; el boro queda positivo al ser menos "
               "electronegativo que el carbono.",
        charge_confidence="representative",
    ),
    "P_sub": LJParams(
        epsilon=0.2000, sigma=3.74, charge=0.20,
        source="OPLS-AA fósforo.",
        charge_confidence="representative",
    ),
}


#: Electrolyte species. These charges ARE established: they define the models.
ELECTROLYTE_PARAMS: dict[str, LJParams] = {
    # SPC/E water. The charges are part of the model definition, not a fit
    # that can be varied independently.
    "OW": LJParams(
        epsilon=0.1553, sigma=3.166, charge=-0.8476,
        source="SPC/E — Berendsen et al., J. Phys. Chem. 91, 6269 (1987).",
    ),
    "HW": LJParams(
        epsilon=0.0, sigma=0.0, charge=0.4238,
        source="SPC/E — el hidrógeno no lleva LJ por construcción.",
    ),
    # Joung-Cheatham ions, parameterised specifically against SPC/E water.
    # Pairing them with a different water model is a known way to get the
    # solvation free energies wrong.
    "Na": LJParams(
        epsilon=0.3526, sigma=2.159, charge=1.0,
        source="Joung & Cheatham, J. Phys. Chem. B 112, 9020 (2008), "
               "conjunto para SPC/E.",
    ),
    "Cl": LJParams(
        epsilon=0.0128, sigma=4.830, charge=-1.0,
        source="Joung & Cheatham (2008), conjunto para SPC/E.",
    ),
    "K": LJParams(
        epsilon=0.4297, sigma=2.838, charge=1.0,
        source="Joung & Cheatham (2008), conjunto para SPC/E.",
    ),
    "Li": LJParams(
        epsilon=0.3367, sigma=1.409, charge=1.0,
        source="Joung & Cheatham (2008), conjunto para SPC/E.",
    ),
    # Coarse-grained ionic liquid: one site per ion. Crude, but it is what
    # makes microsecond EDLC simulations affordable, and it reproduces
    # capacitance trends.
    "BMIM": LJParams(
        epsilon=0.4300, sigma=5.04, charge=0.78,
        source="Modelo de grano grueso tipo Merlet/Salanne: un solo sitio "
               "por catión, carga reducida a 0.78 e para representar la "
               "polarización de forma efectiva.",
        charge_confidence="representative",
    ),
    "PF6": LJParams(
        epsilon=0.4300, sigma=5.06, charge=-0.78,
        source="Anión de grano grueso, emparejado con BMIM.",
        charge_confidence="representative",
    ),
}

#: Geometry of the SPC/E water molecule, which is rigid by construction.
SPCE_GEOMETRY = {
    "oh_distance": 1.0,      # Å
    "hoh_angle": 109.47,     # degrees
}


def get_params(type_name: str) -> LJParams:
    """Look up a type's parameters, from either table."""
    if type_name in ELECTRODE_PARAMS:
        return ELECTRODE_PARAMS[type_name]
    if type_name in ELECTROLYTE_PARAMS:
        return ELECTROLYTE_PARAMS[type_name]
    raise ValueError(
        f"Tipo sin parámetros: '{type_name}'. Disponibles: "
        f"{', '.join(sorted(set(ELECTRODE_PARAMS) | set(ELECTROLYTE_PARAMS)))}."
    )


def total_charge(type_names: list[str]) -> float:
    """Net charge of a set of typed atoms, in units of e."""
    return sum(get_params(name).charge for name in type_names)


def describe_provenance(type_names: Optional[list[str]] = None) -> str:
    """Print where every parameter in use comes from.

    Worth reading before publishing: the honest answer to "where did these
    charges come from?" is different for the water model than for the doped
    carbon, and this makes the difference visible.
    """
    names = sorted(set(type_names)) if type_names else sorted(
        set(ELECTRODE_PARAMS) | set(ELECTROLYTE_PARAMS)
    )
    lines = ["Procedencia de los parámetros en uso:", ""]
    representative: list[str] = []

    for name in names:
        try:
            params = get_params(name)
        except ValueError:
            continue
        lines.append(
            f"  {name:12s} ε={params.epsilon:.4f} kcal/mol  "
            f"σ={params.sigma:.3f} Å  q={params.charge:+.4f} e"
        )
        lines.append(f"               {params.source}")
        if params.charge_confidence == "representative" and params.charge:
            representative.append(name)

    if representative:
        lines += [
            "",
            "⚠️  Las cargas de estos tipos son REPRESENTATIVAS, no definitivas:",
            f"    {', '.join(representative)}",
            "",
            "    No existe un conjunto consensuado de cargas para carbono "
            "dopado o funcionalizado.",
            "    Los valores publicados dependen del esquema de partición "
            "(Bader, Mulliken, Löwdin,",
            "    RESP discrepan entre sí, a veces al doble), del funcional y "
            "de la supercelda.",
            "",
            "    Sirven para explorar tendencias. Para resultados "
            "cuantitativos, derívalas de tu",
            "    propio DFT sobre ESTA estructura:",
            "      carbonforge charges pdos_dir --apply-to estructura.xyz",
        ]
    return "\n".join(lines)


def mixing_note() -> str:
    """Explain the mixing rule and its main pitfall."""
    return (
        "Las interacciones cruzadas se obtienen por la regla de "
        "Lorentz-Berthelot\n"
        "(σ_ij = (σ_i+σ_j)/2, ε_ij = sqrt(ε_i·ε_j)), que es lo que hace "
        "LAMMPS por defecto.\n\n"
        "Cuidado con una cosa: los parámetros iónicos de Joung-Cheatham se "
        "ajustaron\n"
        "CONTRA el agua SPC/E. Combinarlos con TIP3P o TIP4P cambia las "
        "energías de\n"
        "solvatación y es una fuente conocida de resultados erróneos. Si "
        "cambias de\n"
        "modelo de agua, cambia también el conjunto de iones."
    )
