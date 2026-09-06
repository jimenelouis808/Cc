"""Ready-made recipes for the calculations people actually run.

Setting up a DFT calculation correctly means getting a dozen decisions right
at once — functional, dispersion, spin, k-mesh, cutoff, whether to relax
first, how many pools to parallelise over. Getting any one wrong can be
silent. A preset makes those decisions together, **adapting to the structure
it is given**, and explains what it chose.

The most important adaptation: a preset applied to a zigzag nanoribbon turns
on spin polarisation and sets up the antiferromagnetic edge guess, because
that is the ground state. You do not have to know to ask.

Presets are a starting point, not a substitute for judgement. Every one
reports its cutoff and k-mesh as *unconverged defaults* and points at
``carbonforge converge``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

from ase import Atoms

from ..calculations.dos import DOSSpec
from ..calculations.electronic import (
    ElectronicSpec,
    setup_antiferromagnetic_edges,
)
from ..calculations.spectroscopy import SpectroscopySpec, phonon_setup, raman_setup
from ..exports.qe import QESettings

TaskKind = Literal["relax", "scf", "bands", "dos", "phonon", "raman", "md"]


@dataclass
class PresetResult:
    """What a preset decided, ready to be written out."""

    atoms: Atoms
    settings: QESettings
    electronic: ElectronicSpec
    task: TaskKind
    relax_first: bool
    spectroscopy: Optional[SpectroscopySpec] = None
    dos: Optional[DOSSpec] = None
    decisions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def explain(self) -> str:
        """Human-readable account of every choice the preset made."""
        lines = ["Lo que se ha decidido por ti:"]
        lines.extend(f"  • {decision}" for decision in self.decisions)
        if self.warnings:
            lines.append("\nTen en cuenta:")
            lines.extend(f"  ⚠️  {warning}" for warning in self.warnings)
        lines.append(
            "\nLos valores de cutoff y malla k son puntos de PARTIDA, no "
            "están convergidos.\nAntes de publicar: carbonforge converge "
            "<estructura> --parameter cutoff"
        )
        return "\n".join(lines)


def _magnetic_setup(atoms: Atoms) -> tuple[Atoms, ElectronicSpec, list[str]]:
    """Turn on spin polarisation when the structure needs it.

    Zigzag ribbon edges are magnetic and antiferromagnetically coupled; that
    is the ground state, and a non-polarised run silently misses it. Any
    other structure is left non-polarised, which is correct and cheaper.
    """
    kind = str(atoms.info.get("structure_type", ""))
    edge = str(atoms.info.get("edge", ""))

    if kind == "nanoribbon" and edge == "zigzag":
        try:
            tagged, spec = setup_antiferromagnetic_edges(atoms)
        except ValueError as exc:
            return atoms, ElectronicSpec(), [
                f"Se intentó activar el estado antiferromagnético de los "
                f"bordes pero no fue posible ({exc}). El cálculo saldrá sin "
                "polarizar: revísalo."
            ]
        return tagged, spec, [
            "Polarización de espín ACTIVADA con bordes antiferromagnéticos. "
            "Una cinta zigzag tiene estados de borde magnéticos; sin esto el "
            "SCF converge a un estado que no es el fundamental y las bandas "
            "salen mal.",
        ]
    return atoms, ElectronicSpec(), []


def _needs_dispersion(atoms: Atoms) -> bool:
    """Whether the structure is held together partly by dispersion."""
    return str(atoms.info.get("structure_type", "")) in (
        "carbon_foam", "nanocoil",
    )


@dataclass(frozen=True)
class Preset:
    """A named recipe."""

    key: str
    label: str
    description: str
    task: TaskKind
    relax_first: bool = False
    when_to_use: str = ""


PRESETS: dict[str, Preset] = {
    "quick": Preset(
        key="quick",
        label="Comprobación rápida",
        description="Un SCF barato para ver que todo corre.",
        task="scf",
        when_to_use="Primer contacto con una estructura nueva. No publiques esto.",
    ),
    "geometry": Preset(
        key="geometry",
        label="Relajar geometría",
        description="Relajación de posiciones con dispersión incluida.",
        task="relax",
        when_to_use="Siempre, antes de cualquier propiedad. Nada de lo demás "
                    "vale sobre una geometría sin relajar.",
    ),
    "bands": Preset(
        key="bands",
        label="Estructura de bandas",
        description="Relajar → scf → bands, con espín si la estructura lo pide.",
        task="bands",
        relax_first=True,
        when_to_use="La pregunta habitual: ¿hay gap y cómo dispersan las bandas?",
    ),
    "bands-hse": Preset(
        key="bands-hse",
        label="Bandas con híbrido (HSE06)",
        description="Como 'bands' pero con HSE06, que da gaps mucho mejores.",
        task="bands",
        relax_first=True,
        when_to_use="Cuando el valor del gap es el resultado. Caro: relaja con "
                    "PBE primero y deja el híbrido para el paso final.",
    ),
    "dos": Preset(
        key="dos",
        label="Densidad de estados proyectada",
        description="Relajar → scf → nscf denso → dos.x + projwfc.x.",
        task="dos",
        relax_first=True,
        when_to_use="Para saber QUÉ átomos aportan estados en el nivel de "
                    "Fermi. Lo natural tras dopar con nitrógeno.",
    ),
    "raman": Preset(
        key="raman",
        label="Espectro Raman e IR",
        description="Relajar bien → scf → ph.x → dynmat.x, con pseudos NC.",
        task="raman",
        relax_first=True,
        when_to_use="Solo en sistemas con gap. En metálicos usa 'phonon'.",
    ),
    "phonon": Preset(
        key="phonon",
        label="Frecuencias de vibración",
        description="Fonones en Γ sin intensidades. Vale para metálicos.",
        task="phonon",
        relax_first=True,
        when_to_use="Comprobar estabilidad (modos imaginarios) o vibraciones "
                    "en un sistema sin gap.",
    ),
    "adsorption": Preset(
        key="adsorption",
        label="Adsorción / apilamiento",
        description="Relajación con dispersión, que es lo que une las piezas.",
        task="relax",
        when_to_use="Moléculas sobre la superficie, bicapas, apilamiento. Sin "
                    "vdW estos sistemas no ligan.",
    ),
}


def apply_preset(
    atoms: Atoms,
    preset_key: str,
    accurate: bool = False,
) -> PresetResult:
    """Turn a structure plus a preset name into a complete, explained setup.

    Parameters
    ----------
    atoms
        Structure to compute.
    preset_key
        Key into :data:`PRESETS`.
    accurate
        Tighten the defaults: denser k-mesh, higher cutoff, tighter
        convergence. Roughly 3-5x the cost.

    Returns
    -------
    PresetResult
        The (possibly modified) structure, the settings, and a record of
        every decision made.

    Raises
    ------
    ValueError
        For an unknown preset key.
    """
    if preset_key not in PRESETS:
        raise ValueError(
            f"Preset desconocido: '{preset_key}'. "
            f"Opciones: {', '.join(sorted(PRESETS))}."
        )
    preset = PRESETS[preset_key]
    decisions: list[str] = [f"Receta '{preset.label}': {preset.description}"]
    warnings: list[str] = []

    # --- spin -----------------------------------------------------------
    atoms, electronic, spin_notes = _magnetic_setup(atoms)
    decisions.extend(spin_notes)
    if not spin_notes:
        decisions.append("Sin polarización de espín: esta estructura no la necesita.")

    # --- dispersion ------------------------------------------------------
    if preset_key == "adsorption" or _needs_dispersion(atoms):
        electronic.vdw_correction = "grimme-d3"
        decisions.append(
            "Dispersión Grimme-D3 activada: sin ella PBE apenas liga las "
            "piezas que se mantienen unidas por van der Waals. Cuesta "
            "prácticamente nada."
        )

    # --- functional ------------------------------------------------------
    if preset_key == "bands-hse":
        electronic.functional = "hse"
        decisions.append(
            "Funcional HSE06: PBE subestima los gaps ~50 %. Esto los corrige, "
            "a un coste del orden de 30x."
        )
        warnings.append(
            "Relaja primero con PBE ('geometry') y aplica el híbrido solo al "
            "paso final; relajar con HSE casi nunca compensa."
        )

    # --- numerical settings ---------------------------------------------
    if preset_key == "quick":
        cutoff, density, conv = 40.0, 0.35, 1.0e-6
        decisions.append(
            "Ajustes deliberadamente bastos (40 Ry, malla gruesa): esto es "
            "para ver que arranca, no para obtener números."
        )
    elif accurate:
        cutoff, density, conv = 80.0, 0.12, 1.0e-10
        decisions.append("Modo preciso: 80 Ry, malla densa, convergencia estricta.")
    else:
        cutoff, density, conv = 60.0, 0.20, 1.0e-8
        decisions.append("Ajustes estándar: 60 Ry, malla media.")

    # Phonons need a tighter SCF than a plain energy: the force constants are
    # second derivatives, so noise in the density is amplified.
    if preset.task in ("phonon", "raman"):
        conv = min(conv, 1.0e-10)
        decisions.append(
            "Convergencia SCF apretada a 1e-10: las constantes de fuerza son "
            "segundas derivadas y amplifican cualquier ruido en la densidad."
        )

    settings = QESettings(
        calculation="scf",
        ecutwfc=cutoff,
        ecutrho=cutoff * 8.0,
        kpoint_density=density,
        conv_thr=conv,
        electronic=electronic,
    )

    # --- relaxation constraints -----------------------------------------
    if preset.relax_first or preset.task == "relax":
        dim = int(sum(atoms.get_pbc()))
        if dim in (1, 2):
            settings.cell_dofree = "z" if dim == 1 else "2Dxy"
            decisions.append(
                f"Si pasas a vc-relax, la celda queda restringida a "
                f"'{settings.cell_dofree}': en un sistema {dim}D relajarla "
                "libremente comprimiría el vacío."
            )

    # --- task-specific specs --------------------------------------------
    spectroscopy = None
    dos_spec = None
    if preset.task == "raman":
        spectroscopy = raman_setup()
        settings.pseudopotentials = {
            symbol: f"{symbol}_ONCV_PBE-1.2.upf"
            for symbol in set(atoms.get_chemical_symbols())
        }
        decisions.append(
            "Pseudopotenciales norm-conserving: el Raman por DFPT en QE no "
            "admite PAW ni ultrasoft. Descárgalos con 'carbonforge pseudos'."
        )
    elif preset.task == "phonon":
        spectroscopy = phonon_setup()
    elif preset.task == "dos":
        dos_spec = DOSSpec(projected=True, kmesh_factor=2)
        decisions.append(
            "El paso nscf usa una malla k 2x más densa: la que converge la "
            "densidad es demasiado gruesa para resolver una curva de DOS."
        )

    if preset.relax_first:
        decisions.append(
            "Se relajará la geometría primero y la propiedad se calculará "
            "sobre la estructura relajada, no sobre la construida."
        )

    return PresetResult(
        atoms=atoms,
        settings=settings,
        electronic=electronic,
        task=preset.task,
        relax_first=preset.relax_first,
        spectroscopy=spectroscopy,
        dos=dos_spec,
        decisions=decisions,
        warnings=warnings,
    )


def describe_presets() -> str:
    """Return a formatted table of the available presets."""
    lines = [f"{'clave':14s} {'qué hace':44s} cuándo usarlo", "-" * 100]
    for key in sorted(PRESETS):
        preset = PRESETS[key]
        lines.append(f"{preset.key:14s} {preset.description:44s} {preset.when_to_use}")
    return "\n".join(lines)
