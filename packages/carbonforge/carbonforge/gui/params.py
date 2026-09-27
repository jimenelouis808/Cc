"""Declarative parameter specs and build logic behind the GUI.

This module deliberately contains **no Tk code**: the GUI reads
:data:`STRUCTURES` to lay out its widgets, and calls :func:`build_structure`
/ :func:`apply_modifiers` / :func:`export_structure` to do the actual work.
That keeps every non-cosmetic code path unit-testable on a headless machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Sequence

from ase import Atoms

from ..builders import (
    build_carbon_foam,
    build_cnt,
    build_graphene_supercell,
    build_nanocoil,
    build_nanoribbon,
)
from ..defects import introduce_vacancies
from ..dopants import dope_random
import numpy as np

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D, DEFAULT_VACUUM_2D

ParamKind = Literal["int", "float", "bool", "choice", "text"]

#: Site kinds for dopants and groups, from carbonforge.placement.REGIONS.
_REGION_CHOICES: tuple[str, ...] = (
    "any", "edge", "edge_armchair", "edge_zigzag", "basal", "pentagon", "heptagon",
    "defect_57", "near_defect", "vacancy_rim",
)
_REGION_HELP = (
    "edge / edge_armchair / edge_zigzag: bordes (en un C–H, el grupo sustituye al H). "
    "basal: plano (sp3). pentagon / heptagon / defect_57: anillos del defecto 5-7. "
    "near_defect: vecinos de un defecto. vacancy_rim: borde de una vacante."
)


@dataclass(frozen=True)
class ParamSpec:
    """One user-editable parameter.

    Attributes
    ----------
    key
        Keyword argument name passed to the builder.
    label
        Human-readable label shown in the GUI.
    kind
        ``"int"``, ``"float"``, ``"bool"`` or ``"choice"``.
    default
        Initial value.
    minimum, maximum
        Optional inclusive bounds, enforced by :func:`coerce_value`.
    choices
        Allowed values when ``kind == "choice"``.
    help
        Short tooltip / hint describing the physical meaning.
    """

    key: str
    label: str
    kind: ParamKind
    default: Any
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: Optional[Sequence[str]] = None
    help: str = ""


@dataclass(frozen=True)
class StructureSpec:
    """A structure type exposed by the GUI."""

    key: str
    label: str
    description: str
    builder: Callable[..., Atoms]
    params: Sequence[ParamSpec] = field(default_factory=tuple)
    supports_modifiers: bool = True


_BOND = ParamSpec(
    "bond", "Longitud de enlace C-C (Å)", "float", CC_BOND,
    minimum=1.0, maximum=1.8,
    help="Distancia de equilibrio sp2. 1.42 Å salvo que sepas lo que haces.",
)
_VACUUM_1D = ParamSpec(
    "vacuum", "Vacío (Å)", "float", DEFAULT_VACUUM_1D,
    minimum=0.0, maximum=60.0,
    help="Separación entre imágenes periódicas. >=10 Å para DFT.",
)
_VACUUM_2D = ParamSpec(
    "vacuum", "Vacío (Å)", "float", DEFAULT_VACUUM_2D,
    minimum=0.0, maximum=60.0,
    help="Separación entre láminas. >=15 Å recomendado para DFT.",
)


STRUCTURES: dict[str, StructureSpec] = {
    "cnt": StructureSpec(
        key="cnt",
        label="Nanotubo (CNT)",
        description=(
            "Nanotubo de pared simple. (n,n) = armchair, (n,0) = zigzag, "
            "resto = quiral. Periódico a lo largo de z."
        ),
        builder=build_cnt,
        params=(
            ParamSpec("n", "Índice quiral n", "int", 6, minimum=1, maximum=40,
                      help="Debe cumplirse n >= 1."),
            ParamSpec("m", "Índice quiral m", "int", 6, minimum=0, maximum=40,
                      help="Debe cumplirse 0 <= m <= n."),
            ParamSpec("length", "Longitud objetivo (Å)", "float", 10.0,
                      minimum=1.0, maximum=200.0,
                      help="Se redondea al múltiplo entero del periodo."),
            _BOND,
            _VACUUM_1D,
        ),
    ),
    "graphene": StructureSpec(
        key="graphene",
        label="Grafeno (lámina)",
        description="Supercelda ortogonal de grafeno, periódica en x e y.",
        builder=build_graphene_supercell,
        params=(
            ParamSpec("nx", "Repeticiones en x", "int", 4, minimum=1, maximum=30,
                      help="Cada celda ortogonal aporta 4 átomos."),
            ParamSpec("ny", "Repeticiones en y", "int", 4, minimum=1, maximum=30),
            _BOND,
            _VACUUM_2D,
        ),
    ),
    "nanoribbon": StructureSpec(
        key="nanoribbon",
        label="Nanocinta (nanoribbon)",
        description=(
            "Cinta de grafeno con bordes definidos, periódica a lo largo de z."
        ),
        builder=build_nanoribbon,
        params=(
            ParamSpec("width", "Ancho", "int", 6, minimum=1, maximum=30,
                      help="Número de líneas de dímeros (convención ASE)."),
            ParamSpec("length", "Largo (celdas)", "int", 4, minimum=1, maximum=40),
            ParamSpec("edge", "Tipo de borde", "choice", "zigzag",
                      choices=("zigzag", "armchair")),
            ParamSpec("passivate", "Pasivar bordes con H", "bool", False,
                      help="Satura los carbonos del borde (C-H = 1.09 Å)."),
            _BOND,
            _VACUUM_2D,
        ),
    ),
    "nanocoil": StructureSpec(
        key="nanocoil",
        label="Nanoespiral (nanocoil)",
        description=(
            "CNT enrollado sobre una hélice. Estructura finita, pensada para "
            "relajarse después con AIREBO/Tersoff o DFT."
        ),
        builder=build_nanocoil,
        params=(
            ParamSpec("n", "Índice quiral n", "int", 6, minimum=1, maximum=30),
            ParamSpec("m", "Índice quiral m", "int", 6, minimum=0, maximum=30),
            ParamSpec("coil_radius", "Radio de la hélice R (Å)", "float", 25.0,
                      minimum=5.0, maximum=300.0,
                      help="Debe ser >= 2x el radio del tubo. R grande = menos tensión."),
            ParamSpec("pitch", "Paso P (Å)", "float", 12.0,
                      minimum=1.0, maximum=200.0,
                      help="Avance vertical por vuelta."),
            ParamSpec("n_turns", "Número de vueltas", "float", 1.0,
                      minimum=0.1, maximum=10.0,
                      help="Admite decimales: 1.5 = vuelta y media."),
            ParamSpec("stone_wales_density", "Densidad Stone-Wales", "float", 0.0,
                      minimum=0.0, maximum=0.02,
                      help="Fracción de enlaces rotados, sesgada a la pared externa. Máx 0.02."),
            _BOND,
            _VACUUM_1D,
        ),
    ),
    "foam": StructureSpec(
        key="foam",
        label="Espuma 3D (foam)",
        description=(
            "Red desordenada de fragmentos grafíticos en una caja cúbica "
            "periódica. Requiere relajación posterior."
        ),
        builder=build_carbon_foam,
        params=(
            ParamSpec("box_size", "Lado de la caja (Å)", "float", 30.0,
                      minimum=10.0, maximum=200.0),
            ParamSpec("n_flakes", "Número de fragmentos", "int", 20,
                      minimum=1, maximum=500),
            ParamSpec("flake_radius", "Radio del fragmento (Å)", "float", 4.0,
                      minimum=1.5, maximum=30.0,
                      help="La caja debe medir al menos 3x este radio."),
            ParamSpec("seed", "Semilla aleatoria", "int", 0,
                      minimum=0, maximum=10_000_000,
                      help="Misma semilla = misma estructura, siempre."),
        ),
        supports_modifiers=False,
    ),
}


#: The friendly path: pick a goal, let the preset choose the physics.
PRESET_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("preset", "Receta", "choice", "ninguna",
              choices=("ninguna", "quick", "geometry", "bands", "bands-hse",
                       "dos", "phonon", "raman", "adsorption"),
              help="Elige un objetivo y la receta decide funcional, "
                   "dispersión, espín y malla, adaptándose a tu estructura. "
                   "Con 'ninguna' controlas todo a mano abajo."),
    ParamSpec("accurate", "Ajustes precisos", "bool", False,
              help="Malla más densa y cutoff más alto. Unas 3-5 veces más caro."),
    ParamSpec("cores", "Núcleos previstos", "int", 8, minimum=1, maximum=4096,
              help="Para dimensionar los pools de puntos k de Quantum ESPRESSO."),
)


#: What to compute, on top of the structure itself.
CALCULATION_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("task", "Tipo de cálculo", "choice", "scf",
              choices=("scf", "relax", "vc-relax", "bandas",
                       "fonones", "infrarrojo", "raman"),
              help="'bandas' escribe scf+bands+bands.x; los espectroscópicos, scf+ph.x+dynmat.x."),
    ParamSpec("spinorbit", "Acoplamiento espín-órbita", "bool", False,
              help="Requiere pseudopotenciales relativistas. En carbono puro el "
                   "efecto es ~0.01 meV."),
    ParamSpec("kpoint_density", "Densidad de puntos k (1/Å)", "float", 0.20,
              minimum=0.02, maximum=1.0,
              help="Menor = malla más densa y cálculo más caro."),
    ParamSpec("ecutwfc", "Cutoff de ondas planas (Ry)", "float", 60.0,
              minimum=20.0, maximum=200.0,
              help="Solo Quantum ESPRESSO. 60 Ry es un punto de partida, no un valor convergido."),
    ParamSpec("band_npoints", "Puntos k por segmento (bandas)", "int", 30,
              minimum=5, maximum=200),
    ParamSpec("laser_nm", "Longitud de onda del láser (nm)", "float", 532.0,
              minimum=200.0, maximum=1200.0,
              help="Solo para intensidades Raman."),
    ParamSpec("spin", "Espín", "choice", "auto",
              choices=("auto", "none", "afm_edges", "ferro"),
              help="auto: antiferromagnético en los bordes de una cinta zigzag, sin "
                   "espín en lo demás. afm_edges lo fuerza; ferro, paralelo."),
    ParamSpec("functional", "Funcional", "choice", "pbe",
              choices=("pbe", "pbesol", "revpbe", "hse", "b3lyp", "vdw-df2"),
              help="Los híbridos (hse, b3lyp) cuestan 10-100 veces un PBE."),
    ParamSpec("vdw", "Corrección de van der Waals", "choice", "none",
              choices=("none", "grimme-d2", "grimme-d3", "ts", "xdm"),
              help="Imprescindible en espumas, espirales, apilamientos y fisisorción."),
    ParamSpec("occupations", "Ocupaciones", "choice", "smearing",
              choices=("smearing", "fixed"),
              help="'fixed' solo en sistemas con gap."),
    ParamSpec("degauss", "Ancho del smearing (Ry)", "float", 0.01,
              minimum=0.0005, maximum=0.1),
    ParamSpec("ecutrho", "Cutoff de densidad (Ry, 0 = automático)", "float", 0.0,
              minimum=0.0, maximum=2400.0,
              help="Automático: 8x ecutwfc con PAW, 4x con norm-conserving."),
    ParamSpec("cell_dofree", "Celda en vc-relax", "choice", "auto",
              choices=("auto", "all", "2Dxy", "2Dshape", "xy", "x", "y", "z"),
              help="auto: solo las direcciones periódicas, para no comprimir el vacío."),
    ParamSpec("pseudo_family", "Pseudopotenciales", "choice", "auto",
              choices=("auto", "PAW", "NC"),
              help="auto: norm-conserving para Raman (ph.x no admite PAW), PAW en "
                   "lo demás."),
)

#: How a Fix's value maps onto this form, where the labels differ.
_FIX_VALUES: dict[str, dict[Any, Any]] = {
    "task": {"phonon": "fonones", "relax": "relax"},
}


def apply_fix(raw_values: dict[str, Any], fix) -> dict[str, Any]:
    """Return a copy of the form values with one :class:`Fix` applied.

    Raises
    ------
    KeyError
        If the fix names a setting this form does not have.
    """
    known = {spec.key for spec in (*CALCULATION_PARAMS, *PRESET_PARAMS,
                                   *FUNCTIONALIZATION_PARAMS, *MODIFIER_PARAMS)}
    if fix.setting not in known:
        raise KeyError(f"La corrección cambia '{fix.setting}', que este formulario no tiene.")
    value = _FIX_VALUES.get(fix.setting, {}).get(fix.value, fix.value)
    return {**raw_values, fix.setting: value}


#: Maps the GUI's task label onto (QE calculation, spectroscopy mode).
_TASK_MAP: dict[str, tuple[str, Optional[str]]] = {
    "scf": ("scf", None),
    "relax": ("relax", None),
    "vc-relax": ("vc-relax", None),
    "bandas": ("bands", None),
    "fonones": ("scf", "phonon"),
    "infrarrojo": ("scf", "ir"),
    "raman": ("scf", "ir+raman"),
}


#: Functional groups and nitrogen configurations. Kept separate from
#: MODIFIER_PARAMS because they are different chemistry: a group is attached
#: to a carbon, a nitrogen configuration sits inside the lattice.
FUNCTIONALIZATION_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("group", "Grupo funcional", "choice", "ninguno",
              choices=("ninguno", "H", "OH", "NH2", "NO2", "CN", "COOH",
                       "CHO", "CONH2", "O", "SH", "CH3", "epoxy"),
              help="Se ANCLA al carbono. Los nitrogenados son NH2, NO2, CN y CONH2."),
    ParamSpec("group_count", "Cuántos grupos", "int", 1, minimum=1, maximum=50),
    ParamSpec("group_site", "Dónde anclarlos", "choice", "edge",
              choices=tuple(r for r in _REGION_CHOICES if r != "any"),
              help=_REGION_HELP),
    ParamSpec("group_indices", "Átomos exactos (opcional)", "text", "",
              help="Índices separados por comas o rangos, p. ej. '12, 30-32'. "
                   "Si se rellena, manda sobre la región y la cantidad."),
    ParamSpec("group_face", "Cara del plano (basal)", "choice", "+",
              choices=("+", "-"),
              help="En una lámina plana, a qué lado van los grupos basales."),
    ParamSpec("group_avoid", "Distancia mínima a dopantes/grupos (Å)", "float", 2.6,
              minimum=0.0, maximum=10.0,
              help="Evita anclar junto a un heteroátomo o a otro grupo. 0 lo permite."),
    ParamSpec("nitrogen", "Nitrógeno en la red", "choice", "ninguno",
              choices=("ninguno", "graphitic", "pyridinic", "pyrrolic", "n-oxide"),
              help="Esto NO es un grupo anclado: el N va DENTRO de los anillos. "
                   "El XPS del N 1s las separa."),
    ParamSpec("nitrogen_count", "Cuántos sitios de N", "int", 1,
              minimum=1, maximum=20),
    ParamSpec("passivate", "Pasivar bordes con H antes", "bool", False,
              help="Satura los bordes sueltos, que si no dan estados espurios "
                   "en el gap."),
)


MODIFIER_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("dopant", "Dopante", "choice", "ninguno",
              choices=("ninguno", "N", "B", "S", "P"),
              help="Sustitución de carbonos por el elemento elegido."),
    ParamSpec("dopant_concentration", "Concentración de dopante", "float", 0.0,
              minimum=0.0, maximum=0.5,
              help="Fracción de carbonos sustituidos (0.05 = 5%). Se ignora si se "
                   "da un número de dopantes."),
    ParamSpec("dopant_count", "Número de dopantes (0 = usar concentración)", "int", 0,
              minimum=0, maximum=500),
    ParamSpec("dopant_region", "Dónde doparlos", "choice", "any",
              choices=_REGION_CHOICES, help=_REGION_HELP),
    ParamSpec("dopant_indices", "Átomos exactos (opcional)", "text", "",
              help="Índices, p. ej. '5, 17'. Mandan sobre región y cantidad."),
    ParamSpec("dopant_separation", "Separación entre dopantes (Å)", "float", 2.5,
              minimum=0.0, maximum=20.0),
    ParamSpec("vacancies", "Vacancias", "int", 0, minimum=0, maximum=200,
              help="Número de átomos eliminados."),
    ParamSpec("seed", "Semilla (dopaje/defectos)", "int", 0,
              minimum=0, maximum=10_000_000,
              help="Garantiza que el resultado sea reproducible."),
)


def coerce_value(spec: ParamSpec, raw: Any) -> Any:
    """Convert and bounds-check a raw GUI value against its spec.

    Parameters
    ----------
    spec
        The parameter definition.
    raw
        Whatever the widget produced (usually a string).

    Returns
    -------
    Any
        Value converted to the spec's type.

    Raises
    ------
    ValueError
        If the value cannot be converted or falls outside the bounds. The
        message is user-facing and written in Spanish, since it is surfaced
        directly in the GUI.
    """
    if spec.kind == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "sí", "si", "yes", "on"}

    if spec.kind == "text":
        return str(raw).strip()

    if spec.kind == "choice":
        value = str(raw).strip()
        if spec.choices and value not in spec.choices:
            raise ValueError(
                f"'{spec.label}': '{value}' no es válido. "
                f"Opciones: {', '.join(spec.choices)}."
            )
        return value

    text = str(raw).strip().replace(",", ".")
    if not text:
        raise ValueError(f"'{spec.label}' no puede estar vacío.")
    try:
        value = int(text) if spec.kind == "int" else float(text)
    except ValueError:
        tipo = "un número entero" if spec.kind == "int" else "un número"
        raise ValueError(f"'{spec.label}' debe ser {tipo} (recibido: '{raw}').") from None

    if spec.minimum is not None and value < spec.minimum:
        raise ValueError(f"'{spec.label}' debe ser >= {spec.minimum} (recibido: {value}).")
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f"'{spec.label}' debe ser <= {spec.maximum} (recibido: {value}).")
    return value


def collect_values(specs: Sequence[ParamSpec], raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce a dict of raw widget values using ``specs``.

    Missing keys fall back to the spec default.
    """
    out: dict[str, Any] = {}
    for spec in specs:
        out[spec.key] = coerce_value(spec, raw.get(spec.key, spec.default))
    return out


def build_structure(kind: str, raw_values: dict[str, Any]) -> Atoms:
    """Build the requested structure from raw GUI values.

    Parameters
    ----------
    kind
        Key into :data:`STRUCTURES`.
    raw_values
        Mapping of parameter key to raw value.

    Returns
    -------
    ase.Atoms

    Raises
    ------
    ValueError
        For an unknown ``kind`` or invalid parameter values. Builder-level
        physical checks (e.g. a coil radius too small) propagate unchanged.
    """
    if kind not in STRUCTURES:
        raise ValueError(
            f"Tipo de estructura desconocido: '{kind}'. "
            f"Opciones: {', '.join(STRUCTURES)}."
        )
    spec = STRUCTURES[kind]
    kwargs = collect_values(spec.params, raw_values)
    return spec.builder(**kwargs)


def apply_modifiers(atoms: Atoms, raw_values: dict[str, Any]) -> Atoms:
    """Apply doping and vacancies according to the modifier values.

    A ``dopant`` of ``"ninguno"`` or a zero concentration is a no-op, as is a
    zero vacancy count. Both operations are seeded, so the same inputs always
    give the same structure.
    """
    from ..dopants import dope_at_sites
    from ..placement import parse_indices

    values = collect_values(MODIFIER_PARAMS, raw_values)
    seed = int(values["seed"])
    out = atoms

    dopant = values["dopant"]
    concentration = float(values["dopant_concentration"])
    indices = parse_indices(values["dopant_indices"])
    count = int(values["dopant_count"])
    region = values["dopant_region"]
    if dopant != "ninguno":
        if indices:
            out = dope_at_sites(out, dopant, indices=indices, seed=seed)
        elif count > 0 or (region != "any" and concentration > 0):
            if count == 0:
                n_carbon = sum(1 for s in out.get_chemical_symbols() if s == "C")
                count = max(1, int(round(concentration * n_carbon)))
            out = dope_at_sites(out, dopant, count=count, region=region, seed=seed,
                                min_separation=float(values["dopant_separation"]))
        elif concentration > 0:
            out = dope_random(out, dopant, concentration, seed=seed)

    n_vac = int(values["vacancies"])
    if n_vac > 0:
        out = introduce_vacancies(out, n_defects=n_vac, seed=seed)
    return out


def import_and_repair(path: str, autofix_it: bool = True) -> tuple[Atoms, str]:
    """Import a structure file, diagnose it and optionally repair it.

    Returns the structure plus a report, so the GUI can show exactly what
    arrived and what was changed before anything is calculated on it.
    """
    from ..io import autofix as run_autofix
    from ..io import import_structure

    result = import_structure(path)
    report = [result.summary()]

    atoms = result.atoms
    if autofix_it and result.fixable_issues:
        fixed = run_autofix(atoms, result.issues)
        atoms = fixed.atoms
        report += ["", "--- Reparación ---", fixed.summary()]
    return atoms, "\n".join(report)


def scan_pseudopotentials(
    directory: str,
    atoms: Optional[Atoms] = None,
    needs_raman: bool = False,
    needs_soc: bool = False,
) -> str:
    """Scan a pseudopotential folder and match it against a calculation.

    Reads the UPF headers rather than trusting filenames, so "you have a
    carbon file but it is PAW and Raman cannot use it" is distinguished from
    "you have no carbon file".
    """
    from ..io import download_instructions, match_requirements, scan_directory

    try:
        catalog = scan_directory(directory)
    except ValueError as exc:
        return f"No se pudo leer la carpeta: {exc}"

    lines = [catalog.summary()]
    if atoms is not None:
        match = match_requirements(
            catalog, atoms, needs_raman=needs_raman, needs_soc=needs_soc
        )
        lines += ["", "--- Para tu cálculo ---", match.summary()]
        if match.missing:
            lines += ["", download_instructions(
                match.missing, needs_raman=needs_raman, needs_soc=needs_soc
            )]
    return "\n".join(lines)


def check_parameter_constraints(
    atoms: Optional[Atoms],
    raw_values: dict[str, Any],
) -> str:
    """Report incompatible parameter combinations, before building."""
    from .constraints import check_constraints, format_violations

    text = format_violations(check_constraints(raw_values, atoms))
    for code in ("qe", "siesta"):
        typed, report = advanced_overrides(raw_values, code)
        if typed or report.errors or report.warnings:
            lines = [f"\n--- Parámetros avanzados ({code}) ---"]
            lines += [f"  ✗ {e}" for e in report.errors]
            lines += [f"  ! {w}" for w in report.warnings]
            lines += [f"  {k} = {v}" for k, v in typed.items()]
            text += "\n".join(lines)
    return text


def preview_preset(atoms: Atoms, raw_values: dict[str, Any]) -> str:
    """Explain what a preset would do to this structure, before running it.

    Shown live in the GUI so the physics choices are visible at the moment
    they are made, rather than buried in a generated input file.
    """
    from ..workflows.presets import apply_preset

    values = collect_values(PRESET_PARAMS, raw_values)
    key = values["preset"]
    if key == "ninguna":
        return (
            "Sin receta: controlas los ajustes a mano en el panel de cálculo.\n"
            "Si no tienes claro qué elegir, prueba con una receta: se adaptan "
            "a la estructura.\n\nPor ejemplo, en una cinta zigzag activan "
            "solo el estado antiferromagnético de los bordes, que es el "
            "fundamental."
        )
    try:
        result = apply_preset(
            atoms, key, accurate=bool(values["accurate"])
        )
    except ValueError as exc:
        return f"No se pudo aplicar la receta: {exc}"

    lines = [result.explain(), "", "--- Física incluida ---",
             result.electronic.describe()]

    from ..workflows.pipeline import plan_run

    plan = plan_run(result.atoms, result.settings, n_cores=int(values["cores"]))
    lines += ["", "--- Paralelización ---", plan.explain()]
    return "\n".join(lines)


def build_calculation_specs(raw_values: dict[str, Any]) -> dict[str, Any]:
    """Turn raw calculation widgets into the objects the exporters consume.

    Returns a dict with keys ``calculation`` (the QE task name),
    ``spectroscopy`` (a :class:`SpectroscopySpec` or ``None``),
    ``spinorbit`` (a :class:`SpinOrbitSpec` or ``None``) and the numeric
    settings.
    """
    from ..calculations.spectroscopy import SpectroscopySpec
    from ..calculations.spinorbit import SpinOrbitSpec

    values = collect_values(CALCULATION_PARAMS, raw_values)
    task = values["task"]
    calculation, spectro_mode = _TASK_MAP[task]

    spectroscopy = None
    if spectro_mode is not None:
        spectroscopy = SpectroscopySpec(
            mode=spectro_mode,
            laser_wavelength_nm=float(values["laser_nm"]),
        )

    spinorbit = SpinOrbitSpec() if values["spinorbit"] else None

    return {
        "task": task,
        "calculation": calculation,
        "spectroscopy": spectroscopy,
        "spinorbit": spinorbit,
        "kpoint_density": float(values["kpoint_density"]),
        "ecutwfc": float(values["ecutwfc"]),
        "band_npoints": int(values["band_npoints"]),
    }


def electronic_setup(atoms: Atoms, raw_values: dict[str, Any]):
    """The electronic-structure choices of the form, applied to ``atoms``.

    Returns ``(atoms, spec, notes)``: the structure (tagged by edge when the
    antiferromagnetic edge state is set up), the
    :class:`~carbonforge.calculations.electronic.ElectronicSpec`, and what
    was decided, in words. The same function feeds validation and export,
    so what is validated is what gets written.
    """
    from ..calculations.electronic import ElectronicSpec, setup_antiferromagnetic_edges

    values = collect_values(CALCULATION_PARAMS, raw_values)
    kind = str(atoms.info.get("structure_type", ""))
    zigzag = kind == "nanoribbon" and str(atoms.info.get("edge", "")) == "zigzag"
    spin = values["spin"]
    if spin == "auto":
        spin = "afm_edges" if zigzag else "none"
    notes: list[str] = []

    if spin == "afm_edges":
        try:
            atoms, spec = setup_antiferromagnetic_edges(atoms)
        except ValueError as exc:
            raise ValueError(
                f"No se pudo activar el espín antiferromagnético de los bordes: {exc}"
            ) from exc
        notes.append("Espín: antiferromagnético entre los dos bordes (C1/C2).")
    elif spin == "ferro":
        elements = sorted(set(atoms.get_chemical_symbols()) - {"H"})
        spec = ElectronicSpec(spin="collinear",
                              starting_magnetization={el: 0.3 for el in elements})
        notes.append("Espín: ferromagnético (momentos iniciales paralelos).")
    else:
        spec = ElectronicSpec()
    functional = values["functional"]
    spec.functional = None if functional == "pbe" else functional
    spec.vdw_correction = values["vdw"]
    spec.__post_init__()                      # re-validate the edited choices
    return atoms, spec, notes


#: Key of the form dict holding advanced parameters: ``{code: {name: raw}}``.
ADVANCED_KEY = "advanced"


def advanced_overrides(raw_values: dict[str, Any], code: str):
    """The advanced parameters of ``code`` in the form, typed and checked.

    Returns ``(typed, report)``: ``typed`` is what the writer receives
    (``QESettings.extra``, ``SiestaSettings.extra``, ``CalcSpec.extra``);
    ``report`` carries unknown names, wrong types and overrides of values
    carbonforge decides, as errors and warnings.
    """
    from ..codes import load_catalog
    from ..validation.checks import ValidationReport

    raw = dict((raw_values.get(ADVANCED_KEY) or {}).get(code) or {})
    if not raw:
        return {}, ValidationReport()
    typed, report = load_catalog(code).check(raw)
    if code == "qe":
        calculation = build_calculation_specs(raw_values)["calculation"]
        written = {"control", "system", "electrons"}
        written |= {"ions"} if calculation in ("relax", "vc-relax") else set()
        written |= {"cell"} if calculation == "vc-relax" else set()
        for key in list(typed):
            section = key.rpartition(".")[0].lower()
            if section not in written:
                report.errors.append(
                    f"'{key}': un cálculo '{calculation}' no escribe "
                    f"&{(section or '?').upper()}; cambia el tipo de cálculo o quita el parámetro."
                )
                del typed[key]
    return typed, report


def qe_settings_for(atoms: Atoms, raw_values: dict[str, Any], electronic=None):
    """:class:`~carbonforge.exports.qe.QESettings` from the form.

    Pseudopotentials: norm-conserving when asked for, or automatically when
    the task computes Raman (ph.x refuses PAW); otherwise the PAW defaults.
    The density cutoff, when left automatic, follows the family: 8x ecutwfc
    for PAW, 4x for norm-conserving.
    """
    from ..exports.pseudos import pseudopotential_map, requirements_for
    from ..exports.qe import QESettings, infer_qe_settings

    specs = build_calculation_specs(raw_values)
    values = collect_values(CALCULATION_PARAMS, raw_values)
    spectro = specs["spectroscopy"]
    family = values["pseudo_family"]
    if family == "auto":
        family = "NC" if spectro is not None and spectro.needs_raman else "PAW"
    pseudos = None
    if family == "NC":
        pseudos = pseudopotential_map(requirements_for(
            atoms, needs_raman=True, needs_soc=specs["spinorbit"] is not None))
    ecutrho = float(values["ecutrho"]) or specs["ecutwfc"] * (4.0 if family == "NC" else 8.0)

    dofree = values["cell_dofree"]
    if dofree == "auto":
        dofree = None
        pbc = atoms.get_pbc()
        if specs["calculation"] == "vc-relax" and 0 < int(sum(pbc)) < 3:
            dofree = "2Dxy" if int(sum(pbc)) == 2 else "xyz"[int(np.flatnonzero(pbc)[0])]
    settings = QESettings(
        calculation=specs["calculation"],
        spinorbit=specs["spinorbit"],
        kpoint_density=specs["kpoint_density"],
        ecutwfc=specs["ecutwfc"],
        ecutrho=ecutrho,
        occupations=values["occupations"],
        degauss=float(values["degauss"]),
        cell_dofree=dofree,
        electronic=electronic,
        pseudopotentials=pseudos,
        extra=advanced_overrides(raw_values, "qe")[0] or None,
    )
    return infer_qe_settings(atoms, base=settings)


def validate_calculation_report(atoms: Atoms, raw_values: dict[str, Any]):
    """The physics report for the requested calculation, with its fixes.

    Built from exactly the settings :func:`export_structure` will write.
    """
    from ..calculations.kpaths import suggest_band_path
    from ..validation.calculations import check_full_setup
    from ..validation.checks import ValidationReport

    specs = build_calculation_specs(raw_values)
    try:
        tagged, electronic, notes = electronic_setup(atoms, raw_values)
    except ValueError as exc:
        report = ValidationReport()
        report.errors.append(str(exc))
        return report
    settings = qe_settings_for(tagged, raw_values, electronic)
    band_path = (
        suggest_band_path(atoms, npoints_per_segment=specs["band_npoints"])
        if specs["task"] == "bandas"
        else None
    )
    # An advanced override of a checked setting is what gets written, so it
    # is what gets checked.
    extra = {k.lower(): v for k, v in (settings.extra or {}).items()}
    report = check_full_setup(
        tagged,
        calculation=specs["calculation"],
        occupations=str(extra.get("system.occupations", settings.occupations)),
        cell_dofree=extra.get("cell.cell_dofree", settings.cell_dofree),
        spectroscopy=specs["spectroscopy"],
        spinorbit=specs["spinorbit"],
        band_path=band_path,
        pseudopotentials=settings.pseudopotentials,
        electronic=electronic,
    )
    for k, note in enumerate(notes):
        report.info[f"decisión {k + 1}"] = note
    for code in ("qe", "siesta"):
        report.merge(advanced_overrides(raw_values, code)[1])
    return report


def collect_fixes(atoms: Optional[Atoms], raw_values: dict[str, Any]) -> list:
    """Every fixable problem with the current form, for the GUI's fix panel.

    Returns a list of ``(severity, message, Fix)``: parameter constraints
    first, then the physics of the calculation. Problems without a known
    settings cure are left out here (they still appear in the reports).
    """
    from .constraints import check_constraints

    out = [(v.severity, v.message, v.fix) for v in check_constraints(raw_values, atoms)
           if v.fix is not None]
    if atoms is not None and str(raw_values.get("preset", "ninguna")) == "ninguna":
        report = validate_calculation_report(atoms, raw_values)
        messages = report.errors + report.warnings
        for fix in report.fixes:
            severity = "error" if report.errors else "warning"
            out.append((severity, messages[0] if messages else fix.label, fix))
    # The same cure can be proposed by a constraint and by the physics check.
    unique, seen = [], set()
    for severity, message, fix in out:
        if (fix.setting, fix.value) not in seen:
            seen.add((fix.setting, fix.value))
            unique.append((severity, message, fix))
    return unique


def validate_calculation(atoms: Atoms, raw_values: dict[str, Any]) -> str:
    """Return a human-readable physics report for the requested calculation.

    This is what tells the user, *before* they queue anything, that Raman on
    an armchair nanotube with PAW pseudopotentials cannot work -- and, where
    the cure is a setting, which one (:func:`validate_calculation_report`
    carries them as structured fixes the GUI can apply).
    """
    report = validate_calculation_report(atoms, raw_values)
    if report.ok and not report.warnings:
        return "✅ El cálculo solicitado no presenta problemas conocidos."
    return report.summary()


def apply_functionalization(atoms: Atoms, raw_values: dict[str, Any]) -> Atoms:
    """Apply nitrogen configurations, edge passivation and functional groups.

    Order matters and mirrors the CLI: lattice changes first (nitrogen
    configurations create vacancies), then passivation, then attached groups.
    Attaching first would decorate carbons that a later vacancy removes.
    """
    from ..functionalization import (
        functionalize_bridges,
        make_graphitic_n,
        make_pyridinic_n,
        make_pyridinic_n_oxide,
        make_pyrrolic_like,
        passivate_edges,
    )

    values = collect_values(FUNCTIONALIZATION_PARAMS, raw_values)
    seed = int(collect_values(MODIFIER_PARAMS, raw_values)["seed"])
    out = atoms

    nitrogen = values["nitrogen"]
    if nitrogen != "ninguno":
        count = int(values["nitrogen_count"])
        if nitrogen == "graphitic":
            out = make_graphitic_n(out, n_sites=count, seed=seed)
        elif nitrogen == "pyridinic":
            out = make_pyridinic_n(out, n_defects=count, seed=seed)
        elif nitrogen == "pyrrolic":
            out = make_pyrrolic_like(out, n_defects=count, seed=seed)
        else:
            out = make_pyridinic_n_oxide(out, n_defects=count, seed=seed)

    if values["passivate"]:
        out = passivate_edges(out)

    group = values["group"]
    if group != "ninguno":
        from ..functionalization import functionalize_at_sites
        from ..placement import parse_indices

        count = int(values["group_count"])
        if group == "epoxy":
            out = functionalize_bridges(out, n_groups=count, seed=seed)
        else:
            indices = parse_indices(values["group_indices"])
            avoid = float(values["group_avoid"])
            out = functionalize_at_sites(
                out, group, count=count, region=values["group_site"],
                indices=indices or None, seed=seed, avoid_radius=avoid,
                avoid_occupied=avoid > 0, face=values["group_face"],
            )
    return out


def export_structure(
    atoms: Atoms,
    outdir: str | Path,
    formats: Sequence[str],
    calculation: str = "scf",
    force: bool = False,
    calculation_values: Optional[dict[str, Any]] = None,
) -> list[Path]:
    """Write the structure in every requested format.

    Parameters
    ----------
    atoms
        Structure to export.
    outdir
        Destination directory (created if missing).
    formats
        Any of ``"qe"``, ``"lammps"``, ``"xyz"``, ``"cif"``.
    calculation
        QE calculation type, ignored by the other writers.
    force
        Bypass validation errors. The GUI exposes this as a checkbox and
        warns the user, mirroring the library-level behaviour.

    Returns
    -------
    list[pathlib.Path]
        Every file written.
    """
    from ase.io import write as ase_write

    from ..exports.lammps import write_lammps
    from ..exports.qe import (
        QESettings,
        write_qe_bands,
        write_qe_input,
        write_qe_spectroscopy,
    )
    from ..exports.siesta import SiestaSettings, write_siesta

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if calculation_values is not None:
        preset_key = str(calculation_values.get("preset", "ninguna"))
        if preset_key != "ninguna":
            from ..workflows.pipeline import write_preset_project

            _, _, files = write_preset_project(
                atoms, outdir, preset_key,
                accurate=bool(calculation_values.get("accurate", False)),
                n_cores=int(calculation_values.get("cores", 8)),
                force=force,
            )
            return list(files.values())

    specs = (
        build_calculation_specs(calculation_values)
        if calculation_values is not None
        else {
            "task": calculation,
            "calculation": calculation,
            "spectroscopy": None,
            "spinorbit": None,
            "kpoint_density": 0.20,
            "ecutwfc": 60.0,
            "band_npoints": 30,
        }
    )

    if calculation_values is not None:
        # The same electronic choices the validation saw (spin, functional,
        # vdW), applied to the structure that gets written.
        atoms, electronic, _ = electronic_setup(atoms, calculation_values)

    siesta_extra: dict[str, Any] = {}
    if calculation_values is not None:
        for code in ("qe", "siesta"):
            if code not in formats:
                continue
            typed, advanced_report = advanced_overrides(calculation_values, code)
            if advanced_report.errors and not force:
                raise ValueError("Parámetros avanzados inválidos:\n"
                                 + "\n".join(advanced_report.errors))
            if code == "siesta":
                siesta_extra = typed

    def _qe_settings() -> QESettings:
        if calculation_values is not None:
            return qe_settings_for(atoms, calculation_values, electronic)
        return QESettings(
            calculation=specs["calculation"],
            spinorbit=specs["spinorbit"],
            kpoint_density=specs["kpoint_density"],
            ecutwfc=specs["ecutwfc"],
            ecutrho=specs["ecutwfc"] * 8.0,
        )

    for fmt in formats:
        if fmt == "qe":
            qe_dir = outdir / "qe"
            if specs["task"] == "bandas":
                written.extend(
                    write_qe_bands(
                        atoms, qe_dir, settings=_qe_settings(),
                        npoints_per_segment=specs["band_npoints"],
                        force=force,
                    ).values()
                )
            elif specs["spectroscopy"] is not None:
                written.extend(
                    write_qe_spectroscopy(
                        atoms, qe_dir, specs["spectroscopy"],
                        settings=_qe_settings(), force=force,
                    ).values()
                )
            else:
                written.append(
                    write_qe_input(atoms, qe_dir, settings=_qe_settings(),
                                   force=force)
                )
        elif fmt == "siesta":
            run_type = {
                "bandas": "bands",
                "fonones": "phonon",
                "infrarrojo": "phonon",
                "raman": "phonon",
            }.get(specs["task"], specs["calculation"])
            written.append(
                write_siesta(
                    atoms, outdir / "siesta",
                    settings=SiestaSettings(
                        run_type=run_type,
                        spinorbit=specs["spinorbit"],
                        kpoint_density=specs["kpoint_density"],
                        extra=siesta_extra or None,
                    ),
                    spectroscopy=specs["spectroscopy"],
                    force=force,
                )
            )
        elif fmt == "lammps":
            data, inp = write_lammps(atoms, outdir / "lammps", force=force)
            written.extend([data, inp])
        elif fmt == "xyz":
            path = outdir / "structure.xyz"
            ase_write(path, atoms, format="extxyz")
            written.append(path)
        elif fmt == "cif":
            path = outdir / "structure.cif"
            ase_write(path, atoms, format="cif")
            written.append(path)
        else:
            raise ValueError(f"Formato de exportación desconocido: '{fmt}'.")
    # Each engine directory describes its own run (job.json), so the window's
    # queue, a terminal and a cluster all run it the same way.
    from ..jobs.manifest import write_manifest

    for fmt in formats:
        if fmt in ("qe", "siesta", "lammps") and (outdir / fmt).is_dir():
            written.append(write_manifest(outdir / fmt))
    return written


def describe_structure(atoms: Atoms) -> str:
    """Return a short multi-line human summary shown under the 3D preview."""
    from ..topology.graph import coordination_numbers
    from ..validation.checks import run_basic_checks

    report = run_basic_checks(atoms)
    coord = coordination_numbers(atoms)
    pbc = atoms.get_pbc()
    dim = int(sum(pbc))
    ejes = "".join(eje for eje, on in zip("xyz", pbc) if on) or "ninguno"

    lines = [
        f"Fórmula: {atoms.get_chemical_formula()}   ({len(atoms)} átomos)",
        f"Dimensionalidad: {dim}D   (periódico en: {ejes})",
        f"Coordinación media: {coord.mean():.3f}",
    ]
    densidad = report.info.get("density_g_cm3")
    if isinstance(densidad, float) and densidad == densidad:  # descarta NaN
        lines.append(f"Densidad: {densidad:.3f} g/cm³")
    dmin = report.info.get("min_interatomic_distance")
    if isinstance(dmin, float):
        lines.append(f"Distancia mínima: {dmin:.3f} Å")

    if report.ok:
        lines.append("")
        lines.append("✅ Validación superada.")
    else:
        lines.append("")
        lines.append("❌ Validación fallida:")
        lines.extend(f"   • {e}" for e in report.errors)
    if report.warnings:
        lines.append("⚠️  Advertencias:")
        lines.extend(f"   • {w}" for w in report.warnings)
    return "\n".join(lines)
