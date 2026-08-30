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
from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D, DEFAULT_VACUUM_2D

ParamKind = Literal["int", "float", "bool", "choice"]


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


#: What to compute, on top of the structure itself.
CALCULATION_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec("task", "Tipo de cálculo", "choice", "scf",
              choices=("scf", "relax", "vc-relax", "bandas",
                       "fonones", "infrarrojo", "raman"),
              help="'bandas' escribe scf+bands+bands.x; los espectroscópicos, scf+ph.x+dynmat.x."),
    ParamSpec("spinorbit", "Acoplamiento espín-órbita", "bool", False,
              help="Requiere pseudopotenciales relativistas. En carbono puro el efecto es ~0.01 meV."),
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
)


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
              choices=("edge", "basal"),
              help="'edge' es lo habitual. 'basal' fuerza sp3 y arruga la "
                   "lámina: así es el óxido de grafeno."),
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
              help="Fracción de carbonos sustituidos (0.05 = 5%)."),
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
    values = collect_values(MODIFIER_PARAMS, raw_values)
    seed = int(values["seed"])
    out = atoms

    dopant = values["dopant"]
    concentration = float(values["dopant_concentration"])
    if dopant != "ninguno" and concentration > 0:
        out = dope_random(out, dopant, concentration, seed=seed)

    n_vac = int(values["vacancies"])
    if n_vac > 0:
        out = introduce_vacancies(out, n_defects=n_vac, seed=seed)
    return out


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


def validate_calculation(atoms: Atoms, raw_values: dict[str, Any]) -> str:
    """Return a human-readable physics report for the requested calculation.

    This is what tells the user, *before* they queue anything, that Raman on
    an armchair nanotube with PAW pseudopotentials cannot work.
    """
    from ..exports.qe import QESettings, infer_qe_settings
    from ..validation.calculations import check_full_setup
    from ..calculations.kpaths import suggest_band_path

    specs = build_calculation_specs(raw_values)
    settings = infer_qe_settings(
        atoms,
        base=QESettings(
            calculation=specs["calculation"],
            spinorbit=specs["spinorbit"],
        ),
    )
    band_path = (
        suggest_band_path(atoms, npoints_per_segment=specs["band_npoints"])
        if specs["task"] == "bandas"
        else None
    )
    report = check_full_setup(
        atoms,
        calculation=specs["calculation"],
        spectroscopy=specs["spectroscopy"],
        spinorbit=specs["spinorbit"],
        band_path=band_path,
        pseudopotentials=settings.pseudopotentials,
    )
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
        functionalize_random,
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
        count = int(values["group_count"])
        if group == "epoxy":
            out = functionalize_bridges(out, n_groups=count, seed=seed)
        else:
            out = functionalize_random(
                out, group, n_groups=count,
                site_kind=values["group_site"], seed=seed,
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

    def _qe_settings() -> QESettings:
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
