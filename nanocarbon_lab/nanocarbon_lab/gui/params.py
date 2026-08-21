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
            "Cinta de grafeno con bordes definidos, periódica a lo largo de y."
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


def export_structure(
    atoms: Atoms,
    outdir: str | Path,
    formats: Sequence[str],
    calculation: str = "scf",
    force: bool = False,
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
    from ..exports.qe import QESettings, write_qe_input

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for fmt in formats:
        if fmt == "qe":
            written.append(
                write_qe_input(atoms, outdir / "qe",
                               settings=QESettings(calculation=calculation),
                               force=force)
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
