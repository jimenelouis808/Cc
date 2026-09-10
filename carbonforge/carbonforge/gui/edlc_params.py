"""Parameter specs and logic behind the GUI's EDLC tab.

Like :mod:`carbonforge.gui.params`, this module contains **no Tk code**. The
tab reads :data:`EDLC_PARAMS` to lay out its widgets and calls the functions
here to do the work, so every non-cosmetic path is unit-testable headless.

The point of the tab is that an EDLC cell has more ways to be silently wrong
than a structure does. A separation that is too narrow, a potential past the
electrolyte's stability window, or a cross-section too small to hold a
statistically meaningful double layer all produce a simulation that runs
happily and reports a capacitance that means nothing.
:func:`check_edlc_constraints` states those before anything is built, and
:func:`describe_edlc` repeats the checks on the finished cell.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ase import Atoms

from ..forcefields import EDLCCell, build_edlc_cell, check_edlc_setup
from .params import ParamSpec, collect_values

#: Electrochemical stability windows, in volts. Past these the electrolyte
#: decomposes — water electrolyses at 1.23 V — and classical MD, which has no
#: bond breaking, keeps running and returns a number anyway.
STABILITY_WINDOW = {"aqueous": 1.23, "ionic_liquid": 4.0}

#: Minimum electrode separation, in Å, for the two double layers not to
#: overlap. Aqueous double layers are a few Å thick; ionic liquids order into
#: layers that reach much further from the surface.
MIN_SEPARATION = {"aqueous": 30.0, "ionic_liquid": 45.0}

EDLC_PARAMS: tuple[ParamSpec, ...] = (
    ParamSpec(
        "electrolyte", "Electrolito", "choice", "aqueous",
        choices=("aqueous", "ionic_liquid", "vacuum"),
        help=(
            "aqueous = agua SPC/E con sal disuelta. ionic_liquid = BMIM-PF6 "
            "de grano grueso (un sitio por ion). vacuum = solo los "
            "electrodos, para comprobar el montaje."
        ),
    ),
    ParamSpec(
        "separation", "Separación entre electrodos (Å)", "float", 40.0,
        minimum=8.0, maximum=200.0,
        help=(
            "Hueco entre las dos superficies. Debe ser mayor que el doble "
            "del grosor de la doble capa: >=30 Å en agua, >=45 Å con "
            "líquido iónico."
        ),
    ),
    ParamSpec(
        "potential", "Potencial aplicado (V)", "float", 1.0,
        minimum=0.0, maximum=10.0,
        help=(
            "Diferencia entre electrodos, aplicada como ±V/2 para que la "
            "celda quede neutra. El agua se electroliza hacia 1.23 V."
        ),
    ),
    ParamSpec(
        "salt", "Sal disuelta", "choice", "NaCl",
        choices=("NaCl", "KCl", "LiCl"),
        help="Solo se usa con electrolito acuoso.",
    ),
    ParamSpec(
        "molarity", "Molaridad (mol/L)", "float", 1.0,
        minimum=0.0, maximum=6.0,
        help=(
            "Concentración de sal. 1 M es el valor habitual de referencia; "
            "por encima de ~5 M no cabe el agua que la disolvería."
        ),
    ),
    ParamSpec(
        "temperature", "Temperatura (K)", "float", 298.0,
        minimum=100.0, maximum=600.0,
        help="298 K = ambiente. El agua SPC/E hierve muy por encima de 373 K.",
    ),
    ParamSpec(
        "wall_gap", "Hueco electrodo-electrolito (Å)", "float", 3.0,
        minimum=1.5, maximum=8.0,
        help=(
            "Espacio libre dejado sobre cada superficie al rellenar. 3 Å es "
            "donde se asienta la primera capa de agua sobre grafeno, así que "
            "la configuración inicial arranca cerca del sitio correcto."
        ),
    ),
    ParamSpec(
        "equilibration", "Pasos de equilibrado", "int", 500000,
        minimum=1000, maximum=100_000_000,
        help=(
            "Con paso de 1 fs, 500 000 pasos = 500 ps. La doble capa tarda "
            "cientos de ps en formarse; los líquidos iónicos, mucho más."
        ),
    ),
    ParamSpec(
        "production", "Pasos de producción", "int", 2000000,
        minimum=1000, maximum=100_000_000,
        help=(
            "Solo esta etapa se promedia. Debe ser bastante más larga que el "
            "equilibrado para que la media tenga poco ruido."
        ),
    ),
    ParamSpec(
        "seed", "Semilla", "int", 0,
        minimum=0, maximum=10_000_000,
        help="Fija el relleno inicial: la misma semilla da la misma celda.",
    ),
    ParamSpec(
        "mirror_top", "Reflejar electrodo superior", "bool", True,
        help=(
            "Refleja el electrodo de arriba para que las dos caras "
            "funcionalizadas miren al electrolito. Desactívalo para una "
            "celda asimétrica."
        ),
    ),
)


def check_edlc_constraints(electrode: Atoms, raw_values: dict[str, Any]) -> str:
    """Report what is wrong with the requested cell *before* building it.

    Building an EDLC cell packs thousands of molecules, which is slow enough
    that finding out afterwards is a poor trade. This runs the cheap checks
    on the parameters and the electrode alone.

    Parameters
    ----------
    electrode
        Structure that would be used as the electrode.
    raw_values
        Raw widget values, coerced here against :data:`EDLC_PARAMS`.

    Returns
    -------
    str
        A multi-line, user-facing Spanish report. Errors are prefixed ``❌``
        and warnings ``⚠️``; when nothing is wrong it says so.
    """
    values = collect_values(EDLC_PARAMS, raw_values)
    kind = values["electrolyte"]
    errors: list[str] = []
    warnings: list[str] = []

    pbc = electrode.get_pbc()
    if not (pbc[0] and pbc[1]):
        ejes = "".join(e for e, on in zip("xyz", pbc) if on) or "ninguno"
        errors.append(
            f"El electrodo es periódico en '{ejes}' y hace falta que lo sea "
            "en x e y: tiene que teselar la sección transversal. Usa una "
            "lámina de grafeno, no un nanotubo ni un fragmento finito."
        )

    separation = float(values["separation"])
    wall_gap = float(values["wall_gap"])
    if separation <= 2 * wall_gap:
        errors.append(
            f"Separación de {separation:.0f} Å con {wall_gap:.1f} Å de hueco "
            "a cada pared no deja sitio para el electrolito."
        )
    elif kind != "vacuum":
        floor = MIN_SEPARATION[kind]
        if separation < floor:
            warnings.append(
                f"Separación de {separation:.0f} Å: por debajo de {floor:.0f} Å "
                "las dos dobles capas se solapan y la capacitancia deja de "
                "ser la de una interfaz aislada."
            )

    potential = float(values["potential"])
    if kind != "vacuum":
        window = STABILITY_WINDOW[kind]
        if potential > window:
            warnings.append(
                f"{potential:.2f} V supera la ventana de estabilidad del "
                f"electrolito ({window:.2f} V). La simulación clásica no "
                "rompe enlaces, así que no se descompondrá nada: seguirá "
                "corriendo y dará una capacitancia sin sentido físico."
            )

    if kind == "aqueous" and float(values["molarity"]) > 5.0:
        warnings.append(
            f"{values['molarity']:.1f} M está por encima de la solubilidad de "
            "cualquiera de estas sales. Cabrán los iones, pero no es una "
            "disolución que exista."
        )

    if kind != "vacuum":
        area = _cross_section_area(electrode)
        if area < 200.0:
            warnings.append(
                f"Sección transversal de {area:.0f} Å²: muy estrecha para "
                "promediar una doble capa. Amplía la supercelda del "
                "electrodo — es lo que más reduce el ruido."
            )

    if int(values["production"]) <= int(values["equilibration"]):
        warnings.append(
            "La producción no es más larga que el equilibrado. Solo se "
            "promedia la producción, así que estás gastando la mayor parte "
            "del tiempo en una etapa que se descarta."
        )

    if not errors and not warnings:
        return "✅ Los parámetros no presentan problemas conocidos."

    lines: list[str] = []
    if errors:
        lines.append("❌ Errores (impiden construir la celda):")
        lines.extend(f"   • {e}" for e in errors)
    if warnings:
        if lines:
            lines.append("")
        lines.append("⚠️  Advertencias (se puede construir, pero léelas):")
        lines.extend(f"   • {w}" for w in warnings)
    return "\n".join(lines)


def _cross_section_area(atoms: Atoms) -> float:
    """In-plane area of the cell, in Å².

    The z component of a1 x a2, written out: numpy 2 deprecated the
    two-dimensional cross product, and the determinant is what it meant
    anyway.
    """
    import numpy as np

    cell = np.array(atoms.cell)
    return float(abs(cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0]))


def estimate_size(electrode: Atoms, raw_values: dict[str, Any]) -> str:
    """Estimate how big the cell will be, before paying to build it.

    Packing is the slow step and the count is what decides whether the run is
    minutes or days, so it is worth showing up front.
    """
    import numpy as np

    values = collect_values(EDLC_PARAMS, raw_values)
    kind = values["electrolyte"]
    if kind == "vacuum":
        return f"Celda estimada: {2 * len(electrode)} átomos (sin electrolito)."

    from ..forcefields.electrolyte import DENSITIES, _MOLAR_MASS, _N_PER_A3

    area = _cross_section_area(electrode)
    thickness = float(values["separation"]) - 2 * float(values["wall_gap"])
    volume = area * max(thickness, 0.0)

    if kind == "aqueous":
        n_water = int(
            DENSITIES["water"] * _N_PER_A3 * volume / _MOLAR_MASS["water"] * 1000
        )
        n_pairs = int(float(values["molarity"]) * volume * 1e-27 * 6.02214e23)
        n_electrolyte = 3 * n_water + 2 * n_pairs
        detail = f"{n_water} H₂O + {n_pairs} pares iónicos"
    else:
        pair_mass = _MOLAR_MASS["BMIM"] + _MOLAR_MASS["PF6"]
        n_pairs = int(
            DENSITIES["ionic_liquid"] * _N_PER_A3 * volume / pair_mass * 1000
        )
        n_electrolyte = 2 * n_pairs
        detail = f"{n_pairs} pares iónicos (un sitio por ion)"

    total = 2 * len(electrode) + n_electrolyte
    return (
        f"Celda estimada: {total} átomos "
        f"({2 * len(electrode)} de electrodo + {n_electrolyte} de electrolito).\n"
        f"Electrolito: {detail}."
    )


def build_edlc(electrode: Atoms, raw_values: dict[str, Any]) -> EDLCCell:
    """Assemble the EDLC cell described by the GUI values.

    Raises
    ------
    ValueError
        With a user-facing Spanish message, when the electrode or the
        parameters make the cell impossible.
    """
    values = collect_values(EDLC_PARAMS, raw_values)
    kind = values["electrolyte"]

    electrolyte_kwargs: dict[str, Any] = {}
    if kind != "vacuum":
        electrolyte_kwargs["seed"] = int(values["seed"])
    if kind == "aqueous":
        electrolyte_kwargs["salt"] = values["salt"]
        electrolyte_kwargs["molarity"] = float(values["molarity"])

    return build_edlc_cell(
        electrode,
        separation=float(values["separation"]),
        electrolyte=kind,
        potential_v=float(values["potential"]),
        mirror_top=bool(values["mirror_top"]),
        wall_gap=float(values["wall_gap"]),
        electrolyte_kwargs=electrolyte_kwargs,
    )


def export_edlc(
    cell: EDLCCell,
    outdir: str | Path,
    raw_values: Optional[dict[str, Any]] = None,
) -> list[Path]:
    """Write the LAMMPS data file, input script and notes.

    Returns
    -------
    list[pathlib.Path]
        Every file written.
    """
    from ..exports.lammps_edlc import EDLCSettings, write_edlc

    values = collect_values(EDLC_PARAMS, raw_values or {})
    settings = EDLCSettings(
        temperature_k=float(values["temperature"]),
        equilibration_steps=int(values["equilibration"]),
        production_steps=int(values["production"]),
    )
    written = write_edlc(cell, Path(outdir), settings=settings)
    return list(written.values())


def describe_edlc(cell: EDLCCell) -> str:
    """Summarise a built cell and repeat the checks on the real geometry."""
    lines = [cell.summary()]
    warnings = check_edlc_setup(cell)
    lines.append("")
    if warnings:
        lines.append("⚠️  Revisa esto antes de lanzar el cálculo:")
        lines.extend(f"   • {w}" for w in warnings)
    else:
        lines.append("✅ El montaje no presenta problemas conocidos.")
    lines.append("")
    lines.append(
        "Para medir la capacitancia: promedia la carga del electrodo en "
        "electrode_charge.dat\nsobre la etapa de producción, y divide entre "
        "el potencial aplicado."
    )
    return "\n".join(lines)
