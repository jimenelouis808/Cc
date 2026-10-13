"""Fitted spectra in and out, as tables.

What leaves a fitting program is not the fit, it is a table: the binding
energy, the measured counts, the background, the envelope and one column per
component. Everything anybody does afterwards — replot it in their own
style, compare two samples, hand it to a co-author, put it in the
supplementary information — is done on that table, and a program that
cannot write one keeps its results hostage.

Two tables, because they answer different questions:

:func:`fit_table`
    The curves. One row per channel, one column per component. This is what
    gets plotted.
:func:`components_table`
    The parameters. One row per component: state, binding energy, width,
    area, percentage, and the literature window it was held in. This is
    what goes in the paper.

And the way back: :func:`read_components` turns a components table into a
model. That closes the loop that matters for a series of samples — fit the
first one carefully, export it, and apply the same model to the other
eleven, so that what differs between them is the sample rather than the
operator's choices on a Tuesday.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..dataio.export import Table
from .elements import Doublet, XPSDatabase, load_xps_database
from .fitting import XPSComponent, XPSFitResult
from .spectrum import XPSError


def reference_table(
    region: str,
    database: Optional[XPSDatabase] = None,
    present: Optional[Sequence[str]] = None,
) -> Table:
    """The catalogue for one region, as a table anyone can read.

    The constraints a fit runs under are not an implementation detail:
    they decide what the answer can be. A window of 288.0-289.7 eV on the
    O-C=O family is the reason a component came out at 289.7 and not at
    290.4, and somebody reading the result has to be able to see that
    number without opening the source. Section 65 of the user's
    specification: every constraint inspectable.

    ``present`` marks which states the sample's composition allows,
    rather than hiding the rest: what was ruled out, and why, is part of
    the answer.
    """
    database = database or load_xps_database()
    states = database.states_for(region, include_satellites=True)
    if not states:
        raise XPSError(
            f"la base de datos no tiene estados para {region!r}; tiene: "
            + ", ".join(database.region_names()))
    default = set(database.default_model(region))

    columns = ["estado", "asignación", "BE", "BE_min", "BE_max", "FWHM_min",
               "FWHM_max", "confianza", "evidencia", "familia", "uso",
               "necesita", "posible", "nota", "referencia"]
    units = ["", "", "eV", "eV", "eV", "eV", "eV", "", "A-D", "", "", "", "",
             "", ""]
    rows = []
    for state in states:
        rows.append([
            state.key,
            state.name,
            round(float(state.energy_ev), 2),
            round(float(state.window[0]), 2),
            round(float(state.window[1]), 2),
            round(float(state.fwhm[0]), 2),
            round(float(state.fwhm[1]), 2),
            state.confidence,
            state.evidence_level,
            state.family,
            ("por defecto" if state.key in default
             else "satélite" if state.is_satellite
             else state.fitting_priority),
            " o ".join(state.requires_any),
            ("sí" if state.possible_in(present) else "no"),
            state.note,
            state.reference or state.source,
        ])
    return Table(columns=columns, rows=rows, units=units,
                 title=f"Catálogo de estados químicos — {region}")


#: The two consistent ways of exporting a fit, and there are only two.
#: ``"incluido"`` leaves the measured counts as measured and RAISES every
#: component onto the background, so every curve in the file shares the
#: measured baseline and the file plots straight over the data.
#: ``"restado"`` lowers the data instead: counts minus background, and the
#: components as they were fitted, so the envelope is exactly their sum.
#: What is not allowed is the mixture — raw data beside subtracted
#: components — because then the file holds two baselines and nothing that
#: reads it can tell which curve is on which.
BACKGROUND_EXPORT = ("incluido", "restado")


def _check_background_choice(choice: str) -> str:
    if choice not in BACKGROUND_EXPORT:
        raise XPSError(
            f"el fondo se exporta «incluido» o «restado», no {choice!r}")
    return choice


def fit_table(result: XPSFitResult, digits: int = 6,
              background: str = "incluido",
              separator: str = ",", decimal: str = ".") -> Table:
    """The fitted curves, one column per component.

    Parameters
    ----------
    background:
        One of :data:`BACKGROUND_EXPORT`. This is the fix for the one
        thing that made the old export unusable without editing it by
        hand: the data came out WITH the background and the components
        came out without it, so the two could not be drawn on the same
        axes and the columns did not add up to anything.
    separator, decimal:
        The delimiter and the decimal mark. A spreadsheet whose locale
        expects the comma as the decimal mark puts a comma-delimited file
        of dotted numbers in a single column — which is what «mal
        acomodadas» looks like — so both are a choice here.

    Notes
    -----
    Either way the table closes on itself, and which identity holds is
    stated in the notes: with the background included, ``envolvente``
    equals any one component column plus the others minus the background
    counted once per extra component; with it subtracted, ``envolvente``
    is the plain sum of the component columns.
    """
    _check_background_choice(background)
    unit = result_unit(result)
    floor = result.background.values
    subtract = background == "restado"
    columns = ["energia_enlace_eV", "cuentas", "fondo", "envolvente", "residuo"]
    units = ["eV", unit, unit, unit, unit]
    data = [
        result.energy,
        result.counts - floor if subtract else result.counts,
        floor,
        result.fitted - floor if subtract else result.fitted,
        result.residual,
    ]
    for component in result.components:
        columns.append(component.name)
        units.append(unit)
        curve = component.curve(result.energy)
        data.append(curve if subtract else curve + floor)

    rows = [[float(column[index]) for column in data]
            for index in range(result.energy.size)]
    notes = [
        f"región {result.region_label}, ventana "
        f"{result.window[0]:.2f}–{result.window[1]:.2f} eV",
        result.background.describe(),
        f"R² = {result.r_squared:.5f}, χ²_red = {result.reduced_chi2:.3f}"
        + (f", DW = {result.durbin_watson:.2f}" if result.durbin_watson else ""),
    ]
    if subtract:
        notes.append(
            "fondo RESTADO: «cuentas» y «envolvente» son la medida y el "
            "ajuste menos el fondo, y cada componente es ella sola, así que "
            "la envolvente es la suma exacta de las columnas de componente. "
            "La columna «fondo» se deja para poder deshacerlo"
        )
    else:
        notes.append(
            "fondo INCLUIDO: «cuentas» es la medida tal cual y cada columna "
            "de componente va SOBRE el fondo, para que todas las curvas se "
            "dibujen en la misma línea base. Sumarlas cuenta el fondo una "
            "vez de más por cada componente"
        )
    notes.append(
        "cada columna de componente incluye su pareja de espín-órbita")
    if result.links:
        notes.append("ligaduras: " + "; ".join(result.links))
    notes.extend(result.warnings)
    return Table(columns=columns, rows=rows, units=units,
                 title=f"Ajuste XPS {result.region_label}", notes=notes,
                 digits=digits, separator=separator, decimal=decimal)


def result_unit(result: XPSFitResult) -> str:
    """The intensity unit of a fit, for the table's units row."""
    return str(result.acquisition.get("unidad", "cuentas"))


def components_table(result: XPSFitResult, digits: int = 5,
                     separator: str = ",", decimal: str = ".") -> Table:
    """One row per component: what it is, where it is, and how much of it.

    The literature window each component was held in travels with it. That
    is the difference between a table somebody can check and a table
    somebody has to believe: a centre of 400.2 eV means nothing on its own
    and means a great deal beside "pyrrolic, 399.8–400.8 eV, confidence
    high".
    """
    columns = [
        "componente", "estado", "elemento", "perfil",
        "energia_enlace_eV", "error_eV", "FWHM_eV", "area", "porcentaje",
        "doblete_eV", "razon_doblete", "satelite", "justificacion",
    ]
    units = ["", "", "", "", "eV", "eV", "eV", f"{result_unit(result)}·eV",
             "%", "eV", "", "", ""]
    rows = []
    for component in result.components:
        rows.append([
            component.label,
            component.state or "",
            component.element or "",
            component.profile,
            component.centre,
            component.errors.get("centre", float("nan")),
            component.true_fwhm,
            component.area,
            100.0 * component.area_fraction,
            component.doublet.splitting_ev if component.doublet else "",
            component.doublet.ratio if component.doublet else "",
            "sí" if component.satellite else "no",
            component.justification,
        ])
    notes = [
        f"región {result.region_label}, ventana "
        f"{result.window[0]:.2f}–{result.window[1]:.2f} eV, fondo "
        f"{result.background.kind}",
        "las áreas son de la ventana ajustada y con ESE fondo: cambiar los "
        "extremos las cambia",
        "el porcentaje es del total ajustado en esta región, no de la muestra",
        "las áreas son por encima del fondo, se exporte éste incluido o "
        "restado: restarlo mueve las curvas, no las áreas",
    ]
    if result.links:
        notes.append("ligaduras: " + "; ".join(result.links))
    notes.extend(result.warnings)
    return Table(columns=columns, rows=rows, units=units,
                 title=f"Componentes XPS {result.region_label}", notes=notes,
                 digits=digits, separator=separator, decimal=decimal)


def write_fit(result: XPSFitResult, path: str | Path,
              digits: int = 6, background: str = "incluido",
              separator: str = ",", decimal: str = ".") -> tuple[Path, Path]:
    """Write both tables beside each other.

    ``path`` names the curve table; the component table gets
    ``_componentes`` before the extension. Two files rather than one
    because they have different shapes, and a single file holding both is
    a file no spreadsheet opens correctly.

    ``background``, ``separator`` and ``decimal`` go to both tables, so a
    pair of files written together cannot disagree about either the
    baseline or the number format.
    """
    path = Path(path)
    curves = fit_table(result, digits=digits, background=background,
                       separator=separator, decimal=decimal)
    parameters = components_table(result, separator=separator,
                                  decimal=decimal)
    second = path.with_name(f"{path.stem}_componentes{path.suffix}")
    curves.save(path)
    parameters.save(second)
    return path, second


def read_components(
    path: str | Path,
    database: Optional[XPSDatabase] = None,
    region: str = "",
) -> tuple[list[XPSComponent], str]:
    """Rebuild a model's components from a components table.

    Reads back what :func:`components_table` wrote, so a model refined on
    one sample can be applied unchanged to the rest of a series. The
    parameters come back as **starting values**, not as fixed ones: the
    point is that every sample gets the same model, not that every sample
    gets the same answer.

    Returns
    -------
    tuple
        ``(components, region label)``.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    # A note containing a comma is written as a quoted cell, so the comment
    # marker ends up inside the quotes: `"# región N 1s, ventana ..."`.
    # Stripping the quote before looking for the marker is what keeps a
    # comment from being read as the header row.
    lines = [line for line in text.splitlines()
             if line.strip() and not line.strip().lstrip('"').startswith("#")]
    if len(lines) < 2:
        raise XPSError(f"{path.name}: no hay filas de datos en la tabla")
    delimiter = "\t" if "\t" in lines[0] else ","
    header = [cell.strip().strip('"') for cell in lines[0].split(delimiter)]
    required = {"componente", "energia_enlace_eV", "FWHM_eV"}
    missing = required - set(header)
    if missing:
        raise XPSError(
            f"{path.name}: a la tabla le faltan columnas ({', '.join(sorted(missing))}). "
            "Tiene que ser una tabla de componentes escrita por este programa"
        )
    index = {name: position for position, name in enumerate(header)}

    database = database or load_xps_database()
    components: list[XPSComponent] = []
    label = region
    for line in lines[1:]:
        cells = [cell.strip().strip('"') for cell in line.split(delimiter)]
        if len(cells) < len(header):
            continue
        name = cells[index["componente"]]
        if not name or name in ("eV", ""):      # the units row
            continue
        try:
            centre = float(cells[index["energia_enlace_eV"]])
            width = float(cells[index["FWHM_eV"]])
        except ValueError:
            continue                            # the units row, or a note
        state = cells[index["estado"]] if "estado" in index else ""
        element = cells[index["elemento"]] if "elemento" in index else ""
        profile = cells[index["perfil"]] if "perfil" in index else "gl"
        doublet = None
        if "doblete_eV" in index and cells[index["doblete_eV"]]:
            try:
                doublet = Doublet(
                    splitting_ev=float(cells[index["doblete_eV"]]),
                    ratio=float(cells[index["razon_doblete"]]),
                    labels=(name, f"{name} (pareja)"),
                )
            except (ValueError, KeyError):
                doublet = None
        if state and not label:
            for candidate in database.region_names():
                if any(item.key == state for item in database.states_for(candidate)):
                    label = candidate
                    break
        window = None
        if state and label:
            try:
                window = database.state(label, state).window
            except Exception:                   # pragma: no cover - unknown key
                window = None
        components.append(
            XPSComponent(
                name="_".join(name.split()) or f"C{len(components) + 1}",
                label=name,
                centre=centre,
                height=1.0,
                fwhm=max(width, 0.3),
                profile=profile or "gl",
                centre_bounds=window or (centre - 1.0, centre + 1.0),
                doublet=doublet,
                state=state or None,
                element=element or None,
                line=label or None,
                satellite=(cells[index["satelite"]].lower().startswith("s")
                           if "satelite" in index else False),
                justification=(cells[index["justificacion"]]
                               if "justificacion" in index else ""),
            )
        )
    if not components:
        raise XPSError(f"{path.name}: la tabla no contiene ninguna componente")
    return components, label


def heights_from(spectrum, components: Sequence[XPSComponent],
                 window: tuple[float, float], background: str = "shirley"):
    """Re-seed imported components' heights from a new spectrum.

    A model imported from another sample carries that sample's intensities,
    which are meaningless here. The positions and widths are the model; the
    heights have to come from the spectrum in front of you.
    """
    from .background import estimate_background

    estimate = estimate_background(spectrum, window, background)
    axis, counts = spectrum.region_of(*window)
    above = counts - estimate.values
    for component in components:
        component.height = max(
            float(np.interp(component.centre, axis, above)), 1.0)
    return components


__all__ = [
    "BACKGROUND_EXPORT",
    "reference_table",
    "components_table",
    "fit_table",
    "heights_from",
    "read_components",
    "result_unit",
    "write_fit",
]
