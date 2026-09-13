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


def fit_table(result: XPSFitResult, digits: int = 6) -> Table:
    """The fitted curves, one column per component.

    The numbers are the ones **drawn**: the background is the one that was
    subtracted and each component column is that component as fitted,
    spin–orbit partner included, on top of nothing. Adding every component
    column to the background column reproduces the envelope exactly, which
    is the property that makes the table checkable by whoever receives it.
    """
    columns = ["energia_enlace_eV", "cuentas", "fondo", "envolvente", "residuo"]
    units = ["eV", result_unit(result), result_unit(result),
             result_unit(result), result_unit(result)]
    data = [result.energy, result.counts, result.background.values,
            result.fitted, result.residual]
    for component in result.components:
        columns.append(component.name)
        units.append(result_unit(result))
        data.append(component.curve(result.energy))

    rows = [[float(column[index]) for column in data]
            for index in range(result.energy.size)]
    notes = [
        f"región {result.region_label}, ventana "
        f"{result.window[0]:.2f}–{result.window[1]:.2f} eV",
        result.background.describe(),
        f"R² = {result.r_squared:.5f}, χ²_red = {result.reduced_chi2:.3f}"
        + (f", DW = {result.durbin_watson:.2f}" if result.durbin_watson else ""),
        "la envolvente es fondo + todas las componentes; cada columna de "
        "componente incluye su pareja de espín-órbita",
    ]
    if result.links:
        notes.append("ligaduras: " + "; ".join(result.links))
    notes.extend(result.warnings)
    return Table(columns=columns, rows=rows, units=units,
                 title=f"Ajuste XPS {result.region_label}", notes=notes,
                 digits=digits)


def result_unit(result: XPSFitResult) -> str:
    """The intensity unit of a fit, for the table's units row."""
    return str(result.acquisition.get("unidad", "cuentas"))


def components_table(result: XPSFitResult, digits: int = 5) -> Table:
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
    ]
    if result.links:
        notes.append("ligaduras: " + "; ".join(result.links))
    notes.extend(result.warnings)
    return Table(columns=columns, rows=rows, units=units,
                 title=f"Componentes XPS {result.region_label}", notes=notes,
                 digits=digits)


def write_fit(result: XPSFitResult, path: str | Path,
              digits: int = 6) -> tuple[Path, Path]:
    """Write both tables beside each other.

    ``path`` names the curve table; the component table gets
    ``_componentes`` before the extension. Two files rather than one
    because they have different shapes, and a single file holding both is
    a file no spreadsheet opens correctly.
    """
    path = Path(path)
    curves = fit_table(result, digits=digits)
    parameters = components_table(result)
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
    "components_table",
    "fit_table",
    "heights_from",
    "read_components",
    "result_unit",
    "write_fit",
]
