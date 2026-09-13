"""Getting the numbers out, in the shapes people actually need them.

Four different consumers, four formats:

``components``
    One row per fitted band: position, height, width, area, shape,
    uncertainties, and the percentage of the total fitted area. This is
    the table that goes into a paper.
``curves``
    One column per component, on the fit's own abscissa, plus the data, the
    total and the background. This is what you paste into Origin or
    SigmaPlot to redraw the figure yourself, and it is the export people
    ask for first and find missing most often.
``indices``
    One row per spectrum with every ratio and index. This is the batch
    table.
``json``
    Everything, including the model definition, the bounds, the parameter
    correlations and the warnings — enough to reproduce the fit.

Every export carries a header naming the spectrum, the laser, the
preprocessing history and the model, because a table of areas with no
record of which baseline produced them is not a result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from ..models.fitting import FitResult

if TYPE_CHECKING:  # pragma: no cover
    from .report import AnalysisResult

#: Columns of the component table, in order.
COMPONENT_COLUMNS = (
    "componente",
    "banda",
    "perfil",
    "centro_cm-1",
    "err_centro",
    "posicion_max_cm-1",
    "altura",
    "err_altura",
    "altura_max",
    "FWHM_cm-1",
    "err_FWHM",
    "area",
    "area_pct",
    "eta_o_invq",
    "fijados",
)


def component_rows(fit: FitResult) -> list[dict[str, Any]]:
    """The fitted components as a list of plain dictionaries.

    ``area_pct`` is each component's share of the total fitted area, which
    is what most papers actually quote and what most people compute by hand
    afterwards.
    """
    total = fit.total_area() or 1.0
    rows: list[dict[str, Any]] = []
    for peak in fit.peaks:
        extra = peak.extra[0] if peak.extra else None
        rows.append(
            {
                "componente": peak.name,
                "banda": peak.band or "",
                "perfil": peak.profile,
                "centro_cm-1": peak.centre,
                "err_centro": peak.errors.get("centre"),
                "posicion_max_cm-1": peak.peak_position,
                "altura": peak.height,
                "err_altura": peak.errors.get("height"),
                "altura_max": peak.peak_height,
                "FWHM_cm-1": peak.fwhm,
                "err_FWHM": peak.errors.get("fwhm"),
                "area": peak.area,
                "area_pct": 100.0 * peak.area / total,
                "eta_o_invq": extra,
                "fijados": ";".join(peak.fixed),
            }
        )
    return rows


def _cell(value: Any, decimals: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not np.isfinite(value):
            return ""
        return f"{value:.{decimals}g}"
    text = str(value)
    if "," in text or '"' in text or "\n" in text:
        return '"' + text.replace('"', '""') + '"'
    return text


def _header(
    result: Optional["AnalysisResult"],
    fit: Optional[FitResult],
    what: str,
    comment: str = "#",
) -> list[str]:
    """Provenance block written at the top of every text export."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"{comment} ramancarbon — {what}",
        f"{comment} generado: {stamp}",
    ]
    if result is not None:
        laser = (
            f"{result.raw.laser_nm:g} nm" if result.raw.laser_nm else "desconocido"
        )
        lines += [
            f"{comment} espectro: {result.raw.name}",
            f"{comment} laser: {laser}",
            f"{comment} material: {result.classification.label} "
            f"(confianza {result.classification.confidence})",
            f"{comment} base de intensidades: {result.basis}",
            f"{comment} procesado: {' -> '.join(result.processed.history) or 'ninguno'}",
        ]
    if fit is not None:
        lines += [
            f"{comment} modelo: {len(fit.peaks)} componentes, "
            f"{fit.n_parameters} parametros",
            f"{comment} ventana: {fit.x[0]:.0f}-{fit.x[-1]:.0f} cm-1",
            f"{comment} R2 = {fit.r_squared:.6f}, chi2_red = {fit.reduced_chi2:.4g}",
        ]
        for warning in fit.warnings:
            lines.append(f"{comment} AVISO: {warning}")
    return lines


def export_components(
    fit: FitResult,
    path: str | Path,
    result: Optional["AnalysisResult"] = None,
    delimiter: str = ",",
) -> Path:
    """Write the fitted components as a CSV table.

    Parameters
    ----------
    fit:
        The deconvolution to export.
    path:
        Output file.
    result:
        The analysis it came from, used only for the provenance header.
    delimiter:
        Column separator. Use ``";"`` if your spreadsheet is set to a
        Spanish locale, where the comma is the decimal separator.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = _header(result, fit, "componentes del ajuste")
    lines.append(delimiter.join(COMPONENT_COLUMNS))
    for row in component_rows(fit):
        lines.append(delimiter.join(_cell(row[c]) for c in COMPONENT_COLUMNS))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def export_curves(
    fit: FitResult,
    path: str | Path,
    result: Optional["AnalysisResult"] = None,
    delimiter: str = ",",
    include_background: bool = True,
) -> Path:
    """Write the fitted curves point by point, one column per component.

    The columns are ``desplazamiento_cm-1``, ``datos``, ``ajuste_total``,
    ``residuo``, optionally ``fondo``, then one per component. Component
    curves **include** the background, so that plotting any of them against
    the data lines up without further arithmetic — the way they appear in
    the program's own figures.

    Parameters
    ----------
    fit:
        The deconvolution to export.
    path:
        Output file.
    result:
        The analysis it came from, for the header.
    delimiter:
        Column separator.
    include_background:
        Write the background column. Turn it off if you subtracted the
        background before fitting and it is identically zero.

    Returns
    -------
    pathlib.Path
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    x = fit.x
    columns: list[tuple[str, np.ndarray]] = [
        ("desplazamiento_cm-1", x),
        ("datos", fit.y),
        ("ajuste_total", fit.fitted),
        ("residuo", fit.residual),
    ]
    if include_background:
        columns.append(("fondo", fit.background))
    for peak in fit.peaks:
        columns.append((peak.name, peak.curve(x) + fit.background))

    lines = _header(result, fit, "curvas del ajuste")
    lines.append(
        "# las curvas de cada componente INCLUYEN el fondo, para que se puedan "
        "dibujar directamente sobre los datos"
    )
    lines.append(delimiter.join(name for name, _ in columns))
    for index in range(x.size):
        lines.append(delimiter.join(_cell(values[index]) for _, values in columns))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def export_json(
    result: "AnalysisResult",
    path: str | Path,
    include_curves: bool = False,
) -> Path:
    """Write the whole analysis as JSON.

    Everything a report shows plus the machine-readable detail behind it:
    the model definition, per-parameter uncertainties, the correlations
    that make some areas non-independent, and every warning.

    Parameters
    ----------
    result:
        The analysis to serialise.
    path:
        Output file.
    include_curves:
        Also embed the fitted curves. Off by default because it multiplies
        the file size by a hundred; use :func:`export_curves` for those.

    Returns
    -------
    pathlib.Path
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(analysis_to_dict(result, include_curves), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target


def analysis_to_dict(
    result: "AnalysisResult", include_curves: bool = False
) -> dict[str, Any]:
    """The full analysis as nested plain data, ready for JSON."""
    payload: dict[str, Any] = {
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "espectro": {
            "nombre": result.raw.name,
            "laser_nm": result.raw.laser_nm,
            "rango_cm-1": list(result.raw.range),
            "puntos": int(result.raw.shift.size),
            "procesado": list(result.processed.history),
            "metadatos": {
                k: v for k, v in result.raw.metadata.items() if isinstance(v, (str, int, float, bool))
            },
        },
        "identificacion": {
            "material": result.classification.best,
            "etiqueta": result.classification.label,
            "confianza": result.classification.confidence,
            "evidencia": [
                {"regla": e.rule, "peso": e.weight, "texto": e.statement}
                for e in result.classification.evidence
            ],
            "reglas_no_aplicables": list(result.classification.blocked_rules),
        },
        "bandas": {
            key: {
                "posicion_cm-1": entry.position,
                "esperada_cm-1": entry.expected_position,
                "desviacion_cm-1": entry.deviation,
                "altura": entry.height,
                "FWHM_cm-1": entry.fwhm,
                "area": entry.area,
                "origen": entry.origin,
            }
            for key, entry in result.assignment.bands.items()
        },
        "cocientes": {
            key: {
                "valor": ratio.value,
                "base": ratio.basis,
                "valor_otra_base": ratio.alternate,
                "disponible": ratio.available,
                "razon": ratio.reason,
            }
            for key, ratio in result.ratios.items()
        },
        "base_intensidad": result.basis,
        "avisos": list(result.warnings),
    }

    if result.fit is not None:
        payload["ajuste"] = {
            "modelo": result.comparison.best if result.comparison else "manual",
            "ventana_cm-1": [float(result.fit.x[0]), float(result.fit.x[-1])],
            "R2": result.fit.r_squared,
            "chi2_reducido": result.fit.reduced_chi2,
            "AIC": result.fit.aic,
            "BIC": result.fit.bic,
            "n_parametros": result.fit.n_parameters,
            "convergido": result.fit.success,
            "componentes": component_rows(result.fit),
            "correlaciones_altas": {
                f"{a} / {b}": value
                for (a, b), value in result.fit.correlations.items()
            },
            "avisos": list(result.fit.warnings),
        }
        if result.comparison:
            payload["ajuste"]["comparacion"] = {
                name: {
                    "BIC": fit.bic,
                    "AIC": fit.aic,
                    "R2": fit.r_squared,
                    "n_parametros": fit.n_parameters,
                }
                for name, fit in result.comparison.results.items()
            }
        if include_curves:
            payload["ajuste"]["curvas"] = {
                "desplazamiento_cm-1": result.fit.x.tolist(),
                "datos": result.fit.y.tolist(),
                "total": result.fit.fitted.tolist(),
                "fondo": result.fit.background.tolist(),
                "componentes": {
                    peak.name: (peak.curve(result.fit.x)).tolist()
                    for peak in result.fit.peaks
                },
            }

    if result.crystallite:
        payload["estructura"] = {
            "La_nm": result.crystallite.la_low_defect_nm,
            "LD_nm": result.crystallite.ld_low_defect_nm,
            "nD_cm-2": result.crystallite.defect_density_cm2,
            "rama": result.crystallite.likely_branch,
            "razon_rama": result.crystallite.branch_reason,
            "avisos": list(result.crystallite.warnings),
        }
    if result.indices is not None:
        payload["indices"] = result.indices.to_dict()
    if result.defects:
        payload["tipo_defecto"] = {
            "I_D/I_D'": result.defects.ratio,
            "tipo": result.defects.best_match,
            "concluyente": result.defects.confident,
        }
    if result.rbm.diameters:
        payload["diametros"] = [
            {
                "RBM_cm-1": e.input_value,
                "d_nm": e.diameter_nm,
                "incertidumbre_nm": e.uncertainty_nm,
                "parametrizacion": e.parameterisation,
            }
            for e in result.rbm.diameters
        ]
    if result.shifts:
        payload["desplazamientos"] = {
            "referencia": result.shifts.reference_material,
            "tipo_referencia": result.shifts.reference_kind,
            "bandas": {
                key: {
                    "medida": shift.measured,
                    "referencia": shift.reference,
                    "delta_cm-1": shift.delta,
                    "significativo": shift.significant,
                }
                for key, shift in result.shifts.shifts.items()
            },
            "interpretacion": list(result.shifts.interpretation),
            "compatible_con": [s.label for s, _ in result.shifts.dopant_matches],
        }
    return payload


def export_analysis(
    result: "AnalysisResult",
    directory: str | Path,
    stem: Optional[str] = None,
    delimiter: str = ",",
) -> list[Path]:
    """Write every export for one analysis into a directory.

    Produces ``<stem>_informe.txt``, ``<stem>_componentes.csv``,
    ``<stem>_curvas.csv``, ``<stem>_espectro.txt`` (the preprocessed data)
    and ``<stem>_analisis.json``.

    Returns
    -------
    list[pathlib.Path]
        The files written, in that order.
    """
    from ..core.io import write_spectrum

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    name = stem or result.raw.name
    written: list[Path] = []

    report = out / f"{name}_informe.txt"
    report.write_text(result.report(), encoding="utf-8")
    written.append(report)

    if result.fit is not None:
        written.append(
            export_components(result.fit, out / f"{name}_componentes.csv",
                              result, delimiter)
        )
        written.append(
            export_curves(result.fit, out / f"{name}_curvas.csv", result, delimiter)
        )
    written.append(write_spectrum(result.processed, out / f"{name}_espectro.txt"))
    written.append(export_json(result, out / f"{name}_analisis.json"))
    return written


def export_batch(
    results: Sequence["AnalysisResult"],
    path: str | Path,
    delimiter: str = ",",
) -> Path:
    """One row per analysed spectrum, every scalar quantity as a column."""
    rows = [r.to_dict() for r in results]
    if not rows:
        raise ValueError("no hay resultados que exportar")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = _header(None, None, "tabla de lote")
    lines.append(delimiter.join(columns))
    for row in rows:
        lines.append(delimiter.join(_cell(row.get(c)) for c in columns))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


__all__ = [
    "COMPONENT_COLUMNS",
    "analysis_to_dict",
    "component_rows",
    "export_analysis",
    "export_batch",
    "export_components",
    "export_curves",
    "export_json",
]
