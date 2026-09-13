"""One call from files to an answer, and the written report.

The order of the report is the order the numbers have to be *read*, which
is not the order they are computed:

1. **What was measured**, including the things a text export throws away.
   A fit whose pass energy is unknown has no resolution floor and every
   width in it is unjudged.
2. **The charge reference**, before any binding energy. A binding energy
   quoted without saying what put the axis where it is is not a
   measurement, and the reference carries a published spread that every
   number below inherits.
3. **The survey**, before the regions. Which elements are there decides
   which regions are worth fitting, and — more usefully — the *unexplained*
   peaks say whether something is present that nobody went looking for.
4. **The regions**, each with the literature windows its components were
   held in and the statistics that say whether the model is complete.
5. **The composition**, last, because it is the number that depends on
   everything above being right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .calibrate import Calibration, calibrate, calibrate_to_state
from .elements import XPSDatabase, load_xps_database
from .fitting import XPSFitResult, fit_region
from .presets import count_model, state_model
from .quantify import Quantification, areas_from_fits, quantify
from .spectrum import XPSError, XPSSpectrum
from .survey import SurveyResult, identify


@dataclass
class XPSAnalysis:
    """Everything the XPS analysis of one sample produced."""

    name: str = "muestra"
    survey: Optional[SurveyResult] = None
    regions: list[XPSFitResult] = field(default_factory=list)
    composition: Optional[Quantification] = None
    calibration: Optional[Calibration] = None
    warnings: list[str] = field(default_factory=list)

    def region(self, label: str) -> Optional[XPSFitResult]:
        for item in self.regions:
            if item.region_label == label:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Flat row for a batch table."""
        row: dict[str, Any] = {"nombre": self.name}
        if self.calibration:
            row["referencia"] = self.calibration.reference
            row["desplazamiento_carga_eV"] = self.calibration.shift_ev
        if self.survey:
            row["elementos"] = ", ".join(self.survey.symbols())
            row["picos_sin_explicar"] = len(self.survey.unexplained)
        if self.composition:
            for item in self.composition.abundances:
                row[f"at%_{item.element}"] = item.atomic_percent
        for result in self.regions:
            for component in result.components:
                if component.satellite:
                    continue
                key = f"{result.region_label}:{component.name}"
                row[f"area%_{key}"] = 100.0 * component.area_fraction
                row[f"BE_{key}"] = component.centre
        return row

    def report(self, verbose: bool = True) -> str:
        return build_report(self, verbose=verbose)


def analyse_xps(
    spectra: Sequence[XPSSpectrum],
    reference: Optional[str] = "C1s_adventitious",
    reference_state: Optional[tuple[str, str]] = None,
    regions: Optional[dict[str, Any]] = None,
    transmission: str = "potencia",
    exponent: float = -0.65,
    name: str = "",
    database: Optional[XPSDatabase] = None,
) -> XPSAnalysis:
    """Analyse a survey plus its high-resolution regions.

    Parameters
    ----------
    spectra:
        Any mixture of survey and region scans. Which is which is decided
        by their span, not by what the file called them.
    reference:
        Charge reference to apply, or ``None`` to take the axis as it is.
        The shift is measured **once**, on whichever spectrum carries the
        reference line at the best resolution, and applied to all of them —
        because they were measured in one session on one sample and a
        per-spectrum shift would be fitting the charging separately to each
        region, which is how a chemical shift gets invented.
    reference_state:
        ``(region, state)``, e.g. ``("C 1s", "C-C sp2")``: reference on a
        **fitted component** instead of on a whole peak, which is the right
        thing for a sample made of the element that carries the reference.
        Takes precedence over ``reference``. On the synthetic
        nitrogen-doped tube it recovers a 1.8 eV charge shift to 0.09 eV
        where referencing the whole C 1s peak leaves 0.55 eV — the
        difference being the oxidised tail dragging the centroid.
    regions:
        What to fit, as ``{"N 1s": ["pyridinic", "pyrrolic"]}`` or
        ``{"N 1s": 3}`` for "the three strongest states the database
        knows". ``None`` fits every region whose label the database
        recognises, with every state it lists — which is more components
        than most samples justify, and the warnings will say so.

    Returns
    -------
    XPSAnalysis
    """
    database = database or load_xps_database()
    if not spectra:
        raise XPSError("no hay ningún espectro que analizar")
    analysis = XPSAnalysis(name=name or spectra[0].name)

    surveys = [item for item in spectra if item.is_survey]
    scans = [item for item in spectra if not item.is_survey]

    # -- charge referencing, once, on the best-resolved spectrum --------
    shifted = list(spectra)
    if reference_state:
        region_label, state_key = reference_state
        target = next(
            (item for item in scans
             if (database.region_for_line(item.region) or item.region) == region_label),
            None,
        )
        if target is None:
            analysis.warnings.append(
                f"se pidió referenciar sobre {region_label}:{state_key} y no "
                f"hay ningún espectro de esa región; se usa {reference!r}"
            )
        else:
            chosen = (regions or {}).get(region_label)
            states = chosen if isinstance(chosen, list) else None
            _, calibration = calibrate_to_state(
                target, region_label, state_key, states, database=database)
            analysis.calibration = calibration
            shifted = [item.shifted(calibration.shift_ev,
                                    f"{region_label}:{state_key}")
                       for item in spectra]
            surveys = [item for item in shifted if item.is_survey]
            scans = [item for item in shifted if not item.is_survey]
            reference = None
    if reference:
        entry = database.reference(reference)
        candidates = [
            item for item in scans + surveys
            if item.covers(entry.energy_ev - 5.0, entry.energy_ev + 5.0, 0.8)
        ]
        if not candidates:
            analysis.warnings.append(
                f"ningún espectro cubre la referencia {entry.line or reference} "
                f"({entry.energy_ev:.1f} eV), así que el eje se deja como "
                "venía. Las energías de enlace de abajo son las del "
                "instrumento, con la carga dentro"
            )
        else:
            # Best resolved: smallest step, and a region before a survey.
            best = min(candidates, key=lambda item: (item.is_survey, item.step))
            if reference == "C1s_adventitious" and any(
                (database.region_for_line(item.region) or item.region) == "C 1s"
                for item in scans
            ):
                analysis.warnings.append(
                    "la muestra trae su propia región C 1s y se está "
                    "referenciando contra el pico C 1s adventicio: en un "
                    "material hecho de carbono eso es circular, porque el "
                    "pico contra el que se referencia ES la muestra. Usa "
                    "reference_state=(\"C 1s\", \"C-C sp2\") — en el demo "
                    "sintético la diferencia entre las dos formas es 0.55 eV "
                    "frente a 0.05 eV"
                )
            _, calibration = calibrate(best, reference, database=database)
            analysis.calibration = calibration
            shifted = [
                item.shifted(calibration.shift_ev,
                             f"{entry.line or reference} a {entry.energy_ev:.2f} eV")
                for item in spectra
            ]
            surveys = [item for item in shifted if item.is_survey]
            scans = [item for item in shifted if not item.is_survey]

    # -- survey --------------------------------------------------------
    if surveys:
        analysis.survey = identify(surveys[0], database=database)
    else:
        analysis.warnings.append(
            "no hay barrido ancho. Sin él no se sabe qué elementos hay, solo "
            "los que alguien decidió medir: un survey es lo que encuentra lo "
            "que no se esperaba"
        )

    # -- regions -------------------------------------------------------
    wanted = dict(regions or {})
    for spectrum in scans:
        label = database.region_for_line(spectrum.region) or spectrum.region
        if not database.states_for(label):
            analysis.warnings.append(
                f"la región «{spectrum.region}» no está en la base de datos, "
                "así que no se ajusta sola. Ajústala con free_model y nombra "
                "sus componentes en el informe"
            )
            continue
        choice = wanted.get(label, wanted.get(spectrum.region))
        try:
            if isinstance(choice, int):
                model, notes = count_model(spectrum, label, choice,
                                           database=database)
                analysis.warnings.extend(f"{label}: {note}" for note in notes)
            else:
                model = state_model(spectrum, label, choice, database=database,
                                    include_satellites=True)
            analysis.regions.append(fit_region(spectrum, model, database=database))
        except XPSError as error:
            analysis.warnings.append(f"{label}: {error}")

    # -- composition ---------------------------------------------------
    if analysis.regions:
        photon = next((s.photon_energy for s in shifted if s.photon_energy), None)
        if photon is None:
            analysis.warnings.append(
                "sin energía del fotón no hay escala cinética y por tanto no "
                "hay corrección de transmisión ni cuantificación"
            )
        else:
            try:
                analysis.composition = quantify(
                    areas_from_fits(analysis.regions), photon_energy=photon,
                    transmission=transmission, exponent=exponent,
                    database=database,
                )
            except XPSError as error:
                analysis.warnings.append(f"cuantificación: {error}")
    return analysis


def build_report(analysis: XPSAnalysis, verbose: bool = True) -> str:
    """The written report, in the order the numbers have to be read."""
    lines = [
        f"INFORME XPS — {analysis.name}",
        "=" * 60,
        "",
    ]

    if analysis.calibration:
        lines.append("REFERENCIA DE CARGA")
        lines.append("  " + analysis.calibration.describe())
        if verbose:
            lines.extend("  ⚠ " + text for text in analysis.calibration.warnings)
        lines.append("")
    else:
        lines.append("REFERENCIA DE CARGA: ninguna. El eje es el del "
                     "instrumento y lleva la carga de la muestra dentro.")
        lines.append("")

    if analysis.survey:
        lines.append("SURVEY")
        for item in analysis.survey.elements:
            lines.append("  " + item.describe())
        if analysis.survey.uncorroborated:
            lines.append("  sin corroborar: " + ", ".join(
                item.symbol for item in analysis.survey.uncorroborated))
        if analysis.survey.unexplained:
            lines.append("  picos sin explicar: " + ", ".join(
                f"{peak.binding_energy:.1f} eV"
                for peak in analysis.survey.unexplained[:8]))
        if verbose:
            lines.extend("  ⚠ " + text for text in analysis.survey.warnings)
            lines.extend("  solape: " + text
                         for text in analysis.survey.overlaps)
        lines.append("")

    for result in analysis.regions:
        lines.append(f"REGIÓN {result.region_label}")
        lines.append("  " + result.background.describe())
        lines.append(
            f"  R² = {result.r_squared:.5f}   χ²_red = {result.reduced_chi2:.3f}"
            + (f"   DW = {result.durbin_watson:.2f}" if result.durbin_watson else "")
        )
        for component in result.components:
            lines.append("  " + component.summary())
            if verbose and component.justification:
                lines.append("      " + component.justification)
        if result.links:
            lines.append("  ligaduras: " + "; ".join(result.links))
        if verbose:
            lines.extend("  ⚠ " + text for text in result.warnings)
        lines.append("")

    if analysis.composition:
        lines.append("COMPOSICIÓN")
        for item in analysis.composition.abundances:
            lines.append("  " + item.summary())
        if verbose:
            lines.extend("  ⚠ " + text for text in analysis.composition.warnings)
        lines.append("")

    if analysis.warnings:
        lines.append("AVISOS")
        lines.extend("  ⚠ " + text for text in analysis.warnings)
        lines.append("")

    lines.append(
        "Las energías de enlace de este informe valen lo que valga la "
        "referencia de carga de arriba. Las áreas son las de la ventana "
        "ajustada y con el fondo que dice cada región: cambiar los extremos "
        "de la ventana las cambia, y eso no es un defecto del ajuste sino lo "
        "que significa «área de un pico sobre un fondo»."
    )
    return "\n".join(lines)


__all__ = ["XPSAnalysis", "analyse_xps", "build_report"]
