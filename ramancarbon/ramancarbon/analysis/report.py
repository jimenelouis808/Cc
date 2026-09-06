"""The complete analysis: from a raw spectrum to a written report.

:func:`analyse` runs the whole chain in the order the physics requires and
returns one object carrying every intermediate result, so a caller — the
GUI, the CLI, a script — can show as much or as little as it wants without
re-running anything.

The order is not arbitrary:

1. **Preprocess.** Baseline first, because every intensity ratio is a ratio
   of background-free intensities.
2. **Find peaks** and, separately, look in the RBM window with tighter
   settings, since RBMs are narrow and weak and the general-purpose finder
   would miss them.
3. **Deconvolve** the D–G region, choosing the number of components by
   information criterion rather than by habit.
4. **Assign** the fitted components to named bands, dispersion-corrected to
   the actual laser.
5. **Interpret the G region** — is the second component G⁻ or D′? — because
   the answer decides both the classification and whether a diameter may be
   extracted from the splitting.
6. **Ratios**, then the structural quantities derived from them.
7. **Diameters**, from RBM and, where legitimate, from the G splitting.
8. **Classify.**
9. **Shifts**, last, because the reference to compare against depends on
   what the material turned out to be.

Each step degrades rather than fails: a spectrum that starts at 800 cm⁻¹
gets no RBM analysis and a classification that says so, instead of an
exception or a fabricated answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from ..core.peaks import PeakMeasurement, find_peaks
from ..core.preprocess import preprocess
from ..core.spectrum import Spectrum
from ..database import Database, load_database
from ..models.deconvolution import ModelComparison, build_model, compare_models
from ..models.fitting import FitModel, FitResult, fit_model
from .assignment import Assignment, GRegionInterpretation, assign_bands, resolve_g_region
from .classify import Classification, classify
from .diameter import (
    DiameterEstimate,
    WallPair,
    assign_chirality,
    ChiralityCandidate,
    diameter_from_g_splitting,
    rbm_diameter_with_spread,
)
from .ratios import (
    CrystalliteSize,
    DefectType,
    Ratio,
    crystallite_size,
    defect_type,
    graphene_layers,
    intensity_ratios,
)
from .shifts import ShiftAnalysis, analyse_shifts


@dataclass
class RBMResult:
    """Everything the RBM region yielded."""

    peaks: list[PeakMeasurement]
    fit: Optional[FitResult]
    diameters: list[DiameterEstimate]
    chiralities: dict[float, list[ChiralityCandidate]]
    wall_pairs: list[WallPair]
    covered: bool
    """Whether the spectrum actually reached the RBM window."""
    note: str = ""


@dataclass
class AnalysisResult:
    """The complete analysis of one spectrum."""

    raw: Spectrum
    processed: Spectrum
    diagnostics: dict
    peaks: list[PeakMeasurement]
    comparison: Optional[ModelComparison]
    fit: Optional[FitResult]
    assignment: Assignment
    g_region: Optional[GRegionInterpretation]
    ratios: dict[str, Ratio]
    crystallite: Optional[CrystalliteSize]
    defects: Optional[DefectType]
    rbm: RBMResult
    g_split_diameter: Optional[DiameterEstimate]
    classification: Classification
    shifts: Optional[ShiftAnalysis]
    layer_count: Optional[tuple[str, list[str]]]
    basis: str
    warnings: list[str] = field(default_factory=list)

    # -- convenience accessors ----------------------------------------
    @property
    def id_ig(self) -> Optional[float]:
        """I_D/I_G, or ``None`` if it could not be computed."""
        entry = self.ratios.get("ID_IG")
        return entry.value if entry and entry.available else None

    @property
    def i2d_ig(self) -> Optional[float]:
        """I_2D/I_G, or ``None``."""
        entry = self.ratios.get("I2D_IG")
        return entry.value if entry and entry.available else None

    @property
    def id_idprime(self) -> Optional[float]:
        """I_D/I_D', or ``None``."""
        entry = self.ratios.get("ID_IDp")
        return entry.value if entry and entry.available else None

    def to_dict(self) -> dict[str, Any]:
        """Flat dictionary of the headline numbers, for CSV export.

        Only scalars, so a batch of spectra can be written as one table.
        """
        g_entry = self.assignment.g_like()
        d_entry = self.assignment.get("D")
        two_d = self.assignment.get("2D")
        row: dict[str, Any] = {
            "nombre": self.raw.name,
            "laser_nm": self.raw.laser_nm,
            "material": self.classification.label,
            "confianza": self.classification.confidence,
            "base_intensidad": self.basis,
            "pos_G": g_entry.position if g_entry else None,
            "fwhm_G": g_entry.fwhm if g_entry else None,
            "pos_D": d_entry.position if d_entry else None,
            "fwhm_D": d_entry.fwhm if d_entry else None,
            "pos_2D": two_d.position if two_d else None,
            "fwhm_2D": two_d.fwhm if two_d else None,
            "ID_IG": self.id_ig,
            "I2D_IG": self.i2d_ig,
            "ID_IDp": self.id_idprime,
            "modelo": self.comparison.best if self.comparison else None,
            "R2": self.fit.r_squared if self.fit else None,
        }
        if self.crystallite:
            row["La_nm"] = self.crystallite.la_low_defect_nm
            row["LD_nm"] = self.crystallite.ld_low_defect_nm
            row["nD_cm2"] = self.crystallite.defect_density_cm2
            row["rama"] = self.crystallite.likely_branch
        if self.defects:
            row["tipo_defecto"] = self.defects.best_match
        if self.rbm.diameters:
            row["d_RBM_nm"] = self.rbm.diameters[0].diameter_nm
            row["n_RBM"] = len(self.rbm.diameters)
        if self.g_split_diameter:
            row["d_Gsplit_nm"] = self.g_split_diameter.diameter_nm
        return row

    def report(self, verbose: bool = True) -> str:
        """The full written report."""
        return build_report(self, verbose=verbose)


def analyse(
    spectrum: Spectrum,
    basis: str = "area",
    presets: Sequence[str] = ("two_band", "three_band", "four_band", "five_band"),
    metallic: Optional[bool] = None,
    rbm_parameterisation: Optional[str] = None,
    material_hint: Optional[str] = None,
    control: Optional["AnalysisResult"] = None,
    preprocess_kwargs: Optional[dict] = None,
    db: Optional[Database] = None,
) -> AnalysisResult:
    """Run the complete analysis on one spectrum.

    Parameters
    ----------
    spectrum:
        Raw spectrum, straight from the file. Preprocessing happens here so
        that the report can show what was done.
    basis:
        ``"area"`` or ``"height"`` for the intensity ratios. Areas are the
        default because the crystallite-size relation was calibrated on
        them and because they are insensitive to instrument resolution.
    presets:
        Deconvolution models to compare for the D–G region.
    metallic:
        Force the metallic (Breit–Wigner–Fano) G⁻ model. ``None`` decides
        from the fitted G⁻ width.
    rbm_parameterisation:
        Which RBM ↔ diameter relation to use; see ``rbm.json``.
    material_hint:
        Skip the classifier's verdict for the shift reference and use this
        material instead.
    control:
        An analysis of a control sample, used as the reference for the
        shift analysis. This is much better than comparing against
        literature values; see :mod:`ramancarbon.analysis.shifts`.
    preprocess_kwargs:
        Passed to :func:`~ramancarbon.core.preprocess.preprocess`.
    db:
        Loaded database.

    Returns
    -------
    AnalysisResult
    """
    database = db or load_database()
    warnings: list[str] = []

    if spectrum.laser_nm is None:
        warnings.append(
            "no se ha indicado la longitud de onda del láser. Sin ella no se "
            "corrigen las posiciones por dispersión, no se puede calcular L_a "
            "ni la densidad de defectos, y los desplazamientos de D y 2D no son "
            "interpretables. Indícala antes de usar estos resultados"
        )

    processed, diagnostics = preprocess(spectrum, **(preprocess_kwargs or {}))
    peaks = find_peaks(processed)

    # -- RBM region, with its own tighter search -----------------------
    rbm = _analyse_rbm(processed, rbm_parameterisation, database)

    # -- deconvolution of the D-G region -------------------------------
    comparison: Optional[ModelComparison] = None
    fit: Optional[FitResult] = None
    try:
        comparison = compare_models(processed, presets=presets, db=database)
        fit = comparison.results[comparison.best]
    except ValueError as exc:
        warnings.append(f"no se ha podido deconvolucionar la región D–G: {exc}")

    # -- the G region, and a dedicated SWCNT G fit when it makes sense --
    g_region: Optional[GRegionInterpretation] = None
    g_split_diameter: Optional[DiameterEstimate] = None
    swcnt_fit: Optional[FitResult] = None
    metallicity_note = ""
    if rbm.peaks:
        try:
            if metallic is None:
                is_metallic, swcnt_fit, metallicity_note = _decide_metallicity(
                    processed, database
                )
            else:
                is_metallic = metallic
                swcnt_fit = fit_model(
                    processed,
                    build_model(processed, preset="swcnt_g", metallic=metallic, db=database),
                )
                metallicity_note = (
                    "metalicidad impuesta por el usuario: "
                    + ("metálico" if metallic else "semiconductor")
                )
            if swcnt_fit is None:
                raise ValueError(metallicity_note)
            g_region = resolve_g_region(swcnt_fit, rbm_present=True, db=database)
            if metallicity_note:
                g_region.reasons.append(metallicity_note)
            if g_region.interpretation == "swcnt_split" and g_region.partner_position:
                g_split_diameter = diameter_from_g_splitting(
                    omega_g_plus=g_region.g_position,
                    omega_g_minus=g_region.partner_position,
                    metallic=is_metallic,
                    db=database,
                )
        except ValueError as exc:
            warnings.append(f"no se ha podido ajustar la región G como SWCNT: {exc}")
    elif fit is not None:
        g_region = resolve_g_region(fit, rbm_present=False, db=database)

    # -- the 2D band, fitted in its own window -------------------------
    two_d_single, two_d_fit = _fit_two_d(processed, database)

    # -- assignment ----------------------------------------------------
    # Every fitted region contributes its components; the peak finder fills
    # in the bands that no fit covered.
    sources: list[Any] = [f for f in (fit, two_d_fit, rbm.fit) if f is not None]
    sources.extend(peaks)
    assignment = assign_bands(processed, sources, db=database)
    if swcnt_fit is not None and g_region is not None and g_region.interpretation == "swcnt_split":
        _merge_swcnt_g(assignment, swcnt_fit, processed, database)

    # -- ratios and derived structure ----------------------------------
    ratios = intensity_ratios(assignment, basis=basis)
    g_entry = assignment.g_like()
    g_fwhm = g_entry.fwhm if g_entry else None

    crystallite: Optional[CrystalliteSize] = None
    id_ig_entry = ratios.get("ID_IG")
    if id_ig_entry and id_ig_entry.available and spectrum.laser_nm:
        try:
            crystallite = crystallite_size(
                id_ig_entry.value, spectrum.laser_nm, basis=basis, g_fwhm=g_fwhm, db=database
            )
        except ValueError as exc:
            warnings.append(f"no se ha podido estimar el tamaño de cristalito: {exc}")
    elif id_ig_entry and id_ig_entry.available:
        warnings.append(
            "no se calcula L_a ni la densidad de defectos: ambas dependen de "
            "λ⁴ y hace falta la longitud de onda del láser"
        )

    defects: Optional[DefectType] = None
    idp_entry = ratios.get("ID_IDp")
    if idp_entry and idp_entry.available:
        try:
            defects = defect_type(idp_entry.value, db=database)
        except ValueError:
            pass

    # -- classification -------------------------------------------------
    classification = classify(
        processed,
        assignment,
        rbm_peaks=rbm.peaks if rbm.covered else None,
        ratios=ratios,
        two_d_single_lorentzian=two_d_single,
        g_region=g_region.interpretation if g_region else None,
        db=database,
    )

    layer_count = None
    material_key = material_hint or classification.best
    if material_key in {"graphene_1L", "graphene_2L", "FLG", "graphite"}:
        i2d = ratios.get("I2D_IG")
        two_d_entry = assignment.get("2D")
        if i2d and i2d.available:
            layer_count = graphene_layers(
                i2d.value,
                two_d_fwhm=two_d_entry.fwhm if two_d_entry else None,
                two_d_single_lorentzian=two_d_single,
            )

    # -- shifts ---------------------------------------------------------
    shifts: Optional[ShiftAnalysis] = None
    if assignment.g_like() is not None:
        shifts = analyse_shifts(
            assignment,
            material_key=material_key or "graphene_1L",
            control=control.assignment if control else None,
            id_ig=id_ig_entry.value if id_ig_entry and id_ig_entry.available else None,
            control_id_ig=control.id_ig if control else None,
            g_fwhm=g_fwhm,
            control_g_fwhm=(
                control.assignment.g_like().fwhm
                if control and control.assignment.g_like()
                else None
            ),
            db=database,
        )

    return AnalysisResult(
        raw=spectrum,
        processed=processed,
        diagnostics=diagnostics,
        peaks=peaks,
        comparison=comparison,
        fit=fit,
        assignment=assignment,
        g_region=g_region,
        ratios=ratios,
        crystallite=crystallite,
        defects=defects,
        rbm=rbm,
        g_split_diameter=g_split_diameter,
        classification=classification,
        shifts=shifts,
        layer_count=layer_count,
        basis=basis,
        warnings=warnings,
    )


def _analyse_rbm(
    spectrum: Spectrum, parameterisation: Optional[str], db: Database
) -> RBMResult:
    """Search the RBM window, fit it, and turn the peaks into diameters."""
    from .diameter import find_wall_pairs

    covered = spectrum.covers(120.0, 350.0, fraction=0.7)
    if not covered:
        lo, _ = spectrum.range
        return RBMResult(
            peaks=[],
            fit=None,
            diameters=[],
            chiralities={},
            wall_pairs=[],
            covered=False,
            note=(
                f"el espectro empieza en {lo:.0f} cm⁻¹ y no cubre la región RBM "
                "(120–350 cm⁻¹). No se puede estimar el diámetro por RBM ni "
                "concluir nada sobre el número de paredes a partir de su ausencia"
            ),
        )

    window = (max(80.0, spectrum.range[0]), min(400.0, spectrum.range[1]))
    raw_peaks = find_peaks(
        spectrum, window=window, min_distance_cm=5.0, min_fwhm_cm=3.0
    )
    band = db.band("RBM")
    peaks = [p for p in raw_peaks if p.fwhm is None or p.fwhm <= band.typical_fwhm[1] * 2.0]

    if not peaks:
        return RBMResult(
            peaks=[],
            fit=None,
            diameters=[],
            chiralities={},
            wall_pairs=[],
            covered=True,
            note="no se detectan RBM en 120–350 cm⁻¹",
        )

    fit: Optional[FitResult] = None
    try:
        model = build_model(spectrum, preset="rbm", seeds=peaks, db=db, window=window)
        fit = fit_model(spectrum, model)
        positions = [p.peak_position for p in fit.peaks]
    except ValueError:
        positions = [p.position for p in peaks]

    diameters: list[DiameterEstimate] = []
    chiralities: dict[float, list[ChiralityCandidate]] = {}
    for omega in positions:
        try:
            estimate = rbm_diameter_with_spread(omega, parameterisation, db=db)
        except ValueError:
            continue
        diameters.append(estimate)
        chiralities[omega] = assign_chirality(estimate.diameter_nm, db=db)[:6]

    wall_pairs = find_wall_pairs(positions, parameterisation, db=db) if len(positions) > 1 else []

    return RBMResult(
        peaks=peaks,
        fit=fit,
        diameters=diameters,
        chiralities=chiralities,
        wall_pairs=wall_pairs,
        covered=True,
    )


def _decide_metallicity(
    spectrum: Spectrum, db: Database
) -> tuple[bool, Optional[FitResult], str]:
    """Decide whether G⁻ is metallic by fitting it both ways and comparing.

    A metallic tube's G⁻ couples to the free-electron continuum and takes
    the asymmetric Breit–Wigner–Fano shape; a semiconducting one stays
    Lorentzian. Rather than guessing from the height of the low-frequency
    shoulder — which a broad amorphous D3 band also produces — both models
    are fitted and compared by BIC, and the asymmetry parameter is required
    to be resolved (|1/q| > 0.02) before the metallic verdict is accepted.

    Getting this wrong is not cosmetic: the two G-splitting constants differ
    by 40 %, so a wrong call moves the deduced diameter by about 30 %.

    Returns
    -------
    (bool, FitResult or None, str)
        Whether it is metallic, the better fit, and a sentence explaining
        the decision.
    """
    fits: dict[bool, FitResult] = {}
    for metallic in (False, True):
        try:
            model = build_model(spectrum, preset="swcnt_g", metallic=metallic, db=db)
            fits[metallic] = fit_model(spectrum, model)
        except ValueError:
            continue
    if not fits:
        return False, None, "no se ha podido ajustar la región G"
    if len(fits) == 1:
        only = next(iter(fits))
        return only, fits[only], "solo se ha podido ajustar un modelo de la región G"

    lorentzian, bwf = fits[False], fits[True]
    g_minus = bwf.peak("G-")
    q_inverse = abs(g_minus.extra[0]) if g_minus and g_minus.extra else 0.0
    delta_bic = lorentzian.bic - bwf.bic

    if delta_bic > 6.0 and q_inverse > 0.02:
        return (
            True,
            bwf,
            f"G⁻ se describe mejor con un perfil Breit-Wigner-Fano "
            f"(ΔBIC = {delta_bic:.1f}, 1/q = {g_minus.extra[0]:.3f}): "
            "acoplamiento fonón-continuo electrónico, tubos metálicos",
        )
    return (
        False,
        lorentzian,
        f"G⁻ no mejora con un perfil asimétrico (ΔBIC = {delta_bic:.1f}): "
        "se trata como semiconductor",
    )


def _fit_two_d(
    spectrum: Spectrum, db: Database
) -> tuple[Optional[bool], Optional[FitResult]]:
    """Fit the 2D band and decide whether one Lorentzian is enough.

    Fits one and several Lorentzians over the 2D window and compares by
    BIC. A single adequate Lorentzian of normal width is the monolayer
    graphene signature; needing more components means Bernal-stacked
    multilayer, or a nanotube/graphitic envelope.

    Returns
    -------
    (bool or None, FitResult or None)
        Whether one Lorentzian sufficed, and the better of the two fits.
        ``(None, None)`` when the band is absent or its window is not
        covered — which is different from "no", and the classifier treats
        it as such.
    """
    band = db.band("2D")
    lo, hi = band.window_at(spectrum.laser_ev, pad=60.0)
    if not spectrum.covers(lo, hi, fraction=0.7):
        return None, None
    peak = spectrum.max_in(*band.window_at(spectrum.laser_ev))
    if peak is None or peak[1] <= 5 * spectrum.noise_estimate():
        return None, None
    try:
        one = fit_model(spectrum, build_model(spectrum, preset="two_d", db=db))
    except ValueError:
        return None, None
    seeds = find_peaks(spectrum, window=(lo, hi))
    if len(seeds) < 2:
        return True, one
    try:
        many = fit_model(
            spectrum, build_model(spectrum, preset="two_d", seeds=seeds[:4], db=db)
        )
    except ValueError:
        return True, one
    if one.bic <= many.bic:
        return True, one
    return False, many


def _merge_swcnt_g(
    assignment: Assignment, swcnt_fit: FitResult, spectrum: Spectrum, db: Database
) -> None:
    """Fold a dedicated G⁻/G⁺ fit into an assignment built from the D–G fit.

    The wide D–G model fits the G region with a single component; when the
    material turns out to be single- or double-walled, the split fit is the
    better description of that region and its G⁺ becomes the intensity
    reference. Only the G-region entries are replaced, so the D band and
    everything below it keep the areas from the model that was selected on
    evidence.
    """
    from .assignment import _make  # noqa: PLC0415 - internal helper, intentionally shared

    for component in swcnt_fit.peaks:
        if component.band not in {"G-", "G+"}:
            continue
        band = db.band(component.band)
        assignment.bands[component.band] = _make(
            band,
            {
                "position": component.peak_position,
                "height": component.peak_height,
                "fwhm": component.fwhm,
                "area": component.area,
                "origin": "fit",
                "band": component.band,
                "profile": component.profile,
            },
            assignment.laser_ev,
            db,
        )
    for key in ("G-", "G+"):
        if key in assignment.bands and key in assignment.missing:
            assignment.missing.remove(key)

    # With G+ present, the plain G entry would double-count the same
    # intensity in every ratio; drop it and say so.
    if "G+" in assignment.bands and "G" in assignment.bands:
        del assignment.bands["G"]
        assignment.warnings.append(
            "la región G se ha reajustado como G⁻/G⁺ (material de pared simple "
            "o doble); los cocientes usan G⁺ como referencia"
        )


# ----------------------------------------------------------------------
# report rendering
# ----------------------------------------------------------------------
def _rule(title: str) -> str:
    return f"\n{title}\n{'─' * max(len(title), 8)}"


def build_report(result: AnalysisResult, verbose: bool = True) -> str:
    """Render an :class:`AnalysisResult` as plain text."""
    lines: list[str] = []
    lines.append("═" * 72)
    lines.append(f"  ANÁLISIS RAMAN — {result.raw.name}")
    lines.append("═" * 72)
    lines.append(result.processed.describe())

    lines.append(_rule("1. IDENTIFICACIÓN"))
    lines.append(result.classification.summary())

    lines.append(_rule("2. BANDAS IDENTIFICADAS"))
    lines.append(result.assignment.summary())

    lines.append(_rule("3. DECONVOLUCIÓN D–G"))
    if result.comparison:
        lines.append(result.comparison.summary())
        lines.append("")
        lines.append(result.fit.summary())
    else:
        lines.append("no realizada")

    lines.append(_rule("4. COCIENTES DE INTENSIDAD"))
    lines.append(f"base: {result.basis} (áreas integradas)" if result.basis == "area"
                 else f"base: {result.basis} (alturas de pico)")
    for entry in result.ratios.values():
        lines.append("  " + str(entry))
    if result.crystallite:
        lines.append("")
        lines.append(result.crystallite.summary())
    if result.defects:
        lines.append("")
        lines.append(result.defects.summary())

    lines.append(_rule("5. DIÁMETROS"))
    if not result.rbm.covered:
        lines.append(result.rbm.note)
    elif not result.rbm.diameters:
        lines.append(result.rbm.note or "sin RBM detectados")
    else:
        for estimate in result.rbm.diameters:
            lines.append(f"  RBM {estimate.input_value:7.1f} cm⁻¹ → d = {estimate}")
            for warning in estimate.warnings:
                lines.append(f"      ⚠ {warning}")
            candidates = result.rbm.chiralities.get(estimate.input_value, [])
            if candidates and verbose:
                lines.append(
                    "      (n,m) compatibles: "
                    + ", ".join(f"{c.label} {c.electronic[:4]}" for c in candidates[:5])
                )
        if result.rbm.chiralities and verbose:
            lines.append(
                "      nota: el diámetro por sí solo no fija (n,m); hace falta "
                "además la condición de resonancia (Kataura), que necesita "
                "medir con varios láseres"
            )
    if result.rbm.wall_pairs:
        plausible = [p for p in result.rbm.wall_pairs if p.plausible]
        if plausible:
            lines.append("  Paredes concéntricas compatibles:")
            lines.extend("    " + str(p) for p in plausible[:4])
    if result.g_split_diameter:
        lines.append(f"  Por desdoblamiento G: d = {result.g_split_diameter}")
        for warning in result.g_split_diameter.warnings:
            lines.append(f"      ⚠ {warning}")
        if result.rbm.diameters:
            lines.append("      " + _cross_check(result))

    if result.layer_count:
        lines.append(_rule("6. NÚMERO DE CAPAS"))
        verdict, reasons = result.layer_count
        lines.append(f"  {verdict}")
        lines.extend("    – " + r for r in reasons)

    if result.shifts:
        lines.append(_rule("7. DESPLAZAMIENTOS RESPECTO A LA REFERENCIA"))
        lines.append(result.shifts.summary())

    if result.warnings:
        lines.append(_rule("AVISOS"))
        lines.extend("  ⚠ " + w for w in result.warnings)

    lines.append("")
    return "\n".join(lines)


def _cross_check(result: AnalysisResult) -> str:
    """Compare the RBM and G-splitting diameters and say what it means."""
    rbm_d = result.rbm.diameters[0].diameter_nm
    g_d = result.g_split_diameter.diameter_nm
    difference = abs(rbm_d - g_d)
    tolerance = max(
        0.15,
        (result.rbm.diameters[0].uncertainty_nm or 0.0)
        + (result.g_split_diameter.uncertainty_nm or 0.0),
    )
    if difference <= tolerance:
        return (
            f"✓ los dos métodos coinciden ({rbm_d:.2f} vs {g_d:.2f} nm): el "
            "diámetro es sólido"
        )
    return (
        f"✗ los dos métodos discrepan ({rbm_d:.2f} vs {g_d:.2f} nm, "
        f"tolerancia {tolerance:.2f}). Revisa si el tubo es metálico o "
        "semiconductor, si la parametrización RBM corresponde al entorno de "
        "la muestra, y si lo que se ha ajustado como G⁻ no es en realidad D3"
    )


__all__ = ["AnalysisResult", "RBMResult", "analyse", "build_report"]
