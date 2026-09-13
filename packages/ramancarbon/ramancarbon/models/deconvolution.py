"""Ready-made deconvolution models for the D–G region and the 2D band.

Choosing how many components to put in the 1000–1800 cm⁻¹ window is the
central decision of a carbon Raman analysis, and adding components always
improves χ². The presets here correspond to the models actually used in the
literature, each with the material class it was designed for:

``two_band``
    D + G. The honest minimum. Correct for clean graphene and well-ordered
    nanotubes, where there is nothing else to fit.
``three_band``
    D + G + D′. Add D′ whenever the G band has a shoulder above it — which
    is almost always in multi-walled and defective material. Without D′ the
    fitted G band is pulled upwards by several cm⁻¹ and its width inflated,
    which then corrupts every G-position and I_D/I_G number downstream.
``four_band``
    D + D3 + G + D′. Adds the amorphous band in the D–G valley.
``five_band``
    D4 + D + D3 + G + D′, the Sadezky model for soot and carbon black. The
    most complete and the most easily abused: five overlapping components
    over an 800 cm⁻¹ window are strongly correlated, and the fitter reports
    that correlation rather than hiding it.
``swcnt_full``
    D + G⁻ + G⁺ + D′ over the whole D–G window. The right model whenever
    the sample has an RBM: the four-band model above has no G⁻, so on a
    nanotube spectrum the D band stretches to its width limit trying to
    absorb the G⁻ intensity, and the resulting I_D/I_G is far too large.
    That failure is not hypothetical — it is what the four-band model does
    to a metallic tube, whose broad Breit–Wigner–Fano G⁻ reaches well below
    1500 cm⁻¹.
``swcnt_g``
    G⁻ + G⁺ (+ D′) over the G region alone, with G⁻ optionally
    Breit–Wigner–Fano for metallic tubes. Use when only the G region
    matters.
``rbm``
    A variable number of narrow Lorentzians over the RBM window, seeded
    from the peak finder.
``two_d``
    The 2D band, as one Lorentzian (monolayer test) or several.

Every preset is a *starting point*: it returns a
:class:`~ramancarbon.models.fitting.FitModel` whose components can be
edited, fixed or removed before fitting. :func:`compare_models` fits several
and ranks them by an information criterion, which is how to decide the
number of components on evidence rather than on habit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ..core.peaks import PeakMeasurement, find_peaks
from ..core.spectrum import Spectrum
from ..database import Database, load_database
from .fitting import FitModel, FitResult, PeakSpec, fit_model
from .lineshapes import resolve_profile

#: Preset identifiers, in the order the GUI offers them.
PRESETS = (
    "two_band",
    "three_band",
    "four_band",
    "five_band",
    "swcnt_full",
    "swcnt_g",
    "rbm",
    "two_d",
)

#: Human labels for the presets.
PRESET_LABELS = {
    "two_band": "2 bandas: D + G",
    "three_band": "3 bandas: D + G + D'",
    "four_band": "4 bandas: D + D3 + G + D'",
    "five_band": "5 bandas (Sadezky): D4 + D + D3 + G + D'",
    "swcnt_full": "Nanotubo: D + G⁻ + G⁺ + D'",
    "swcnt_g": "Región G de SWCNT: G⁻ + G⁺ + D'",
    "rbm": "Región RBM: lorentzianas estrechas",
    "two_d": "Banda 2D",
}

#: Which band keys each preset uses, in ascending frequency.
PRESET_BANDS = {
    "two_band": ("D", "G"),
    "three_band": ("D", "G", "D'"),
    "four_band": ("D", "D3", "G", "D'"),
    "five_band": ("D4", "D", "D3", "G", "D'"),
    "swcnt_full": ("D", "G-", "G+", "D'"),
}

#: Default fit windows in cm⁻¹, at the database's reference excitation.
PRESET_WINDOWS = {
    "two_band": (1100.0, 1750.0),
    "three_band": (1100.0, 1750.0),
    "four_band": (1050.0, 1750.0),
    "five_band": (900.0, 1800.0),
    "swcnt_full": (1100.0, 1750.0),
    "swcnt_g": (1450.0, 1700.0),
    "two_d": (2500.0, 2850.0),
}


def build_model(
    spectrum: Spectrum,
    preset: str = "three_band",
    metallic: bool = False,
    seeds: Optional[Sequence[PeakMeasurement]] = None,
    background: str = "linear",
    db: Optional[Database] = None,
    window: Optional[tuple[float, float]] = None,
    profile: Optional[str] = None,
) -> FitModel:
    """Build a deconvolution model for one of the presets.

    Initial centres come from the database (dispersion-corrected to the
    spectrum's laser), initial heights from the data at those centres, and
    initial widths from the middle of each band's typical range. Centre
    bounds are the band's own database window, so the fit cannot move a
    component out of physical reach — a D band cannot slide to 1450 cm⁻¹ to
    mop up amorphous intensity.

    Parameters
    ----------
    spectrum:
        The spectrum to be fitted; used for its laser energy and for the
        initial heights.
    preset:
        One of :data:`PRESETS`.
    metallic:
        For ``swcnt_g``: fit G⁻ with a Breit–Wigner–Fano profile. Use it
        when G⁻ is broad and asymmetric; the resulting 1/q is a direct
        measure of the electron–phonon coupling.
    seeds:
        For ``rbm`` and ``two_d``: peaks to build components around,
        normally from :func:`~ramancarbon.core.peaks.find_peaks`.
    background:
        Polynomial background inside the fit; see
        :class:`~ramancarbon.models.fitting.FitModel`.
    db:
        Loaded database.
    window:
        Override the preset's fit window.
    profile:
        Force every component to one lineshape, overriding the per-band
        defaults in the database. ``None`` keeps those defaults, which are
        Lorentzian for the true phonon bands and Gaussian for the D3 and D4
        envelopes.

        ``"pseudo_voigt"`` is the pragmatic choice for real, noisy spectra
        and it is worth understanding why. A pseudo-Voigt with a free
        mixing fraction η contains both the Lorentzian (η = 1) and the
        Gaussian (η = 0) as special cases, so instead of asserting the
        lineshape you let the fit report it — and the fitted η is then a
        result you can look at. A band that comes back with η near 1 is
        behaving like a phonon with a finite lifetime; one that comes back
        near 0 is an inhomogeneous envelope. The cost is one extra free
        parameter per band, which on a five-component model over a noisy
        spectrum is a real cost: watch the correlation warnings.

        A metallic G⁻ is the one component this does **not** override,
        because Breit–Wigner–Fano asymmetry is not a shape preference but
        a different physical mechanism, and no symmetric profile can
        represent it.

    Returns
    -------
    FitModel

    Raises
    ------
    ValueError
        If the preset is unknown, the profile name is not recognised, or
        the spectrum does not cover the preset's window.
    """
    if profile is not None:
        profile = resolve_profile(profile)
    database = db or load_database()
    laser_ev = spectrum.laser_ev

    if preset == "rbm":
        return _rbm_model(spectrum, seeds, background, database, window, profile)
    if preset == "two_d":
        return _two_d_model(spectrum, seeds, background, database, window, profile)
    if preset == "swcnt_g":
        return _swcnt_g_model(spectrum, metallic, background, database, window, profile)
    if preset not in PRESET_BANDS:
        raise ValueError(
            f"unknown preset {preset!r}; available: {', '.join(PRESETS)}"
        )

    lo, hi = window or _shifted_window(PRESET_WINDOWS[preset], database, laser_ev)
    _require_coverage(spectrum, lo, hi, preset)

    peaks: list[PeakSpec] = []
    for key in PRESET_BANDS[preset]:
        # BWF for a metallic G- is mechanism, not preference: it survives an
        # explicit profile override.
        chosen = "bwf" if (key == "G-" and metallic) else profile
        spec = _spec_for_band(spectrum, key, database, laser_ev, profile=chosen)
        if key == "G-" and metallic:
            spec.fwhm = 60.0
            spec.fwhm_bounds = (20.0, 160.0)
            spec.extra = (-0.12,)
            spec.extra_bounds = ((-0.45, 0.0),)
        peaks.append(spec)
    return FitModel(peaks=peaks, window=(lo, hi), background=background, name=preset)


def _shifted_window(
    window: tuple[float, float], db: Database, laser_ev: Optional[float]
) -> tuple[float, float]:
    """Translate a fit window by the D band's dispersion.

    The window has to follow the bands: at 785 nm the whole D–G complex sits
    ~38 cm⁻¹ lower, and a window fixed in absolute cm⁻¹ would clip it.
    """
    if laser_ev is None:
        return window
    shift = db.band("D").dispersion * (laser_ev - db.reference_laser_ev)
    return (window[0] + shift, window[1] + shift)


def _require_coverage(spectrum: Spectrum, low: float, high: float, preset: str) -> None:
    lo, hi = spectrum.range
    if lo > low + 0.25 * (high - low) or hi < high - 0.25 * (high - low):
        raise ValueError(
            f"el preset «{preset}» necesita la ventana {low:.0f}–{high:.0f} cm⁻¹ "
            f"y el espectro solo cubre {lo:.0f}–{hi:.0f} cm⁻¹"
        )


def _spec_for_band(
    spectrum: Spectrum,
    key: str,
    db: Database,
    laser_ev: Optional[float],
    profile: Optional[str] = None,
) -> PeakSpec:
    """One component seeded from the database and the data."""
    band = db.band(key)
    centre = band.position_at(laser_ev)
    bounds = band.window_at(laser_ev)
    observed = spectrum.max_in(*bounds)
    height = observed[1] if observed else 1.0
    if height <= 0:
        height = max(abs(height), 1e-3)
    fwhm_lo, fwhm_hi = band.typical_fwhm
    fwhm = float(np.sqrt(fwhm_lo * fwhm_hi))  # geometric mean of the range
    return PeakSpec(
        name=key,
        profile=profile or band.default_profile,
        centre=centre,
        height=height,
        fwhm=fwhm,
        centre_bounds=bounds,
        height_bounds=(0.0, height * 20.0),
        fwhm_bounds=(fwhm_lo, fwhm_hi),
        band=key,
    )


def _swcnt_g_model(
    spectrum: Spectrum,
    metallic: bool,
    background: str,
    db: Database,
    window: Optional[tuple[float, float]],
    profile: Optional[str] = None,
) -> FitModel:
    laser_ev = spectrum.laser_ev
    lo, hi = window or _shifted_window(PRESET_WINDOWS["swcnt_g"], db, laser_ev)
    _require_coverage(spectrum, lo, hi, "swcnt_g")
    g_minus = _spec_for_band(
        spectrum, "G-", db, laser_ev,
        profile="bwf" if metallic else (profile or "lorentzian"),
    )
    if metallic:
        # A metallic G- is broad; let it be, and let 1/q run negative only.
        g_minus.fwhm = 60.0
        g_minus.fwhm_bounds = (20.0, 160.0)
        g_minus.extra = (-0.12,)
        g_minus.extra_bounds = ((-0.45, 0.0),)
    g_plus = _spec_for_band(spectrum, "G+", db, laser_ev, profile=profile)
    d_prime = _spec_for_band(spectrum, "D'", db, laser_ev, profile=profile)
    return FitModel(
        peaks=[g_minus, g_plus, d_prime],
        window=(lo, hi),
        background=background,
        name="swcnt_g",
    )


def _rbm_model(
    spectrum: Spectrum,
    seeds: Optional[Sequence[PeakMeasurement]],
    background: str,
    db: Database,
    window: Optional[tuple[float, float]],
    profile: Optional[str] = None,
) -> FitModel:
    band = db.band("RBM")
    lo, hi = window or band.window
    lo = max(lo, spectrum.range[0])
    hi = min(hi, spectrum.range[1])
    if hi - lo < 40.0:
        raise ValueError(
            f"el espectro solo cubre {lo:.0f}–{hi:.0f} cm⁻¹ de la región RBM; "
            "hace falta al menos una ventana de 40 cm⁻¹"
        )
    found = list(seeds) if seeds is not None else find_peaks(
        spectrum, window=(lo, hi), min_distance_cm=5.0, min_fwhm_cm=3.0
    )
    found = [p for p in found if lo <= p.position <= hi]
    if not found:
        raise ValueError(
            f"no se han encontrado picos en la región RBM ({lo:.0f}–{hi:.0f} cm⁻¹); "
            "no hay nada que deconvolucionar"
        )
    peaks = []
    for i, peak in enumerate(sorted(found, key=lambda p: p.position)):
        fwhm = peak.fwhm or 10.0
        peaks.append(
            PeakSpec(
                name=f"RBM{i + 1}",
                profile=profile or "lorentzian",
                centre=peak.position,
                height=peak.height,
                fwhm=float(np.clip(fwhm, 3.0, 25.0)),
                centre_bounds=(peak.position - 8.0, peak.position + 8.0),
                height_bounds=(0.0, peak.height * 20.0),
                fwhm_bounds=(2.0, 40.0),
                band="RBM",
            )
        )
    return FitModel(peaks=peaks, window=(lo, hi), background=background, name="rbm")


def _two_d_model(
    spectrum: Spectrum,
    seeds: Optional[Sequence[PeakMeasurement]],
    background: str,
    db: Database,
    window: Optional[tuple[float, float]],
    profile: Optional[str] = None,
) -> FitModel:
    laser_ev = spectrum.laser_ev
    band = db.band("2D")
    lo, hi = window or band.window_at(laser_ev, pad=60.0)
    _require_coverage(spectrum, lo, hi, "two_d")
    if seeds:
        specs = []
        for i, peak in enumerate(sorted(seeds, key=lambda p: p.position)):
            specs.append(
                PeakSpec(
                    name=f"2D_{i + 1}",
                    profile=profile or "lorentzian",
                    centre=peak.position,
                    height=peak.height,
                    fwhm=float(np.clip(peak.fwhm or 35.0, 10.0, 120.0)),
                    centre_bounds=(peak.position - 25.0, peak.position + 25.0),
                    height_bounds=(0.0, peak.height * 20.0),
                    fwhm_bounds=(10.0, 150.0),
                    band="2D",
                )
            )
    else:
        specs = [_spec_for_band(spectrum, "2D", db, laser_ev, profile=profile)]
    return FitModel(peaks=specs, window=(lo, hi), background=background, name="two_d")


@dataclass
class ModelComparison:
    """Several fits of the same window, ranked."""

    results: dict[str, FitResult]
    ranking: list[tuple[str, float]]
    """``(preset, criterion value)``, best first."""
    criterion: str
    best: str
    verdict: str

    def summary(self) -> str:
        lines = [f"Comparación de modelos por {self.criterion} (menor es mejor):", ""]
        for name, value in self.ranking:
            result = self.results[name]
            mark = "←" if name == self.best else " "
            lines.append(
                f"  {mark} {PRESET_LABELS.get(name, name):<42s} "
                f"{self.criterion}={value:9.1f}  R²={result.r_squared:.5f}  "
                f"{result.n_parameters} par."
            )
        lines.append("")
        lines.append(self.verdict)
        return "\n".join(lines)


def compare_models(
    spectrum: Spectrum,
    presets: Iterable[str] = ("two_band", "three_band", "four_band", "five_band"),
    criterion: str = "bic",
    db: Optional[Database] = None,
    metallic: bool = False,
    profile: Optional[str] = None,
    **fit_kwargs,
) -> ModelComparison:
    """Fit several deconvolution models and rank them.

    R² always improves when a component is added, so it cannot choose the
    number of components. The Bayesian information criterion penalises each
    extra parameter by ``ln(n)``, which for a thousand-point spectrum is a
    stiff penalty and gives a conservative answer — the right bias when the
    failure mode is over-fitting. AIC penalises by 2 and is more permissive.

    Parameters
    ----------
    spectrum:
        Preprocessed spectrum.
    presets:
        Which models to compare. Only presets covering the same window are
        comparable: an information criterion computed over different data
        is meaningless, so models whose window differs from the first one's
        are refitted on the widest common window.
    criterion:
        ``"bic"`` or ``"aic"``.
    db:
        Loaded database.
    metallic:
        Forwarded to :func:`build_model`, so a ``swcnt_full`` candidate gets
        a Breit–Wigner–Fano G⁻ when the tubes are metallic.
    profile:
        Forwarded to :func:`build_model`. Every model in a comparison must
        use the same lineshape, or the information criteria are comparing
        two things at once.
    **fit_kwargs:
        Passed to :func:`~ramancarbon.models.fitting.fit_model`.

    Returns
    -------
    ModelComparison

    Raises
    ------
    ValueError
        If no preset could be fitted, or the criterion is unknown.
    """
    if criterion not in {"aic", "bic"}:
        raise ValueError(f"criterion must be 'aic' or 'bic', got {criterion!r}")
    database = db or load_database()
    names = [p for p in presets]
    if not names:
        raise ValueError("no presets to compare")

    laser_ev = spectrum.laser_ev
    windows = [
        _shifted_window(PRESET_WINDOWS[p], database, laser_ev)
        for p in names
        if p in PRESET_WINDOWS
    ]
    common = (min(w[0] for w in windows), max(w[1] for w in windows))
    common = (max(common[0], spectrum.range[0]), min(common[1], spectrum.range[1]))

    results: dict[str, FitResult] = {}
    failures: dict[str, str] = {}
    for name in names:
        try:
            model = build_model(
                spectrum,
                preset=name,
                metallic=metallic,
                db=database,
                window=common,
                profile=profile,
            )
            results[name] = fit_model(spectrum, model, **fit_kwargs)
        except ValueError as exc:
            failures[name] = str(exc)
    if not results:
        raise ValueError(
            "no se ha podido ajustar ningún modelo: "
            + "; ".join(f"{k}: {v}" for k, v in failures.items())
        )

    ranking = sorted(
        ((name, getattr(result, criterion)) for name, result in results.items()),
        key=lambda item: item[1],
    )
    best = ranking[0][0]
    verdict = _comparison_verdict(ranking, results, best, criterion)
    return ModelComparison(
        results=results, ranking=ranking, criterion=criterion, best=best, verdict=verdict
    )


def _comparison_verdict(
    ranking: list[tuple[str, float]],
    results: dict[str, FitResult],
    best: str,
    criterion: str,
) -> str:
    """Say how decisive the comparison was, in words."""
    if len(ranking) < 2:
        return f"Solo se ha ajustado un modelo ({PRESET_LABELS.get(best, best)})."
    gap = ranking[1][1] - ranking[0][1]
    label = PRESET_LABELS.get(best, best)
    runner = PRESET_LABELS.get(ranking[1][0], ranking[1][0])
    if gap > 10:
        head = (
            f"«{label}» gana con claridad (Δ{criterion.upper()} = {gap:.0f} sobre "
            f"«{runner}»)."
        )
    elif gap > 2:
        head = (
            f"«{label}» es preferible a «{runner}», pero el margen es moderado "
            f"(Δ{criterion.upper()} = {gap:.1f})."
        )
    else:
        head = (
            f"«{label}» y «{runner}» son indistinguibles "
            f"(Δ{criterion.upper()} = {gap:.1f}): elige el más simple y no leas "
            "significado físico en las componentes extra."
        )
    warnings = results[best].warnings
    if any("degener" in w for w in warnings):
        head += (
            " Ojo: el modelo ganador tiene parámetros casi degenerados, así que "
            "sus áreas individuales no están determinadas de forma independiente."
        )
    return head


__all__ = [
    "D_REGION_LADDER",
    "G_REGION_LADDER",
    "REGION_BOUNDS",
    "ModelComparison",
    "PRESETS",
    "PRESET_BANDS",
    "PRESET_LABELS",
    "PRESET_WINDOWS",
    "build_model",
    "build_region_model",
    "compare_models",
]


# ----------------------------------------------------------------------
# free-form region models
# ----------------------------------------------------------------------
#: Order in which named components enter the D region as the count grows.
#:
#: This is the order the literature adds them in, not an arbitrary one.
#: One component is the D band itself. The second is D3, the amorphous
#: envelope that fills the D–G valley. The third is D4, the sp³/polyene
#: shoulder below 1250 cm⁻¹. Together those three are the D-region half of
#: the Sadezky five-band model for soot and disordered carbons — which is
#: what "three peaks in the D band" usually means in practice.
D_REGION_LADDER = ("D", "D3", "D4")

#: Order in which named components enter the G region.
#:
#: G alone, then D′ — the defect-activated shoulder above it. A third
#: component is the curvature-split G⁻ of a nanotube; beyond that there is
#: no named band left and extra components are unnamed.
G_REGION_LADDER = ("G", "D'", "G-")

#: Where each region is taken to start and end, cm⁻¹ at 2.33 eV.
REGION_BOUNDS = {"D": (1050.0, 1500.0), "G": (1500.0, 1700.0)}


def build_region_model(
    spectrum: Spectrum,
    n_d: int = 3,
    n_g: int = 2,
    profile: Optional[str] = "pseudo_voigt",
    background: str = "linear",
    window: Optional[tuple[float, float]] = None,
    db: Optional[Database] = None,
    metallic: bool = False,
) -> FitModel:
    """Build a D–G model with a chosen number of components in each region.

    Many groups fit a fixed convention rather than a named preset — three
    components across the D region and two across the G region is a common
    one — and the presets in this module cannot express an arbitrary
    choice. This does.

    Components are **named where the physics names them** and unnamed
    beyond that. The first three in the D region are D, D3 and D4, in that
    order, because that is the order the literature adds them in and each
    one carries a meaning, a default lineshape and a position window from
    the database. The first three in the G region are G, D′ and G⁻. Ask
    for more than the ladder provides and the extras come back as ``Dx1``,
    ``Gx1`` and so on, evenly spaced in the gaps, with wide bounds and no
    band assignment — because a component with no name has no physical
    interpretation and should not be given one by the software.

    That distinction is carried through: unnamed components appear in the
    fit and in the exports, but they do not feed the band assignment, the
    ratios or the classifier.

    Parameters
    ----------
    spectrum:
        The spectrum to be fitted.
    n_d, n_g:
        How many components in the D and G regions. Both must be at least
        one; the sum is capped at 12, beyond which no spectrum constrains
        the model.
    profile:
        Lineshape for every component. ``"pseudo_voigt"`` by default here
        — unlike the named presets, which keep each band's own default —
        because a free-form model is usually being fitted to broad,
        overlapping material where letting the fit choose the shape is the
        safer option.
    background:
        Polynomial background inside the fit.
    window:
        Fit window. Defaults to the union of the two regions, padded.
    db:
        Loaded database.
    metallic:
        Give G⁻, if the G count reaches three, a Breit–Wigner–Fano profile.

    Returns
    -------
    FitModel

    Raises
    ------
    ValueError
        If the counts are out of range or the spectrum does not cover the
        window.
    """
    database = db or load_database()
    if n_d < 1 or n_g < 1:
        raise ValueError("hace falta al menos una componente en cada región")
    if n_d + n_g > 12:
        raise ValueError(
            f"{n_d + n_g} componentes es más de lo que un espectro Raman puede "
            "determinar; por encima de ~8 las áreas dejan de ser independientes"
        )
    if profile is not None:
        profile = resolve_profile(profile)

    laser_ev = spectrum.laser_ev
    shift = 0.0
    if laser_ev is not None:
        shift = database.band("D").dispersion * (laser_ev - database.reference_laser_ev)

    d_low, d_high = (v + shift for v in REGION_BOUNDS["D"])
    g_low, g_high = (v + shift for v in REGION_BOUNDS["G"])
    lo, hi = window or (d_low - 50.0, g_high + 50.0)
    _require_coverage(spectrum, lo, hi, f"{n_d}D+{n_g}G")

    peaks: list[PeakSpec] = []
    peaks.extend(
        _region_specs(
            spectrum, n_d, D_REGION_LADDER, (d_low, d_high), "Dx",
            database, laser_ev, profile, metallic,
        )
    )
    peaks.extend(
        _region_specs(
            spectrum, n_g, G_REGION_LADDER, (g_low, g_high), "Gx",
            database, laser_ev, profile, metallic,
        )
    )
    peaks.sort(key=lambda spec: spec.centre)
    return FitModel(
        peaks=peaks, window=(lo, hi), background=background,
        name=f"{n_d}D+{n_g}G",
    )


def _region_specs(
    spectrum: Spectrum,
    count: int,
    ladder: Sequence[str],
    bounds: tuple[float, float],
    prefix: str,
    db: Database,
    laser_ev: Optional[float],
    profile: Optional[str],
    metallic: bool,
) -> list[PeakSpec]:
    """Named components from the ladder, then evenly spaced unnamed ones."""
    specs: list[PeakSpec] = []
    for key in ladder[: min(count, len(ladder))]:
        chosen = "bwf" if (key == "G-" and metallic) else profile
        specs.append(_spec_for_band(spectrum, key, db, laser_ev, profile=chosen))

    extra = count - len(specs)
    if extra <= 0:
        return specs

    low, high = bounds
    taken = sorted(spec.centre for spec in specs)
    for index, centre in enumerate(_fill_gaps(low, high, taken, extra), start=1):
        observed = spectrum.max_in(centre - 40.0, centre + 40.0)
        height = observed[1] if observed and observed[1] > 0 else 1.0
        specs.append(
            PeakSpec(
                name=f"{prefix}{index}",
                profile=profile or "pseudo_voigt",
                centre=centre,
                height=height,
                fwhm=80.0,
                centre_bounds=(max(low, centre - 60.0), min(high, centre + 60.0)),
                height_bounds=(0.0, height * 20.0),
                fwhm_bounds=(10.0, 300.0),
                band=None,  # deliberately unnamed: no physical assignment
            )
        )
    return specs


def _fill_gaps(
    low: float, high: float, taken: Sequence[float], count: int
) -> list[float]:
    """Positions for extra components, in the widest gaps between the named ones.

    Spreading them evenly across the whole region would stack them on top
    of the named bands. Putting each new one at the midpoint of whatever
    gap is currently largest keeps them apart and away from components
    that already exist.
    """
    edges = sorted([low, *taken, high])
    positions: list[float] = []
    for _ in range(count):
        widths = [(edges[i + 1] - edges[i], i) for i in range(len(edges) - 1)]
        width, index = max(widths)
        centre = 0.5 * (edges[index] + edges[index + 1])
        positions.append(centre)
        edges.insert(index + 1, centre)
    return sorted(positions)
