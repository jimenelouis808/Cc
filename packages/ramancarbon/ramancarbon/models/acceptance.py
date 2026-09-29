"""Whether a deconvolution may be believed, and how much.

A fit that converged is not a fit that means anything. This module turns
the rules a Raman analyst applies by eye into checks that run on every
result: is each component physically placed, is its width plausible, does
it carry enough area to be real, and does the model as a whole earn its
parameters.

The rules come from the reference tables a user of this package supplied
for 532 nm nanocarbon work, and they are deliberately conservative. Two
of them matter more than the rest:

* **R-squared alone is not an argument.** Adding components always
  raises it. What decides is an information criterion, which charges for
  every parameter, together with the physics.
* **A component pinned against its own bound is a warning, not a
  result.** The optimiser wanted to go further and the model would not
  let it, which means the model is wrong, not that the bound is right.

Nothing here rejects a fit on its own. Each finding carries a severity
and an explanation, and the caller -- or the person reading the report --
decides. Silently dropping a component would be exactly the kind of
automatic assignment these rules exist to prevent.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

#: Plausible FWHM range per band, in cm-1, at 532 nm.
#:
#: These are *orientation*, not law: the reference tables call them
#: "FWHM orientativa" and warn against fixed universal values. A width
#: outside the range is reported, never clamped, because a genuinely
#: broad D band in a turbostratic carbon is a measurement and not an
#: error. What the check catches is the other case -- a width driven to
#: an extreme because it is mopping up intensity that belongs to a
#: component the model does not have.
FWHM_WINDOWS: dict[str, tuple[float, float]] = {
    "RBM": (5.0, 40.0),
    "D": (40.0, 180.0),
    "D4": (50.0, 250.0),
    "D3": (50.0, 200.0),
    "G": (10.0, 80.0),
    "G-": (10.0, 120.0),
    "G+": (10.0, 80.0),
    "D'": (20.0, 100.0),
    "2D": (20.0, 200.0),
    "D+G": (50.0, 200.0),
    "D+D'": (50.0, 200.0),
    "2D'": (20.0, 150.0),
}

#: Below this share of the strongest component's area, a component is
#: doing nothing that a slightly different baseline would not also do.
#: The tables list "area cercana a cero" as an overfitting alert; this
#: puts a number on "cercana".
NEGLIGIBLE_AREA_FRACTION = 0.02

#: How close to a bound counts as pinned, as a fraction of the bound's
#: own span.
PINNED_TOLERANCE = 0.01

#: How close to its own ceiling a width has to get before the gap stops
#: meaning anything, as a fraction of the ceiling.
#:
#: `PINNED_TOLERANCE` is deliberately tight, because "this parameter
#: stopped exactly on its bound" is a statement about the optimiser and
#: has to be literal. A width is the one parameter where that tightness
#: hides the finding: on a real disordered carbon the D band came back at
#: 200.0 of 200 in the three-band model, 195.7 in the four-band and 191.0
#: in the five-band -- the same wall in all three, and only the first was
#: reported, because the other two are 2 % and 5 % short of it. Those two
#: numbers are not measurements of a width either: they are where the
#: optimiser stopped pushing. Five per cent of the ceiling is the gap
#: below which the distinction is not worth making, and the check only
#: fires when the width is ALSO above its usual range, so a band that is
#: legitimately broad and nowhere near its bound says nothing.
NO_ROOM_FRACTION = 0.05

#: Correlation above which two parameters are not independently
#: determined, so their individual values must not be quoted.
DEGENERATE_CORRELATION = 0.95

SEVERITY = ("info", "aviso", "grave")
CONFIDENCE = ("HIGH", "MODERATE", "LOW", "AMBIGUOUS")


@dataclass(frozen=True)
class Finding:
    """One thing noticed about a fit."""

    code: str
    severity: str
    component: str | None
    message: str

    def __str__(self) -> str:
        where = f"{self.component}: " if self.component else ""
        return f"[{self.severity}] {where}{self.message}"


@dataclass
class Audit:
    """The findings, and what they add up to."""

    findings: list[Finding] = field(default_factory=list)
    confidence: str = "MODERATE"
    reason: str = ""

    @property
    def grave(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "grave"]

    @property
    def avisos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "aviso"]

    def __str__(self) -> str:
        head = f"Confianza: {self.confidence} — {self.reason}"
        if not self.findings:
            return head + "\n  sin observaciones"
        return head + "\n" + "\n".join(f"  {f}" for f in self.findings)


def aicc(aic: float, n_points: int, n_parameters: int) -> float:
    """AIC corrected for finite samples.

    The reference tables ask for AICc rather than AIC, and the reason is
    the small-sample regime: the correction is ``2k(k+1)/(n-k-1)``, which
    is negligible for a few thousand spectral points and decisive for a
    narrow window with a handful. Returning AIC unchanged when the
    correction is undefined (``n <= k + 1``) would hide exactly the case
    it exists for, so that returns infinity instead: a model with more
    parameters than the data can support is not a model.
    """
    if n_points <= n_parameters + 1:
        return math.inf
    return aic + (2.0 * n_parameters * (n_parameters + 1.0)
                  / (n_points - n_parameters - 1.0))


def _band_key(name: str) -> str:
    """The band a component name refers to, for the width windows."""
    clean = name.strip()
    for key in ("D+D'", "D+G", "2D'", "2D", "D'", "D4", "D3", "G+", "G-",
                "RBM", "D", "G"):
        if clean == key or clean.startswith((key + " ", key + "_")):
            return key
    return clean


def audit_fit(result, bounds: dict | None = None) -> Audit:
    """Check one fit against the acceptance rules.

    Parameters
    ----------
    result:
        A :class:`~ramancarbon.models.fitting.FitResult`.
    bounds:
        ``{f"{component}.{parameter}": (low, high)}`` used to report a
        parameter that finished against its own limit. Defaults to the
        limits the fit recorded on the result itself, so the check runs
        without the caller having to reconstruct them; passing a dict
        overrides those, and an empty dict skips the check.
    """
    if bounds is None:
        bounds = getattr(result, "bounds", None)
    findings: list[Finding] = []
    peaks = list(getattr(result, "peaks", []) or [])
    if not peaks:
        return Audit([Finding("sin-componentes", "grave", None,
                              "el ajuste no dejó ningún componente")],
                     "LOW", "no hay nada que evaluar")

    areas = [abs(float(getattr(p, "area", 0.0))) for p in peaks]
    largest = max(areas) if areas else 0.0

    for peak, area in zip(peaks, areas, strict=True):
        name = str(getattr(peak, "name", "?"))
        key = _band_key(name)
        width = float(getattr(peak, "fwhm", float("nan")))
        centre = float(getattr(peak, "centre", float("nan")))

        window = FWHM_WINDOWS.get(key)
        if window and np.isfinite(width):
            low, high = window
            if width < low:
                findings.append(Finding(
                    "fwhm-estrecha", "aviso", name,
                    f"FWHM {width:.1f} cm⁻¹ por debajo de lo habitual para "
                    f"{key} ({low:.0f}–{high:.0f}); un componente demasiado "
                    "estrecho suele estar ajustando ruido"))
            elif width > high:
                ceiling = _fwhm_ceiling(bounds, name)
                if ceiling is not None and width >= ceiling * (1.0 - NO_ROOM_FRACTION):
                    findings.append(Finding(
                        "anchura-sin-sitio", "grave", name,
                        f"FWHM {width:.1f} cm⁻¹ contra un techo de "
                        f"{ceiling:.0f}: el ajuste ha llevado la anchura "
                        f"hasta donde se le deja, así que {width:.0f} cm⁻¹ no "
                        "es una medida de la anchura sino el borde. Un "
                        f"componente tan ancho como {key} está aquí se está "
                        "comiendo intensidad que pertenece a otro: prueba un "
                        "preajuste con más componentes antes de leer ningún "
                        "cociente de este ajuste"))
                else:
                    findings.append(Finding(
                        "fwhm-ancha", "aviso", name,
                        f"FWHM {width:.1f} cm⁻¹ por encima de lo habitual para "
                        f"{key} ({low:.0f}–{high:.0f}); puede estar absorbiendo "
                        "intensidad de un componente que falta en el modelo"))

        if largest > 0 and area < NEGLIGIBLE_AREA_FRACTION * largest:
            findings.append(Finding(
                "area-despreciable", "grave", name,
                f"su área es el {100.0 * area / largest:.1f} % de la mayor; "
                "la tabla de sobreajuste llama a esto «área cercana a cero» "
                "y el componente probablemente no hace falta"))

        if not np.isfinite(centre) or not np.isfinite(width):
            findings.append(Finding(
                "no-finito", "grave", name,
                "centro o anchura no son números finitos"))

    if bounds:
        for peak in peaks:
            name = str(getattr(peak, "name", "?"))
            for parameter in ("centre", "fwhm", "height"):
                limits = bounds.get(f"{name}.{parameter}")
                value = getattr(peak, parameter, None)
                if not limits or value is None:
                    continue
                low, high = float(limits[0]), float(limits[1])
                span = high - low
                if span <= 0 or not np.isfinite(span):
                    continue
                if abs(float(value) - low) <= PINNED_TOLERANCE * span:
                    findings.append(Finding(
                        "pegado-al-limite", "grave", name,
                        f"«{parameter}» terminó en su límite inferior "
                        f"({low:g}): el ajuste quería ir más allá, de modo "
                        "que el modelo es el que está mal, no el límite"))
                elif abs(float(value) - high) <= PINNED_TOLERANCE * span:
                    findings.append(Finding(
                        "pegado-al-limite", "grave", name,
                        f"«{parameter}» terminó en su límite superior "
                        f"({high:g}): el ajuste quería ir más allá, de modo "
                        "que el modelo es el que está mal, no el límite"))

    correlations = getattr(result, "correlations", None) or {}
    degenerate = [(pair, value) for pair, value in correlations.items()
                  if abs(float(value)) >= DEGENERATE_CORRELATION]
    if degenerate:
        worst = ", ".join(f"{a}–{b} ({v:+.2f})" for (a, b), v in degenerate[:4])
        findings.append(Finding(
            "degenerado", "grave", None,
            f"parámetros no separables: {worst}. Las áreas individuales no "
            "están determinadas, así que no deben citarse por separado"))

    r2 = getattr(result, "r_squared", None)
    if r2 is not None and float(r2) > 0.999 and len(peaks) >= 5:
        findings.append(Finding(
            "r2-solo", "info", None,
            f"R² = {float(r2):.5f} con {len(peaks)} componentes: un R² alto "
            "no es argumento por sí mismo, compare AICc/BIC y estabilidad"))

    if not getattr(result, "success", True):
        findings.append(Finding(
            "no-converge", "grave", None,
            f"el optimizador no convergió: {getattr(result, 'message', '')}"))

    confidence, reason = _confidence(findings, result)
    return Audit(findings, confidence, reason)


def _fwhm_ceiling(bounds, name: str) -> Optional[float]:
    """The upper width bound the fit actually used, if it is known.

    ``bounds`` is the fit's own record of what each parameter was allowed
    to do, which is not the same thing as the band's usual range: the
    first is what constrained this fit and the second is what the
    literature reports. The check that calls this needs the first.
    """
    if not bounds:
        return None
    limits = bounds.get(f"{name}.fwhm")
    if not limits:
        return None
    high = float(limits[1])
    return high if np.isfinite(high) and high > 0 else None


def _confidence(findings: Sequence[Finding], result) -> tuple[str, str]:
    """Table 13 of the reference, applied to what was found.

    HIGH needs a clean fit, not merely a converged one. Anything the
    checks call `grave` takes it out of HIGH by definition, because every
    `grave` is a reason the numbers would change under a reasonable
    change of model or baseline.
    """
    graves = [f for f in findings if f.severity == "grave"]
    avisos = [f for f in findings if f.severity == "aviso"]

    if any(f.code in ("sin-componentes", "no-converge", "no-finito")
           for f in graves):
        return "LOW", "el ajuste no es utilizable"
    if any(f.code == "degenerado" for f in graves):
        return ("AMBIGUOUS",
                ("hay parámetros que los datos no separan: existen modelos "
                 "distintos igual de compatibles con el espectro"))
    if graves:
        detalle = "; ".join(sorted({f.code for f in graves}))
        return ("LOW", f"{len(graves)} observación(es) graves: {detalle}")
    if avisos:
        return ("MODERATE",
                (f"{len(avisos)} aviso(s) sobre anchuras o posiciones; el "
                 "espectro es compatible pero caben interpretaciones "
                 "alternativas"))
    r2 = getattr(result, "r_squared", None)
    if r2 is not None and float(r2) < 0.95:
        return ("MODERATE",
                (f"sin observaciones, pero R² = {float(r2):.4f} deja "
                 "estructura sin describir"))
    return ("HIGH",
            ("componentes físicamente situados, anchuras plausibles y "
             "parámetros separables"))


__all__ = [
    "CONFIDENCE",
    "DEGENERATE_CORRELATION",
    "FWHM_WINDOWS",
    "NEGLIGIBLE_AREA_FRACTION",
    "NO_ROOM_FRACTION",
    "SEVERITY",
    "Audit",
    "Finding",
    "aicc",
    "audit_fit",
]
