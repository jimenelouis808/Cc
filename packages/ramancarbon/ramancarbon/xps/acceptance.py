"""The checklist before a fit is accepted, applied to the numbers.

The master tables end with ten questions to ask before believing an XPS
fit, and every one of them is answerable from what the fit already
returns. Until now none of them was asked. The default region model put
in every state the database lists — fourteen for C 1s — and reported
per-cent compositions for all of them; on a real PHI Quantera spectrum
four of those states came back with binding-energy uncertainties of
278 822, 5 613 136 and 203 002 971 eV, which is the arithmetic saying the
parameter is not determined at all, printed as a chemical result.

The questions, and where each is answered:

============================================== =========================
"¿El BE está dentro del rango químico?"        the state's own window
"¿FWHM es razonable?"                          its published range, and
                                               the instrument's floor
"¿Hay doblete? Aplicar ΔSO, ratio y FWHM"      the region's Doublet
"¿El pico requiere otro elemento?"             ``requires_any``
"¿Hay evidencia en otra región?"               named, for the reader
"¿El área cambia con perturbaciones?"          ``area_drift``
"¿Agregar el pico solo mejora residual?"       BIC, in `selection`
"¿Es metal de transición?"                     multiplet regions
"¿Es π–π*?"                                    excluded from the %
"¿Hay dos interpretaciones equivalentes?"      correlations, families
============================================== =========================

One question is not on the list and belongs with them: **did a parameter
finish against its own limit?** That is a stronger statement than "this
value is unusual" — it means the optimiser wanted to go further, so the
model is wrong and not the limit — and it is only answerable because the
fit now records the bounds it used. On the C 1s above, four parameters
are pinned after the state selection has done its work, including the
asymmetry of the sp² component at exactly zero. The reference table says
of that line: *"en grafito y grafeno la línea es ASIMÉTRICA … ajustarla
con una forma simétrica obliga a meter una componente extra a ~285.5 que
luego se lee como C sp3. Es el error más común en XPS de carbono."* An
asymmetry pinned at zero is that error happening, and nothing was saying
so.

The verdict is deliberately not a score. A fit is ALTA, MODERADA, BAJA or
AMBIGUA, and the last of those is its own category rather than a low
score because it means something different: not "this is poorly
measured" but "these data are equally compatible with a different
answer", which no amount of counting longer will fix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .fitting import XPSFitResult, resolution_floor
from .spectrum import XPSSpectrum
from .tables import XPSDatabase, load_xps_database

#: Share of the total peak area below which a component is not a state.
NEGLIGIBLE_AREA = 0.02

#: Correlation at which two components stop being independently measured.
DEGENERATE = 0.95

#: How close to a bound counts as being on it, as a share of the range.
PINNED = 0.02

#: How many times the counting noise a component's height must reach.
MIN_SIGNAL_SIGMA = 3.0

#: Largest relative area drift between background iterations before the
#: areas are called unstable. The checklist asks for it by name: "¿el
#: área cambia mucho con perturbaciones? → marcar inestable".
MAX_AREA_DRIFT = 0.05

#: Tolerance on a doublet's splitting, in eV, and on its area ratio.
SPLIT_TOLERANCE = 0.15
RATIO_TOLERANCE = 0.25

#: Regions whose 2p lines are multiplet-split, where a list of symmetric
#: peaks is not a model. The tables say it in capitals for both: "NO usar
#: lista simple de gaussianas".
MULTIPLET_REGIONS = ("Fe 2p", "Ni 2p", "Co 2p", "Mn 2p", "Cr 2p", "Cu 2p")


@dataclass
class Finding:
    """One thing wrong, or worth knowing, about a fit."""

    code: str
    severity: str
    """``grave``, ``aviso`` or ``info``."""
    component: Optional[str]
    message: str

    def __str__(self) -> str:
        where = f"{self.component}: " if self.component else ""
        return f"[{self.severity}] {where}{self.message}"


@dataclass
class Audit:
    """The findings and what they add up to."""

    findings: list[Finding] = field(default_factory=list)
    confidence: str = "MODERADA"
    reason: str = ""

    @property
    def grave(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "grave"]

    @property
    def avisos(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "aviso"]

    def __str__(self) -> str:
        head = f"Confianza del ajuste: {self.confidence} — {self.reason}"
        if not self.findings:
            return head + "\n  sin observaciones"
        return head + "\n" + "\n".join(f"  {f}" for f in self.findings)


def aicc(aic: float, n_points: int, n_parameters: int) -> float:
    """AIC with the finite-sample correction.

    An XPS region is a few hundred points and a model is a dozen or more
    parameters, which is exactly where the ``2k(k+1)/(n−k−1)`` term stops
    being negligible. Undefined when the parameters outnumber the data, and
    that returns infinity rather than the uncorrected value: a model with
    more parameters than points is not a model, and hiding it behind an
    AIC that still looks finite is how it gets accepted.
    """
    if n_points <= n_parameters + 1:
        return math.inf
    return aic + (2.0 * n_parameters * (n_parameters + 1.0)
                  / (n_points - n_parameters - 1.0))


def _state_of(component, database: XPSDatabase):
    """The catalogued state a fitted component came from, or ``None``.

    Looked up by region and key, and forgiving about both: a component
    keeps the region label it was built under, and a fit the user
    assembled by hand may carry a state key the database does not have.
    A missing entry means the literature checks are skipped for that
    component, not that the audit fails.
    """
    if not component.state:
        return None
    for region in (component.line, getattr(component, "region_label", None)):
        if not region:
            continue
        try:
            return database.state(region, component.state)
        except Exception:                        # noqa: BLE001
            continue
    for region in database.region_names():
        try:
            return database.state(region, component.state)
        except Exception:                        # noqa: BLE001
            continue
    return None


def _value_of(component, parameter: str) -> Optional[float]:
    if parameter in ("centre", "height", "fwhm"):
        return float(getattr(component, parameter))
    extras = dict(zip(component.extra_names, component.extra))
    value = extras.get(parameter)
    return None if value is None else float(value)


def _check_bounds(result: XPSFitResult, findings: list[Finding]) -> None:
    """Parameters that finished on a limit."""
    for key, limits in (result.bounds or {}).items():
        name, _, parameter = key.partition(".")
        component = result.component(name)
        if component is None:
            continue
        value = _value_of(component, parameter)
        if value is None:
            continue
        low, high = float(limits[0]), float(limits[1])
        span = high - low
        if span <= 0 or not np.isfinite(span):
            continue
        if abs(value - low) <= PINNED * span:
            edge, limit = "inferior", low
        elif abs(value - high) <= PINNED * span:
            edge, limit = "superior", high
        else:
            continue
        extra = ""
        if parameter == "asymmetry" and limit == 0.0:
            extra = (". Una asimetría en cero convierte el perfil metálico en "
                     "una lorentziana simétrica, y entonces la cola del pico "
                     "tiene que taparse con otra componente que se lee como "
                     "un estado químico que no está")
        findings.append(Finding(
            "pegado-al-limite", "grave", name,
            f"«{parameter}» terminó en su límite {edge} ({limit:g}): el "
            f"ajuste quería ir más allá, así que lo que está mal es el "
            f"modelo, no el límite{extra}"))


def _check_windows(result: XPSFitResult, database: XPSDatabase,
                   findings: list[Finding]) -> None:
    """Binding energies and widths against the literature ranges."""
    for component in result.components:
        if not component.state or component.satellite:
            # A shake-up satellite sits several eV ABOVE its parent by
            # construction — that offset is what makes it a satellite — so
            # measuring it against the parent state's window accuses it of
            # exactly the thing it is supposed to do. On an Fe 2p fit the
            # audit reported the Fe(III) satellite at 720.0 eV as outside
            # the 710.4–712.0 eV window of Fe(III), which is true and
            # means nothing.
            continue
        state = _state_of(component, database)
        if state is None:
            continue
        low, high = state.window
        if not low - 0.05 <= component.centre <= high + 0.05:
            findings.append(Finding(
                "be-fuera-de-rango", "grave", component.name,
                f"BE {component.centre:.2f} eV fuera del rango químico de "
                f"{state.name} ({low:.1f}–{high:.1f}): o es otra especie o "
                "la referencia de carga está mal"))
        width = component.true_fwhm or component.fwhm
        wide_low, wide_high = state.fwhm
        if width < wide_low - 0.05:
            findings.append(Finding(
                "fwhm-estrecha", "aviso", component.name,
                f"FWHM {width:.2f} eV por debajo de lo publicado para "
                f"{state.name} ({wide_low:.1f}–{wide_high:.1f})"))
        elif width > wide_high + 0.05:
            findings.append(Finding(
                "fwhm-ancha", "aviso", component.name,
                f"FWHM {width:.2f} eV por encima de lo publicado para "
                f"{state.name} ({wide_low:.1f}–{wide_high:.1f}): suele ser "
                "una componente que falta, o el fondo"))


def _check_resolution(result: XPSFitResult, spectrum: Optional[XPSSpectrum],
                      findings: list[Finding]) -> None:
    """Components narrower than the instrument can produce."""
    if spectrum is None:
        return
    floor = resolution_floor(spectrum)
    if not floor:
        findings.append(Finding(
            "sin-resolucion", "info", None,
            "no se ha declarado la energía de paso, así que no se puede "
            "comprobar si alguna componente sale más estrecha de lo que el "
            "equipo puede dar. Es el dato que una exportación en texto tira"))
        return
    for component in result.components:
        width = component.true_fwhm or component.fwhm
        if width < 0.9 * floor:
            findings.append(Finding(
                "bajo-la-resolucion", "grave", component.name,
                f"FWHM {width:.2f} eV por debajo de la resolución del equipo "
                f"({floor:.2f} eV a {spectrum.pass_energy:g} eV de paso, "
                f"fuente {'monocromada' if spectrum.monochromated else 'sin monocromar'}): "
                "ninguna especie real puede salir así. Comprueba la energía de "
                "paso declarada antes de tocar el modelo: es el número del que "
                "cuelga este límite"))


def _check_signal(result: XPSFitResult, spectrum: Optional[XPSSpectrum],
                  findings: list[Finding]) -> None:
    """Components whose height does not clear the counting noise.

    A different measurement from "its area is a small share of the total",
    and it catches a different failure: on a real N 1s the oxidised-nitrogen
    component held 3.7 % of the area — comfortably above the area floor —
    with a peak height under three times the noise. The fitter already
    warned about it and the audit still called the region ALTA, which is
    the kind of disagreement that makes a verdict worth nothing.

    Three sigma, and the reason is the same as everywhere else in this
    package: least squares returns as many peaks as it is given, so a
    component has to clear the noise before it is a species.
    """
    if spectrum is None:
        return
    noise = spectrum.noise_estimate()
    if not noise or not np.isfinite(noise) or noise <= 0:
        return
    for component in result.components:
        if component.satellite:
            continue
        if component.peak_height < MIN_SIGNAL_SIGMA * noise:
            findings.append(Finding(
                "bajo-el-ruido", "grave", component.name,
                f"su altura ({component.peak_height:.0f} cuentas) no llega a "
                f"{MIN_SIGNAL_SIGMA:.0f} veces el ruido ({noise:.0f}): no se "
                "sostiene como especie, por mucho que su área parezca un "
                "porcentaje respetable"))


def _check_areas(result: XPSFitResult, findings: list[Finding]) -> None:
    total = sum(abs(item.area) for item in result.components) or 1.0
    for component in result.components:
        if component.satellite:
            continue
        fraction = abs(component.area) / total
        if fraction < NEGLIGIBLE_AREA:
            findings.append(Finding(
                "area-despreciable", "grave", component.name,
                f"su área es el {100.0 * fraction:.1f} % del total: no es un "
                "estado químico, es una corrección al fondo con nombre"))
    if result.area_drift > MAX_AREA_DRIFT:
        findings.append(Finding(
            "areas-inestables", "grave", None,
            f"las áreas se movieron un {100.0 * result.area_drift:.1f} % en la "
            "última iteración del fondo: el reparto entre componentes depende "
            "del fondo y no de la química"))


def _check_degeneracy(result: XPSFitResult, findings: list[Finding]) -> None:
    pairs = [(names, value) for names, value in (result.correlations or {}).items()
             if abs(float(value)) >= DEGENERATE]
    if not pairs:
        return
    worst = ", ".join(f"{a}–{b} ({v:+.2f})" for (a, b), v in pairs[:4])
    findings.append(Finding(
        "no-separables", "grave", None,
        f"parámetros que los datos no separan: {worst}. Las áreas de esas "
        "componentes no están determinadas por separado, así que sus "
        "porcentajes no se pueden citar como estados independientes"))


def _check_doublets(result: XPSFitResult, findings: list[Finding]) -> None:
    """ΔSO, area ratio and width, for the regions that have a partner."""
    for component in result.components:
        doublet = component.doublet
        if doublet is None or component.satellite:
            continue
        expected = doublet.expected_ratio
        if expected is None:
            continue
        # The partner is inside the same component here — the fitter builds
        # a doublet as one object with a fixed ratio — so what is worth
        # checking is that the ratio and splitting in force are the
        # published ones, not that the fit found them.
        if abs(doublet.ratio - expected) > RATIO_TOLERANCE * expected:
            findings.append(Finding(
                "doblete-ratio", "grave", component.name,
                f"la razón de áreas del doblete es {doublet.ratio:.2f} y la "
                f"degeneración exige {expected:.2f}"))


def _check_multiplets(result: XPSFitResult, findings: list[Finding]) -> None:
    label = (result.region_label or "")
    if not any(label.startswith(region.split()[0] + " " + region.split()[1][:2])
               or label.startswith(region) for region in MULTIPLET_REGIONS):
        return
    symmetric = [c.name for c in result.components
                 if c.profile in ("gl", "sgl", "gauss", "lorentz")
                 and not c.satellite]
    if symmetric:
        findings.append(Finding(
            "multiplete", "grave", None,
            f"{label} está ajustada con componentes simétricas "
            f"({', '.join(symmetric[:4])}). Las tablas lo dicen en "
            "mayúsculas: en un 2p de metal de transición los estados están "
            "desdoblados por multipletes y acompañados de satélites, así que "
            "una lista de gaussianas reparte una estructura física entre "
            "«estados químicos» inventados. Hace falta un modelo de "
            "referencia medido"))


def _check_loss_features(result: XPSFitResult, findings: list[Finding]) -> None:
    """π–π* and shake-up intensity must not be counted as chemistry."""
    total = sum(abs(c.area) for c in result.components) or 1.0
    loss = [c for c in result.components
            if c.satellite or (c.state or "").startswith("pi-pi")]
    if not loss:
        return
    share = sum(abs(c.area) for c in loss) / total
    findings.append(Finding(
        "perdidas", "info", None,
        f"{len(loss)} componente(s) de pérdida (π–π* o shake-up) con el "
        f"{100.0 * share:.1f} % del área. No son especies: su intensidad no "
        "entra en el reparto químico, y la cuantificación las excluye"))


def _check_cross_evidence(result: XPSFitResult, database: XPSDatabase,
                          present: Optional[Sequence[str]],
                          complete: bool,
                          findings: list[Finding]) -> None:
    """States that need another element, and whether it is there.

    Presence and absence are not symmetric here. A measured region proves
    its own element is in the sample, so a C–S component in an S 2p fit of
    a sample that also has a C 1s region needs no further defence. Absence
    is the hard direction: only a survey can say an element is NOT there,
    which is why ``complete`` gates the severe finding and everything else
    gets an ``aviso`` naming what to check.
    """
    known = None if present is None else {str(s).strip() for s in present}
    for component in result.components:
        if not component.state:
            continue
        state = _state_of(component, database)
        if state is None or not state.requires_any:
            continue
        need = " o ".join(state.requires_any)
        if known is not None and known.intersection(state.requires_any):
            continue                              # confirmed: nothing to say
        if known is not None and complete:
            findings.append(Finding(
                "falta-el-elemento", "grave", component.name,
                f"{state.name} exige {need} y el survey dice que la muestra no "
                "lo tiene: esta componente ajusta una química que no puede "
                "existir"))
        else:
            findings.append(Finding(
                "sin-confirmar", "aviso", component.name,
                f"{state.name} exige {need}, y no hay constancia de que la "
                "muestra lo tenga. Mídelo o míralo en el survey: mientras no, "
                "esta componente es una hipótesis, no un resultado"))


def _verdict(findings: Sequence[Finding], result: XPSFitResult) -> tuple[str, str]:
    """Four outcomes, and AMBIGUA is not the worst one — it is a different one."""
    if not result.success:
        return "BAJA", "el ajuste no convergió"
    codes = {f.code for f in findings if f.severity == "grave"}
    if {"no-separables", "multiplete"} & codes:
        return ("AMBIGUA",
                "hay componentes que los datos no separan: existen reparticiones "
                "distintas igual de compatibles con el espectro, y medir más "
                "tiempo no lo resuelve")
    grave = [f for f in findings if f.severity == "grave"]
    if grave:
        return "BAJA", (f"{len(grave)} observación(es) graves: "
                        + ", ".join(sorted(codes)))
    if any(f.severity == "aviso" for f in findings):
        return "MODERADA", "hay avisos, ninguno descalificante"
    if result.reduced_chi2 > 3.0:
        return "MODERADA", (f"χ²_red = {result.reduced_chi2:.2f}: el modelo "
                            "deja estructura sin ajustar")
    return "ALTA", "dentro de los rangos, sin parámetros indeterminados"


def audit_region(
    result: XPSFitResult,
    spectrum: Optional[XPSSpectrum] = None,
    present: Optional[Sequence[str]] = None,
    complete: bool = False,
    database: Optional[XPSDatabase] = None,
) -> Audit:
    """Run the acceptance checklist over one fitted region.

    Parameters
    ----------
    result:
        The fit.
    spectrum:
        The region it was fitted to, for the instrument resolution. Without
        it that check is reported as skipped rather than passed.
    present:
        Elements known to be in the sample: the survey's, plus those whose
        regions were measured. A state whose requirement is in this set
        needs no further defence.
    complete:
        Whether ``present`` is the whole composition, which only a survey
        can establish. Without it, a missing element is an unverified
        hypothesis (``aviso``); with it, a state needing an element the
        survey did not find is fitting a chemistry that cannot exist
        (``grave``).
    database:
        Chemical-state database.
    """
    database = database or load_xps_database()
    findings: list[Finding] = []
    if not result.components:
        return Audit([Finding("sin-componentes", "grave", None,
                              "el ajuste no dejó ninguna componente")],
                     "BAJA", "no hay nada que evaluar")

    _check_bounds(result, findings)
    _check_windows(result, database, findings)
    _check_resolution(result, spectrum, findings)
    _check_signal(result, spectrum, findings)
    _check_areas(result, findings)
    _check_degeneracy(result, findings)
    _check_doublets(result, findings)
    _check_multiplets(result, findings)
    _check_loss_features(result, findings)
    _check_cross_evidence(result, database, present, complete, findings)

    if np.isfinite(result.aic):
        findings.append(Finding(
            "criterios", "info", None,
            f"{len(result.components)} componentes, {result.n_parameters} "
            f"parámetros, χ²_red = {result.reduced_chi2:.2f}, "
            f"AICc = {aicc(result.aic, result.energy.size, result.n_parameters):.1f}, "
            f"BIC = {result.bic:.1f}"))

    confidence, reason = _verdict(findings, result)
    return Audit(findings, confidence, reason)


__all__ = [
    "DEGENERATE",
    "MAX_AREA_DRIFT",
    "MIN_SIGNAL_SIGMA",
    "MULTIPLET_REGIONS",
    "NEGLIGIBLE_AREA",
    "PINNED",
    "Audit",
    "Finding",
    "aicc",
    "audit_region",
]
