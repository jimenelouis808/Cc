"""Rietveld refinement: fitting the whole pattern, not the peaks.

The idea is old and simple. Compute the entire diffractogram from the
crystal structures, point by point, and adjust the structures and the
instrument until the calculated curve sits on top of the measured one. No
peak is ever integrated, so overlapping reflections — which in a
multi-phase nanomaterial means most of them — are handled by the model
instead of by a decision about where one peak ends and the next begins.

What can be refined, and why each one is here:

``scale``            one per phase; what weight fractions are computed from
``lattice``          a, b, c as relative factors, so one step size suits a
                     2.5 Å axis and a 14 Å one
``profile``          Caglioti U, V, W and the pseudo-Voigt mixing, **per
                     phase**, because a nanocrystalline carbon and a
                     well-crystallised selenide in the same sample have
                     genuinely different peak widths and forcing them to
                     share a profile makes both wrong
``preferred``        March–Dollase r; layered powders are textured as a
                     rule, not as an exception
``u_iso``            one overall displacement parameter per phase
``zero``             a constant 2θ offset — the diffractometer's zero
``displacement``     a cos θ offset — the sample sitting off the reference
                     plane, which is a *different* function of angle and
                     the reason a single zero never quite fixes a
                     mis-mounted sample
``background``       Chebyshev coefficients

Two warnings that the code enforces rather than merely prints.

**Refining everything at once diverges.** Scale and background are
correlated, background and displacement parameters are correlated, and
turning them all loose from a poor starting point walks the fit into a
local minimum that looks converged and is nonsense. :func:`auto_refine`
therefore follows the standard staged protocol, freeing parameters in the
order that has been the recommended practice since the method existed, and
it is the default.

**Weight fractions assume you found every phase.** The Hill–Howard
relation converts scale factors to weight fractions *of the crystalline,
identified part of the sample*. An unmodelled phase does not reduce the
total to below 100 %: its intensity is shared out among the phases you did
model. Nor does amorphous content appear at all. So a "78 % FeSe" from a
sample with an unexplained peak in the residual is not 78 % of the sample,
and :attr:`RietveldResult.warnings` says so whenever the residual suggests
it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .pattern import Pattern
from .powder import Profile, Reflection, pseudo_voigt, reflections, scherrer
from .structure import ATOMIC_WEIGHT, Crystal

#: Default Chebyshev background order. Six terms describe air scatter and
#: an amorphous hump without being able to follow a Bragg peak.
DEFAULT_BACKGROUND_ORDER = 6

#: Peak profiles are evaluated out to this many FWHM either side. Ten is
#: generous for the Gaussian part and necessary for the Lorentzian one,
#: whose tails carry a few per cent of the area well past the visible peak.
PROFILE_CUTOFF = 10.0

#: Reflections below this fraction of a phase's strongest are dropped
#: during refinement. Weaker than 1/1000 of the strongest line is under
#: the background of any laboratory pattern, and keeping them multiplies
#: the cost of every residual evaluation for nothing.
REFINE_MIN_RELATIVE = 1e-3

#: Crystallite size above which a line width carries no information, nm.
SIZE_CEILING = 120.0

#: A goodness of fit above this is a bad fit whatever the R factors say.
GOF_LIMIT = 4.0

#: March–Dollase r outside this range is severe texture.
TEXTURE_LIMITS = (0.7, 1.4)


class RefinementError(ValueError):
    """Raised when a refinement cannot be set up."""


@dataclass
class Parameter:
    """One refinable quantity, with its bounds and whether it is free."""

    name: str
    value: float
    free: bool = False
    lower: float = -np.inf
    upper: float = np.inf
    phase: Optional[int] = None
    kind: str = ""
    error: Optional[float] = None

    def __str__(self) -> str:
        mark = "libre " if self.free else "fijo  "
        error = f" ± {self.error:.3g}" if self.error is not None else ""
        return f"  {mark} {self.name:<28s} {self.value:14.6g}{error}"


@dataclass
class PhaseModel:
    """One phase in a refinement, with its own profile and texture."""

    crystal: Crystal
    scale: float = 1.0
    profile: Profile = field(default_factory=Profile)
    u_iso: float = 0.008
    preferred_axis: Optional[tuple[int, int, int]] = None
    preferred_r: float = 1.0
    lattice_factors: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def current_crystal(self) -> Crystal:
        """The structure at the current lattice factors."""
        if self.lattice_factors == (1.0, 1.0, 1.0):
            return self.crystal
        return self.crystal.with_lattice(
            self.crystal.lattice.scaled(self.lattice_factors)
        )

    @property
    def cell_mass(self) -> Optional[float]:
        """Mass of the unit cell contents in u, or ``None`` if unknown."""
        total = 0.0
        for element, number in self.current_crystal().cell_composition().items():
            weight = ATOMIC_WEIGHT.get(element)
            if weight is None:
                return None
            total += weight * number
        return total


@dataclass
class RietveldResult:
    """The outcome of a refinement."""

    pattern: Pattern
    phases: list[PhaseModel]
    calculated: np.ndarray
    background: np.ndarray
    parameters: list[Parameter]
    zero: float = 0.0
    displacement: float = 0.0
    r_p: float = 0.0
    r_wp: float = 0.0
    r_expected: float = 0.0
    gof: float = 0.0
    converged: bool = False
    message: str = ""
    stages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def difference(self) -> np.ndarray:
        """Observed minus calculated — the curve that is actually read."""
        return self.pattern.intensity - self.calculated

    @property
    def free_parameters(self) -> int:
        return sum(1 for p in self.parameters if p.free)

    def weight_fractions(self) -> dict[str, Optional[float]]:
        """Hill–Howard weight fractions of the *identified crystalline* part.

        ``W_p ∝ S_p · (ZM)_p · V_p``. Returns ``None`` for every phase if
        any structure contains an element with no tabulated atomic weight,
        rather than silently leaving it out of the normalisation.
        """
        products: list[float] = []
        for phase in self.phases:
            mass = phase.cell_mass
            if mass is None:
                return {p.crystal.name: None for p in self.phases}
            products.append(
                max(phase.scale, 0.0) * mass * phase.current_crystal().lattice.volume
            )
        total = sum(products)
        if total <= 0.0:
            return {p.crystal.name: None for p in self.phases}
        return {
            phase.crystal.name: value / total
            for phase, value in zip(self.phases, products)
        }

    def crystallite_sizes(
        self, instrument_fwhm: float = 0.06
    ) -> dict[str, Optional[float]]:
        """Volume-averaged crystallite size per phase, in nm.

        Evaluated at each phase's strongest reflection, with the
        instrument's own width removed first. ``None`` when the phase's
        peaks are no broader than the instrument — which is the honest
        answer, not a large number: above roughly 100 nm with a laboratory
        diffractometer, the line width carries no size information at all.
        """
        sizes: dict[str, Optional[float]] = {}
        for phase in self.phases:
            crystal = phase.current_crystal()
            lines = reflections(
                crystal,
                wavelength=self.pattern.wavelength,
                two_theta_range=self.pattern.range,
            )
            if not lines:
                sizes[crystal.name] = None
                continue
            strongest = max(lines, key=lambda r: r.intensity)
            observed = float(phase.profile.fwhm(np.array([strongest.two_theta]))[0])
            if observed <= instrument_fwhm:
                sizes[crystal.name] = None
                continue
            size = scherrer(
                observed - instrument_fwhm, strongest.two_theta, self.pattern.wavelength
            )
            sizes[crystal.name] = size if size <= SIZE_CEILING else None
        return sizes

    def summary(self, instrument_fwhm: float = 0.06) -> str:
        lines = [
            f"Refinamiento Rietveld — {self.pattern.name}",
            f"  {'convergido' if self.converged else 'NO convergido'}: {self.message}",
            f"  parámetros libres: {self.free_parameters} sobre "
            f"{self.pattern.n} puntos",
            "",
            f"  Rp   = {100 * self.r_p:6.2f} %",
            f"  Rwp  = {100 * self.r_wp:6.2f} %",
            f"  Rexp = {100 * self.r_expected:6.2f} %",
            f"  GOF  = {self.gof:6.3f}   (χ² = {self.gof ** 2:.3f})",
        ]
        if self.stages:
            lines.append("")
            lines.append("  Etapas: " + " → ".join(self.stages))
        lines.append("")
        lines.append(f"  Cero: {self.zero:+.4f}°   Desplazamiento de muestra: "
                     f"{self.displacement:+.4f}°")
        fractions = self.weight_fractions()
        sizes = self.crystallite_sizes(instrument_fwhm)
        lines.append("")
        lines.append("  Fases:")
        for phase in self.phases:
            crystal = phase.current_crystal()
            fraction = fractions.get(crystal.name)
            size = sizes.get(crystal.name)
            lines.append(f"    {crystal.name} [{crystal.formula}]")
            lines.append(
                "      fracción en peso : "
                + (f"{100 * fraction:5.1f} %" if fraction is not None else "no calculable")
            )
            lines.append(f"      celda            : {crystal.lattice.describe()}")
            lines.append(
                "      tamaño cristalito: "
                + (
                    f"{size:.1f} nm"
                    if size is not None
                    else (
                        f"no medible: por encima de ~{SIZE_CEILING:.0f} nm el "
                        "ensanchamiento por tamaño es menor que la resolución "
                        "del equipo y la anchura ya no lleva información"
                    )
                )
            )
            if abs(phase.preferred_r - 1.0) > 1e-6:
                lines.append(
                    f"      orientación pref.: r = {phase.preferred_r:.3f} "
                    f"sobre {phase.preferred_axis}"
                )
            lines.append(f"      U_iso            : {phase.u_iso:.5f} Å²")
        free = [p for p in self.parameters if p.free]
        if free:
            lines.append("")
            lines.append("  Parámetros refinados:")
            lines.extend(str(p) for p in free)
        if self.warnings:
            lines.append("")
            lines.extend("  ⚠ " + w for w in self.warnings)
        return "\n".join(lines)


# -- the model ---------------------------------------------------------


def chebyshev_background(
    two_theta: np.ndarray, coefficients: Sequence[float]
) -> np.ndarray:
    """Chebyshev polynomial background on the pattern's own 2θ range.

    Chebyshev rather than an ordinary polynomial because the terms stay
    bounded and roughly orthogonal over the interval, so adding a term
    changes the fit locally instead of rescaling every other coefficient —
    which is what makes a background order safe to increase.
    """
    angles = np.asarray(two_theta, dtype=float)
    if angles.size < 2:
        return np.zeros_like(angles)
    scaled = 2.0 * (angles - angles[0]) / (angles[-1] - angles[0]) - 1.0
    return np.polynomial.chebyshev.chebval(scaled, list(coefficients))


def _angle_shift(two_theta: float, zero: float, displacement: float) -> float:
    """The same correction as :func:`_angle_correction`, for one angle.

    Scalar because the refinement applies it to every reflection on every
    residual evaluation, and a one-element array round trip there is pure
    overhead.
    """
    return zero + displacement * math.cos(math.radians(two_theta) / 2.0)


def _angle_correction(two_theta: np.ndarray, zero: float, displacement: float) -> np.ndarray:
    """Instrumental 2θ corrections, as functions of angle.

    A zero error is constant. A sample sitting off the focusing circle
    produces ``Δ2θ ∝ cos θ``, largest at low angle and vanishing at 180°.
    They are different functions, which is exactly why refining only a
    zero against a mis-mounted sample leaves a systematic residual that no
    amount of further refinement removes.
    """
    return zero + displacement * np.cos(np.radians(np.asarray(two_theta)) / 2.0)


def calculate_pattern(
    two_theta: np.ndarray,
    phases: Sequence[PhaseModel],
    wavelength: float,
    kalpha2_ratio: float = 0.0,
    kalpha2_wavelength: Optional[float] = None,
    zero: float = 0.0,
    displacement: float = 0.0,
    background: Optional[Sequence[float]] = None,
    reflection_cache: Optional[dict] = None,
) -> np.ndarray:
    """The calculated diffractogram of a set of phases."""
    angles = np.asarray(two_theta, dtype=float)
    total = np.zeros_like(angles)
    lines: list[tuple[float, float]] = [(wavelength, 1.0)]
    if kalpha2_ratio > 0.0:
        second = kalpha2_wavelength or wavelength * 1.002486
        lines.append((second, kalpha2_ratio))

    span = (float(angles[0]) - 1.5, float(angles[-1]) + 1.5)
    for index, phase in enumerate(phases):
        crystal = phase.current_crystal()
        for line_wavelength, weight in lines:
            key = (
                index,
                round(line_wavelength, 8),
                phase.lattice_factors,
                round(phase.u_iso, 8),
                phase.preferred_axis,
                round(phase.preferred_r, 8),
            )
            computed: Optional[list[Reflection]] = (
                reflection_cache.get(key) if reflection_cache is not None else None
            )
            if computed is None:
                computed = reflections(
                    crystal,
                    wavelength=line_wavelength,
                    two_theta_range=span,
                    u_override=phase.u_iso,
                    preferred_axis=phase.preferred_axis,
                    preferred_r=phase.preferred_r,
                    normalise=False,
                    min_relative=REFINE_MIN_RELATIVE,
                )
                if reflection_cache is not None:
                    reflection_cache[key] = computed
            for reflection in computed:
                centre = reflection.two_theta + _angle_shift(
                    reflection.two_theta, zero, displacement)
                width = phase.profile.fwhm_at(centre)
                mixing = phase.profile.eta_at(centre)
                window = (
                    (angles > centre - PROFILE_CUTOFF * width)
                    & (angles < centre + PROFILE_CUTOFF * width)
                )
                if not window.any():
                    continue
                total[window] += (
                    phase.scale
                    * reflection.intensity
                    * weight
                    * pseudo_voigt(angles[window], centre, width, mixing)
                )
    if background is not None:
        total = total + chebyshev_background(angles, background)
    return total


# -- parameter handling ------------------------------------------------

#: Parameter groups, in the order the staged protocol frees them.
STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("escala y fondo", ("scale", "background")),
    ("cero", ("scale", "background", "zero")),
    ("celda", ("scale", "background", "zero", "lattice")),
    ("anchura", ("scale", "background", "zero", "lattice", "profile_w")),
    ("perfil", ("scale", "background", "zero", "lattice", "profile_w", "profile_uv", "eta")),
    ("textura", ("scale", "background", "zero", "lattice", "profile_w", "profile_uv",
                 "eta", "preferred")),
    ("desplazamiento y U", ("scale", "background", "zero", "displacement", "lattice",
                            "profile_w", "profile_uv", "eta", "preferred", "u_iso")),
)


def build_parameters(
    phases: Sequence[PhaseModel],
    background_order: int = DEFAULT_BACKGROUND_ORDER,
    background: Optional[Sequence[float]] = None,
) -> list[Parameter]:
    """The full parameter list, all fixed. Free them by ``kind`` or name."""
    parameters: list[Parameter] = []
    coefficients = list(background) if background is not None else [0.0] * background_order
    for order, value in enumerate(coefficients):
        parameters.append(
            Parameter(f"fondo_c{order}", value, kind="background")
        )
    parameters.append(Parameter("cero", 0.0, lower=-1.0, upper=1.0, kind="zero"))
    parameters.append(
        Parameter("desplazamiento", 0.0, lower=-1.0, upper=1.0, kind="displacement")
    )
    for index, phase in enumerate(phases):
        tag = phase.crystal.name
        parameters.append(
            Parameter(f"escala[{tag}]", phase.scale, lower=0.0, phase=index, kind="scale")
        )
        # Only the symmetry-independent axes get a parameter. A hexagonal
        # cell has one, not two: refining a and b separately returns two
        # numbers differing by several standard errors that symmetry says
        # are the same number.
        constraint = phase.crystal.lattice_constraint()
        for axis, factor, label in zip("abc", phase.lattice_factors, constraint):
            if label != axis:
                continue
            parameters.append(
                Parameter(
                    f"{axis}[{tag}]", factor, lower=0.95, upper=1.05,
                    phase=index, kind="lattice",
                )
            )
        parameters.append(
            Parameter(f"W[{tag}]", phase.profile.w, lower=1e-5, upper=2.0,
                      phase=index, kind="profile_w")
        )
        parameters.append(
            Parameter(f"U[{tag}]", phase.profile.u, lower=-0.5, upper=2.0,
                      phase=index, kind="profile_uv")
        )
        parameters.append(
            Parameter(f"V[{tag}]", phase.profile.v, lower=-1.0, upper=1.0,
                      phase=index, kind="profile_uv")
        )
        parameters.append(
            Parameter(f"eta[{tag}]", phase.profile.eta0, lower=0.0, upper=1.0,
                      phase=index, kind="eta")
        )
        parameters.append(
            Parameter(f"r_textura[{tag}]", phase.preferred_r, lower=0.3, upper=3.0,
                      phase=index, kind="preferred")
        )
        parameters.append(
            Parameter(f"U_iso[{tag}]", phase.u_iso, lower=0.0, upper=0.15,
                      phase=index, kind="u_iso")
        )
    return parameters


def free_kinds(parameters: Sequence[Parameter], kinds: Sequence[str]) -> None:
    """Free every parameter of the named kinds and fix the rest."""
    wanted = set(kinds)
    for parameter in parameters:
        parameter.free = parameter.kind in wanted


def _apply(parameters: Sequence[Parameter], phases: Sequence[PhaseModel]) -> tuple[
    list[float], float, float
]:
    """Push parameter values into the phase models. Returns the background."""
    background: list[float] = []
    zero = displacement = 0.0
    lattice: dict[int, list[float]] = {i: list(p.lattice_factors) for i, p in enumerate(phases)}
    for parameter in parameters:
        if parameter.kind == "background":
            background.append(parameter.value)
        elif parameter.kind == "zero":
            zero = parameter.value
        elif parameter.kind == "displacement":
            displacement = parameter.value
        elif parameter.phase is None:
            continue
        else:
            phase = phases[parameter.phase]
            if parameter.kind == "scale":
                phase.scale = parameter.value
            elif parameter.kind == "lattice":
                axis = parameter.name[0]
                constraint = phase.crystal.lattice_constraint()
                for position, label in enumerate(constraint):
                    if label == axis:
                        lattice[parameter.phase][position] = parameter.value
            elif parameter.name.startswith("W["):
                phase.profile.w = parameter.value
            elif parameter.name.startswith("U["):
                phase.profile.u = parameter.value
            elif parameter.name.startswith("V["):
                phase.profile.v = parameter.value
            elif parameter.kind == "eta":
                phase.profile.eta0 = parameter.value
            elif parameter.kind == "preferred":
                phase.preferred_r = parameter.value
            elif parameter.kind == "u_iso":
                phase.u_iso = parameter.value
    for index, factors in lattice.items():
        phases[index].lattice_factors = tuple(factors)
    return background, zero, displacement


def r_factors(
    observed: np.ndarray,
    calculated: np.ndarray,
    weights: np.ndarray,
    free: int,
) -> tuple[float, float, float, float]:
    """``(Rp, Rwp, Rexp, GOF)``.

    ``Rexp`` is the R factor the counting statistics alone would produce,
    so ``GOF = Rwp/Rexp`` is the honest measure: an Rwp of 8 % is
    excellent on a noisy pattern and poor on a long overnight count, and
    only the ratio distinguishes them.
    """
    residual = observed - calculated
    denominator = float(np.abs(observed).sum())
    r_p = float(np.abs(residual).sum() / denominator) if denominator else float("inf")
    weighted = float((weights * observed**2).sum())
    if weighted <= 0.0:
        return r_p, float("inf"), float("inf"), float("inf")
    r_wp = math.sqrt(float((weights * residual**2).sum()) / weighted)
    points = observed.size
    r_expected = math.sqrt(max(points - free, 1) / weighted)
    gof = r_wp / r_expected if r_expected > 0 else float("inf")
    return r_p, r_wp, r_expected, gof


# -- the refinement ----------------------------------------------------


def refine(
    pattern: Pattern,
    phases: Sequence[PhaseModel] | Sequence[Crystal],
    parameters: Optional[list[Parameter]] = None,
    background_order: int = DEFAULT_BACKGROUND_ORDER,
    max_iterations: int = 200,
    instrument_fwhm: float = 0.06,
    callback: Optional[Callable[[str], None]] = None,
) -> RietveldResult:
    """One refinement of whatever parameters are marked free.

    This is the manual entry point: build the parameter list with
    :func:`build_parameters`, set ``free`` on the ones you want, and call
    this. :func:`auto_refine` is the staged protocol built on top of it.

    Parameters
    ----------
    pattern:
        The measured diffractogram. Do **not** background-subtract or
        smooth it first: the background is part of the model, and
        smoothing correlates neighbouring points, after which χ² and every
        uncertainty are meaningless.
    phases:
        :class:`PhaseModel` objects, or bare structures which are wrapped
        with default settings.
    parameters:
        The parameter list. Defaults to everything fixed except the scales
        and the background, which is the only combination that is always
        safe to start from.

    Returns
    -------
    RietveldResult
    """
    models = [
        p if isinstance(p, PhaseModel) else PhaseModel(crystal=p) for p in phases
    ]
    if not models:
        raise RefinementError("no hay ninguna fase que refinar")
    if parameters is None:
        parameters = build_parameters(models, background_order)
        free_kinds(parameters, ("scale", "background"))

    observed = np.asarray(pattern.intensity, dtype=float)
    sigma = np.asarray(pattern.sigma, dtype=float)
    weights = 1.0 / np.maximum(sigma, 1e-9) ** 2
    cache: dict = {}

    free = [p for p in parameters if p.free]
    if not free:
        raise RefinementError(
            "no hay ningún parámetro libre. Marca al menos la escala"
        )

    def unpack(values: np.ndarray) -> None:
        for parameter, value in zip(free, values):
            parameter.value = float(value)

    def model() -> np.ndarray:
        background, zero, displacement = _apply(parameters, models)
        return calculate_pattern(
            pattern.two_theta,
            models,
            wavelength=pattern.wavelength,
            kalpha2_ratio=pattern.kalpha2_ratio,
            kalpha2_wavelength=pattern.kalpha2_lambda if pattern.has_doublet else None,
            zero=zero,
            displacement=displacement,
            background=background,
            reflection_cache=cache,
        )

    def residual(values: np.ndarray) -> np.ndarray:
        unpack(values)
        return (observed - model()) * np.sqrt(weights)

    start = np.array([p.value for p in free], dtype=float)
    lower = np.array([p.lower for p in free], dtype=float)
    upper = np.array([p.upper for p in free], dtype=float)
    start = np.clip(start, lower + 1e-12, upper - 1e-12)

    outcome = least_squares(
        residual,
        start,
        bounds=(lower, upper),
        max_nfev=max_iterations * max(len(free), 1),
        x_scale="jac",
        xtol=1e-9,
        ftol=1e-9,
        gtol=1e-9,
    )
    unpack(outcome.x)
    _estimate_errors(outcome, free, observed.size)

    background, zero, displacement = _apply(parameters, models)
    calculated = model()
    r_p, r_wp, r_expected, gof = r_factors(observed, calculated, weights, len(free))

    result = RietveldResult(
        pattern=pattern,
        phases=models,
        calculated=calculated,
        background=chebyshev_background(pattern.two_theta, background),
        parameters=parameters,
        zero=zero,
        displacement=displacement,
        r_p=r_p,
        r_wp=r_wp,
        r_expected=r_expected,
        gof=gof,
        converged=bool(outcome.success),
        message=str(outcome.message),
    )
    _check(result, instrument_fwhm)
    if callback:
        callback(f"Rwp = {100 * r_wp:.2f} %, GOF = {gof:.3f}")
    return result


def _estimate_errors(outcome, free: Sequence[Parameter], points: int) -> None:
    """Standard errors from the Jacobian, scaled by the reduced χ².

    These are the *statistical* uncertainties of the fit and they are
    famously optimistic in Rietveld refinement, by factors of two to
    three, because the residuals of a diffraction pattern are serially
    correlated and the derivation assumes they are not. Quote them, but do
    not quote them as the accuracy of a lattice parameter.
    """
    try:
        jacobian = outcome.jac
        _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
        threshold = np.finfo(float).eps * max(jacobian.shape) * singular[0]
        keep = singular > threshold
        covariance = (vt[keep].T / singular[keep] ** 2) @ vt[keep]
        dof = max(points - len(free), 1)
        scale = 2.0 * outcome.cost / dof
        errors = np.sqrt(np.maximum(np.diag(covariance) * scale, 0.0))
    except (np.linalg.LinAlgError, ValueError, IndexError):  # pragma: no cover
        return
    for parameter, error in zip(free, errors):
        parameter.error = float(error)


def auto_refine(
    pattern: Pattern,
    phases: Sequence[PhaseModel] | Sequence[Crystal],
    background_order: int = DEFAULT_BACKGROUND_ORDER,
    stages: Sequence[tuple[str, tuple[str, ...]]] = STAGES,
    preferred_axis: Optional[Sequence[int]] = None,
    instrument_fwhm: float = 0.06,
    callback: Optional[Callable[[str], None]] = None,
) -> RietveldResult:
    """The staged protocol: free parameters in a safe order.

    Scale and background first, then the zero, the cell, the widths, the
    shape, the texture, and only at the end the displacement parameters —
    which correlate with the background and with U_iso and will absorb
    both if let loose early. Each stage starts from the previous stage's
    answer, so the fit walks towards the minimum instead of jumping at it.

    This is the standard recommended order and the reason it exists is
    worth stating plainly: freeing everything at once on a real pattern
    converges, reports plausible R factors, and returns a wrong structure.
    """
    models = [
        p if isinstance(p, PhaseModel) else PhaseModel(crystal=p) for p in phases
    ]
    if preferred_axis is not None:
        for phase in models:
            if phase.preferred_axis is None:
                phase.preferred_axis = tuple(int(v) for v in preferred_axis)

    parameters = build_parameters(models, background_order)
    # A sensible background start: the low percentile of the data, so the
    # first stage does not have to climb from zero.
    baseline = float(np.percentile(pattern.intensity, 5))
    for parameter in parameters:
        if parameter.name == "fondo_c0":
            parameter.value = baseline
    for phase in models:
        phase.scale = max(phase.scale, 1e-8)

    result: Optional[RietveldResult] = None
    done: list[str] = []
    for label, kinds in stages:
        active = list(kinds)
        if any(p.preferred_axis is None for p in models) and "preferred" in active:
            active = [k for k in active if k != "preferred"]
        free_kinds(parameters, active)
        try:
            result = refine(
                pattern,
                models,
                parameters=parameters,
                background_order=background_order,
                instrument_fwhm=instrument_fwhm,
            )
        except (RefinementError, ValueError) as exc:  # pragma: no cover - defensive
            if result is None:
                raise
            result.warnings.append(f"la etapa «{label}» falló: {exc}")
            break
        done.append(label)
        if callback:
            callback(f"[{label}] Rwp = {100 * result.r_wp:.2f} %, GOF = {result.gof:.3f}")
    assert result is not None
    result.stages = done
    return result


def _check(result: RietveldResult, instrument_fwhm: float) -> None:
    """Say what the numbers mean, including when they mean nothing."""
    if result.gof > GOF_LIMIT:
        result.warnings.append(
            f"la bondad del ajuste es {result.gof:.2f}, muy por encima de 1. "
            "El modelo no describe los datos: falta una fase, el fondo se "
            "queda corto, o el perfil no vale. Mira la curva diferencia antes "
            "que cualquier número: los R son un resumen y la diferencia dice "
            "DÓNDE falla"
        )
    elif result.gof < 0.9 and result.pattern.sigma_origin == "poisson":
        result.warnings.append(
            f"la bondad del ajuste es {result.gof:.2f}, por debajo de 1, lo "
            "que significa ajustar mejor que el ruido. Casi siempre es que "
            "las incertidumbres están sobrestimadas: el patrón no son cuentas "
            "crudas (¿lo has suavizado o restado fondo antes?) y entonces √N "
            "no es su error"
        )
    if result.pattern.sigma_origin == "flat":
        result.warnings.append(
            "el patrón no está en cuentas, así que las incertidumbres son una "
            "estimación plana y ni Rwp ni GOF tienen su significado "
            "estadístico habitual. Las posiciones y las fracciones siguen "
            "valiendo; los R, como comparación entre ajustes de ESTE patrón y "
            "no como medida absoluta"
        )
    for phase in result.phases:
        name = phase.crystal.name
        if phase.u_iso <= 1e-6:
            result.warnings.append(
                f"{name}: U_iso ha caído a cero. Suele indicar que algo más "
                "está absorbiendo intensidad —el fondo, la absorción, o una "
                "fase que falta— y no que los átomos no vibren"
            )
        elif phase.u_iso > 0.05:
            result.warnings.append(
                f"{name}: U_iso = {phase.u_iso:.3f} Å² es enorme (lo normal es "
                "0.005–0.02). Es el parámetro que se traga los errores de los "
                "demás: comprueba el fondo y la orientación preferente antes "
                "de creerte esta cifra"
            )
        if phase.preferred_axis is not None and not (
            TEXTURE_LIMITS[0] <= phase.preferred_r <= TEXTURE_LIMITS[1]
        ):
            result.warnings.append(
                f"{name}: la orientación preferente refinada (r = "
                f"{phase.preferred_r:.2f}) es severa. Con textura así de "
                "fuerte, la fracción en peso de esta fase es poco fiable; "
                "vuelve a preparar la muestra con carga lateral o mide en "
                "capilar"
            )
        factors = phase.lattice_factors
        if max(abs(f - 1.0) for f in factors) > 0.02:
            result.warnings.append(
                f"{name}: la celda se ha movido más de un 2 % respecto a la "
                "referencia. O es otra fase, o la longitud de onda que has "
                "declarado no es la del equipo. Comprueba λ antes que la "
                "estructura: un λ equivocado escala TODA la celda por el "
                "mismo factor y nada en el ajuste protesta"
            )
    fractions = result.weight_fractions()
    if any(value is None for value in fractions.values()):
        result.warnings.append(
            "no se pueden calcular fracciones en peso: alguna fase contiene "
            "un elemento sin masa atómica tabulada"
        )
    else:
        result.warnings.append(
            "las fracciones en peso son de la parte CRISTALINA E IDENTIFICADA "
            "de la muestra. Una fase que no hayas modelado no baja el total de "
            "100 %: su intensidad se reparte entre las que sí están. Y el "
            "material amorfo no aparece en absoluto. Para cuantificar amorfo "
            "hace falta un patrón interno"
        )


__all__ = [
    "DEFAULT_BACKGROUND_ORDER",
    "GOF_LIMIT",
    "PROFILE_CUTOFF",
    "REFINE_MIN_RELATIVE",
    "SIZE_CEILING",
    "STAGES",
    "TEXTURE_LIMITS",
    "Parameter",
    "PhaseModel",
    "RefinementError",
    "RietveldResult",
    "auto_refine",
    "build_parameters",
    "calculate_pattern",
    "chebyshev_background",
    "free_kinds",
    "r_factors",
    "refine",
]
