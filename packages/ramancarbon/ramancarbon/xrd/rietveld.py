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
from pathlib import Path
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


class RefinementCancelled(RuntimeError):
    """Raised inside the residual to unwind out of ``least_squares``.

    SciPy has no stop hook, so the only way out of a running fit is to
    raise from the function it is calling. This is caught immediately by
    :func:`refine`, which then reports whatever the solver had reached --
    losing the work would make stopping useless, and a user who cannot
    stop a refinement will kill the window instead.
    """

#: Report progress every this many residual evaluations. Small enough that
#: a slow refinement visibly moves, large enough that the reporting itself
#: is not the cost.
PROGRESS_EVERY = 10

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
    cancelled: bool = False
    """The user stopped it. The parameters are wherever the solver had
    got to, which is a real intermediate state and not a failure -- but
    the uncertainties are not meaningful, because they come from a
    Jacobian at a minimum the fit had not reached."""
    message: str = ""
    n_evaluations: int = 0
    """Residual evaluations the least-squares solver used. A refinement
    that looks instantaneous usually IS: with a good starting point and
    few free parameters it converges in a handful of steps. Reporting the
    number is what distinguishes that from one that never started."""
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    """Parameter-to-parameter correlations from the fit's own covariance.
    Empty when the refinement was cancelled, since there is no converged
    Jacobian to take them from."""
    stages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def difference(self) -> np.ndarray:
        """Observed minus calculated — the curve that is actually read."""
        return self.pattern.intensity - self.calculated

    @property
    def chi_squared(self) -> float:
        """Reduced χ², which is the goodness of fit squared.

        The two are the same statement — ``GOF = Rwp/Rexp`` and
        ``χ²_red = GOF²`` — and both are quoted because different
        communities read one or the other. What neither means, on its own,
        is that the structure is right: a χ² of 1.05 with a systematic
        ripple in the difference curve is a worse refinement than a χ² of
        3 with structureless residuals, and the difference curve is where
        to look first.
        """
        return float(self.gof ** 2)

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

    @property
    def background_share(self) -> float:
        """How much of the calculated pattern is background, 0 to 1.

        The single most useful number for reading a weight fraction, and
        the one nobody computes. Every fraction this class returns is a
        ratio between scale factors, and a scale factor is only fixed by
        the part of the pattern the phases actually account for. When that
        part is small, the fractions are conclusions drawn from a sliver.

        On the user's real CVD pattern — iron, cementite, two iron
        selenides and a 6 nm turbostratic carbon over 10–90° — this came
        out at **0.968**: the whole five-phase model explains 3.2 % of the
        counts, and the carbon's 52 % by weight is a share of that 3.2 %.
        """
        total = float(np.sum(np.abs(self.calculated)))
        if total <= 0.0:
            return 0.0
        return float(np.sum(np.abs(self.background)) / total)

    def phase_contributions(self) -> dict[str, float]:
        """Each phase's share of the calculated intensity above background.

        Not the same question as the weight fraction and much closer to
        what the eye asks of the plot: the weight fraction answers "how
        much of the sample", weighted by scattering power per unit mass,
        while this answers "how much of what you can see". A phase can be
        half the sample by weight and a twentieth of the visible signal —
        carbon against iron is exactly that case, because the scattering
        goes as Z² and carbon's Z is 6 against iron's 26 — and a reader
        who is told only the first number has no way to judge it.
        """
        shares: dict[str, float] = {}
        for phase in self.phases:
            curve = calculate_pattern(
                self.pattern.two_theta,
                [phase],
                wavelength=self.pattern.wavelength,
                kalpha2_ratio=self.pattern.kalpha2_ratio,
                kalpha2_wavelength=(
                    self.pattern.kalpha2_lambda
                    if self.pattern.has_doublet else None
                ),
                zero=self.zero,
                displacement=self.displacement,
            )
            shares[phase.crystal.name] = float(np.sum(np.abs(curve)))
        total = sum(shares.values())
        if total <= 0.0:
            return {name: 0.0 for name in shares}
        return {name: value / total for name, value in shares.items()}

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
            f"  GOF  = {self.gof:6.3f}   (χ² reducida = {self.chi_squared:.3f})",
            f"  {self.n_evaluations} evaluaciones del residuo",
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
    should_stop: Optional[Callable[[], bool]] = None,
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

    evaluations = 0
    root_weights = np.sqrt(weights)
    # The ceiling SciPy is given. Quoting the counter against it turns
    # "iteración 2840" -- which says nothing about whether to keep
    # waiting -- into a fraction of a known budget. With five phases and
    # forty free parameters that budget is eight thousand evaluations,
    # and a user who cannot see that will read a working fit as a hang.
    budget = max_iterations * max(len(free), 1)
    best = {"values": None, "rwp": float("inf")}

    def residual(values: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        unpack(values)
        difference = (observed - model()) * root_weights
        evaluations += 1
        current = math.sqrt(
            float(np.sum(difference ** 2))
            / max(float(np.sum(weights * observed ** 2)), 1e-30)
        )
        if current < best["rwp"]:
            best["rwp"] = current
            best["values"] = np.array(values, dtype=float)
        if callback and evaluations % PROGRESS_EVERY == 0:
            # Rwp from the residual already in hand, so reporting progress
            # costs no extra pattern calculation -- which would otherwise
            # be the single most expensive thing in the loop.
            callback(f"iteración {evaluations} de {budget}: "
                     f"Rwp = {100 * current:.2f} %")
        if should_stop is not None and should_stop():
            raise RefinementCancelled(
                f"detenido en la iteración {evaluations}")
        return difference

    start = np.array([p.value for p in free], dtype=float)
    lower = np.array([p.lower for p in free], dtype=float)
    upper = np.array([p.upper for p in free], dtype=float)
    start = np.clip(start, lower + 1e-12, upper - 1e-12)

    cancelled = False
    stop_message = ""
    try:
        outcome = least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=budget,
            x_scale="jac",
            xtol=1e-9,
            ftol=1e-9,
            gtol=1e-9,
        )
    except RefinementCancelled as stopped:
        # Keep the best point the solver actually reached rather than
        # the half-step it was standing on when the stop arrived: a
        # least-squares trial can be far worse than the last accepted
        # one, and handing that back would make stopping destructive.
        cancelled = True
        stop_message = str(stopped)
        unpack(best["values"] if best["values"] is not None else start)
        correlations = {}
        for parameter in free:
            parameter.error = None
    else:
        unpack(outcome.x)
        correlations = _estimate_errors(outcome, free, observed.size)

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
        converged=False if cancelled else bool(outcome.success),
        cancelled=cancelled,
        message=stop_message if cancelled else str(outcome.message),
        n_evaluations=evaluations,
        correlations=correlations,
    )
    if cancelled:
        result.warnings.append(
            "Refinamiento detenido por el usuario: los parámetros son los "
            "del mejor punto alcanzado, y las incertidumbres NO se "
            "calcularon porque salen del jacobiano en un mínimo al que el "
            "ajuste no llegó.")
    else:
        _check(result, instrument_fwhm)
    if callback:
        callback(
            f"{evaluations} iteraciones: Rwp = {100 * r_wp:.2f} %, "
            f"GOF = {gof:.3f}, χ² = {gof ** 2:.3f}"
            + (" (detenido)" if cancelled else "")
        )
    return result


def _estimate_errors(
    outcome, free: Sequence[Parameter], points: int
) -> dict[tuple[str, str], float]:
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
        variance = np.diag(covariance)
        errors = np.sqrt(np.maximum(variance * scale, 0.0))
    except (np.linalg.LinAlgError, ValueError, IndexError):  # pragma: no cover
        return {}
    for parameter, error in zip(free, errors):
        parameter.error = float(error)
    # The off-diagonal was being thrown away, and it holds the answer to
    # the question a weight fraction cannot answer for itself: whether
    # the scale factor that produced it was determined by the data or by
    # the background polynomial. Computing it costs nothing here -- the
    # covariance is already in hand -- and reconstructing it afterwards
    # would mean refitting.
    spread = np.sqrt(np.maximum(variance, 0.0))
    correlations: dict[tuple[str, str], float] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, first in enumerate(free):
            for j in range(i + 1, len(free)):
                denominator = spread[i] * spread[j]
                if denominator <= 0.0 or not np.isfinite(denominator):
                    continue
                value = float(covariance[i, j] / denominator)
                if np.isfinite(value):
                    correlations[(first.name, free[j].name)] = value
    return correlations


def auto_refine(
    pattern: Pattern,
    phases: Sequence[PhaseModel] | Sequence[Crystal],
    background_order: int = DEFAULT_BACKGROUND_ORDER,
    stages: Sequence[tuple[str, tuple[str, ...]]] = STAGES,
    preferred_axis: Optional[Sequence[int]] = None,
    instrument_fwhm: float = 0.06,
    callback: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
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
    skipped_texture = False
    for label, kinds in stages:
        active = list(kinds)
        if any(p.preferred_axis is None for p in models) and "preferred" in active:
            active = [k for k in active if k != "preferred"]
            skipped_texture = True
        free_kinds(parameters, active)
        # The stage's own name goes in front of the inner counter, so a
        # long refinement says which stage it is in and how far into it,
        # rather than going quiet for twenty seconds.
        stage_number = len(done) + 1
        inner = (
            lambda text, stage=label, n=stage_number:
            callback(f"[{n}/{len(stages)} {stage}] {text}")
        ) if callback else None
        try:
            result = refine(
                pattern,
                models,
                parameters=parameters,
                background_order=background_order,
                instrument_fwhm=instrument_fwhm,
                callback=inner,
                should_stop=should_stop,
            )
        except (RefinementError, ValueError) as exc:  # pragma: no cover - defensive
            if result is None:
                raise
            result.warnings.append(f"la etapa «{label}» falló: {exc}")
            break
        done.append(label)
        if callback:
            callback(f"[{label}] Rwp = {100 * result.r_wp:.2f} %, GOF = {result.gof:.3f}")
        if result.cancelled:
            # The stop was asked for inside this stage. Running the next
            # one would free more parameters from a point the fit had not
            # settled at, which is the one thing the staged order exists
            # to prevent.
            result.warnings.append(
                f"se detuvo en la etapa «{label}»; quedaron sin correr: "
                + ", ".join(name for name, _ in stages[len(done):]) + ".")
            break
    assert result is not None
    result.stages = done
    if skipped_texture and not result.cancelled:
        # It was being skipped in silence, and silence reads as "there was
        # nothing to do". There was: the texture stage is the only one
        # that touches relative intensities, and a laminar material on a
        # flat plate is where preferred orientation actually bites -- a
        # turbostratic carbon lying flat shows its 00l and hides its hk0.
        # A user looking at a difference curve whose peak HEIGHTS do not
        # match has no way to learn from the program that the one stage
        # for that never ran.
        result.warnings.append(
            "la etapa de orientación preferente no se ha corrido, porque "
            "ninguna fase declara un eje de textura. Es la única etapa que "
            "toca las intensidades RELATIVAS: si las alturas de los picos "
            "calculados no casan con las medidas mientras las posiciones sí, "
            "empieza por ahí. En un material laminar —un carbono "
            "turbostrático montado en plano es el caso de libro— la "
            "orientación preferente es grande y no es opcional"
        )
    return result


#: Columns a refinement is plotted from, in the order every Rietveld
#: figure in the literature draws them.
FIT_COLUMNS = ("2theta", "observado", "calculado", "fondo", "diferencia",
               "sigma")


def fit_columns(result: RietveldResult) -> tuple[list[str], np.ndarray]:
    """The arrays a Rietveld figure is drawn from.

    A saved picture and a saved report answer "what happened"; neither
    lets the work be redrawn beside someone else's data, put on a
    journal's axes, or replotted at a different scale. These are the six
    columns that do, and they are exactly what the window's own plot
    uses -- not a recalculation that could drift from it.

    ``sigma`` comes along because the weighting is part of the result:
    the difference curve is usually shown divided by it, and without the
    column that plot cannot be made.
    """
    pattern = result.pattern
    return list(FIT_COLUMNS), np.column_stack([
        np.asarray(pattern.two_theta, dtype=float),
        np.asarray(pattern.intensity, dtype=float),
        np.asarray(result.calculated, dtype=float),
        np.asarray(result.background, dtype=float),
        np.asarray(result.difference, dtype=float),
        np.asarray(pattern.sigma, dtype=float),
    ])


def reflection_ticks(result: RietveldResult) -> dict[str, np.ndarray]:
    """Allowed reflection positions per phase, in 2theta.

    The zero shift is already applied, so these are where the ticks sit
    on the measured axis rather than where an ideal cell would put them.
    Without them the figure cannot be redrawn: the tick rows are how a
    reader tells which phase owns which peak.
    """
    pattern = result.pattern
    ticks: dict[str, np.ndarray] = {}
    for phase in result.phases:
        crystal = phase.current_crystal()
        lines = reflections(crystal, wavelength=pattern.wavelength,
                            two_theta_range=pattern.range)
        ticks[crystal.name] = np.array(
            [line.two_theta + result.zero for line in lines], dtype=float)
    return ticks


def write_fit(result: RietveldResult, path) -> list[Path]:
    """Write the fit as text, and the tick marks beside it.

    Two files rather than one, because they have different shapes: the
    curves are one row per measured point and the ticks are one row per
    allowed reflection. Forcing both into a single table would mean
    padding one of them, and a column of blanks is a thing every plotting
    program reads differently.

    Returns the paths written, the curves first.
    """
    destination = Path(path)
    names, table = fit_columns(result)
    header = [
        f"# {result.pattern.name}",
        f"# lambda = {result.pattern.wavelength:.6f} A",
        (f"# Rwp = {100 * result.r_wp:.3f} %  Rp = {100 * result.r_p:.3f} %  "
         f"Rexp = {100 * result.r_expected:.3f} %  GOF = {result.gof:.4f}"),
        (f"# desplazamiento cero = {result.zero:+.5f} deg, "
         f"muestra = {result.displacement:+.5f}"),
        f"# fases: {', '.join(p.current_crystal().name for p in result.phases)}",
    ]
    if result.cancelled:
        header.append("# ATENCION: refinamiento DETENIDO antes de converger")
    header.append("# " + "  ".join(f"{n:>14s}" for n in names).lstrip())
    rows = ["  ".join(f"{value:14.6f}" for value in row) for row in table]
    destination.write_text("\n".join(header + rows) + "\n", encoding="utf-8")

    ticks = reflection_ticks(result)
    companion = destination.with_name(destination.stem + "_reflexiones"
                                      + (destination.suffix or ".txt"))
    lines = [
        "# posiciones de reflexion permitidas, 2theta en grados",
        "# con el desplazamiento cero ya aplicado",
        "# fase  2theta",
    ]
    for name, angles in ticks.items():
        for angle in angles:
            lines.append(f"{name}  {angle:.5f}")
    companion.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [destination, companion]


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
        _warn_about_the_background(result)


#: Above this share of the calculated pattern, the background is most of
#: what was fitted and every weight fraction comes from the remainder.
#:
#: Measured on a real CVD pattern of carbon on FeSe: 0.968, with a
#: five-phase model and a 6 nm turbostratic carbon. A "51 % carbon" read
#: off that fit is 51 % of the 3.2 % that the phases explain.
BACKGROUND_DOMINATES = 0.90

#: Correlation above which a phase's scale factor and the background are
#: not separable, so the phase's weight fraction is a choice about the
#: background and not a measurement.
#:
#: The same number the Raman side uses for the same reason
#: (`models.acceptance.DEGENERATE_CORRELATION`): above it, two parameters
#: are not independently determined and their values must not be quoted
#: apart.
SCALE_BACKGROUND_DEGENERATE = 0.95

#: How many times its share of the diffracted intensity a phase's weight
#: fraction has to be before the gap is worth stating.
#:
#: Measured, not guessed: on the pattern that prompted this the
#: turbostratic carbon came back at 52.4 % by weight and 17.9 % of the
#: calculated intensity, a ratio of 2.9, and a first attempt with the
#: threshold at 3 missed it by four hundredths -- which is the wrong way
#: for a check whose entire purpose is to explain a number the user
#: already found surprising. Two and a half sits below the case that
#: matters and above the ordinary spread between phases of similar
#: scattering power (alpha-Fe here was 20.8 % by weight and 29.3 % of the
#: intensity, a ratio of 0.7).
WEIGHT_OVER_SIGNAL = 2.5


def _warn_about_the_background(result: RietveldResult) -> None:
    """Say when a weight fraction is really a statement about the background.

    A broad reflection from a nanocrystalline phase and a flexible
    polynomial background describe the same shape, and nothing in the
    refinement objects: the fit converges, Rwp looks good, and the phase
    comes back with a weight fraction that reads like a measurement.

    This is not hypothetical and it is not a small effect. On the user's
    own pattern — α-Fe, a 6 nm turbostratic carbon, cementite and two
    iron selenides, over 10–90° — changing ONLY the order of the
    background polynomial, which is a modelling choice and not a
    measurement, moved the answer like this:

    ======  ======  ======  ========
    orden   Rwp %   C wt%   Fe3C wt%
    ======  ======  ======  ========
    2       5.34    25.9    52.7
    4       4.90    36.6    32.3
    6       4.39    52.4    16.3
    ======  ======  ======  ========

    and deleting the carbon phase entirely — half the sample, by that
    fit's own account — cost 0.29 percentage points of Rwp.

    Three things are checked, because they fail independently: how much
    of the pattern the background accounts for, which parameters the fit
    could not separate, and whether a phase that is large by weight is
    small in the visible signal. Nothing is changed; every one is
    reported.

    One limitation, stated because assuming otherwise was this function's
    first design: **the covariance cannot see the effect above.** On the
    pattern quoted, the carbon's scale factor correlated with the
    background coefficients at no more than 0.33 — locally, given a
    sixth-order polynomial, the scale is perfectly well determined. What
    is not determined is the ORDER, and an order is a discrete choice
    about the model, not a parameter with a derivative. So the correlation
    check here is worth having for the degeneracies it does catch (on that
    same refinement, zero against sample displacement came back at
    -1.0000) and `background_order_sensitivity` is what catches this one.
    """
    share = result.background_share
    if share >= BACKGROUND_DOMINATES:
        result.warnings.append(
            f"el fondo es el {100 * share:.0f} % del patrón calculado: las "
            f"fases modeladas explican el {100 * (1 - share):.1f} % restante y "
            "TODAS las fracciones en peso salen de ese resto. Un fondo "
            "flexible y una reflexión ancha describen la misma forma, así que "
            "comprueba la cifra cambiando el orden del polinomio de fondo: si "
            "se mueve, es una elección tuya y no una medida"
        )

    degenerate = sorted(
        (
            (abs(value), first, second, value)
            for (first, second), value in result.correlations.items()
            if abs(value) >= SCALE_BACKGROUND_DEGENERATE
        ),
        reverse=True,
    )
    if degenerate:
        seen: set[str] = set()
        shown: list[str] = []
        for _, first, second, value in degenerate:
            # One line per parameter, not per pair: zero, displacement and
            # a lattice constant mutually at 0.9999 is ONE degeneracy in
            # three costumes, and printing all three pairs buries it.
            if first in seen and second in seen:
                continue
            seen.update((first, second))
            shown.append(f"{first}–{second} ({value:+.3f})")
            if len(shown) >= 4:
                break
        result.warnings.append(
            "parámetros que los datos NO separan: " + ", ".join(shown)
            + (". Cada uno de esos pares es un solo grado de libertad "
               "repartido entre dos nombres: sus valores individuales no se "
               "pueden citar, y sus incertidumbres son las de un ajuste que "
               "cree tener más información de la que tiene")
        )
    scale_tied = {
        (first if "escala" in first else second)
        for _, first, second, _v in degenerate
        if (("escala" in first and "fondo" in second)
            or ("fondo" in first and "escala" in second))
    }
    for scale_name in sorted(scale_tied):
        result.warnings.append(
            f"{scale_name} está atado al fondo: los datos no separan esta "
            "fase del fondo, así que su fracción en peso no está determinada "
            "por la medida. Es el caso típico de una fase nanocristalina, "
            "cuyas reflexiones son tan anchas como la propia flexibilidad "
            "del fondo"
        )

    fractions = result.weight_fractions()
    visible = result.phase_contributions()
    for phase in result.phases:
        name = phase.crystal.name
        weight = fractions.get(name)
        seen = visible.get(name)
        if weight is None or seen is None or seen <= 0.0:
            continue
        if weight >= 0.25 and weight >= WEIGHT_OVER_SIGNAL * seen:
            result.warnings.append(
                f"{name}: {100 * weight:.0f} % en peso pero solo el "
                f"{100 * seen:.0f} % de la intensidad difractada. No es "
                "incoherente —la dispersión va como Z² y un elemento ligero "
                "pesa mucho por cada cuenta que produce— pero significa que "
                "esa fracción descansa sobre poca señal, y por eso hay que "
                "juzgarla contra el fondo antes de citarla"
            )


#: Background polynomial orders a sensitivity scan tries by default.
#:
#: Spread around the package default rather than centred on it, because
#: the point is to see whether the default is on a plateau or on a spike,
#: and on the pattern that prompted this it was on a spike.
SENSITIVITY_ORDERS = (4, 6, 8, 10)


@dataclass
class BackgroundSensitivity:
    """What the weight fractions did as the background was allowed to flex.

    The honest answer to "is this fraction a measurement?" is not a
    standard error — the refinement's own errors are famously optimistic
    and, worse, they are conditional on the background model being right,
    which is precisely what is in question. The answer is to change the
    background model and look.
    """

    orders: tuple[int, ...]
    r_wp: dict[int, float]
    gof: dict[int, float]
    fractions: dict[int, dict[str, Optional[float]]]

    def spread(self) -> dict[str, float]:
        """Largest minus smallest weight fraction, per phase, over the scan."""
        out: dict[str, float] = {}
        for name in next(iter(self.fractions.values()), {}):
            values = [
                f[name] for f in self.fractions.values()
                if f.get(name) is not None
            ]
            out[name] = (max(values) - min(values)) if values else 0.0
        return out

    def best_order(self) -> int:
        """The order with the lowest Rwp — the background that fits best."""
        return min(self.r_wp, key=lambda order: self.r_wp[order])

    def summary(self) -> str:
        lines = [
            "Sensibilidad de las fracciones al orden del fondo",
            "",
            "  El fondo no es un dato: es un modelo, y una reflexión ancha y",
            "  un polinomio flexible describen la misma forma. Si la fracción",
            "  se mueve al cambiar el orden, esa fracción es una elección.",
            "",
        ]
        names = sorted(next(iter(self.fractions.values()), {}))
        header = "  %-7s %8s %8s   " % ("orden", "Rwp %", "GOF")
        lines.append(header + "  ".join("%-10s" % n[:10] for n in names))
        for order in self.orders:
            row = "  %-7d %8.3f %8.4f   " % (
                order, 100 * self.r_wp[order], self.gof[order])
            cells = []
            for name in names:
                value = self.fractions[order].get(name)
                cells.append("%-10s" % ("n/d" if value is None
                                        else "%.1f %%" % (100 * value)))
            lines.append(row + "  ".join(cells))
        lines.append("")
        lines.append(f"  Mejor Rwp: orden {self.best_order()}")
        lines.append("")
        for name, width in sorted(self.spread().items(),
                                  key=lambda kv: -kv[1]):
            verdict = ("NO está determinada por la medida" if width >= 0.10
                       else "estable")
            lines.append("  %-24s varía %5.1f puntos — %s"
                         % (name, 100 * width, verdict))
        return "\n".join(lines)


def background_order_sensitivity(
    pattern: Pattern,
    phases: Sequence[PhaseModel] | Sequence[Crystal],
    orders: Sequence[int] = SENSITIVITY_ORDERS,
    callback: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    **options,
) -> BackgroundSensitivity:
    """Refine the same phases against several background orders.

    One refinement per order, so this costs what a refinement costs times
    the length of ``orders``. It is a separate call and not part of
    `auto_refine` for that reason.

    What it is for: on a real CVD pattern of carbon grown on FeSe —
    alpha-Fe, a 6 nm turbostratic carbon, cementite and two iron
    selenides — the turbostratic carbon came back at 25.9 % of the
    sample by weight with a second-order background, 36.6 % at fourth,
    52.4 % at sixth, 33.1 % at eighth and 33.1 % at tenth, while Rwp fell
    monotonically from 5.34 to 4.00. The package's default sixth order
    was the WORST of the five and the one that produced the outlier, and
    nothing in a single refinement could have shown that: it converged,
    its GOF was 1.05, and its difference curve looked like the others.

    Each refinement starts fresh from the same phases, so an order does
    not inherit the previous one's minimum.
    """
    import copy

    models = [
        p if isinstance(p, PhaseModel) else PhaseModel(crystal=p) for p in phases
    ]
    chosen = tuple(int(order) for order in orders)
    r_wp: dict[int, float] = {}
    gof: dict[int, float] = {}
    fractions: dict[int, dict[str, Optional[float]]] = {}
    for index, order in enumerate(chosen, start=1):
        if should_stop is not None and should_stop():
            raise RefinementCancelled(
                f"detenido antes del orden {order}")
        inner = (
            lambda text, n=index, o=order:
            callback(f"[fondo {n}/{len(chosen)}, orden {o}] {text}")
        ) if callback else None
        result = auto_refine(
            pattern,
            copy.deepcopy(models),
            background_order=order,
            callback=inner,
            should_stop=should_stop,
            **options,
        )
        r_wp[order] = result.r_wp
        gof[order] = result.gof
        fractions[order] = result.weight_fractions()
    return BackgroundSensitivity(chosen, r_wp, gof, fractions)


__all__ = [
    "BACKGROUND_DOMINATES",
    "BackgroundSensitivity",
    "SENSITIVITY_ORDERS",
    "background_order_sensitivity",
    "DEFAULT_BACKGROUND_ORDER",
    "GOF_LIMIT",
    "SCALE_BACKGROUND_DEGENERATE",
    "WEIGHT_OVER_SIGNAL",
    "PROFILE_CUTOFF",
    "REFINE_MIN_RELATIVE",
    "SIZE_CEILING",
    "STAGES",
    "TEXTURE_LIMITS",
    "Parameter",
    "PhaseModel",
    "RefinementCancelled",
    "RefinementError",
    "RietveldResult",
    "auto_refine",
    "build_parameters",
    "calculate_pattern",
    "chebyshev_background",
    "fit_columns",
    "free_kinds",
    "r_factors",
    "refine",
    "reflection_ticks",
    "write_fit",
]
