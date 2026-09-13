"""Constrained peak fitting for photoelectron spectra.

A high-resolution XPS region is fitted with a chosen number of components,
and the number is a decision the person makes, not one the software should
make quietly: three components in an N 1s is a claim about chemistry, and a
fourth will always improve the residual. What this module does is make the
claim checkable — every component carries the literature state it is meant
to be, every constraint is written down, and the statistics that actually
detect an over-fitted model (χ² against counting statistics, the
Durbin–Watson statistic of the residual) are reported beside the areas.

Four things are built in because leaving them out is how XPS fits go wrong:

**Spin–orbit doublets are one component, not two.** A 2p line is two peaks
whose separation and area ratio are properties of the atom — 13.1 eV and
1:2 for iron — not free parameters. Here a component with a doublet
evaluates both peaks from one set of parameters, so the ratio cannot drift
and the fitter cannot buy residual by breaking it. Fitting Fe 2p3/2 and
2p1/2 as independent peaks is the fastest known way to invent a
stoichiometry.

**The background is inside the fit.** The Shirley step depends on the peak
envelope, and the peak envelope depends on the background. Computing the
background once from the raw data and then fitting is the common shortcut;
this module iterates the two, so the final background is the one implied by
the final peaks (the "active" approach), and reports how much the areas
moved between iterations. If they moved a lot, the region is too narrow.

**The weights are counting statistics.** Least squares with equal weights
treats a point at 40 000 counts and one at 400 as equally informative, and
the peak maximum is where the fit is least certain in relative terms. With
σ = √N the reduced χ² also becomes interpretable: near 1 the model explains
the data to within the noise, and well above 1 it does not — which is the
one statistic in XPS that can say "this fit is wrong" out loud. R², which
sits at 0.999 for a visibly bad fit, cannot.

**A width below the instrument's resolution is not a measurement.** The
analyser pass energy and the X-ray line width set a floor. A component
fitted narrower than that floor is the fitter exploiting a degree of
freedom, and it is flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..core.compat import trapezoid
from ..models.constraints import Link, order as order_links, parse_link, validate
from .background import Background, estimate_background
from .elements import Doublet, XPSDatabase, load_xps_database
from .lineshapes import XPS_PROFILES, ds_peak, profile_fwhm, resolve_xps_profile
from .spectrum import XPSError, XPSSpectrum

#: X-ray line widths in eV, as the contribution to the instrument function.
#: A monochromated aluminium source is quoted between 0.16 and 0.5 eV
#: depending on how hard the monochromator is being pushed; 0.30 is a fair
#: middle for a modern instrument and errs towards permissive.
XRAY_LINEWIDTH = {"Al mono": 0.30, "Al": 0.85, "Mg": 0.70}

#: Analyser resolution as a fraction of the pass energy. Real spectrometers
#: run between about 0.015 and 0.04 depending on slit and lens mode; the
#: lower end is used so the warning fires only when a width is clearly
#: impossible rather than merely optimistic.
ANALYSER_RESOLUTION_FRACTION = 0.015

#: A component peaking below this many times the local noise is not a
#: component. A least-squares fit always returns as many peaks as it is
#: given, and on a flat region it returns them with confident-looking
#: parameters.
MIN_SIGNIFICANCE = 3.0


@dataclass
class XPSComponent:
    """One component of a region fit.

    Parameters
    ----------
    name:
        Identifier, e.g. ``"piridínico"``. It is what the constraint syntax
        refers to, so it carries **no spaces**; the readable name goes in
        ``label``.
    label:
        What the report prints. Defaults to ``name``.
    profile:
        A key of :data:`~ramancarbon.xps.lineshapes.XPS_PROFILES`.
    centre, height, fwhm:
        Starting values. ``centre`` is the binding energy in eV of the
        **low**-binding-energy member when the component is a doublet.
    centre_bounds, height_bounds, fwhm_bounds:
        ``(low, high)``. ``None`` lets the fitter choose: ±1 eV on the
        centre, [0, 10×] on the height, [0.3, 6] eV on the width — narrow
        by Raman standards because XPS chemical shifts are small and a
        centre allowed to wander 5 eV will find another state's peak.
    extra, extra_bounds:
        The profile's extra parameters (mixing, asymmetry…).
    fixed:
        Parameter names held constant.
    doublet:
        The spin–orbit partner, from the database. Evaluated from the same
        parameters: the splitting and the area ratio are not fitted.
    state, element, line:
        What this component is claimed to be. ``state`` is a key in
        ``xps.json``'s chemical states, and carrying it here is what lets
        the report say *why* a component sits where it does.
    """

    name: str
    centre: float
    height: float = 1.0
    fwhm: float = 1.2
    profile: str = "gl"
    label: str = ""
    centre_bounds: Optional[tuple[float, float]] = None
    height_bounds: Optional[tuple[float, float]] = None
    fwhm_bounds: Optional[tuple[float, float]] = None
    extra: Optional[tuple[float, ...]] = None
    extra_bounds: Optional[tuple[tuple[float, float], ...]] = None
    fixed: tuple[str, ...] = ()
    doublet: Optional[Doublet] = None
    state: Optional[str] = None
    element: Optional[str] = None
    line: Optional[str] = None
    satellite: bool = False
    """Whether this component is a shake-up satellite of another. Satellites
    belong to the same species as their parent — their area counts with it,
    not separately — and they are exempt from the check that a component
    still sits inside its state's published window, because a satellite
    sits several eV outside it by definition."""
    justification: str = ""
    """Why this component is in the model. Printed in the report."""

    def __post_init__(self) -> None:
        self.label = self.label or self.name
        if " " in self.name:
            raise XPSError(
                f"el nombre de componente {self.name!r} lleva espacios, y es "
                "lo que usa la sintaxis de ligaduras para referirse a él. Usa "
                "un nombre sin espacios y pon el texto legible en label="
            )
        self.profile = resolve_xps_profile(self.profile)
        spec = XPS_PROFILES[self.profile]
        if self.extra is None:
            self.extra = tuple(spec["defaults"])
        elif len(self.extra) != len(spec["defaults"]):
            raise XPSError(
                f"el perfil {self.profile!r} lleva {len(spec['defaults'])} "
                f"parámetro(s) extra y se han dado {len(self.extra)}"
            )
        if self.extra_bounds is None:
            self.extra_bounds = tuple(spec["bounds"])
        if self.fwhm <= 0:
            raise XPSError(f"componente {self.name!r}: la anchura tiene que ser positiva")

    @property
    def extra_names(self) -> tuple[str, ...]:
        return tuple(XPS_PROFILES[self.profile]["extra"])

    @property
    def is_asymmetric(self) -> bool:
        return bool(XPS_PROFILES[self.profile]["asymmetric"])

    def evaluate(self, x: np.ndarray, centre: Optional[float] = None,
                 height: Optional[float] = None, fwhm: Optional[float] = None,
                 extra: Optional[Sequence[float]] = None) -> np.ndarray:
        """The component's curve, spin–orbit partner included."""
        function = XPS_PROFILES[self.profile]["function"]
        c = self.centre if centre is None else centre
        h = self.height if height is None else height
        w = self.fwhm if fwhm is None else fwhm
        e = tuple(self.extra if extra is None else extra)
        curve = function(x, c, h, w, *e)
        if self.doublet is not None:
            curve = curve + function(
                x, c + self.doublet.splitting_ev, h * self.doublet.ratio, w, *e
            )
        return curve


@dataclass
class XPSModel:
    """Components, window, background and constraints for one region."""

    components: list[XPSComponent]
    window: tuple[float, float]
    background: str = "shirley"
    name: str = "región"
    links: tuple[Link, ...] = ()
    """Parameter links, in the same ``objetivo = factor × origen +
    desplazamiento`` syntax the Raman side uses. The classic XPS uses are
    tying all the components of one region to a common width, and tying a
    shake-up satellite's position to its parent's."""
    region_label: str = ""
    """The database region this model is for, e.g. ``"N 1s"``."""

    def __post_init__(self) -> None:
        if not self.components:
            raise XPSError("un modelo necesita al menos una componente")
        low, high = self.window
        if high <= low:
            raise XPSError(f"ventana vacía [{low}, {high}]")
        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise XPSError(
                "dos componentes con el mismo nombre; las ligaduras y el "
                f"informe las identifican por nombre ({', '.join(names)})"
            )
        outside = [c.name for c in self.components
                   if not low - 2.0 <= c.centre <= high + 2.0]
        if outside:
            raise XPSError(
                f"estas componentes empiezan fuera de la ventana "
                f"[{low:g}, {high:g}] eV: {', '.join(outside)}"
            )
        self.links = tuple(
            parse_link(item) if isinstance(item, str) else item for item in self.links
        )
        if self.links:
            validate(self.links, self.parameter_names())
            self.links = tuple(order_links(self.links))

    def parameter_names(self) -> dict[str, tuple[str, ...]]:
        return {
            c.name: ("centre", "height", "fwhm") + c.extra_names
            for c in self.components
        }

    @property
    def linked(self) -> set[tuple[str, str]]:
        return {link.target for link in self.links}


@dataclass
class XPSFittedComponent:
    """One component after fitting."""

    name: str
    label: str
    profile: str
    centre: float
    height: float
    fwhm: float
    extra: tuple[float, ...]
    extra_names: tuple[str, ...]
    area: float
    """Integrated over the fit window, partner included. Windowed on
    purpose: with an asymmetric profile there is no other kind."""
    area_fraction: float
    true_fwhm: float
    """The profile's actual width, which is not the ``fwhm`` parameter for
    the GL product form."""
    peak_position: float
    peak_height: float
    doublet: Optional[Doublet] = None
    state: Optional[str] = None
    element: Optional[str] = None
    line: Optional[str] = None
    satellite: bool = False
    justification: str = ""
    errors: dict[str, float] = field(default_factory=dict)
    fixed: tuple[str, ...] = ()

    def curve(self, x: np.ndarray) -> np.ndarray:
        function = XPS_PROFILES[self.profile]["function"]
        out = function(x, self.centre, self.height, self.fwhm, *self.extra)
        if self.doublet is not None:
            out = out + function(
                x, self.centre + self.doublet.splitting_ev,
                self.height * self.doublet.ratio, self.fwhm, *self.extra
            )
        return out

    def summary(self) -> str:
        error = self.errors.get("centre")
        centre = f"{self.centre:.2f}" + (f" ± {error:.2f}" if error else "")
        tail = ""
        if self.extra_names:
            tail = "  " + ", ".join(
                f"{n}={v:.3f}" for n, v in zip(self.extra_names, self.extra))
        doublet = f"  doblete Δ={self.doublet.splitting_ev:.2f} eV" if self.doublet else ""
        return (
            f"{self.label:<26s} {centre:>13s} eV  FWHM={self.true_fwhm:.2f}  "
            f"A={self.area:.4g}  {100 * self.area_fraction:5.1f} %{doublet}{tail}"
        )


@dataclass
class XPSFitResult:
    """A fitted region, with the grounds for believing it or not."""

    components: list[XPSFittedComponent]
    background: Background
    energy: np.ndarray
    counts: np.ndarray
    fitted: np.ndarray
    """Peaks plus background, on the same scale as ``counts``."""
    residual: np.ndarray
    r_squared: float
    reduced_chi2: float
    durbin_watson: Optional[float]
    n_parameters: int
    success: bool
    message: str
    window: tuple[float, float]
    iterations: int = 1
    """Background/peak iterations. More than one means the active Shirley."""
    area_drift: float = 0.0
    """Largest relative change of any component area in the last iteration."""
    links: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)
    region_label: str = ""
    weighted: bool = True
    acquisition: dict = field(default_factory=dict)
    """Dwell time, sweeps and intensity unit of the spectrum that was fitted.

    Carried because an area is only comparable with another area measured
    the same way: a region collected with four times the sweeps has four
    times the area at the same concentration, and a quantification that
    mixes the two without dividing by the acquisition time is wrong by
    exactly that factor."""

    @property
    def peaks(self) -> np.ndarray:
        """The sum of the components, background excluded."""
        return self.fitted - self.background.values

    def component(self, name: str) -> Optional[XPSFittedComponent]:
        for item in self.components:
            if name in (item.name, item.state, item.label):
                return item
        return None

    def total_area(self) -> float:
        return float(sum(c.area for c in self.components))

    def summary(self) -> str:
        lines = [
            f"{self.region_label or 'Región'} {self.window[0]:.1f}–{self.window[1]:.1f} eV: "
            f"{len(self.components)} componentes, {self.n_parameters} parámetros libres",
            self.background.describe()
            + (f"; {self.iterations} pasadas fondo↔picos, las áreas se movieron "
               f"{100 * self.area_drift:.1f} % en la última"
               if self.iterations > 1 else ""),
            f"R² = {self.r_squared:.5f}   χ²_red = {self.reduced_chi2:.3f}"
            + (f"   DW = {self.durbin_watson:.2f}" if self.durbin_watson else "")
            + ("" if self.weighted else "   (sin pesar por estadística de conteo)"),
            "",
        ]
        lines.extend(c.summary() for c in self.components)
        if self.links:
            lines.append("")
            lines.append("Ligaduras: " + "; ".join(self.links))
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


# ----------------------------------------------------------------------
# packing
# ----------------------------------------------------------------------
def _pack(model: XPSModel) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    list[tuple[int, str]]]:
    """Free parameters, their bounds, and which slot is which."""
    x0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    layout: list[tuple[int, str]] = []
    linked = model.linked
    for index, component in enumerate(model.components):
        defaults = {
            "centre": (component.centre, component.centre_bounds
                       or (component.centre - 1.0, component.centre + 1.0)),
            "height": (component.height, component.height_bounds
                       or (0.0, max(abs(component.height) * 10.0, 1e-6))),
            "fwhm": (component.fwhm, component.fwhm_bounds or (0.3, 6.0)),
        }
        for key in ("centre", "height", "fwhm"):
            if key in component.fixed or (component.name, key) in linked:
                continue
            value, (low, high) = defaults[key]
            x0.append(float(np.clip(value, low, high)))
            lower.append(float(low))
            upper.append(float(high))
            layout.append((index, key))
        for position, name in enumerate(component.extra_names):
            if name in component.fixed or (component.name, name) in linked:
                continue
            low, high = component.extra_bounds[position]
            x0.append(float(np.clip(component.extra[position], low, high)))
            lower.append(float(low))
            upper.append(float(high))
            layout.append((index, name))
    if not x0:
        raise XPSError(
            "el modelo no tiene ningún parámetro libre: todo está fijo o ligado"
        )
    return np.asarray(x0), np.asarray(lower), np.asarray(upper), layout


class _Evaluator:
    """Resolved model evaluation for the least-squares inner loop.

    Same design as the Raman fitter's: the mapping from parameter slot to
    profile argument is fixed once the model is built, and the links are
    applied on a flat table so a link never needs its source component to
    have been evaluated first.
    """

    __slots__ = ("_x", "_components", "_flat_indices", "_safe", "_defaults",
                 "_links", "_doublets")

    def __init__(self, model: XPSModel, layout: list[tuple[int, str]],
                 x: np.ndarray) -> None:
        self._x = x
        slots = {(index, key): slot for slot, (index, key) in enumerate(layout)}
        flat_indices: list[int] = []
        defaults: list[float] = []
        position: dict[tuple[str, str], int] = {}
        self._components = []
        self._doublets = []
        for index, component in enumerate(model.components):
            names = ("centre", "height", "fwhm") + component.extra_names
            values = (component.centre, component.height, component.fwhm) \
                + tuple(component.extra)
            start = len(flat_indices)
            for name, value in zip(names, values):
                position[(component.name, name)] = len(flat_indices)
                flat_indices.append(slots.get((index, name), -1))
                defaults.append(float(value))
            self._components.append(
                (XPS_PROFILES[component.profile]["function"], start, len(flat_indices))
            )
            self._doublets.append(
                None if component.doublet is None
                else (float(component.doublet.splitting_ev),
                      float(component.doublet.ratio))
            )
        self._flat_indices = np.asarray(flat_indices, dtype=np.intp)
        self._safe = np.maximum(self._flat_indices, 0)
        self._defaults = np.asarray(defaults, dtype=float)
        self._links = [
            (position[link.target], position[link.source],
             float(link.factor), float(link.offset))
            for link in model.links
        ]

    def flat(self, params: np.ndarray) -> np.ndarray:
        values = np.where(self._flat_indices >= 0, params[self._safe], self._defaults)
        for target, source, factor, offset in self._links:
            values[target] = factor * values[source] + offset
        return values

    def curves(self, params: np.ndarray) -> list[np.ndarray]:
        flat = self.flat(params)
        out = []
        for (function, start, stop), doublet in zip(self._components, self._doublets):
            block = flat[start:stop]
            curve = function(self._x, *block)
            if doublet is not None:
                splitting, ratio = doublet
                curve = curve + function(
                    self._x, block[0] + splitting, block[1] * ratio, *block[2:]
                )
            out.append(curve)
        return out

    def __call__(self, params: np.ndarray) -> np.ndarray:
        total = np.zeros_like(self._x)
        for curve in self.curves(params):
            total += curve
        return total


# ----------------------------------------------------------------------
# fitting
# ----------------------------------------------------------------------
def fit_region(
    spectrum: XPSSpectrum,
    model: XPSModel,
    weighted: bool = True,
    background_iterations: int = 3,
    max_nfev: int = 8000,
    database: Optional[XPSDatabase] = None,
) -> XPSFitResult:
    """Fit one region, iterating the background against the peaks.

    Parameters
    ----------
    spectrum:
        The measurement. Charge-reference it **before** fitting: the
        literature windows in ``xps.json`` are on the referenced scale, and
        a component matched to a state on an unreferenced axis is matched
        to the wrong state.
    model:
        Components, window, background and constraints.
    weighted:
        Weight each point by ``1/√N`` — counting statistics. Switched off
        only for a spectrum whose intensity scale is not counts; the result
        records which was used, because χ² means nothing without it.
    background_iterations:
        How many times to recompute the background from the fitted
        envelope. One means the background is computed from the raw data
        and left alone.

    Returns
    -------
    XPSFitResult
    """
    database = database or load_xps_database()
    low, high = model.window
    energy, counts = spectrum.region_of(low, high)
    if energy.size < 10:
        raise XPSError(
            f"la ventana {low:g}–{high:g} eV deja {energy.size} puntos; el "
            f"espectro va de {spectrum.range[0]:.1f} a {spectrum.range[1]:.1f} eV"
        )

    x0, lower, upper, layout = _pack(model)
    if energy.size <= x0.size:
        raise XPSError(
            f"{x0.size} parámetros libres y solo {energy.size} puntos en "
            f"[{low:g}, {high:g}] eV: el ajuste está indeterminado"
        )

    sigma, weighted = _weights(spectrum, energy, counts, weighted)
    evaluate = _Evaluator(model, layout, energy)

    background = estimate_background(spectrum, (low, high), model.background)
    params = x0
    previous_areas: Optional[np.ndarray] = None
    drift = 0.0
    result = None
    iterations = max(1, int(background_iterations))
    used = 0
    for used in range(1, iterations + 1):
        target = counts - background.values

        def residual(values: np.ndarray) -> np.ndarray:
            return (evaluate(values) - target) / sigma

        result = least_squares(
            residual, params, bounds=(lower, upper), max_nfev=max_nfev,
            x_scale="jac",
        )
        params = result.x
        areas = np.array([
            float(trapezoid(curve, energy)) for curve in evaluate.curves(params)
        ])
        if previous_areas is not None:
            reference = np.where(previous_areas > 0, previous_areas, np.inf)
            drift = float(np.max(np.abs(areas - previous_areas) / reference))
        previous_areas = areas
        if used == iterations or model.background not in ("shirley", "Shirley"):
            break
        envelope = evaluate(params)
        background = estimate_background(
            spectrum, (low, high), model.background, envelope=envelope
        )
        if drift and drift < 1e-3:
            break

    assert result is not None
    fitted_peaks = evaluate(params)
    fitted = fitted_peaks + background.values
    residual_values = counts - fitted
    weighted_residual = residual_values / sigma

    n, k = energy.size, params.size
    ss_res = float(np.sum(residual_values**2))
    ss_tot = float(np.sum((counts - np.mean(counts)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(n - k, 1)
    reduced_chi2 = float(np.sum(weighted_residual**2)) / dof
    watson = _durbin_watson(residual_values)

    errors = _uncertainties(result, weighted_residual, layout, model, weighted)
    flat = evaluate.flat(params)
    components = _finish(model, evaluate, flat, energy, previous_areas)

    warnings = _warnings(
        spectrum, model, components, background, result, reduced_chi2, watson,
        residual_values, sigma, weighted, drift, used, database,
    )
    for index, component in enumerate(components):
        component.errors = errors.get(index, {})

    return XPSFitResult(
        components=components,
        background=background,
        energy=energy,
        counts=counts,
        fitted=fitted,
        residual=residual_values,
        r_squared=r_squared,
        reduced_chi2=reduced_chi2,
        durbin_watson=watson,
        n_parameters=k,
        success=bool(result.success),
        message=str(result.message),
        window=(float(energy[0]), float(energy[-1])),
        iterations=used,
        area_drift=drift,
        links=tuple(str(link) for link in model.links),
        warnings=warnings,
        region_label=model.region_label or model.name,
        weighted=weighted,
        acquisition={
            "dwell_s": spectrum.dwell_s,
            "sweeps": spectrum.sweeps,
            "unidad": spectrum.intensity_unit,
            "energía_paso_ev": spectrum.pass_energy,
        },
    )


def _weights(spectrum: XPSSpectrum, energy: np.ndarray, counts: np.ndarray,
             weighted: bool) -> tuple[np.ndarray, bool]:
    """σ per point, from counting statistics where they are recoverable.

    A spectrum stored in counts per second has had its statistics divided
    away; multiplying back by the dwell time recovers the number of counts
    that were actually collected, and σ in the stored units is then
    ``√N / t``. Without the dwell time there is no way back, and the fit
    falls back to equal weights and says so rather than treating counts per
    second as if they were counts — which would understate σ by √t and
    hand back uncertainties several times too small.
    """
    if not weighted:
        return np.ones_like(counts), False
    scale = 1.0
    if spectrum.intensity_unit.startswith("cuentas/"):
        if spectrum.dwell_s is None:
            return np.ones_like(counts), False
        scale = float(spectrum.dwell_s) * float(spectrum.sweeps or 1)
        if scale <= 0:
            return np.ones_like(counts), False
    accumulated = np.clip(counts * scale, 1.0, None)
    return np.sqrt(accumulated) / scale, True


def _finish(model: XPSModel, evaluate: _Evaluator, flat: np.ndarray,
            energy: np.ndarray, areas: np.ndarray) -> list[XPSFittedComponent]:
    """Turn the fitted vector into reportable components."""
    total = float(np.sum(areas)) if areas is not None else 0.0
    out: list[XPSFittedComponent] = []
    cursor = 0
    for index, component in enumerate(model.components):
        width = 3 + len(component.extra_names)
        block = flat[cursor:cursor + width]
        cursor += width
        centre, height, fwhm = float(block[0]), float(block[1]), float(block[2])
        extras = tuple(float(value) for value in block[3:])
        if component.is_asymmetric:
            position, apex = ds_peak(centre, fwhm, extras[0], height)
        else:
            position, apex = centre, height
        out.append(
            XPSFittedComponent(
                name=component.name,
                label=component.label,
                profile=component.profile,
                centre=centre,
                height=height,
                fwhm=fwhm,
                extra=extras,
                extra_names=component.extra_names,
                area=float(areas[index]) if areas is not None else 0.0,
                area_fraction=(float(areas[index]) / total) if total > 0 else 0.0,
                true_fwhm=profile_fwhm(component.profile, fwhm, extras),
                peak_position=position,
                peak_height=apex,
                doublet=component.doublet,
                state=component.state,
                element=component.element,
                line=component.line,
                satellite=component.satellite,
                justification=component.justification,
                fixed=component.fixed,
            )
        )
    return out


def _durbin_watson(residual: np.ndarray) -> Optional[float]:
    """``Σ(rᵢ − rᵢ₋₁)² / Σrᵢ²`` — 2 for white noise, low for structure."""
    if residual.size < 3:
        return None
    denominator = float(np.sum(residual**2))
    if denominator <= 0:
        return None
    return float(np.sum(np.diff(residual) ** 2) / denominator)


def _uncertainties(result, weighted_residual: np.ndarray,
                   layout: list[tuple[int, str]], model: XPSModel,
                   weighted: bool) -> dict[int, dict[str, float]]:
    """Standard errors from the Jacobian at the solution.

    With counting-statistics weights the covariance is ``(JᵀJ)⁻¹``
    directly; unweighted it has to be scaled by the residual variance.
    Either way these are the fit's own *analytic* errors, which are known
    to be optimistic whenever components overlap — and in XPS they always
    overlap. They are reported because they are the cheapest available
    signal, not because they are the last word.
    """
    try:
        _, singular, vt = np.linalg.svd(result.jac, full_matrices=False)
    except np.linalg.LinAlgError:               # pragma: no cover
        return {}
    threshold = np.finfo(float).eps * max(result.jac.shape) * singular[0]
    keep = singular > threshold
    if not keep.any():                          # pragma: no cover
        return {}
    covariance = (vt[keep].T / singular[keep] ** 2) @ vt[keep]
    if not weighted:
        dof = max(weighted_residual.size - result.x.size, 1)
        covariance = covariance * float(np.sum(weighted_residual**2)) / dof
    diagonal = np.clip(np.diag(covariance), 0.0, None)
    errors: dict[int, dict[str, float]] = {}
    for slot, (index, key) in enumerate(layout):
        errors.setdefault(index, {})[key] = float(np.sqrt(diagonal[slot]))
    return errors


def resolution_floor(spectrum: XPSSpectrum) -> Optional[float]:
    """The narrowest FWHM the instrument could produce, in eV.

    ``√(ΔE_analyser² + ΔE_photon²)``, with the analyser term taken as a
    conservative 1.5 % of the pass energy. Returns ``None`` when the pass
    energy is unknown, which is the usual state of a text export — and the
    reason to record it.
    """
    if spectrum.pass_energy is None:
        return None
    analyser = ANALYSER_RESOLUTION_FRACTION * float(spectrum.pass_energy)
    if spectrum.photon_energy and spectrum.photon_energy < 1300:
        photon = XRAY_LINEWIDTH["Mg"]
    elif spectrum.monochromated:
        photon = XRAY_LINEWIDTH["Al mono"]
    else:
        photon = XRAY_LINEWIDTH["Al"]
    return float(np.hypot(analyser, photon))


def _warnings(spectrum: XPSSpectrum, model: XPSModel,
              components: list[XPSFittedComponent], background: Background,
              result, reduced_chi2: float, watson: Optional[float],
              residual: np.ndarray, sigma: np.ndarray, weighted: bool,
              drift: float, iterations: int,
              database: XPSDatabase) -> list[str]:
    """Everything about this fit that should stop somebody publishing it."""
    out: list[str] = list(background.warnings)
    if not result.success:
        out.append(f"el ajuste no ha convergido: {result.message}")
    if not weighted:
        out.append(
            "el ajuste va sin pesar por estadística de conteo (el espectro "
            "está en cuentas por segundo y no se sabe el tiempo por canal, o "
            "se ha pedido así): el χ² reducido de arriba no se puede "
            "comparar con 1"
        )

    noise = float(np.median(sigma))
    weak = [c.label for c in components
            if c.peak_height < MIN_SIGNIFICANCE * noise]
    if weak:
        out.append(
            f"estas componentes no llegan a {MIN_SIGNIFICANCE:.0f} veces el "
            f"ruido ({noise:.0f} cuentas) y no se sostienen: "
            f"{', '.join(weak)}. Un ajuste por mínimos cuadrados SIEMPRE "
            "devuelve tantos picos como se le den"
        )

    floor = resolution_floor(spectrum)
    if floor is not None:
        narrow = [f"{c.label} ({c.true_fwhm:.2f} eV)" for c in components
                  if c.true_fwhm < floor]
        if narrow:
            out.append(
                f"la resolución del equipo (E_paso {spectrum.pass_energy:.0f} eV, "
                f"fuente incluida) no baja de {floor:.2f} eV y estas "
                f"componentes salen más estrechas: {', '.join(narrow)}. No es "
                "una anchura medida, es el ajuste aprovechando un grado de "
                "libertad"
            )
    elif spectrum.pass_energy is None:
        out.append(
            "sin energía de paso no hay suelo de resolución con el que "
            "juzgar las anchuras ajustadas"
        )

    if weighted and reduced_chi2 > 4.0:
        out.append(
            f"χ²_red = {reduced_chi2:.1f}. Con pesos de conteo eso significa "
            "que el modelo se aparta de los datos varias veces más de lo que "
            "explica el ruido. Por orden de probabilidad: falta una "
            "componente; hay anchuras ligadas que los datos no comparten "
            "(suelta la ligadura y mira si baja); el fondo no es el que toca; "
            "o la forma de línea es simétrica donde no debería"
        )
    if watson is not None and watson < 1.0:
        out.append(
            f"los residuos están correlacionados (Durbin-Watson = {watson:.2f}, "
            "sería ~2 si fueran ruido): el modelo deja estructura sin ajustar"
        )

    if iterations > 1 and drift > 0.02:
        out.append(
            f"las áreas se han movido un {100 * drift:.0f} % en la última "
            "iteración del fondo activo: el fondo y los picos no están de "
            "acuerdo, que casi siempre es una ventana demasiado estrecha"
        )

    # Components that are not resolved from each other.
    ordered = sorted(components, key=lambda c: c.centre)
    for first, second in zip(ordered, ordered[1:]):
        gap = second.centre - first.centre
        width = 0.5 * (first.true_fwhm + second.true_fwhm)
        if gap < 0.5 * width:
            out.append(
                f"«{first.label}» y «{second.label}» están a {gap:.2f} eV con "
                f"anchuras de ~{width:.2f} eV: no están resueltas, y el "
                "reparto de área entre ellas lo decide el modelo, no la medida"
            )

    asymmetric = [item for item in components if item.extra_names
                  and "asymmetry" in item.extra_names]
    if asymmetric and background.kind.startswith(("Shirley", "Tougaard")):
        worst = max(asymmetric,
                    key=lambda item: item.extra[item.extra_names.index("asymmetry")])
        alpha = worst.extra[worst.extra_names.index("asymmetry")]
        out.append(
            f"«{worst.label}» es asimétrica (α = {alpha:.2f}) y el fondo es "
            f"{background.kind}: esas dos cosas NO son independientes sobre "
            "una ventana finita. Una Doniach-Šunjić no decae a cero por "
            "ninguno de los dos lados —va como |u|^(α−1)— así que el fondo se "
            "come parte de la cola y la α ajustada sale BAJA. Medido sobre "
            "picos sintéticos con α conocida, el área de la componente "
            "metálica sale corta un 2 % con α = 0.05, un 5 % con 0.15 y un "
            "12 % con 0.30, y siempre en esa dirección. No es de este "
            "programa, es de la función: para acotarlo, repite con fondo "
            "lineal sobre una ventana estrecha y compara"
        )

    if not spectrum.monochromated:
        out.append(
            "la fuente no es monocromada: hay satélites de rayos X entre 8 y "
            "12 eV por debajo de cada línea en energía de enlace, y son "
            "instrumento, no química. Réstalos o modélalos antes de leer un "
            "estado en esa zona"
        )

    smoothed = spectrum.looks_smoothed()
    if smoothed:
        out.append(smoothed)

    # Components claiming a literature state that they no longer sit in.
    for component in components:
        if not component.state or not model.region_label or component.satellite:
            continue
        try:
            state = database.state(model.region_label, component.state)
        except Exception:                       # pragma: no cover - unknown key
            continue
        if not state.contains(component.centre, tolerance=0.15):
            out.append(
                f"«{component.label}» se ha ido a {component.centre:.2f} eV y "
                f"la ventana de {state.name} es "
                f"{state.window[0]:.1f}–{state.window[1]:.1f} eV: ya no es ese "
                "estado, o la referencia de carga está mal"
            )
    return out


__all__ = [
    "ANALYSER_RESOLUTION_FRACTION",
    "MIN_SIGNIFICANCE",
    "XRAY_LINEWIDTH",
    "XPSComponent",
    "XPSFitResult",
    "XPSFittedComponent",
    "XPSModel",
    "fit_region",
    "resolution_floor",
]
