"""Building a region model from the literature, instead of from scratch.

The whole difficulty of a high-resolution XPS fit is that the answer is
chosen before the fitting starts. How many components, at what binding
energies, with what widths — those are the claim; least squares only polishes
it. So this module builds models out of the states in ``xps.json``, where
every window carries a source and a confidence, and every component comes out
carrying the reason it is there.

Three ways in, in decreasing order of how much the literature does for you:

:func:`state_model`
    Pick states by name from the database. Each becomes a component whose
    centre is bounded by that state's published window and whose width is
    bounded by its published range. A component that then walks out of its
    window is reported as no longer being that state, which is the useful
    failure: it usually means the charge reference is wrong.

:func:`count_model`
    Say only how many components you want. The states of the region are
    taken in order of how well they match what is actually in the data, and
    you are told which were used and which were left out.

:func:`free_model`
    No literature at all: *n* components seeded from the second derivative.
    For a region the database does not cover, or for deliberately letting
    the data speak first.

And :func:`compare_counts`, which fits several component counts and reports
what each one buys — because "how many peaks" is the question this package
cannot answer for you, and the honest help is the evidence, not a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..models.constraints import Link
from .background import estimate_background
from .elements import ChemicalState, Doublet, XPSDatabase, load_xps_database
from .fitting import XPSComponent, XPSFitResult, XPSModel, fit_region
from .spectrum import XPSError, XPSSpectrum

#: Padding in eV added around the states' windows to make the fit window.
WINDOW_PAD = 4.0

#: How different two states' published widths may be and still be tied to
#: each other. A ratio of midpoints, so 1.5 means one may be half again as
#: broad as the other.
WIDTH_LINK_RATIO = 1.5

#: Default Gaussian–Lorentzian mixing, CasaXPS's GL(30). Held fixed unless
#: asked otherwise: with overlapping components the mixing and the width
#: trade against each other almost freely, and letting both go is how a
#: component ends up pure Gaussian next to one that is pure Lorentzian
#: without anything physical having changed.
DEFAULT_MIXING = 0.3


def _identifier(text: str) -> str:
    """A link-safe name: no spaces, since links refer to components by name."""
    return "_".join(text.split())


def doublet_for(region: str, database: XPSDatabase) -> Optional[Doublet]:
    """The spin–orbit partner implied by a region name, if there is one.

    ``"Fe 2p3/2"`` yields iron's 2p doublet, so a component placed on the
    3/2 line automatically carries its 1/2 partner at the right separation
    and the right area ratio. ``"C 1s"`` yields ``None``: an s level has no
    partner to carry.
    """
    parts = region.split()
    if len(parts) < 2:
        return None
    symbol, orbital = parts[0], parts[1]
    base = orbital.split("/")[0][:-1] if "/" in orbital else orbital
    try:
        element = database.element(symbol)
    except Exception:
        return None
    for line in element.lines:
        if line.label == f"{symbol} {base}" and line.doublet is not None:
            return line.doublet
    return None


def _height_guess(spectrum: XPSSpectrum, window: tuple[float, float],
                  energy: float, background: str) -> float:
    """Intensity above the background at one binding energy."""
    try:
        estimate = estimate_background(spectrum, window, background)
    except XPSError:                            # pragma: no cover - tiny window
        return float(np.max(spectrum.counts))
    axis, counts = spectrum.region_of(*window)
    value = float(np.interp(energy, axis, counts - estimate.values))
    return max(value, 0.05 * float(np.max(counts - estimate.values)), 1.0)


def _component_from_state(
    state: ChemicalState,
    spectrum: XPSSpectrum,
    window: tuple[float, float],
    background: str,
    doublet: Optional[Doublet],
    profile: Optional[str],
    fix_mixing: bool,
) -> XPSComponent:
    shape = profile or ("ds_gauss" if state.asymmetric else "gl")
    extra: tuple[float, ...]
    fixed: tuple[str, ...] = ()
    if shape in ("gl", "sgl"):
        extra = (DEFAULT_MIXING,)
        fixed = ("mixing",) if fix_mixing else ()
    elif shape == "ds_gauss":
        extra = (0.12, 0.6)
    elif shape == "ds":
        extra = (0.12,)
    else:
        extra = ()
    return XPSComponent(
        name=_identifier(state.key),
        label=state.name,
        centre=state.energy_ev,
        height=_height_guess(spectrum, window, state.energy_ev, background),
        fwhm=float(np.mean(state.fwhm)),
        profile=shape,
        centre_bounds=state.window,
        fwhm_bounds=state.fwhm,
        extra=extra,
        fixed=fixed,
        doublet=doublet,
        state=state.key,
        element=state.region.split()[0],
        line=state.region,
        justification=(
            f"{state.name}: {state.window[0]:.1f}–{state.window[1]:.1f} eV, "
            f"FWHM {state.fwhm[0]:.1f}–{state.fwhm[1]:.1f} eV, confianza "
            f"{state.confidence}"
            + (f". {state.note}" if state.note else "")
        ),
    )


def state_model(
    spectrum: XPSSpectrum,
    region: str,
    states: Optional[Sequence[str]] = None,
    window: Optional[tuple[float, float]] = None,
    background: str = "shirley",
    profile: Optional[str] = None,
    link_widths: bool = True,
    fix_mixing: bool = True,
    include_satellites: bool = False,
    database: Optional[XPSDatabase] = None,
) -> XPSModel:
    """Build a model from named literature states of one region.

    Parameters
    ----------
    region:
        A key of the database's chemical states, e.g. ``"N 1s"``,
        ``"Fe 2p3/2"``.
    states:
        Which states to include, by key. ``None`` takes every non-satellite
        state of the region, which is usually too many — the point of the
        argument is that the choice is yours and is recorded.
    link_widths:
        Tie together the widths of components the literature gives the
        same width to. Chemically shifted states of one element share a
        core hole, an analyser and a sample, so their widths really are
        similar, and tying them removes the parameter that overlapping
        components fight over.

        It is **not** applied blindly. Two states are tied only if their
        published FWHM ranges overlap and their midpoints are within a
        factor of :data:`WIDTH_LINK_RATIO` — because some states are
        genuinely broader than others for a physical reason, and forcing
        them equal is a worse error than leaving the width free. Fe(III) is
        multiplet-broadened to 1.5–4.5 eV while Fe–Se sits at 0.9–2.0 eV;
        tied together they fit a common 1.5 eV, leave a visible residual on
        both, and raise χ² by a factor of fifty. Asymmetric (metallic)
        components are never tied: their width parameter is the Lorentzian
        part alone and means something different.
    fix_mixing:
        Hold the Gaussian–Lorentzian mixing at :data:`DEFAULT_MIXING`.
    include_satellites:
        Add the shake-up satellites the database attaches to these states,
        each tied to its parent's position by a link.

    Returns
    -------
    XPSModel
    """
    database = database or load_xps_database()
    available = database.states_for(region, include_satellites=True)
    if not available:
        raise XPSError(
            f"la base de datos no tiene estados para la región {region!r}; "
            f"tiene: {', '.join(database.region_names())}"
        )
    index = {state.key: state for state in available}
    if states is None:
        chosen = [s for s in available if not s.is_satellite]
    else:
        missing = [key for key in states if key not in index]
        if missing:
            raise XPSError(
                f"estados desconocidos en {region!r}: {', '.join(missing)}; "
                f"hay: {', '.join(index)}"
            )
        chosen = [index[key] for key in states]
    if not chosen:
        raise XPSError(f"no se ha elegido ningún estado de {region!r}")

    if window is None:
        low = min(state.window[0] for state in chosen) - WINDOW_PAD
        high = max(state.window[1] for state in chosen) + WINDOW_PAD
        doublet = doublet_for(region, database)
        if doublet is not None:
            high += doublet.splitting_ev
        window = (low, high)
    window = (float(min(window)), float(max(window)))
    if not spectrum.covers(*window, fraction=0.95):
        raise XPSError(
            f"el espectro va de {spectrum.range[0]:.1f} a "
            f"{spectrum.range[1]:.1f} eV y la región {region} necesita "
            f"{window[0]:.1f}–{window[1]:.1f} eV. Un ajuste sobre una ventana "
            "truncada ancla el fondo dentro de un pico"
        )

    doublet = doublet_for(region, database)
    components = [
        _component_from_state(state, spectrum, window, background, doublet,
                              profile, fix_mixing)
        for state in chosen
    ]
    links: list[Link] = []
    if link_widths:
        links.extend(_width_links(
            [(component, state) for component, state in zip(components, chosen)
             if not component.is_asymmetric]
        ))
    if include_satellites:
        for state, component in zip(chosen, list(components)):
            if state.satellite_ev is None:
                continue
            satellite = XPSComponent(
                name=f"{component.name}_sat",
                label=f"{state.name} (satélite shake-up)",
                centre=state.satellite_ev,
                height=0.15 * component.height,
                fwhm=2.0 * component.fwhm,
                profile="gl",
                fwhm_bounds=(component.fwhm, 8.0),
                extra=(DEFAULT_MIXING,),
                fixed=("mixing",),
                doublet=doublet,
                state=state.key,
                element=component.element,
                line=component.line,
                satellite=True,
                justification=(
                    f"satélite shake-up de {state.name}, a "
                    f"{state.satellite_ev:.1f} eV ({state.satellite_offset:+.1f} "
                    "eV del principal). Es la misma especie: su área cuenta "
                    "con la del pico principal, no aparte"
                ),
            )
            components.append(satellite)
            links.append(Link(satellite.name, "centre", component.name, "centre",
                              1.0, float(state.satellite_offset)))
    return XPSModel(
        components=components,
        window=window,
        background=background,
        name=region,
        links=tuple(links),
        region_label=region,
    )


def _width_links(pairs: list[tuple[XPSComponent, ChemicalState]]) -> list[Link]:
    """Tie together the widths of states the literature widens alike.

    Greedy grouping: each component joins the first existing group whose
    leader it is width-compatible with, and starts a new group otherwise.
    A group of one produces no link, which is the point — a state nobody
    else matches keeps its own free width.
    """
    groups: list[list[tuple[XPSComponent, ChemicalState]]] = []
    for component, state in pairs:
        for group in groups:
            if _width_compatible(group[0][1], state):
                group.append((component, state))
                break
        else:
            groups.append([(component, state)])
    links: list[Link] = []
    for group in groups:
        leader = group[0][0]
        for component, _ in group[1:]:
            links.append(Link(component.name, "fwhm", leader.name, "fwhm"))
    return links


def _width_compatible(first: ChemicalState, second: ChemicalState) -> bool:
    """Whether two states' published widths are close enough to tie."""
    if min(first.fwhm[1], second.fwhm[1]) <= max(first.fwhm[0], second.fwhm[0]):
        return False
    a = 0.5 * (first.fwhm[0] + first.fwhm[1])
    b = 0.5 * (second.fwhm[0] + second.fwhm[1])
    return max(a, b) / max(min(a, b), 1e-9) <= WIDTH_LINK_RATIO


def count_model(
    spectrum: XPSSpectrum,
    region: str,
    n: int,
    database: Optional[XPSDatabase] = None,
    **options,
) -> tuple[XPSModel, list[str]]:
    """A model with exactly ``n`` components, chosen from the region's states.

    The states are ranked by how much intensity the spectrum actually has
    at each one — background removed — and the top ``n`` are kept. This is
    a starting point, not a decision: the states left out are returned so
    the choice can be argued with, and a state with real intensity that
    falls below the cut is exactly the thing to look at next.

    Returns
    -------
    tuple
        ``(model, notes)`` — the notes name what was kept and what was not.
    """
    database = database or load_xps_database()
    available = [s for s in database.states_for(region, include_satellites=True)
                 if not s.is_satellite]
    if not available:
        raise XPSError(f"la base de datos no tiene estados para {region!r}")
    if n < 1:
        raise XPSError("hacen falta al menos una componente")
    if n > len(available):
        raise XPSError(
            f"se piden {n} componentes y la base de datos solo describe "
            f"{len(available)} estados de {region}: "
            f"{', '.join(s.key for s in available)}. Para más componentes que "
            "estados conocidos, usa free_model y explica en el informe qué "
            "es cada una"
        )
    low = min(s.window[0] for s in available) - WINDOW_PAD
    high = max(s.window[1] for s in available) + WINDOW_PAD
    background = options.get("background", "shirley")
    scores = [
        (_height_guess(spectrum, (low, high), state.energy_ev, background), state)
        for state in available
    ]
    ranked = sorted(scores, key=lambda item: item[0], reverse=True)
    kept = sorted((state for _, state in ranked[:n]),
                  key=lambda state: state.energy_ev)
    dropped = [state for _, state in ranked[n:]]
    notes = [
        "componentes elegidas por la intensidad que hay en su ventana: "
        + ", ".join(f"{s.name} ({s.energy_ev:.1f} eV)" for s in kept)
    ]
    if dropped:
        notes.append(
            "fuera del modelo, y la base de datos las describe: "
            + ", ".join(f"{s.name} ({s.energy_ev:.1f} eV)" for s in dropped)
        )
    model = state_model(spectrum, region, [s.key for s in kept],
                        database=database, **options)
    return model, notes


def free_model(
    spectrum: XPSSpectrum,
    window: tuple[float, float],
    n: int,
    background: str = "shirley",
    profile: str = "gl",
    fwhm: float = 1.4,
    fix_mixing: bool = True,
    link_widths: bool = False,
    region: str = "",
) -> XPSModel:
    """``n`` components seeded from the data, with no literature attached.

    The seeds come from the second derivative of a lightly smoothed
    spectrum, which finds shoulders that the spectrum itself does not show
    as maxima — and shoulders are what an XPS region is mostly made of. If
    fewer than ``n`` shoulders are found, the rest are spread across the
    region, which is exactly as arbitrary as it sounds and is the reason
    this function exists separately from :func:`state_model`.
    """
    if n < 1:
        raise XPSError("hacen falta al menos una componente")
    window = (float(min(window)), float(max(window)))
    estimate = estimate_background(spectrum, window, background)
    energy, counts = spectrum.region_of(*window)
    signal = counts - estimate.values

    centres = _second_derivative_seeds(energy, signal, n)
    components = []
    for position, centre in enumerate(centres, start=1):
        height = max(float(np.interp(centre, energy, signal)), 1.0)
        extra: tuple[float, ...] = ()
        fixed: tuple[str, ...] = ()
        if profile in ("gl", "sgl"):
            extra = (DEFAULT_MIXING,)
            fixed = ("mixing",) if fix_mixing else ()
        elif profile == "ds_gauss":
            extra = (0.12, 0.6)
        elif profile == "ds":
            extra = (0.12,)
        components.append(
            XPSComponent(
                name=f"C{position}",
                label=f"componente {position}",
                centre=float(centre),
                height=height,
                fwhm=fwhm,
                profile=profile,
                centre_bounds=(window[0], window[1]),
                extra=extra,
                fixed=fixed,
                justification=(
                    "sin asignación: sembrada en un hombro de la segunda "
                    "derivada. Nombrarla es cosa del informe, no del ajuste"
                ),
            )
        )
    links = [Link(c.name, "fwhm", components[0].name, "fwhm")
             for c in components[1:]] if link_widths else []
    return XPSModel(components, window, background=background,
                    name=region or "libre", links=tuple(links),
                    region_label=region)


def _second_derivative_seeds(energy: np.ndarray, signal: np.ndarray,
                             n: int) -> list[float]:
    """Up to ``n`` component positions, from minima of the second derivative.

    A peak is a minimum of the second derivative whether or not it is a
    maximum of the spectrum, which is the whole point: in a C 1s the
    carbonyl component is never a maximum.
    """
    if energy.size < 7:
        return list(np.linspace(energy[0], energy[-1], n + 2)[1:-1])
    window = max(5, int(round(0.4 / max(np.median(np.diff(energy)), 1e-6))) | 1)
    kernel = np.ones(window) / window
    smooth = np.convolve(np.pad(signal, window // 2, mode="edge"), kernel, "valid")
    second = np.gradient(np.gradient(smooth, energy), energy)
    candidates = []
    for index in range(1, second.size - 1):
        if second[index] < second[index - 1] and second[index] <= second[index + 1] \
                and second[index] < 0:
            candidates.append((float(-second[index]), float(energy[index])))
    candidates.sort(reverse=True)
    chosen = sorted(position for _, position in candidates[:n])
    if len(chosen) < n:
        extra = np.linspace(energy[0], energy[-1], n - len(chosen) + 2)[1:-1]
        chosen = sorted(chosen + [float(value) for value in extra])
    return chosen


@dataclass
class CountComparison:
    """What each number of components bought."""

    n: int
    result: XPSFitResult
    reduced_chi2: float
    durbin_watson: Optional[float]
    n_parameters: int

    @property
    def significant(self) -> bool:
        """Whether every component in this fit stands above the noise."""
        return not any("no llegan a" in w for w in self.result.warnings)


def compare_counts(
    spectrum: XPSSpectrum,
    region: str,
    counts: Sequence[int] = (2, 3, 4),
    database: Optional[XPSDatabase] = None,
    **options,
) -> tuple[list[CountComparison], str]:
    """Fit several component counts and report what each one is worth.

    There is no statistic that decides how many chemical states a sample
    has. Adding a component always lowers the residual, and χ² falls even
    when the extra component is fitting noise. What *can* be said, and is
    said here, is:

    * whether the model is still leaving structure in the residual
      (Durbin–Watson well below 2, χ²_red well above 1) — that is evidence
      that a component is **missing**;
    * whether every component stands above the noise — if one does not,
      the model has too many;
    * how much χ² actually improved, so a 0.5 % gain for a whole extra
      chemical state can be seen for what it is.

    Returns
    -------
    tuple
        ``(comparisons, verdict)``. The verdict is a sentence in Spanish
        for the report, and it says what the evidence supports rather than
        naming a winner when the evidence does not.
    """
    database = database or load_xps_database()
    out: list[CountComparison] = []
    for n in counts:
        try:
            model, _ = count_model(spectrum, region, n, database=database, **options)
        except XPSError:
            continue
        result = fit_region(spectrum, model, database=database)
        out.append(CountComparison(n, result, result.reduced_chi2,
                                   result.durbin_watson, result.n_parameters))
    if not out:
        raise XPSError(
            f"no se ha podido ajustar ninguna de las cuentas pedidas en {region}"
        )

    good = [item for item in out
            if item.significant and item.reduced_chi2 < 2.0
            and (item.durbin_watson or 2.0) > 1.2]
    if not good:
        verdict = (
            "ninguno de los modelos probados describe la región dentro del "
            "ruido: o falta una componente que la base de datos no tiene, o "
            "el fondo o la forma de línea no son los que toca"
        )
    else:
        best = min(good, key=lambda item: item.n)
        richer = [item for item in good if item.n > best.n]
        gain = (
            100.0 * (1.0 - min(i.reduced_chi2 for i in richer) / best.reduced_chi2)
            if richer else 0.0
        )
        verdict = (
            f"con {best.n} componentes el residuo ya es ruido "
            f"(χ²_red = {best.reduced_chi2:.2f}, DW = "
            f"{best.durbin_watson:.2f}); añadir más baja el χ² un "
            f"{gain:.1f} % y eso no es prueba de otro estado químico"
            if richer else
            f"con {best.n} componentes el residuo ya es ruido "
            f"(χ²_red = {best.reduced_chi2:.2f})"
        )
    return out, verdict


__all__ = [
    "CountComparison",
    "DEFAULT_MIXING",
    "WIDTH_LINK_RATIO",
    "WINDOW_PAD",
    "compare_counts",
    "count_model",
    "doublet_for",
    "free_model",
    "state_model",
]
