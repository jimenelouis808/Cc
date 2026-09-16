"""Peak profiles for photoelectron spectra, and when each one is right.

A core-level line is not a Lorentzian. It is a Lorentzian (the core-hole
lifetime) convolved with a Gaussian (the analyser, the photon line width and
the sample's own inhomogeneity), and for a **metal** it also carries an
asymmetric tail to high binding energy from the electron–hole pairs the
photoemission event excites in the conduction band.

That last point is not a refinement, it is the difference between a right
answer and a wrong one. Fit metallic iron with symmetric peaks and the tail
has to be covered by something; the fitter covers it with an extra
component, and the extra component gets reported as an oxide that is not
there. Any spectrum with metal in it needs an asymmetric shape for the
metallic component and symmetric ones for the oxidised components — mixed,
in the same fit.

The profiles:

``gl``
    Gaussian–Lorentzian **product**, CasaXPS's GL(p). The default for
    everything non-metallic. ``mixing`` is CasaXPS's p/100: 0 is pure
    Gaussian, 1 pure Lorentzian, and 0.3 is the usual starting point for a
    polymer or an oxide on a modern monochromated instrument.
``sgl``
    The **sum** form, SGL(p) — the pseudo-Voigt. Product and sum agree at
    the two extremes and differ by a few per cent in the wings in between;
    CasaXPS defaults to the product, so a fit meant to be compared with one
    made there should use ``gl``.
``ds``
    Doniach–Šunjić, for metals. ``asymmetry`` is the singularity index α:
    0 reduces exactly to a Lorentzian, and 0.05–0.20 covers the transition
    metals. Its integral **diverges** — see :func:`ds` — so its area is
    only ever a windowed area.
``ds_gauss``
    Doniach–Šunjić convolved with a Gaussian, which is what a real
    instrument measures. Use this rather than ``ds`` when the metallic
    component's width matters, because a pure DS fitted to a real spectrum
    absorbs the whole instrument function into α and γ.

This registry is deliberately **separate** from
:data:`ramancarbon.models.lineshapes.PROFILES`. Merging them would let the
Raman fitter offer Doniach–Šunjić for a D band, where it means nothing, and
let the XPS fitter offer Breit–Wigner–Fano, where it means something else.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.compat import trapezoid
from ..models.lineshapes import GAUSS_FWHM_FACTOR, gaussian, lorentzian, pseudo_voigt

#: Widest asymmetry the fitter will consider. Above roughly 0.3 the profile
#: stops having a recognisable maximum and the fit walks into the background.
MAX_ASYMMETRY = 0.35


def gl(x: np.ndarray, centre: float, height: float, fwhm: float,
       mixing: float = 0.3) -> np.ndarray:
    """Gaussian–Lorentzian product profile, CasaXPS's ``GL(p)``.

    ``GL(u) = h · exp(−4 ln2 (1−m) u²) / (1 + 4 m u²)`` with ``u = (x−c)/w``.

    At ``m = 0`` this is exactly a Gaussian of FWHM ``w`` and at ``m = 1``
    exactly a Lorentzian of FWHM ``w``. In between the true FWHM is up to
    5 % narrower than ``w`` (the minimum is at m ≈ 0.5), which is why
    :func:`profile_fwhm` exists and why a fitted width has to be quoted
    with the lineshape that produced it.
    """
    w = max(float(fwhm), 1e-9)
    m = float(np.clip(mixing, 0.0, 1.0))
    u = (np.asarray(x, dtype=float) - float(centre)) / w
    return float(height) * np.exp(-4.0 * np.log(2.0) * (1.0 - m) * u * u) / (
        1.0 + 4.0 * m * u * u
    )


def sgl(x: np.ndarray, centre: float, height: float, fwhm: float,
        mixing: float = 0.3) -> np.ndarray:
    """Gaussian–Lorentzian **sum**, ``SGL(p)`` — the pseudo-Voigt."""
    return pseudo_voigt(x, centre, height, fwhm, mixing)


def ds(x: np.ndarray, centre: float, height: float, fwhm: float,
       asymmetry: float = 0.1) -> np.ndarray:
    """Doniach–Šunjić profile for a metallic core level.

    ``DS(u) = h · cos[πα/2 + (1−α) arctan u] / (1 + u²)^((1−α)/2)``
    with ``u = (x − c)/γ`` and ``γ = w/2``.

    Sign convention: positive ``u`` is *higher* binding energy, which is
    where the tail goes. That is the direction the electron–hole
    excitations push intensity, because they take energy away from the
    photoelectron.

    Notes
    -----
    Two honest inconveniences, both consequences of the physics rather than
    of the parameterisation:

    * The integral **diverges**. The tail falls off as ``u^(α−1)``, so the
      area over the whole line is infinite for any α > 0 and any "DS area"
      is a windowed area. Quote the window with it, the same way this
      package insists for Breit–Wigner–Fano.
    * ``height`` is an amplitude, not the maximum, and ``centre`` is not
      where the maximum is: the asymmetry displaces the peak towards the
      tail. :func:`ds_peak` reports both honestly.
    """
    gamma = max(float(fwhm), 1e-9) / 2.0
    alpha = float(np.clip(asymmetry, 0.0, MAX_ASYMMETRY))
    u = (np.asarray(x, dtype=float) - float(centre)) / gamma
    numerator = np.cos(0.5 * np.pi * alpha + (1.0 - alpha) * np.arctan(u))
    return float(height) * numerator / (1.0 + u * u) ** (0.5 * (1.0 - alpha))


def convolve_gaussian(x: np.ndarray, y: np.ndarray, fwhm: float) -> np.ndarray:
    """Convolve a curve on a **uniform** grid with a Gaussian.

    The grid must be uniform: a convolution on an uneven axis is a
    different operation at every point, and no XPS region is measured on
    one. Edges are padded with their own end values, so a profile that has
    not decayed to zero at the window edge is not given an artificial step
    to ring against.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return y
    steps = np.diff(x)
    step = float(np.median(steps))
    if step <= 0 or np.max(np.abs(steps - step)) > 0.05 * abs(step):
        raise ValueError(
            "la convolución gaussiana necesita una rejilla uniforme; esta "
            f"varía del paso mediano ({step:.4g}) en más de un 5 %"
        )
    sigma = max(float(fwhm), 1e-9) / GAUSS_FWHM_FACTOR
    half = max(int(np.ceil(4.0 * sigma / step)), 1)
    offsets = np.arange(-half, half + 1) * step
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    total = kernel.sum()
    if total <= 0:                              # pragma: no cover - sigma > 0
        return y
    kernel = kernel / total
    padded = np.concatenate([np.full(half, y[0]), y, np.full(half, y[-1])])
    return np.convolve(padded, kernel, mode="valid")


def ds_gauss(x: np.ndarray, centre: float, height: float, fwhm: float,
             asymmetry: float = 0.1, gaussian_fwhm: float = 0.5) -> np.ndarray:
    """Doniach–Šunjić broadened by a Gaussian — what an instrument measures.

    ``fwhm`` is the Lorentzian part (the core-hole lifetime) and
    ``gaussian_fwhm`` the instrumental and inhomogeneous part. Separating
    them is worth the extra parameter: the Lorentzian width is a property
    of the element and should come out roughly the same in everybody's
    spectra, while the Gaussian width is a property of *your* analyser
    setting and should not.
    """
    return convolve_gaussian(x, ds(x, centre, height, fwhm, asymmetry), gaussian_fwhm)


def ds_peak(centre: float, fwhm: float, asymmetry: float,
            height: float = 1.0) -> tuple[float, float]:
    """Where a Doniach–Šunjić profile actually peaks, and how high.

    Found numerically on a fine grid around the centre and then refined by
    a parabola through the three best points — there is no closed form, and
    the displacement is not negligible: at α = 0.2 and a 1 eV width the
    maximum sits about 0.1 eV below the ``centre`` parameter, which is the
    same order as the chemical shifts being measured.

    Returns
    -------
    tuple
        ``(position in eV, value there)``.
    """
    gamma = max(float(fwhm), 1e-9) / 2.0
    grid = centre + np.linspace(-3.0 * gamma, 3.0 * gamma, 1201)
    values = ds(grid, centre, height, fwhm, asymmetry)
    top = int(np.argmax(values))
    if 0 < top < grid.size - 1:
        y0, y1, y2 = values[top - 1], values[top], values[top + 1]
        denominator = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denominator if denominator != 0 else 0.0
        step = grid[1] - grid[0]
        return float(grid[top] + offset * step), float(y1 - 0.25 * (y0 - y2) * offset)
    return float(grid[top]), float(values[top])


def profile_fwhm(name: str, fwhm: float, extra: tuple[float, ...] = ()) -> float:
    """The true full width at half maximum of a profile, in eV.

    For ``gl`` the width parameter is not the width: the product of a
    Gaussian and a Lorentzian of equal FWHM is narrower than either. The
    gap is 5 % at mixing 0.5 and zero at both extremes — small, but it is a
    systematic 5 % on every width reported from a GL fit, and it goes the
    same way for everybody's spectra, so it survives averaging.
    """
    spec = XPS_PROFILES[name]
    function = spec["function"]
    width = max(float(fwhm), 1e-9)
    grid = np.linspace(-8.0 * width, 8.0 * width, 4001)
    values = function(grid, 0.0, 1.0, width, *extra)
    peak = float(values.max())
    above = grid[values >= 0.5 * peak]
    if above.size < 2:                          # pragma: no cover - degenerate
        return width
    return float(above[-1] - above[0])


def fwhm_for_total(name: str, total: float,
                   extra: tuple[float, ...] = ()) -> float:
    """The width PARAMETER that gives a profile a given true width.

    The inverse of :func:`profile_fwhm`, and it has to exist because the
    two numbers are not the same and the published one is the total: a
    GL product is 4 % narrower than its parameter at mixing 0.3, and a
    Doniach-Šunjić is 13 % wider. Anything that lets a user type a width
    has to convert, or they are setting a different quantity from the one
    they read off the table — typing the displayed number back would
    change the peak.

    Solved numerically because ``ds_gauss`` is not scale-invariant: it
    convolves with a Gaussian of fixed width, so its true width is not
    proportional to its parameter and the one-point rescaling that works
    for every other profile is wrong for it by up to a third.

    Accurate to about half a per cent, which is the resolution of the
    forward function rather than of this one: :func:`profile_fwhm`
    measures the half-maximum crossings on a 4001-point grid, so it is
    quantised at roughly 0.4 % and no inverse of it can be tighter. On a
    1.4 eV component that is six thousandths of an electronvolt.
    """
    target = max(float(total), 1e-9)
    spec = XPS_PROFILES[name]
    extra = tuple(extra) if extra else tuple(spec["defaults"])

    guess = target * target / max(profile_fwhm(name, target, extra), 1e-9)
    if abs(profile_fwhm(name, guess, extra) - target) <= 1e-6 * target:
        return float(guess)

    low, high = 1e-4 * target, 10.0 * target
    for _ in range(60):
        middle = 0.5 * (low + high)
        if profile_fwhm(name, middle, extra) < target:
            low = middle
        else:
            high = middle
    return float(0.5 * (low + high))


def window_area(name: str, height: float, fwhm: float,
                extra: tuple[float, ...] = (),
                window: Optional[tuple[float, float]] = None,
                centre: float = 0.0, points: int = 4001) -> float:
    """Area of a profile, integrated over a window.

    Every XPS area is a windowed area whether anybody says so or not: the
    background under a region is defined by the region's endpoints, so the
    intensity assigned to a component depends on where the region was cut.
    This function makes the window explicit. With ``window`` ``None`` it
    integrates over ±10 widths, which is effectively the whole line for the
    symmetric profiles and is **not** for the Doniach–Šunjić ones, whose
    integral diverges — for those, pass the fit window.
    """
    spec = XPS_PROFILES[name]
    width = max(float(fwhm), 1e-9)
    if window is None:
        low, high = centre - 10.0 * width, centre + 10.0 * width
    else:
        low, high = float(min(window)), float(max(window))
    grid = np.linspace(low, high, int(points))
    return float(trapezoid(spec["function"](grid, centre, height, width, *extra), grid))


#: The XPS profile registry, shaped like the Raman one but separate from it.
XPS_PROFILES: dict[str, dict] = {
    "gl": {
        "function": gl,
        "extra": ("mixing",),
        "defaults": (0.3,),
        "bounds": ((0.0, 1.0),),
        "label": "Gaussiana-Lorentziana (producto, GL)",
        "asymmetric": False,
        "diverges": False,
    },
    "sgl": {
        "function": sgl,
        "extra": ("mixing",),
        "defaults": (0.3,),
        "bounds": ((0.0, 1.0),),
        "label": "Gaussiana-Lorentziana (suma, SGL)",
        "asymmetric": False,
        "diverges": False,
    },
    "gaussian": {
        "function": gaussian,
        "extra": (),
        "defaults": (),
        "bounds": (),
        "label": "Gaussiana",
        "asymmetric": False,
        "diverges": False,
    },
    "lorentzian": {
        "function": lorentzian,
        "extra": (),
        "defaults": (),
        "bounds": (),
        "label": "Lorentziana",
        "asymmetric": False,
        "diverges": False,
    },
    "ds": {
        "function": ds,
        "extra": ("asymmetry",),
        "defaults": (0.1,),
        "bounds": ((0.0, MAX_ASYMMETRY),),
        "label": "Doniach-Šunjić (metal)",
        "asymmetric": True,
        "diverges": True,
    },
    "ds_gauss": {
        "function": ds_gauss,
        "extra": ("asymmetry", "gaussian_fwhm"),
        "defaults": (0.1, 0.5),
        "bounds": ((0.0, MAX_ASYMMETRY), (0.05, 4.0)),
        "label": "Doniach-Šunjić con ensanchamiento gaussiano",
        "asymmetric": True,
        "diverges": True,
    },
}

#: Names accepted from the CLI and the GUI.
XPS_PROFILE_ALIASES = {
    "glp": "gl", "gl30": "gl", "producto": "gl", "product": "gl",
    "pseudovoigt": "sgl", "pv": "sgl", "suma": "sgl", "sum": "sgl",
    "doniach": "ds", "doniachsunjic": "ds", "asimetrica": "ds",
    "dsg": "ds_gauss", "doniachgauss": "ds_gauss",
    "g": "gaussian", "l": "lorentzian",
}


def resolve_xps_profile(name: str) -> str:
    """Normalise an XPS profile name, accepting the common aliases."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = XPS_PROFILE_ALIASES.get(key.replace("_", ""), key)
    if key not in XPS_PROFILES:
        raise ValueError(
            f"perfil XPS desconocido {name!r}; hay: {', '.join(XPS_PROFILES)}"
        )
    return key


__all__ = [
    "MAX_ASYMMETRY",
    "XPS_PROFILES",
    "XPS_PROFILE_ALIASES",
    "convolve_gaussian",
    "ds",
    "ds_gauss",
    "ds_peak",
    "fwhm_for_total",
    "gl",
    "profile_fwhm",
    "resolve_xps_profile",
    "sgl",
    "window_area",
]
