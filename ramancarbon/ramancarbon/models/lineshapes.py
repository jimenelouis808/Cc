"""Peak profiles, and when each one is physically right.

Choosing a lineshape for a carbon band is a physical statement, not a
cosmetic one, and the choice changes the fitted areas by tens of percent:

``lorentzian``
    The profile of a damped harmonic oscillator: a phonon with a finite
    lifetime. This is the correct default for the D, D′, 2D and RBM bands
    and for the G band of a well-ordered sp² carbon.
``gaussian``
    What you get when many slightly different oscillators are summed — an
    inhomogeneous distribution. Justified for the D3 (~1500 cm⁻¹)
    amorphous-carbon band and the D4 (~1200 cm⁻¹) band, which are not
    single modes but envelopes over a distribution of local environments.
``pseudo_voigt``
    A linear blend of the two, parameterised by a mixing fraction η. Use it
    when the instrument's slit function is a significant part of the width
    (a low-resolution spectrometer broadens everything towards Gaussian)
    or when you want to let the data choose.
``bwf``
    Breit–Wigner–Fano: an asymmetric profile arising from a discrete phonon
    coupled to an electronic continuum. This is the *only* correct shape
    for the G⁻ feature of a **metallic** SWCNT, where the phonon couples to
    the free-electron continuum, and it is also used for the G band of
    nanocrystalline and heavily doped graphite. Fitting a metallic G⁻ with
    a Lorentzian pushes the fitted position several cm⁻¹ high and inflates
    the neighbouring D′.

Every profile is parameterised the same way — ``(centre, height, fwhm,
[shape])`` — so the fitter can treat them interchangeably. Note that
*height* here means the amplitude parameter; for the BWF profile the
maximum of the curve is not at the centre and is not equal to the
amplitude, which :func:`bwf_peak_position` and :func:`bwf_peak_height`
report honestly rather than papering over.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

#: 2*sqrt(2*ln 2) — converts a Gaussian standard deviation to its FWHM.
GAUSS_FWHM_FACTOR = 2.3548200450309493


def lorentzian(x: np.ndarray, centre: float, height: float, fwhm: float) -> np.ndarray:
    """Lorentzian profile, normalised so the value at ``centre`` is ``height``.

    ``L(x) = h / (1 + 4((x−c)/w)²)`` with ``w`` the FWHM.

    The analytic area is ``π h w / 2``.
    """
    w = max(float(fwhm), 1e-9)
    return float(height) / (1.0 + 4.0 * ((np.asarray(x, dtype=float) - float(centre)) / w) ** 2)


def gaussian(x: np.ndarray, centre: float, height: float, fwhm: float) -> np.ndarray:
    """Gaussian profile with peak value ``height`` at ``centre``.

    The analytic area is ``h w sqrt(π / (4 ln 2))`` ≈ ``1.0645 h w``.
    """
    w = max(float(fwhm), 1e-9)
    sigma = w / GAUSS_FWHM_FACTOR
    z = (np.asarray(x, dtype=float) - float(centre)) / sigma
    return float(height) * np.exp(-0.5 * z * z)


def pseudo_voigt(
    x: np.ndarray, centre: float, height: float, fwhm: float, eta: float = 0.5
) -> np.ndarray:
    """Linear pseudo-Voigt: ``η·Lorentzian + (1−η)·Gaussian``.

    Both components share the same FWHM, which is the standard (Thompson–
    Cox–Hastings-style) convention and makes ``fwhm`` the FWHM of the
    result to better than 1 % for any η.

    Parameters
    ----------
    eta:
        Lorentzian fraction, clipped to [0, 1]. η = 1 is a pure Lorentzian,
        η = 0 a pure Gaussian.
    """
    e = float(np.clip(eta, 0.0, 1.0))
    return e * lorentzian(x, centre, height, fwhm) + (1.0 - e) * gaussian(x, centre, height, fwhm)


def bwf(
    x: np.ndarray, centre: float, height: float, fwhm: float, q_inverse: float = -0.1
) -> np.ndarray:
    """Breit–Wigner–Fano profile.

    ``I(x) = h · (1 + (x−c)/(q Γ))² / (1 + ((x−c)/Γ)²)`` with ``Γ = w/2``.

    Parameterised by **1/q** rather than q, for two reasons: the
    uncoupled limit is ``1/q → 0``, which is an interior point of the
    parameter space and therefore fittable, whereas ``q → ∞`` is not; and
    the coupling strength is proportional to ``1/q``, so the fitted
    uncertainty on ``1/q`` is directly the uncertainty on the coupling.

    Sign convention: ``1/q < 0`` puts the tail on the **low-frequency**
    side, which is the direction observed for the G⁻ band of metallic
    SWCNTs and for nanocrystalline graphite. Reported values of *q* for
    metallic SWCNT bundles cluster around −5 to −20, i.e. ``1/q`` from
    −0.2 to −0.05.

    Parameters
    ----------
    q_inverse:
        The asymmetry, ``1/q``. Zero reduces exactly to a Lorentzian.

    Notes
    -----
    ``height`` is the amplitude parameter, not the maximum of the curve;
    see :func:`bwf_peak_height`. ``centre`` is the BWF ``ω_0`` parameter,
    not the position of the maximum; see :func:`bwf_peak_position`. Papers
    quoting "the BWF peak position" usually mean ω_0, but not always, so
    this package reports both.
    """
    gamma = max(float(fwhm), 1e-9) / 2.0
    reduced = (np.asarray(x, dtype=float) - float(centre)) / gamma
    return float(height) * (1.0 + reduced * float(q_inverse)) ** 2 / (1.0 + reduced**2)


def bwf_peak_position(centre: float, fwhm: float, q_inverse: float) -> float:
    """Where a BWF profile actually peaks, in cm⁻¹.

    With ``u = (x − ω₀)/Γ`` and ``b = 1/q`` the derivative factors exactly::

        dI/du ∝ (1 + b u)(b − u)

    so the profile has a zero at ``u = −1/b`` and its **maximum at ``u = b``**.
    The peak therefore sits at ``ω₀ + Γ/q``, displaced towards the tail
    side — to *lower* wavenumber for the negative ``1/q`` of metallic
    SWCNTs.

    For a G⁻ band with Γ = 20 cm⁻¹ and 1/q = −0.1 that is a 2 cm⁻¹
    displacement, the same order as the doping-induced shifts this package
    is asked to detect, which is why it must not be ignored.

    Parameters
    ----------
    centre:
        The BWF ``ω₀`` parameter.
    fwhm:
        ``2Γ``, the width parameter.
    q_inverse:
        The asymmetry ``1/q``.

    Returns
    -------
    float
        Position of the maximum in cm⁻¹.
    """
    gamma = max(float(fwhm), 1e-9) / 2.0
    return float(centre) + gamma * float(q_inverse)


def bwf_peak_height(height: float, q_inverse: float) -> float:
    """The maximum value of a BWF profile with amplitude ``height``.

    Substituting ``u = b`` into ``(1 + b u)² / (1 + u²)`` gives exactly
    ``1 + b²``, so the maximum is ``h (1 + q⁻²)`` — always ``>= h``. Using
    the amplitude parameter as "the peak intensity" when computing I_D/I_G
    with a BWF G band underestimates I_G, and by more the stronger the
    coupling.
    """
    b = float(q_inverse)
    return float(height) * (1.0 + b * b)


# ----------------------------------------------------------------------
# analytic areas
# ----------------------------------------------------------------------
def lorentzian_area(height: float, fwhm: float) -> float:
    """Exact integral of :func:`lorentzian` over the real line."""
    return float(np.pi * height * fwhm / 2.0)


def gaussian_area(height: float, fwhm: float) -> float:
    """Exact integral of :func:`gaussian` over the real line."""
    sigma = fwhm / GAUSS_FWHM_FACTOR
    return float(height * sigma * np.sqrt(2.0 * np.pi))


def pseudo_voigt_area(height: float, fwhm: float, eta: float) -> float:
    """Exact integral of :func:`pseudo_voigt`."""
    e = float(np.clip(eta, 0.0, 1.0))
    return e * lorentzian_area(height, fwhm) + (1.0 - e) * gaussian_area(height, fwhm)


def bwf_area(height: float, fwhm: float, q_inverse: float) -> float:
    """Integral of a BWF profile over a finite window.

    The BWF profile does **not** have a convergent integral over the real
    line: its tails go to the constant ``h/q²`` rather than to zero, so
    ``∫ I dx`` diverges linearly. Any "BWF area" in the literature is
    therefore a *windowed* area, and quoting one without its window is
    meaningless.

    This function returns the integral of the convergent part only — the
    profile minus its asymptote ``h/q²`` — which is the quantity that
    behaves like an area and reduces to the Lorentzian area as ``1/q → 0``:

    ``A = (π h Γ) (1 − 1/q²) + 2 h Γ / q · [arctan]``… evaluated in closed
    form as ``π h Γ (1 − q⁻²)``.

    Use :func:`window_area` when you need the number an integration over a
    stated window would give.
    """
    gamma = max(float(fwhm), 1e-9) / 2.0
    b = float(q_inverse)
    return float(np.pi * height * gamma * (1.0 - b * b))


def window_area(
    profile: Callable[..., np.ndarray],
    low: float,
    high: float,
    *args,
    points: int = 4001,
    **kwargs,
) -> float:
    """Numerically integrate any profile over ``[low, high]``.

    The escape hatch for BWF and for reporting areas over the same finite
    window the experiment actually covered.
    """
    if high <= low:
        raise ValueError("empty integration window")
    grid = np.linspace(float(low), float(high), int(points))
    return float(np.trapezoid(profile(grid, *args, **kwargs), grid))


#: Registry consumed by :mod:`ramancarbon.models.fitting`. Each entry maps a
#: name to (callable, extra parameter names, default extra values, bounds).
PROFILES: dict[str, dict] = {
    "lorentzian": {
        "function": lorentzian,
        "extra": (),
        "defaults": (),
        "bounds": (),
        "area": lambda h, w, extra: lorentzian_area(h, w),
        "label": "Lorentziana",
    },
    "gaussian": {
        "function": gaussian,
        "extra": (),
        "defaults": (),
        "bounds": (),
        "area": lambda h, w, extra: gaussian_area(h, w),
        "label": "Gaussiana",
    },
    "pseudo_voigt": {
        "function": pseudo_voigt,
        "extra": ("eta",),
        "defaults": (0.7,),
        "bounds": ((0.0, 1.0),),
        "area": lambda h, w, extra: pseudo_voigt_area(h, w, extra[0]),
        "label": "Pseudo-Voigt",
    },
    "bwf": {
        "function": bwf,
        "extra": ("q_inverse",),
        "defaults": (-0.1,),
        # |1/q| is kept below 0.5. Beyond that the profile stops looking
        # like a peak at all and the fit runs away into the background.
        "bounds": ((-0.5, 0.5),),
        "area": lambda h, w, extra: bwf_area(h, w, extra[0]),
        "label": "Breit-Wigner-Fano",
    },
}

#: Human-facing aliases accepted from the CLI and the GUI.
PROFILE_ALIASES = {
    "l": "lorentzian",
    "lor": "lorentzian",
    "lorentz": "lorentzian",
    "g": "gaussian",
    "gauss": "gaussian",
    "pv": "pseudo_voigt",
    "voigt": "pseudo_voigt",
    "pseudovoigt": "pseudo_voigt",
    "fano": "bwf",
    "breit-wigner-fano": "bwf",
}


def resolve_profile(name: str) -> str:
    """Normalise a profile name, accepting the common aliases."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = PROFILE_ALIASES.get(key.replace("_", ""), PROFILE_ALIASES.get(key, key))
    if key not in PROFILES:
        raise ValueError(
            f"unknown lineshape {name!r}; available: {', '.join(sorted(PROFILES))}"
        )
    return key


__all__ = [
    "GAUSS_FWHM_FACTOR",
    "PROFILES",
    "bwf",
    "bwf_area",
    "bwf_peak_height",
    "bwf_peak_position",
    "gaussian",
    "gaussian_area",
    "lorentzian",
    "lorentzian_area",
    "pseudo_voigt",
    "pseudo_voigt_area",
    "resolve_profile",
    "window_area",
]
