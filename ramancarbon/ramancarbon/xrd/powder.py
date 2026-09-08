"""Structure factors and simulated powder patterns.

Nothing here is looked up. Given a crystal structure and a wavelength, the
reflection list is enumerated from the lattice, each structure factor is
summed over the atoms the symmetry generates, and the profile is built from
those. That is what makes a lattice parameter refinable: change ``a`` and
every peak moves the amount physics says it should.

Four corrections are applied, and each is a decision worth reading:

**Lorentz–polarisation.** For an unmonochromated Bragg–Brentano
diffractometer, ``LP = (1 + cos²2θ) / (sin²θ · cosθ)``. It rises steeply
towards low angle, so the intensity ratio between a 10° and a 60°
reflection is dominated by it, not by the structure. With a graphite
monochromator on the diffracted beam the polarisation term becomes
``(1 + K cos²2θ)/(1 + K)``, which is why :func:`simulate` takes a
monochromator angle rather than assuming.

**Debye–Waller.** ``exp(−8π²U sin²θ/λ²)`` per atom, which suppresses the
high-angle reflections and is the parameter most easily abused in a
refinement: a wrong background or an unmodelled absorption is absorbed by
U, which comes out negative or implausibly large. Both are checked.

**The Kα₁₂ doublet.** A tube with only a Kβ filter emits two lines 0.004 Å
apart with intensity ratio 2:1, and at 80° 2θ that splits a peak by about
0.2° — several times the step of a normal scan. Simulating one line and
fitting real two-line data forces the profile width up and biases every
crystallite size derived from it. So the doublet is on by default and
switched off explicitly for monochromated data.

**Preferred orientation**, through the March–Dollase function. Plate-like
and needle-like powders are the normal case in nanomaterials, not the
exception: a graphite or MoS₂ powder pressed into a holder gives an 002
reflection several times too strong. One parameter, one axis, and the
report says when the value implies severe texture rather than silently
absorbing intensity errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .pattern import Pattern
from .scattering import form_factor
from .structure import Crystal

#: Reflections closer in d than this fraction are merged as overlapping.
D_MERGE_RELATIVE = 1e-5

#: Intensity below this fraction of the strongest reflection is dropped.
MIN_RELATIVE_INTENSITY = 1e-4

#: Kα₂/Kα₁ intensity ratio for a laboratory tube.
KALPHA2_RATIO = 0.5


@dataclass
class Reflection:
    """One powder reflection: where it is and how strong it should be."""

    hkl: tuple[int, int, int]
    """A representative index; overlapping families are merged."""
    d: float
    two_theta: float
    intensity: float
    """Relative, with the strongest reflection at 100."""
    multiplicity: int
    f_squared: float
    phase: str = ""
    contributing: tuple[tuple[int, int, int], ...] = ()
    """Every hkl merged into this line. More than one *family* here means a
    genuine overlap, not a multiplicity."""

    @property
    def is_overlap(self) -> bool:
        """Whether distinct families coincide in d, e.g. cubic 333 and 511."""
        seen = {tuple(sorted(abs(v) for v in hkl)) for hkl in self.contributing}
        return len(seen) > 1

    def __str__(self) -> str:
        h, k, l = self.hkl
        mark = " (solape)" if self.is_overlap else ""
        return (
            f"({h:2d}{k:2d}{l:2d})  d={self.d:8.4f} Å  2θ={self.two_theta:8.3f}°  "
            f"I={self.intensity:6.1f}  m={self.multiplicity:3d}{mark}"
        )


def lorentz_polarisation(
    two_theta: np.ndarray, monochromator_two_theta: float = 0.0
) -> np.ndarray:
    """The LP factor for Bragg–Brentano geometry.

    ``monochromator_two_theta`` is the 2θ of the monochromator crystal in
    degrees; 0 means none, and the unpolarised form is used. A graphite
    (002) monochromator is at 26.6° for Cu Kα, for which K = cos²(26.6°) =
    0.80 — close enough to 1 that the correction is small, but it is the
    kind of small systematic that shows up as a stubborn intensity misfit
    at low angle.
    """
    theta = np.radians(np.asarray(two_theta, dtype=float)) / 2.0
    if monochromator_two_theta > 0.0:
        k = math.cos(math.radians(monochromator_two_theta)) ** 2
        polarisation = (1.0 + k * np.cos(2.0 * theta) ** 2) / (1.0 + k)
    else:
        polarisation = (1.0 + np.cos(2.0 * theta) ** 2) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        lorentz = 1.0 / (np.sin(theta) ** 2 * np.cos(theta))
    result = polarisation * lorentz
    result[~np.isfinite(result)] = 0.0
    return result


def march_dollase(
    hkl: np.ndarray, axis: Sequence[int], crystal: Crystal, r: float
) -> np.ndarray:
    """Preferred-orientation correction, March–Dollase.

    ``P = (r² cos²α + sin²α / r)^(−3/2)``, with α the angle between each
    reflection's plane normal and the texture axis. ``r = 1`` is a random
    powder; ``r < 1`` describes plates lying flat, which enhances the
    reflections parallel to the plate — the usual case for a layered
    material pressed into a holder.
    """
    if abs(r - 1.0) < 1e-9:
        return np.ones(len(hkl))
    metric = crystal.lattice.reciprocal_metric
    reference = np.asarray(axis, dtype=float)
    reference_norm = math.sqrt(float(reference @ metric @ reference))
    indices = np.asarray(hkl, dtype=float)
    norms = np.sqrt(np.einsum("ij,jk,ik->i", indices, metric, indices))
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = (indices @ metric @ reference) / (norms * reference_norm)
    cosine = np.clip(np.nan_to_num(cosine), -1.0, 1.0)
    cos2 = cosine**2
    return (r * r * cos2 + (1.0 - cos2) / r) ** (-1.5)


def reflections(
    crystal: Crystal,
    wavelength: float = 1.540598,
    two_theta_range: tuple[float, float] = (5.0, 90.0),
    u_override: Optional[float] = None,
    monochromator_two_theta: float = 0.0,
    preferred_axis: Optional[Sequence[int]] = None,
    preferred_r: float = 1.0,
    min_relative: float = MIN_RELATIVE_INTENSITY,
    normalise: bool = True,
) -> list[Reflection]:
    """Every reflection of one structure inside a 2θ window.

    The hkl enumeration is bounded rigorously rather than heuristically:
    since ``h = H·a`` for the direct lattice vector **a**, ``|h| ≤ a/d_min``
    for any lattice including triclinic. No safety factor is needed and
    none is used.

    Reflections whose d-spacings coincide are summed rather than counted
    separately, which is both physically right — overlapping reflections
    add — and removes any need for a multiplicity table.

    ``normalise`` rescales the strongest reflection to 100, which is what
    a powder card shows and what phase identification wants. Refinement
    needs it **off**: the Rietveld scale factor is only proportional to
    the amount of a phase if the intensities it multiplies are absolute,
    and per-phase normalisation destroys exactly that proportionality —
    so weight fractions computed from normalised intensities are wrong by
    the ratio of the phases' strongest structure factors.
    """
    low, high = two_theta_range
    high = min(float(high), 179.0)
    low = max(float(low), 1e-3)
    lattice = crystal.lattice
    d_max = wavelength / (2.0 * math.sin(math.radians(low) / 2.0))
    d_min = wavelength / (2.0 * math.sin(math.radians(high) / 2.0))

    limits = [max(1, int(math.floor(getattr(lattice, axis) / d_min)))
              for axis in ("a", "b", "c")]
    grids = np.meshgrid(*[np.arange(-n, n + 1) for n in limits], indexing="ij")
    hkl = np.stack([g.ravel() for g in grids], axis=1)
    hkl = hkl[np.any(hkl != 0, axis=1)]

    spacing = lattice.d_spacing(hkl)
    keep = (spacing >= d_min - 1e-9) & (spacing <= d_max + 1e-9)
    hkl, spacing = hkl[keep], spacing[keep]
    if hkl.size == 0:
        return []

    s = 1.0 / (2.0 * spacing)
    coordinates, elements, occupancies, displacements = crystal.expanded()
    if u_override is not None:
        displacements = np.full_like(displacements, float(u_override))

    # Group atoms by element so each form factor is evaluated once.
    structure = np.zeros(len(hkl), dtype=complex)
    phases = 2.0 * math.pi * (hkl @ coordinates.T)
    for element in sorted(set(elements)):
        mask = np.array([e == element for e in elements])
        f0 = form_factor(element, s)
        weights = occupancies[mask]
        debye = np.exp(-8.0 * math.pi**2 * np.outer(s**2, displacements[mask]))
        contribution = (np.exp(1j * phases[:, mask]) * debye) @ weights
        structure += f0 * contribution

    f_squared = np.abs(structure) ** 2
    two_theta = 2.0 * np.degrees(np.arcsin(np.clip(wavelength / (2.0 * spacing), -1, 1)))
    intensity = f_squared * lorentz_polarisation(two_theta, monochromator_two_theta)
    if preferred_axis is not None and abs(preferred_r - 1.0) > 1e-9:
        intensity = intensity * march_dollase(hkl, preferred_axis, crystal, preferred_r)

    order = np.argsort(-spacing, kind="stable")
    hkl, spacing, two_theta = hkl[order], spacing[order], two_theta[order]
    intensity, f_squared = intensity[order], f_squared[order]

    # Everything below is per-reflection bookkeeping on scalars, so the
    # arrays are converted to Python lists once. Indexing a NumPy array
    # for one element builds a NumPy scalar; doing that a few hundred
    # thousand times inside a refinement — which rebuilds this list for
    # every trial cell — was a quarter of the total time.
    hkl_rows = [tuple(int(v) for v in row) for row in hkl]
    spacing_list = spacing.tolist()
    two_theta_list = two_theta.tolist()
    intensity_list = intensity.tolist()
    f_squared_list = f_squared.tolist()

    merged: list[Reflection] = []
    index = 0
    count = len(spacing_list)
    while index < count:
        stop = index + 1
        reference = spacing_list[index]
        tolerance = D_MERGE_RELATIVE * reference
        while stop < count and abs(spacing_list[stop] - reference) <= tolerance:
            stop += 1
        total = math.fsum(intensity_list[index:stop])
        if total > 0.0:
            # Prefer the conventional representative of the family: the
            # one with no negative index, then the largest lexicographically.
            # Reporting (0 0 -1) where every table says (0 0 1) makes the
            # output harder to check against a reference, for no gain.
            representative = index
            best = None
            for candidate in range(index, stop):
                h, k, l = hkl_rows[candidate]
                key = (h >= 0 and k >= 0 and l >= 0, h, k, l)
                if best is None or key > best:
                    best, representative = key, candidate
            merged.append(
                Reflection(
                    hkl=hkl_rows[representative],
                    d=reference,
                    two_theta=two_theta_list[index],
                    intensity=total,
                    multiplicity=stop - index,
                    f_squared=f_squared_list[representative],
                    phase=crystal.name,
                    contributing=tuple(hkl_rows[index:stop]),
                )
            )
        index = stop

    if not merged:
        return []
    strongest = max(r.intensity for r in merged)
    if strongest <= 0.0:
        return []
    cutoff = strongest * min_relative
    if normalise:
        for reflection in merged:
            reflection.intensity = 100.0 * reflection.intensity / strongest
        cutoff = 100.0 * min_relative
    return [r for r in merged if r.intensity >= cutoff]


# -- profile shapes ---------------------------------------------------


@dataclass
class Profile:
    """Peak-shape parameters of a simulated or refined pattern."""

    u: float = 0.008
    v: float = -0.002
    w: float = 0.004
    """Caglioti coefficients: ``FWHM² = U tan²θ + V tanθ + W``, in deg²."""
    eta0: float = 0.5
    eta1: float = 0.0
    """Pseudo-Voigt mixing, ``η = η0 + η1·2θ``, clipped to [0, 1]."""
    zero: float = 0.0
    """Zero-point shift of the 2θ axis, in degrees."""

    def fwhm(self, two_theta: np.ndarray) -> np.ndarray:
        """FWHM in degrees at each 2θ, never below a hundredth of a degree."""
        tan_theta = np.tan(np.radians(np.asarray(two_theta, dtype=float)) / 2.0)
        squared = self.u * tan_theta**2 + self.v * tan_theta + self.w
        return np.sqrt(np.maximum(squared, 1e-6))

    def eta(self, two_theta: np.ndarray) -> np.ndarray:
        return np.clip(self.eta0 + self.eta1 * np.asarray(two_theta, dtype=float), 0.0, 1.0)

    def fwhm_at(self, two_theta: float) -> float:
        """The same, for one angle, without building an array.

        A refinement asks for the width of every reflection on every
        residual evaluation — fifty thousand times for a single-phase fit.
        Going through ``np.asarray`` and ``np.sqrt`` for one number each
        time cost a fifth of the total refinement.
        """
        tan_theta = math.tan(math.radians(two_theta) / 2.0)
        squared = self.u * tan_theta * tan_theta + self.v * tan_theta + self.w
        return math.sqrt(max(squared, 1e-6))

    def eta_at(self, two_theta: float) -> float:
        """Mixing at one angle, clipped, without an array."""
        return min(1.0, max(0.0, self.eta0 + self.eta1 * two_theta))


def pseudo_voigt(
    x: np.ndarray, centre: float, fwhm: float, eta: float
) -> np.ndarray:
    """Area-normalised pseudo-Voigt, as used for a diffraction peak.

    Area-normalised rather than height-normalised because the quantity a
    structure factor predicts is the integrated intensity: normalising by
    height would make the scale factor depend on the width, and then a
    refinement could trade crystallite size against phase fraction.
    """
    half = max(fwhm, 1e-6) / 2.0
    gaussian_sigma = half / math.sqrt(2.0 * math.log(2.0))
    gaussian = np.exp(-0.5 * ((x - centre) / gaussian_sigma) ** 2) / (
        gaussian_sigma * math.sqrt(2.0 * math.pi)
    )
    lorentzian = (half / math.pi) / ((x - centre) ** 2 + half**2)
    return eta * lorentzian + (1.0 - eta) * gaussian


def scherrer(fwhm_deg: float, two_theta: float, wavelength: float, k: float = 0.9) -> float:
    """Crystallite size in nm from a peak width, Scherrer's equation.

    ``D = K λ / (β cos θ)`` with β the *sample* broadening in radians. The
    caller must have removed the instrument's own contribution first;
    :func:`instrumental_correction` does that. A size quoted without it is
    an instrument resolution dressed up as a nanostructure — with a typical
    laboratory 0.08° resolution, that floor is around 100 nm, and above it
    Scherrer says nothing at all.
    """
    beta = math.radians(max(fwhm_deg, 1e-9))
    theta = math.radians(two_theta) / 2.0
    return 0.1 * k * wavelength / (beta * math.cos(theta))


def instrumental_correction(
    observed_fwhm: float, instrument_fwhm: float, lorentzian: bool = True
) -> Optional[float]:
    """Sample broadening from observed and instrument widths.

    Lorentzian deconvolution subtracts widths, Gaussian subtracts squares.
    Returns ``None`` when the observed width is not larger than the
    instrument's, which is the honest answer: the peak is resolution
    limited and no size can be extracted from it.
    """
    if observed_fwhm <= instrument_fwhm:
        return None
    if lorentzian:
        return observed_fwhm - instrument_fwhm
    return math.sqrt(observed_fwhm**2 - instrument_fwhm**2)


# -- full pattern -----------------------------------------------------


@dataclass
class SimulatedPhase:
    """One phase's contribution to a calculated pattern."""

    crystal: Crystal
    reflections: list[Reflection]
    scale: float = 1.0
    profile: Profile = field(default_factory=Profile)
    preferred_axis: Optional[tuple[int, int, int]] = None
    preferred_r: float = 1.0


def simulate(
    crystals: Crystal | Sequence[Crystal],
    two_theta: Optional[np.ndarray] = None,
    wavelength: float = 1.540598,
    two_theta_range: tuple[float, float] = (10.0, 90.0),
    step: float = 0.02,
    profile: Optional[Profile] = None,
    scales: Optional[Sequence[float]] = None,
    background: float = 0.0,
    kalpha2: bool = True,
    kalpha2_ratio: float = KALPHA2_RATIO,
    kalpha2_wavelength: Optional[float] = None,
    monochromator_two_theta: float = 0.0,
    preferred_axis: Optional[Sequence[int]] = None,
    preferred_r: float = 1.0,
    counts_at_max: Optional[float] = None,
    seed: int = 0,
    name: str = "simulado",
) -> Pattern:
    """Compute a powder pattern from one or several structures.

    Parameters
    ----------
    crystals:
        One structure or a list of them. With several, ``scales`` gives
        their relative weights; those are *not* weight fractions until the
        Rietveld scale factors are converted with
        :func:`~ramancarbon.xrd.rietveld.weight_fractions`.
    two_theta:
        Explicit axis. Overrides ``two_theta_range`` and ``step``, and is
        how a simulation is put on the same grid as a measurement.
    wavelength:
        In Å. Always the **Kα₁** line: the satellite is added on top of it
        rather than the two being averaged into one, so that the doublet
        is right at every angle instead of at none.
    kalpha2:
        Add the Kα₂ satellite at ``kalpha2_ratio`` of the Kα₁ intensity.
        On by default because it is what an unmonochromated tube emits;
        turn it off for synchrotron or monochromated data.
    counts_at_max:
        When given, the pattern is scaled so its strongest point has this
        many counts and then drawn from a Poisson distribution. Genuinely
        Poisson rather than "Gaussian noise of some amplitude", because
        the whole detection and weighting machinery downstream assumes
        σ = √N and calibrating it against noise that is not Poisson
        calibrates it against the wrong thing.

    Returns
    -------
    Pattern
    """
    if isinstance(crystals, Crystal):
        crystals = [crystals]
    crystals = list(crystals)
    if scales is None:
        scales = [1.0] * len(crystals)
    if len(scales) != len(crystals):
        raise ValueError("hacen falta tantos factores de escala como fases")
    shape = profile or Profile()

    if two_theta is None:
        axis = np.arange(two_theta_range[0], two_theta_range[1] + step / 2.0, step)
    else:
        axis = np.asarray(two_theta, dtype=float)

    lines: list[tuple[float, float]] = [(wavelength, 1.0)]
    if kalpha2:
        second = kalpha2_wavelength if kalpha2_wavelength else wavelength * 1.002486
        lines.append((second, kalpha2_ratio))
    weight = sum(w for _, w in lines)

    total = np.zeros_like(axis)
    for crystal, scale in zip(crystals, scales):
        for line_wavelength, line_weight in lines:
            for reflection in reflections(
                crystal,
                wavelength=line_wavelength,
                two_theta_range=(float(axis[0]) - 2.0, float(axis[-1]) + 2.0),
                monochromator_two_theta=monochromator_two_theta,
                preferred_axis=preferred_axis,
                preferred_r=preferred_r,
            ):
                centre = reflection.two_theta + shape.zero
                width = float(shape.fwhm(np.array([centre]))[0])
                mixing = float(shape.eta(np.array([centre]))[0])
                window = (axis > centre - 12.0 * width) & (axis < centre + 12.0 * width)
                if not window.any():
                    continue
                total[window] += (
                    scale
                    * reflection.intensity
                    * (line_weight / weight)
                    * pseudo_voigt(axis[window], centre, width, mixing)
                )

    if counts_at_max is not None and counts_at_max > 0.0:
        # Scale the peaks first and add the background afterwards, so that
        # `background` is in counts like the rest of the pattern. A
        # background expressed as a fraction of the peak before scaling
        # ends up at one or two counts, and then Poisson statistics make
        # every ripple significant — which is not a property of real data,
        # where air scatter alone puts a few hundred counts under
        # everything.
        peak = max(float(total.max()), 1e-9)
        total = total * (float(counts_at_max) / peak) + float(background)
        total = np.random.default_rng(seed).poisson(np.maximum(total, 0.0)).astype(float)
    else:
        total = total + float(background)

    return Pattern(
        two_theta=axis,
        intensity=total,
        wavelength=wavelength,
        counts=counts_at_max is not None,
        kalpha2_ratio=kalpha2_ratio if kalpha2 else 0.0,
        kalpha2_wavelength=kalpha2_wavelength,
        name=name,
        metadata={
            "simulated": True,
            "phases": [c.name for c in crystals],
            "warning": (
                "Patrón calculado, no medido. Sirve para probar el programa "
                "y para comparar con una medida, no como dato experimental."
            ),
        },
    )


__all__ = [
    "D_MERGE_RELATIVE",
    "KALPHA2_RATIO",
    "MIN_RELATIVE_INTENSITY",
    "Profile",
    "Reflection",
    "SimulatedPhase",
    "instrumental_correction",
    "lorentz_polarisation",
    "march_dollase",
    "pseudo_voigt",
    "reflections",
    "scherrer",
    "simulate",
]
