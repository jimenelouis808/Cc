"""Tube diameters from the RBM and from the G-band splitting.

Two independent routes, which is the point of having both:

1. **RBM.** ``ω = A/d + B``. Direct, precise, and the standard method — but
   it needs the spectrum to reach below ~150 cm⁻¹, it only sees tubes the
   laser happens to resonate with, and the constants depend on the tube's
   environment (bundled, surfactant-wrapped, suspended). Choosing the wrong
   (A, B) pair biases every diameter by up to 10 %, so this module always
   reports which parameterisation was used and can quote the spread across
   all of them as an honest systematic.

2. **G-band splitting.** ``ω_G⁻ = ω_G⁺ − C/d²``. Needs no low-frequency
   access at all, so it works on spectra that start at 800 cm⁻¹ — but the
   constant C differs by 40 % between semiconducting and metallic tubes,
   and the splitting must be a genuine G⁻/G⁺ pair rather than a G band with
   a D′ shoulder. Applied to multi-walled material it produces a
   meaningless number, so :func:`diameter_from_g_splitting` refuses when
   the caller says the sample is multi-walled.

Where the two agree, the diameter is solid. Where they disagree, something
in the assignment is wrong, and saying so is more useful than averaging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence


from ..database import Database, load_database


@dataclass
class DiameterEstimate:
    """One diameter, with the assumption that produced it."""

    diameter_nm: float
    method: str
    """``"RBM"`` or ``"G-splitting"``."""
    parameterisation: str
    """Which database entry was used."""
    input_value: float
    """The ω_RBM or ΔG that went in, cm⁻¹."""
    uncertainty_nm: Optional[float] = None
    """Systematic spread, where one could be estimated."""
    in_calibration_range: bool = True
    """Whether the result lies inside the range the relation was calibrated on."""
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        unc = f" ± {self.uncertainty_nm:.3f}" if self.uncertainty_nm else ""
        return f"{self.diameter_nm:.3f}{unc} nm ({self.method}, {self.parameterisation})"


@dataclass
class ChiralityCandidate:
    """A ``(n, m)`` index pair compatible with a measured diameter."""

    n: int
    m: int
    diameter_nm: float
    chiral_angle_deg: float
    kind: str
    """``"armchair"``, ``"zigzag"`` or ``"chiral"``."""
    electronic: str
    """``"metallic"``, ``"quasi-metallic"`` or ``"semiconducting"``."""
    mismatch_nm: float
    """|d(n,m) − d_measured|."""

    @property
    def label(self) -> str:
        return f"({self.n},{self.m})"

    def __str__(self) -> str:
        return (
            f"({self.n},{self.m})  d={self.diameter_nm:.3f} nm  "
            f"θ={self.chiral_angle_deg:.1f}°  {self.kind}, {self.electronic}"
        )


def rbm_to_diameter(
    omega: float,
    parameterisation: Optional[str] = None,
    db: Optional[Database] = None,
) -> DiameterEstimate:
    """Diameter from an RBM frequency.

    Inverts ``ω = A/d + B``, or for the Araujo form ``ω = (A/d)·sqrt(1 +
    C_e d²)``, which rearranges to ``d = sqrt(A² / (ω² − A² C_e))``.

    Parameters
    ----------
    omega:
        RBM frequency in cm⁻¹.
    parameterisation:
        Key from ``rbm.json``. ``None`` uses the database default
        (bundled powder). Choose the one matching your sample's
        environment; see :func:`compare_parameterisations` to see how much
        it matters for your particular frequency.
    db:
        Loaded database; one is loaded if omitted.

    Returns
    -------
    DiameterEstimate

    Raises
    ------
    ValueError
        If the frequency is not positive, or is below the offset ``B`` (in
        which case the relation gives a negative or infinite diameter — a
        sign that the peak is not an RBM).
    """
    database = db or load_database()
    param = database.rbm_parameterisation(parameterisation)
    if omega <= 0:
        raise ValueError(f"RBM frequency must be positive, got {omega!r}")

    warnings: list[str] = []
    if param.is_multiplicative:
        c_e = float(param.environment_correction)
        denom = omega**2 - param.A**2 * c_e
        if denom <= 0:
            raise ValueError(
                f"ω = {omega:g} cm⁻¹ is too low for the {param.key} relation "
                "(the environment term dominates); this peak is probably not an RBM"
            )
        diameter = param.A / math.sqrt(denom)
    else:
        denom = omega - param.B
        if denom <= 0:
            raise ValueError(
                f"ω = {omega:g} cm⁻¹ is at or below the offset B = {param.B:g} cm⁻¹ "
                f"of the {param.key} relation, which would give an infinite "
                "diameter; this peak is probably not an RBM"
            )
        diameter = param.A / denom

    low, high = param.diameter_range_nm
    in_range = low <= diameter <= high
    if not in_range:
        warnings.append(
            f"d = {diameter:.3f} nm cae fuera del rango de calibración "
            f"{low:g}–{high:g} nm de «{param.key}»; extrapolación"
        )

    return DiameterEstimate(
        diameter_nm=float(diameter),
        method="RBM",
        parameterisation=param.key,
        input_value=float(omega),
        in_calibration_range=in_range,
        source=param.source,
        warnings=warnings,
    )


def diameter_to_rbm(
    diameter_nm: float,
    parameterisation: Optional[str] = None,
    db: Optional[Database] = None,
) -> float:
    """Forward direction: expected RBM frequency for a given diameter, cm⁻¹."""
    database = db or load_database()
    param = database.rbm_parameterisation(parameterisation)
    if diameter_nm <= 0:
        raise ValueError("diameter must be positive")
    if param.is_multiplicative:
        c_e = float(param.environment_correction)
        return float((param.A / diameter_nm) * math.sqrt(1.0 + c_e * diameter_nm**2))
    return float(param.A / diameter_nm + param.B)


def compare_parameterisations(
    omega: float, db: Optional[Database] = None
) -> dict[str, DiameterEstimate]:
    """Diameter from every parameterisation in the database.

    The spread between them is the systematic uncertainty that a single
    quoted diameter hides. For a typical 200 cm⁻¹ RBM the answers differ by
    ~0.15 nm, which is larger than the statistical uncertainty on the peak
    position by an order of magnitude.

    Returns
    -------
    dict[str, DiameterEstimate]
        Keyed by parameterisation. Entries that cannot be inverted at this
        frequency are omitted rather than reported as errors.
    """
    database = db or load_database()
    out: dict[str, DiameterEstimate] = {}
    for key in database.rbm:
        try:
            out[key] = rbm_to_diameter(omega, key, db=database)
        except ValueError:
            continue
    return out


def rbm_diameter_with_spread(
    omega: float,
    parameterisation: Optional[str] = None,
    db: Optional[Database] = None,
) -> DiameterEstimate:
    """Diameter from the chosen relation, with the inter-relation spread as ±.

    This is the function the report uses. The central value comes from the
    parameterisation the user chose; the uncertainty is the half-range over
    all parameterisations whose calibration covers the result, which is a
    fair statement of "how much does this depend on a choice I made".
    """
    database = db or load_database()
    central = rbm_to_diameter(omega, parameterisation, db=database)
    others = [
        est.diameter_nm
        for est in compare_parameterisations(omega, db=database).values()
        if est.in_calibration_range
    ]
    if len(others) >= 2:
        central.uncertainty_nm = float((max(others) - min(others)) / 2.0)
    return central


def diameter_from_g_splitting(
    omega_g_plus: float,
    omega_g_minus: float,
    metallic: bool = False,
    walls: Optional[int] = None,
    db: Optional[Database] = None,
) -> DiameterEstimate:
    """Diameter from the curvature-induced G⁻/G⁺ splitting.

    ``ω_G⁻ = ω_G⁺ − C/d²`` so ``d = sqrt(C / Δω)``, with C = 47.7 cm⁻¹nm²
    for semiconducting tubes and 79.5 for metallic ones.

    Parameters
    ----------
    omega_g_plus, omega_g_minus:
        Fitted positions of the two G components, cm⁻¹.
    metallic:
        Whether the tube is metallic. Getting this wrong changes the
        diameter by ~30 %, so the caller should base it on the BWF
        lineshape of G⁻ (metallic tubes give an asymmetric, broad G⁻) or on
        the RBM/Kataura assignment, not on a guess.
    walls:
        Number of walls, when known. Passing ``walls >= 3`` raises: in
        multi-walled material what looks like a split G band is G + D′, and
        this formula would turn a defect band into a fictitious diameter.
    db:
        Loaded database.

    Returns
    -------
    DiameterEstimate

    Raises
    ------
    ValueError
        If the splitting is not positive, or the sample is multi-walled.
    """
    database = db or load_database()
    constants = database.g_splitting
    if walls is not None and walls >= 3:
        raise ValueError(
            "the G-splitting diameter relation does not apply to multi-walled "
            "material: the feature above the G band there is D′ (a defect band), "
            "not a curvature-split G⁻, and this formula would report a diameter "
            "for a defect"
        )
    delta = float(omega_g_plus) - float(omega_g_minus)
    if delta <= 0:
        raise ValueError(
            f"G⁺ ({omega_g_plus:g}) must lie above G⁻ ({omega_g_minus:g}) cm⁻¹"
        )

    c = float(constants["C_metallic" if metallic else "C_semiconducting"])
    diameter = math.sqrt(c / delta)

    warnings: list[str] = []
    if delta < 5.0:
        warnings.append(
            f"la separación G⁺−G⁻ es de solo {delta:.1f} cm⁻¹; por debajo de "
            "~5 cm⁻¹ el diámetro deducido es muy sensible al ajuste"
        )
    if diameter > 3.0:
        warnings.append(
            f"d = {diameter:.2f} nm; por encima de ~3 nm la separación G es "
            "demasiado pequeña para medirse con fiabilidad"
        )
    other = math.sqrt(
        float(constants["C_semiconducting" if metallic else "C_metallic"]) / delta
    )
    warnings.append(
        f"si el tubo fuera {'semiconductor' if metallic else 'metálico'} en lugar de "
        f"{'metálico' if metallic else 'semiconductor'}, d sería {other:.3f} nm"
    )

    return DiameterEstimate(
        diameter_nm=float(diameter),
        method="G-splitting",
        parameterisation="metallic" if metallic else "semiconducting",
        input_value=delta,
        uncertainty_nm=abs(other - diameter) / 2.0,
        in_calibration_range=0.7 <= diameter <= 3.0,
        source=str(constants.get("source", "")),
        warnings=warnings,
    )


# ----------------------------------------------------------------------
# chirality
# ----------------------------------------------------------------------
def chiral_diameter(n: int, m: int, a_nm: float = 0.2459512146747803) -> float:
    """Diameter of an ``(n, m)`` tube in nm: ``d = (a/π)·sqrt(n²+nm+m²)``."""
    if n < 0 or m < 0 or (n == 0 and m == 0):
        raise ValueError(f"invalid chiral indices ({n}, {m})")
    return a_nm / math.pi * math.sqrt(n * n + n * m + m * m)


def chiral_angle(n: int, m: int) -> float:
    """Chiral angle in degrees, 0° (zigzag) to 30° (armchair)."""
    if n == 0 and m == 0:
        raise ValueError("invalid chiral indices (0, 0)")
    return math.degrees(math.atan2(math.sqrt(3.0) * m, 2.0 * n + m))


def electronic_type(n: int, m: int) -> str:
    """Metallic, quasi-metallic or semiconducting, from ``(n − m) mod 3``.

    Armchair tubes (n = m) are truly metallic. Other tubes with
    ``(n − m) mod 3 == 0`` are metallic in the zone-folding picture but
    curvature opens a small gap of a few tens of meV, so they are called
    quasi-metallic — the distinction matters for Raman because both show
    the broad Breit–Wigner–Fano G⁻ lineshape.
    """
    if (n - m) % 3 != 0:
        return "semiconducting"
    return "metallic" if n == m else "quasi-metallic"


def chirality_kind(n: int, m: int) -> str:
    """``"armchair"``, ``"zigzag"`` or ``"chiral"``."""
    if n == m:
        return "armchair"
    if m == 0:
        return "zigzag"
    return "chiral"


def assign_chirality(
    diameter_nm: float,
    tolerance_nm: float = 0.03,
    max_index: int = 40,
    only: Optional[str] = None,
    db: Optional[Database] = None,
) -> list[ChiralityCandidate]:
    """List the ``(n, m)`` tubes whose diameter matches a measurement.

    Parameters
    ----------
    diameter_nm:
        Measured diameter.
    tolerance_nm:
        How close a candidate must come. The default 0.03 nm corresponds to
        roughly ±3 cm⁻¹ in the RBM at 200 cm⁻¹ — about the reproducibility
        of a well-calibrated spectrometer.
    max_index:
        Largest chiral index searched. 40 covers diameters up to ~3.2 nm.
    only:
        Restrict to ``"metallic"`` (includes quasi-metallic) or
        ``"semiconducting"``. Use it when the G⁻ lineshape has already
        settled the question.
    db:
        Loaded database, for the lattice constant.

    Returns
    -------
    list[ChiralityCandidate]
        Sorted by how well the diameter matches.

    Notes
    -----
    Diameter alone does **not** determine (n, m): several tubes share a
    diameter to within any realistic tolerance. A unique assignment needs
    the resonance condition too — which transition energy E_ii the laser
    matches — and that is a Kataura-plot analysis this package does not
    attempt, because doing it properly needs an excitation-dependent
    dataset rather than one spectrum. What is returned is the candidate
    *set*, honestly labelled as such.
    """
    database = db or load_database()
    a_nm = float(database.chirality["lattice_constant_nm"])
    if diameter_nm <= 0:
        raise ValueError("diameter must be positive")
    if tolerance_nm <= 0:
        raise ValueError("tolerance must be positive")

    out: list[ChiralityCandidate] = []
    for n in range(1, max_index + 1):
        for m in range(0, n + 1):
            d = chiral_diameter(n, m, a_nm)
            mismatch = abs(d - diameter_nm)
            if mismatch > tolerance_nm:
                continue
            kind = electronic_type(n, m)
            if only == "metallic" and kind == "semiconducting":
                continue
            if only == "semiconducting" and kind != "semiconducting":
                continue
            out.append(
                ChiralityCandidate(
                    n=n,
                    m=m,
                    diameter_nm=d,
                    chiral_angle_deg=chiral_angle(n, m),
                    kind=chirality_kind(n, m),
                    electronic=kind,
                    mismatch_nm=mismatch,
                )
            )
    out.sort(key=lambda c: c.mismatch_nm)
    return out


# ----------------------------------------------------------------------
# multi-wall pairing
# ----------------------------------------------------------------------
@dataclass
class WallPair:
    """A candidate inner/outer tube pair in a double-walled nanotube."""

    inner_omega: float
    outer_omega: float
    inner_diameter_nm: float
    outer_diameter_nm: float
    spacing_nm: float
    """Half the diameter difference: the wall-to-wall separation."""
    plausible: bool

    def __str__(self) -> str:
        mark = "✓" if self.plausible else "✗"
        return (
            f"{mark} {self.outer_omega:.1f} cm⁻¹ (d={self.outer_diameter_nm:.3f} nm) "
            f"⊃ {self.inner_omega:.1f} cm⁻¹ (d={self.inner_diameter_nm:.3f} nm), "
            f"separación {self.spacing_nm:.3f} nm"
        )


def find_wall_pairs(
    rbm_frequencies: Sequence[float],
    parameterisation: Optional[str] = None,
    inner_parameterisation: Optional[str] = "milnera_dwcnt_inner",
    db: Optional[Database] = None,
    tolerance_nm: float = 0.02,
) -> list[WallPair]:
    """Find RBM pairs consistent with concentric walls.

    Two tubes are concentric if their radii differ by one wall spacing, so
    their **diameters** differ by twice that: 0.66–0.72 nm. This is the
    quantitative test that separates a genuine DWCNT from a mixture of
    single-walled tubes of two different diameters, which otherwise produce
    an identical-looking pair of RBM clusters.

    Parameters
    ----------
    rbm_frequencies:
        Fitted RBM positions in cm⁻¹.
    parameterisation:
        RBM relation for the outer tubes.
    inner_parameterisation:
        RBM relation for the inner tubes. Inner walls are shielded by the
        outer one and sit in a stiffer environment, so they have their own
        calibration; passing ``None`` uses the same relation for both,
        which will systematically overestimate the inner diameter.
    db:
        Loaded database.
    tolerance_nm:
        How far outside the published spacing range a pair may fall and
        still be reported (flagged as implausible).

    Returns
    -------
    list[WallPair]
        Every ordered pair, sorted with the plausible ones first. An empty
        list means no pair in the input is consistent with concentricity.
    """
    database = db or load_database()
    low, high = database.wall_spacing_nm
    freqs = sorted(float(f) for f in rbm_frequencies)
    pairs: list[WallPair] = []
    for outer_omega in freqs:
        for inner_omega in freqs:
            if inner_omega <= outer_omega:
                continue  # inner tubes are smaller, hence higher frequency
            try:
                outer = rbm_to_diameter(outer_omega, parameterisation, db=database)
                inner = rbm_to_diameter(
                    inner_omega, inner_parameterisation or parameterisation, db=database
                )
            except ValueError:
                continue
            spacing = (outer.diameter_nm - inner.diameter_nm) / 2.0
            if spacing <= 0:
                continue
            plausible = (low - tolerance_nm) <= spacing <= (high + tolerance_nm)
            if spacing > high + 5 * tolerance_nm:
                continue  # far too far apart to be worth listing
            pairs.append(
                WallPair(
                    inner_omega=inner_omega,
                    outer_omega=outer_omega,
                    inner_diameter_nm=inner.diameter_nm,
                    outer_diameter_nm=outer.diameter_nm,
                    spacing_nm=spacing,
                    plausible=plausible,
                )
            )
    pairs.sort(key=lambda p: (not p.plausible, abs(p.spacing_nm - 0.5 * (low + high))))
    return pairs


__all__ = [
    "ChiralityCandidate",
    "DiameterEstimate",
    "WallPair",
    "assign_chirality",
    "chiral_angle",
    "chiral_diameter",
    "chirality_kind",
    "compare_parameterisations",
    "diameter_from_g_splitting",
    "diameter_to_rbm",
    "electronic_type",
    "find_wall_pairs",
    "rbm_diameter_with_spread",
    "rbm_to_diameter",
]
