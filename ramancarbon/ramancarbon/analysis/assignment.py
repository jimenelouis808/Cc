"""Matching observed features to named Raman bands.

Assignment is where the laser wavelength stops being metadata and starts
mattering. Every window in the database is defined at 2.33 eV and is
translated by the band's dispersion before anything is compared, so a D band
measured at 785 nm (1.58 eV) is looked for near 1312 cm⁻¹, not 1350. Without
that correction a 785 nm spectrum has "no D band" and a "20 cm⁻¹ downshift"
that is pure arithmetic.

Two ambiguities are handled explicitly rather than silently:

* **D versus diamond.** Both sit near 1330–1350 cm⁻¹. The discriminators
  are width (diamond is narrow, < 20 cm⁻¹) and dispersion (diamond does not
  move with the laser, D does). With one spectrum only the width is
  available, so both candidates are reported when the width is borderline.
* **G versus G⁺/G⁻ versus D′.** A shoulder above the G band is D′ in
  multi-walled and graphitic material but G⁺ in single-walled material,
  and a shoulder below it is G⁻. Getting this backwards turns a defect band
  into a tube diameter. :func:`resolve_g_region` applies the width and
  spacing tests that separate the cases and says which one it took.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.peaks import PeakMeasurement
from ..core.spectrum import Spectrum
from ..database import Band, Database, load_database
from ..models.fitting import FitResult, FittedPeak


@dataclass
class BandAssignment:
    """One observed feature identified as a named band."""

    key: str
    band: Band
    position: float
    height: float
    fwhm: Optional[float]
    area: Optional[float]
    origin: str
    """``"fit"`` when it came from a deconvolution, ``"peak"`` from the peak finder."""
    expected_position: float
    """Where the database says it should be, at this laser."""
    deviation: float
    """``position − expected_position``, cm⁻¹. The number the shift analysis uses."""
    width_ok: bool
    """Whether the FWHM falls inside the band's typical range."""
    alternatives: tuple[str, ...] = ()
    """Other bands whose window also contains this feature."""
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.band.name

    def __str__(self) -> str:
        width = f"{self.fwhm:.1f}" if self.fwhm else "n/a"
        if self.band.position_is_reference:
            reference = (
                f"(esperado {self.expected_position:.1f}, Δ={self.deviation:+.1f})"
            )
        else:
            # The RBM has no nominal position to deviate from: its frequency is
            # the measurement, not a shift.
            reference = "(posición fijada por el diámetro)"
        return (
            f"{self.key:>7s}  {self.position:8.1f} cm⁻¹  {reference}  "
            f"I={self.height:.4g}  FWHM={width}"
        )


@dataclass
class Assignment:
    """The full set of identified bands for one spectrum."""

    bands: dict[str, BandAssignment]
    laser_ev: Optional[float]
    unassigned: list[float]
    """Positions of features that matched no band window."""

    extras: dict[str, list[BandAssignment]] = field(default_factory=dict)
    """Additional instances of bands a spectrum may legitimately show more
    than once. A sample with a spread of tube diameters has one RBM per
    resonant diameter, and reporting only the strongest — or listing the
    rest as "unassigned" — throws away exactly the information the RBM
    region is there to provide."""
    missing: list[str] = field(default_factory=list)
    """Bands looked for inside the measured range but not found."""
    out_of_range: list[str] = field(default_factory=list)
    """Bands the spectrum does not cover, so their absence means nothing."""
    warnings: list[str] = field(default_factory=list)

    def __contains__(self, key: str) -> bool:
        return key in self.bands

    def get(self, key: str) -> Optional[BandAssignment]:
        """Look up one assigned band."""
        return self.bands.get(key)

    def position(self, key: str) -> Optional[float]:
        """Position of one assigned band, or ``None``."""
        entry = self.bands.get(key)
        return entry.position if entry else None

    def height(self, key: str) -> Optional[float]:
        """Height of one assigned band, or ``None``."""
        entry = self.bands.get(key)
        return entry.height if entry else None

    def area(self, key: str) -> Optional[float]:
        """Area of one assigned band, or ``None``."""
        entry = self.bands.get(key)
        return entry.area if entry else None

    def all_of(self, key: str) -> list[BandAssignment]:
        """Every instance of a band: the primary one plus any extras."""
        primary = self.bands.get(key)
        return ([primary] if primary else []) + self.extras.get(key, [])

    def g_like(self) -> Optional[BandAssignment]:
        """Whichever of G, G⁺ is present — the intensity reference.

        Ratios are quoted against "the G band". In a single-walled spectrum
        that means G⁺, in everything else the single G. Getting this wrong
        changes I_D/I_G by however much G⁻ contributes, so it is resolved
        in one place rather than at each call site.
        """
        return self.bands.get("G") or self.bands.get("G+")

    def summary(self) -> str:
        """Table of assigned bands, ordered by position."""
        every = list(self.bands.values())
        for group in self.extras.values():
            every.extend(group)
        lines = [str(a) for a in sorted(every, key=lambda b: b.position)]
        if self.missing:
            lines.append("no detectadas (dentro del rango medido): " + ", ".join(self.missing))
        if self.out_of_range:
            lines.append("fuera del rango medido: " + ", ".join(self.out_of_range))
        if self.unassigned:
            lines.append(
                "sin asignar: " + ", ".join(f"{p:.0f}" for p in self.unassigned) + " cm⁻¹"
            )
        for w in self.warnings:
            lines.append("⚠ " + w)
        return "\n".join(lines)


#: Bands that :func:`assign_bands` will not assign from a bare peak search,
#: because they are shoulders that only a deconvolution can locate.
SHOULDER_BANDS = frozenset({"D3", "D4", "G-", "G+"})


def assign_bands(
    spectrum: Spectrum,
    features: Sequence[PeakMeasurement | FitResult] | FitResult,
    db: Optional[Database] = None,
    keys: Optional[Iterable[str]] = None,
    include_shoulders: Optional[bool] = None,
) -> Assignment:
    """Identify observed features as named bands.

    Parameters
    ----------
    spectrum:
        The spectrum the features came from; used for its laser energy and
        for deciding which bands were even measurable.
    features:
        Any mixture of :class:`~ramancarbon.models.fitting.FitResult` objects
        and :class:`~ramancarbon.core.peaks.PeakMeasurement` objects, or a
        single fit result.

        Mixing is the normal case and the reason this accepts a sequence: a
        real analysis fits the D–G region with one model, the 2D band with
        another and the RBM region with a third, and each fit only knows
        about its own window. A peak that falls inside some fit's window is
        dropped in favour of that fit's component, because a deconvolution
        separates overlapping bands and a peak finder does not; peaks
        outside every fitted window are kept, which is how the 2D band
        still gets assigned when only the D–G region was deconvolved.
    db:
        Loaded database.
    keys:
        Restrict to these band keys. ``None`` considers every band.
    include_shoulders:
        Whether to assign the shoulder bands (D3, D4, G⁺, G⁻) from a bare
        peak list. Defaults to ``True`` for a fit result and ``False`` for a
        peak list, because a peak finder cannot resolve a shoulder and
        would attach the label to whatever noise ripple sits nearest.

    Returns
    -------
    Assignment
    """
    database = db or load_database()
    laser_ev = spectrum.laser_ev
    sources = [features] if isinstance(features, FitResult) else list(features)
    from_fit = any(isinstance(item, FitResult) for item in sources)
    if include_shoulders is None:
        include_shoulders = from_fit

    warnings: list[str] = []
    if laser_ev is None:
        warnings.append(
            "no se conoce la longitud de onda del láser: las posiciones "
            "esperadas no se han corregido por dispersión, así que las "
            "desviaciones de las bandas D, 2D y D+D' no son fiables"
        )

    candidates = _candidate_bands(database, keys, include_shoulders)
    observed = _as_observations(sources)

    assigned: dict[str, BandAssignment] = {}
    used: set[int] = set()

    # A component that came out of a deconvolution already knows which band
    # it was meant to be; trust that over a window match.
    for index, obs in enumerate(observed):
        if obs["band"] and obs["band"] in candidates:
            key = obs["band"]
            if key in assigned:
                continue
            assigned[key] = _make(candidates[key], obs, laser_ev, database)
            used.add(index)

    for key, band in candidates.items():
        if key in assigned:
            continue
        low, high = band.window_at(laser_ev)
        pool = [
            (i, o)
            for i, o in enumerate(observed)
            if i not in used and low <= o["position"] <= high
        ]
        if not pool:
            continue
        expected = band.position_at(laser_ev)
        index, obs = min(pool, key=lambda item: abs(item[1]["position"] - expected))
        assigned[key] = _make(band, obs, laser_ev, database)
        used.add(index)

    # Record every other band whose window also contained the chosen feature.
    for key, entry in assigned.items():
        others = tuple(
            other.key
            for other in candidates.values()
            if other.key != key
            and other.window_at(laser_ev)[0] <= entry.position <= other.window_at(laser_ev)[1]
        )
        if others:
            entry.alternatives = others

    extras: dict[str, list[BandAssignment]] = {}
    for key, band in candidates.items():
        if not band.multi_valued or key not in assigned:
            continue
        low, high = band.window_at(laser_ev)
        for index, obs in enumerate(observed):
            if index in used or not low <= obs["position"] <= high:
                continue
            extras.setdefault(key, []).append(_make(band, obs, laser_ev, database))
            used.add(index)

    unassigned = [o["position"] for i, o in enumerate(observed) if i not in used]

    missing: list[str] = []
    out_of_range: list[str] = []
    for key, band in candidates.items():
        if key in assigned:
            continue
        low, high = band.window_at(laser_ev)
        if spectrum.covers(low, high, fraction=0.6):
            missing.append(key)
        else:
            out_of_range.append(key)

    warnings.extend(_ambiguity_warnings(assigned, laser_ev))

    return Assignment(
        bands=assigned,
        laser_ev=laser_ev,
        unassigned=unassigned,
        extras=extras,
        missing=missing,
        out_of_range=out_of_range,
        warnings=warnings,
    )


def _candidate_bands(
    db: Database, keys: Optional[Iterable[str]], include_shoulders: bool
) -> dict[str, Band]:
    if keys is not None:
        wanted = {k: db.band(k) for k in keys}
    else:
        wanted = dict(db.bands)
    if not include_shoulders:
        wanted = {k: v for k, v in wanted.items() if k not in SHOULDER_BANDS}
    return wanted


def _as_observations(sources: Sequence[PeakMeasurement | FitResult]) -> list[dict]:
    """Normalise fitted components and peak measurements to one shape.

    Fitted components win over peaks in the same window; see
    :func:`assign_bands`.
    """
    observations: list[dict] = []
    fitted_windows: list[tuple[float, float]] = []

    for item in sources:
        if not isinstance(item, FitResult):
            continue
        if item.x.size:
            fitted_windows.append((float(item.x[0]), float(item.x[-1])))
        for component in item.peaks:
            observations.append(
                {
                    "position": component.peak_position,
                    "height": component.peak_height,
                    "fwhm": component.fwhm,
                    "area": component.area,
                    "origin": "fit",
                    "band": component.band,
                    "profile": component.profile,
                }
            )

    for item in sources:
        if isinstance(item, FitResult):
            continue
        if any(lo <= item.position <= hi for lo, hi in fitted_windows):
            continue
        observations.append(
            {
                "position": item.position,
                "height": item.height,
                "fwhm": item.fwhm,
                "area": item.area,
                "origin": "peak",
                "band": None,
                "profile": None,
            }
        )

    observations.sort(key=lambda o: o["position"])
    return observations


def _make(band: Band, obs: dict, laser_ev: Optional[float], db: Database) -> BandAssignment:
    expected = band.position_at(laser_ev)
    fwhm = obs["fwhm"]
    lo, hi = band.typical_fwhm
    width_ok = fwhm is None or (lo <= fwhm <= hi)
    notes: list[str] = []
    if fwhm is not None and not width_ok:
        notes.append(
            f"FWHM {fwhm:.1f} cm⁻¹ fuera del rango típico {lo:g}–{hi:g} cm⁻¹ "
            f"para {band.key}"
        )
    return BandAssignment(
        key=band.key,
        band=band,
        position=float(obs["position"]),
        height=float(obs["height"]),
        fwhm=float(fwhm) if fwhm is not None else None,
        area=float(obs["area"]) if obs["area"] is not None else None,
        origin=obs["origin"],
        expected_position=float(expected),
        deviation=float(obs["position"] - expected),
        width_ok=width_ok,
        notes=notes,
    )


def _ambiguity_warnings(
    assigned: dict[str, BandAssignment], laser_ev: Optional[float]
) -> list[str]:
    """The two assignments that go wrong most often, called out by name."""
    out: list[str] = []

    d_band = assigned.get("D")
    diamond = assigned.get("diamond")
    if d_band is not None and d_band.fwhm is not None and d_band.fwhm < 20.0:
        out.append(
            f"la banda asignada a D es muy estrecha (FWHM {d_band.fwhm:.1f} cm⁻¹). "
            "Una línea estrecha cerca de 1332 cm⁻¹ que NO se desplaza al cambiar "
            "el láser es diamante (sp3), no la banda D. Mide con dos láseres "
            "para distinguirlas"
        )
    if diamond is not None and d_band is not None:
        out.append(
            "se han asignado a la vez diamante (1332) y banda D; con un solo "
            "espectro no se pueden separar de forma concluyente"
        )

    g_plus = assigned.get("G+")
    d_prime = assigned.get("D'")
    if g_plus is not None and d_prime is not None:
        gap = d_prime.position - g_plus.position
        if gap < 15.0:
            out.append(
                f"G⁺ y D' están separadas solo {gap:.1f} cm⁻¹; probablemente son "
                "la misma característica ajustada dos veces"
            )
    g_minus = assigned.get("G-")
    if g_minus is not None and g_minus.fwhm is not None and g_minus.fwhm > 60.0:
        out.append(
            f"G⁻ es ancha (FWHM {g_minus.fwhm:.1f} cm⁻¹): compatible con tubos "
            "metálicos (perfil Breit-Wigner-Fano). Reajusta G⁻ con perfil BWF "
            "antes de deducir el diámetro a partir de la separación G"
        )
    return out


@dataclass
class GRegionInterpretation:
    """What the structure of the 1500–1650 cm⁻¹ region means."""

    interpretation: str
    """``"single"``, ``"swcnt_split"``, ``"g_plus_dprime"`` or ``"unresolved"``."""
    g_position: Optional[float]
    partner_position: Optional[float]
    partner_role: Optional[str]
    """``"G-"``, ``"D'"`` or ``None``."""
    reasons: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.interpretation}: " + "; ".join(self.reasons)


def resolve_g_region(
    fit: FitResult,
    rbm_present: bool,
    db: Optional[Database] = None,
) -> GRegionInterpretation:
    """Decide what the components in the G region actually are.

    The rules, in the order they are applied:

    1. A component **above** the main G peak by 20–50 cm⁻¹ with a width
       under ~40 cm⁻¹ is D′ — unless the main peak is itself already at
       1590 and the extra component is below it.
    2. A component **below** the main G peak, together with an RBM, is G⁻:
       curvature-split single-walled material.
    3. A component below the main G peak with **no** RBM anywhere in a
       spectrum that reaches low enough to have seen one is not G⁻. In
       multi-walled and graphitic material that intensity is the D3
       amorphous band or the tail of D, and calling it G⁻ would produce a
       fictitious tube diameter.

    Parameters
    ----------
    fit:
        A deconvolution covering the G region.
    rbm_present:
        Whether a genuine RBM was detected. Pass ``False`` only when the
        spectrum actually covered the RBM window; otherwise the third rule
        fires on a spectrum that simply started too high.
    db:
        Loaded database.

    Returns
    -------
    GRegionInterpretation
    """
    database = db or load_database()
    components = sorted(
        (p for p in fit.peaks if 1450.0 <= p.peak_position <= 1700.0),
        key=lambda p: p.peak_height,
        reverse=True,
    )
    if not components:
        return GRegionInterpretation("unresolved", None, None, None, ["no hay componentes en 1450–1700 cm⁻¹"])

    main = components[0]
    others = [p for p in components[1:]]
    if not others:
        return GRegionInterpretation(
            "single", main.peak_position, None, None, ["una sola componente en la región G"]
        )

    partner = max(others, key=lambda p: p.peak_height)
    gap = partner.peak_position - main.peak_position
    reasons: list[str] = []

    if gap > 0:
        if 12.0 <= gap <= 60.0 and partner.fwhm <= 45.0:
            reasons.append(
                f"componente {gap:.1f} cm⁻¹ por encima de G con FWHM "
                f"{partner.fwhm:.1f} cm⁻¹ → D′ (banda de defectos)"
            )
            return GRegionInterpretation(
                "g_plus_dprime", main.peak_position, partner.peak_position, "D'", reasons
            )
        reasons.append(f"componente {gap:.1f} cm⁻¹ por encima de G, no concluyente")
        return GRegionInterpretation(
            "unresolved", main.peak_position, partner.peak_position, None, reasons
        )

    # partner is below the main peak
    if rbm_present:
        reasons.append(
            f"componente {abs(gap):.1f} cm⁻¹ por debajo de G y hay RBM → "
            "desdoblamiento G⁻/G⁺ por curvatura (material de pared simple o doble)"
        )
        return GRegionInterpretation(
            "swcnt_split", main.peak_position, partner.peak_position, "G-", reasons
        )

    reasons.append(
        f"componente {abs(gap):.1f} cm⁻¹ por debajo de G pero sin RBM: se "
        "interpreta como banda amorfa D3 o cola de D, no como G⁻. No se deduce "
        "diámetro a partir de esta separación"
    )
    return GRegionInterpretation(
        "unresolved", main.peak_position, partner.peak_position, None, reasons
    )


__all__ = [
    "Assignment",
    "BandAssignment",
    "GRegionInterpretation",
    "SHOULDER_BANDS",
    "assign_bands",
    "resolve_g_region",
]
