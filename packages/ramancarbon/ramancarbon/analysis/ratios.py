"""Intensity ratios, and the structural quantities derived from them.

The three ratios this module computes answer three different questions:

``I_D/I_G``
    *How much* disorder. Converts to a crystallite size or a defect density
    — but only inside the right regime, and the conversion depends on
    whether the ratio was taken from heights or from areas.
``I_2D/I_G``
    *How many layers*, for graphene-like material. Also sensitive to
    doping, which is why the 2D **lineshape** is the primary layer counter
    and this ratio is the corroboration.
``I_D/I_D'``
    *What kind* of defect: sp³, vacancy, grain boundary or substitutional.
    The most informative of the three and the least used, because it needs
    a deconvolution to separate D′ from the G band at all.

Two traps are handled here rather than left to the user.

**Heights versus areas.** For the same spectrum, an area-based I_D/I_G is
typically 1.5–2.5 times the height-based one, because the D band is broader
than the G band. Much of the published scatter in I_D/I_G is really this.
Every ratio here records its basis, and the formulas refuse a basis they
were not calibrated for.

**The two branches of the Tuinstra–Koenig relation.** I_D/I_G does not
increase monotonically with disorder. It rises as defects are introduced
into good graphite, peaks when the defect spacing falls to about 3 nm, and
then *falls* as the material amorphises. A measured I_D/I_G of 1.0
therefore corresponds to two possible structures, and quoting only the
low-defect one — as the bare Tuinstra–Koenig formula does — can be wrong by
an order of magnitude. :func:`crystallite_size` returns both branches and
uses the G-band width to say which is more likely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..database import Database, load_database
from .assignment import Assignment

#: Cançado's constant in ``L_a = C λ⁴ / (I_D/I_G)`` with λ in nm and L_a in nm.
CANCADO_LA = 2.4e-10

#: Cançado's constant in ``L_D² = C λ⁴ / (I_D/I_G)`` with λ in nm, L_D in nm.
CANCADO_LD2 = 1.8e-9

#: Uncertainty on :data:`CANCADO_LD2`, as published.
CANCADO_LD2_ERROR = 0.5e-9

#: Defect spacing at which I_D/I_G peaks and the relation turns over, nm.
TURNOVER_LD_NM = 3.0

#: Empirical factor by which an area-based I_D/I_G exceeds a height-based one.
#: Used only to warn, never to convert — the true factor is FWHM_D/FWHM_G and
#: is available directly whenever a deconvolution was run.
AREA_TO_HEIGHT_TYPICAL = 2.0


@dataclass
class Ratio:
    """One intensity ratio, with the basis it was computed on."""

    name: str
    value: Optional[float]
    basis: str
    """``"height"`` or ``"area"``."""
    numerator: Optional[float] = None
    denominator: Optional[float] = None
    available: bool = True
    reason: str = ""
    """Why it is unavailable, when it is."""

    alternate: Optional[float] = None
    """The same ratio computed on the *other* basis, when both were available.

    Carrying both is not redundancy. Published ranges are quoted sometimes
    on heights and sometimes on areas, and the two differ by the ratio of
    the two bands' widths — a factor of 2 or more for D versus G. Comparing
    an area-based measurement against a height-based literature range is a
    silent, systematic error, so every consumer of a ratio can ask for the
    basis it needs via :meth:`on_basis`."""

    @property
    def alternate_basis(self) -> str:
        """The basis :attr:`alternate` was computed on."""
        return "area" if self.basis == "height" else "height"

    def on_basis(self, basis: str) -> Optional[float]:
        """This ratio expressed on the requested basis, or ``None``.

        Exact rather than approximate: for two Lorentzians the area is
        ``(π/2)·h·w``, so the two ratios differ by exactly the ratio of the
        fitted widths, which a deconvolution provides.
        """
        if not self.available:
            return None
        if basis == self.basis:
            return self.value
        return self.alternate

    def __str__(self) -> str:
        if not self.available or self.value is None:
            return f"{self.name}: no disponible ({self.reason})"
        text = f"{self.name} = {self.value:.3f} ({self.basis})"
        if self.alternate is not None:
            text += f"  [{self.alternate:.3f} en {self.alternate_basis}]"
        return text


@dataclass
class CrystalliteSize:
    """Result of converting I_D/I_G into a structural length scale."""

    la_low_defect_nm: Optional[float]
    """In-plane crystallite size on the low-defect (Tuinstra–Koenig) branch."""
    ld_low_defect_nm: Optional[float]
    """Mean distance between point defects, low-defect branch."""
    defect_density_cm2: Optional[float]
    """Point-defect areal density, low-defect branch."""
    ld_high_defect_nm: Optional[float]
    """The other solution: the amorphisation branch, where I_D/I_G ∝ L_D²."""
    likely_branch: str
    """``"low-defect"``, ``"amorphous"`` or ``"ambiguous"``."""
    branch_reason: str
    laser_nm: float
    basis: str
    warnings: list[str] = field(default_factory=list)
    source: str = (
        "Cançado et al., Appl. Phys. Lett. 88 (2006) 163106 (L_a); "
        "Cançado et al., Nano Lett. 11 (2011) 3190 (L_D, n_D); "
        "Lucchese et al., Carbon 48 (2010) 1592 (two-branch behaviour)"
    )

    def summary(self) -> str:
        lines = []
        if self.la_low_defect_nm is not None:
            lines.append(f"L_a (rama de bajo desorden) = {self.la_low_defect_nm:.1f} nm")
        if self.ld_low_defect_nm is not None:
            lines.append(f"L_D (distancia entre defectos) = {self.ld_low_defect_nm:.1f} nm")
        if self.defect_density_cm2 is not None:
            lines.append(f"n_D = {self.defect_density_cm2:.2e} cm⁻²")
        if self.ld_high_defect_nm is not None:
            lines.append(
                f"rama alternativa (amorfización): L_D = {self.ld_high_defect_nm:.1f} nm"
            )
        lines.append(f"rama probable: {self.likely_branch} — {self.branch_reason}")
        lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


@dataclass
class DefectType:
    """Interpretation of I_D/I_D' as a defect character."""

    ratio: float
    best_match: str
    candidates: list[tuple[str, float]]
    """``(name, reference value)`` sorted by closeness."""
    confident: bool
    labels: dict[str, str] = field(default_factory=dict)
    source: str = "Eckmann et al., Nano Lett. 12 (2012) 3925"
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        name = self.labels.get(self.best_match, self.best_match)
        head = f"I_D/I_D' = {self.ratio:.2f} → defectos tipo «{name}»"
        if not self.confident:
            head += " (asignación no concluyente)"
        rest = ", ".join(
            f"{self.labels.get(k, k)}≈{v:g}" for k, v in self.candidates
        )
        lines = [head, "referencias: " + rest]
        lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


#: Human labels for the defect classes in ``perturbations.json``.
DEFECT_LABELS = {
    "sp3": "sp³ (funcionalización covalente)",
    "vacancy": "vacantes",
    "boundary": "bordes / fronteras de grano",
    "on_site_substitutional": "sustitucional en el sitio (dopante en red)",
}


def _pick(assignment: Assignment, key: str, basis: str) -> Optional[float]:
    entry = assignment.get(key)
    if entry is None:
        return None
    return _value_of(entry, basis)


def _value_of(entry, basis: str) -> Optional[float]:
    value = entry.area if basis == "area" else entry.height
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def _ratio(name: str, entry_a, entry_b, basis: str) -> Ratio:
    """Build a Ratio on ``basis``, filling in the other basis when possible."""
    primary_a, primary_b = _value_of(entry_a, basis), _value_of(entry_b, basis)
    if primary_a is None or primary_b is None:
        missing = "numerador" if primary_a is None else "denominador"
        return Ratio(name, None, basis, available=False, reason=f"{missing} no disponible")
    other = "area" if basis == "height" else "height"
    alt_a, alt_b = _value_of(entry_a, other), _value_of(entry_b, other)
    alternate = alt_a / alt_b if alt_a is not None and alt_b is not None else None
    return Ratio(name, primary_a / primary_b, basis, primary_a, primary_b, alternate=alternate)


def intensity_ratios(
    assignment: Assignment,
    basis: str = "height",
) -> dict[str, Ratio]:
    """Compute I_D/I_G, I_2D/I_G and I_D/I_D' from an assignment.

    Parameters
    ----------
    assignment:
        Output of :func:`~ramancarbon.analysis.assignment.assign_bands`.
        For I_D/I_D' this must have come from a deconvolution: D′ is a
        shoulder on the G band and a peak finder will not separate it.
    basis:
        ``"height"`` or ``"area"``. Heights are the older convention and
        the one the layer-counting ranges assume; areas are more robust to
        instrument resolution and are what the crystallite-size formula was
        calibrated on. Whichever is chosen, it is recorded and carried
        through, and mixing the two between samples is the single most
        common way an I_D/I_G comparison becomes meaningless.

    Returns
    -------
    dict[str, Ratio]
        Keys ``"ID_IG"``, ``"I2D_IG"``, ``"ID_IDp"``. Ratios that cannot be
        computed are present with ``available=False`` and a reason, rather
        than absent — so a report can say *why* a number is missing.
    """
    if basis not in {"height", "area"}:
        raise ValueError(f"basis must be 'height' or 'area', got {basis!r}")

    g_entry = assignment.g_like()
    d_entry = assignment.get("D")
    two_d_entry = assignment.get("2D")
    dp_entry = assignment.get("D'")

    out: dict[str, Ratio] = {}
    if g_entry is None:
        reason = "no se ha identificado la banda G"
        out["ID_IG"] = Ratio("I_D/I_G", None, basis, available=False, reason=reason)
        out["I2D_IG"] = Ratio("I_2D/I_G", None, basis, available=False, reason=reason)
    else:
        out["ID_IG"] = (
            _ratio("I_D/I_G", d_entry, g_entry, basis)
            if d_entry is not None
            else Ratio("I_D/I_G", None, basis, available=False, reason="banda D no detectada")
        )
        out["I2D_IG"] = (
            _ratio("I_2D/I_G", two_d_entry, g_entry, basis)
            if two_d_entry is not None
            else Ratio("I_2D/I_G", None, basis, available=False, reason="banda 2D no detectada")
        )

    if d_entry is not None and dp_entry is not None:
        out["ID_IDp"] = _ratio("I_D/I_D'", d_entry, dp_entry, basis)
    else:
        missing = "D" if d_entry is None else "D'"
        out["ID_IDp"] = Ratio(
            "I_D/I_D'",
            None,
            basis,
            available=False,
            reason=(
                f"banda {missing} no disponible; D' es un hombro sobre G y "
                "requiere deconvolución"
            ),
        )
    return out


def crystallite_size(
    id_ig: float,
    laser_nm: float,
    basis: str = "area",
    g_fwhm: Optional[float] = None,
    db: Optional[Database] = None,
) -> CrystalliteSize:
    """Convert I_D/I_G into a crystallite size and a defect density.

    Uses Cançado's generalisations of the Tuinstra–Koenig relation::

        L_a  (nm)  = 2.4e-10 · λ⁴ / (I_D/I_G)
        L_D² (nm²) = 1.8e-9  · λ⁴ / (I_D/I_G)
        n_D (cm⁻²) = 1.8e22  · (I_D/I_G) / λ⁴

    with λ the excitation wavelength in nm. The λ⁴ dependence is not
    cosmetic: the same sample measured at 633 nm gives an I_D/I_G roughly
    2.3 times larger than at 532 nm, and comparing raw ratios across
    wavelengths without this correction is meaningless.

    Parameters
    ----------
    id_ig:
        The measured ratio.
    laser_nm:
        Excitation wavelength in nm.
    basis:
        Whether ``id_ig`` came from areas or heights. The L_a relation was
        calibrated on **integrated areas**; passing heights overestimates
        L_a by roughly a factor of two to four, and this function warns
        loudly rather than silently rescaling, because the true factor is
        FWHM_D/FWHM_G and only your own fit knows it.
    g_fwhm:
        G-band width in cm⁻¹, used to decide which branch of the
        I_D/I_G-versus-disorder curve the sample sits on. A G band wider
        than ~60 cm⁻¹ means nanocrystalline or amorphous carbon, where the
        relation has turned over.
    db:
        Loaded database (unused today; accepted so the signature stays
        stable if the constants move into JSON).

    Returns
    -------
    CrystalliteSize

    Raises
    ------
    ValueError
        If the ratio or the wavelength is not positive.
    """
    if id_ig <= 0:
        raise ValueError(f"I_D/I_G must be positive, got {id_ig!r}")
    if laser_nm <= 0:
        raise ValueError(f"laser wavelength must be positive, got {laser_nm!r}")

    warnings: list[str] = []
    if basis == "height":
        warnings.append(
            "el cociente se ha tomado de alturas, pero L_a se calibró con áreas "
            "integradas. El valor de L_a que sigue está sobreestimado, "
            "típicamente en un factor 2–4 (el factor real es FWHM_D/FWHM_G). "
            "Repite con basis='area' si has hecho deconvolución"
        )

    lam4 = laser_nm**4
    la = CANCADO_LA * lam4 / id_ig
    ld2 = CANCADO_LD2 * lam4 / id_ig
    ld = float(np.sqrt(ld2))
    n_d = 1.8e22 * id_ig / lam4

    # The amorphisation branch: I_D/I_G ∝ L_D², matched to the low-defect
    # branch at the turnover so the two solutions agree there.
    id_ig_at_turnover = CANCADO_LD2 * lam4 / TURNOVER_LD_NM**2
    if id_ig <= id_ig_at_turnover:
        ld_high = TURNOVER_LD_NM * float(np.sqrt(id_ig / id_ig_at_turnover))
    else:
        ld_high = None
        warnings.append(
            f"I_D/I_G = {id_ig:.2f} supera el máximo teórico "
            f"({id_ig_at_turnover:.2f}) que la relación alcanza a {laser_nm:g} nm. "
            "Eso indica que la muestra no está en ninguna de las dos ramas "
            "simples, o que el cociente incluye intensidad de bandas amorfas "
            "(D3) que la deconvolución debería haber separado"
        )

    if ld_high is not None and ld_high < 1.0:
        warnings.append(
            f"la rama de amorfización daría L_D = {ld_high:.2f} nm, es decir "
            "defectos cada pocos enlaces. Eso ya no es «grafeno con defectos» "
            "sino carbono amorfo, y ninguna de las dos ramas describe bien la "
            "estructura: interprétalo solo como «muy desordenado»"
        )

    branch, reason = _choose_branch(g_fwhm, ld)
    if branch == "amorphous":
        warnings.append(
            "en la rama de amorfización la relación de Tuinstra-Koenig se "
            "invierte: un I_D/I_G menor significa MÁS desorden, no menos. "
            "El valor de L_a de arriba no es aplicable"
        )
    if ld < TURNOVER_LD_NM:
        warnings.append(
            f"L_D = {ld:.1f} nm está por debajo del punto de inversión "
            f"({TURNOVER_LD_NM:g} nm); las dos ramas se confunden aquí"
        )

    return CrystalliteSize(
        la_low_defect_nm=float(la),
        ld_low_defect_nm=ld,
        defect_density_cm2=float(n_d),
        ld_high_defect_nm=ld_high,
        likely_branch=branch,
        branch_reason=reason,
        laser_nm=float(laser_nm),
        basis=basis,
        warnings=warnings,
    )


def _choose_branch(g_fwhm: Optional[float], ld_low: float) -> tuple[str, str]:
    """Decide which side of the I_D/I_G turnover the sample is on."""
    if g_fwhm is None:
        return (
            "ambiguous",
            "no se ha dado la anchura de G; sin ella no se puede saber en qué "
            "rama está la muestra",
        )
    if g_fwhm > 70.0:
        return (
            "amorphous",
            f"la banda G es muy ancha (FWHM {g_fwhm:.0f} cm⁻¹), típico de carbono "
            "nanocristalino o amorfo",
        )
    if g_fwhm < 30.0:
        return (
            "low-defect",
            f"la banda G es estrecha (FWHM {g_fwhm:.0f} cm⁻¹), compatible con "
            "material grafítico bien ordenado",
        )
    return (
        "ambiguous",
        f"la anchura de G ({g_fwhm:.0f} cm⁻¹) está en la zona intermedia; "
        "ambas ramas son posibles",
    )


def defect_type(
    id_idprime: float,
    db: Optional[Database] = None,
) -> DefectType:
    """Identify the defect character from I_D/I_D'.

    The reference values, measured on graphene with deliberately introduced
    defects of each kind, are approximately 13 for sp³ (covalent
    functionalisation), 7 for vacancies, 3.5 for grain boundaries and 1.3
    for on-site substitutional dopants.

    Parameters
    ----------
    id_idprime:
        Measured ratio. It must come from a deconvolution: D′ overlaps the
        G band and its height cannot be read off the raw spectrum.
    db:
        Loaded database, which holds the reference values and their
        tolerance.

    Returns
    -------
    DefectType

    Raises
    ------
    ValueError
        If the ratio is not positive.

    Notes
    -----
    The published values were taken at 2.41 eV and the ratio has some
    excitation dependence, so the class boundaries are soft. A sample with
    more than one kind of defect gives an intermediate value that belongs
    to none of the classes — which is why ``confident`` exists.
    """
    database = db or load_database()
    if id_idprime <= 0:
        raise ValueError(f"I_D/I_D' must be positive, got {id_idprime!r}")

    reference = database.defect_type_ratios
    tolerance = float(reference.get("tolerance", 1.5))
    entries = [
        (k, float(v))
        for k, v in reference.items()
        if k in DEFECT_LABELS and isinstance(v, (int, float))
    ]
    entries.sort(key=lambda item: abs(item[1] - id_idprime))
    best, best_value = entries[0]
    confident = abs(best_value - id_idprime) <= tolerance

    warnings: list[str] = []
    if not confident:
        warnings.append(
            f"I_D/I_D' = {id_idprime:.2f} no cae dentro de ±{tolerance:g} de "
            "ningún valor de referencia; puede tratarse de una mezcla de tipos "
            "de defecto, para la que este cociente da un valor intermedio sin "
            "significado propio"
        )
    measured_at = reference.get("measured_at_ev")
    if measured_at:
        warnings.append(
            f"los valores de referencia se midieron a {measured_at} eV; el "
            "cociente depende algo de la excitación"
        )

    return DefectType(
        ratio=float(id_idprime),
        best_match=best,
        candidates=entries,
        confident=confident,
        labels=dict(DEFECT_LABELS),
        warnings=warnings,
    )


def graphene_layers(
    i2d_ig: float,
    two_d_fwhm: Optional[float] = None,
    two_d_single_lorentzian: Optional[bool] = None,
) -> tuple[str, list[str]]:
    """Estimate the graphene layer count, with the caveats attached.

    Parameters
    ----------
    i2d_ig:
        Height-based I_2D/I_G.
    two_d_fwhm:
        2D band width in cm⁻¹. **This is the primary indicator**, not the
        ratio: monolayer graphene has a 2D FWHM of 24–35 cm⁻¹ regardless of
        doping, whereas I_2D/I_G can drop from 4 to below 1 through doping
        alone without a single extra layer.
    two_d_single_lorentzian:
        Whether the 2D band is well fitted by one Lorentzian. Decisive when
        available.

    Returns
    -------
    (str, list[str])
        A verdict and the reasoning behind it.
    """
    reasons: list[str] = []
    verdict = "indeterminado"

    if two_d_fwhm is not None:
        if two_d_fwhm <= 35.0:
            verdict = "monocapa"
            reasons.append(f"FWHM(2D) = {two_d_fwhm:.0f} cm⁻¹ ≤ 35 → monocapa")
        elif two_d_fwhm <= 45.0:
            verdict = "monocapa o bicapa"
            reasons.append(f"FWHM(2D) = {two_d_fwhm:.0f} cm⁻¹, zona de transición")
        elif two_d_fwhm <= 65.0:
            verdict = "bicapa o pocas capas"
            reasons.append(f"FWHM(2D) = {two_d_fwhm:.0f} cm⁻¹ → 2–5 capas")
        else:
            verdict = "multicapa / grafito"
            reasons.append(f"FWHM(2D) = {two_d_fwhm:.0f} cm⁻¹ > 65 → grafítico")
    if two_d_single_lorentzian is not None:
        if two_d_single_lorentzian and two_d_fwhm is not None and two_d_fwhm <= 40.0:
            verdict = "monocapa"
            reasons.append("la banda 2D se ajusta con una sola lorentziana")
        elif not two_d_single_lorentzian:
            reasons.append(
                "la banda 2D necesita varias componentes → apilamiento Bernal "
                "de 2 o más capas"
            )

    if i2d_ig >= 2.0:
        reasons.append(f"I_2D/I_G = {i2d_ig:.2f} ≥ 2, compatible con monocapa")
        if verdict == "indeterminado":
            verdict = "monocapa (solo por el cociente)"
    elif i2d_ig >= 0.8:
        reasons.append(f"I_2D/I_G = {i2d_ig:.2f}, compatible con 2–4 capas")
        if verdict == "indeterminado":
            verdict = "bicapa o pocas capas (solo por el cociente)"
    else:
        reasons.append(f"I_2D/I_G = {i2d_ig:.2f} < 0.8, compatible con multicapa")
        if verdict == "indeterminado":
            verdict = "multicapa (solo por el cociente)"

    reasons.append(
        "aviso: I_2D/I_G también baja con el dopado sin cambiar el número de "
        "capas. La anchura y la forma de 2D mandan sobre el cociente"
    )
    reasons.append(
        "aviso: un bicapa turbostrático (capas giradas entre sí) presenta una "
        "2D de una sola lorentziana y se confunde con monocapa"
    )
    return verdict, reasons


__all__ = [
    "AREA_TO_HEIGHT_TYPICAL",
    "CANCADO_LA",
    "CANCADO_LD2",
    "CrystalliteSize",
    "DEFECT_LABELS",
    "DefectType",
    "Ratio",
    "TURNOVER_LD_NM",
    "crystallite_size",
    "defect_type",
    "graphene_layers",
    "intensity_ratios",
]
