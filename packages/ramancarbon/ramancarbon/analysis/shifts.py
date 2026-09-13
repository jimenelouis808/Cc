"""Band shifts against the literature: doping, strain, or neither.

A measured band position only means something relative to a reference. This
module compares against two kinds:

* a **pristine reference** — undoped, unstrained material of the same type,
  from ``perturbations.json``;
* a **user-supplied control** — the same sample before treatment, measured
  on the same instrument on the same day, which is a far better reference
  and the one a careful experiment provides.

Then it interprets the shift. The physics that makes interpretation possible:

* The G band **stiffens for both electron and hole doping**. That is not
  intuitive. It happens because moving the Fermi level away from the Dirac
  point Pauli-blocks the virtual electron–hole pairs that soften the phonon
  (the Kohn anomaly), so the phonon hardens either way. The G linewidth
  *narrows* at the same time.
* The 2D band moves in **opposite directions** for the two carriers: up for
  holes, down for electrons. So the pair (ΔG, Δ2D) carries the sign of the
  doping, which ΔG alone does not.
* **Strain softens both** bands, and moves them along a line of slope
  Δω_2D/Δω_G ≈ 2.2, whereas doping moves them along a line of slope ≈ 0.7.
  Two different directions in the same plane means a measured pair can be
  decomposed into a strain part and a doping part — the single most useful
  quantitative shift analysis available.

Three warnings this module always issues, because each of them has produced
published nonsense:

1. **Calibrate.** A spectrometer drifts by several cm⁻¹ between sessions,
   which is the size of the effects being measured. Without a same-session
   reference line (the 520.7 cm⁻¹ line of silicon is the usual choice), an
   absolute comparison against a literature number measures the instrument,
   not the sample.
2. **Check the laser power.** A few mW on a black powder heats it by
   hundreds of kelvin, downshifting G by several cm⁻¹ and mimicking tensile
   strain. If halving the power moves the band, the shift was thermal.
3. **The strain/doping decomposition is calibrated for monolayer graphene.**
   Applying it to nanotubes or multi-walled material gives numbers that
   look precise and mean nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..database import Database, DopantSignature, load_database
from .assignment import Assignment


@dataclass
class BandShift:
    """One band's displacement from its reference position."""

    key: str
    measured: float
    reference: float
    delta: float
    reference_kind: str
    """``"literature"`` or ``"control"``."""
    significant: bool
    """Whether |delta| exceeds the calibration uncertainty."""

    def __str__(self) -> str:
        mark = "" if self.significant else " (dentro del error de calibración)"
        return (
            f"{self.key}: {self.measured:.1f} vs {self.reference:.1f} cm⁻¹ → "
            f"Δ = {self.delta:+.1f} cm⁻¹{mark}"
        )


@dataclass
class StrainDopingDecomposition:
    """Separation of a (ΔG, Δ2D) pair into strain and doping components."""

    delta_g: float
    delta_2d: float
    strain_component_g: float
    """The part of ΔG attributable to strain, cm⁻¹."""
    doping_component_g: float
    """The part of ΔG attributable to doping, cm⁻¹."""
    strain_percent: Optional[float]
    """Biaxial strain in %, if a conversion rate is available. Sign: positive is tensile."""
    carrier: str
    """``"hole"``, ``"electron"`` or ``"negligible"``."""
    valid_for: str
    warnings: list[str] = field(default_factory=list)
    source: str = "Lee et al., Nature Commun. 3 (2012) 1024"

    def summary(self) -> str:
        lines = [
            f"ΔG = {self.delta_g:+.1f} cm⁻¹, Δ2D = {self.delta_2d:+.1f} cm⁻¹",
            f"  componente de deformación en G: {self.strain_component_g:+.1f} cm⁻¹",
            f"  componente de dopado en G:      {self.doping_component_g:+.1f} cm⁻¹",
        ]
        if self.strain_percent is not None:
            if abs(self.strain_percent) < 0.005:
                lines.append("  deformación biaxial ≈ despreciable (< 0.005 %)")
            else:
                kind = "tracción" if self.strain_percent > 0 else "compresión"
                lines.append(f"  deformación biaxial ≈ {self.strain_percent:+.3f} % ({kind})")
        lines.append(f"  portadores: {self.carrier}")
        lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


@dataclass
class ShiftAnalysis:
    """The complete shift report for one spectrum."""

    shifts: dict[str, BandShift]
    reference_material: str
    reference_kind: str
    interpretation: list[str]
    dopant_matches: list[tuple[DopantSignature, str]]
    decomposition: Optional[StrainDopingDecomposition]
    calibration_uncertainty: float
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Referencia: {self.reference_material} ({self.reference_kind})",
            f"Incertidumbre de calibración asumida: ±{self.calibration_uncertainty:.1f} cm⁻¹",
            "",
        ]
        lines.extend(str(s) for s in self.shifts.values())
        if self.interpretation:
            lines.append("")
            lines.extend("• " + i for i in self.interpretation)
        if self.dopant_matches:
            lines.append("")
            lines.append("Compatible con:")
            for signature, why in self.dopant_matches:
                lines.append(f"  – {signature.label}: {why}")
                lines.append(f"    ({signature.source}; confianza {signature.confidence})")
        if self.decomposition:
            lines.append("")
            lines.append(self.decomposition.summary())
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


#: Which pristine reference goes with which classified material.
REFERENCE_FOR_MATERIAL = {
    "SWCNT": "SWCNT_G_plus",
    "DWCNT": "SWCNT_G_plus",
    "MWCNT": "MWCNT_G",
    "graphene_1L": "graphene_G",
    "graphene_2L": "graphene_G",
    "FLG": "graphene_G",
    "graphite": "graphite_G",
    "GO": "graphene_G",
    "rGO": "graphene_G",
}


def analyse_shifts(
    assignment: Assignment,
    material_key: str = "graphene_1L",
    control: Optional[Assignment] = None,
    calibration_uncertainty: float = 2.0,
    id_ig: Optional[float] = None,
    control_id_ig: Optional[float] = None,
    g_fwhm: Optional[float] = None,
    control_g_fwhm: Optional[float] = None,
    db: Optional[Database] = None,
) -> ShiftAnalysis:
    """Compare measured band positions against a reference and interpret.

    Parameters
    ----------
    assignment:
        The sample's assigned bands.
    material_key:
        Which reference material the sample is (from the classifier, or
        chosen by the user). Selects the pristine G reference.
    control:
        An assignment for a control measurement — the same material before
        doping or before treatment, measured on the same instrument. When
        given, it is used as the reference instead of the literature value,
        which removes the instrument's calibration offset entirely and is
        the only way to trust a shift of a few cm⁻¹.
    calibration_uncertainty:
        How far a band may move before the shift is called real, cm⁻¹.
        2 cm⁻¹ is realistic for a well-calibrated instrument compared
        against a literature number; with a same-session control, 0.5 is
        defensible.
    id_ig, control_id_ig:
        I_D/I_G for sample and control. The discriminator between
        substitutional doping (which creates defects, so I_D/I_G rises) and
        charge-transfer doping (which does not).
    g_fwhm, control_g_fwhm:
        G-band widths, in cm⁻¹. Doping narrows the G band while heating
        broadens it, so the width separates the two causes of a G shift.
    db:
        Loaded database.

    Returns
    -------
    ShiftAnalysis
    """
    database = db or load_database()
    pristine = database.pristine

    warnings: list[str] = []
    if assignment.laser_ev is None:
        warnings.append(
            "sin longitud de onda del láser no se corrigen las posiciones de "
            "referencia por dispersión; los desplazamientos de D, 2D y D+D' "
            "no son interpretables"
        )
    warnings.append(
        "calibra el equipo en la misma sesión (p. ej. con la línea de 520.7 "
        "cm⁻¹ del silicio) antes de interpretar desplazamientos de pocos cm⁻¹: "
        "la deriva típica de un espectrómetro es de ese mismo orden"
    )
    warnings.append(
        "comprueba el efecto de la potencia del láser: unos pocos mW sobre un "
        "polvo negro lo calientan cientos de kelvin y desplazan G hacia abajo, "
        "imitando una deformación de tracción. Si al bajar la potencia la banda "
        "se mueve, el desplazamiento era térmico"
    )

    shifts: dict[str, BandShift] = {}
    reference_kind = "control" if control is not None else "literature"

    reference_name = REFERENCE_FOR_MATERIAL.get(material_key, "graphene_G")

    material = database.materials.get(material_key)

    def reference_for(key: str) -> Optional[float]:
        if control is not None:
            other = control.g_like() if key in {"G", "G+"} else control.get(key)
            return other.position if other else None
        if key in {"G", "G+"}:
            return float(pristine[reference_name])
        if key == "2D":
            # The 2D band of a nanotube is not graphene's 2D band: it sits
            # ~15 cm-1 higher and is a layer-averaged feature. Comparing a
            # nanotube against graphene's 2676 cm-1 manufactures a shift of
            # exactly that size out of nothing, so prefer the reference
            # material's own range when the database has one.
            span = material.band_window("2D") if material else None
            if span is not None:
                centre = 0.5 * (span[0] + span[1])
                band = database.bands.get("2D")
                if band is not None and assignment.laser_ev is not None:
                    centre += band.dispersion * (
                        assignment.laser_ev - database.reference_laser_ev
                    )
                return float(centre)
            return float(pristine["graphene_2D"])
        band = database.bands.get(key)
        return band.position_at(assignment.laser_ev) if band else None

    for key in ("G", "G+", "2D", "D", "D'"):
        entry = assignment.get(key)
        if entry is None:
            continue
        ref = reference_for(key)
        if ref is None:
            continue
        delta = entry.position - ref
        shifts[key] = BandShift(
            key=key,
            measured=entry.position,
            reference=ref,
            delta=delta,
            reference_kind=reference_kind,
            significant=abs(delta) > calibration_uncertainty,
        )

    g_shift = shifts.get("G") or shifts.get("G+")
    d2_shift = shifts.get("2D")

    interpretation = _interpret(
        g_shift,
        d2_shift,
        id_ig=id_ig,
        control_id_ig=control_id_ig,
        g_fwhm=g_fwhm,
        control_g_fwhm=control_g_fwhm,
        calibration_uncertainty=calibration_uncertainty,
    )

    dopant_matches = []
    if g_shift is not None and g_shift.significant:
        for signature in database.dopants.values():
            if signature.matches(g_shift.delta, d2_shift.delta if d2_shift else None):
                why = (
                    f"ΔG = {g_shift.delta:+.1f} cm⁻¹ en el rango "
                    f"[{signature.g_shift[0]:g}, {signature.g_shift[1]:g}]"
                )
                if d2_shift:
                    why += (
                        f"; Δ2D = {d2_shift.delta:+.1f} en "
                        f"[{signature.d2_shift[0]:g}, {signature.d2_shift[1]:g}]"
                    )
                else:
                    why += "; sin banda 2D no se puede fijar el signo del dopado"
                dopant_matches.append((signature, why))

    decomposition = None
    if g_shift is not None and d2_shift is not None:
        decomposition = decompose_strain_doping(
            g_shift.delta, d2_shift.delta, material_key=material_key, db=database
        )

    return ShiftAnalysis(
        shifts=shifts,
        reference_material=(
            f"{material_key} / {reference_name}" if control is None else "control del usuario"
        ),
        reference_kind=reference_kind,
        interpretation=interpretation,
        dopant_matches=dopant_matches,
        decomposition=decomposition,
        calibration_uncertainty=calibration_uncertainty,
        warnings=warnings,
    )


def _interpret(
    g_shift: Optional[BandShift],
    d2_shift: Optional[BandShift],
    id_ig: Optional[float],
    control_id_ig: Optional[float],
    g_fwhm: Optional[float],
    control_g_fwhm: Optional[float],
    calibration_uncertainty: float,
) -> list[str]:
    """Turn a pair of shifts into physical statements."""
    out: list[str] = []
    if g_shift is None:
        out.append("no se ha identificado la banda G: no hay nada que comparar")
        return out

    if not g_shift.significant:
        out.append(
            f"ΔG = {g_shift.delta:+.1f} cm⁻¹ no supera la incertidumbre de "
            f"calibración (±{calibration_uncertainty:g}): compatible con material "
            "sin perturbar"
        )
    elif g_shift.delta > 0:
        out.append(
            f"G desplazada hacia arriba ({g_shift.delta:+.1f} cm⁻¹): endurecimiento "
            "del fonón, característico de dopado (de cualquier signo) por bloqueo "
            "de Pauli de la anomalía de Kohn"
        )
    else:
        out.append(
            f"G desplazada hacia abajo ({g_shift.delta:+.1f} cm⁻¹): ablandamiento, "
            "característico de deformación de tracción o de calentamiento; el "
            "dopado casi nunca baja la G"
        )

    if d2_shift is None:
        out.append(
            "sin banda 2D no se puede determinar el SIGNO del dopado: G sube "
            "tanto con electrones como con huecos, y solo 2D los distingue "
            "(sube con huecos, baja con electrones)"
        )
    elif g_shift.significant and g_shift.delta > 0:
        if d2_shift.delta > calibration_uncertainty:
            out.append(
                f"2D también sube ({d2_shift.delta:+.1f} cm⁻¹) → dopado tipo p "
                "(huecos)"
            )
        elif d2_shift.delta < -calibration_uncertainty:
            out.append(
                f"2D baja ({d2_shift.delta:+.1f} cm⁻¹) mientras G sube → dopado "
                "tipo n (electrones)"
            )
        else:
            out.append(
                f"2D apenas se mueve ({d2_shift.delta:+.1f} cm⁻¹): dopado bajo, o "
                "una mezcla de deformación y dopado que se compensan en 2D"
            )

    if g_fwhm is not None:
        if control_g_fwhm is not None:
            change = g_fwhm - control_g_fwhm
            if change < -1.0:
                out.append(
                    f"la G se ha estrechado ({change:+.1f} cm⁻¹), lo que confirma "
                    "dopado: el calentamiento la ensancharía"
                )
            elif change > 2.0:
                out.append(
                    f"la G se ha ensanchado ({change:+.1f} cm⁻¹): apunta a "
                    "calentamiento o a más desorden, no a dopado limpio"
                )
        elif g_fwhm > 40.0 and g_shift.significant:
            out.append(
                f"la G es ancha (FWHM {g_fwhm:.0f} cm⁻¹); en material desordenado "
                "la posición de G también depende del tamaño de dominio, así que "
                "el desplazamiento no es atribuible solo a dopado"
            )

    if id_ig is not None and control_id_ig is not None:
        change = id_ig - control_id_ig
        relative = change / control_id_ig if control_id_ig > 0 else float("inf")
        if relative > 0.2:
            out.append(
                f"I_D/I_G ha subido de {control_id_ig:.2f} a {id_ig:.2f}: el "
                "tratamiento ha creado defectos en la red (dopado sustitucional, "
                "funcionalización covalente o daño), no solo transferencia de carga"
            )
        elif abs(relative) <= 0.2 and g_shift.significant:
            out.append(
                f"I_D/I_G prácticamente sin cambio ({control_id_ig:.2f} → "
                f"{id_ig:.2f}) con la G desplazada: transferencia de carga por "
                "adsorción o intercalación, sin sustitución en la red"
            )
    elif id_ig is not None and g_shift.significant:
        out.append(
            "para distinguir dopado sustitucional de transferencia de carga hace "
            "falta el I_D/I_G del material SIN tratar: el sustitucional crea "
            "defectos y sube I_D/I_G, la transferencia de carga no"
        )
    return out


def decompose_strain_doping(
    delta_g: float,
    delta_2d: float,
    material_key: str = "graphene_1L",
    db: Optional[Database] = None,
) -> StrainDopingDecomposition:
    """Split a (ΔG, Δ2D) pair into a strain part and a doping part.

    Pure strain moves the sample along a line of slope
    ``Δω_2D/Δω_G ≈ 2.2`` in the (ω_G, ω_2D) plane; pure hole doping along a
    line of slope ``≈ 0.7``. Two non-parallel directions span the plane, so
    any measured pair decomposes uniquely::

        Δ_G = Δ_G^strain + Δ_G^doping
        Δ_2D = k_s Δ_G^strain + k_d Δ_G^doping

    solved for the two components.

    Parameters
    ----------
    delta_g, delta_2d:
        Measured shifts in cm⁻¹.
    material_key:
        Used only to decide whether to warn: the calibration is for
        monolayer graphene.
    db:
        Loaded database, which holds the two slopes and their provenance.

    Returns
    -------
    StrainDopingDecomposition

    Notes
    -----
    The decomposition itself is robust — it is linear algebra on two
    measured numbers. What is *not* robust is the conversion of the strain
    component into a strain percentage, which uses a rate measured on
    graphene on a flexible substrate with imperfect strain transfer. Treat
    the percentage as an order of magnitude.
    """
    database = db or load_database()
    constants = database.strain_doping
    k_strain = float(constants["slope_strain"])
    k_doping = float(constants["slope_hole_doping"])

    warnings: list[str] = []
    if material_key not in {"graphene_1L", "graphene_2L", "FLG", "graphite"}:
        warnings.append(
            f"la descomposición deformación/dopado está calibrada para grafeno "
            f"monocapa; aplicarla a «{material_key}» da números que parecen "
            "precisos pero no lo son. Úsala aquí solo de forma cualitativa"
        )

    determinant = k_strain - k_doping
    if abs(determinant) < 1e-9:  # pragma: no cover - guards a corrupted database
        raise ValueError("the two slopes in the database are identical; cannot decompose")

    doping_g = (delta_2d - k_strain * delta_g) / (k_doping - k_strain)
    strain_g = delta_g - doping_g

    effect = database.effect("biaxial_strain")
    rate = float(effect.data["G_rate_cm1_per_percent"])
    strain_percent = strain_g / rate if rate != 0 else None

    if doping_g > 1.0:
        carrier = "compatible con dopado (G endurecida)"
    elif doping_g < -1.0:
        carrier = "componente de dopado negativa: revisa la referencia, el dopado no ablanda G"
    else:
        carrier = "negligible"

    warnings.append(
        "las pendientes (2.2 deformación, 0.7 dopado) se midieron a 514 nm en "
        "grafeno exfoliado; la DESCOMPOSICIÓN es robusta, los valores absolutos "
        "que produce no lo son"
    )

    return StrainDopingDecomposition(
        delta_g=float(delta_g),
        delta_2d=float(delta_2d),
        strain_component_g=float(strain_g),
        doping_component_g=float(doping_g),
        strain_percent=float(strain_percent) if strain_percent is not None else None,
        carrier=carrier,
        valid_for="grafeno monocapa",
        warnings=warnings,
        source=str(constants.get("source", "")),
    )


__all__ = [
    "BandShift",
    "REFERENCE_FOR_MATERIAL",
    "ShiftAnalysis",
    "StrainDopingDecomposition",
    "analyse_shifts",
    "decompose_strain_doping",
]
