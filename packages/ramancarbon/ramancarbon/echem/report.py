"""One call from files to an answer, and the written report.

The order of the report is the order the numbers should be *read*, and
that is not the order they are computed. The mechanism comes first,
because it decides whether the capacitance below is even the right
quantity to quote. The Kramers–Kronig verdict comes before the circuit
parameters, because a circuit fitted to inconsistent data has none. And
the iR-correction status comes before the Tafel slope, because without it
the slope is not a kinetic quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .capacitance import (
    CapacitanceComparison,
    ComplexCapacitance,
    capacitance_from_eis,
    complex_capacitance,
    compare as compare_capacitance,
    specific,
)
from .curve import ChargeDischarge, Impedance, Voltammogram
from .cv import CVResult, DunnAnalysis, RateStudy, analyse_cv, analyse_rate_study, dunn_analysis
from .eis import DRTResult, EISResult, analyse_eis, drt
from .evaluate import CatalysisResult, StorageVerdict, analyse_catalysis, classify_storage
from .curve import CurveError
from .gcd import GCDResult, analyse_gcd


@dataclass
class EchemResult:
    """Everything the electrochemical analysis of one sample produced."""

    name: str = "muestra"
    cv: Optional[CVResult] = None
    rates: Optional[RateStudy] = None
    gcd: Optional[GCDResult] = None
    eis: Optional[EISResult] = None
    catalysis: Optional[CatalysisResult] = None
    storage: Optional[StorageVerdict] = None
    electrode: Optional[Any] = None
    """The cell it was all measured on. Kept so the report can state the
    reference scale and what it converts to: two papers on the same
    material quoting potentials against Ag/AgCl saturated and Ag/AgCl 3 M
    differ by 13 mV before anything physical happens."""
    dunn: Optional[DunnAnalysis] = None
    capacitance: Optional[CapacitanceComparison] = None
    """The same capacitance by every method that was measured. It is the
    disagreement between them that is informative, not any one of them."""
    complex_capacitance: Optional[ComplexCapacitance] = None
    """C(ω) = 1/(jωZ), straight from the data. A circuit is a
    hypothesis; this is not one, and its C″ maximum gives τ₀ without
    needing the mass."""
    drt: Optional[DRTResult] = None
    """The distribution of relaxation times. Model-free in the sense that
    it assumes no circuit, but NOT assumption-free: the regularisation is
    a choice about how much structure to believe, and it is reported."""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Flat row for a batch table."""
        row: dict[str, Any] = {"nombre": self.name}
        if self.storage:
            row["mecanismo"] = self.storage.mechanism
            row["confianza_mecanismo"] = self.storage.confidence
        if self.cv:
            row["C_cv_F"] = self.cv.capacitance_loop.farads if self.cv.capacitance_loop else None
            if self.cv.capacitance_loop:
                row["C_cv_F_g"] = self.cv.capacitance_loop.specific_f_per_g
            row["q_cv_C_g"] = self.cv.capacity_c_per_g
            row["dEp_V"] = self.cv.peak_separation
        if self.gcd:
            row["C_gcd_F"] = self.gcd.capacitance_f
            row["C_gcd_F_g"] = self.gcd.specific_f_per_g
            row["q_gcd_mAh_g"] = self.gcd.capacity_mah_per_g
            row["eficiencia_culombica"] = self.gcd.coulombic_efficiency
            row["E_Wh_kg"] = self.gcd.energy_wh_per_kg
            row["P_W_kg"] = self.gcd.power_w_per_kg
            row["R_IR_ohm"] = self.gcd.resistance_ohm
        if self.dunn and self.dunn.fractions:
            fastest = max(self.dunn.fractions)
            slowest = min(self.dunn.fractions)
            row["capacitivo_rapido"] = self.dunn.fractions[fastest]
            row["capacitivo_lento"] = self.dunn.fractions[slowest]
        if self.capacitance:
            for entry in self.capacitance.entries:
                row[f"C_{entry.method}_F"] = entry.farads
                if entry.per_gram is not None:
                    row[f"C_{entry.method}_F_g"] = entry.per_gram
            row["dispersion_capacitancia"] = self.capacitance.spread
        if self.rates:
            row["ECSA_cm2"] = self.rates.ecsa_cm2
            row["Cdl_F"] = self.rates.cdl_farads
            if self.rates.b_values:
                row["b_min"] = min(v.b for v in self.rates.b_values)
        if self.eis:
            row["Rs_ohm"] = self.eis.series_resistance
            row["Rct_ohm"] = self.eis.charge_transfer_resistance
            if self.eis.kk:
                row["KK_ok"] = self.eis.kk.passes
        if self.catalysis:
            row["eta10_mV"] = (
                1e3 * self.catalysis.overpotential_at_benchmark
                if self.catalysis.overpotential_at_benchmark is not None
                else None
            )
            if self.catalysis.tafel and self.catalysis.tafel.valid:
                row["tafel_mV_dec"] = self.catalysis.tafel.slope_mv_per_decade
        return row

    def report(self, verbose: bool = True) -> str:
        return build_report(self, verbose=verbose)


def analyse_sample(
    name: str = "muestra",
    cv: Optional[Voltammogram] = None,
    rate_series: Optional[Sequence[Voltammogram]] = None,
    gcd: Optional[ChargeDischarge] = None,
    eis: Optional[Impedance] = None,
    catalysis_curve: Optional[Voltammogram] = None,
    reaction: str = "OER",
    circuit: Optional[str] = "randles_cpe",
    circuit_initial: Optional[dict[str, float]] = None,
    circuit_fixed: Optional[Sequence[str]] = None,
    non_faradaic: bool = False,
    drt_regularisation: Optional[float] = None,
) -> EchemResult:
    """Analyse whatever measurements are available for one electrode.

    Every argument is optional; the mechanism classification uses whichever
    of the three storage measurements were given and says how confident
    that makes it. Passing more raises the confidence rather than changing
    the criteria.
    """
    result = EchemResult(name=name)
    for measurement in (cv, gcd, eis, catalysis_curve,
                        *(rate_series or ())):
        if measurement is not None and getattr(measurement, "electrode", None):
            result.electrode = measurement.electrode
            break
    if cv is not None:
        result.cv = analyse_cv(cv)
        result.warnings.extend(f"CV: {w}" for w in result.cv.warnings)
    if rate_series:
        result.rates = analyse_rate_study(rate_series, non_faradaic=non_faradaic)
        result.warnings.extend(f"velocidades: {w}" for w in result.rates.warnings)
    if gcd is not None:
        result.gcd = analyse_gcd(gcd)
        result.warnings.extend(f"GCD: {w}" for w in result.gcd.warnings)
    if eis is not None:
        result.eis = analyse_eis(eis, circuit=circuit, initial=circuit_initial,
                                 fixed=circuit_fixed)
        result.warnings.extend(f"EIS: {w}" for w in result.eis.warnings)
        # Both of these are model-free, so they are computed whenever
        # there is a spectrum rather than being a separate action: a
        # circuit is a hypothesis, and the user should be able to see
        # what the data say before choosing one.
        try:
            result.complex_capacitance = complex_capacitance(eis)
            result.warnings.extend(f"C(ω): {w}"
                                   for w in result.complex_capacitance.warnings)
        except (CurveError, ValueError) as error:
            result.warnings.append(f"C(ω): {error}")
        try:
            result.drt = drt(eis, regularisation=drt_regularisation)
            result.warnings.extend(f"DRT: {w}" for w in result.drt.warnings)
        except (CurveError, ValueError) as error:
            result.warnings.append(f"DRT: {error}")
    if catalysis_curve is not None:
        result.catalysis = analyse_catalysis(
            catalysis_curve,
            reaction=reaction,
            ecsa_cm2=result.rates.ecsa_cm2 if result.rates else None,
        )
        result.warnings.extend(f"{reaction}: {w}" for w in result.catalysis.warnings)

    if rate_series and len(rate_series) >= 3:
        try:
            result.dunn = dunn_analysis(rate_series)
            result.warnings.extend(f"Dunn: {w}" for w in result.dunn.warnings)
        except CurveError as error:
            result.warnings.append(f"Dunn: {error}")

    result.capacitance = _capacitance_comparison(result, cv, gcd, eis)
    if result.capacitance:
        result.warnings.extend(f"capacitancia: {w}"
                               for w in result.capacitance.warnings)

    if result.cv or result.gcd or result.rates:
        result.storage = classify_storage(result.cv, result.gcd, result.rates)
        _cross_check(result)
    return result


def _capacitance_comparison(
    result: EchemResult,
    cv: Optional[Voltammogram],
    gcd: Optional[ChargeDischarge],
    eis: Optional[Impedance],
) -> Optional[CapacitanceComparison]:
    """Collect whatever capacitances the measurements support.

    Only measurements that were actually made contribute, and each carries
    the condition it was measured at — a scan rate, a current, a frequency.
    A capacitance without its condition cannot be compared with anybody
    else's, including the same electrode measured twice.
    """
    entries = []
    if cv is not None and result.cv and result.cv.capacitance_loop:
        entries.append(specific(
            result.cv.capacitance_loop.farads, cv.electrode, "CV",
            f"{1e3 * cv.scan_rate:.0f} mV/s",
        ))
    if gcd is not None and result.gcd and result.gcd.capacitance_f:
        current = getattr(result.gcd, "current_a", None)
        condition = (f"{1e3 * abs(current):.2f} mA" if current
                     else "descarga galvanostática")
        entries.append(specific(result.gcd.capacitance_f, gcd.electrode, "GCD",
                                condition))
    if eis is not None:
        try:
            entries.append(capacitance_from_eis(eis))
        except CurveError:
            pass
    if not entries:
        return None
    return compare_capacitance(entries)


def _cross_check(result: EchemResult) -> None:
    """Compare what the different measurements say about the same thing.

    Two capacitances of the same electrode that disagree by more than a
    quarter are the most useful disagreement in the whole analysis, and
    each direction has a specific cause worth naming.
    """
    verdict = result.storage
    if verdict is not None and not verdict.farads_are_appropriate:
        if result.cv and result.cv.capacitance_loop:
            result.warnings.append(
                "se ha calculado una capacitancia en F/g y el mecanismo es de "
                "tipo BATERÍA. Ese número está en el informe porque el "
                "programa lo calcula siempre, no porque deba citarse: para "
                "este electrodo la magnitud correcta es la capacidad, en C/g "
                "o mAh/g, que también está"
            )

    cv_value = (
        result.cv.capacitance_loop.farads
        if result.cv and result.cv.capacitance_loop
        else None
    )
    gcd_value = result.gcd.capacitance_f if result.gcd else None
    if cv_value and gcd_value:
        ratio = cv_value / gcd_value
        if ratio > 1.25:
            result.warnings.append(
                f"la capacitancia del voltamperograma es {ratio:.2f} veces la "
                "de la curva galvanostática. La integral del CV recoge TODA la "
                "corriente, incluida la faradaica de los picos y cualquier "
                "reacción parásita; la galvanostática mide sólo la carga que "
                "se devuelve. La diferencia entre las dos es una medida de "
                "cuánta de esa corriente no es almacenamiento reversible"
            )
        elif ratio < 0.8:
            result.warnings.append(
                f"la capacitancia del voltamperograma es sólo {ratio:.2f} veces "
                "la galvanostática. Suele significar que la ventana del CV era "
                "más estrecha, o que la velocidad de barrido era demasiado "
                "alta para que el electrodo respondiera entero"
            )

    if result.eis and result.gcd and result.gcd.resistance_ohm:
        series = result.eis.series_resistance
        if series:
            ratio = result.gcd.resistance_ohm / series
            if ratio > 3.0 or ratio < 0.33:
                result.warnings.append(
                    f"la resistencia de la caída IR ({result.gcd.resistance_ohm:.3g} Ω) "
                    f"y la de alta frecuencia de la impedancia ({series:.3g} Ω) "
                    "difieren en más de un factor 3. La primera incluye "
                    "resistencias que a alta frecuencia se cortocircuitan (la "
                    "de contacto, la del propio material), así que salga mayor "
                    "es normal; salga menor, no, y entonces alguna de las dos "
                    "medidas está mal escalada"
                )


def build_report(result: EchemResult, verbose: bool = True) -> str:
    """The written report."""

    def section(title: str) -> str:
        return "\n" + "─" * 72 + f"\n  {title}\n" + "─" * 72

    lines = ["═" * 72, f"  ELECTROQUÍMICA — {result.name}", "═" * 72]

    if result.electrode is not None:
        lines.append(section("LA CELDA"))
        lines.append(result.electrode.describe())
        conversions = []
        for target in ("SHE", "RHE"):
            values, why = result.electrode.to_reference([0.0], target)
            conversions.append(f"  → {target}: {why}" if values is not None
                               else f"  → {target}: no se puede — {why}")
        lines.extend(conversions)
        lines.append(
            "  Di siempre el relleno de la referencia: Ag/AgCl 3 M y "
            "saturado están a 13 mV, y el SCE a 31 mV del de 3 M."
        )

    if result.storage:
        lines.append(section("MECANISMO DE ALMACENAMIENTO"))
        lines.append(
            "Esto va primero porque decide si las magnitudes de abajo son las "
            "correctas para este material."
        )
        lines.append("")
        lines.append(result.storage.summary())

    if result.cv:
        lines.append(section("VOLTAMPEROMETRÍA CÍCLICA"))
        lines.append(result.cv.summary())
    if result.rates:
        lines.append(section("ESTUDIO DE VELOCIDAD"))
        lines.append(result.rates.summary())
    if result.dunn:
        lines.append(section("SEPARACIÓN CAPACITIVO / DIFUSIVO (DUNN)"))
        lines.append(result.dunn.summary())
    if result.gcd:
        lines.append(section("CARGA-DESCARGA GALVANOSTÁTICA"))
        lines.append(result.gcd.summary())
    if result.eis:
        lines.append(section("IMPEDANCIA"))
        lines.append(result.eis.summary())
    if result.catalysis:
        lines.append(section(f"ELECTROCATÁLISIS ({result.catalysis.reaction})"))
        lines.append(result.catalysis.summary())
    if result.capacitance and len(result.capacitance.entries) > 1:
        lines.append(section("LA MISMA CAPACITANCIA, POR CADA MÉTODO"))
        lines.append(result.capacitance.summary())

    if result.warnings and verbose:
        lines.append(section("AVISOS"))
        seen: set[str] = set()
        for warning in result.warnings:
            if warning in seen:
                continue
            seen.add(warning)
            lines.append("⚠ " + warning)
    return "\n".join(lines)


__all__ = ["EchemResult", "analyse_sample", "build_report"]
