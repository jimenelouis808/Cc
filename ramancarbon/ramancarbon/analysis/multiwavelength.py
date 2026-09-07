"""Combining measurements of the same sample at two or more excitations.

Measuring at 532 and 633 nm is not just two chances at the same answer. It
makes three things possible that a single wavelength cannot do at all:

**Telling a real band from an impostor.** The D band is a double-resonance
feature: its position depends on the laser energy, moving about 50 cm⁻¹/eV.
Almost nothing else in the spectrum does. A narrow line near 1332 cm⁻¹ that
does not move is diamond, not D. A line at 225 or 292 cm⁻¹ that does not
move is not a radial breathing mode. With one laser these are arguments;
with two they are measurements.

**Detecting amorphous carbon.** The G band does **not** disperse in
graphite or in nanocrystalline graphite — its dispersion is zero, because
it is a zone-centre mode. It only starts to disperse when the material
contains sp² configurations of varying bond length and order, i.e. amorphous
carbon, reaching 6–10 cm⁻¹/eV. So a measured G dispersion is close to a
direct assay for the amorphous fraction, and it is invisible to any
single-wavelength measurement.

**Checking that I_D/I_G is being used correctly.** The ratio scales as λ⁴,
so 633 nm gives a value roughly 2.3 times the 532 nm one on the *same
sample*. The crystallite sizes derived from the two must agree; if they do
not, either the sampled spots differ, or the ratios were taken on different
bases, or something is resonant.

This module does not need the spectra to be of the same spot — but it is far
more informative if they are, and it says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from ..core.spectrum import laser_energy_ev
from ..database import Database, load_database

if TYPE_CHECKING:  # pragma: no cover
    from .report import AnalysisResult

#: Expected G-band dispersion, cm⁻¹/eV, by material class.
#: Zero for anything graphitic; non-zero means amorphous sp² is present.
G_DISPERSION_REFERENCE = {
    "graphite": (0.0, 1.0),
    "nanocrystalline": (0.0, 2.0),
    "amorphous": (6.0, 12.0),
    "tetrahedral": (12.0, 25.0),
}


@dataclass
class BandDispersion:
    """How one band moved between excitations."""

    key: str
    positions: dict[float, float]
    """Fitted position in cm⁻¹, keyed by laser wavelength in nm."""
    slope: Optional[float]
    """Measured dω/dE_laser, cm⁻¹/eV. Positive means the band moves up with
    increasing photon energy."""
    expected: Optional[float]
    """What the database says this band should do."""
    residual: Optional[float]
    """Measured minus expected."""
    consistent: bool
    verdict: str = ""

    def __str__(self) -> str:
        if self.slope is None:
            return f"{self.key}: medida en un solo láser, no hay dispersión que medir"
        measured = f"{self.slope:+.1f} cm⁻¹/eV"
        expected = f"{self.expected:+.1f}" if self.expected is not None else "—"
        mark = "✓" if self.consistent else "✗"
        return f"{mark} {self.key}: medida {measured}, esperada {expected} cm⁻¹/eV"


@dataclass
class MultiWavelengthResult:
    """Everything the combination of excitations yielded."""

    lasers: list[float]
    dispersions: dict[str, BandDispersion]
    g_dispersion_verdict: str = ""
    crystallite_sizes: dict[float, float] = field(default_factory=dict)
    crystallite_verdict: str = ""
    impostors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Excitaciones combinadas: "
            + ", ".join(f"{v:g} nm ({laser_energy_ev(v):.2f} eV)" for v in self.lasers),
            "",
            "Dispersión de las bandas:",
        ]
        lines.extend("  " + str(d) for d in self.dispersions.values())
        if self.g_dispersion_verdict:
            lines.append("")
            lines.append(self.g_dispersion_verdict)
        if self.crystallite_sizes:
            lines.append("")
            lines.append("Tamaño de cristalito por excitación:")
            for laser, value in sorted(self.crystallite_sizes.items()):
                lines.append(f"  {laser:g} nm → L_a = {value:.1f} nm")
            lines.append("  " + self.crystallite_verdict)
        if self.impostors:
            lines.append("")
            lines.append("Bandas que NO se desplazan (no son de doble resonancia):")
            lines.extend("  " + text for text in self.impostors)
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def measure_dispersion(
    positions: dict[float, float],
) -> tuple[Optional[float], Optional[float]]:
    """Least-squares slope of position against photon energy.

    Parameters
    ----------
    positions:
        ``{laser_nm: position_cm1}``, at least two entries.

    Returns
    -------
    (float or None, float or None)
        Slope in cm⁻¹/eV and its standard error. With exactly two points
        the slope is exact and the error is ``None`` — there is no spare
        degree of freedom, which is worth being explicit about rather than
        reporting a zero uncertainty.
    """
    if len(positions) < 2:
        return None, None
    energies = np.array([laser_energy_ev(v) for v in positions], dtype=float)
    values = np.array(list(positions.values()), dtype=float)
    if np.ptp(energies) < 1e-6:
        return None, None
    slope, _ = np.polyfit(energies, values, 1)
    if len(positions) == 2:
        return float(slope), None
    fit = np.polyval(np.polyfit(energies, values, 1), energies)
    residual = values - fit
    dof = len(positions) - 2
    variance = float(np.sum(residual**2)) / dof
    spread = float(np.sum((energies - energies.mean()) ** 2))
    return float(slope), float(np.sqrt(variance / spread)) if spread > 0 else None


def compare_excitations(
    results: Sequence["AnalysisResult"],
    tolerance: float = 15.0,
    db: Optional[Database] = None,
) -> MultiWavelengthResult:
    """Combine analyses of the same sample taken at different lasers.

    Parameters
    ----------
    results:
        Two or more :class:`~ramancarbon.analysis.report.AnalysisResult`,
        each with a known and distinct excitation wavelength.
    tolerance:
        How far, in cm⁻¹/eV, a measured dispersion may sit from the
        database value and still count as consistent. The default 15 is
        deliberately generous: with only two lasers the slope is a
        two-point estimate, and a 2 cm⁻¹ error on either position becomes
        ~5 cm⁻¹/eV between 532 and 633 nm.
    db:
        Loaded database.

    Returns
    -------
    MultiWavelengthResult

    Raises
    ------
    ValueError
        If fewer than two results carry a laser wavelength, or they all
        share the same one.
    """
    database = db or load_database()
    usable = [r for r in results if r.raw.laser_nm is not None]
    lasers = sorted({float(r.raw.laser_nm) for r in usable})
    if len(usable) < 2 or len(lasers) < 2:
        raise ValueError(
            "hacen falta al menos dos análisis con longitudes de onda de "
            "excitación distintas y conocidas"
        )

    warnings: list[str] = []
    if len(lasers) == 2 and abs(
        laser_energy_ev(lasers[0]) - laser_energy_ev(lasers[1])
    ) < 0.3:
        warnings.append(
            f"las dos excitaciones ({lasers[0]:g} y {lasers[1]:g} nm) difieren en "
            "poca energía; la pendiente medida amplifica cualquier error de "
            "calibración. Con 532 y 633 nm la separación es de 0.37 eV, que es "
            "suficiente pero no holgada"
        )

    dispersions: dict[str, BandDispersion] = {}
    impostors: list[str] = []
    for key in ("D", "G", "G+", "D'", "2D", "D+D'", "D+D''"):
        positions: dict[float, float] = {}
        for result in usable:
            entry = result.assignment.get(key)
            if entry is not None:
                positions[float(result.raw.laser_nm)] = entry.position
        if len(positions) < 2:
            continue
        slope, _ = measure_dispersion(positions)
        band = database.bands.get(key)
        expected = band.dispersion if band else None
        residual = None if expected is None or slope is None else slope - expected
        consistent = residual is None or abs(residual) <= tolerance
        entry = BandDispersion(
            key=key,
            positions=positions,
            slope=slope,
            expected=expected,
            residual=residual,
            consistent=consistent,
        )
        entry.verdict = _verdict_for(entry, tolerance)
        dispersions[key] = entry
        if (
            slope is not None
            and expected is not None
            and abs(expected) > 20.0
            and abs(slope) < 10.0
        ):
            impostors.append(
                f"{key}: se esperaba {expected:+.0f} cm⁻¹/eV y se mide "
                f"{slope:+.1f}. Una banda de doble resonancia SIEMPRE se "
                "desplaza; si esta no lo hace, no es la banda que se ha "
                "asignado. Candidatos: diamante (1332), línea de un óxido, o "
                "una banda del sustrato"
            )

    g_verdict = _interpret_g_dispersion(dispersions)
    sizes, size_verdict = _crystallite_consistency(usable)

    return MultiWavelengthResult(
        lasers=lasers,
        dispersions=dispersions,
        g_dispersion_verdict=g_verdict,
        crystallite_sizes=sizes,
        crystallite_verdict=size_verdict,
        impostors=impostors,
        warnings=warnings,
    )


def _verdict_for(entry: BandDispersion, tolerance: float) -> str:
    if entry.slope is None or entry.expected is None:
        return ""
    if entry.consistent:
        return (
            f"compatible con la asignación de {entry.key} como banda de doble "
            "resonancia"
        )
    return (
        f"la dispersión medida ({entry.slope:+.1f} cm⁻¹/eV) no encaja con "
        f"{entry.key} ({entry.expected:+.1f} ± {tolerance:g}). Revisa la "
        "asignación o la calibración del eje"
    )


def _interpret_g_dispersion(dispersions: dict[str, BandDispersion]) -> str:
    """The G-band dispersion as an amorphous-carbon assay."""
    entry = dispersions.get("G") or dispersions.get("G+")
    if entry is None or entry.slope is None:
        return ""
    slope = entry.slope
    head = f"Dispersión de la banda G: {slope:+.1f} cm⁻¹/eV. "
    if abs(slope) <= 2.0:
        return head + (
            "Compatible con cero, que es lo que hace la G en grafito y en "
            "grafito nanocristalino: es un modo del centro de zona y no "
            "depende de la energía del láser. No hay indicio de carbono "
            "amorfo por esta vía."
        )
    if abs(slope) <= 6.0:
        return head + (
            "Ligeramente distinta de cero. En material puramente grafítico la "
            "G no dispersa, así que esto sugiere algo de carbono amorfo — pero "
            "está dentro de lo que puede producir un error de calibración de "
            "2 cm⁻¹ entre las dos medidas. Calibra con silicio y repite antes "
            "de concluir."
        )
    return head + (
        "Claramente distinta de cero. La banda G solo dispersa cuando hay "
        "configuraciones sp² con longitudes de enlace variadas, es decir "
        "carbono amorfo. Esto es un indicio de fase amorfa que NINGUNA medida "
        "a un solo láser puede dar, porque a una sola energía la posición de G "
        "también se mueve por dopado y por deformación."
    )


def _crystallite_consistency(
    results: Sequence["AnalysisResult"],
) -> tuple[dict[float, float], str]:
    """Do the two excitations agree on the crystallite size?"""
    sizes: dict[float, float] = {}
    for result in results:
        if result.crystallite and result.crystallite.la_low_defect_nm:
            sizes[float(result.raw.laser_nm)] = result.crystallite.la_low_defect_nm
    if len(sizes) < 2:
        return sizes, ""
    values = list(sizes.values())
    spread = (max(values) - min(values)) / max(np.mean(values), 1e-9)
    if spread <= 0.25:
        return sizes, (
            f"Los dos valores coinciden dentro del {spread * 100:.0f} %, lo que "
            "confirma que la corrección λ⁴ se ha aplicado bien y que ambas "
            "medidas ven la misma estructura."
        )
    return sizes, (
        f"Discrepan un {spread * 100:.0f} %. I_D/I_G escala como λ⁴, así que L_a "
        "debería salir igual desde las dos excitaciones. Que no lo haga apunta a: "
        "puntos distintos de una muestra heterogénea (lo más probable), cocientes "
        "tomados con bases distintas (áreas frente a alturas), o efectos de "
        "resonancia. Promedia sobre varios puntos antes de dar un número."
    )


__all__ = [
    "BandDispersion",
    "G_DISPERSION_REFERENCE",
    "MultiWavelengthResult",
    "compare_excitations",
    "measure_dispersion",
]
