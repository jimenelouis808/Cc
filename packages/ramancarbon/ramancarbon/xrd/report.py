"""One call from a file to an answer, and the written report.

:func:`analyse_pattern` runs the whole diffraction path — peak search,
phase identification, and optionally a Rietveld refinement of the phases
it found — and returns everything the GUI and the CLI need, including the
calculated curve and the difference curve.

The report is written in the order the numbers should be *read*, which is
not the order they are computed. The difference curve comes before the R
factors, because an R factor is a single number summarising thousands of
points and the difference curve says where the model fails; a refinement
with a fine Rwp and a systematic wave in the residual is worse than one
with a poor Rwp and structureless noise. The unexplained peaks come before
the weight fractions, because a fraction computed without a phase that is
present is wrong by an amount nothing else in the output reveals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .pattern import Pattern
from .rietveld import PhaseModel, RietveldResult, auto_refine
from .search import PhaseSearchResult, XRDPeak, find_peaks, identify_phases
from .structure import Crystal


@dataclass
class XRDResult:
    """Everything the diffraction analysis of one pattern produced."""

    pattern: Pattern
    peaks: list[XRDPeak]
    search: PhaseSearchResult
    refinement: Optional[RietveldResult] = None
    warnings: list[str] = field(default_factory=list)

    @property
    def calculated(self) -> Optional[np.ndarray]:
        """The theoretical pattern of the identified phases, or ``None``."""
        return None if self.refinement is None else self.refinement.calculated

    @property
    def difference(self) -> Optional[np.ndarray]:
        return None if self.refinement is None else self.refinement.difference

    def to_dict(self) -> dict:
        """Flat row for a batch table."""
        row: dict = {
            "nombre": self.pattern.name,
            "lambda_A": self.pattern.wavelength,
            "n_picos": len(self.peaks),
            "fases": ";".join(m.crystal.name for m in self.search.accepted),
            "sin_explicar": len(self.search.unexplained),
            "cero_deg": self.search.zero_shift,
        }
        if self.refinement is not None:
            row["Rwp_pct"] = 100.0 * self.refinement.r_wp
            row["Rp_pct"] = 100.0 * self.refinement.r_p
            row["GOF"] = self.refinement.gof
            for name, fraction in self.refinement.weight_fractions().items():
                if fraction is not None:
                    row[f"w_{name}"] = 100.0 * fraction
            for name, size in self.refinement.crystallite_sizes().items():
                if size is not None:
                    row[f"D_{name}_nm"] = size
        return row

    def report(self, verbose: bool = True) -> str:
        return build_report(self, verbose=verbose)


def analyse_pattern(
    pattern: Pattern,
    candidates: Optional[Sequence[Crystal]] = None,
    extra_directories: Optional[Sequence[str]] = None,
    refine: bool = True,
    max_phases: int = 4,
    preferred_axis: Optional[Sequence[int]] = None,
    instrument_fwhm: float = 0.06,
) -> XRDResult:
    """Identify the phases in a diffractogram and, optionally, refine them.

    Parameters
    ----------
    pattern:
        The measured pattern, **not** background-subtracted: the
        refinement models the background, and subtracting it first
        destroys the counting statistics the weighting needs.
    candidates:
        Structures to test. Defaults to the reference library plus
        ``extra_directories``.
    refine:
        Run a staged Rietveld refinement of the identified phases.
    preferred_axis:
        Texture axis, e.g. ``(0, 0, 1)`` for a layered material. Given
        one, the refinement frees a March–Dollase parameter; without one
        it cannot, and a textured sample then shows up as a stubborn
        intensity misfit that U_iso will try to absorb.

    Returns
    -------
    XRDResult
    """
    peaks = find_peaks(pattern)
    search = identify_phases(
        pattern,
        candidates=candidates,
        max_phases=max_phases,
        extra_directories=extra_directories,
        peaks=peaks,
    )
    result = XRDResult(pattern=pattern, peaks=peaks, search=search)
    result.warnings.extend(search.warnings)

    if refine and search.accepted:
        models = [PhaseModel(crystal=m.crystal) for m in search.accepted]
        result.refinement = auto_refine(
            pattern,
            models,
            preferred_axis=preferred_axis,
            instrument_fwhm=instrument_fwhm,
        )
        result.warnings.extend(result.refinement.warnings)
    elif refine:
        result.warnings.append(
            "no se ha refinado porque no se ha identificado ninguna fase. "
            "Rietveld ajusta un modelo, no lo descubre: primero hace falta "
            "saber qué fases hay"
        )
    return result


def theoretical_pattern(
    pattern: Pattern,
    phases: Sequence[Crystal],
    preferred_axis: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray, RietveldResult]:
    """Calculated pattern and difference for a given set of phases.

    A thin wrapper on the refinement for the common request "show me what
    these phases would look like on my data". The scales, background and
    profile are refined because they must be — a calculated pattern on an
    arbitrary vertical scale cannot be compared to anything — while the
    structures stay as given.

    Returns ``(calculated, difference, refinement)``.
    """
    refinement = auto_refine(
        pattern, [PhaseModel(crystal=c) for c in phases], preferred_axis=preferred_axis
    )
    return refinement.calculated, refinement.difference, refinement


def build_report(result: XRDResult, verbose: bool = True) -> str:
    """The written report, ordered the way the numbers should be read."""

    def section(title: str) -> str:
        return "\n" + "─" * 72 + f"\n  {title}\n" + "─" * 72

    lines = ["═" * 72, f"  DIFRACCIÓN DE RAYOS X — {result.pattern.name}", "═" * 72]
    lines.append(result.pattern.describe())

    lines.append(section("IDENTIFICACIÓN DE FASES"))
    lines.append(result.search.summary())

    if result.refinement is not None:
        refinement = result.refinement
        lines.append(section("AJUSTE DE RIETVELD"))
        lines.append(
            "Mira PRIMERO la curva diferencia. Los factores R resumen miles de "
            "puntos en un número y no dicen dónde falla el modelo; un ajuste "
            "con Rwp bonito y una ondulación sistemática en la diferencia es "
            "peor que uno con Rwp feo y diferencia sin estructura."
        )
        lines.append("")
        difference = refinement.difference
        weighted = difference / np.maximum(result.pattern.sigma, 1e-9)
        lines.append(
            f"Diferencia: máxima {np.max(np.abs(difference)):.0f} cuentas "
            f"({np.max(np.abs(weighted)):.1f} σ), media "
            f"{np.mean(np.abs(weighted)):.2f} σ"
        )
        runs = _runs_statistic(weighted)
        lines.append(
            f"Rachas de signo en el residuo: {runs:.2f} veces lo esperado por "
            "azar"
            + (
                "  — hay estructura sistemática sin modelar"
                if runs < 0.7
                else "  — sin estructura evidente"
            )
        )
        lines.append("")
        lines.append(refinement.summary())

    if result.warnings and verbose:
        lines.append(section("AVISOS"))
        seen: set[str] = set()
        for warning in result.warnings:
            if warning in seen:
                continue
            seen.add(warning)
            lines.append("⚠ " + warning)
    return "\n".join(lines)


def _runs_statistic(weighted_residual: np.ndarray) -> float:
    """Observed sign runs over the number expected from independent noise.

    A value near 1 means the residual changes sign as often as noise
    would. A value well below 1 means long stretches of one sign — a peak
    the model puts in the wrong place, a background that is too stiff, a
    missing phase — which is precisely the failure that a small Rwp can
    hide.
    """
    signs = np.sign(np.asarray(weighted_residual, dtype=float))
    signs = signs[signs != 0]
    if signs.size < 20:
        return 1.0
    runs = 1 + int(np.count_nonzero(np.diff(signs) != 0))
    positive = int(np.count_nonzero(signs > 0))
    negative = signs.size - positive
    if positive == 0 or negative == 0:
        return 0.0
    expected = 1.0 + 2.0 * positive * negative / signs.size
    return float(runs / expected)


__all__ = [
    "XRDResult",
    "analyse_pattern",
    "build_report",
    "theoretical_pattern",
]
