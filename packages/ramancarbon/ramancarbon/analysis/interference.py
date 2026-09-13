"""Bands that are not carbon: catalyst oxides, dopant residue, substrate.

**Off by default.** Nothing in the standard analysis calls this. It exists
for samples where you already suspect the spectrum contains something other
than the carbon you are measuring, and it costs a little honesty to run:
telling a peak finder in advance what it might find biases what it reports.

When it is worth switching on: as-synthesised CVD material still carrying
its catalyst, samples doped from a sulfur, selenium or phosphorus precursor,
and anything measured through a substrate.

The failure it prevents is specific. Iron oxide has strong lines at 225 and
293 cm⁻¹, cobalt oxide at 194, elemental sulfur at 219, trigonal selenium
at 237, anatase at 144 — all inside the radial-breathing-mode window. Fed
to ``ω = A/d + B`` any of them yields a confident, entirely fictitious tube
diameter. Sulfur is the sharpest case: in a sulfur-doped sample, that
219 cm⁻¹ line means the dopant did **not** enter the lattice, which is the
opposite of what its presence is usually taken to show.

**A single matching line is not evidence.** The first version of this module
matched each peak against every catalogued line within 8 cm⁻¹ and, run on a
clean single-walled sample, threw away four of its five genuine radial
breathing modes — calling one cobalt oxide and another selenium. The
arithmetic explains it: the 100–400 cm⁻¹ window holds a dozen catalogued
lines, and a ±8 cm⁻¹ window around each covers more than half of it, so a
coincidence is likelier than not.

The fix is what the physics already says: a crystalline phase has a
*spectrum*, not a line. A species with several strong lines is only reported
when **several of them are present**, in the right places, together. One
line out of three is a coincidence; three out of three is hematite. Species
that genuinely have only one strong line in range are matched on it but
reported at lower confidence, and never used to exclude anything.

A matched peak is never deleted. It is reported, with what it probably is,
and — only when the identification is corroborated — excluded from the
diameter analysis. Deleting would hide evidence about the sample; keeping it
as an RBM would manufacture a result.

Every position here is non-dispersive, so two lasers settle any doubt: a
real D band moves ~50 cm⁻¹/eV and an oxide line does not move at all. See
:mod:`ramancarbon.analysis.multiwavelength`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from ..core.peaks import PeakMeasurement
from ..database.loader import DATA_DIR

#: Groups considered when the caller does not narrow it down.
DEFAULT_GROUPS = ("catalyst_oxides", "dopant_residue", "substrate", "carbon_impurity")

#: Match window in cm⁻¹.
#:
#: Tighter than the 8 the data file suggests. These are first-order modes of
#: ordinary crystals, known to about a wavenumber, and the instrument is
#: calibrated to a couple more; anything further away is a different band.
DEFAULT_TOLERANCE = 4.0

#: Fraction of a species' strong lines that must be found, when more than one
#: of them lies inside the measured range, before the identification counts
#: as corroborated.
CORROBORATION = 0.6


@dataclass(frozen=True)
class Species:
    """One non-carbon material with Raman lines in the range of interest."""

    key: str
    label: str
    formula: str
    group: str
    bands: tuple[float, ...]
    strong: tuple[float, ...]
    confidence: str
    source: str
    notes: str


@dataclass
class Match:
    """An observed peak identified as something other than carbon."""

    peak: PeakMeasurement
    species: Species
    line: float
    """The catalogue line it matched."""
    offset: float
    """Observed minus catalogue position, cm⁻¹."""
    is_strong_line: bool
    in_rbm_window: bool
    corroborated: bool = False
    """Whether enough of this species' other lines were also found."""
    lines_found: int = 1
    lines_expected: int = 1

    def __str__(self) -> str:
        if not self.corroborated:
            mark = "?"
        elif self.in_rbm_window:
            mark = "‼"
        else:
            mark = "·"
        support = f"{self.lines_found}/{self.lines_expected} líneas"
        return (
            f"{mark} {self.peak.position:7.1f} cm⁻¹  →  {self.species.label} "
            f"({self.species.formula}), línea de {self.line:g} "
            f"[{self.offset:+.1f} cm⁻¹, {support}]"
        )


@dataclass
class InterferenceReport:
    """What the non-carbon scan found."""

    matches: list[Match] = field(default_factory=list)
    excluded_from_rbm: list[float] = field(default_factory=list)
    species_present: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def found_anything(self) -> bool:
        return bool(self.matches)

    def summary(self) -> str:
        if not self.enabled:
            return (
                "Búsqueda de bandas no carbonosas: desactivada (es opcional y "
                "por defecto no se ejecuta)."
            )
        if not self.matches:
            return "Búsqueda de bandas no carbonosas: ninguna coincidencia."
        lines = [
            "Bandas que probablemente NO son carbono "
            "(‼ descartada de los diámetros, · identificada, ? sin corroborar):"
        ]
        lines.extend("  " + str(m) for m in self.matches)
        if self.excluded_from_rbm:
            lines.append("")
            lines.append(
                "Excluidas del análisis de diámetros: "
                + ", ".join(f"{p:.0f}" for p in self.excluded_from_rbm)
                + " cm⁻¹ (‼ arriba). Convertirlas con ω = A/d + B habría dado "
                "un diámetro de nanotubo inventado"
            )
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


@lru_cache(maxsize=2)
def _load(directory: str) -> tuple[dict, tuple[Species, ...]]:
    path = Path(directory) / "interferences.json"
    if not path.is_file():
        raise FileNotFoundError(f"falta el archivo de interferencias: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    species: list[Species] = []
    for group in payload["groups"]:
        for entry in group["species"]:
            species.append(
                Species(
                    key=entry["key"],
                    label=entry["label"],
                    formula=entry.get("formula", ""),
                    group=group["key"],
                    bands=tuple(float(v) for v in entry["bands"]),
                    strong=tuple(float(v) for v in entry.get("strong", ())),
                    confidence=entry.get("confidence", "unknown"),
                    source=entry.get("source", ""),
                    notes=entry.get("notes", ""),
                )
            )
    return payload, tuple(species)


def load_species(
    groups: Sequence[str] = DEFAULT_GROUPS, directory: Optional[str | Path] = None
) -> list[Species]:
    """Every catalogued non-carbon species in the requested groups."""
    _, species = _load(str(Path(directory) if directory else DATA_DIR))
    wanted = set(groups)
    return [s for s in species if s.group in wanted]


def find_interferences(
    peaks: Sequence[PeakMeasurement],
    groups: Sequence[str] = DEFAULT_GROUPS,
    tolerance: float = DEFAULT_TOLERANCE,
    spectrum_range: Optional[tuple[float, float]] = None,
    rbm_window: tuple[float, float] = (80.0, 400.0),
    directory: Optional[str | Path] = None,
) -> InterferenceReport:
    """Identify observed peaks as non-carbon phases, with corroboration.

    A species is accepted only when enough of the strong lines it *should*
    show inside the measured range actually appear. That is the difference
    between "there is a peak near a hematite line" and "this is hematite",
    and without it the matcher discards genuine radial breathing modes.

    Parameters
    ----------
    peaks:
        Detected peaks, from :func:`~ramancarbon.core.peaks.find_peaks`.
    groups:
        Which catalogue groups to consider. Narrow this to what your
        synthesis could plausibly have left behind.
    tolerance:
        Match window in cm⁻¹.
    spectrum_range:
        ``(low, high)`` actually measured. Needed to know which of a
        species' lines *could* have been seen: a species is not penalised
        for lines outside the window. Defaults to the span of the peaks,
        which understates it — pass the real range.
    rbm_window:
        Range in which a corroborated match is excluded from the diameter
        analysis.
    directory:
        Where to load the catalogue from.

    Returns
    -------
    InterferenceReport
    """
    species = load_species(groups, directory)
    positions = [p.position for p in peaks]
    if spectrum_range is None:
        spectrum_range = (
            (min(positions) - 20.0, max(positions) + 20.0) if positions else (0.0, 0.0)
        )

    report = InterferenceReport()
    claimed: dict[int, Match] = {}

    for item in species:
        reference = item.strong or item.bands
        # Only lines that could have been observed count for or against.
        observable = [
            line for line in reference
            if spectrum_range[0] <= line <= spectrum_range[1]
        ]
        if not observable:
            continue

        hits: list[tuple[int, float, float]] = []
        for index, peak in enumerate(peaks):
            for line in observable:
                offset = peak.position - line
                if abs(offset) <= tolerance:
                    hits.append((index, line, offset))
                    break
        if not hits:
            continue

        fraction = len(hits) / len(observable)
        corroborated = len(observable) == 1 or fraction >= CORROBORATION
        if len(observable) == 1 and len(item.bands) > 1:
            # One strong line in range but the species has others outside it:
            # nothing corroborates the identification, so say so.
            corroborated = False

        for index, line, offset in hits:
            existing = claimed.get(index)
            candidate = Match(
                peak=peaks[index],
                species=item,
                line=line,
                offset=offset,
                is_strong_line=line in item.strong,
                in_rbm_window=rbm_window[0] <= peaks[index].position <= rbm_window[1],
                corroborated=corroborated,
                lines_found=len(hits),
                lines_expected=len(observable),
            )
            # A corroborated identification beats an isolated coincidence,
            # and among equals the closer match wins.
            if existing is None or (
                (candidate.corroborated, -abs(candidate.offset))
                > (existing.corroborated, -abs(existing.offset))
            ):
                claimed[index] = candidate

    report.matches = [claimed[i] for i in sorted(claimed)]
    for match in report.matches:
        if match.corroborated:
            report.species_present[match.species.key] = (
                report.species_present.get(match.species.key, 0) + 1
            )
            if match.in_rbm_window:
                report.excluded_from_rbm.append(match.peak.position)

    uncorroborated = [m for m in report.matches if not m.corroborated]
    if uncorroborated:
        report.warnings.append(
            f"{len(uncorroborated)} coincidencia(s) sin corroborar: solo encaja "
            "una línea de la especie y faltan las demás que deberían verse. "
            "Se listan pero NO se descartan del análisis de diámetros, porque "
            "una coincidencia aislada en esta región es más probable que la "
            "especie"
        )
    _add_context(report)
    return report


def _add_context(report: InterferenceReport) -> None:
    """Say what a match implies, not just that it happened."""
    present = report.species_present
    if not present:
        return

    if "S8" in present:
        report.warnings.append(
            "hay azufre elemental sin reaccionar (S₈). En una muestra dopada "
            "con azufre eso indica que parte del precursor NO se incorporó a la "
            "red — es lo contrario de lo que su presencia se suele tomar como "
            "prueba. Contrasta con XPS: el azufre tiofénico y el S₈ se "
            "distinguen bien en el S2p"
        )
    if {"Se_trigonal", "Se_amorphous"} & set(present):
        report.warnings.append(
            "hay selenio en fase separada. Como con el azufre, indica "
            "precursor sin incorporar, no dopado sustitucional"
        )
    if {"Fe2O3_hematite", "Fe3O4_magnetite", "Co3O4", "NiO", "MoO3"} & set(present):
        report.warnings.append(
            "hay óxido de catalizador. Además de añadir bandas propias, las "
            "nanopartículas metálicas amplifican por SERS el carbono que está "
            "en contacto con ellas, y no por igual: los cocientes de intensidad "
            "pueden variar mucho entre puntos de la misma muestra. Compara "
            "zonas con y sin estas bandas antes de cuantificar"
        )
    if "Fe3O4_magnetite" in present and "Fe2O3_hematite" in present:
        report.warnings.append(
            "aparecen a la vez magnetita y hematita. La magnetita se convierte "
            "en hematita bajo el propio láser: puede que estés quemando la "
            "muestra. Baja la potencia y comprueba si las líneas de hematita "
            "se debilitan"
        )
    if "diamond" in present:
        report.warnings.append(
            "hay una línea estrecha compatible con diamante en 1332 cm⁻¹. Con "
            "un solo láser no se separa de la banda D; con dos, sí, porque la D "
            "se desplaza ~50 cm⁻¹/eV y el diamante no se mueve"
        )
    if "Si" in present:
        report.warnings.append(
            "se ve el silicio del sustrato. Aprovéchalo: su línea está en "
            "520.7 cm⁻¹ exactos, así que la desviación respecto a ese valor es "
            "el error de calibración del equipo hoy (ver analysis.quality)"
        )


__all__ = [
    "DEFAULT_GROUPS",
    "InterferenceReport",
    "Match",
    "Species",
    "find_interferences",
    "load_species",
]
