"""Oxide + chalcogenide composites: MoO₃@MoSe₂, MoO₂@MoSe₂ and relatives.

Real dichalcogenide samples are rarely one phase. The precursor was an
oxide and some of it survived; or the sample sat in air; or — and this is
the one that ruins measurements quietly — the laser oxidised it while the
spectrum was being taken. All three put oxide bands next to the
chalcogenide's, and the first job of this module is to say which oxide.

**Raman cannot see topology.** A core–shell MoO₃@MoSe₂ particle and a
physical mixture of MoO₃ and MoSe₂ powders give the same point spectrum,
because a phonon does not know what its grain is touching a micron away.
Nothing here will tell the two apart and nothing here pretends to. What it
*can* do is measure the one thing that differs: an intimate interface
strains or dopes the chalcogenide and shifts its modes, and a mixture of
powders does not. That is evidence, not proof — the proof is a microscope —
and the report says so in those words.

The second job is to separate the three origins above, because they call
for different actions. A grown-in oxide is a synthesis result; an air-grown
one is a storage problem; a laser-grown one is an artefact of your own
measurement and invalidates the numbers you just took. The module cannot
distinguish them from a single spectrum either, but it says what to measure
next: repeat on a fresh spot at lower power and watch whether the oxide
bands grow.

The quantity it reports, ``oxidation_index``, is a ratio of intensities and
**not** a mass fraction. The 819 cm⁻¹ band of α-MoO₃ is one of the
strongest Raman lines in inorganic chemistry, and the modes of a
dichalcogenide measured off-resonance are not. Comparing the two across
samples measured identically is meaningful; converting either into "percent
oxide" is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from ..core.peaks import PeakMeasurement
from ..database.loader import DATA_DIR

#: Match window in cm⁻¹ for an oxide line.
OXIDE_TOLERANCE = 6.0

#: Fraction of an oxide's observable strong lines needed to call it present.
CORROBORATION = 0.6

#: An oxide whose *signature* lines are catalogued is only accepted when at
#: least one of them is found. MoO₂ and MoSe₂ both have bands near 230 cm⁻¹;
#: without this, every MoSe₂ spectrum contains MoO₂.
REQUIRE_SIGNATURE = True


class HeterostructureError(ValueError):
    """Raised when the oxide catalogue is missing or malformed."""


@dataclass(frozen=True)
class Oxide:
    """One metal oxide that can accompany a dichalcogenide."""

    key: str
    label: str
    formula: str
    metal: str
    crystal_system: str
    space_group: Optional[str]
    bands: tuple[float, ...]
    strong: tuple[float, ...]
    signature: tuple[float, ...]
    """Lines no other catalogued oxide of the same metal has. Without one of
    these an identification rests on bands the chalcogenide itself could
    have produced."""
    confidence: str
    source: str
    colour: str
    notes: str
    min_fwhm: Optional[float] = None
    """Lower bound on the width of the matched band, for amorphous entries.
    Only a *lower* bound is evidence; see
    :mod:`ramancarbon.analysis.phases`."""

    @property
    def is_metallic(self) -> bool:
        """Whether the oxide conducts, which decides doping vs strain."""
        return "metálico" in self.colour.lower() or "METÁLICO" in self.notes


@dataclass
class OxideMatch:
    """One oxide, and how well the spectrum supports it."""

    oxide: Oxide
    found: list[tuple[float, float]] = field(default_factory=list)
    """``(catalogue line, observed position)`` pairs."""
    expected: int = 0
    signature_found: list[float] = field(default_factory=list)
    intensity: float = 0.0
    corroborated: bool = False
    reason: str = ""

    @property
    def support(self) -> float:
        return len(self.found) / self.expected if self.expected else 0.0

    def __str__(self) -> str:
        mark = "✓" if self.corroborated else "?"
        lines = ", ".join(f"{obs:.0f}" for _, obs in self.found)
        return (
            f"{mark} {self.oxide.label} [{self.oxide.formula}] — "
            f"{len(self.found)}/{self.expected} líneas fuertes ({lines} cm⁻¹)"
            f"{'; ' + self.reason if self.reason else ''}"
        )


@dataclass
class ModeShift:
    """Displacement of one chalcogenide mode from its pristine position."""

    mode: str
    observed: float
    reference: float

    @property
    def shift(self) -> float:
        return self.observed - self.reference

    def __str__(self) -> str:
        return (
            f"{self.mode}: {self.observed:.1f} cm⁻¹ "
            f"(referencia {self.reference:.1f}, {self.shift:+.1f})"
        )


@dataclass
class HeterostructureResult:
    """Oxide content and interface evidence for one dichalcogenide spectrum."""

    chalcogenide: Optional[str] = None
    matches: list[OxideMatch] = field(default_factory=list)
    shifts: list[ModeShift] = field(default_factory=list)
    oxidation_index: Optional[float] = None
    """Strongest oxide line over strongest chalcogenide line. An index for
    comparing like with like, never a mass fraction."""
    index_mode: Optional[str] = None
    """Which host mode the index was measured against. Two indices computed
    against different modes are not comparable."""
    interface_verdict: str = ""
    interface_reason: str = ""
    warnings: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def oxides_present(self) -> list[Oxide]:
        return [m.oxide for m in self.matches if m.corroborated]

    @property
    def found_anything(self) -> bool:
        return bool(self.oxides_present)

    def composition(self, topology: str = "unknown") -> str:
        """Composition string, in the notation the topology justifies.

        ``topology="core_shell"`` produces ``MoO₃@MoSe₂``; anything else
        produces ``MoO₃ + MoSe₂``, because a point Raman spectrum does not
        establish the ``@``.
        """
        oxides = self.oxides_present
        if not oxides or not self.chalcogenide:
            return self.chalcogenide or "—"
        names = " + ".join(_pretty(o.formula) for o in oxides)
        host = _pretty(self.chalcogenide)
        joiner = "@" if topology == "core_shell" else " + "
        return f"{names}{joiner}{host}"

    def summary(self) -> str:
        if not self.enabled:
            return "Búsqueda de óxidos: desactivada."
        lines: list[str] = []
        shown = [m for m in self.matches if m.corroborated]
        if not shown:
            lines.append("Óxidos: ninguno corroborado.")
            rejected = [m for m in self.matches if m.reason]
            lines.extend(f"  · descartado {m}" for m in rejected[:3])
        else:
            lines.append("Óxidos detectados:")
            lines.extend("  " + str(m) for m in shown)
            lines.append("")
            lines.append(f"Composición: {self.composition()}")
            if self.oxidation_index is not None:
                lines.append(
                    f"Índice de oxidación (I_óxido/I_{self.index_mode or '?'}): "
                    f"{self.oxidation_index:.3g}"
                )
                lines.append(
                    "  NO es una fracción másica. Las secciones eficaces "
                    "Raman de un óxido y de un dicalcogenuro difieren en "
                    "órdenes de magnitud (la banda de 819 cm⁻¹ del α-MoO₃ es "
                    "de las más intensas de la química inorgánica) y encima "
                    "dependen de la resonancia con tu láser. Sirve para "
                    "comparar muestras medidas igual, no para decir cuánto hay"
                )
        if self.shifts:
            lines.append("")
            lines.append("Modos del calcogenuro frente al material puro:")
            lines.extend("  " + str(s) for s in self.shifts)
        if self.interface_verdict:
            lines.append("")
            lines.append(f"Intercara: {self.interface_verdict}")
            lines.append(f"  {self.interface_reason}")
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _pretty(formula: str) -> str:
    """``MoO3`` → ``MoO₃``."""
    return "".join(
        c.translate(_SUBSCRIPT) if c.isdigit() else c for c in formula
    )


@lru_cache(maxsize=2)
def _load(directory: str) -> tuple[tuple[Oxide, ...], dict, dict]:
    path = Path(directory) / "tmd.json"
    if not path.is_file():
        raise HeterostructureError(f"falta el archivo de TMD: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "oxides" not in payload:
        raise HeterostructureError(f"{path} no tiene bloque 'oxides'")
    oxides = tuple(
        Oxide(
            key=entry["key"],
            label=entry["label"],
            formula=entry["formula"],
            metal=entry["metal"],
            crystal_system=entry.get("crystal_system", ""),
            space_group=entry.get("space_group"),
            bands=tuple(float(v) for v in entry["bands"]),
            strong=tuple(float(v) for v in entry.get("strong", ())),
            signature=tuple(float(v) for v in entry.get("signature", ())),
            confidence=entry.get("confidence", "unknown"),
            source=entry.get("source", ""),
            colour=entry.get("colour", ""),
            notes=entry.get("notes", ""),
            min_fwhm=(
                float(entry["min_fwhm"]) if entry.get("min_fwhm") is not None else None
            ),
        )
        for entry in payload["oxides"]
    )
    products = {
        k: tuple(v) for k, v in payload.get("oxidation_products", {}).items()
        if not k.startswith("_")
    }
    return oxides, products, payload.get("interface", {})


def load_oxides(directory: Optional[str | Path] = None) -> list[Oxide]:
    """Every catalogued oxide."""
    oxides, _, _ = _load(str(Path(directory) if directory else DATA_DIR))
    return list(oxides)


def expected_oxides(
    chalcogenide: str, directory: Optional[str | Path] = None
) -> list[Oxide]:
    """The oxides that this dichalcogenide can chemically produce.

    Restricting the search to those is not a convenience: MoO₂ has a band
    at 228 cm⁻¹ and MoSe₂'s A₁g is at 240, so an unrestricted catalogue
    finds tungsten oxides in molybdenum samples by coincidence alone.
    """
    oxides, products, _ = _load(str(Path(directory) if directory else DATA_DIR))
    keys = products.get(chalcogenide)
    by_key = {o.key: o for o in oxides}
    if keys is None:
        return list(oxides)
    return [by_key[k] for k in keys if k in by_key]


def find_oxides(
    peaks: Sequence[PeakMeasurement],
    chalcogenide: Optional[str] = None,
    spectrum_range: Optional[tuple[float, float]] = None,
    tolerance: float = OXIDE_TOLERANCE,
    directory: Optional[str | Path] = None,
) -> list[OxideMatch]:
    """Match observed peaks against the oxide catalogue, with corroboration.

    Two gates, and both exist because of a specific false positive:

    * enough of the oxide's strong lines that lie inside the measured
      range must be present, and
    * when the oxide has *signature* lines — ones no other oxide of the
      same metal shares — at least one of them must be among them.

    Without the second gate every MoSe₂ spectrum contains MoO₂, because
    MoO₂'s 203 and 228 cm⁻¹ bands sit on top of the dichalcogenide's own
    modes. The signature lines of MoO₂ are at 495 and 744 cm⁻¹, where the
    chalcogenide has nothing.
    """
    catalogue = (
        expected_oxides(chalcogenide, directory)
        if chalcogenide
        else load_oxides(directory)
    )
    positions = [p.position for p in peaks]
    if spectrum_range is None:
        spectrum_range = (
            (min(positions) - 20.0, max(positions) + 20.0) if positions else (0.0, 0.0)
        )

    matches: list[OxideMatch] = []
    for oxide in catalogue:
        reference = oxide.strong or oxide.bands
        observable = [
            line for line in reference
            if spectrum_range[0] <= line <= spectrum_range[1]
        ]
        if not observable:
            continue
        match = OxideMatch(oxide=oxide, expected=len(observable))
        used: set[int] = set()
        for line in observable:
            best, best_offset = None, None
            for index, peak in enumerate(peaks):
                if index in used:
                    continue
                offset = peak.position - line
                if abs(offset) <= tolerance and (
                    best_offset is None or abs(offset) < abs(best_offset)
                ):
                    best, best_offset = index, offset
            if best is None:
                continue
            if oxide.min_fwhm is not None:
                fwhm = peaks[best].fwhm
                if fwhm is not None and fwhm < oxide.min_fwhm:
                    match.reason = (
                        f"la banda de {line:g} cm⁻¹ mide {fwhm:.0f} cm⁻¹ de "
                        f"FWHM, menos de los {oxide.min_fwhm:g} que exige una "
                        "fase amorfa: eso es el óxido cristalino"
                    )
                    continue
            used.add(best)
            match.found.append((line, peaks[best].position))
            match.intensity = max(match.intensity, max(0.0, peaks[best].height))
            if any(abs(line - s) < 1e-6 for s in oxide.signature):
                match.signature_found.append(line)

        if not match.found:
            continue
        enough = match.support >= CORROBORATION
        signature_in_range = [
            s for s in oxide.signature
            if spectrum_range[0] <= s <= spectrum_range[1]
        ]
        has_signature = bool(match.signature_found) or not signature_in_range
        if REQUIRE_SIGNATURE and signature_in_range and not match.signature_found:
            match.reason = (
                "faltan sus líneas exclusivas ("
                + ", ".join(f"{s:g}" for s in signature_in_range)
                + " cm⁻¹); las que coinciden podría producirlas el propio "
                "calcogenuro"
            )
        elif not enough:
            match.reason = (
                f"solo {len(match.found)} de {match.expected} líneas fuertes; "
                "una fase es un espectro, no una línea"
            )
        match.corroborated = enough and has_signature
        matches.append(match)

    matches.sort(key=lambda m: (-int(m.corroborated), -m.support, -m.intensity))
    return matches


def analyse_heterostructure(
    peaks: Sequence[PeakMeasurement],
    chalcogenide: Optional[str],
    mode_positions: Optional[dict[str, float]] = None,
    reference_positions: Optional[dict[str, float]] = None,
    spectrum_range: Optional[tuple[float, float]] = None,
    directory: Optional[str | Path] = None,
) -> HeterostructureResult:
    """Oxide content, oxidation index and interface evidence.

    Parameters
    ----------
    peaks:
        Detected peaks over the whole measured range. The oxide signatures
        live between 400 and 1000 cm⁻¹, well above the dichalcogenide's own
        modes, so a spectrum that stops at 500 cm⁻¹ cannot see them — and
        the result says so rather than reporting "no oxide".
    chalcogenide:
        Key of the host, e.g. ``"MoSe2"``. Restricts the oxide search to
        what that metal can produce.
    mode_positions:
        Fitted positions of the host's modes, from
        :func:`~ramancarbon.analysis.tmd.analyse_tmd`.
    reference_positions:
        Pristine positions of the same modes, from the database.
    spectrum_range:
        ``(low, high)`` actually measured, cm⁻¹.
    directory:
        Where to load the catalogue from.

    Returns
    -------
    HeterostructureResult
    """
    _, _, interface = _load(str(Path(directory) if directory else DATA_DIR))
    result = HeterostructureResult(chalcogenide=chalcogenide)
    result.matches = find_oxides(
        peaks, chalcogenide, spectrum_range=spectrum_range, directory=directory
    )

    if spectrum_range is not None and spectrum_range[1] < 830.0:
        result.warnings.append(
            f"el espectro llega solo a {spectrum_range[1]:.0f} cm⁻¹. Las "
            "líneas que identifican un óxido sin ambigüedad están entre 640 y "
            "1000 cm⁻¹ (819 y 995 del MoO₃, 744 del MoO₂, 807 del WO₃). Sin "
            "esa región, «no hay óxido» no significa nada: mide hasta al "
            "menos 1050 cm⁻¹"
        )

    _oxidation_index(result, peaks, mode_positions)
    _interface_evidence(result, mode_positions, reference_positions, interface)
    _add_context(result)
    return result


def _oxidation_index(
    result: HeterostructureResult,
    peaks: Sequence[PeakMeasurement],
    mode_positions: Optional[dict[str, float]],
) -> None:
    """Strongest oxide line over the strongest *uncontaminated* host line.

    The denominator is the trap. MoO₃ has bands at 246 and 291 cm⁻¹ and
    MoSe₂'s modes are at 240 and 287; measure the "host" peak there and you
    have measured the oxide, and the ratio comes out as 1 no matter how
    much oxide there is. So a host mode within :data:`OXIDE_TOLERANCE` of
    any line of a detected oxide is not used, and when every host mode is
    contaminated the index is refused with a reason instead of reported
    wrong.
    """
    present = [m for m in result.matches if m.corroborated]
    if not present or not mode_positions:
        return
    contaminating = [
        line for match in present for line in match.oxide.bands
    ]

    clean_height = 0.0
    clean_mode: Optional[str] = None
    blocked: list[str] = []
    for name, position in sorted(mode_positions.items()):
        if any(abs(position - line) <= OXIDE_TOLERANCE for line in contaminating):
            blocked.append(name)
            continue
        near = [p for p in peaks if abs(p.position - position) <= OXIDE_TOLERANCE]
        if near:
            height = max(p.height for p in near)
            if height > clean_height:
                clean_height, clean_mode = height, name

    if clean_height <= 0.0:
        result.warnings.append(
            "no se puede dar un índice de oxidación: "
            + (
                "los modos del calcogenuro ("
                + ", ".join(blocked)
                + ") caen encima de bandas del óxido, así que medir su altura "
                "es medir la del óxido"
                if blocked
                else "no se ha medido ningún modo del calcogenuro"
            )
        )
        return

    result.oxidation_index = max(m.intensity for m in present) / clean_height
    result.index_mode = clean_mode
    if blocked:
        result.warnings.append(
            f"el índice de oxidación se calcula con el modo {clean_mode}, "
            f"porque {', '.join(blocked)} solapa(n) con bandas del óxido y su "
            "altura no sería del calcogenuro. Compara solo índices calculados "
            "con el mismo modo"
        )


def _interface_evidence(
    result: HeterostructureResult,
    mode_positions: Optional[dict[str, float]],
    reference_positions: Optional[dict[str, float]],
    interface: dict,
) -> None:
    """Read the host's mode shifts as strain, doping, or neither."""
    if not mode_positions or not reference_positions:
        return
    threshold = float(interface.get("significant_shift_cm1", 1.5))
    large = float(interface.get("large_shift_cm1", 4.0))
    out_key = interface.get("out_of_plane_mode", "A1g")
    in_key = interface.get("in_plane_mode", "E2g")

    for key, observed in sorted(mode_positions.items()):
        reference = reference_positions.get(key)
        if reference is None:
            continue
        result.shifts.append(ModeShift(mode=key, observed=observed, reference=reference))
    if not result.shifts:
        return

    by_mode = {s.mode: s.shift for s in result.shifts}
    out_shift = by_mode.get(out_key)
    in_shift = by_mode.get(in_key)
    # Judge on the two principal modes only. The secondary ones (B12g, 2LA)
    # sit where oxide bands are and a fit pulled by an overlapping oxide
    # line is not interface physics.
    principal = [abs(v) for k, v in by_mode.items() if k in (out_key, in_key)]
    biggest = max(principal) if principal else max(abs(s.shift) for s in result.shifts)

    if biggest < threshold:
        result.interface_verdict = "sin evidencia de intercara"
        result.interface_reason = (
            f"ningún modo se desplaza más de {threshold:g} cm⁻¹ respecto al "
            "material puro. Eso es lo que se espera de una MEZCLA física de "
            "polvos, y también de un recubrimiento muy fino o mal acoplado. "
            "No distingue entre las dos: Raman no ve topología"
        )
        return

    metallic = any(o.is_metallic for o in result.oxides_present)
    if out_shift is not None and in_shift is not None:
        same_sign = out_shift * in_shift > 0
        if abs(out_shift) >= threshold and abs(in_shift) < threshold:
            mechanism = (
                f"solo se mueve el {out_key} (fuera del plano), que es lo que "
                "responde a la transferencia de carga"
            )
            verdict = "compatible con dopado por la intercara"
        elif same_sign and abs(in_shift) >= threshold:
            mechanism = (
                "los dos modos se desplazan en el mismo sentido, que es la "
                "firma de una deformación biaxial"
            )
            verdict = "compatible con deformación en la intercara"
        else:
            mechanism = (
                "los modos se mueven de forma discordante; puede haber "
                "deformación y dopado a la vez, o el ajuste de una de las dos "
                "bandas es malo"
            )
            verdict = "intercara presente, mecanismo no separable"
    else:
        mechanism = "no se han podido medir los dos modos principales"
        verdict = "desplazamiento presente, mecanismo no separable"

    expectation = ""
    if result.oxides_present:
        expectation = (
            " Con un óxido metálico encima (conduce) lo esperable es dopado."
            if metallic
            else " Con un óxido aislante encima lo esperable es deformación."
        )
    result.interface_verdict = verdict
    result.interface_reason = (
        f"desplazamiento máximo {biggest:.1f} cm⁻¹ (umbral {threshold:g}); "
        f"{mechanism}.{expectation} Un desplazamiento así indica una "
        "intercara real, no una mezcla de polvos — pero es un indicio, no "
        "una prueba de estructura núcleo-corteza. Eso lo decide la "
        "microscopía"
    )
    if biggest >= large:
        result.warnings.append(
            f"el desplazamiento de {biggest:.1f} cm⁻¹ es grande. Antes de "
            "atribuirlo a la intercara, comprueba la calibración del equipo "
            "con una referencia (silicio a 520.7 cm⁻¹): un error de eje "
            "produce exactamente esto y afecta a todos los modos por igual"
        )


def _add_context(result: HeterostructureResult) -> None:
    """Say what the oxide implies and what to measure next."""
    present = {o.key for o in result.oxides_present}
    if not present:
        return
    result.warnings.append(
        "un óxido en el espectro puede venir de tres sitios distintos y no se "
        "distinguen en una sola medida: precursor sin reaccionar, oxidación "
        "al aire, u oxidación PROVOCADA POR EL LÁSER durante esta medida. "
        "Compruébalo: repite en un punto virgen a la mitad de potencia y mira "
        "si las bandas del óxido crecen con el tiempo de exposición. Si "
        "crecen, el óxido lo estás haciendo tú y los cocientes de esta medida "
        "no valen"
    )
    if {"MoO3_alpha", "MoO2"} <= present:
        result.warnings.append(
            "aparecen a la vez MoO₃ y MoO₂. Puede ser una mezcla real de los "
            "dos óxidos o un óxido subestequiométrico MoO₃₋ₓ; Raman no los "
            "separa. La pista macroscópica es el color: el MoO₃ es blanco y "
            "aislante, el MoO₂ violeta oscuro y conductor, y un MoO₃₋ₓ con "
            "vacantes es azul intenso"
        )
    if any(o.is_metallic for o in result.oxides_present):
        result.warnings.append(
            "el óxido identificado es un CONDUCTOR metálico (MoO₂/WO₂). En un "
            "compuesto con el calcogenuro eso significa transferencia de carga "
            "real y, si vas a medir electroquímica, una contribución propia a "
            "la capacitancia y a la conductividad que no es del calcogenuro. "
            "Tenlo en cuenta al interpretar la pestaña de electroquímica"
        )
    if "MoOx_amorphous" in present:
        result.warnings.append(
            "el óxido es amorfo: bandas muy anchas y sin estructura fina. DRX "
            "no lo verá (no da picos de Bragg), así que Raman es aquí la "
            "técnica sensible y no al revés"
        )


__all__ = [
    "CORROBORATION",
    "OXIDE_TOLERANCE",
    "HeterostructureError",
    "HeterostructureResult",
    "ModeShift",
    "Oxide",
    "OxideMatch",
    "analyse_heterostructure",
    "expected_oxides",
    "find_oxides",
    "load_oxides",
]
