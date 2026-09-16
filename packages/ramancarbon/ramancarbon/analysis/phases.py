"""Crystalline phases that are part of the sample rather than dirt on it.

This is the sibling of :mod:`ramancarbon.analysis.interference` and the
difference is intent, not method. There, a non-carbon band is a nuisance to
be kept out of the diameter analysis. Here it is the point of the
measurement: a nanotube decorated with FeSe is an FeSe sample as much as a
carbon one, and the program should say which FeSe.

**This scan runs by default**, unlike the interference scan. The reason is
a collision that is specific and severe. Tetragonal β-FeSe puts its two
Raman modes at 181 and 196 cm⁻¹; elemental selenium at 237 and 254; and
cementite at 212 and 280. Every one of those sits inside the radial
breathing mode window. Pushed through ``ω = A/d + B`` they give tube
diameters of 1.42, 1.30, 1.05, 0.97, 1.20 and 0.88 nm — six believable
numbers, all fabricated. A user measuring exactly this kind of sample gets
that wrong answer silently unless something looks for the phases first.

The bias objection that keeps the interference scan switched off still
applies, and the same defence answers it: **a phase is a spectrum, not a
line**. A candidate is reported as identified only when enough of the lines
it should show, in the measured range, are actually there. One match out of
three is arithmetic; three out of three is a phase.

Polymorphs get a second layer on top of that. Phases are grouped into
families (all the FeSe, all the elemental Se), and the module reports the
family whenever it can and the polymorph only when a *discriminating* line
separates it from its siblings. β-FeSe tetragonal and δ-FeSe hexagonal are
the honest hard case: the first is well characterised, the second is not,
and no amount of curve fitting fixes that. Each phase therefore carries an
``xrd_hint`` — the reflection and 2θ that settles it — so the report can
hand the question to the diffraction tab instead of guessing. Raman
proposes a phase; a diffractogram demonstrates it.

Nothing here is dispersive. Measuring at two lasers separates any of these
from carbon in one step: the D band moves ~50 cm⁻¹/eV and a lattice mode of
an ordinary crystal does not move at all.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

from ..core.peaks import PeakMeasurement
from ..database.loader import DATA_DIR

#: Match window in cm⁻¹. Slightly wider than the interference matcher's,
#: because several of these phases are non-stoichiometric or strained and
#: their modes genuinely move by a few wavenumbers between samples.
DEFAULT_TOLERANCE = 5.0

#: Fraction of a phase's observable strong lines that must be found before
#: the identification counts as corroborated.
CORROBORATION = 0.6

#: Range in which a matched line collides with the radial breathing mode.
RBM_WINDOW = (80.0, 400.0)

#: Minimum FWHM in cm⁻¹ below which an "amorphous" assignment is refused.
#: Amorphous selenium is distinguished from the monoclinic ring crystal by
#: the *width* of the 250 cm⁻¹ band, not by its position.
AMORPHOUS_MIN_FWHM = 15.0


class PhaseDatabaseError(ValueError):
    """Raised when ``phases.json`` is missing or malformed."""


@dataclass(frozen=True)
class PhaseBand:
    """One catalogued Raman line of a phase."""

    position: float
    window: tuple[float, float]
    assignment: str
    relative: float
    """Intensity relative to the strongest line of the same phase."""


@dataclass(frozen=True)
class Phase:
    """One crystalline (or amorphous) phase from ``phases.json``."""

    key: str
    label: str
    formula: str
    family: str
    crystal_system: str
    space_group: Optional[str]
    bands: tuple[PhaseBand, ...]
    strong: tuple[float, ...]
    discriminating: tuple[float, ...]
    confidence: str
    source: str
    xrd_hint: dict
    notes: str
    width_rule: Optional[dict] = None
    origin: str = "programa"
    """``"programa"`` for a phase that ships with the package and
    ``"usuario"`` for one the user added. Shown everywhere a phase is
    named, because the two do not carry the same warranty: the bundled
    ones were checked against the literature they cite, and a phase
    somebody typed in last Tuesday was not."""
    """Optional band-width test, ``{"line", "min_fwhm"|"max_fwhm", "reason"}``.

    Some polymorphs differ in the width of a band rather than its position.
    Amorphous and monoclinic selenium both peak near 252 cm⁻¹; only the
    FWHM separates them. A failed width test rejects the *polymorph*, never
    the composition, and an unmeasurable FWHM produces "could not check"
    rather than a decision."""

    @property
    def raman_silent(self) -> bool:
        """Whether the phase has no first-order Raman modes at all.

        True for metallic α-iron. An entry with no bands is deliberate: it
        lets the report answer "is there iron?" with the correct "Raman
        cannot see it" instead of silence.
        """
        return not self.bands

    def band_at(self, line: float) -> Optional[PhaseBand]:
        """The catalogued band whose position equals ``line``."""
        for band in self.bands:
            if abs(band.position - line) < 1e-6:
                return band
        return None

    def xrd_advice(self) -> str:
        """One sentence naming the reflection that would settle this phase."""
        hint = self.xrd_hint or {}
        two_theta = hint.get("two_theta_cu")
        reflection = hint.get("reflection", "?")
        note = hint.get("note", "")
        if two_theta is None:
            return f"{self.label}: {note}" if note else f"{self.label}: sin reflexión de referencia."
        text = f"{self.label}: reflexión {reflection} a 2θ ≈ {two_theta:.1f}° (Cu Kα)"
        return f"{text}. {note}" if note else text + "."


@dataclass(frozen=True)
class Family:
    """A group of polymorphs of the same composition."""

    key: str
    label: str
    note: str
    resolved_by: str
    group: str = ""
    """Optional grouping above the family, e.g. ``"polimeros"``."""
    optional: bool = False
    """Whether the family is searched only on request.

    The polymers are, and the reason is specific rather than tidiness:
    polyaniline has bands at 1340 and 1590, polypyrrole at 1330 and 1590,
    PET at 1615. Those sit on the D band and on the G band. Searched by
    default they would match every carbon spectrum ever measured, and the
    match would be a coincidence of position between a conjugated polymer
    and a graphitic lattice — two different physics that happen to
    vibrate at the same frequency. So they are there when you ask for
    them and absent when you do not."""


@dataclass
class PhaseHit:
    """One observed peak matched to one catalogued line."""

    peak: PeakMeasurement
    line: PhaseBand
    offset: float
    """Observed minus catalogue position, cm⁻¹."""

    @property
    def is_discriminating(self) -> bool:
        return bool(getattr(self, "_discriminating", False))


@dataclass
class PhaseIdentification:
    """A phase, and how well the spectrum supports it."""

    phase: Phase
    hits: list[PhaseHit] = field(default_factory=list)
    lines_expected: int = 0
    """Strong lines that lie inside the measured range."""
    discriminating_found: int = 0
    discriminating_expected: int = 0
    corroborated: bool = False
    intensity: float = 0.0
    """Summed height of the matched peaks. Comparable between phases only
    with the caveat spelled out in :meth:`PhaseReport.abundance_caveat`."""

    strong_found: int = 0
    """Matched lines that belong to the phase's strong set."""
    strong_out_of_range: bool = False
    """True when none of the phase's strong lines lies inside the measured
    range. Such a phase cannot be corroborated — only its minor lines were
    ever measurable — and saying so is different from saying it is absent."""
    width_ok: bool = True
    """False when the phase's band-width test failed."""
    width_confirmed: bool = False
    """True when a *lower* bound on the band width was met.

    Only a lower bound corroborates. A band too broad to be anything else
    is positive evidence — an amorphous phase has exactly one broad band
    and can never produce a second line, so without this it could never be
    identified at all. An upper bound is not evidence: "narrower than
    15 cm⁻¹" is satisfied by every radial breathing mode there is, and
    letting it corroborate turned the 254 cm⁻¹ RBM of the clean single-wall
    demo into monoclinic selenium."""
    width_note: str = ""

    @property
    def lines_found(self) -> int:
        return len(self.hits)

    @property
    def support(self) -> float:
        """Fraction of the expected strong lines that were found, 0–1."""
        if self.lines_expected == 0:
            return 0.0
        return min(1.0, self.strong_found / self.lines_expected)

    @property
    def mean_offset(self) -> float:
        """Mean signed position error, cm⁻¹. A systematic offset here is a
        calibration error, not a wrong identification."""
        if not self.hits:
            return 0.0
        return sum(h.offset for h in self.hits) / len(self.hits)

    @property
    def discriminated(self) -> bool:
        """Whether something separated this polymorph from its siblings.

        Either a line unique to it was seen, or its band-width test passed
        where a sibling's failed. Both are discriminators; the width one is
        the only discriminator amorphous selenium has.
        """
        return self.discriminating_found > 0 or self.width_confirmed

    def __str__(self) -> str:
        mark = "✓" if self.corroborated else "?"
        if not self.width_ok:
            mark = "✗"
        elif self.corroborated and not self.discriminated and self.discriminating_expected:
            mark = "~"
        weak = self.lines_found - self.strong_found
        extra = f" +{weak} débil(es)" if weak > 0 else ""
        lines = ", ".join(f"{h.peak.position:.0f}" for h in self.hits)
        text = (
            f"{mark} {self.phase.label} [{self.phase.formula}] — "
            f"{self.strong_found}/{self.lines_expected} líneas fuertes{extra} "
            f"({lines} cm⁻¹), desviación media {self.mean_offset:+.1f} cm⁻¹, "
            f"confianza de la referencia: {self.phase.confidence}"
        )
        if self.width_note:
            marker = "✗ anchura" if not self.width_ok else "anchura"
            text += f"\n      {marker}: {self.width_note}"
        return text


@dataclass
class FamilyVerdict:
    """What can be said about one composition, across its polymorphs."""

    family: Family
    candidates: list[PhaseIdentification] = field(default_factory=list)
    best: Optional[PhaseIdentification] = None
    resolved: bool = False
    """Whether one polymorph is singled out by a discriminating line."""
    tentative: bool = False
    """True when no candidate in the family was even corroborated."""
    reason: str = ""

    @property
    def present(self) -> bool:
        return bool(self.candidates)

    def __str__(self) -> str:
        if not self.candidates:
            return f"{self.family.label}: no detectado."
        if self.tentative:
            names = " / ".join(c.phase.label for c in self.candidates)
            head = f"{self.family.label} → posible {names}, sin corroborar"
            return f"{head}\n    {self.reason}"
        if self.resolved and self.best is not None:
            head = f"{self.family.label} → {self.best.phase.label}"
        else:
            names = " o ".join(
                c.phase.label for c in self.candidates if c.width_ok
            ) or " o ".join(c.phase.label for c in self.candidates)
            head = f"{self.family.label} → sin resolver entre {names}"
        return f"{head}\n    {self.reason}"


@dataclass
class PhaseReport:
    """Everything the phase scan concluded."""

    identifications: list[PhaseIdentification] = field(default_factory=list)
    families: list[FamilyVerdict] = field(default_factory=list)
    rbm_conflicts: list[tuple[float, str]] = field(default_factory=list)
    """``(position, phase label)`` for matched lines inside the RBM window."""
    rbm_suspects: list[tuple[float, str]] = field(default_factory=list)
    """``(position, phase label)`` for *uncorroborated* leads in that window.

    Not enough evidence to name the phase, and far too much to convert the
    peak into a nanotube diameter without saying so. A single catalogued
    line landing on a peak is exactly the case that put four invented
    diameters in a spectrum of carbon decorated with FeSe: the two FeSe
    lines were corroborated and caught, and the two selenium ones were a
    lone line each and went through in silence.
    """
    unmatched_peaks: list[PeakMeasurement] = field(default_factory=list)
    near_misses: dict[float, list[tuple[str, float, float]]] = field(
        default_factory=dict)
    """For each unexplained peak, the catalogued lines that came closest:
    ``position -> [(phase label, band position, distance)]``.

    "SIN EXPLICAR" is the result, not a failure — it is the phase you were
    not expecting — but it is a result nobody can act on. A peak at
    580 cm⁻¹ with nothing beside it leaves the user to search the
    literature from scratch; the same peak with "el 550 de la goethita
    está a 30 cm⁻¹, el 612 de la hematita a 32" tells them which two
    cards to pull and how far off each is. It is the same reasoning as
    reporting the strongest sub-threshold maximum on a diffractogram that
    identified nothing: a number the user can judge costs nothing and
    silence costs them the afternoon."""
    elements: list[str] = field(default_factory=list)
    """The elements the search was restricted to, empty when it was not.
    A restricted search is both narrower and LOUDER: see
    :func:`find_phases`."""
    warnings: list[str] = field(default_factory=list)
    xrd_questions: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def found_anything(self) -> bool:
        return any(i.corroborated for i in self.identifications)

    @property
    def excluded_from_rbm(self) -> list[float]:
        """Positions the diameter analysis must not convert."""
        return [position for position, _ in self.rbm_conflicts]

    def drop_explained(self, positions: Sequence[float],
                       tolerance: float = 12.0) -> None:
        """Forget peaks another part of the analysis already accounts for.

        The phase scan runs before the band assignment and knows nothing
        about it, so the D and the G of a carbon sample arrive here as
        "unexplained" — and then the near-miss list helpfully offers
        hexagonal boron nitride for the D band and graphitic carbon
        nitride for the G. Those are not leads, they are noise generated
        by asking the wrong question of the best understood bands in the
        spectrum.
        """
        if not positions:
            return
        def spoken_for(value: float) -> bool:
            return any(abs(value - other) <= tolerance for other in positions)

        self.unmatched_peaks = [
            peak for peak in self.unmatched_peaks
            if not spoken_for(peak.position)
        ]
        self.near_misses = {
            position: hits for position, hits in self.near_misses.items()
            if not spoken_for(position)
        }

    def abundance_caveat(self) -> str:
        """Why the intensity numbers are not weight fractions."""
        return (
            "Las intensidades relativas NO son fracciones másicas. Las "
            "secciones eficaces Raman de estas fases difieren en órdenes de "
            "magnitud entre sí y respecto al carbono sp², y además dependen "
            "de la resonancia con el láser que uses. Sirven para comparar el "
            "MISMO par de fases entre puntos o entre muestras medidas igual; "
            "no para decir cuánto hay. Para eso, DRX con Rietveld."
        )

    def summary(self, include_tentative: bool = False) -> str:
        """Human-readable report.

        Uncorroborated single-line coincidences are hidden by default and
        counted instead. They are real information — a lead worth chasing
        with a second laser — but on a clean spectrum there are always two
        or three of them, and printing them next to a genuine call makes
        the genuine call harder to see. Pass ``include_tentative=True`` to
        get all of them.
        """
        if not self.enabled:
            return "Identificación de fases: desactivada."
        lines: list[str] = []
        shown = [
            i for i in self.identifications if include_tentative or i.corroborated
        ]
        hidden = len(self.identifications) - len(shown)
        if not shown:
            lines.append("Identificación de fases no carbonosas: ninguna corroborada.")
        else:
            lines.append(
                "Fases detectadas (✓ corroborada, ~ familia sin resolver, "
                "? sin corroborar):"
            )
            lines.extend("  " + str(i) for i in shown)
        if hidden:
            lines.append(
                f"  ({hidden} coincidencia(s) aislada(s) sin corroborar, "
                "ocultas: una línea suelta en esta región es más probable "
                "que la fase)"
            )
        families = [
            f for f in self.families if include_tentative or not f.tentative
        ]
        if families:
            lines.append("")
            lines.append("Por composición:")
            lines.extend("  " + str(f) for f in families)
        if self.rbm_conflicts:
            lines.append("")
            lines.append(
                "‼ Líneas dentro de la ventana RBM, excluidas del cálculo de "
                "diámetros:"
            )
            for position, label in self.rbm_conflicts:
                lines.append(f"    {position:7.1f} cm⁻¹  →  {label}")
        if self.xrd_questions:
            lines.append("")
            lines.append("Lo que zanjaría la duda en DRX:")
            lines.extend("  · " + q for q in self.xrd_questions)
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def user_phase_file() -> Path:
    """Where the user's own Raman phases live.

    One fixed, documented place next to the preferences file, read on
    every load. Same answer as the CIF drop folder on the diffraction
    side, for the same reason: "where do I put my own references" should
    have an answer, not a dialog to find again every session.

    The bundled catalogue is never written to. A package upgrade replaces
    it, and a user's phases living inside it would be lost without
    warning — which is a good way to lose a year of somebody's
    measurements.
    """
    from ..core.history import config_directory

    return config_directory() / "fases_usuario.json"


def _merge_user_phases(payload: dict) -> dict:
    """Fold the user's own phases into the catalogue payload.

    A user phase whose key matches a bundled one REPLACES it, so a
    position the user has measured better than the reference can be
    corrected without editing the package. That is deliberate and it is
    why every phase carries its origin: an overridden entry says
    "usuario", and a report that names it says so too.

    A malformed user file is reported and skipped, never fatal. The
    bundled catalogue has to keep working when somebody's hand-written
    JSON has a trailing comma in it.
    """
    path = user_phase_file()
    if not path.is_file():
        return payload
    try:
        extra = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        payload.setdefault("_problems", []).append(
            f"no se ha podido leer {path}: {error}. Se ignoran las fases "
            "propias y se sigue con el catálogo del programa"
        )
        return payload
    if not isinstance(extra, dict):
        payload.setdefault("_problems", []).append(
            f"{path} no contiene un objeto JSON; se ignora")
        return payload

    families = {f["key"]: f for f in payload.get("families", [])}
    for family in extra.get("families", []):
        if isinstance(family, dict) and family.get("key"):
            families[family["key"]] = family
    payload["families"] = list(families.values())

    phases = {p["key"]: p for p in payload.get("phases", [])}
    for entry in extra.get("phases", []):
        if not isinstance(entry, dict) or not entry.get("key"):
            continue
        entry = dict(entry)
        entry["origin"] = "usuario"
        phases[entry["key"]] = entry
    payload["phases"] = list(phases.values())
    return payload


@lru_cache(maxsize=2)
def _load(directory: str) -> tuple[dict[str, Family], tuple[Phase, ...], float]:
    path = Path(directory) / "phases.json"
    if not path.is_file():
        raise PhaseDatabaseError(f"falta el archivo de fases: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - hand edits
        raise PhaseDatabaseError(f"{path} no es JSON válido: {exc}") from exc
    payload = _merge_user_phases(payload)

    families = {
        entry["key"]: Family(
            key=entry["key"],
            label=entry["label"],
            note=entry.get("note", ""),
            resolved_by=entry.get("resolved_by", ""),
            group=entry.get("group", ""),
            optional=bool(entry.get("optional", False)),
        )
        for entry in payload.get("families", [])
    }

    phases: list[Phase] = []
    for entry in payload["phases"]:
        if entry["family"] not in families:
            raise PhaseDatabaseError(
                f"la fase {entry['key']} declara la familia {entry['family']}, "
                "que no está definida"
            )
        bands = tuple(
            PhaseBand(
                position=float(b["position"]),
                window=(float(b["window"][0]), float(b["window"][1])),
                assignment=b.get("assignment", ""),
                relative=float(b.get("relative", 1.0)),
            )
            for b in entry.get("bands", [])
        )
        phases.append(
            Phase(
                key=entry["key"],
                label=entry["label"],
                formula=entry.get("formula", ""),
                family=entry["family"],
                crystal_system=entry.get("crystal_system", ""),
                space_group=entry.get("space_group"),
                bands=bands,
                strong=tuple(float(v) for v in entry.get("strong", ())),
                discriminating=tuple(float(v) for v in entry.get("discriminating", ())),
                confidence=entry.get("confidence", "unknown"),
                source=entry.get("source", ""),
                xrd_hint=entry.get("xrd_hint", {}),
                notes=entry.get("notes", ""),
                width_rule=entry.get("width_rule"),
                origin=entry.get("origin", "programa"),
            )
        )
    tolerance = float(payload.get("match_tolerance_cm1", DEFAULT_TOLERANCE))
    return families, tuple(phases), tolerance


def save_user_phase(entry: dict) -> Path:
    """Add or replace one phase in the user's own catalogue.

    The entry is the same shape as one in ``phases.json``. It is checked
    hard before being written, because a phase with no discriminating
    line or no source is not a reference — it is a guess that will be
    reported with the same confidence as everything else.
    """
    required = ("key", "label", "family", "bands")
    missing = [field for field in required if not entry.get(field)]
    if missing:
        raise PhaseDatabaseError(
            "faltan campos obligatorios: " + ", ".join(missing))
    if not entry.get("source"):
        raise PhaseDatabaseError(
            "una fase sin 'source' no es una referencia. Pon de dónde salen "
            "las posiciones, aunque sea «medido en mi muestra el 3/4/26»")
    for band in entry["bands"]:
        low, high = band["window"]
        if not low < band["position"] < high:
            raise PhaseDatabaseError(
                f"la banda de {band['position']} cm⁻¹ está fuera de su propia "
                f"ventana [{low}, {high}]")

    path = user_phase_file()
    payload = {"families": [], "phases": []}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {"families": [], "phases": []}
    phases = [p for p in payload.get("phases", []) if p.get("key") != entry["key"]]
    phases.append(entry)
    payload["phases"] = phases
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    _load.cache_clear()
    return path


def delete_user_phase(key: str) -> bool:
    """Remove one phase from the user's catalogue.

    Only from the user's. A bundled phase cannot be deleted, because the
    file it lives in is replaced on the next package upgrade and the
    deletion would come back — which is worse than refusing. To suppress
    a bundled phase, override it with a user entry of the same key.
    """
    path = user_phase_file()
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    phases = payload.get("phases", [])
    kept = [p for p in phases if p.get("key") != key]
    if len(kept) == len(phases):
        return False
    payload["phases"] = kept
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    _load.cache_clear()
    return True


def load_phases(directory: Optional[str | Path] = None) -> list[Phase]:
    """Every catalogued phase."""
    _, phases, _ = _load(str(Path(directory) if directory else DATA_DIR))
    return list(phases)


def load_families(directory: Optional[str | Path] = None) -> list[Family]:
    """Every polymorph family."""
    families, _, _ = _load(str(Path(directory) if directory else DATA_DIR))
    return list(families.values())


#: Element symbols, longest first so "Se" is not read as "S" + "e".
_SYMBOL = re.compile(r"[A-Z][a-z]?")


def match_tolerance(directory: Optional[str | Path] = None) -> float:
    """The catalogue's own match window, in cm⁻¹.

    Exposed so that anything drawing the result — the spectrum plot marks
    explained peaks in their own colour — uses the same window the search
    used, instead of a second constant that drifts away from it.
    """
    return _load(str(Path(directory) if directory else DATA_DIR))[2]


def elements_of(formula: str) -> set[str]:
    """The element symbols in a chemical formula.

    Deliberately crude: it reads capital-then-optional-lowercase runs and
    ignores counts, charges and brackets, which is everything a formula in
    this catalogue actually contains (``Fe3C``, ``FeOOH``, ``C3H6N6``,
    ``CaCO3``). It exists to answer one question — could this phase be in
    a sample made of these elements — and that question does not need a
    parser.
    """
    return set(_SYMBOL.findall(formula or ""))


def find_phases(
    peaks: Sequence[PeakMeasurement],
    spectrum_range: Optional[tuple[float, float]] = None,
    tolerance: Optional[float] = None,
    families: Optional[Sequence[str]] = None,
    rbm_window: tuple[float, float] = RBM_WINDOW,
    directory: Optional[str | Path] = None,
    elements: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[str]] = None,
) -> PhaseReport:
    """Identify sample phases from detected peaks, family by family.

    Parameters
    ----------
    peaks:
        Detected peaks, from :func:`~ramancarbon.core.peaks.find_peaks`.
    spectrum_range:
        ``(low, high)`` actually measured, in cm⁻¹. This decides which of a
        phase's lines *could* have been seen, and therefore whether a
        missing line counts against it. Defaults to the span of the peaks,
        which understates the range and makes the test too lenient — pass
        the real one.
    tolerance:
        Match window in cm⁻¹. Defaults to the value in ``phases.json``.
    families:
        Restrict to these composition families. Narrow it to what your
        synthesis could plausibly contain; the whole catalogue is the
        default because a user who did not ask for FeSe still benefits from
        being told its lines are not radial breathing modes.
    rbm_window:
        Range in which a corroborated match blocks a diameter calculation.
    directory:
        Where to load ``phases.json`` from.
    elements:
        The elements the sample can contain, e.g. ``["C", "Fe", "Se"]``.
        Phases needing an element that is not on the list are not searched
        at all — a manganese oxide cannot be in a sample with no
        manganese, and with three dozen phases catalogued those
        impossible matches are most of the noise.

        It also changes what counts as an RBM suspect, and that is the
        point of it. Without a composition the program has to be cautious:
        nearly any genuine radial breathing mode lands within tolerance of
        *some* catalogued line, so only a high-confidence phase's own
        exclusive line raises a flag. That caution silences the case this
        whole module exists for — iron carbide's bands at 212 and 280 sit
        inside the RBM window, and its catalogue entry is low confidence
        because cementite's Raman cross-section is poor, so a sample full
        of Fe₃C got no warning at all. Once you have said the sample
        contains iron, a line in the RBM window is worth interrupting for
        whatever the literature's confidence in it, and the report names
        that confidence instead of hiding behind it.

    groups:
        Optional groups of families to switch ON, e.g. ``["polimeros"]``.
        A family marked optional is not searched unless its group is named
        here or the family itself is named in ``families``.

        There is one such group and it exists for a concrete reason:
        polyaniline has bands at 1340 and 1590 cm⁻¹, polypyrrole at 1330
        and 1590, PET at 1615. Those are the D band and the G band. A
        polymer library searched by default would match every carbon
        spectrum in existence, on a coincidence between how a conjugated
        polymer vibrates and how a graphitic lattice does.

    Returns
    -------
    PhaseReport
    """
    family_map, catalogue, file_tolerance = _load(
        str(Path(directory) if directory else DATA_DIR)
    )
    tol = float(tolerance) if tolerance is not None else file_tolerance
    wanted = set(families) if families else None
    enabled_groups = {str(g) for g in groups} if groups else set()
    optional_off = {
        key for key, family in family_map.items()
        if family.optional
        and family.group not in enabled_groups
        and not (wanted and key in wanted)
    }
    if optional_off:
        catalogue = [p for p in catalogue if p.family not in optional_off]
    allowed = {symbol.strip().title() for symbol in elements} if elements else None
    if allowed:
        catalogue = [phase for phase in catalogue
                     if elements_of(phase.formula) <= allowed]

    positions = [p.position for p in peaks]
    if spectrum_range is None:
        spectrum_range = (
            (min(positions) - 20.0, max(positions) + 20.0) if positions else (0.0, 0.0)
        )

    report = PhaseReport()
    report.elements = sorted(allowed) if allowed else []
    identifications: list[PhaseIdentification] = []

    for phase in catalogue:
        if wanted is not None and phase.family not in wanted:
            continue
        if phase.raman_silent:
            continue
        in_range = [
            band for band in phase.bands
            if spectrum_range[0] <= band.position <= spectrum_range[1]
        ]
        if not in_range:
            continue
        # Corroboration is judged on the strong lines, but the weak ones are
        # matched too and count as support. Judging on the strong set alone
        # made trigonal selenium — one strong line at 237 and one weak at
        # 143 — permanently uncorroborable even with both of them present.
        strong_bands = [
            band for band in in_range
            if any(abs(band.position - s) < 1e-6 for s in phase.strong)
        ]
        # When not one strong line was measurable, the minor ones are all
        # there is, and they are judged against each other so the phase can
        # still be reported as a lead. What it cannot do is corroborate: a
        # carbon spectrum measured from 1000 cm⁻¹ matches two of the three
        # g-C₃N₄ bands that sit on the D and the G, and calling that carbon
        # nitride would be an identification made entirely out of the bands
        # the two materials share.
        strong_in_range = strong_bands or in_range

        ident = PhaseIdentification(phase=phase, lines_expected=len(strong_in_range))
        ident.strong_out_of_range = not strong_bands
        ident.discriminating_expected = sum(
            1 for line in phase.discriminating
            if spectrum_range[0] <= line <= spectrum_range[1]
        )
        used: set[int] = set()
        strong_found = 0
        for band in in_range:
            best_index, best_offset = None, None
            for index, peak in enumerate(peaks):
                if index in used:
                    continue
                offset = peak.position - band.position
                if abs(offset) <= tol and (
                    best_offset is None or abs(offset) < abs(best_offset)
                ):
                    best_index, best_offset = index, offset
            if best_index is None:
                continue
            used.add(best_index)
            hit = PhaseHit(peak=peaks[best_index], line=band, offset=best_offset or 0.0)
            object.__setattr__(
                hit, "_discriminating",
                any(abs(band.position - d) < 1e-6 for d in phase.discriminating),
            )
            ident.hits.append(hit)
            ident.intensity += max(0.0, peaks[best_index].height)
            if hit.is_discriminating:
                ident.discriminating_found += 1
            if band in strong_in_range:
                strong_found += 1

        ident.strong_found = strong_found
        _apply_width_rule(ident, tol)
        if strong_found == 0:
            # Only weak lines matched. A phase whose strong lines are all
            # missing while a minor one lands on a peak is arithmetic, not
            # evidence: listing it would bury the real calls in noise. This
            # holds even when the minor line is a discriminating one — it
            # discriminates between polymorphs, it does not establish that
            # the composition is there at all.
            continue
        if ident.strong_out_of_range:
            ident.corroborated = False
            identifications.append(ident)
            continue
        if len(in_range) == 1:
            # A single catalogued line inside the measured range. Nothing can
            # corroborate it, so it is a lead unless the phase has no other
            # line anywhere — in which case there was never more to find.
            ident.corroborated = len(phase.bands) == 1 and ident.width_ok
        else:
            ident.corroborated = (
                strong_found / len(strong_in_range) >= CORROBORATION
                and (len(ident.hits) >= 2 or ident.width_confirmed)
                and ident.width_ok
            )
        identifications.append(ident)

    identifications = _resolve_conflicts(identifications)
    identifications.sort(key=lambda i: (-int(i.corroborated), -i.support, -i.intensity))
    report.identifications = identifications

    report.families = _family_verdicts(identifications, family_map)
    _flag_rbm_conflicts(report, rbm_window)
    _collect_xrd_questions(report)
    _blocked_by_range(report)
    _add_context(report)

    matched = {id(h.peak) for i in identifications if i.corroborated for h in i.hits}
    report.unmatched_peaks = [p for p in peaks if id(p) not in matched]
    report.near_misses = _near_misses(report.unmatched_peaks, catalogue)
    return report


#: How far from a catalogued line a peak can sit and still be worth
#: naming as a near miss, in cm⁻¹.
#:
#: Well outside the matching tolerance on purpose — this is not a match
#: and is never presented as one. It is the distance at which "look at
#: this card" is still useful advice: Raman positions of the same phase
#: move by tens of wavenumbers with crystallite size, strain and
#: stoichiometry, so 40 covers the real spread while staying far short of
#: the "any peak is near something" regime.
NEAR_MISS_CM = 40.0


def _near_misses(
    peaks: Sequence[PeakMeasurement], catalogue: Sequence["Phase"]
) -> dict[float, list[tuple[str, float, float]]]:
    """The catalogued lines nearest each unexplained peak."""
    out: dict[float, list[tuple[str, float, float]]] = {}
    for peak in peaks:
        candidates: list[tuple[str, float, float]] = []
        for phase in catalogue:
            for band in phase.bands:
                distance = abs(peak.position - band.position)
                if distance <= NEAR_MISS_CM:
                    candidates.append((phase.label, band.position, distance))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[2])
        # One line per phase, closest first: a phase with four bands in
        # the neighbourhood would otherwise fill the list on its own.
        seen: set[str] = set()
        trimmed = []
        for label, position, distance in candidates:
            if label in seen:
                continue
            seen.add(label)
            trimmed.append((label, position, distance))
            if len(trimmed) == 3:
                break
        out[float(peak.position)] = trimmed
    return out


def _apply_width_rule(ident: PhaseIdentification, tolerance: float) -> None:
    """Test a polymorph's band-width criterion, when it declares one.

    The test rejects the polymorph, not the composition: selenium is still
    selenium whether it is amorphous or monoclinic. When the matched peak
    has no measurable FWHM — a shoulder, or a band running off the edge of
    the window — the rule reports that it could not be checked instead of
    silently passing or silently failing.
    """
    rule = ident.phase.width_rule
    if not rule:
        return
    line = float(rule["line"])
    hit = next(
        (h for h in ident.hits if abs(h.line.position - line) <= tolerance + 1e-6),
        None,
    )
    if hit is None:
        return
    fwhm = hit.peak.fwhm
    if fwhm is None or fwhm <= 0.0:
        ident.width_note = (
            f"no se pudo comprobar la anchura de la banda de {line:g} cm⁻¹ "
            "(el pico no tiene FWHM medible), así que el polimorfo queda sin "
            "confirmar por esa vía"
        )
        return
    minimum = rule.get("min_fwhm")
    maximum = rule.get("max_fwhm")
    if minimum is not None and fwhm < float(minimum):
        ident.width_ok = False
        ident.width_note = (
            f"FWHM medida {fwhm:.1f} cm⁻¹ < {float(minimum):g} exigida. "
            + str(rule.get("reason", ""))
        )
    elif maximum is not None and fwhm > float(maximum):
        ident.width_ok = False
        ident.width_note = (
            f"FWHM medida {fwhm:.1f} cm⁻¹ > {float(maximum):g} admisible. "
            + str(rule.get("reason", ""))
        )
    elif minimum is not None:
        ident.width_confirmed = True
        ident.width_note = (
            f"banda ancha ({fwhm:.1f} cm⁻¹ ≥ {float(minimum):g}), demasiado "
            "para un modo de red estrecho: eso, y no la posición, es lo que "
            "identifica este polimorfo"
        )
    else:
        ident.width_note = (
            f"anchura compatible ({fwhm:.1f} cm⁻¹ ≤ {float(maximum):g}), pero "
            "un límite superior de anchura no corrobora nada por sí solo: lo "
            "cumple también cualquier RBM"
        )


def _resolve_conflicts(
    identifications: list[PhaseIdentification],
) -> list[PhaseIdentification]:
    """Drop a candidate whose every line is claimed better by another phase.

    Two phases with a line in the same place (β-FeSe at 181 and FeSe₂ at
    175, six wavenumbers apart) both match a peak in between. Reporting
    both is right — that ambiguity is real and the report says so — but
    reporting a phase whose *only* support is a peak another phase explains
    better is noise.

    There are two versions of that, and they need different strictness.

    **An uncorroborated candidate** goes as soon as another *corroborated*
    phase claims every peak it matched, unless it has a discriminating line
    of its own.

    **A corroborated one** takes more. Some phases of entirely different
    chemistry sit a few wavenumbers apart on every band they have:
    magnetite at 668/540/310 and β-MnO₂ at 665/535, or α-S₈ at 473/219/153
    and α-MoO₃ at 471/217/158. Both really do match, both really do
    corroborate, and a magnetite sample was being reported as manganese
    dioxide. It is dropped only when another phase *of a different family*
    explains the same peaks with strictly more lines and a smaller mean
    offset — three conditions, all of them, because the alternative to a
    strict rule here is losing a genuine minority phase.

    Between polymorphs of one family the rule is deliberately not applied:
    there, reporting both and letting the family verdict say whether the
    polymorph can be named *is* the answer.
    """
    keep: list[PhaseIdentification] = []
    for ident in identifications:
        peaks_here = {id(h.peak) for h in ident.hits}
        if ident.corroborated:
            eclipsed = any(
                other is not ident
                and other.corroborated
                and other.phase.family != ident.phase.family
                and peaks_here < {id(h.peak) for h in other.hits}
                and len(other.hits) > len(ident.hits)
                and abs(other.mean_offset) < abs(ident.mean_offset)
                for other in identifications
            )
            if not eclipsed:
                keep.append(ident)
            continue
        if ident.discriminated:
            keep.append(ident)
            continue
        dominated = any(
            other is not ident
            and other.corroborated
            and peaks_here <= {id(h.peak) for h in other.hits}
            for other in identifications
        )
        if not dominated:
            keep.append(ident)
    return keep


def _family_verdicts(
    identifications: Sequence[PhaseIdentification], families: dict[str, Family]
) -> list[FamilyVerdict]:
    """Decide, per composition, whether the polymorph can be named."""
    by_family: dict[str, list[PhaseIdentification]] = {}
    for ident in identifications:
        by_family.setdefault(ident.phase.family, []).append(ident)

    verdicts: list[FamilyVerdict] = []
    for key, candidates in by_family.items():
        family = families[key]
        verdict = FamilyVerdict(family=family, candidates=candidates)
        corroborated = [c for c in candidates if c.corroborated]
        discriminated = [c for c in corroborated if c.discriminated]

        if len(discriminated) == 1:
            verdict.best = discriminated[0]
            verdict.resolved = True
            winner = discriminated[0]
            if winner.discriminating_found:
                how = (
                    f"aparecen {winner.discriminating_found} línea(s) propias "
                    "de este polimorfo"
                )
            else:
                how = "lo decide la anchura de banda, no la posición"
            verdict.reason = f"Resuelto: {how}. {family.resolved_by}"
        elif len(discriminated) > 1:
            verdict.resolved = False
            verdict.best = max(discriminated, key=lambda c: (c.support, c.intensity))
            verdict.reason = (
                "Hay líneas discriminantes de más de un polimorfo a la vez: o "
                "coexisten en la muestra o alguna coincidencia es fortuita. "
                f"{family.resolved_by}"
            )
        elif corroborated:
            verdict.resolved = False
            verdict.best = max(corroborated, key=lambda c: (c.support, c.intensity))
            verdict.reason = (
                "La composición está, pero ninguna línea discriminante la "
                f"separa de sus polimorfos. {family.resolved_by}"
            )
        else:
            verdict.resolved = False
            verdict.reason = (
                "Solo coincidencias aisladas, sin corroborar. En esta región "
                "una coincidencia suelta es más probable que la fase."
            )
            verdict.tentative = True
        rejected = [c for c in candidates if not c.width_ok]
        if rejected:
            verdict.reason += " " + " ".join(
                f"Descartado {c.phase.label}: {c.width_note}" for c in rejected
            )
        verdicts.append(verdict)
    verdicts.sort(key=lambda v: (not v.resolved, v.family.key))
    return verdicts


def _flag_rbm_conflicts(report: PhaseReport, rbm_window: tuple[float, float]) -> None:
    """Record matched lines that a diameter calculation must not convert.

    Corroborated matches are excluded outright. Uncorroborated ones are
    recorded separately as suspects rather than dropped: the evidence is
    too thin to name the phase and far too strong to convert the peak into
    a diameter without a word. Dropping them is what let elemental
    selenium's 237 cm-1 -- a line the catalogue holds exactly, on a phase
    with only one other line, usually below the filter cut -- come back as
    a 1.0 nm nanotube.
    """
    seen: set[float] = set()
    for ident in report.identifications:
        if not ident.corroborated:
            continue
        for hit in ident.hits:
            position = hit.peak.position
            if not (rbm_window[0] <= position <= rbm_window[1]):
                continue
            if any(abs(position - p) < 1e-6 for p in seen):
                continue
            seen.add(position)
            report.rbm_conflicts.append((position, ident.phase.label))
    report.rbm_conflicts.sort()

    # A suspect has to be worth interrupting for, or it interrupts every
    # real nanotube spectrum instead. With three dozen phases catalogued,
    # nearly any genuine RBM lands within tolerance of *some* single
    # line: a clean SWCNT at 220 cm-1 drew a delta-FeSe hexagonal flag,
    # and that phase's own catalogue entry says it has no reliable Raman
    # literature. So without a stated composition only a high-confidence
    # phase's own discriminating line raises one.
    #
    # WITH a stated composition the caution is misplaced, and silently so.
    # Cementite's bands at 212 and 280 sit inside the RBM window and its
    # entry is low confidence -- because cementite's Raman cross-section
    # is poor, not because the positions are doubtful -- so a CVD sample
    # grown on an iron catalyst, which is the case this module was written
    # for, got no warning at all while the program quietly converted 212
    # into a 1.2 nm tube. Once the user has said the sample contains iron,
    # any catalogued line of an allowed phase inside the window is worth
    # raising, and the report states the confidence rather than hiding
    # behind it.
    restricted = bool(report.elements)
    for ident in report.identifications:
        if ident.corroborated:
            continue
        if not restricted and ident.phase.confidence != "high":
            continue
        for hit in ident.hits:
            position = hit.peak.position
            if not (rbm_window[0] <= position <= rbm_window[1]):
                continue
            if not restricted and not hit.is_discriminating:
                continue
            if any(abs(position - p) < 1e-6 for p in seen):
                continue
            seen.add(position)
            label = ident.phase.label
            if restricted and ident.phase.confidence != "high":
                label += f" — confianza de la referencia: {ident.phase.confidence}"
            report.rbm_suspects.append((position, label))
    report.rbm_suspects.sort()


def _collect_xrd_questions(report: PhaseReport) -> None:
    """Name the diffraction measurement that would settle each open call."""
    for verdict in report.families:
        if verdict.resolved or not verdict.present:
            continue
        for candidate in verdict.candidates:
            if candidate.corroborated:
                report.xrd_questions.append(candidate.phase.xrd_advice())
    for ident in report.identifications:
        if ident.corroborated and ident.phase.confidence == "low":
            advice = ident.phase.xrd_advice()
            if advice not in report.xrd_questions:
                report.xrd_questions.append(advice)


def _blocked_by_range(report: PhaseReport) -> None:
    """Name the phases that only a wider scan could confirm or rule out."""
    blocked = [
        ident for ident in report.identifications
        if ident.strong_out_of_range and ident.hits
    ]
    if not blocked:
        return
    for ident in blocked[:3]:
        lines = ", ".join(f"{line:g}" for line in ident.phase.strong)
        report.warnings.append(
            f"{ident.phase.label} coincide en {len(ident.hits)} banda(s) "
            "menor(es), pero ninguna de sus líneas fuertes "
            f"({lines} cm⁻¹) entra en el rango medido. Ni se confirma ni se "
            "descarta: para eso hay que medir esa región"
        )


def _add_context(report: PhaseReport) -> None:
    """Say what the identification implies for the rest of the analysis."""
    present = {i.phase.key for i in report.identifications if i.corroborated}
    if report.rbm_suspects:
        listed = ", ".join(
            f"{position:.0f} cm⁻¹ ({label})"
            for position, label in report.rbm_suspects[:4]
        )
        report.warnings.append(
            f"{len(report.rbm_suspects)} pico(s) de la ventana RBM coinciden "
            f"con una línea catalogada sin que nada la corrobore: {listed}. "
            "No basta para nombrar la fase y sobra para no convertirlos en "
            "diámetros a ciegas. Míralos antes de citar esos tubos: si la "
            "fase está, sus otras líneas suelen quedar por debajo del corte "
            "del filtro, así que baja el corte o mide a dos láseres — un RBM "
            "cambia de intensidad con la resonancia y una línea de fase no."
        )
    if report.rbm_conflicts:
        report.warnings.append(
            f"{len(report.rbm_conflicts)} línea(s) de fase caen en la ventana "
            "RBM. Se han excluido del cálculo de diámetros: convertirlas con "
            "ω = A/d + B habría dado diámetros de nanotubo inventados. Si tu "
            "muestra además TIENE nanotubos de pared única, mide a dos "
            "láseres: el RBM cambia de intensidad con la resonancia y estas "
            "líneas no"
        )
    if {"Se_trigonal", "Se_monoclinic", "Se_amorphous"} & present:
        report.warnings.append(
            "hay selenio en fase separada. En una muestra dopada con selenio "
            "eso significa que ese selenio NO está en la red del carbono: es "
            "precursor segregado, no dopado sustitucional. Contrástalo con "
            "XPS (Se3d) antes de citar un porcentaje de dopado"
        )
    if "Se_amorphous" in present:
        report.warnings.append(
            "el selenio amorfo cristaliza bajo el láser con pocos mW. Si la "
            "banda de ~252 cm⁻¹ se estrecha o migra hacia 237 cm⁻¹ entre "
            "medidas repetidas, la fase la estás creando tú. Baja la potencia "
            "y repite sobre un punto virgen"
        )
    if {"FeSe_tetragonal", "FeSe_hexagonal"} <= present:
        report.warnings.append(
            "aparecen líneas compatibles con las DOS fases del FeSe. Es "
            "perfectamente posible que coexistan (es lo normal si el recocido "
            "fue corto), pero la referencia Raman de la fase hexagonal tiene "
            "confianza baja. No publiques la coexistencia solo con Raman"
        )
    if {"FeSe_tetragonal", "FeSe2_marcasite"} <= present:
        report.warnings.append(
            "β-FeSe y FeSe₂ tienen líneas separadas solo 6 cm⁻¹ (181 y 175). "
            "Con resolución peor que ~4 cm⁻¹ no son separables y el ajuste "
            "puede repartirlas a capricho. Comprueba el paso de tu espectro "
            "antes de creerte la coexistencia"
        )
    if "Fe3C_cementite" in present:
        report.warnings.append(
            "cementita detectada. Es la fase que suele haber realmente dentro "
            "de las nanopartículas de la punta de un nanotubo crecido con "
            "hierro. Su sección eficaz Raman es muy baja: verla es "
            "informativo, NO verla no prueba nada"
        )


__all__ = [
    "AMORPHOUS_MIN_FWHM",
    "CORROBORATION",
    "DEFAULT_TOLERANCE",
    "Family",
    "FamilyVerdict",
    "Phase",
    "PhaseBand",
    "PhaseDatabaseError",
    "PhaseHit",
    "PhaseIdentification",
    "PhaseReport",
    "RBM_WINDOW",
    "elements_of",
    "find_phases",
    "delete_user_phase",
    "match_tolerance",
    "save_user_phase",
    "user_phase_file",
    "load_families",
    "load_phases",
]
