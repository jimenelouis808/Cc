"""Transition-metal chalcogenides: S, Se and Te of Mo, W, Ti, Nb, Ta and Fe.

A separate physics from the carbon side of this package, sharing its
machinery. The spectra look nothing alike — two sharp modes a few cm⁻¹ wide
instead of broad overlapping envelopes — and so does the analysis.

**Not all of them are 2H and not all of them are layered.** MoS₂, WS₂,
MoSe₂, WSe₂ and MoTe₂ are the prismatic 2H semiconductors the layer-count
method was built for. TiS₂ and TiSe₂ are octahedral 1T, WTe₂ is Td, TaS₂
comes in both 2H and a 1T with charge-density-wave modes below 130 cm⁻¹,
NbSe₂ and TaSe₂ are CDW metals whose two modes are 6 cm⁻¹ apart, and pyrite
FeS₂ and marcasite FeSe₂ have no layers at all. Each entry carries what it
is, and the analysis declines to count layers where the question has no
answer rather than producing a number.

**The measurement is a separation, not a position.** Stacking layers softens
the in-plane E¹₂g mode and stiffens the out-of-plane A₁g mode, so the gap
between them grows monotonically with layer count. In MoS₂ it runs from
about 19 cm⁻¹ for a monolayer to 25 cm⁻¹ for bulk. Because it is a
*difference*, any common offset in the axis calibration cancels — which
makes it far more robust than either position on its own, and is why this
is the standard method.

The two modes move in opposite directions for a physical reason worth
knowing: the A₁g displaces the chalcogen atoms perpendicular to the layer,
so a neighbouring layer pushes back and stiffens it; the E¹₂g moves atoms
within the plane, where the dominant effect is not van der Waals contact
but increased dielectric screening of the long-range Coulomb interaction
between effective charges, which softens it.

**Where the method fails, this module says so.** In WSe₂ the two modes are
nearly degenerate near 250 cm⁻¹ and an ordinary spectrometer sees one band;
layer counting there rests on the B¹₂g mode at 308 cm⁻¹, which symmetry
forbids in a monolayer. In MoSe₂ and MoTe₂ the separation barely changes
with thickness and the same B¹₂g argument is used instead.

**Resolution matters more than for carbon.** These bands are 2–6 cm⁻¹ wide
and the layer-count boundaries are 2–3 cm⁻¹ apart. A spectrometer that
cannot resolve 3 cm⁻¹ cannot count layers, however good the fit looks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence


from ..core.peaks import find_peaks
from ..core.preprocess import preprocess
from ..core.spectrum import Spectrum
from .heterostructure import HeterostructureResult, analyse_heterostructure
from ..database.loader import DATA_DIR
from ..models.fitting import FitModel, FitResult, PeakSpec, fit_model

#: Instrumental resolution, in cm⁻¹, below which layer counting by
#: separation is not meaningful. The class boundaries are 2–3 cm⁻¹ apart.
RESOLUTION_LIMIT = 3.0


@dataclass(frozen=True)
class TMDMode:
    """One vibrational mode of a TMD."""

    key: str
    label: str
    position: float
    window: tuple[float, float]
    discriminating: bool = False
    """Whether this mode separates the material from its neighbours in the
    catalogue. With fifteen materials many windows overlap — the A₁g of
    MoSe₂ at 240 and of TaSe₂ at 234, the A₁g of TiSe₂ at 198 and the B₁g
    of β-FeSe at 196 — and a sum of coincidences on shared windows is not
    an identification."""


@dataclass(frozen=True)
class TMDMaterial:
    """One transition-metal chalcogenide from the database."""

    key: str
    label: str
    formula: str
    modes: dict[str, TMDMode]
    separation_by_layers: dict[str, tuple[float, float]]
    separation_source: str
    confidence: str
    notes: str
    metal: str = ""
    chalcogen: str = ""
    polytype: str = ""
    layered: bool = True
    """Whether the structure has van der Waals layers at all. Pyrite FeS₂
    and marcasite FeSe₂ do not, and asking how many layers they have is not
    a question with an answer."""
    source: str = ""

    @property
    def counts_layers_by_separation(self) -> bool:
        """Whether the E₂g–A₁g gap is a usable layer counter here."""
        return bool(self.separation_by_layers)

    @property
    def discriminating_modes(self) -> list[TMDMode]:
        return [m for m in self.modes.values() if m.discriminating]

    def mode(self, key: str) -> Optional[TMDMode]:
        return self.modes.get(key)


@dataclass
class TMDResult:
    """The analysis of one TMD spectrum."""

    material: Optional[str]
    label: str
    positions: dict[str, float] = field(default_factory=dict)
    widths: dict[str, float] = field(default_factory=dict)
    separation: Optional[float] = None
    layers: Optional[str] = None
    layer_reason: str = ""
    phase: str = "2H"
    phase_reason: str = ""
    fit: Optional[FitResult] = None
    candidates: list[tuple[str, float]] = field(default_factory=list)
    oxides: Optional[HeterostructureResult] = None
    """Metal oxides accompanying the dichalcogenide, and what their presence
    implies for the interface. See
    :mod:`ramancarbon.analysis.heterostructure`."""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Flat row for the batch table."""
        row: dict = {
            "nombre": "",
            "material": self.label,
            "fase": self.phase,
            "capas": self.layers,
            "separacion_cm-1": self.separation,
        }
        if self.oxides is not None and self.oxides.found_anything:
            row["composicion"] = self.oxides.composition()
            row["oxidos"] = ";".join(o.key for o in self.oxides.oxides_present)
            row["indice_oxidacion"] = self.oxides.oxidation_index
        for key, value in self.positions.items():
            row[f"pos_{key}"] = value
        for key, value in self.widths.items():
            row[f"fwhm_{key}"] = value
        return row

    def summary(self) -> str:
        lines = [f"Material: {self.label}"]
        if self.candidates and len(self.candidates) > 1:
            others = ", ".join(
                f"{k} ({s:.0f})" for k, s in self.candidates[1:4]
            )
            lines.append(f"  otros candidatos: {others}")
        if self.positions:
            lines.append("")
            lines.append("Modos:")
            for key, value in sorted(self.positions.items(), key=lambda kv: kv[1]):
                width = self.widths.get(key)
                text = f"  {key:<10s} {value:8.2f} cm⁻¹"
                if width:
                    text += f"   FWHM {width:5.2f}"
                lines.append(text)
        if self.separation is not None:
            lines.append("")
            lines.append(f"Separación E₂g–A₁g: {self.separation:.2f} cm⁻¹")
        if self.layers:
            lines.append(f"Número de capas   : {self.layers}")
            lines.append(f"  {self.layer_reason}")
        if self.oxides is not None and self.oxides.enabled and (
            self.oxides.found_anything or self.oxides.warnings
        ):
            lines.append("")
            lines.append(self.oxides.summary())
        lines.append("")
        lines.append(f"Fase: {self.phase}")
        if self.phase_reason:
            lines.append(f"  {self.phase_reason}")
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


@lru_cache(maxsize=2)
def _load(directory: str) -> tuple[dict, tuple[TMDMaterial, ...]]:
    path = Path(directory) / "tmd.json"
    if not path.is_file():
        raise FileNotFoundError(f"falta la base de datos de TMD: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    materials = []
    for entry in payload["materials"]:
        modes = {
            key: TMDMode(
                key=key,
                label=value.get("label", key),
                position=float(value["position"]),
                window=(float(value["window"][0]), float(value["window"][1])),
                discriminating=bool(value.get("discriminating", False)),
            )
            for key, value in entry["modes"].items()
        }
        materials.append(
            TMDMaterial(
                key=entry["key"],
                label=entry["label"],
                formula=entry.get("formula", ""),
                modes=modes,
                separation_by_layers={
                    k: (float(v[0]), float(v[1]))
                    for k, v in entry.get("separation_by_layers", {}).items()
                },
                separation_source=entry.get("separation_source", ""),
                confidence=entry.get("confidence", "unknown"),
                notes=entry.get("notes", ""),
                metal=entry.get("metal", ""),
                chalcogen=entry.get("chalcogen", ""),
                polytype=entry.get("polytype", ""),
                layered=bool(entry.get("layered", True)),
                source=entry.get("source", ""),
            )
        )
    return payload, tuple(materials)


def load_tmd_database(directory: Optional[str | Path] = None):
    """The TMD materials and the raw payload behind them."""
    return _load(str(Path(directory) if directory else DATA_DIR))


def tmd_materials(directory: Optional[str | Path] = None) -> list[TMDMaterial]:
    """Every TMD in the database."""
    return list(_load(str(Path(directory) if directory else DATA_DIR))[1])


def identify_material(
    peaks: Sequence,
    directory: Optional[str | Path] = None,
    spectrum_range: Optional[tuple[float, float]] = None,
) -> list[tuple[str, float]]:
    """Score each material by how well the observed peaks match its modes.

    A simple, explainable score: each mode found inside its window scores
    one, weighted by how close it lands to the catalogue position.

    **With one gate.** When the catalogue marks some of a material's modes
    as *discriminating*, at least one of them must be found or the material
    scores nothing at all. That gate is what keeps the library honest as it
    grows. With five materials in well-separated regions a bare sum was
    enough; with fifteen it is not, because the windows overlap — the A₁g
    of MoSe₂ sits at 240 and that of TaSe₂ at 234, the A₁g of TiSe₂ at 198
    lands on the B₁g of β-FeSe at 196, and the two-phonon band of NbSe₂ is
    broad enough to swallow half the RBM window. Without the gate a MoSe₂
    spectrum scores on TaSe₂ and NbSe₂ by coincidence alone.

    It is the same rule the oxide side already uses for signature lines
    (:mod:`ramancarbon.analysis.heterostructure`) and the phase side for
    discriminating lines (:mod:`ramancarbon.analysis.phases`).

    **And one accounting rule.** An observed peak is claimed by at most one
    mode. Some materials have modes whose windows overlap each other — in
    2H-NbSe₂ the A₁g and E¹₂g are 6 cm⁻¹ apart, closer than either window is
    wide — and letting one peak satisfy both of them scores a single band as
    though it were two. That is not a small effect: it was enough to make a
    MoTe₂ spectrum, whose E₂g sits at 234, score higher as NbSe₂ than as
    itself.

    Parameters
    ----------
    peaks:
        Peaks detected over the whole measured range.
    directory:
        Where to load the TMD database from.
    spectrum_range:
        The measured range. When given, the gate above is only applied to
        materials that actually had a discriminating mode *inside* it. A
        band you never measured is not evidence against a material: the
        discriminating modes of 1T-TaS₂ are at 63 and 75 cm⁻¹, and a
        spectrum that starts at 100 — which is most of them, with an edge
        filter — cannot see them. Such a material is still scored, and
        :func:`analyse_tmd` says out loud that the identification rests on
        shared bands. Without the range this is unknowable, so the gate
        applies to everyone.

    Returns
    -------
    list[(str, float)]
        Material keys with scores, best first.
    """
    materials = tmd_materials(directory)
    scores: list[tuple[str, float]] = []
    for material in materials:
        # Every (mode, peak) pair that falls inside the mode's window, best
        # first; then take them greedily, retiring both sides each time.
        pairs = []
        for mode in material.modes.values():
            width = max(mode.window[1] - mode.window[0], 1e-9)
            for index, peak in enumerate(peaks):
                if not mode.window[0] <= peak.position <= mode.window[1]:
                    continue
                closeness = 1.0 - abs(peak.position - mode.position) / width
                if closeness > 0:
                    pairs.append((closeness, mode.key, index, mode.discriminating))
        pairs.sort(key=lambda item: -item[0])
        used_modes: set[str] = set()
        used_peaks: set[int] = set()
        score = 0.0
        found_discriminating = False
        for closeness, mode_key, index, discriminating in pairs:
            if mode_key in used_modes or index in used_peaks:
                continue
            used_modes.add(mode_key)
            used_peaks.add(index)
            score += closeness
            found_discriminating = found_discriminating or discriminating
        if material.discriminating_modes and not found_discriminating:
            if spectrum_range is None or _in_range(material, spectrum_range):
                continue
        if score > 0:
            scores.append((material.key, score))
    scores.sort(key=lambda item: -item[1])
    return scores


def _in_range(
    material: TMDMaterial, spectrum_range: tuple[float, float]
) -> bool:
    """Whether any of the material's discriminating modes was measurable."""
    low, high = spectrum_range
    return any(
        low <= mode.window[0] and mode.window[1] <= high
        for mode in material.discriminating_modes
    )


def count_layers(
    material: TMDMaterial, separation: float
) -> tuple[Optional[str], str]:
    """Layer count from the E₂g–A₁g separation.

    Returns ``(layers, reasoning)``. ``layers`` is ``"1"``, ``"2"``, …,
    ``"bulk"``, or ``None`` when the separation falls outside every
    catalogued range or the material does not support the method.
    """
    if not material.counts_layers_by_separation:
        return None, (
            f"en {material.label} los dos modos están casi degenerados y la "
            "separación no cuenta capas; se usa la presencia del modo B¹₂g"
        )
    table = material.separation_by_layers
    for name, (low, high) in table.items():
        if low <= separation <= high:
            label = "bulk" if name == "bulk" else f"{name} capa(s)"
            return name, (
                f"la separación de {separation:.2f} cm⁻¹ cae en el rango "
                f"{low:g}–{high:g} de {label} ({material.separation_source})"
            )

    values = [v for pair in table.values() for v in pair]
    if separation < min(values):
        return None, (
            f"la separación de {separation:.2f} cm⁻¹ está por DEBAJO de la de "
            f"monocapa ({min(values):g}). Puede ser deformación por tracción, "
            "que ablanda el E₂g y ensancha el hueco al revés, o un ajuste malo "
            "de una de las dos bandas"
        )
    return None, (
        f"la separación de {separation:.2f} cm⁻¹ supera la del bulk "
        f"({max(values):g}); por encima de ~5 capas el método satura y ya no "
        "distingue"
    )


def detect_phase(
    peaks: Sequence, material_key: str, directory: Optional[str | Path] = None
) -> tuple[str, str]:
    """2H or 1T′, from the presence of the J modes.

    The 1T′ distortion doubles the unit cell and produces modes — J1, J2,
    J3 — that simply do not exist in the 2H phase. Seeing them is positive
    evidence of metallic phase. **Not** seeing them is much weaker
    evidence: a small 1T′ fraction sits below the noise.
    """
    payload, _ = _load(str(Path(directory) if directory else DATA_DIR))
    phase = payload["phases"]["1T_prime"]
    markers = phase.get("marker_bands", {}).get(material_key)
    if not markers:
        return "2H", (
            f"no hay bandas J catalogadas para {material_key}; se asume 2H"
        )
    labels = phase.get("marker_labels", ["J1", "J2", "J3"])
    found = []
    for line, name in zip(markers, labels):
        if any(abs(p.position - line) <= 6.0 for p in peaks):
            found.append(f"{name} ({line:g})")
    if len(found) >= 2:
        return "1T'", (
            f"se ven los modos {', '.join(found)}, que no existen en la fase "
            "2H: hay fase metálica 1T′ presente"
        )
    if len(found) == 1:
        return "2H (posible 1T′)", (
            f"solo se ve {found[0]} de los tres modos J. Una sola coincidencia "
            "no basta; mide con más tiempo de integración antes de afirmar que "
            "hay fase 1T′"
        )
    return "2H", (
        "no se ven los modos J1–J3, así que no hay evidencia de fase 1T′. Eso "
        "NO descarta una fracción pequeña: por debajo del ruido no se ve"
    )


def analyse_tmd(
    spectrum: Spectrum,
    material: Optional[str] = None,
    preprocess_kwargs: Optional[dict] = None,
    check_oxides: bool = True,
    directory: Optional[str | Path] = None,
) -> TMDResult:
    """Analyse a TMD spectrum: material, layer count, phase, oxide content.

    Parameters
    ----------
    spectrum:
        The raw spectrum. TMD modes sit between 100 and 500 cm⁻¹, so the
        spectrum must reach down there.
    material:
        Force a material instead of identifying one.
    check_oxides:
        Look for the metal oxides this chalcogenide can produce, and read
        the host's mode shifts as interface evidence. On by default: an
        oxide is a fact about the sample, and one of its three possible
        origins is the laser you are measuring with.
    preprocess_kwargs:
        Passed to :func:`~ramancarbon.core.preprocess.preprocess`.
    directory:
        Where to load the TMD database from.

    Returns
    -------
    TMDResult
    """
    settings = {"baseline_method": "asls", **(preprocess_kwargs or {})}
    processed, _ = preprocess(spectrum, **settings)
    warnings: list[str] = []

    if not processed.covers(150.0, 450.0, fraction=0.6):
        warnings.append(
            f"el espectro va de {processed.range[0]:.0f} a "
            f"{processed.range[1]:.0f} cm⁻¹ y los modos de un TMD están entre "
            "100 y 500. Mide esa región"
        )

    if processed.step > RESOLUTION_LIMIT:
        warnings.append(
            f"el paso de muestreo es de {processed.step:.1f} cm⁻¹, y las "
            f"fronteras entre números de capa están a 2–3 cm⁻¹ unas de otras. "
            "Con esta resolución no se pueden contar capas por la separación, "
            "por muy bien que se vea el ajuste"
        )

    peaks = find_peaks(processed, min_distance_cm=4.0, min_fwhm_cm=1.5)
    scores = identify_material(peaks, directory, spectrum_range=processed.range)
    materials = {m.key: m for m in tmd_materials(directory)}

    chosen_key = material or (scores[0][0] if scores else None)
    if chosen_key is None or chosen_key not in materials:
        return TMDResult(
            material=None,
            label="no identificado",
            candidates=scores,
            warnings=warnings
            + [
                "no se reconoce ningún TMD de la base de datos. Comprueba que "
                "el espectro cubre 100–500 cm⁻¹ y que la muestra es uno de: "
                + ", ".join(sorted(materials))
            ],
        )

    chosen = materials[chosen_key]
    if chosen.discriminating_modes and not _in_range(chosen, processed.range):
        lines = ", ".join(
            f"{m.position:g}" for m in chosen.discriminating_modes
        )
        warnings.append(
            f"ninguna de las bandas que distinguen el {chosen.label} de sus "
            f"vecinos ({lines} cm⁻¹) entra en el rango medido "
            f"({processed.range[0]:.0f}–{processed.range[1]:.0f} cm⁻¹). La "
            "identificación se apoya en bandas que comparte con otras fases: "
            "mira la lista de candidatos y, si puedes, amplía el rango"
        )
    fit = _fit_modes(processed, chosen)
    positions = {p.name: p.peak_position for p in fit.peaks} if fit else {}
    widths = {p.name: p.fwhm for p in fit.peaks} if fit else {}

    separation = None
    layers = None
    layer_reason = ""
    if "E2g" in positions and "A1g" in positions:
        separation = abs(positions["A1g"] - positions["E2g"])
        layers, layer_reason = count_layers(chosen, separation)
    elif chosen.counts_layers_by_separation:
        layer_reason = "no se han podido ajustar los dos modos principales"
    else:
        layers, layer_reason = _layers_from_b12g(chosen, positions, peaks)

    phase, phase_reason = detect_phase(peaks, chosen_key, directory)
    warnings.extend(_quality_warnings(chosen, widths, directory))

    oxides: Optional[HeterostructureResult] = None
    if check_oxides:
        oxides = analyse_heterostructure(
            peaks,
            chosen_key,
            mode_positions=positions,
            reference_positions={k: m.position for k, m in chosen.modes.items()},
            spectrum_range=processed.range,
            directory=directory,
        )
        warnings.extend(f"óxidos: {w}" for w in oxides.warnings)
    else:
        oxides = HeterostructureResult(chalcogenide=chosen_key, enabled=False)

    return TMDResult(
        material=chosen_key,
        label=chosen.label,
        positions=positions,
        widths=widths,
        separation=separation,
        layers=layers,
        layer_reason=layer_reason,
        phase=phase,
        phase_reason=phase_reason,
        fit=fit,
        candidates=scores,
        oxides=oxides,
        warnings=warnings,
    )


def _fit_modes(spectrum: Spectrum, material: TMDMaterial) -> Optional[FitResult]:
    """Fit every catalogued mode that the spectrum covers.

    Lorentzian, not pseudo-Voigt: these are sharp, well-defined phonons of
    a crystal with a real lifetime, and a free mixing fraction on a band
    only a few points wide would trade against the width without adding
    anything.
    """
    specs: list[PeakSpec] = []
    for mode in material.modes.values():
        if not spectrum.covers(*mode.window, fraction=0.8):
            continue
        observed = spectrum.max_in(*mode.window)
        if observed is None or observed[1] <= 0:
            continue
        specs.append(
            PeakSpec(
                name=mode.key,
                profile="lorentzian",
                centre=observed[0],
                height=observed[1],
                fwhm=4.0,
                centre_bounds=mode.window,
                height_bounds=(0.0, observed[1] * 20.0),
                fwhm_bounds=(0.8, 40.0),
                band=mode.key,
            )
        )
    if not specs:
        return None
    low = min(s.centre_bounds[0] for s in specs) - 20.0
    high = max(s.centre_bounds[1] for s in specs) + 20.0
    low, high = max(low, spectrum.range[0]), min(high, spectrum.range[1])
    try:
        return fit_model(
            spectrum,
            FitModel(peaks=specs, window=(low, high), background="linear", name="tmd"),
        )
    except ValueError:
        return None


def _layers_from_b12g(
    material: TMDMaterial, positions: dict[str, float], peaks: Sequence
) -> tuple[Optional[str], str]:
    """Layer count from the B¹₂g mode, for materials where the gap fails.

    B¹₂g is forbidden by symmetry in a monolayer and appears from two
    layers up, so its presence and absence carry opposite information —
    but not equally: absence could also mean it is simply weak.
    """
    if not material.layered:
        return None, (
            f"el {material.label} no es un material laminar de van der Waals "
            "—no tiene capas que contar—, así que la pregunta no tiene "
            "respuesta y el programa no se la inventa"
        )
    mode = material.mode("B12g")
    if mode is None:
        return None, (
            "no hay un modo B¹₂g catalogado para este material, así que no hay "
            "forma de contar capas por Raman aquí. Lo que sí cambia con el "
            "espesor es la anchura de las bandas, y eso no es un contador"
        )
    present = any(mode.window[0] <= p.position <= mode.window[1] for p in peaks)
    if present:
        return "≥2", (
            f"se ve el modo B¹₂g cerca de {mode.position:g} cm⁻¹, prohibido por "
            "simetría en monocapa: hay al menos dos capas"
        )
    return "1 (probable)", (
        f"no se ve el modo B¹₂g de {mode.position:g} cm⁻¹, que aparece desde "
        "dos capas. Compatible con monocapa, aunque una señal débil también "
        "podría quedar bajo el ruido"
    )


def _quality_warnings(
    material: TMDMaterial, widths: dict[str, float], directory
) -> list[str]:
    payload, _ = _load(str(Path(directory) if directory else DATA_DIR))
    good = payload.get("quality_indicators", {})
    low, high = good.get("A1g_fwhm_good", [2.0, 6.0])
    out: list[str] = []
    for key in ("A1g", "E2g"):
        width = widths.get(key)
        if width is None:
            continue
        if width > high * 2:
            out.append(
                f"el modo {key} tiene una anchura de {width:.1f} cm⁻¹, muy por "
                f"encima de los {low:g}–{high:g} de un cristal limpio. Material "
                "policristalino, dominios pequeños de CVD, o dañado"
            )
    if material.confidence == "low":
        out.append(
            f"los datos de {material.label} en la base tienen confianza baja: "
            "hay poca literatura cuantitativa y los rangos son orientativos"
        )
    return out


__all__ = [
    "RESOLUTION_LIMIT",
    "TMDMaterial",
    "TMDMode",
    "TMDResult",
    "analyse_tmd",
    "count_layers",
    "detect_phase",
    "identify_material",
    "load_tmd_database",
    "tmd_materials",
]
