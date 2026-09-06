"""Deciding what the sample is: SWCNT, DWCNT, MWCNT, graphene, oxide…

This is a **rule-based, evidence-reporting** classifier, not a black box.
Each rule contributes a signed, weighted piece of evidence with a sentence
explaining it, and the final answer is accompanied by the whole list. For a
measurement technique whose conclusions end up in a methods section, being
able to read *why* the software said "DWCNT" is worth more than a
probability with no provenance.

The physics the rules encode, in decreasing order of how decisive it is:

**RBM presence is close to definitive for wall count.** A radial breathing
mode requires a small-diameter tube: its frequency goes as 1/d and its
intensity falls with d, so above ~2.5 nm it drops below the Rayleigh cutoff
of an ordinary spectrometer. Present ⇒ single- or double-walled (or a
thin-walled multi-wall). Absent ⇒ multi-walled, graphitic or amorphous —
**but only if the spectrum actually reached that low**. A spectrum starting
at 400 cm⁻¹ says nothing about RBMs, and the classifier refuses to use the
absence as evidence in that case. This one check is the difference between
a classifier that works and one that confidently calls every 500–3000 cm⁻¹
spectrum multi-walled.

**Paired RBMs separate DWCNT from SWCNT.** Two concentric walls differ in
diameter by twice the wall spacing, 0.66–0.72 nm. A sample with RBM peaks
that pair up that way is double-walled; one whose RBMs do not pair is a
mixture of single-walled diameters. Nothing else in a single Raman spectrum
distinguishes those two cases.

**The G band's shape separates single- from multi-walled.** A resolved,
narrow G⁻/G⁺ doublet is curvature splitting in a single wall. A single
broad G with a shoulder above it is G + D′ in multi-walled or graphitic
material. Confusing the two turns a defect band into a tube diameter.

**The 2D band counts graphene layers**, by lineshape rather than by ratio.

Everything else — I_D/I_G, band widths, the G position — is corroboration,
weighted low, because every one of them is dominated by sample quality
rather than by identity. A damaged SWCNT sample with I_D/I_G = 1.2 is still
SWCNT material.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..core.peaks import PeakMeasurement
from ..core.spectrum import Spectrum
from ..database import Database, Material, load_database
from .assignment import Assignment
from .diameter import WallPair, find_wall_pairs, rbm_to_diameter

#: Weight of each kind of evidence. Structural evidence outranks quality
#: metrics by design; see the module docstring.
WEIGHTS = {
    "rbm_present": 4.0,
    "rbm_absent": 3.0,
    "rbm_paired": 4.0,
    "rbm_unpaired": 2.0,
    "g_split": 2.5,
    "g_shape": 1.5,
    "2d_shape": 2.5,
    "band_position": 1.0,
    "ratio": 0.8,
    "width": 0.8,
}


@dataclass
class Evidence:
    """One reason for or against a candidate material."""

    rule: str
    material: str
    weight: float
    """Positive supports the material, negative argues against it."""
    statement: str

    def __str__(self) -> str:
        sign = "+" if self.weight > 0 else ""
        return f"[{sign}{self.weight:.1f}] {self.statement}"


@dataclass
class Candidate:
    """One material with its accumulated score."""

    key: str
    label: str
    score: float
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def support(self) -> float:
        """Sum of the positive evidence only."""
        return sum(e.weight for e in self.evidence if e.weight > 0)


@dataclass
class Classification:
    """The classifier's verdict and everything behind it."""

    best: Optional[str]
    label: str
    confidence: str
    """``"alta"``, ``"media"``, ``"baja"`` or ``"insuficiente"``."""
    candidates: list[Candidate]
    evidence: list[Evidence]
    blocked_rules: list[str]
    """Rules that could not run, and why — usually spectral range."""
    wall_pairs: list[WallPair] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines: list[str] = []
        if self.best is None:
            lines.append(f"Identificación: {self.label}")
        else:
            lines.append(f"Identificación: {self.label}  (confianza {self.confidence})")
        ranked = [c for c in self.candidates if c.score > 0][:4]
        if ranked:
            lines.append("")
            lines.append("Puntuación:")
            for c in ranked:
                lines.append(f"  {c.label:<38s} {c.score:6.1f}")
        if self.evidence:
            lines.append("")
            lines.append("Evidencia:")
            lines.extend("  " + str(e) for e in self.evidence)
        if self.wall_pairs:
            lines.append("")
            lines.append("Emparejamiento de paredes (RBM):")
            lines.extend("  " + str(p) for p in self.wall_pairs[:5])
        if self.blocked_rules:
            lines.append("")
            lines.append("Reglas que no se han podido aplicar:")
            lines.extend("  – " + r for r in self.blocked_rules)
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def classify(
    spectrum: Spectrum,
    assignment: Assignment,
    rbm_peaks: Optional[Sequence[PeakMeasurement]] = None,
    ratios: Optional[dict] = None,
    two_d_single_lorentzian: Optional[bool] = None,
    g_region: Optional[str] = None,
    db: Optional[Database] = None,
) -> Classification:
    """Identify the material from its Raman fingerprint.

    Parameters
    ----------
    spectrum:
        The (preprocessed) spectrum. Its range decides which rules may run.
    assignment:
        Assigned bands, ideally from a deconvolution.
    rbm_peaks:
        Peaks found in the RBM window. Pass the output of
        :func:`~ramancarbon.core.peaks.find_peaks` restricted to
        80–400 cm⁻¹. ``None`` means "not looked for", which is treated as
        missing information rather than as absence.
    ratios:
        Output of :func:`~ramancarbon.analysis.ratios.intensity_ratios`.
    two_d_single_lorentzian:
        Whether a single-Lorentzian fit of the 2D band was adequate.
    g_region:
        The ``interpretation`` field from
        :func:`~ramancarbon.analysis.assignment.resolve_g_region`.
    db:
        Loaded database.

    Returns
    -------
    Classification
    """
    database = db or load_database()
    evidence: list[Evidence] = []
    blocked: list[str] = []
    warnings: list[str] = []
    scores: dict[str, float] = {k: 0.0 for k in database.materials}

    rbm_window = (80.0, 400.0)
    covers_rbm = spectrum.covers(120.0, 350.0, fraction=0.7)
    genuine_rbm = _genuine_rbm_peaks(rbm_peaks, database)
    wall_pairs: list[WallPair] = []

    # -- rule 1: RBM presence ------------------------------------------
    if rbm_peaks is None:
        blocked.append(
            "presencia de RBM: no se ha buscado en la región 80–400 cm⁻¹"
        )
    elif not covers_rbm:
        lo, hi = spectrum.range
        blocked.append(
            f"presencia/ausencia de RBM: el espectro empieza en {lo:.0f} cm⁻¹ y no "
            "cubre la región RBM (120–350 cm⁻¹). La ausencia de RBM aquí NO es "
            "prueba de material multipared"
        )
        warnings.append(
            "para distinguir pared simple/doble de multipared hace falta medir "
            "desde ~100 cm⁻¹. Sin esa región la identificación se apoya solo en "
            "la forma de la banda G y es mucho más débil"
        )
    elif genuine_rbm:
        freqs = [p.position for p in genuine_rbm]
        for key in ("SWCNT", "DWCNT"):
            scores[key] += WEIGHTS["rbm_present"]
        scores["MWCNT"] -= WEIGHTS["rbm_absent"]
        for key in ("graphene_1L", "graphene_2L", "FLG", "graphite", "GO", "rGO",
                    "amorphous_carbon", "carbon_black"):
            scores[key] -= WEIGHTS["rbm_present"]
        evidence.append(
            Evidence(
                "rbm_present",
                "SWCNT/DWCNT",
                WEIGHTS["rbm_present"],
                f"hay {len(genuine_rbm)} modo(s) de respiración radial "
                f"({', '.join(f'{f:.0f}' for f in freqs)} cm⁻¹). El RBM solo existe "
                "en tubos de diámetro pequeño: descarta material multipared "
                "grueso, grafeno, grafito y carbono amorfo",
            )
        )

        # -- rule 2: paired RBMs -> DWCNT ------------------------------
        if len(freqs) >= 2:
            wall_pairs = find_wall_pairs(freqs, db=database)
            plausible = [p for p in wall_pairs if p.plausible]
            low, high = database.wall_spacing_nm
            if plausible:
                scores["DWCNT"] += WEIGHTS["rbm_paired"]
                scores["SWCNT"] -= WEIGHTS["rbm_unpaired"]
                best = plausible[0]
                evidence.append(
                    Evidence(
                        "rbm_paired",
                        "DWCNT",
                        WEIGHTS["rbm_paired"],
                        f"dos RBM emparejan como paredes concéntricas: "
                        f"{best.outer_omega:.0f} cm⁻¹ (d={best.outer_diameter_nm:.2f} nm) "
                        f"y {best.inner_omega:.0f} cm⁻¹ (d={best.inner_diameter_nm:.2f} nm), "
                        f"separación {best.spacing_nm:.3f} nm, dentro del rango "
                        f"pared-pared {low:g}–{high:g} nm",
                    )
                )
            else:
                scores["SWCNT"] += WEIGHTS["rbm_unpaired"]
                scores["DWCNT"] -= WEIGHTS["rbm_unpaired"]
                evidence.append(
                    Evidence(
                        "rbm_unpaired",
                        "SWCNT",
                        WEIGHTS["rbm_unpaired"],
                        f"los {len(freqs)} RBM no emparejan como paredes "
                        f"concéntricas (ninguna separación cae en {low:g}–{high:g} nm): "
                        "distribución de diámetros de pared simple, no doble pared",
                    )
                )
        else:
            blocked.append(
                "emparejamiento de paredes: hace falta más de un RBM para "
                "distinguir pared doble de pared simple"
            )
    else:
        # Absence of an RBM rules out thin tubes. It says nothing about which
        # of the remaining materials this is, so every non-thin-tube material
        # gets the same credit; weighting MWCNT above graphene here would be
        # inventing evidence.
        for key in ("MWCNT", "graphene_1L", "graphene_2L", "FLG", "graphite",
                    "GO", "rGO", "amorphous_carbon", "carbon_black"):
            scores[key] += WEIGHTS["rbm_absent"]
        scores["SWCNT"] -= WEIGHTS["rbm_absent"]
        scores["DWCNT"] -= WEIGHTS["rbm_absent"]
        evidence.append(
            Evidence(
                "rbm_absent",
                "MWCNT/grafítico",
                WEIGHTS["rbm_absent"],
                "no hay RBM aunque el espectro cubre la región 120–350 cm⁻¹: "
                "el diámetro externo es grande (multipared) o el material no es "
                "tubular",
            )
        )

    # -- rule 3: the G region ------------------------------------------
    if g_region == "swcnt_split":
        scores["SWCNT"] += WEIGHTS["g_split"]
        scores["DWCNT"] += WEIGHTS["g_split"] * 0.8
        scores["MWCNT"] -= WEIGHTS["g_split"]
        evidence.append(
            Evidence(
                "g_split",
                "SWCNT/DWCNT",
                WEIGHTS["g_split"],
                "la banda G aparece desdoblada en G⁻/G⁺ por la curvatura de la "
                "pared, característica de tubos de pared simple o doble",
            )
        )
    elif g_region == "g_plus_dprime":
        scores["MWCNT"] += WEIGHTS["g_split"]
        scores["SWCNT"] -= WEIGHTS["g_split"] * 0.5
        evidence.append(
            Evidence(
                "g_shape",
                "MWCNT",
                WEIGHTS["g_split"],
                "lo que acompaña a la banda G es D′ (por encima, estrecha), no "
                "un G⁻ de curvatura: material multipared o grafítico con defectos",
            )
        )
    elif g_region is not None:
        blocked.append(f"forma de la región G: interpretación «{g_region}», no concluyente")

    g_entry = assignment.g_like()
    if g_entry is not None and g_entry.fwhm is not None:
        width = g_entry.fwhm
        if width <= 25.0:
            for key in ("SWCNT", "DWCNT", "graphene_1L", "graphene_2L", "graphite"):
                scores[key] += WEIGHTS["width"]
            for key in ("amorphous_carbon", "GO", "carbon_black"):
                scores[key] -= WEIGHTS["width"] * 2
            evidence.append(
                Evidence("width", "ordenado", WEIGHTS["width"],
                         f"banda G estrecha (FWHM {width:.0f} cm⁻¹): material bien "
                         "ordenado, no amorfo")
            )
        elif width >= 80.0:
            for key in ("amorphous_carbon", "GO", "carbon_black", "rGO"):
                scores[key] += WEIGHTS["width"] * 2
            for key in ("SWCNT", "DWCNT", "graphene_1L", "graphite"):
                scores[key] -= WEIGHTS["width"] * 2
            evidence.append(
                Evidence("width", "desordenado", WEIGHTS["width"] * 2,
                         f"banda G muy ancha (FWHM {width:.0f} cm⁻¹): carbono "
                         "amorfo, óxido de grafeno o negro de carbono")
            )

    # -- rule 4: the 2D band -------------------------------------------
    two_d = assignment.get("2D")
    covers_2d = spectrum.covers(2550.0, 2800.0, fraction=0.7)
    if not covers_2d:
        blocked.append(
            "forma de la banda 2D: el espectro no llega a 2550–2800 cm⁻¹, así "
            "que no se puede contar capas ni confirmar la conjugación"
        )
    elif two_d is None:
        for key in ("GO", "amorphous_carbon"):
            scores[key] += WEIGHTS["2d_shape"]
        for key in ("graphene_1L", "graphene_2L", "graphite"):
            scores[key] -= WEIGHTS["2d_shape"]
        evidence.append(
            Evidence("2d_shape", "GO/amorfo", WEIGHTS["2d_shape"],
                     "no hay banda 2D aunque el espectro cubre su región: la red "
                     "conjugada está rota (óxido de grafeno, carbono amorfo)")
        )
    elif two_d.fwhm is not None:
        if two_d_single_lorentzian and two_d.fwhm <= 40.0:
            scores["graphene_1L"] += WEIGHTS["2d_shape"]
            scores["graphite"] -= WEIGHTS["2d_shape"]
            evidence.append(
                Evidence("2d_shape", "graphene_1L", WEIGHTS["2d_shape"],
                         f"banda 2D única y estrecha (FWHM {two_d.fwhm:.0f} cm⁻¹), "
                         "ajustable con una sola lorentziana: grafeno monocapa "
                         "(o bicapa turbostrático, que no se distingue así)")
            )
        elif two_d.fwhm > 60.0:
            for key in ("graphite", "FLG", "MWCNT"):
                scores[key] += WEIGHTS["2d_shape"] * 0.6
            scores["graphene_1L"] -= WEIGHTS["2d_shape"]
            evidence.append(
                Evidence("2d_shape", "grafítico", WEIGHTS["2d_shape"] * 0.6,
                         f"banda 2D ancha (FWHM {two_d.fwhm:.0f} cm⁻¹): apilamiento "
                         "de varias capas o paredes")
            )

    # -- rule 5: ratios and positions ----------------------------------
    if ratios:
        _score_ratios(ratios, database, scores, evidence)
    _score_positions(assignment, database, scores, evidence)

    candidates = [
        Candidate(key=k, label=database.material(k).label, score=v)
        for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]
    for candidate in candidates:
        candidate.evidence = [e for e in evidence if candidate.key in e.material]

    best, confidence, label = _verdict(candidates, database, evidence)
    if best is None:
        warnings.append(
            "la evidencia disponible no permite identificar el material. "
            "Mide desde ~100 cm⁻¹ hasta ~3000 cm⁻¹ y haz deconvolución de la "
            "región D–G para poder aplicar todas las reglas"
        )

    return Classification(
        best=best,
        label=label,
        confidence=confidence,
        candidates=candidates,
        evidence=evidence,
        blocked_rules=blocked,
        wall_pairs=wall_pairs,
        warnings=warnings,
    )


def _genuine_rbm_peaks(
    peaks: Optional[Sequence[PeakMeasurement]], db: Database
) -> list[PeakMeasurement]:
    """Filter RBM candidates to those that could really be breathing modes.

    Rejects anything too wide (a real RBM is 3–25 cm⁻¹) and anything whose
    frequency implies a diameter outside 0.5–3 nm, since a tube outside
    that range either cannot exist or cannot show an RBM.
    """
    if not peaks:
        return []
    band = db.band("RBM")
    lo, hi = band.typical_fwhm
    out: list[PeakMeasurement] = []
    for peak in peaks:
        if not 80.0 <= peak.position <= 400.0:
            continue
        if peak.fwhm is not None and peak.fwhm > hi * 2.0:
            continue
        try:
            estimate = rbm_to_diameter(peak.position, db=db)
        except ValueError:
            continue
        if not 0.4 <= estimate.diameter_nm <= 3.5:
            continue
        out.append(peak)
    return out


def _score_ratios(
    ratios: dict, db: Database, scores: dict[str, float], evidence: list[Evidence]
) -> None:
    """Corroborating evidence from I_D/I_G and I_2D/I_G, weighted low."""
    for ratio_key, db_key in (("ID_IG", "ID_IG"), ("I2D_IG", "I2D_IG")):
        entry = ratios.get(ratio_key)
        if entry is None or not entry.available or entry.value is None:
            continue
        matches: list[str] = []
        shown: Optional[float] = None
        for material in db.materials.values():
            span = material.ratio_range(db_key)
            if not span:
                continue
            # Compare on the basis the reference range was quoted on. An
            # area-based ratio is 2-3x its height-based counterpart, so
            # mixing the two silently shifts every material out of range.
            value = entry.on_basis(material.intensity_basis)
            if value is None:
                continue
            if span[0] <= value <= span[1]:
                scores[material.key] += WEIGHTS["ratio"]
                matches.append(material.key)
                shown = value
        if matches and shown is not None:
            evidence.append(
                Evidence(
                    "ratio",
                    "/".join(matches),
                    WEIGHTS["ratio"],
                    f"{entry.name} = {shown:.2f} cae en el rango de: "
                    f"{', '.join(matches)}",
                )
            )


def _score_positions(
    assignment: Assignment, db: Database, scores: dict[str, float], evidence: list[Evidence]
) -> None:
    """Corroborating evidence from where the G band actually sits."""
    g_entry = assignment.g_like()
    if g_entry is None:
        return
    matches: list[str] = []
    for material in db.materials.values():
        for key in ("G", "G+"):
            span = material.band_window(key)
            if span and span[0] <= g_entry.position <= span[1]:
                scores[material.key] += WEIGHTS["band_position"]
                matches.append(material.key)
                break
    if matches:
        evidence.append(
            Evidence(
                "band_position",
                "/".join(matches),
                WEIGHTS["band_position"],
                f"la banda G está en {g_entry.position:.1f} cm⁻¹, dentro del rango "
                f"de: {', '.join(matches)}",
            )
        )


def _verdict(
    candidates: list[Candidate], db: Database, evidence: list[Evidence]
) -> tuple[Optional[str], str, str]:
    """Turn scores into a verdict, refusing to answer when the margin is thin."""
    if not candidates or candidates[0].score <= 0:
        return None, "identificación no concluyente", "insuficiente"

    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    margin = best.score - (runner_up.score if runner_up else 0.0)

    structural = [e for e in evidence if e.rule in {"rbm_present", "rbm_absent", "rbm_paired", "g_split"}]
    if not structural:
        confidence = "baja"
    elif margin >= 3.0 and best.score >= 6.0:
        confidence = "alta"
    elif margin >= 1.5:
        confidence = "media"
    else:
        confidence = "baja"

    label = db.material(best.key).label
    if confidence == "baja" and runner_up is not None and margin < 1.5:
        label = f"{label} o {db.material(runner_up.key).label}"
    return best.key, confidence, label


__all__ = ["Candidate", "Classification", "Evidence", "WEIGHTS", "classify"]
