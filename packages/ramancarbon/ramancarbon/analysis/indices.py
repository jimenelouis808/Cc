"""Structural indices beyond I_D/I_G.

I_D/I_G is the number everyone quotes and the worst-behaved of the family.
It is not monotonic in disorder: it rises as defects are introduced into
good graphite, peaks when they are about 3 nm apart, then falls again as the
material amorphises. For heavily disordered samples — multi-walled tubes,
carbon nanofibres, anything doped hard enough to matter — that non-monotonic
behaviour means two very different structures give the same number, and
nothing in the ratio itself says which one you have.

The indices here are the ones that behave better, or that answer a question
I_D/I_G cannot:

``Γ_G`` (G-band width)
    **Monotonic across the whole disorder range**, from ~15 cm⁻¹ in
    graphite to >150 cm⁻¹ in amorphous carbon. It is the single most
    useful number in a heavily disordered sample and the one most often
    left unreported. Where I_D/I_G is ambiguous, Γ_G is not.
``R1``, ``R2``
    The Beyssac ordering parameters. ``R2 = A_D/(A_D + A_G + A_D')`` is an
    *area* ratio over the whole D–G complex, so it is far less sensitive
    than I_D/I_G to how the fit divided intensity between the components.
``I_D3/I_G``, ``I_D4/I_G``
    How much of the spectrum is amorphous carbon and sp³/polyene material
    rather than graphitic domains. In doped and functionalised samples
    these grow while I_D/I_G may not move.
``amorphisation stage``
    Which branch of the Ferrari–Robertson three-stage trajectory the
    sample sits on, read from ω_G together with I_D/I_G. This is what
    tells you whether an I_D/I_G of 1.0 means "moderately defective
    graphite" or "nearly amorphous".
``sp²-domain fraction``
    The share of the fitted D–G intensity carried by ordered graphitic
    components.

Everything here needs a **deconvolution**, not a peak search: these are
ratios between components that overlap. Where a required component is
missing the index is returned as unavailable with a reason, never
silently as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.fitting import FitResult
from .assignment import Assignment

#: G-band width, in cm⁻¹, at the ends of the disorder scale.
#: Used to place a sample on a monotonic ordering axis.
G_WIDTH_SCALE = {
    "graphite": 15.0,
    "nanocrystalline": 35.0,
    "defective": 60.0,
    "amorphous": 120.0,
}


@dataclass
class Index:
    """One computed index, or the reason it could not be computed."""

    key: str
    label: str
    value: Optional[float]
    available: bool = True
    reason: str = ""
    interpretation: str = ""
    source: str = ""

    def __str__(self) -> str:
        if not self.available or self.value is None:
            return f"{self.label}: no disponible ({self.reason})"
        text = f"{self.label} = {self.value:.4g}"
        if self.interpretation:
            text += f"  → {self.interpretation}"
        return text


@dataclass
class IndexSet:
    """Every index computed for one spectrum."""

    indices: dict[str, Index] = field(default_factory=dict)
    stage: Optional[str] = None
    stage_reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def get(self, key: str) -> Optional[float]:
        """Value of one index, or ``None`` if unavailable."""
        entry = self.indices.get(key)
        return entry.value if entry and entry.available else None

    def summary(self) -> str:
        lines = [str(entry) for entry in self.indices.values()]
        if self.stage:
            lines.append("")
            lines.append(f"Etapa de amorfización: {self.stage}")
            lines.append(f"  {self.stage_reason}")
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Optional[float]]:
        """Flat mapping for the CSV export."""
        row: dict[str, Optional[float]] = {
            key: entry.value if entry.available else None
            for key, entry in self.indices.items()
        }
        row["etapa"] = self.stage
        return row


def _area(fit: Optional[FitResult], assignment: Assignment, key: str) -> Optional[float]:
    """Integrated area of one band, preferring the fit's own component."""
    if fit is not None:
        component = fit.peak(key)
        if component is not None and np.isfinite(component.area) and component.area > 0:
            return float(component.area)
    entry = assignment.get(key)
    if entry is None or entry.area is None or entry.area <= 0:
        return None
    return float(entry.area)


def _height(assignment: Assignment, key: str) -> Optional[float]:
    entry = assignment.get(key)
    if entry is None or entry.height is None or entry.height <= 0:
        return None
    return float(entry.height)


def compute_indices(
    assignment: Assignment,
    fit: Optional[FitResult] = None,
) -> IndexSet:
    """Compute every index the available components allow.

    Parameters
    ----------
    assignment:
        Assigned bands, from :func:`~ramancarbon.analysis.assignment.assign_bands`.
    fit:
        The D–G deconvolution. Required for the area-based indices and for
        anything involving D3, D4 or D′, which a peak search cannot
        separate.

    Returns
    -------
    IndexSet
    """
    result = IndexSet()
    warnings: list[str] = []

    g_entry = assignment.g_like()
    g_width = g_entry.fwhm if g_entry else None
    g_position = g_entry.position if g_entry else None

    # -- G width: the monotonic one ------------------------------------
    if g_width is None:
        result.indices["gamma_G"] = Index(
            "gamma_G", "Γ_G (anchura de G)", None, available=False,
            reason="no se ha ajustado la banda G",
        )
    else:
        result.indices["gamma_G"] = Index(
            "gamma_G",
            "Γ_G (anchura de G)",
            float(g_width),
            interpretation=_interpret_g_width(g_width),
            source="Ferrari & Robertson, Phys. Rev. B 61 (2000) 14095",
        )

    # -- Beyssac ordering parameters -----------------------------------
    a_d = _area(fit, assignment, "D")
    a_g = _area(fit, assignment, "G") or _area(fit, assignment, "G+")
    a_dp = _area(fit, assignment, "D'")
    a_d3 = _area(fit, assignment, "D3")
    a_d4 = _area(fit, assignment, "D4")

    i_d, i_g = _height(assignment, "D"), (
        _height(assignment, "G") or _height(assignment, "G+")
    )
    if i_d is not None and i_g is not None:
        result.indices["R1"] = Index(
            "R1", "R1 = I_D/I_G (alturas)", i_d / i_g,
            source="Beyssac et al., J. Metamorph. Geol. 20 (2002) 859",
        )
    else:
        result.indices["R1"] = Index(
            "R1", "R1 = I_D/I_G (alturas)", None, available=False,
            reason="faltan las alturas de D o de G",
        )

    if a_d is not None and a_g is not None and a_dp is not None:
        total = a_d + a_g + a_dp
        r2 = a_d / total
        result.indices["R2"] = Index(
            "R2", "R2 = A_D/(A_D+A_G+A_D')", r2,
            interpretation=_interpret_r2(r2),
            source="Beyssac et al., J. Metamorph. Geol. 20 (2002) 859",
        )
        warnings.append(
            "R2 se usa en geología como termómetro (T ≈ 641 − 445·R2 °C). Esa "
            "calibración es para materia carbonosa metamórfica y NO se aplica a "
            "nanotubos ni nanofibras: aquí R2 es solo un parámetro de orden"
        )
    else:
        missing = [n for n, v in (("D", a_d), ("G", a_g), ("D'", a_dp)) if v is None]
        result.indices["R2"] = Index(
            "R2", "R2 = A_D/(A_D+A_G+A_D')", None, available=False,
            reason=f"falta el área de {', '.join(missing)}; requiere deconvolución",
        )

    # -- amorphous and sp3 fractions -----------------------------------
    for key, area, label, note in (
        ("ID3_IG", a_d3, "A_D3/A_G (carbono amorfo)",
         "fracción de la señal que viene de carbono amorfo, no de dominios grafíticos"),
        ("ID4_IG", a_d4, "A_D4/A_G (sp³ / polieno)",
         "fracción de fragmentos sp³ y poliénicos fuera de la red grafítica"),
    ):
        if area is not None and a_g is not None:
            result.indices[key] = Index(
                key, label, area / a_g, interpretation=note,
                source="Sadezky et al., Carbon 43 (2005) 1731",
            )
        else:
            result.indices[key] = Index(
                key, label, None, available=False,
                reason="la componente no está en el modelo ajustado",
            )

    # -- ordered fraction -----------------------------------------------
    components = [v for v in (a_d, a_g, a_dp, a_d3, a_d4) if v is not None]
    if a_g is not None and len(components) >= 3:
        ordered = (a_g + (a_dp or 0.0)) / sum(components)
        result.indices["sp2_ordered"] = Index(
            "sp2_ordered", "Fracción sp² ordenada (A_G+A_D')/A_total", ordered,
            interpretation="cuánta de la intensidad total viene de la red grafítica",
            source="derivado del modelo de Sadezky",
        )

    # -- amorphisation stage --------------------------------------------
    ratio = result.get("R1")
    if g_position is not None and ratio is not None:
        result.stage, result.stage_reason = amorphisation_stage(
            g_position, ratio, g_width
        )
    else:
        result.stage_reason = (
            "hace falta la posición de G y I_D/I_G para situar la muestra en la "
            "trayectoria de amorfización"
        )

    if fit is None:
        warnings.append(
            "sin deconvolución solo se calculan los índices que no necesitan "
            "separar componentes solapadas"
        )
    result.warnings = warnings
    return result


def _interpret_g_width(width: float) -> str:
    """Words for a G-band width. Monotonic, unlike I_D/I_G."""
    if width <= 20.0:
        return "grafítico bien ordenado"
    if width <= 35.0:
        return "grafítico con defectos moderados"
    if width <= 60.0:
        return "nanocristalino"
    if width <= 100.0:
        return "muy desordenado (típico de MWCNT y CNF)"
    return "carbono amorfo"


def _interpret_r2(r2: float) -> str:
    if r2 <= 0.2:
        return "muy ordenado"
    if r2 <= 0.4:
        return "ordenado"
    if r2 <= 0.6:
        return "desordenado"
    return "muy desordenado"


def amorphisation_stage(
    g_position: float,
    id_ig: float,
    g_width: Optional[float] = None,
) -> tuple[str, str]:
    """Place a sample on the Ferrari–Robertson amorphisation trajectory.

    Disorder does not move a carbon along a single axis. It follows a
    trajectory with three stages, and I_D/I_G means something different on
    each:

    **Etapa 1 — grafito → grafito nanocristalino.** ω_G rises from 1581
    towards 1600 cm⁻¹ and I_D/I_G rises with disorder. The
    Tuinstra–Koenig relation applies and a crystallite size can be quoted.

    **Etapa 2 — nanocristalino → carbono amorfo.** ω_G falls from 1600
    back towards 1510 cm⁻¹ and I_D/I_G *falls*, because six-membered rings
    are being destroyed and the D band needs them. Here a smaller I_D/I_G
    means **more** disorder, and quoting a crystallite size is wrong.

    **Etapa 3 — amorfo → tetraédrico.** ω_G rises again from 1510 and
    I_D/I_G is near zero as sp³ content climbs above 20 %.

    The two stages that matter for nanotubes and nanofibres are 1 and 2,
    and the G position is what separates them: the same I_D/I_G of 1.0
    sits in stage 1 if ω_G is near 1600 and in stage 2 if ω_G has fallen
    below ~1560.

    Parameters
    ----------
    g_position:
        Fitted G-band position, cm⁻¹.
    id_ig:
        Height-based I_D/I_G.
    g_width:
        G-band FWHM, used as a corroborating signal when available.

    Returns
    -------
    (str, str)
        The stage and the reasoning.

    References
    ----------
    Ferrari & Robertson, *Phys. Rev. B* **61** (2000) 14095.
    """
    wide = g_width is not None and g_width > 90.0
    narrow = g_width is not None and g_width < 40.0

    if g_position >= 1570.0 and not wide:
        if id_ig < 0.05:
            return (
                "etapa 1 (grafito casi perfecto)",
                f"ω_G = {g_position:.0f} cm⁻¹ con I_D/I_G = {id_ig:.2f}, casi sin "
                "banda D: material muy ordenado",
            )
        return (
            "etapa 1 (grafito → nanocristalino)",
            f"ω_G = {g_position:.0f} cm⁻¹ (subiendo hacia 1600) con "
            f"I_D/I_G = {id_ig:.2f}. En esta rama I_D/I_G crece con el desorden y "
            "la relación de Tuinstra-Koenig sí es aplicable"
            + (f"; Γ_G = {g_width:.0f} cm⁻¹ lo confirma" if narrow else ""),
        )
    if g_position >= 1540.0:
        return (
            "etapa 1–2 (frontera)",
            f"ω_G = {g_position:.0f} cm⁻¹ está en la zona donde la trayectoria da "
            "la vuelta. No se puede decir si I_D/I_G sube o baja con el desorden "
            "aquí; usa Γ_G, que sí es monótona",
        )
    return (
        "etapa 2 (nanocristalino → amorfo)",
        f"ω_G = {g_position:.0f} cm⁻¹ ha caído por debajo de 1540, señal de que se "
        "están destruyendo los anillos de seis. En esta rama un I_D/I_G MENOR "
        f"significa MÁS desorden ({id_ig:.2f} aquí), y no se debe citar un tamaño "
        "de cristalito",
    )


__all__ = [
    "G_WIDTH_SCALE",
    "Index",
    "IndexSet",
    "amorphisation_stage",
    "compute_indices",
]
