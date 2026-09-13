"""Atomic per cent from peak areas, and what that number is not.

The arithmetic is short: the number of atoms of an element in the sampled
volume is proportional to the area of one of its lines divided by that
line's sensitivity factor and by the analyser's transmission at that
kinetic energy. Normalise over the elements you found and you have atomic
per cent.

Everything difficult is in the assumptions, and every one of them is
reported rather than buried:

**It is a per cent of what you detected.** Hydrogen has no core level and is
invisible to XPS; helium and lithium are effectively so. An element you did
not fit does not lower anybody else's percentage — its share is simply
redistributed. A composition summing to 100 % is an artefact of
normalisation, never evidence of completeness.

**The sensitivity factors assume a geometry.** The Scofield cross-sections
in the database are calculations and are reliable; what is not transferable
is the angular factor and the transmission function, which belong to the
spectrometer. If the instrument supplies its own sensitivity factors, they
are better than these, and the difference reaches tens of per cent between
the ends of the energy range.

**It is a surface composition, and not even a single surface.** The
information depth goes as roughly the 0.7 power of the kinetic energy, so a
C 1s measured at 1200 eV of kinetic energy samples about 40 % deeper than
an Fe 2p at 775 eV with the same anode. For a homogeneous solid that
difference is already inside the sensitivity factors. For anything layered
— a decorated nanotube, an oxidised particle, a catalyst on a support,
which is to say the samples this package exists for — it is not, and the
element nearer the surface is over-represented. :func:`quantify` reports the
spread of information depths across the lines it used so the size of that
effect can be seen.

**Adventitious carbon inflates carbon.** Every sample that has seen air
carries a nanometre or two of hydrocarbon. On a carbon sample it cannot be
separated from the sample at all, and the C 1s area is the sum of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


from .elements import XPSDatabase, load_xps_database
from .fitting import XPSFitResult
from .spectrum import XPSError, XPSSpectrum

#: Exponent of the kinetic energy in the inelastic mean free path,
#: ``λ ∝ KE^p``. Between 0.6 and 0.8 in every parameterisation in use; the
#: exact value matters only for the size of the warning this module prints,
#: not for any number it returns.
IMFP_EXPONENT = 0.7

#: Relative systematic uncertainty of an XPS quantification done with
#: tabulated sensitivity factors. Not a number this package can improve on:
#: it is the accumulated uncertainty of the cross-sections, the angular
#: term and the transmission function, and it is why two laboratories
#: quoting 12 % and 15 % nitrogen have not disagreed about anything.
SYSTEMATIC_UNCERTAINTY = 0.15


@dataclass
class LineArea:
    """One measured area, and which line it belongs to."""

    element: str
    line: str
    area: float
    uncertainty: Optional[float] = None
    """Statistical uncertainty of the area in the same units, if known."""
    label: str = ""
    notes: list[str] = field(default_factory=list)
    """How the area was obtained, and anything known to be wrong with it —
    an overlapping line, above all. Carried through to the report."""


@dataclass
class Abundance:
    """One element's share, with the numbers that produced it."""

    element: str
    line: str
    area: float
    rsf: float
    transmission: float
    corrected: float
    """``area / (rsf · transmission)`` — proportional to the atom count."""
    atomic_percent: float
    uncertainty_percent: Optional[float]
    kinetic_ev: float
    information_depth: float
    """Relative, normalised to the deepest line in the set. Dimensionless
    on purpose: the absolute IMFP needs a material, and the ratio does not."""
    confidence: str = "media"
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        text = f"{self.element:<3s} {self.line:<9s} {self.atomic_percent:6.2f} %"
        if self.uncertainty_percent:
            text += f" ± {self.uncertainty_percent:.2f}"
        return text + (f"   (A={self.area:.4g}, RSF={self.rsf:.2f}, "
                       f"T={self.transmission:.3f})")


@dataclass
class Quantification:
    """A composition, with everything needed to qualify it."""

    abundances: list[Abundance]
    transmission_model: str
    photon_energy: float
    rsf_basis: dict
    warnings: list[str] = field(default_factory=list)

    def percent(self, element: str) -> Optional[float]:
        for item in self.abundances:
            if item.element == element:
                return item.atomic_percent
        return None

    def ratio(self, numerator: str, denominator: str) -> Optional[float]:
        """One element's atomic ratio to another, e.g. N/C.

        More defensible than either percentage on its own: the
        normalisation drops out, and so does anything that scaled the whole
        spectrum. It still carries both sensitivity factors.
        """
        top, bottom = self.percent(numerator), self.percent(denominator)
        if top is None or bottom is None or bottom <= 0:
            return None
        return float(top / bottom)

    def summary(self) -> str:
        lines = [
            f"Composición atómica (hν = {self.photon_energy:.1f} eV, "
            f"transmisión «{self.transmission_model}»):",
            "",
        ]
        lines.extend("  " + item.summary() for item in self.abundances)
        lines.append("")
        lines.append(
            f"Factores de sensibilidad: {self.rsf_basis.get('name', 'desconocidos')}"
        )
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + text for text in self.warnings)
        return "\n".join(lines)


def transmission_factor(
    kinetic_ev: float,
    model: str = "potencia",
    exponent: float = -0.65,
    pass_energy: Optional[float] = None,
    coefficients: Optional[tuple[float, float]] = None,
) -> float:
    """Relative analyser transmission at one kinetic energy.

    Parameters
    ----------
    model:
        ``"ninguna"`` — no correction, which is what most people do and is
        right only if the instrument's sensitivity factors already contain
        the transmission.
        ``"potencia"`` — ``T ∝ (KE/1000)^n``, the usual parameterisation.
        Real analysers run between about n = −1 and n = +0.5 depending on
        lens mode; the number is a property of the instrument and has to
        come from it.
        ``"phi"`` — PHI's form, ``T = E_paso · (a²/(a² + (KE/E_paso)²))^b``,
        with ``a`` and ``b`` the ``IntensityCalCoeff`` pair from a ``.spe``
        header.

    Notes
    -----
    Only ratios matter, so the absolute scale of this function is
    irrelevant — it cancels in the normalisation. What does not cancel is
    its *slope* across the range, and between C 1s and Fe 2p with an
    aluminium anode a ``n = −0.65`` transmission differs by 30 %.
    """
    name = model.strip().lower()
    kinetic = max(float(kinetic_ev), 1.0)
    if name in ("ninguna", "none", "plana", "flat"):
        return 1.0
    if name in ("potencia", "power"):
        return float((kinetic / 1000.0) ** float(exponent))
    if name == "phi":
        if pass_energy is None or coefficients is None:
            raise XPSError(
                "el modelo de transmisión «phi» necesita la energía de paso y "
                "el par IntensityCalCoeff de la cabecera del .spe"
            )
        a, b = float(coefficients[0]), float(coefficients[1])
        ratio = kinetic / float(pass_energy)
        return float(pass_energy * (a * a / (a * a + ratio * ratio)) ** b)
    raise XPSError(
        f"modelo de transmisión desconocido {model!r}; hay: ninguna, "
        "potencia, phi"
    )


def quantify(
    areas: Sequence[LineArea],
    photon_energy: float,
    work_function: float = 4.5,
    transmission: str = "potencia",
    exponent: float = -0.65,
    pass_energy: Optional[float] = None,
    coefficients: Optional[tuple[float, float]] = None,
    database: Optional[XPSDatabase] = None,
) -> Quantification:
    """Atomic per cent from a set of measured line areas.

    Parameters
    ----------
    areas:
        One entry per element. Use the **whole** line: both spin–orbit
        components and any shake-up satellites, because the sensitivity
        factor is for the whole line and leaving the satellite out of an
        Fe 2p costs 10–20 % of the iron.
    photon_energy:
        eV. Needed to turn each line's binding energy into the kinetic
        energy the transmission and the sampling depth depend on.

    Returns
    -------
    Quantification
    """
    database = database or load_xps_database()
    if not areas:
        raise XPSError("no hay áreas que cuantificar")

    prepared = []
    for entry in areas:
        element = database.element(entry.element)
        line = element.line(entry.line) if entry.line else element.primary_line
        kinetic = float(photon_energy) - line.energy_ev - float(work_function)
        if kinetic <= 0:
            raise XPSError(
                f"{line.label} está a {line.energy_ev:.1f} eV de energía de "
                f"enlace y la fuente da {photon_energy:.1f} eV: esa línea no "
                "se excita con este ánodo"
            )
        factor = transmission_factor(kinetic, transmission, exponent,
                                     pass_energy, coefficients)
        if line.rsf <= 0:
            raise XPSError(
                f"{line.label} no tiene factor de sensibilidad en la base de "
                "datos, así que no puede entrar en una cuantificación"
            )
        corrected = float(entry.area) / (line.rsf * factor)
        prepared.append((entry, line, kinetic, factor, corrected))

    total = sum(item[4] for item in prepared)
    if total <= 0:
        raise XPSError("las áreas suman cero: no hay nada que cuantificar")
    deepest = max(item[2] for item in prepared) ** IMFP_EXPONENT

    abundances: list[Abundance] = []
    for entry, line, kinetic, factor, corrected in prepared:
        percent = 100.0 * corrected / total
        statistical = None
        if entry.uncertainty is not None and entry.area > 0:
            statistical = percent * float(entry.uncertainty) / float(entry.area)
        notes = list(entry.notes)
        if not line.primary:
            notes.append(
                f"{line.label} no es la línea principal de {entry.element}; su "
                "factor de sensibilidad es pequeño y el resultado es más "
                "sensible al fondo"
            )
        if line.confidence in ("low", "baja"):
            notes.append(
                f"el factor de sensibilidad de {line.label} lleva confianza "
                f"{line.confidence}"
            )
        abundances.append(
            Abundance(
                element=entry.element, line=line.label, area=float(entry.area),
                rsf=line.rsf, transmission=factor, corrected=corrected,
                atomic_percent=percent, uncertainty_percent=statistical,
                kinetic_ev=kinetic,
                information_depth=float(kinetic**IMFP_EXPONENT / deepest),
                confidence=line.confidence, notes=notes,
            )
        )
    abundances.sort(key=lambda item: item.atomic_percent, reverse=True)

    warnings = _quantification_warnings(abundances, transmission, database)
    for item in abundances:
        warnings.extend(f"{item.element}: {note}" for note in item.notes)
    return Quantification(
        abundances=abundances, transmission_model=transmission,
        photon_energy=float(photon_energy), rsf_basis=dict(database.rsf_basis),
        warnings=warnings,
    )


def _quantification_warnings(abundances: list[Abundance], transmission: str,
                             database: XPSDatabase) -> list[str]:
    out = [
        "es un porcentaje de LO DETECTADO. El hidrógeno no tiene nivel "
        "interno y la técnica no lo ve; un elemento que no se haya ajustado "
        "no baja el porcentaje de los demás, su parte se reparte entre ellos. "
        "Que sume 100 % es la normalización, no que esté todo",
        f"la incertidumbre sistemática de una cuantificación con factores "
        f"tabulados es de un {100 * SYSTEMATIC_UNCERTAINTY:.0f} % relativo, "
        "muy por encima de la estadística. Dos medidas tuyas hechas igual sí "
        "se comparan entre sí mucho mejor que eso",
    ]
    note = database.rsf_basis.get("note")
    if note:
        out.append(str(note))
    if transmission.strip().lower() in ("ninguna", "none", "plana", "flat"):
        out.append(
            "sin corrección de transmisión: solo es correcto si los factores "
            "de sensibilidad que usa la base de datos ya son los de TU equipo. "
            "Con los de Scofield y un analizador corriente, el error entre un "
            "C 1s y un Fe 2p llega al 30 %"
        )
    depths = [item.information_depth for item in abundances]
    if depths and max(depths) / min(depths) > 1.15:
        shallow = min(abundances, key=lambda item: item.information_depth)
        deep = max(abundances, key=lambda item: item.information_depth)
        out.append(
            f"las líneas usadas no miran igual de hondo: {deep.line} llega un "
            f"{100 * (max(depths) / min(depths) - 1):.0f} % más profundo que "
            f"{shallow.line}. En una muestra homogénea eso ya está dentro de "
            "los factores de sensibilidad; en una decorada, recubierta o "
            "segregada, lo que esté arriba sale sobrerrepresentado"
        )
    carbon = next((item for item in abundances if item.element == "C"), None)
    if carbon is not None and carbon.atomic_percent > 20:
        out.append(
            f"el carbono sale al {carbon.atomic_percent:.0f} %. Toda muestra "
            "que haya visto el aire lleva uno o dos nanómetros de hidrocarburo "
            "encima, y en una muestra de carbono no hay forma de separarlo de "
            "la propia muestra: ese número es la suma de los dos"
        )
    return out


def areas_from_fits(
    results: Sequence[XPSFitResult],
    lines: Optional[dict[str, str]] = None,
    include_satellites: bool = True,
    normalise_acquisition: bool = True,
) -> list[LineArea]:
    """Collect one area per element from a set of fitted regions.

    Every component of a region is summed, satellites included, because a
    shake-up satellite is the same atom in the same chemical state: it lost
    some of its energy on the way out, not some of its existence. Leaving
    the Fe 2p satellites out is a 10–20 % error on the iron and it always
    goes the same way.

    Parameters
    ----------
    lines:
        Which tabulated line each region's area belongs to, by element,
        e.g. ``{"Fe": "Fe 2p"}``. Defaults to the region label of the fit
        with the ``3/2`` or similar stripped.
    normalise_acquisition:
        Divide each region's area by its dwell time × sweeps. This is not
        optional in practice: nobody measures a 2 % iron region with the
        same number of sweeps as the carbon, and an area collected over
        four times as long is four times as large at the same
        concentration. When a region does not record its acquisition, it is
        left alone and a note says so — which is the honest failure, since
        the alternative is a composition wrong by whatever the ratio of
        acquisition times happened to be.
    """
    lines = dict(lines or {})
    totals: dict[str, list] = {}
    for result in results:
        factor = 1.0
        note = ""
        if normalise_acquisition:
            dwell = result.acquisition.get("dwell_s")
            sweeps = result.acquisition.get("sweeps") or 1
            if dwell:
                factor = 1.0 / (float(dwell) * float(sweeps))
            else:
                note = (
                    f"la región {result.region_label} no dice cuánto se "
                    "midió, así que su área entra sin normalizar por tiempo "
                    "de adquisición. Si no todas las regiones se midieron "
                    "igual, la composición está mal en esa proporción"
                )
        for component in result.components:
            if component.element is None:
                continue
            if not include_satellites and component.satellite:
                continue
            label = lines.get(component.element) or _line_of(component.line
                                                             or result.region_label)
            entry = totals.setdefault(component.element, [0.0, label, []])
            entry[0] += component.area * factor
            entry[1] = label
            if note and note not in entry[2]:
                entry[2].append(note)
    return [
        LineArea(element=element, line=label, area=area, notes=list(notes))
        for element, (area, label, notes) in totals.items()
    ]


def _line_of(region: str) -> str:
    """``"Fe 2p3/2"`` → ``"Fe 2p"``; ``"C 1s"`` → ``"C 1s"``."""
    parts = region.split()
    if len(parts) < 2:
        return region
    orbital = parts[1]
    if "/" in orbital:
        orbital = orbital.split("/")[0][:-1]
    return f"{parts[0]} {orbital}"


def survey_areas(
    spectrum: XPSSpectrum,
    elements: Sequence[str],
    half_width: float = 8.0,
    background: str = "shirley",
    database: Optional[XPSDatabase] = None,
) -> list[LineArea]:
    """Integrate each element's main line straight off a survey.

    Quick, and worse than fitting: on a survey the lines of neighbouring
    elements overlap, the step is too coarse to place a background well,
    and nothing separates a satellite from the line it belongs to. Use it
    to see whether a composition is in the right ballpark, and fit the
    regions for anything that goes in a paper — the two disagree by 10–30 %
    routinely, and the fit is the one that is right.
    """
    from .background import estimate_background
    from ..core.compat import trapezoid

    database = database or load_xps_database()
    out: list[LineArea] = []
    for symbol in elements:
        line = database.element(symbol).primary_line
        span = line.span
        window = (span[0] - half_width, span[1] + half_width)
        if not spectrum.covers(*window, fraction=0.9):
            continue
        estimate = estimate_background(spectrum, window, background)
        axis, counts = spectrum.region_of(*window)
        area = float(trapezoid(estimate.subtract(counts), axis))
        if area <= 0:
            continue
        intruders = [
            other.label
            for other in database.lines_in_range(window[0], window[1])
            if other.element != symbol
        ]
        notes = []
        if intruders:
            notes.append(
                "la ventana integrada lleva dentro "
                + ", ".join(sorted(set(intruders)))
                + ": el área incluye lo que aporten esas líneas, así que este "
                "elemento sale de más. Ajusta la región en vez de integrarla"
            )
        out.append(LineArea(element=symbol, line=line.label, area=area,
                            label=f"integrada sobre {window[0]:.0f}–{window[1]:.0f} eV",
                            notes=notes))
    if not out:
        raise XPSError(
            "ninguna línea principal de los elementos pedidos cae entera "
            f"dentro del espectro ({spectrum.range[0]:.0f}–"
            f"{spectrum.range[1]:.0f} eV)"
        )
    return out


__all__ = [
    "Abundance",
    "IMFP_EXPONENT",
    "LineArea",
    "Quantification",
    "SYSTEMATIC_UNCERTAINTY",
    "areas_from_fits",
    "quantify",
    "survey_areas",
    "transmission_factor",
]
