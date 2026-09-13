"""Charge referencing: putting the binding-energy axis where it belongs.

An insulating sample loses electrons as it is measured and charges
positive, which retards every photoelectron and moves the whole spectrum to
higher apparent binding energy — by a few tenths of an eV on a thin film
over a conductor, by several eV on a powder. The shift is not a property of
the sample, so it has to be removed, and the only way to remove it is to
decide that some line is *known* and slide the axis until it is there.

That decision is the weakest link in the entire technique, and this module
is written so it cannot be made silently:

* The reference used, its published value, its spread and its confidence
  all travel with the corrected spectrum.
* The C 1s of adventitious carbon — the universal choice — is stored in the
  database with confidence **low** and the reasons why. The literature puts
  it anywhere between 284.4 and 285.2 eV, which is a bigger range than most
  of the chemical shifts people then interpret, and on a sample that is
  *made of carbon* the peak near 284.8 eV may be the sample rather than the
  contaminant, which makes the whole exercise circular. This module looks
  at the width and the asymmetry of the peak it is about to reference
  against and says so when they look like sp² carbon rather than adventitious
  carbon.
* :func:`auger_parameter` exists because it is the way out. The modified
  Auger parameter, ``α′ = BE(photoline) + KE(Auger)``, is a **difference**
  of two features in the same spectrum, so the charge shift cancels
  exactly. It needs no reference at all, and where it is available it
  settles chemical-state questions that referenced binding energies only
  argue about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .background import estimate_background
from .elements import CalibrationReference, XPSDatabase, load_xps_database
from .fitting import XPSComponent, XPSModel, fit_region, resolution_floor
from .spectrum import XPSError, XPSSpectrum

#: Width above which a referencing peak is wider than chemistry explains,
#: in eV. Adventitious carbon on a well-behaved sample is 1.2–1.6 eV; much
#: above 2 eV and the peak is being broadened by *differential* charging,
#: where different parts of the sample sit at different potentials. A single
#: shift cannot fix that, and nothing downstream can detect it.
DIFFERENTIAL_CHARGING_FWHM = 2.2

#: Below this width, and with a visible tail, a peak at ~284.5 eV is far
#: more likely to be the sample's own sp² carbon than adventitious carbon.
SP2_FWHM = 1.05

#: Above this width, in eV, a single peak fitted to a reference line is not
#: one chemical state: it is the envelope of several, and its centroid is
#: not the position of any of them. A C 1s of an oxidised carbon fitted
#: with one peak comes out 0.5–1.5 eV high, and that error goes straight
#: into every binding energy referenced against it.
ENVELOPE_FWHM = 1.9


@dataclass
class Calibration:
    """What was done to the binding-energy axis, and how much to trust it."""

    shift_ev: float
    """Added to every binding energy. Positive means the sample was
    charging positive and the spectrum has been moved down."""
    reference: str
    measured_ev: float
    expected_ev: float
    spread_ev: float
    confidence: str
    source: str
    fwhm_ev: Optional[float] = None
    warnings: list[str] = field(default_factory=list)
    note: str = ""

    def describe(self) -> str:
        return (
            f"referencia {self.reference}: medida en {self.measured_ev:.2f} eV, "
            f"esperada en {self.expected_ev:.2f} ± {self.spread_ev:.2f} eV → "
            f"desplazamiento {self.shift_ev:+.2f} eV (confianza {self.confidence})"
        )


def measure_line(
    spectrum: XPSSpectrum,
    centre: float,
    half_width: float = 5.0,
    profile: str = "gl",
    background: str = "shirley",
) -> tuple[float, float, float]:
    """Fit a single peak near ``centre`` and report where it is.

    Returns
    -------
    tuple
        ``(centre in eV, FWHM in eV, height)``. The centre comes from a fit
        rather than from the highest channel: with a 0.1 eV step the
        highest channel quantises the reference to 0.1 eV, and the shifts
        being argued about are that size.
    """
    window = (centre - half_width, centre + half_width)
    if not spectrum.covers(*window, fraction=0.8):
        raise XPSError(
            f"el espectro ({spectrum.range[0]:.1f}–{spectrum.range[1]:.1f} eV) "
            f"no cubre {window[0]:.1f}–{window[1]:.1f} eV, que es donde habría "
            "que medir esa línea"
        )
    estimate = estimate_background(spectrum, window, background)
    axis, counts = spectrum.region_of(*window)
    above = counts - estimate.values
    start = float(axis[int(np.argmax(above))])
    model = XPSModel(
        [XPSComponent(name="ref", label="línea de referencia", centre=start,
                      height=float(np.max(above)), fwhm=1.4, profile=profile,
                      centre_bounds=(window[0] + 0.5, window[1] - 0.5),
                      fwhm_bounds=(0.3, 8.0))],
        window=window, background=background, name="calibración",
    )
    result = fit_region(spectrum, model, background_iterations=2)
    fitted = result.components[0]
    return fitted.centre, fitted.true_fwhm, fitted.peak_height


def calibrate(
    spectrum: XPSSpectrum,
    reference: str = "C1s_adventitious",
    measured: Optional[float] = None,
    half_width: float = 5.0,
    database: Optional[XPSDatabase] = None,
) -> tuple[XPSSpectrum, Calibration]:
    """Shift a spectrum so a chosen line sits at its published energy.

    Parameters
    ----------
    reference:
        A key in the database's calibration references:
        ``"C1s_adventitious"``, ``"Au4f"``, ``"Ag3d"``, ``"Cu2p"``,
        ``"Fermi"``.
    measured:
        Where the reference line actually is, in eV. ``None`` measures it
        from this spectrum by fitting a single peak near the expected
        position — which only works if the charging is smaller than
        ``half_width``.

    Returns
    -------
    tuple
        ``(shifted spectrum, Calibration)``. The shift is recorded in the
        spectrum's history and metadata, so a binding energy taken from it
        can always say what it was referenced to.
    """
    database = database or load_xps_database()
    entry = database.reference(reference)
    warnings: list[str] = []
    width: Optional[float] = None

    if measured is None:
        measured, width, _ = measure_line(spectrum, entry.energy_ev, half_width)
    shift = float(entry.energy_ev) - float(measured)

    warnings.extend(_reference_warnings(spectrum, entry, shift, width))
    shifted = spectrum.shifted(
        shift, f"{entry.line or reference} a {entry.energy_ev:.2f} eV ({reference})"
    )
    shifted.metadata["calibración"] = {
        "referencia": reference,
        "medida_ev": float(measured),
        "esperada_ev": float(entry.energy_ev),
        "confianza": entry.confidence,
        "fuente": entry.source,
    }
    return shifted, Calibration(
        shift_ev=shift, reference=reference, measured_ev=float(measured),
        expected_ev=float(entry.energy_ev), spread_ev=float(entry.spread_ev),
        confidence=entry.confidence, source=entry.source, fwhm_ev=width,
        warnings=warnings, note=entry.note,
    )


def _reference_warnings(spectrum: XPSSpectrum, entry: CalibrationReference,
                        shift: float, width: Optional[float]) -> list[str]:
    """Everything about this referencing that could make it wrong."""
    out: list[str] = []
    if entry.confidence in ("low", "baja"):
        out.append(
            f"la referencia {entry.key} lleva confianza {entry.confidence} en la "
            f"base de datos. {entry.note}"
        )
    out.append(
        f"la horquilla publicada de esta referencia es ±{entry.spread_ev:.2f} eV, "
        "así que toda energía de enlace de este espectro la arrastra. Las "
        "diferencias más pequeñas que eso no se pueden defender contra otro "
        "laboratorio, aunque sí entre muestras tuyas medidas igual"
    )
    if abs(shift) > 5.0:
        out.append(
            f"el desplazamiento de carga es {shift:+.1f} eV, que es mucho: la "
            "muestra está cargándose de verdad. Comprueba que el pico que se "
            "ha usado como referencia es el que se cree y no otro que ha "
            "entrado en la ventana"
        )
    if width is not None:
        floor = resolution_floor(spectrum)
        if width > DIFFERENTIAL_CHARGING_FWHM:
            out.append(
                f"la línea de referencia sale con {width:.2f} eV de anchura. Por "
                "encima de unos 2 eV eso suele ser carga DIFERENCIAL: distintas "
                "zonas de la muestra a distinto potencial. Un solo "
                "desplazamiento no lo arregla, y los estados químicos que se "
                "lean después estarán ensanchados por el equipo, no por la "
                "química"
            )
        elif floor and width < floor:
            out.append(
                f"la línea de referencia sale más estrecha ({width:.2f} eV) que "
                f"la resolución del equipo ({floor:.2f} eV): revisa la energía "
                "de paso declarada"
            )
        if width > ENVELOPE_FWHM:
            out.append(
                f"el pico de referencia se ha ajustado con UNA componente de "
                f"{width:.2f} eV, y por encima de {ENVELOPE_FWHM:.1f} eV eso "
                "no es un estado químico sino la envolvente de varios. Su "
                "centro no es la posición de ninguno, y en un C 1s oxidado "
                "sale entre 0.5 y 1.5 eV alto. Referencia contra una "
                "componente ajustada (calibrate_to_state) en vez de contra el "
                "pico entero"
            )
        if entry.key == "C1s_adventitious" and width < SP2_FWHM:
            out.append(
                f"el pico usado como C 1s adventicio mide {width:.2f} eV de "
                "ancho, que es estrecho para carbono de contaminación "
                "(1.2–1.6 eV) y típico del carbono sp² de la propia muestra. Si "
                "es la muestra, referenciar contra él es circular: se está "
                "fijando por decreto la energía que se quiere medir. Con "
                "grafeno, nanotubos o carburos, usa el nivel de Fermi, una "
                "referencia interna, o publica también el parámetro Auger"
            )
    if entry.conductor_required:
        out.append(
            f"{entry.key} solo vale si la muestra está en contacto eléctrico "
            "con el portamuestras; sobre un polvo aislante, la referencia "
            "misma se carga"
        )
    return out


def calibrate_to_state(
    spectrum: XPSSpectrum,
    region: str,
    state: str,
    states: Optional[list[str]] = None,
    database: Optional[XPSDatabase] = None,
    **options,
) -> tuple[XPSSpectrum, Calibration]:
    """Reference the axis on one **fitted component**, not on a whole peak.

    This is the right way to reference a sample that is itself made of the
    element carrying the reference line — a graphene, a nanotube, a carbide,
    which is to say most of what this package is for. Fitting a single peak
    to such a C 1s and calling its centre 284.8 eV puts the axis wherever
    the oxidised tail happens to drag the centroid, which is between 0.5 and
    1.5 eV out; and 284.8 eV is the adventitious value anyway, while the
    sp² carbon of the sample sits at 284.4.

    The procedure is two passes, because the published windows a fit is
    bounded by are on the *referenced* axis and the axis is not referenced
    yet: the region is first shifted roughly, by putting its tallest feature
    at the target state's energy, then fitted, then shifted again by
    whatever the fitted component is still out by.

    Parameters
    ----------
    region, state:
        The database region and the state within it to put at its published
        energy, e.g. ``"C 1s"`` and ``"C-C sp2"``.
    states:
        Which states to include in the fit. ``None`` fits the region's
        whole state list, which is usually too many — pass the ones the
        sample plausibly has.

    Returns
    -------
    tuple
        ``(shifted spectrum, Calibration)``.
    """
    from .presets import state_model

    database = database or load_xps_database()
    target = database.state(region, state)
    if states is not None and state not in states:
        states = list(states) + [state]

    estimate = estimate_background(spectrum, spectrum.range, "shirley")
    axis, counts = spectrum.region_of(*spectrum.range)
    rough = float(axis[int(np.argmax(counts - estimate.values))])
    rough_shift = float(target.energy_ev) - rough

    work = spectrum.shifted(rough_shift, "preajuste para calibrar")
    model = state_model(work, region, states, database=database, **options)
    result = fit_region(work, model, database=database)
    component = result.component(state)
    if component is None:                       # pragma: no cover - state_model guarantees it
        raise XPSError(f"el ajuste no ha devuelto la componente {state!r}")
    refinement = float(target.energy_ev) - component.centre
    total = rough_shift + refinement

    warnings: list[str] = []
    if abs(refinement) > 0.5:
        warnings.append(
            f"el preajuste suponía que el pico más alto de la región es "
            f"{target.name}, y el ajuste lo ha corregido en "
            f"{refinement:+.2f} eV. Si la suposición era falsa —una muestra "
            "muy oxidada, por ejemplo— la referencia está sobre la "
            "componente equivocada"
        )
    warnings.extend(
        text for text in result.warnings
        if "χ²" in text or "Durbin" in text
    )
    warnings.append(
        f"referenciado sobre una componente ajustada ({target.name} a "
        f"{target.energy_ev:.2f} eV, confianza {target.confidence}), no sobre "
        "el pico entero. Eso quita el sesgo de la envolvente, pero el ajuste "
        "que lo sitúa es parte de la calibración: si el modelo de la región "
        "cambia, el eje cambia"
    )
    shifted = spectrum.shifted(total, f"{target.name} a {target.energy_ev:.2f} eV")
    shifted.metadata["calibración"] = {
        "referencia": f"{region}:{state}",
        "medida_ev": float(target.energy_ev - total),
        "esperada_ev": float(target.energy_ev),
        "confianza": target.confidence,
        "método": "componente ajustada",
    }
    return shifted, Calibration(
        shift_ev=total, reference=f"{region}:{state}",
        measured_ev=float(target.energy_ev - total),
        expected_ev=float(target.energy_ev),
        spread_ev=float(target.window[1] - target.window[0]) / 2.0,
        confidence=target.confidence, source=target.source or region,
        fwhm_ev=component.true_fwhm, warnings=warnings, note=target.note,
    )


# ----------------------------------------------------------------------
# the reference-free route
# ----------------------------------------------------------------------
@dataclass
class AugerParameter:
    """The modified Auger parameter and what it is worth.

    ``α′ = BE(photoline) + KE(Auger)``. Both terms move by the same amount
    when the sample charges — one up, one down — so the sum does not move
    at all. It is therefore a chemical-state measurement that owes nothing
    to any referencing decision, which is why it is worth the trouble of
    measuring a broad Auger group.
    """

    value_ev: float
    photoline: str
    photoline_ev: float
    auger: str
    auger_kinetic_ev: float
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"α′ = {self.value_ev:.2f} eV  ({self.photoline} a "
            f"{self.photoline_ev:.2f} eV + {self.auger} a "
            f"{self.auger_kinetic_ev:.2f} eV cinéticos)"
        )


def auger_parameter(
    spectrum: XPSSpectrum,
    element: str,
    photoline: Optional[str] = None,
    auger: Optional[str] = None,
    half_width: float = 5.0,
    database: Optional[XPSDatabase] = None,
) -> AugerParameter:
    """Measure the modified Auger parameter of one element.

    Immune to charging by construction, and therefore the right thing to
    quote when the sample is an insulator or when the C 1s reference is
    circular. Its cost is that it needs the Auger group, which is broad and
    weak and is often outside the window somebody chose for a
    high-resolution scan.

    Raises
    ------
    XPSError
        If the photon energy is unknown — without it there is no kinetic
        scale and therefore no Auger parameter — or if either feature falls
        outside the measured range.
    """
    database = database or load_xps_database()
    if spectrum.photon_energy is None:
        raise XPSError(
            "el parámetro Auger necesita la energía del fotón: su gracia es "
            "que suma una energía de enlace y una CINÉTICA, y la conversión "
            "entre las dos es la energía del fotón"
        )
    entry = database.element(element)
    line = entry.line(photoline) if photoline else entry.primary_line
    groups = entry.auger
    if not groups:
        raise XPSError(
            f"la base de datos no tiene líneas Auger de {element}, así que no "
            "hay parámetro Auger que medir"
        )
    group = groups[0] if auger is None else next(
        (g for g in groups if g.label == auger), groups[0]
    )

    photo_centre, _, _ = measure_line(spectrum, line.energy_ev, half_width)
    expected = group.binding_at(spectrum.photon_energy, spectrum.work_function)
    window = (expected - group.width_ev, expected + group.width_ev)
    if not spectrum.covers(*window, fraction=0.7):
        raise XPSError(
            f"el grupo {group.label} caería en {expected:.0f} eV de energía de "
            f"enlace con esta fuente, y el espectro llega a "
            f"{spectrum.range[1]:.0f} eV. Sin él no hay parámetro Auger"
        )
    axis, counts = spectrum.region_of(*window)
    estimate = estimate_background(spectrum, window, "lineal")
    above = counts - estimate.values
    peak = float(axis[int(np.argmax(above))])
    kinetic = spectrum.photon_energy - peak - spectrum.work_function

    warnings = [
        "el máximo del grupo Auger se ha tomado del canal más alto, no de un "
        "ajuste: estos grupos son anchos y estructurados y no tienen una "
        "forma que ajustar. Eso deja una incertidumbre de varias décimas de eV"
    ]
    if group.confidence in ("low", "baja"):
        warnings.append(
            f"la posición tabulada de {group.label} lleva confianza "
            f"{group.confidence}"
        )
    return AugerParameter(
        value_ev=float(photo_centre + kinetic),
        photoline=line.label, photoline_ev=float(photo_centre),
        auger=group.label, auger_kinetic_ev=float(kinetic),
        warnings=warnings,
    )


@dataclass
class DParameter:
    """The C KLL D parameter: how much of the carbon is sp².

    Defined as the energy separation between the maximum and the minimum of
    the **first derivative** of the C KLL Auger group. Graphitic (sp²)
    carbon gives about 21–23 eV and diamond-like (sp³) about 13–14 eV, and
    the scale in between is roughly linear in sp² fraction.

    This is the XPS answer to the question Raman answers with I_D/I_G, and
    it is worth having for the same reason the Raman side gives two
    branches of Tuinstra–Koenig: two independent measurements of the same
    quantity that disagree are telling you something.
    """

    value_ev: float
    sp2_fraction: Optional[float]
    maximum_ke: float
    minimum_ke: float
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        text = f"parámetro D del C KLL = {self.value_ev:.1f} eV"
        if self.sp2_fraction is not None:
            text += f" → ~{100 * self.sp2_fraction:.0f} % sp²"
        return text


#: Anchor points of the D parameter scale, in eV: (sp³ reference, sp²
#: reference). Widely used, and worth distrusting at the ends — the
#: literature quotes diamond between 13 and 14.5 eV and HOPG between 21 and
#: 23, so the scale is good to about ±10 % of sp² fraction, not better.
D_PARAMETER_SCALE = (13.0, 22.0)


def carbon_d_parameter(
    spectrum: XPSSpectrum,
    smooth_ev: float = 1.5,
    database: Optional[XPSDatabase] = None,
) -> DParameter:
    """Measure the C KLL D parameter.

    The derivative amplifies noise ferociously, so the group is smoothed
    first over ``smooth_ev``; the width of that smoothing is part of the
    measurement and is reported, exactly as the smoothing of a dQ/dV curve
    is on the electrochemistry side.
    """
    database = database or load_xps_database()
    if spectrum.photon_energy is None:
        raise XPSError(
            "el parámetro D se mide en energía CINÉTICA y hace falta la "
            "energía del fotón para situar el grupo C KLL"
        )
    group = database.element("C").auger[0]
    expected = group.binding_at(spectrum.photon_energy, spectrum.work_function)
    window = (expected - 22.0, expected + 22.0)
    if not spectrum.covers(*window, fraction=0.8):
        raise XPSError(
            f"el C KLL caería en {expected:.0f} eV de energía de enlace con "
            f"esta fuente y el espectro llega a {spectrum.range[1]:.0f} eV"
        )
    axis, counts = spectrum.region_of(*window)
    kinetic = spectrum.photon_energy - axis - spectrum.work_function
    order = np.argsort(kinetic)
    kinetic, counts = kinetic[order], counts[order]

    step = float(np.median(np.diff(kinetic)))
    span = max(3, int(round(smooth_ev / max(step, 1e-6))) | 1)
    kernel = np.ones(span) / span
    smooth = np.convolve(np.pad(counts, span // 2, mode="edge"), kernel, "valid")
    derivative = np.gradient(smooth, kinetic)
    top = int(np.argmax(derivative))
    bottom = int(np.argmin(derivative))
    value = float(abs(kinetic[bottom] - kinetic[top]))

    low, high = D_PARAMETER_SCALE
    fraction: Optional[float] = None
    warnings = [
        f"medido sobre la derivada del C KLL suavizada en {smooth_ev:.1f} eV. "
        "El suavizado es parte de la medida: con menos sale ruido y con más "
        "se acercan el máximo y el mínimo"
    ]
    if low < value < high:
        fraction = (value - low) / (high - low)
    else:
        warnings.append(
            f"el valor {value:.1f} eV cae fuera de la escala diamante–grafito "
            f"({low:.0f}–{high:.0f} eV), así que no se traduce a una fracción "
            "sp². Suele significar que el grupo está contaminado por otra "
            "línea o que las estadísticas no dan"
        )
    if step > 0.6:
        warnings.append(
            f"el paso del espectro es {step:.2f} eV: para una derivada hacen "
            "falta 0.1–0.2 eV. Esto es un barrido de reconocimiento, no una "
            "medida del parámetro D"
        )
    return DParameter(
        value_ev=value, sp2_fraction=fraction,
        maximum_ke=float(kinetic[top]), minimum_ke=float(kinetic[bottom]),
        warnings=warnings,
    )


__all__ = [
    "AugerParameter",
    "Calibration",
    "DIFFERENTIAL_CHARGING_FWHM",
    "DParameter",
    "D_PARAMETER_SCALE",
    "SP2_FWHM",
    "ENVELOPE_FWHM",
    "auger_parameter",
    "calibrate",
    "calibrate_to_state",
    "carbon_d_parameter",
    "measure_line",
]
