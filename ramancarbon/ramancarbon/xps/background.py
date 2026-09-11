"""Inelastic backgrounds, and the fact that the choice is part of the result.

Under every photoemission peak there is a step: electrons that left the atom
with the full kinetic energy and lost some of it on the way out. They appear
at *higher* binding energy than the peak that produced them, which is why an
XPS background steps **up** across a peak instead of running flat under it.

Three subtractions, in increasing order of how much physics they contain and
how much they ask of the measurement:

``linear``
    A straight line between the endpoints. Honest for a narrow, weak region
    where the step is invisible; wrong for anything intense, where it leaves
    part of the step inside the peaks and inflates the high-binding-energy
    component — which, in a C 1s or an N 1s, is exactly the oxidised state
    somebody is about to draw a conclusion from.
``shirley``
    The step is taken proportional to the peak intensity at higher kinetic
    energy, solved self-consistently. The standard choice, and the one the
    literature values in ``xps.json`` were fitted with.
``tougaard``
    The step is computed from an explicit inelastic-scattering cross
    section, so it does not need the spectrum to return to a flat region and
    it is the right choice over a wide range. It does need 30–50 eV of
    spectrum beyond the peak on the high-binding-energy side; over a narrow
    region it has nothing to work with and is worse than Shirley.

**The endpoints are a parameter.** Every one of these is fixed by where the
region was cut, and Shirley most of all: moving the high-binding-energy
endpoint of a C 1s region by 1 eV moves the fitted area of the
carbonyl component by several per cent. That is not a defect of the
algorithm, it is what "area of a peak on top of a background" means. This
module reports the endpoints it used and warns when they sit somewhere that
makes the answer unstable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .spectrum import XPSError, XPSSpectrum

#: Universal Tougaard cross-section parameters, in eV².
#: ``K(T) = B T / (C + T²)²`` with these values is Tougaard's "universal"
#: curve, fitted to metals and reused everywhere; for polymers the C
#: parameter is nearer 551 eV², which is why it is an argument.
TOUGAARD_B = 2866.0
TOUGAARD_C = 1643.0

#: How many points at each end are averaged to fix an endpoint. One point is
#: one realisation of the counting noise, and the whole background hangs off
#: it.
ENDPOINT_POINTS = 5

#: Minimum span in eV for a Tougaard background to mean anything.
TOUGAARD_MIN_SPAN = 30.0


@dataclass
class Background:
    """A background, with everything needed to argue about it."""

    values: np.ndarray
    kind: str
    window: tuple[float, float]
    endpoints: tuple[float, float]
    """The averaged intensities at the low- and high-binding-energy ends."""
    iterations: int = 0
    parameters: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def subtract(self, counts: np.ndarray) -> np.ndarray:
        """``counts − background``, without clipping.

        Negative values are left in place deliberately. They are the
        signature of a background that is too high — an endpoint on the
        flank of a neighbouring peak, or a Shirley over a region that steps
        down — and clipping them to zero hides the one piece of evidence
        that says so.
        """
        return np.asarray(counts, dtype=float) - self.values

    def describe(self) -> str:
        text = (f"fondo {self.kind} sobre {self.window[0]:.1f}–{self.window[1]:.1f} eV "
                f"({self.endpoints[0]:.0f} → {self.endpoints[1]:.0f} cuentas)")
        if self.iterations > 1:
            text += f", convergido en {self.iterations} iteraciones"
        return text


def _endpoints(counts: np.ndarray, points: int) -> tuple[float, float]:
    n = max(1, min(int(points), counts.size // 3))
    return float(np.mean(counts[:n])), float(np.mean(counts[-n:]))


def linear_background(
    energy: np.ndarray, counts: np.ndarray, points: int = ENDPOINT_POINTS
) -> Background:
    """A straight line between the averaged endpoints."""
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    low, high = _endpoints(counts, points)
    span = energy[-1] - energy[0]
    values = low + (high - low) * (energy - energy[0]) / (span if span else 1.0)
    warnings: list[str] = []
    if high > 1.15 * low and low > 0:
        warnings.append(
            f"el fondo sube un {100 * (high / low - 1):.0f} % a lo ancho de la "
            "región: eso es el escalón inelástico, y una recta deja parte de "
            "él dentro de los picos. El componente de mayor energía de enlace "
            "se lleva casi todo ese exceso — usa Shirley"
        )
    return Background(
        values=values, kind="lineal", window=(float(energy[0]), float(energy[-1])),
        endpoints=(low, high), warnings=warnings,
    )


def shirley_background(
    energy: np.ndarray,
    counts: np.ndarray,
    envelope: Optional[np.ndarray] = None,
    points: int = ENDPOINT_POINTS,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> Background:
    """The Shirley inelastic background.

    ``B(E) = I_low + (I_high − I_low) · A(E) / A_total`` where ``A(E)`` is
    the peak area between the low-binding-energy endpoint and ``E``, taken
    above the background itself — hence the iteration.

    Parameters
    ----------
    envelope:
        The fitted peak envelope, background-free. When given, the
        background is computed from *it* in a single pass instead of
        iterating on the data. This is the "active" Shirley, and it is
        better for a reason worth stating: the iterative form integrates
        the noise as well as the peaks, so a Shirley under a weak region
        carries a slow random walk that changes the fitted areas. Computed
        from a model, it does not.

    Notes
    -----
    Direction matters and is easy to get backwards. Inelastically
    scattered electrons have *lost* kinetic energy, so they appear at
    *higher* binding energy than the peak they came from: the integral runs
    from the low-binding-energy endpoint upwards and the background steps
    up. A Shirley that steps down has been computed on a reversed axis.
    """
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if energy.size < 5:
        raise XPSError("un fondo Shirley necesita al menos cinco puntos")
    low, high = _endpoints(counts, points)
    warnings: list[str] = []

    if envelope is not None:
        source = np.clip(np.asarray(envelope, dtype=float), 0.0, None)
        cumulative = np.concatenate([[0.0], np.cumsum(
            0.5 * (source[1:] + source[:-1]) * np.diff(energy))])
        total = float(cumulative[-1])
        values = low + (high - low) * (cumulative / total if total > 0 else 0.0)
        return Background(
            values=values, kind="Shirley (activo)",
            window=(float(energy[0]), float(energy[-1])),
            endpoints=(low, high), iterations=1,
            parameters={"fuente": "envolvente ajustada"},
            warnings=warnings,
        )

    values = np.full(energy.shape, low, dtype=float)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        above = np.clip(counts - values, 0.0, None)
        cumulative = np.concatenate([[0.0], np.cumsum(
            0.5 * (above[1:] + above[:-1]) * np.diff(energy))])
        total = float(cumulative[-1])
        if total <= 0:
            warnings.append(
                "no queda ninguna intensidad por encima del fondo: la región "
                "no tiene pico, o los extremos están al revés"
            )
            break
        new = low + (high - low) * cumulative / total
        change = float(np.max(np.abs(new - values))) / max(abs(high - low), 1e-12)
        values = new
        if change < tolerance:
            break
    else:                                       # pragma: no cover - rare
        warnings.append(
            f"el fondo Shirley no ha convergido en {max_iterations} iteraciones"
        )

    if high < low:
        warnings.append(
            "la región termina MÁS BAJA de lo que empieza en energía de "
            "enlace, que es lo contrario de lo que hace un escalón "
            "inelástico. O los extremos caen en el flanco de un pico vecino, "
            "o bajo esta región hay un fondo decreciente que el Shirley no "
            "describe: comprueba los límites antes de usar las áreas"
        )
    # Three noise levels, not one: with ordinary counting noise about one
    # point in six sits a whole σ below any correct background, so a
    # one-σ test fires on every good subtraction there is.
    negative = int(np.count_nonzero(counts - values < -3.0 * _noise(counts)))
    if negative > 0.02 * counts.size:
        warnings.append(
            f"{negative} puntos ({100 * negative / counts.size:.0f} %) quedan "
            "por debajo del fondo más de lo que explica el ruido: el fondo "
            "está demasiado alto y las áreas saldrán cortas"
        )
    warnings.extend(_endpoint_warnings(energy, counts, points))
    return Background(
        values=values, kind="Shirley",
        window=(float(energy[0]), float(energy[-1])),
        endpoints=(low, high), iterations=iterations,
        parameters={"tolerancia": tolerance}, warnings=warnings,
    )


def tougaard_background(
    energy: np.ndarray,
    counts: np.ndarray,
    b: float = TOUGAARD_B,
    c: float = TOUGAARD_C,
    points: int = ENDPOINT_POINTS,
) -> Background:
    """The Tougaard universal-cross-section background.

    ``B(E) = k ∫ K(E − E′) I(E′) dE′`` over ``E′ < E``, with the universal
    loss function ``K(T) = B T / (C + T²)²``. The scale ``k`` is fixed by
    requiring the background to meet the spectrum at the
    high-binding-energy end, which is the standard normalisation and is the
    step that makes the answer depend on the region being wide enough.

    Unlike Shirley, this does not assume the spectrum returns to a flat
    region, so it is the right choice across a wide window containing
    several lines. Over a 10 eV region it is not: there is no loss tail
    inside the window to scale against.
    """
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    span = float(energy[-1] - energy[0])
    warnings: list[str] = []
    if span < TOUGAARD_MIN_SPAN:
        warnings.append(
            f"la región mide {span:.0f} eV y un fondo de Tougaard necesita "
            f"unos {TOUGAARD_MIN_SPAN:.0f} eV por encima del pico en energía "
            "de enlace para tener cola de pérdidas con la que escalarse. Aquí "
            "Shirley es mejor elección"
        )
    low, high = _endpoints(counts, points)
    step = float(np.median(np.diff(energy)))
    signal = np.clip(counts - low, 0.0, None)

    # K(E_i − E_j) for j < i, built once as a lower-triangular matrix. The
    # regions are a few hundred points, so the memory is trivial and the
    # matrix product is far faster than a Python loop over lags.
    loss = energy[:, None] - energy[None, :]
    kernel = np.where(loss > 0, b * loss / (c + loss**2) ** 2, 0.0)
    raw = kernel @ signal * step
    scale = (high - low) / raw[-1] if raw[-1] > 0 else 0.0
    if scale <= 0:
        warnings.append(
            "la cola de pérdidas integrada es nula: no se puede escalar un "
            "fondo de Tougaard con esta región"
        )
    values = low + scale * raw
    warnings.extend(_endpoint_warnings(energy, counts, points))
    return Background(
        values=values, kind="Tougaard",
        window=(float(energy[0]), float(energy[-1])),
        endpoints=(low, high),
        parameters={"B_ev2": b, "C_ev2": c, "escala": float(scale)},
        warnings=warnings,
    )


def _noise(counts: np.ndarray) -> float:
    """Robust noise level, from the median absolute second difference."""
    if counts.size < 5:
        return 0.0
    return float(np.median(np.abs(np.diff(counts, n=2)))) / (0.6745 * np.sqrt(6.0))


def _endpoint_warnings(
    energy: np.ndarray, counts: np.ndarray, points: int
) -> list[str]:
    """Complain when an endpoint is not on anything that looks like a floor.

    An endpoint on the flank of a peak is the commonest way an XPS area
    goes wrong, and it is invisible in the fit statistics: the fit is just
    as good, the areas are just different.
    """
    out: list[str] = []
    n = max(2, min(int(points) * 2, counts.size // 5))
    noise = _noise(counts)
    if noise <= 0:
        return out
    for name, block, axis in (
        ("bajo", counts[:n], energy[:n]),
        ("alto", counts[-n:], energy[-n:]),
    ):
        slope = float(np.polyfit(axis, block, 1)[0])
        change = abs(slope) * (axis[-1] - axis[0])
        if change > 4.0 * noise:
            out.append(
                f"el extremo de energía de enlace {name} de la región está en "
                f"pendiente ({change:.0f} cuentas de cambio en {axis[-1] - axis[0]:.1f} "
                "eV, frente a un ruido de "
                f"{noise:.0f}): el fondo se ancla ahí, así que mover ese "
                "límite moverá todas las áreas"
            )
    return out


#: The backgrounds, by the name the CLI and the GUI use.
BACKGROUNDS = {
    "lineal": linear_background,
    "shirley": shirley_background,
    "tougaard": tougaard_background,
}

_BACKGROUND_ALIASES = {
    "linear": "lineal", "recta": "lineal", "line": "lineal",
    "s": "shirley", "t": "tougaard", "universal": "tougaard",
    "ninguno": "ninguno", "none": "ninguno", "sin": "ninguno",
}


def estimate_background(
    spectrum: XPSSpectrum,
    window: Optional[tuple[float, float]] = None,
    kind: str = "shirley",
    envelope: Optional[np.ndarray] = None,
    points: int = ENDPOINT_POINTS,
    **options,
) -> Background:
    """Background under a region of a spectrum.

    Parameters
    ----------
    spectrum:
        The measurement.
    window:
        ``(low, high)`` in eV. Defaults to the whole spectrum, which is
        almost never what you want for a Shirley.
    kind:
        ``"shirley"``, ``"tougaard"``, ``"lineal"`` or ``"ninguno"``.
    envelope:
        Fitted peak envelope, for the active Shirley. Ignored by the other
        kinds, which do not use one.
    """
    name = _BACKGROUND_ALIASES.get(kind.strip().lower(), kind.strip().lower())
    low, high = window if window else spectrum.range
    energy, counts = spectrum.region_of(low, high)
    if energy.size < 5:
        raise XPSError(
            f"la ventana {low:g}–{high:g} eV deja {energy.size} puntos del "
            f"espectro, que va de {spectrum.range[0]:.1f} a "
            f"{spectrum.range[1]:.1f} eV"
        )
    if name == "ninguno":
        return Background(
            values=np.zeros_like(energy), kind="ninguno",
            window=(float(energy[0]), float(energy[-1])), endpoints=(0.0, 0.0),
            warnings=["sin fondo: las áreas incluyen el escalón inelástico"],
        )
    if name not in BACKGROUNDS:
        raise XPSError(
            f"fondo desconocido {kind!r}; hay: {', '.join(BACKGROUNDS)}, ninguno"
        )
    if name == "shirley":
        return shirley_background(energy, counts, envelope=envelope, points=points,
                                  **options)
    return BACKGROUNDS[name](energy, counts, points=points, **options)


__all__ = [
    "BACKGROUNDS",
    "Background",
    "ENDPOINT_POINTS",
    "TOUGAARD_B",
    "TOUGAARD_C",
    "TOUGAARD_MIN_SPAN",
    "estimate_background",
    "linear_background",
    "shirley_background",
    "tougaard_background",
]
