"""Impedance: equivalent circuits, complex non-linear fitting, Kramers–Kronig.

A circuit is written as a string — ``R0-(R1|Q1)-Wo1`` — and parsed into a
tree: ``-`` is series, ``|`` is parallel, parentheses group. That is the
notation everyone already writes on a whiteboard, and it means a model is
specified in one line rather than assembled in code.

Three things here are not decoration.

**Fitting is complex non-linear least squares with modulus weighting.**
The real and imaginary parts are fitted together, not separately, and each
residual is divided by ``|Z|`` at that frequency. Without the weighting a
spectrum spanning four decades of impedance is fitted almost entirely at
its low-frequency end, and the charge-transfer semicircle — the part
anyone cares about — contributes nothing to the objective.

**A CPE is not a capacitor.** Its parameter ``Q`` has units of S·sⁿ and is
not a capacitance; the two are equal only when n = 1. Converting requires
knowing what the CPE is in parallel with, so the conversion is offered
explicitly (:func:`cpe_to_capacitance`) rather than being done silently,
and the ``n`` is always reported next to it.

**Kramers–Kronig comes before the circuit.** The KK relations must hold for
any system that is linear, causal, stable and time-invariant, whatever its
circuit. A spectrum that fails them was measured on a cell that drifted
while it was being measured, and fitting a circuit to it produces
parameters with no meaning — which no goodness of fit will reveal, because
a flexible enough circuit fits drifted data perfectly well. The test is run
first and reported first.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .curve import Impedance

#: Number of RC elements per decade in the linear Kramers–Kronig test.
KK_PER_DECADE = 7

#: RMS residual above which the KK test fails outright, whatever its shape.
KK_LIMIT = 0.02

#: Sign-run ratio below which the residual has systematic structure. The
#: threshold matters more than the residual size: 0.5 % measurement noise
#: on sixty points puts the largest residual near 1.5 % by chance alone, so
#: a fixed 1 % cutoff fails perfectly good data. What drift actually
#: produces is a residual with long stretches of one sign.
KK_RUNS_LIMIT = 0.6

#: Circuits offered by name.
LIBRARY: dict[str, str] = {
    "randles": "R0-(R1|C1)",
    "randles_cpe": "R0-(R1|Q1)",
    "randles_warburg": "R0-(R1-W1|Q1)",
    "supercondensador": "R0-(R1|Q1)-Q2",
    "bateria": "R0-(R1|Q1)-(R2|Q2)-Wo1",
    "dos_semicirculos": "R0-(R1|Q1)-(R2|Q2)",
    "con_inductancia": "L0-R0-(R1|Q1)",
}


class CircuitError(ValueError):
    """Raised when a circuit string cannot be parsed or evaluated."""


# -- circuit elements --------------------------------------------------


def _z_r(omega: np.ndarray, r: float) -> np.ndarray:
    return np.full(omega.shape, complex(r))


def _z_c(omega: np.ndarray, c: float) -> np.ndarray:
    return 1.0 / (1j * omega * max(c, 1e-18))


def _z_l(omega: np.ndarray, inductance: float) -> np.ndarray:
    return 1j * omega * inductance


def _z_q(omega: np.ndarray, q: float, n: float) -> np.ndarray:
    return 1.0 / (max(q, 1e-18) * (1j * omega) ** n)


def _z_w(omega: np.ndarray, sigma: float) -> np.ndarray:
    return sigma * (1.0 - 1j) / np.sqrt(omega)


def _z_ws(omega: np.ndarray, r: float, tau: float) -> np.ndarray:
    """Finite-length transmissive Warburg."""
    root = np.sqrt(1j * omega * max(tau, 1e-18))
    return r * np.tanh(root) / root


def _z_wo(omega: np.ndarray, r: float, tau: float) -> np.ndarray:
    """Finite-length reflective (open) Warburg, ``Z = R·coth(√(jωτ))/√(jωτ)``.

    The right one for a battery electrode: diffusion into a particle ends
    at its centre, which is a reflecting boundary, and the low-frequency
    response is capacitive rather than resistive. Using the transmissive
    form there gives a finite low-frequency resistance where the data show
    a vertical line.
    """
    root = np.sqrt(1j * omega * max(tau, 1e-18))
    return r / (np.tanh(root) * root)


#: Element symbol → (impedance function, parameter names, defaults, bounds,
#: whether each parameter is fitted in logarithmic space).
#:
#: Everything but the CPE exponent is fitted as its logarithm. These
#: parameters span ten or more orders of magnitude — a capacitance of 1e-5 F
#: bounded between 1e-15 and 1 — and a trust-region step that is sensible
#: for a 40 Ω resistance is astronomically wrong for a 2e-4 S·sⁿ CPE. Fitted
#: linearly, a Randles circuit with a capacitive tail walked its resistance
#: to the bound and reported convergence.
ELEMENTS: dict[str, tuple] = {
    "R": (_z_r, ("R",), (10.0,), ((1e-6,), (1e9,)), (True,)),
    "C": (_z_c, ("C",), (1e-5,), ((1e-15,), (1.0,)), (True,)),
    "L": (_z_l, ("L",), (1e-6,), ((1e-12,), (1.0,)), (True,)),
    "Q": (_z_q, ("Q", "n"), (1e-4, 0.85), ((1e-15, 0.3), (10.0, 1.0)), (True, False)),
    "W": (_z_w, ("sigma",), (10.0,), ((1e-6,), (1e9,)), (True,)),
    "Ws": (_z_ws, ("R", "tau"), (10.0, 1.0), ((1e-6, 1e-6), (1e9, 1e6)), (True, True)),
    "Wo": (_z_wo, ("R", "tau"), (10.0, 1.0), ((1e-6, 1e-6), (1e9, 1e6)), (True, True)),
}

_TOKEN = re.compile(r"(Ws|Wo|[RCLQW])(\d*)")


@dataclass
class Element:
    """One circuit element instance."""

    symbol: str
    name: str
    parameters: list[str]
    values: list[float]

    @property
    def labels(self) -> list[str]:
        return [f"{self.name}.{p}" for p in self.parameters]

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        function = ELEMENTS[self.symbol][0]
        return function(omega, *self.values)


class _Node:
    """Series or parallel combination, or a leaf element."""

    def __init__(self, kind: str, children: list) -> None:
        self.kind = kind
        self.children = children

    def impedance(self, omega: np.ndarray) -> np.ndarray:
        if self.kind == "element":
            return self.children[0].impedance(omega)
        parts = [child.impedance(omega) for child in self.children]
        if self.kind == "series":
            return sum(parts)
        admittance = sum(1.0 / np.where(np.abs(p) > 0, p, 1e-30) for p in parts)
        return 1.0 / admittance

    def elements(self) -> list[Element]:
        if self.kind == "element":
            return [self.children[0]]
        return [e for child in self.children for e in child.elements()]


def parse_circuit(text: str) -> _Node:
    """Parse a circuit string into a tree.

    ``-`` is series, ``|`` is parallel, parentheses group, and an element
    is a symbol plus an optional index: ``R0-(R1|Q1)-Wo1``.
    """
    cleaned = text.replace(" ", "")
    if not cleaned:
        raise CircuitError("circuito vacío")

    position = 0

    def peek() -> str:
        return cleaned[position] if position < len(cleaned) else ""

    def parse_series() -> _Node:
        nonlocal position
        parts = [parse_parallel()]
        while peek() == "-":
            position += 1
            parts.append(parse_parallel())
        return parts[0] if len(parts) == 1 else _Node("series", parts)

    def parse_parallel() -> _Node:
        nonlocal position
        parts = [parse_atom()]
        while peek() == "|":
            position += 1
            parts.append(parse_atom())
        return parts[0] if len(parts) == 1 else _Node("parallel", parts)

    def parse_atom() -> _Node:
        nonlocal position
        if peek() == "(":
            position += 1
            inner = parse_series()
            if peek() != ")":
                raise CircuitError(f"falta un paréntesis de cierre en {text!r}")
            position += 1
            return inner
        match = _TOKEN.match(cleaned, position)
        if not match:
            raise CircuitError(
                f"no se entiende {cleaned[position:position + 4]!r} en {text!r}; "
                f"elementos válidos: {', '.join(sorted(ELEMENTS))}"
            )
        position = match.end()
        symbol = match.group(1)
        name = symbol + (match.group(2) or "")
        _, names, defaults, _, _ = ELEMENTS[symbol]
        return _Node(
            "element",
            [Element(symbol=symbol, name=name, parameters=list(names),
                     values=list(defaults))],
        )

    tree = parse_series()
    if position != len(cleaned):
        raise CircuitError(f"sobra {cleaned[position:]!r} al final de {text!r}")
    return tree


@dataclass
class CircuitFit:
    """The result of fitting an equivalent circuit."""

    circuit: str
    elements: list[Element]
    chi_squared: float
    residual_pct: float
    """Mean modulus-weighted residual, as a percentage."""
    errors: dict[str, float] = field(default_factory=dict)
    fitted: Optional[np.ndarray] = None
    converged: bool = False
    message: str = ""
    warnings: list[str] = field(default_factory=list)

    def values(self) -> dict[str, float]:
        return {
            label: value
            for element in self.elements
            for label, value in zip(element.labels, element.values)
        }

    def summary(self) -> str:
        lines = [
            f"Circuito: {self.circuit}",
            f"  {'convergido' if self.converged else 'NO convergido'}: {self.message}",
            f"  χ² = {self.chi_squared:.4g}   residuo medio "
            f"{self.residual_pct:.3f} %",
            "",
        ]
        for element in self.elements:
            for label, value, name in zip(
                element.labels, element.values, element.parameters
            ):
                error = self.errors.get(label)
                unit = _unit(element.symbol, name)
                text = f"  {label:<10s} = {value:12.6g} {unit}"
                if error is not None and value != 0:
                    text += f"  (± {100 * error / abs(value):.1f} %)"
                lines.append(text)
        if self.warnings:
            lines.append("")
            lines.extend("  ⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def _unit(symbol: str, parameter: str) -> str:
    units = {
        ("R", "R"): "Ω", ("C", "C"): "F", ("L", "L"): "H",
        ("Q", "Q"): "S·sⁿ", ("Q", "n"): "", ("W", "sigma"): "Ω·s^-½",
        ("Ws", "R"): "Ω", ("Ws", "tau"): "s",
        ("Wo", "R"): "Ω", ("Wo", "tau"): "s",
    }
    return units.get((symbol, parameter), "")


def cpe_to_capacitance(q: float, n: float, resistance: float) -> float:
    """Effective capacitance of a CPE in parallel with a resistance.

    Brug's relation, ``C = (Q · R^(1−n))^(1/n) / R``. The parameter ``Q`` of
    a constant-phase element has units of S·sⁿ and **is not** a
    capacitance; quoting it as one is wrong by orders of magnitude when n
    is far from 1. The conversion needs the resistance the CPE is in
    parallel with, which is why it is a separate call rather than something
    done silently.
    """
    if not 0.0 < n <= 1.0 or resistance <= 0.0 or q <= 0.0:
        raise CircuitError("parámetros fuera de rango para convertir un CPE")
    return float((q * resistance ** (1.0 - n)) ** (1.0 / n) / resistance)


def fit_circuit(
    spectrum: Impedance,
    circuit: str = "randles_cpe",
    initial: Optional[dict[str, float]] = None,
    max_iterations: int = 4000,
) -> CircuitFit:
    """Fit an equivalent circuit by complex non-linear least squares.

    Parameters
    ----------
    spectrum:
        The measurement.
    circuit:
        A circuit string, or a key of :data:`LIBRARY`.
    initial:
        Starting values by label (``"R1.R"``). Anything not given is
        estimated from the data — the series resistance from the
        high-frequency intercept, the polarisation resistance from the
        low-frequency one, and the capacitances from the frequency of the
        imaginary maximum — which matters, because a CNLS fit started far
        from the answer finds a local minimum and reports it happily.

    Returns
    -------
    CircuitFit
    """
    text = LIBRARY.get(circuit, circuit)
    tree = parse_circuit(text)
    elements = tree.elements()
    if not elements:
        raise CircuitError("el circuito no tiene elementos")

    _seed(elements, spectrum)
    if initial:
        for element in elements:
            for index, label in enumerate(element.labels):
                if label in initial:
                    element.values[index] = float(initial[label])

    labels: list[str] = []
    start: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    logarithmic: list[bool] = []
    for element in elements:
        _, names, _, bounds, logs = ELEMENTS[element.symbol]
        for index, name in enumerate(names):
            labels.append(f"{element.name}.{name}")
            start.append(element.values[index])
            lower.append(bounds[0][index])
            upper.append(bounds[1][index])
            logarithmic.append(bool(logs[index]))

    is_log = np.array(logarithmic)
    start_array = np.clip(
        np.asarray(start, dtype=float),
        np.asarray(lower) * 1.0001,
        np.asarray(upper) * 0.9999,
    )

    def to_internal(values: np.ndarray) -> np.ndarray:
        out = np.array(values, dtype=float)
        out[is_log] = np.log10(np.maximum(out[is_log], 1e-300))
        return out

    def to_external(values: np.ndarray) -> np.ndarray:
        out = np.array(values, dtype=float)
        out[is_log] = 10.0 ** out[is_log]
        return out

    omega = spectrum.omega
    observed = spectrum.z
    weight = 1.0 / np.maximum(np.abs(observed), 1e-12)

    def unpack(internal: np.ndarray) -> None:
        values = to_external(internal)
        position = 0
        for element in elements:
            count = len(element.parameters)
            element.values = [float(v) for v in values[position:position + count]]
            position += count

    def residual(internal: np.ndarray) -> np.ndarray:
        unpack(internal)
        model = tree.impedance(omega)
        difference = (model - observed) * weight
        return np.concatenate([difference.real, difference.imag])

    internal_lower = to_internal(np.asarray(lower))
    internal_upper = to_internal(np.asarray(upper))

    def run(guess: np.ndarray):
        return least_squares(
            residual,
            np.clip(guess, internal_lower + 1e-9, internal_upper - 1e-9),
            bounds=(internal_lower, internal_upper),
            x_scale="jac",
            max_nfev=max_iterations,
            ftol=1e-13,
            xtol=1e-13,
        )

    # A few restarts around the seed. Complex non-linear least squares on a
    # circuit with a Warburg inside a parallel branch has local minima that
    # fit the data to the noise level with entirely wrong parameters, and
    # the only cheap defence is to look in more than one place.
    base = to_internal(start_array)
    rng = np.random.default_rng(0)
    outcome = run(base)
    for _ in range(3):
        perturbed = base.copy()
        perturbed[is_log] += rng.normal(0.0, 0.7, int(is_log.sum()))
        candidate = run(perturbed)
        if candidate.cost < outcome.cost * 0.999:
            outcome = candidate
    unpack(outcome.x)
    model = tree.impedance(omega)

    chi_squared = float((outcome.fun**2).sum())
    residual_pct = 100.0 * float(np.mean(np.abs(model - observed) / np.abs(observed)))

    fit = CircuitFit(
        circuit=text,
        elements=elements,
        chi_squared=chi_squared,
        residual_pct=residual_pct,
        fitted=model,
        converged=bool(outcome.success),
        message=str(outcome.message),
    )
    _errors(outcome, labels, is_log, fit, spectrum.n)
    _check_bounds(fit, labels, lower, upper)
    _circuit_warnings(fit, spectrum)
    return fit


def _seed(elements: Sequence[Element], spectrum: Impedance) -> None:
    """Sensible starting values read off the spectrum.

    Estimated from the shape, not from the endpoints. The classic recipe
    takes the polarisation resistance as the span between the two real
    intercepts, which fails for anything with a capacitive tail — a
    supercapacitor has no low-frequency intercept at all, its Z' runs away
    with the tail, and the seed came out an order of magnitude too large
    and the fit walked its resistance to the bound.

    So the semicircle is located instead: the apex of −Z″ gives its
    diameter (twice the height above the series resistance) and its time
    constant, and the low-frequency end gives the tail's capacitance
    directly from ``|Z″| = 1/ωC``.
    """
    real = spectrum.z.real
    imaginary = -spectrum.z.imag
    series = max(float(real.min()), 1e-6)

    # The semicircle apex is a LOCAL maximum of -Z''. Taking the global one
    # works for a Randles circuit and fails for anything with a capacitive
    # tail, where -Z'' rises without limit towards low frequency and the
    # global maximum is the last point — which seeded the charge-transfer
    # resistance at the value of the tail and sent the fit to its bound.
    apex = _semicircle_apex(imaginary, real, series)
    height = max(float(imaginary[apex]), 1e-9)
    polarisation = max(2.0 * height, 1e-3)
    omega_peak = max(float(spectrum.omega[apex]), 1e-9)
    capacitance = 1.0 / (omega_peak * polarisation)

    # The low-frequency tail, if there is one: |Z''| = 1/(omega C).
    tail = 1.0 / max(spectrum.omega[-1] * abs(spectrum.z.imag[-1]), 1e-12)

    resistors = [e for e in elements if e.symbol == "R"]
    capacitive = [e for e in elements if e.symbol in ("C", "Q")]
    for index, element in enumerate(resistors):
        element.values = [
            series if index == 0
            else polarisation / max(len(resistors) - 1, 1)
        ]
    for index, element in enumerate(capacitive):
        # The first capacitive element is the double layer (fast); a second
        # one in series is the storage tail (slow, much larger).
        value = capacitance if index == 0 else max(tail, capacitance * 10.0)
        element.values = [value] if element.symbol == "C" else [value, 0.9]
    for element in elements:
        if element.symbol == "W":
            element.values = [polarisation]
        elif element.symbol in ("Ws", "Wo"):
            element.values = [polarisation, 1.0 / omega_peak]
        elif element.symbol == "L":
            element.values = [1e-6]


def _semicircle_apex(imaginary: np.ndarray, real: np.ndarray, series: float) -> int:
    """Index of the charge-transfer semicircle's apex.

    The highest-frequency prominent local maximum of −Z″. Falls back to
    the point where Z′ has covered half its range, which is where the apex
    of a single semicircle sits.
    """
    from scipy.signal import find_peaks as _find_peaks

    if imaginary.size > 4:
        indices, _ = _find_peaks(
            imaginary, prominence=0.05 * max(float(imaginary.max()), 1e-12)
        )
        if indices.size:
            # Frequencies are sorted descending, so the first index is the
            # highest-frequency peak.
            return int(indices[0])
    target = series + 0.5 * (float(real.max()) - series)
    return int(np.argmin(np.abs(real - target)))


def _errors(
    outcome, labels: Sequence[str], is_log: np.ndarray, fit: CircuitFit, points: int
) -> None:
    """Standard errors, converted back out of the logarithmic parameters.

    The Jacobian is with respect to log₁₀ p, so its uncertainty is a
    *relative* one: ``σ_p / p = ln(10) · σ_log``. Reporting the log-space
    number as if it were linear turned a 3 % uncertainty into 3 000 000 %.
    """
    try:
        jacobian = outcome.jac
        _, singular, vt = np.linalg.svd(jacobian, full_matrices=False)
        threshold = np.finfo(float).eps * max(jacobian.shape) * singular[0]
        keep = singular > threshold
        covariance = (vt[keep].T / singular[keep] ** 2) @ vt[keep]
        dof = max(2 * points - len(labels), 1)
        scale = 2.0 * outcome.cost / dof
        errors = np.sqrt(np.maximum(np.diag(covariance) * scale, 0.0))
    except (np.linalg.LinAlgError, ValueError, IndexError):  # pragma: no cover
        return
    values = fit.values()
    converted: dict[str, float] = {}
    for label, error, logarithmic in zip(labels, errors, is_log):
        if logarithmic:
            converted[label] = float(math.log(10.0) * error * abs(values.get(label, 0.0)))
        else:
            converted[label] = float(error)
    fit.errors = converted


def _check_bounds(
    fit: CircuitFit, labels: Sequence[str], lower: Sequence[float],
    upper: Sequence[float],
) -> None:
    """Flag any parameter that ended up sitting on its own limit.

    A value at a bound is not a fitted value: the optimiser wanted to go
    further and could not. It almost always means the circuit is wrong for
    the data — fitting a plain Randles to a spectrum with a capacitive tail
    sends the charge-transfer resistance to infinity, because that is the
    only way the model can produce a rising low-frequency impedance — and
    the number that comes out looks like a measurement.
    """
    values = fit.values()
    for label, low, high in zip(labels, lower, upper):
        value = values.get(label)
        if value is None:
            continue
        for limit, side in ((low, "inferior"), (high, "superior")):
            if limit and abs(value - limit) <= abs(limit) * 1e-6:
                fit.warnings.append(
                    f"{label} ha quedado pegado a su límite {side} "
                    f"({limit:g}). Eso NO es un valor ajustado: el optimizador "
                    "quería seguir y no podía. Casi siempre significa que el "
                    "circuito no es el adecuado para estos datos — por ejemplo "
                    "un Randles simple contra un espectro con cola capacitiva, "
                    "donde la única forma que tiene el modelo de subir a baja "
                    "frecuencia es mandar la resistencia a infinito"
                )
                break


def _circuit_warnings(fit: CircuitFit, spectrum: Impedance) -> None:
    values = fit.values()
    for element in fit.elements:
        if element.symbol == "Q":
            n = element.values[1]
            if n > 0.98:
                fit.warnings.append(
                    f"{element.name}: n = {n:.3f}, prácticamente 1. Un CPE con "
                    "n = 1 es un condensador: usa C y tendrás un parámetro "
                    "menos y una incertidumbre menor"
                )
            elif n < 0.6:
                fit.warnings.append(
                    f"{element.name}: n = {n:.3f}, muy lejos de 1. Con esa n "
                    "el elemento ya no describe una doble capa distribuida; "
                    "mira si lo que hay ahí es difusión (Warburg, n = 0.5)"
                )
            fit.warnings.append(
                f"{element.name}: Q = {element.values[0]:.4g} S·sⁿ NO son "
                "faradios. Para dar una capacitancia hace falta convertir con "
                "la resistencia en paralelo (echem.eis.cpe_to_capacitance) y "
                "citar siempre la n al lado"
            )
    for label, error in fit.errors.items():
        value = values.get(label, 0.0)
        if value and error / abs(value) > 0.5:
            fit.warnings.append(
                f"{label} tiene una incertidumbre del "
                f"{100 * error / abs(value):.0f} %: ese elemento no está "
                "determinado por los datos. O sobra en el circuito, o su "
                "constante de tiempo cae fuera del rango de frecuencias que "
                "mediste"
            )
    if fit.residual_pct > 5.0:
        fit.warnings.append(
            f"el residuo medio es {fit.residual_pct:.1f} %. El circuito no "
            "describe estos datos: mira el diagrama de Nyquist y compara la "
            "forma, no el número"
        )
    if -spectrum.z.imag[0] < 0:
        fit.warnings.append(
            "a alta frecuencia el espectro entra en la mitad inductiva del "
            "plano (−Z″ negativo). Casi siempre son los cables, no la muestra: "
            "añade una L en serie al circuito o recorta esas frecuencias"
        )


# -- Kramers-Kronig ----------------------------------------------------


@dataclass
class KKResult:
    """The linear Kramers–Kronig consistency test."""

    residual_real: np.ndarray
    residual_imag: np.ndarray
    max_residual: float
    passes: bool
    n_elements: int
    rms_residual: float = 0.0
    runs_ratio: float = 1.0
    """Sign changes in the residual over the number chance would give. Near
    1 means scatter; well below means systematic structure."""
    message: str = ""

    def summary(self) -> str:
        verdict = "consistente" if self.passes else "NO CONSISTENTE"
        return (
            f"Kramers-Kronig ({self.n_elements} elementos RC): {verdict}\n"
            f"  residuo RMS {100 * self.rms_residual:.2f} %, máximo "
            f"{100 * self.max_residual:.2f} %, rachas de signo "
            f"{self.runs_ratio:.2f}× lo esperado por azar\n  {self.message}"
        )


def kramers_kronig(spectrum: Impedance, per_decade: int = KK_PER_DECADE) -> KKResult:
    """Boukamp's linear Kramers–Kronig test.

    A series of RC elements with time constants fixed on a logarithmic grid
    across the measured range is fitted to the data. Every such circuit is
    KK-compliant by construction, so if the data can be fitted by one, they
    are consistent with linearity, causality and stability; if they cannot,
    they are not — and no equivalent circuit fitted to them means anything,
    however good its χ² looks.

    The usual cause of failure is the cell changing while it was being
    measured: a slow impedance sweep at a potential where something is
    corroding, dissolving or intercalating takes minutes, and the sample at
    the end is not the sample at the start.
    """
    omega = spectrum.omega
    decades = math.log10(float(omega.max() / omega.min()))
    count = max(5, int(round(per_decade * max(decades, 1.0))))
    taus = 1.0 / np.logspace(
        math.log10(float(omega.min())), math.log10(float(omega.max())), count
    )

    # Z(w) = R_inf + 1/(j w C_s) + j w L + sum_k R_k/(1 + j w tau_k), linear
    # in R_inf, 1/C_s, L and the R_k. The series capacitance and the
    # inductance are not optional extras: a finite sum of RC elements
    # cannot reproduce a low-frequency capacitive tail, and without the C_s
    # term a perfectly good supercapacitor spectrum fails the test with an
    # 8 % residual that has nothing to do with drift. Every added term is
    # itself Kramers-Kronig compliant, so the test keeps its meaning.
    basis = 1.0 / (1.0 + 1j * np.outer(omega, taus))
    series_c = 1.0 / (1j * omega)
    inductance = 1j * omega
    design = np.hstack(
        [
            np.vstack([np.ones((omega.size, 1)), np.zeros((omega.size, 1))]),
            np.vstack([series_c.real[:, None], series_c.imag[:, None]]),
            np.vstack([inductance.real[:, None], inductance.imag[:, None]]),
            np.vstack([basis.real, basis.imag]),
        ]
    )
    weight = 1.0 / np.maximum(np.abs(spectrum.z), 1e-12)
    target = np.concatenate([spectrum.z.real * weight, spectrum.z.imag * weight])
    design = design * np.concatenate([weight, weight])[:, None]

    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    model = design @ solution
    residual = target - model
    half = omega.size
    residual_real = residual[:half] / np.maximum(np.abs(weight * spectrum.z), 1e-12)
    residual_imag = residual[half:] / np.maximum(np.abs(weight * spectrum.z), 1e-12)
    worst = float(max(np.abs(residual_real).max(), np.abs(residual_imag).max()))
    both = np.concatenate([residual_real, residual_imag])
    rms = float(np.sqrt(np.mean(both**2)))
    runs = _runs_ratio(both)

    passes = rms <= KK_LIMIT and runs >= KK_RUNS_LIMIT
    message = (
        "los datos son compatibles con un sistema lineal, causal y estable; "
        "cualquier circuito equivalente que ajustes tiene sentido"
        if passes
        else (
            "los datos NO satisfacen Kramers-Kronig. La causa habitual es que "
            "la celda cambió MIENTRAS medías: un barrido de impedancia lento a "
            "un potencial donde algo se corroe, se disuelve o se intercala "
            "tarda minutos, y la muestra del final no es la del principio. "
            "Ajustar un circuito a esto da parámetros sin significado, y el χ² "
            "no lo delata porque un circuito flexible ajusta bien datos que "
            "han derivado. Repite midiendo de alta a baja frecuencia y otra "
            "vez al revés: si las dos no coinciden, ahí está"
        )
    )
    return KKResult(
        residual_real=residual_real,
        residual_imag=residual_imag,
        max_residual=worst,
        passes=passes,
        n_elements=count,
        rms_residual=rms,
        runs_ratio=runs,
        message=message,
    )


def _runs_ratio(residual: np.ndarray) -> float:
    """Sign changes observed over the number independent noise would give."""
    signs = np.sign(np.asarray(residual, dtype=float))
    signs = signs[signs != 0]
    if signs.size < 10:
        return 1.0
    runs = 1 + int(np.count_nonzero(np.diff(signs) != 0))
    positive = int(np.count_nonzero(signs > 0))
    negative = signs.size - positive
    if positive == 0 or negative == 0:
        return 0.0
    expected = 1.0 + 2.0 * positive * negative / signs.size
    return float(runs / expected)


# -- everything together -----------------------------------------------


@dataclass
class EISResult:
    """The analysis of one impedance spectrum."""

    spectrum: Impedance
    kk: Optional[KKResult] = None
    fit: Optional[CircuitFit] = None
    series_resistance: Optional[float] = None
    charge_transfer_resistance: Optional[float] = None
    capacitance_f: Optional[float] = None
    knee_frequency: Optional[float] = None
    """Where the response stops being a semicircle and starts being a line."""
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [self.spectrum.describe(), ""]
        if self.kk:
            lines.append(self.kk.summary())
            lines.append("")
        if self.series_resistance is not None:
            lines.append(
                f"Resistencia en serie (corte a alta f): "
                f"{self.series_resistance:.4g} Ω"
            )
        if self.charge_transfer_resistance is not None:
            lines.append(
                f"Resistencia de transferencia de carga: "
                f"{self.charge_transfer_resistance:.4g} Ω"
            )
        if self.capacitance_f is not None:
            lines.append(
                f"Capacitancia efectiva (Brug, desde el CPE): "
                f"{1e3 * self.capacitance_f:.4g} mF"
            )
        if self.knee_frequency is not None:
            lines.append(
                f"Frecuencia de codo: {self.knee_frequency:.4g} Hz "
                f"(τ = {1.0 / self.knee_frequency:.4g} s)"
            )
        if self.fit:
            lines.append("")
            lines.append(self.fit.summary())
        if self.warnings:
            lines.append("")
            lines.extend("⚠ " + w for w in self.warnings)
        return "\n".join(lines)


def analyse_eis(
    spectrum: Impedance,
    circuit: Optional[str] = "randles_cpe",
    check_kk: bool = True,
) -> EISResult:
    """Impedance analysis: consistency first, then a circuit.

    ``circuit=None`` skips the fit and returns only the model-free
    quantities, which is the right thing to do when the Kramers–Kronig test
    fails.
    """
    result = EISResult(spectrum=spectrum)
    if check_kk:
        result.kk = kramers_kronig(spectrum)
        if not result.kk.passes:
            result.warnings.append(
                "el espectro no pasa Kramers-Kronig. Los parámetros del "
                "circuito que salgan abajo NO son fiables, por muy bien que "
                "ajuste"
            )

    result.series_resistance = float(spectrum.z.real[0])
    imaginary = -spectrum.z.imag
    if imaginary.size > 2:
        knee = int(np.argmax(imaginary))
        result.knee_frequency = float(spectrum.frequency[knee])

    if circuit:
        result.fit = fit_circuit(spectrum, circuit)
        values = result.fit.values()
        resistances = [
            (label, value) for label, value in values.items() if label.endswith(".R")
        ]
        if len(resistances) >= 2:
            result.charge_transfer_resistance = resistances[1][1]
        cpes = [e for e in result.fit.elements if e.symbol == "Q"]
        if cpes and result.charge_transfer_resistance:
            try:
                result.capacitance_f = cpe_to_capacitance(
                    cpes[0].values[0], cpes[0].values[1],
                    result.charge_transfer_resistance,
                )
            except CircuitError:
                pass
        result.warnings.extend(result.fit.warnings)
    return result


__all__ = [
    "DRT_PER_DECADE",
    "DRTResult",
    "drt",
    "ELEMENTS",
    "KK_LIMIT",
    "KK_PER_DECADE",
    "KK_RUNS_LIMIT",
    "LIBRARY",
    "CircuitError",
    "CircuitFit",
    "EISResult",
    "Element",
    "KKResult",
    "analyse_eis",
    "cpe_to_capacitance",
    "fit_circuit",
    "kramers_kronig",
    "parse_circuit",
]


# -- distribution of relaxation times ----------------------------------

#: How many time constants per decade the DRT is solved on. More is not
#: better: the problem is ill-posed, and a finer grid buys resolution the
#: data does not contain while making the regularisation do more work.
DRT_PER_DECADE = 10


@dataclass
class DRTResult:
    """A distribution of relaxation times."""

    tau_s: np.ndarray
    gamma: np.ndarray
    """γ(τ) in ohms: the area under a peak is that process's resistance."""
    regularisation: float
    fitted: np.ndarray
    """The impedance the distribution reproduces."""
    residual: float
    peaks_s: list[float] = field(default_factory=list)
    peak_resistances: list[float] = field(default_factory=list)
    ohmic: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        if not self.peaks_s:
            return (f"DRT: sin picos separados (λ = {self.regularisation:.3g}, "
                    f"residuo {100 * self.residual:.2f} %)")
        parts = ", ".join(
            f"τ = {tau:.3g} s ({resistance:.3g} Ω)"
            for tau, resistance in zip(self.peaks_s, self.peak_resistances))
        return (f"DRT: {len(self.peaks_s)} procesos — {parts}; R_s = "
                f"{self.ohmic:.3g} Ω, residuo {100 * self.residual:.2f} %")


def drt(
    spectrum: Impedance,
    regularisation: Optional[float] = None,
    per_decade: int = DRT_PER_DECADE,
    extend_decades: float = 1.0,
) -> DRTResult:
    """The distribution of relaxation times, by Tikhonov regularisation.

    An impedance spectrum is a sum of RC processes:

    .. math::

        Z(\\omega) = R_s + \\int \\frac{\\gamma(\\tau)}
        {1 + i\\omega\\tau}\\,\\mathrm{d}\\ln\\tau

    and the DRT is that γ(τ). It is worth having because it separates
    processes that a Nyquist plot merges: two semicircles a factor of
    three apart in time constant look like one flattened arc, and a
    circuit fit will happily describe that arc with a constant-phase
    element and call the question answered.

    **The inversion is ill-posed.** Without regularisation the solution
    oscillates wildly between positive and negative values that fit the
    data perfectly and mean nothing. The regularisation parameter is
    therefore not a detail of the implementation, it is a choice about
    how much structure to believe, and it is reported with the result.
    When it is not given, it is chosen by the L-curve — the corner of the
    trade-off between fitting the data and keeping the solution smooth —
    which is the standard automatic choice and still only a choice.

    γ is constrained to be **non-negative**, because a negative relaxation
    strength is not a physical process. That constraint does most of the
    work the regularisation would otherwise have to do.
    """
    frequency = np.asarray(spectrum.frequency, dtype=float)
    z = np.asarray(spectrum.z, dtype=complex)
    if frequency.size < 8:
        raise CircuitError("hacen falta al menos ocho frecuencias para una DRT")

    omega = 2.0 * np.pi * frequency
    low = 1.0 / (omega.max() * 10.0 ** extend_decades)
    high = 10.0 ** extend_decades / omega.min()
    count = max(8, int(per_decade * math.log10(high / low)))
    tau = np.logspace(math.log10(low), math.log10(high), count)

    # The design matrix: one column per time constant, real and imaginary
    # parts stacked, plus a column of ones for the ohmic resistance.
    kernel = 1.0 / (1.0 + 1j * np.outer(omega, tau))
    design = np.concatenate([kernel.real, kernel.imag], axis=0)
    ohmic_column = np.concatenate([np.ones(omega.size), np.zeros(omega.size)])
    design = np.concatenate([design, ohmic_column[:, None]], axis=1)
    target = np.concatenate([z.real, z.imag])

    # First differences: the penalty is on the ROUGHNESS of γ, not on its
    # size. Penalising the size shrinks every process towards zero and
    # biases the resistances low; penalising the roughness merges
    # neighbouring time constants, which is the honest failure mode —
    # it says "these two are not resolved" instead of "both are smaller
    # than they are".
    difference = np.zeros((count - 1, count + 1))
    for index in range(count - 1):
        difference[index, index] = -1.0
        difference[index, index + 1] = 1.0

    if regularisation is None:
        regularisation, curve = _l_curve(design, target, difference)
    else:
        curve = []

    solution = _nnls_tikhonov(design, target, difference, regularisation)
    # The solved coefficients are per-bin WEIGHTS: the model is
    # Z = R_s + Σ g_k/(1+iωτ_k), so g_k = γ(τ_k)·Δlnτ. Dividing by the bin
    # width turns them into the density γ that the documentation promises,
    # the one whose integral over a peak is that process's resistance.
    # Skipping this step is a silent factor of Δlnτ — 0.23 on the default
    # grid, which turned a 20 Ω process into 4.7 Ω.
    log_step = float(math.log(tau[1] / tau[0])) if tau.size > 1 else 1.0
    gamma = solution[:-1] / max(log_step, 1e-12)
    ohmic = float(solution[-1])

    fitted = design @ solution
    residual = float(np.linalg.norm(fitted - target)
                     / max(np.linalg.norm(target), 1e-30))

    peaks, resistances = _drt_peaks(tau, gamma)
    warnings = [
        "la inversión es mal condicionada: la regularización no es un "
        f"detalle de implementación sino una elección sobre cuánta "
        f"estructura creerse. Aquí λ = {regularisation:.3g}"
    ]
    if regularisation is not None and not curve:
        warnings.append("λ lo has fijado tú; con otro valor salen otros picos")
    if len(peaks) >= 2:
        ratios = [peaks[index + 1] / peaks[index] for index in range(len(peaks) - 1)]
        if min(ratios) < 3.0:
            warnings.append(
                "hay picos separados por menos de un factor de tres en τ: a "
                "esa distancia la separación depende de λ más que de los "
                "datos, y un circuito equivalente los describiría igual de "
                "bien con un solo elemento de fase constante"
            )
    if residual > 0.05:
        warnings.append(
            f"el residuo es del {100 * residual:.1f} %: la distribución no "
            "reproduce el espectro, así que sobra regularización o los datos "
            "no son los de un sistema lineal invariante"
        )
    if gamma.size and gamma[0] > 0.1 * float(np.max(gamma)):
        warnings.append(
            "hay peso apreciable en el extremo rápido de la rejilla: la "
            "medida no llega a frecuencia bastante alta para ver ese proceso "
            "entero y lo que se ve es su cola"
        )
    if gamma.size and gamma[-1] > 0.1 * float(np.max(gamma)):
        warnings.append(
            "hay peso apreciable en el extremo lento: falta frecuencia baja, "
            "y la difusión suele estar ahí"
        )
    return DRTResult(
        tau_s=tau, gamma=gamma, regularisation=float(regularisation),
        fitted=fitted, residual=residual, peaks_s=peaks,
        peak_resistances=resistances, ohmic=ohmic, warnings=warnings,
    )


def _nnls_tikhonov(design, target, difference, regularisation):
    """Non-negative least squares with a roughness penalty.

    Solved by stacking the penalty onto the design matrix, which turns the
    regularised problem into an ordinary non-negative least squares — the
    same trick that makes ridge regression an ordinary regression.
    """
    from scipy.optimize import nnls

    penalty = math.sqrt(max(regularisation, 0.0)) * difference
    stacked = np.concatenate([design, penalty], axis=0)
    extended = np.concatenate([target, np.zeros(penalty.shape[0])])
    solution, _ = nnls(stacked, extended, maxiter=50 * stacked.shape[1])
    return solution


def _l_curve(design, target, difference, count: int = 24):
    """The corner of the fit-against-smoothness trade-off.

    The corner is taken as the point of greatest curvature of the
    log-log curve of residual against solution roughness. It is the
    standard automatic choice, and it is still a choice: a decade either
    side of it produces a defensible DRT with a different number of peaks.
    """
    candidates = np.logspace(-6, 2, count)
    residuals, roughness = [], []
    for value in candidates:
        solution = _nnls_tikhonov(design, target, difference, value)
        residuals.append(float(np.linalg.norm(design @ solution - target)))
        roughness.append(float(np.linalg.norm(difference @ solution)))
    x = np.log10(np.maximum(residuals, 1e-30))
    y = np.log10(np.maximum(roughness, 1e-30))
    if x.size < 5:
        return float(candidates[len(candidates) // 2]), []
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum(
        (dx ** 2 + dy ** 2) ** 1.5, 1e-30)
    curvature[:2] = 0.0
    curvature[-2:] = 0.0
    return float(candidates[int(np.argmax(curvature))]), list(zip(x, y))


def _drt_peaks(tau, gamma, min_fraction: float = 0.05):
    """Peaks of γ(τ) and the resistance under each.

    The resistance is the integral of γ over the peak in ln τ, and it is
    integrated over the whole **basin** — from the minimum on one side to
    the minimum on the other — not over the part of the peak above some
    fraction of its height. Truncating at a tenth of the peak looks
    tidier and throws away most of the area: two RC processes of 10 and
    25 Ω came back as 1.2 and 2.9.
    """
    from ..core.compat import trapezoid

    gamma = np.asarray(gamma, dtype=float)
    if gamma.size < 3 or float(np.max(gamma)) <= 0:
        return [], []
    threshold = min_fraction * float(np.max(gamma))
    log_tau = np.log(np.asarray(tau, dtype=float))

    maxima = [
        index for index in range(1, gamma.size - 1)
        if gamma[index] > threshold
        and gamma[index] >= gamma[index - 1]
        and gamma[index] >= gamma[index + 1]
    ]
    if not maxima:
        return [], []

    # Basin boundaries: the lowest point between neighbouring peaks.
    bounds = [0]
    for first, second in zip(maxima, maxima[1:]):
        bounds.append(first + int(np.argmin(gamma[first:second + 1])))
    bounds.append(gamma.size - 1)

    peaks, resistances = [], []
    for index, position in enumerate(maxima):
        left, right = bounds[index], bounds[index + 1]
        if right <= left:
            continue
        peaks.append(float(tau[position]))
        resistances.append(float(trapezoid(gamma[left:right + 1],
                                           log_tau[left:right + 1])))
    return peaks, resistances
