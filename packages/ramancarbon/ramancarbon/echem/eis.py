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
from typing import Callable, Optional, Sequence

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

@dataclass(frozen=True)
class CircuitTemplate:
    """A named circuit, with the measurement it is for.

    The name and the string are not enough. Every equivalent circuit fits
    *something*, and the ones below differ less in how well they fit than
    in what their parameters mean afterwards: a CPE where a capacitor
    belongs still fits, and then ``Q1.Q`` is not a capacitance and cannot
    be quoted as one. So each carries the case it is for and the trap it
    has, and the report prints them next to the numbers.
    """

    name: str
    circuit: str
    use: str
    """When this circuit is the right one, in one line."""
    caution: str = ""
    """What it will do wrong if it is not."""
    origin: str = "programa"


#: Circuits offered by name.
#:
#: Ordered roughly by complexity, because that is the order to try them
#: in: a circuit with more elements always fits at least as well, so
#: adding one and seeing the residual drop proves nothing on its own.
#: What decides is whether the extra element's parameters come out with
#: usable uncertainties and a physical value.
TEMPLATES: tuple[CircuitTemplate, ...] = (
    CircuitTemplate(
        "resistencia", "R0",
        "Sólo la resistencia de la disolución; para comprobar la celda.",
        "No hay electrodo en este modelo. Si ajusta bien, no mediste nada.",
    ),
    CircuitTemplate(
        "edlc_ideal", "R0-C1",
        "Condensador de doble capa ideal: R en serie y un condensador.",
        "Ningún electrodo poroso real lo cumple; la dispersión de "
        "frecuencias pide un CPE.",
    ),
    CircuitTemplate(
        "edlc_cpe", "R0-Q1",
        "Doble capa con dispersión de frecuencias, sin transferencia de "
        "carga. El punto de partida de un supercondensador de carbono.",
        "Q NO son faradios. Sin una R en paralelo no hay conversión de "
        "Brug posible.",
    ),
    CircuitTemplate(
        "randles", "R0-(R1|C1)",
        "Un semicírculo limpio: transferencia de carga con doble capa "
        "ideal.",
        "Un semicírculo achatado no es éste: es randles_cpe.",
    ),
    CircuitTemplate(
        "randles_cpe", "R0-(R1|Q1)",
        "Un semicírculo achatado. El caso corriente de un electrodo "
        "rugoso o poroso.",
        "El achatamiento puede ser rugosidad o puede ser una distribución "
        "de constantes de tiempo; n no dice cuál.",
    ),
    CircuitTemplate(
        "randles_warburg", "R0-(R1-W1|Q1)",
        "Semicírculo más cola de difusión semiinfinita a 45°.",
        "Si la cola se dobla hacia la vertical o hacia el eje real, la "
        "difusión NO es semiinfinita: usa randles_ws o randles_wo.",
    ),
    CircuitTemplate(
        "randles_ws", "R0-(R1-Ws1|Q1)",
        "Difusión de longitud finita con el otro extremo permeable "
        "(transmisiva): capa de difusión con reserva al otro lado.",
        "Termina en una resistencia finita. Si tus datos acaban verticales, "
        "el correcto es el reflectante.",
    ),
    CircuitTemplate(
        "randles_wo", "R0-(R1-Wo1|Q1)",
        "Difusión de longitud finita con el extremo cerrado "
        "(reflectante): el interior de una partícula.",
        "Termina capacitivo, en vertical. Si tus datos acaban en el eje "
        "real, el correcto es el transmisivo.",
    ),
    CircuitTemplate(
        "supercondensador", "R0-(R1|Q1)-Q2",
        "Transferencia de carga de la interfase más la cola de "
        "almacenamiento del dispositivo.",
        "La Q2 es del dispositivo entero, no del material.",
    ),
    CircuitTemplate(
        "poroso", "R0-T1",
        "Línea de transmisión de De Levie/Bisquert para un poro "
        "bloqueante: la recta a 45° que se dobla a vertical. El modelo de "
        "un carbono poroso de verdad.",
        "T1.Ri es la resistencia IÓNICA dentro del poro, no la de la "
        "disolución: es la que limita la potencia.",
    ),
    CircuitTemplate(
        "poroso_con_transferencia", "R0-(R1|Q1)-T1",
        "Electrodo poroso con una reacción de superficie además de la "
        "doble capa.",
        "Con tres tiempos característicos parecidos, R1 y T1 se "
        "correlacionan: mira las incertidumbres antes de creerte el "
        "reparto.",
    ),
    CircuitTemplate(
        "recubrimiento", "R0-(Q1|R1-(R2|Q2))",
        "Una película sobre el electrodo: poros de la capa en paralelo "
        "con su capacidad, y la interfase debajo. El caso de un FeSe "
        "recubierto de carbono.",
        "Los dos arcos se separan sólo si sus constantes de tiempo "
        "difieren en más de una década.",
    ),
    CircuitTemplate(
        "dos_semicirculos", "R0-(R1|Q1)-(R2|Q2)",
        "Dos procesos en serie con constantes de tiempo distintas.",
        "Dos CPE en serie ajustan casi cualquier arco achatado. Que baje "
        "el residuo no prueba que haya dos procesos.",
    ),
    CircuitTemplate(
        "tres_semicirculos", "R0-(R1|Q1)-(R2|Q2)-(R3|Q3)",
        "Tres procesos en serie: película, transferencia y difusión "
        "lenta.",
        "Nueve parámetros libres. Sin una DRT que enseñe tres picos "
        "separados, esto es dibujar.",
    ),
    CircuitTemplate(
        "bateria", "R0-(R1|Q1)-(R2|Q2)-Wo1",
        "Electrodo de batería: SEI, transferencia de carga y difusión en "
        "el sólido.",
        "La cola vertical puede ser difusión o puede ser capacidad de la "
        "celda; Wo1.tau sólo significa algo si mediste hasta bien por "
        "debajo de 1/tau.",
    ),
    CircuitTemplate(
        "bateria_ws", "R0-(R1|Q1)-(R2|Q2)-Ws1",
        "Como el anterior pero con la difusión permeable: un electrodo "
        "en película delgada sobre un colector.",
        "Igual que arriba, con el extremo opuesto.",
    ),
    CircuitTemplate(
        "gerischer", "R0-(G1|Q1)",
        "Reacción química acoplada a la difusión (Gerischer): un arco "
        "deprimido que NO es un CPE. Típico de electrocatálisis y de "
        "conductores mixtos.",
        "Un Gerischer y un CPE se parecen mucho en el Nyquist. Lo que los "
        "separa es el diagrama de Bode a alta frecuencia.",
    ),
    CircuitTemplate(
        # Los paréntesis de (R2-L1) NO son decorativos. En esta notación
        # «|» liga MÁS fuerte que «-», así que «R1|Q1|R2-L1» se lee como
        # «(R1|Q1|R2)-L1»: una inductancia en SERIE. Medido sobre esta
        # misma plantilla antes de arreglarla, con R0=5, R1=40, Q1=2e-4
        # n=0.9, R2=25 y L1=8 H: cero puntos de lazo inductivo y un Z''
        # máximo de +8·10⁵ a 16 kHz, que es la reactancia del inductor en
        # serie disparándose arriba. Con los paréntesis, 252 puntos de
        # lazo y Z'' máximo de +11 a 1.2 Hz, que es el lazo de libro.
        "adsorcion_inductiva", "R0-(R1|Q1|(R2-L1))",
        "El lazo inductivo a baja frecuencia de un intermedio adsorbido: "
        "HER y OER lo dan casi siempre.",
        "Un lazo inductivo también lo produce la deriva del electrodo "
        "durante la medida. Repite el barrido antes de interpretarlo. Y "
        "fíjate en los paréntesis de (R2-L1): sin ellos «|» liga antes "
        "que «-» y lo que queda es una inductancia en serie, que sube "
        "Z″ arriba en vez de dar un lazo abajo. Dibuja el circuito.",
    ),
    CircuitTemplate(
        "con_inductancia", "L0-R0-(R1|Q1)",
        "Con la inductancia de los cables, que domina por encima de unos "
        "kHz.",
        "Si L0 sale mayor que 1 µH no es la muestra, es el montaje: "
        "acorta y trenza los cables.",
    ),
    CircuitTemplate(
        "zarc", "R0-Za1",
        "Un arco deprimido escrito en (R, τ, γ) en vez de en (R, Q, n). "
        "Es el MISMO circuito que R0-(R1|Q1) y devuelve otros números: la "
        "capacitancia efectiva es τ/R exacta, sin Brug y sin conversión.",
        "No es un modelo físico distinto. Si R|Q describía mal el "
        "electrodo, esto también.",
    ),
    CircuitTemplate(
        "dos_zarc", "R0-Za1-Za2",
        "Dos procesos con constantes de tiempo distintas —película y "
        "transferencia de carga, o dos poblaciones de poro—, uno por "
        "arco. Es el circuito al que traduce un DRT de dos picos.",
        "Dos τ a menos de un factor de tres son UN proceso: a esa "
        "distancia el reparto lo decide el punto de partida y no los "
        "datos. Mira la DRT antes de usarlo.",
    ),
    CircuitTemplate(
        "cables_no_ideales", "La0-R0-(R1|Q1)",
        "Como «con_inductancia» pero con la inductancia del montaje "
        "no ideal, (jωL)^α. El caso de un cableado largo o con "
        "electrodo de referencia.",
        "Si α baja de 0.8 esto ya no está describiendo una inductancia; "
        "mira qué más hay en el extremo alto antes de citar L.",
    ),
    CircuitTemplate(
        "supercondensador_completo", "L0-R0-(R1|Q1)-T1",
        "Un supercondensador de carbono medido de MHz a mHz, con los "
        "cables, la interfase y el poro.",
        "Seis parámetros y tres regiones de frecuencia. Si no mediste las "
        "tres, fija los elementos de las que falten.",
    ),
)

#: Circuits offered by name. Kept as a plain mapping because that is what
#: the rest of the package and the tests use; :data:`TEMPLATES` carries
#: everything else.
LIBRARY: dict[str, str] = {t.name: t.circuit for t in TEMPLATES}


class CircuitError(ValueError):
    """Raised when a circuit string cannot be parsed or evaluated."""


def user_circuit_file():
    """Where the user's own circuits live.

    Outside the package, next to the preferences and the user's Raman
    phases, for the same reason: a package upgrade replaces what is
    inside the package, and a circuit somebody worked out for their own
    cell would go with it.
    """
    from ..core.history import config_directory

    return config_directory() / "circuitos_usuario.json"


def user_circuits() -> dict[str, str]:
    """The user's own circuits, by name.

    An unreadable file is reported and ignored, never fatal: a stray
    comma in hand-written JSON cannot be allowed to take the bundled
    circuits down with it.
    """
    import json

    path = user_circuit_file()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(name): str(circuit)
        for name, circuit in payload.items()
        if isinstance(circuit, str) and circuit.strip()
    }


def save_user_circuit(name: str, circuit: str, use: str = "") -> CircuitTemplate:
    """Store a circuit under a name, after checking that it parses.

    Checked before it is written, not when it is next used: a circuit
    saved with a typo comes back as a parse error in the middle of an
    analysis, a week later, with no clue as to which of the saved ones
    is broken.
    """
    import json

    label = name.strip()
    if not label:
        raise CircuitError("un circuito guardado necesita un nombre")
    parse_circuit(LIBRARY.get(circuit, circuit))
    stored = user_circuits()
    stored[label] = circuit.strip()
    path = user_circuit_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return CircuitTemplate(label, circuit.strip(), use or "Circuito propio.",
                           origin="usuario")


def delete_user_circuit(name: str) -> bool:
    """Forget a user circuit. Bundled ones cannot be deleted.

    Same rule as the Raman phases: deleting a bundled entry would mean
    writing to a file the next upgrade replaces, so the deletion would
    silently come back. Shadow it with one of your own instead.
    """
    import json

    stored = user_circuits()
    if name not in stored:
        return False
    del stored[name]
    path = user_circuit_file()
    path.write_text(json.dumps(stored, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return True


def circuit_templates() -> list[CircuitTemplate]:
    """Every circuit on offer, bundled and user, each saying where it is
    from. A user circuit with the name of a bundled one replaces it."""
    by_name = {t.name: t for t in TEMPLATES}
    for name, circuit in user_circuits().items():
        by_name[name] = CircuitTemplate(
            name, circuit, "Circuito propio.", origin="usuario")
    return sorted(by_name.values(), key=lambda t: (t.origin != "programa", t.name))


def circuit_for(name: str) -> str:
    """The circuit string for a name, or the name itself if it is one."""
    return LIBRARY.get(name, user_circuits().get(name, name))


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


def _z_g(omega: np.ndarray, z0: float, k: float) -> np.ndarray:
    """Gerischer: a chemical reaction coupled to diffusion.

    ``Z = Z₀ / √(k + jω)``. It looks like a depressed arc and is
    routinely fitted with a CPE instead, which fits about as well and
    means something else entirely: the CPE's ``n`` is a dispersion of
    time constants, while ``k`` here is a rate constant with units of
    s⁻¹ that can be compared against a kinetic measurement. What
    separates them is the high-frequency end — a Gerischer goes to 45°
    like a Warburg, a CPE to a constant phase of 90n degrees.
    """
    return z0 / np.sqrt(max(k, 1e-12) + 1j * omega)


def _z_za(omega: np.ndarray, r: float, tau: float, gamma: float) -> np.ndarray:
    """ZARC (Cole-Cole) arc: ``Z = R / (1 + (jωτ)^γ)``.

    **The same shape as R|Q, written in the parameters the measurement
    speaks in**, and that is the whole reason it exists. A ZARC and a
    resistance in parallel with a CPE are algebraically identical, with
    ``Q = τ^γ / R``; there is no impedance a ZARC can make that R|Q
    cannot. What changes is what the fit hands back.

    Three things follow from the parameterisation, and all three matter:

    * **The effective capacitance is exact and free.** ``C = τ / R``, by
      definition, with no Brug relation and no conversion. The CPE's
      ``Q`` is not a capacitance and turning it into one needs the
      resistance it is in parallel with plus an assumption about what
      causes the dispersion; the ZARC never leaves the units.
    * **It is what a DRT peak already measures.** A peak of γ(τ) gives a
      time constant and, as the area under it, a resistance — which are
      two of these three parameters. Building the circuit from the DRT is
      then a translation rather than a guess, which is what
      :func:`circuit_from_drt` does.
    * **It is better conditioned.** In R|Q the two parameters are tied
      together through ``Q = τ^γ/R``, so R and Q come back strongly
      correlated and their individual uncertainties mean less than the
      table suggests. R and τ are close to independent.

    What it is NOT is a different physical model. If R|Q was the wrong
    description of the electrode, so is this.
    """
    return max(r, 1e-12) / (1.0 + (1j * omega * max(tau, 1e-18)) ** gamma)


def _z_la(omega: np.ndarray, inductance: float, alpha: float) -> np.ndarray:
    """Non-ideal inductance, ``Z = (jωL)^α``.

    Cable and current-collector inductance is what lifts Z″ positive at
    the top of a spectrum, and a pure inductor describes it only when the
    wiring is a single clean loop. With a cell in a glovebox, a long
    harness and a reference electrode, the high-frequency end comes back
    at a phase that is neither 90° nor constant, and an ideal ``L``
    forced through it shifts the series resistance to absorb the
    mismatch — which lands in R_u and from there in every iR correction
    made with it.

    α is fitted and bounded near 1: below about 0.8 this is no longer
    describing inductance and the right question is what else is up
    there.
    """
    return (1j * omega * max(inductance, 1e-18)) ** alpha


def _z_t(omega: np.ndarray, r_ion: float, q: float, n: float) -> np.ndarray:
    """Blocking porous electrode: the De Levie / Bisquert transmission line.

    ``Z = √(R_ion · Z_c) · coth √(R_ion / Z_c)``, with the pore wall a
    CPE, ``Z_c = 1/(Q (jω)^n)``.

    This is what a porous carbon actually does, and no combination of
    lumped elements reproduces it: at high frequency the ion has only
    entered the mouth of the pore, so the response is the 45° line of a
    distributed RC; at low frequency the whole pore is charged and it
    turns vertical, offset from the series resistance by ``R_ion/3``.
    That factor of three is the point — fitting the 45° region with a
    Warburg instead gives a "diffusion coefficient" for an ion that is
    not diffusing anywhere, and fitting the whole thing with R-(R|Q)
    buries the ionic resistance, which is the quantity that limits the
    device's power.
    """
    z_wall = 1.0 / (max(q, 1e-18) * (1j * omega) ** n)
    ratio = np.sqrt(max(r_ion, 1e-12) / z_wall)
    # coth saturates at 1 for large argument; clip so the high-frequency
    # end does not overflow to nan on a wide spectrum.
    return np.sqrt(max(r_ion, 1e-12) * z_wall) / np.tanh(np.clip(ratio, -30.0, 30.0))


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
    "G": (_z_g, ("Z0", "k"), (10.0, 1.0), ((1e-6, 1e-6), (1e9, 1e6)), (True, True)),
    "T": (
        _z_t,
        ("Ri", "Q", "n"),
        (10.0, 1e-3, 0.9),
        ((1e-6, 1e-15, 0.3), (1e9, 10.0, 1.0)),
        (True, True, False),
    ),
    "Za": (
        _z_za,
        ("R", "tau", "gamma"),
        (10.0, 1e-3, 0.85),
        ((1e-6, 1e-9, 0.3), (1e9, 1e6, 1.0)),
        (True, True, False),
    ),
    "La": (
        _z_la,
        ("L", "alpha"),
        (1e-6, 1.0),
        ((1e-12, 0.5), (1.0, 1.0)),
        (True, False),
    ),
}

# Two-letter symbols FIRST: the alternation is ordered, so a leading
# ``[RCLQWGT]`` would match the ``L`` of ``La`` and leave an ``a`` that
# parses as nothing.
_TOKEN = re.compile(r"(Ws|Wo|Za|La|[RCLQWGT])(\d*)")


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


#: How each element is drawn in a schematic, and what it is.
_GLYPHS: dict[str, tuple[str, str]] = {
    "R": ("─[ {name} ]─", "resistencia"),
    "C": ("─┤├─ {name} ", "condensador"),
    "L": ("─(((─ {name} ", "inductancia"),
    "La": ("─(((─ {name} ", "inductancia no ideal (jωL)^α"),
    "Q": ("─[/{name}/]─", "CPE"),
    "Za": ("─(∩ {name} ∩)─", "ZARC: R‖CPE en (R, τ, γ)"),
    "W": ("─[W {name} ]─", "Warburg semiinfinito"),
    "Ws": ("─[W {name} ]─", "Warburg finito transmisivo"),
    "Wo": ("─[W {name} ]─", "Warburg finito reflectante"),
    "G": ("─[G {name} ]─", "Gerischer"),
    "T": ("─[T {name} ]─", "electrodo poroso (línea de transmisión)"),
}


def draw_circuit(circuit: str, values: Optional[dict[str, float]] = None) -> str:
    """The circuit as a schematic, with a legend of what each element is.

    A circuit string is written left to right and read as a shape, and
    the two stop agreeing as soon as there is nesting:
    ``L0-R0-(R1|Q1|R2-L1)-T1`` is six elements in three levels, and
    whether that last ``L1`` is inside the parallel branch or after it is
    the difference between an adsorbed intermediate and a wiring
    artefact. Typing it and seeing it are not the same check.

    Parallel branches are stacked vertically between ``┬`` and ``┴``
    rails, which is how a circuit diagram is drawn and is why the shape
    is readable at a glance when the string is not.

    With ``values``, each element carries its fitted number, so the
    drawing doubles as the parameter table in the order the current
    actually goes through them.
    """
    tree = parse_circuit(circuit)

    def render(node: _Node) -> list[str]:
        if node.kind == "element":
            element = node.children[0]
            shape, _ = _GLYPHS.get(element.symbol, ("─[ {name} ]─", ""))
            return [shape.format(name=element.name)]
        if node.kind == "series":
            blocks = [render(child) for child in node.children]
            height = max(len(block) for block in blocks)
            padded = []
            for block in blocks:
                width = max(len(line) for line in block)
                middle = len(block) // 2
                top = (height - len(block)) // 2
                rows = [" " * width] * top + [line.ljust(width) for line in block]
                rows += [" " * width] * (height - len(rows))
                # The through-line of a series block is the row the
                # connectors meet at, which is the middle of the child.
                padded.append((rows, top + middle))
            spine = padded[0][1]
            out = []
            for index in range(height):
                out.append("".join(
                    rows[index] if index < len(rows) else ""
                    for rows, _ in padded))
            del spine
            return out
        # Parallel: stack the branches and bracket them with rails.
        blocks = [render(child) for child in node.children]
        width = max(len(line) for block in blocks for line in block)
        lines: list[str] = []
        for position, block in enumerate(blocks):
            for row, line in enumerate(block):
                centre = len(block) // 2
                left = "┬" if (position == 0 and row == centre) else (
                    "┴" if (position == len(blocks) - 1 and row == centre)
                    else "│")
                right = left
                lines.append(f"{left}{line.center(width)}{right}")
        return lines

    drawing = render(tree)
    # The rails have to be one continuous pair of columns, so a parallel
    # block taller than one line gets its verticals filled in.
    lines = ["  " + line for line in drawing]
    used = {e.symbol for e in tree.elements()}
    legend = ["", "  " + "─" * 44]
    for symbol in sorted(used):
        _, what = _GLYPHS.get(symbol, ("", ""))
        names = [e.name for e in tree.elements() if e.symbol == symbol]
        legend.append(f"  {', '.join(names):<10s} {what}")
    if values:
        legend.append("  " + "─" * 44)
        for element in tree.elements():
            for label, parameter in zip(element.labels, element.parameters):
                if label in values:
                    unit = _unit(element.symbol, parameter)
                    legend.append(f"  {label:<12s} {values[label]:12.6g} {unit}")
    return "\n".join([f"  {circuit}", ""] + lines + legend)


def parse_circuit(text: str) -> _Node:
    """Parse a circuit string into a tree.

    ``-`` is series, ``|`` is parallel, parentheses group, and an element
    is a symbol plus an optional index: ``R0-(R1|Q1)-Wo1``.

    **``|`` binds tighter than ``-``**, so ``R1|Q1|R2-L1`` is
    ``(R1|Q1|R2)-L1`` and not ``R1|Q1|(R2-L1)``. That is the usual
    precedence and it is also a trap, because the second is what somebody
    writing an adsorption loop means: the bundled template for one
    shipped without the parentheses and produced a series inductor —
    nothing inductive at low frequency at all. Parenthesise a branch
    that is itself a series, and use :func:`draw_circuit` to check, since
    the string and the shape stop agreeing exactly where it matters.
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
    fixed: list[str] = field(default_factory=list)
    """Parameters held rather than refined. Named because fixing one
    shrinks every other uncertainty in the table, so a reader who does
    not know which were held cannot read the table."""

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
                if label in self.fixed:
                    text += "  (FIJADO)"
                elif error is not None and value != 0:
                    text += f"  (± {100 * error / abs(value):.1f} %)"
                lines.append(text)
        if self.fixed:
            lines.append("")
            lines.append(
                "  Parámetros fijados: " + ", ".join(self.fixed) + ". Las "
                "incertidumbres de los demás salen más pequeñas por eso, y "
                "un valor fijado que esté mal se reparte entre ellos sin que "
                "el ajuste empeore."
            )
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
        ("G", "Z0"): "Ω·s^-½", ("G", "k"): "s⁻¹",
        ("T", "Ri"): "Ω", ("T", "Q"): "S·sⁿ", ("T", "n"): "",
        ("Za", "R"): "Ω", ("Za", "tau"): "s", ("Za", "gamma"): "",
        ("La", "L"): "H", ("La", "alpha"): "",
    }
    return units.get((symbol, parameter), "")


@dataclass
class EffectiveCapacitance:
    """A CPE turned into farads, with the arithmetic that did it.

    The number alone is not reportable. Two accepted relations give
    different answers from the same CPE, they rest on different physics,
    and the literature applies them interchangeably — which is exactly
    the confusion this class exists to stop. So it carries the method,
    the inputs under both naming conventions, and every step.
    """

    farads: float
    method: str
    """``"Brug"`` or ``"Hsu-Mansfeld"``."""
    t: float
    """The CPE's magnitude. ``T`` in ZView and in most commercial
    software, ``Q`` in the electrochemical literature and in this
    package's element table. The same number."""
    p: float
    """The CPE's exponent. ``P`` in ZView, ``n`` or ``α`` elsewhere."""
    resistances: dict[str, float] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    caution: str = ""

    def describe(self) -> str:
        lines = [
            f"CAPACITANCIA EFECTIVA DEL CPE — {self.method}",
            "─" * 46,
            f"  T (= Q) = {self.t:.6g} S·s^P",
            f"  P (= n) = {self.p:.4f}",
        ]
        for label, value in self.resistances.items():
            lines.append(f"  {label} = {value:.6g} Ω")
        lines.append("")
        lines.extend(f"  {step}" for step in self.steps)
        lines.append("")
        lines.append(f"  C = {self.farads:.6g} F")
        if self.caution:
            lines.append("")
            lines.append(f"  ⚠ {self.caution}")
        return "\n".join(lines)


def _check_cpe(t: float, p: float) -> None:
    if not 0.0 < p <= 1.0 or t <= 0.0:
        raise CircuitError(
            f"parámetros fuera de rango para convertir un CPE: T = {t:.6g}, "
            f"P = {p:.4f}. P tiene que estar en (0, 1] y T ser positivo"
        )


def hsu_mansfeld_capacitance(
    t: float, p: float, resistance: float
) -> EffectiveCapacitance:
    """Effective capacitance from the arc's own apex.

    ``C = (T · R^(1−P))^(1/P) = T^(1/P) · R^((1−P)/P)``, where ``R`` is
    the resistance the CPE is in parallel with.

    **It is the characteristic capacitance, derived and not quoted.**
    For ``R`` in parallel with a CPE, ``Z = R / (1 + R T (jω)^P)``, whose
    apex sits where ``|R T (jω)^P| = 1``, so ``ω_peak = (R T)^(−1/P)``.
    Defining ``C`` by the ideal relation ``ω_peak = 1/(R C)`` gives
    exactly the expression above. That derivation is also the check: a
    ZARC written as ``R/(1 + (jωτ)^γ)`` has ``T = τ^γ/R``, and
    substituting gives ``C = τ/R`` — which is what ``τ`` and ``R`` mean.

    Hsu and Mansfeld's relation describes a distribution of time
    constants **normal** to the surface (through the thickness of a
    film). For a distribution ALONG the surface, which is what roughness
    and non-uniform current give, the right one is
    :func:`brug_capacitance`, and the two do not agree.
    """
    _check_cpe(t, p)
    if resistance <= 0.0:
        raise CircuitError("hace falta una resistencia positiva en paralelo")
    farads = float(t ** (1.0 / p) * resistance ** ((1.0 - p) / p))
    return EffectiveCapacitance(
        farads=farads,
        method="Hsu-Mansfeld",
        t=t, p=p,
        resistances={"R (en paralelo con el CPE)": resistance},
        steps=[
            "C = (T · R^(1−P))^(1/P) = T^(1/P) · R^((1−P)/P)",
            f"T^(1/P) = {t:.6g}^(1/{p:.4f}) = {t ** (1.0 / p):.6g}",
            f"R^((1−P)/P) = {resistance:.6g}^({(1.0 - p) / p:.4f}) "
            f"= {resistance ** ((1.0 - p) / p):.6g}",
            f"C = {t ** (1.0 / p):.6g} × "
            f"{resistance ** ((1.0 - p) / p):.6g} = {farads:.6g} F",
        ],
        caution=(
            "vale para una distribución de constantes de tiempo NORMAL a la "
            "superficie (a través del espesor de una película). Si la "
            "dispersión viene de rugosidad o de corriente no uniforme, la "
            "distribución es a lo largo de la superficie y toca Brug"
        ),
    )


def brug_capacitance(
    t: float,
    p: float,
    r_series: float,
    r_charge_transfer: Optional[float] = None,
) -> EffectiveCapacitance:
    """Effective capacitance for a distribution ALONG the surface.

    Blocking electrode (no faradaic path)::

        C = T^(1/P) · R_s^((1−P)/P)

    Faradaic electrode::

        C = T^(1/P) · (1/R_s + 1/R_ct)^((P−1)/P)

    The second reduces to the first as ``R_ct → ∞``, which is what
    "blocking" means, so they are one relation.

    **The solution resistance is in it, and that is the whole
    difference.** Brug's derivation puts the dispersion of time constants
    along the electrode surface — roughness, non-uniform current
    distribution, a polycrystalline face — so the current reaching each
    patch has gone through the electrolyte, and ``R_s`` sets the coupling
    between patches. Hsu and Mansfeld's puts the dispersion through the
    film's thickness, where it does not. Using one where the other
    belongs is a documented and common error in this literature, so both
    are offered and each says what it assumes.

    Since ``R_s ∥ R_ct < R_ct``, Brug always returns a **smaller**
    capacitance than Hsu–Mansfeld on the same CPE.
    """
    _check_cpe(t, p)
    if r_series <= 0.0:
        raise CircuitError(
            "la relación de Brug necesita la resistencia en SERIE (la de la "
            "disolución), que es la que acopla los parches de la superficie. "
            "Sin ella, lo que se puede calcular es Hsu-Mansfeld"
        )
    resistances = {"R_s (disolución, en serie)": r_series}
    if r_charge_transfer is not None and r_charge_transfer > 0.0:
        conductance = 1.0 / r_series + 1.0 / r_charge_transfer
        resistances["R_ct (transferencia de carga)"] = r_charge_transfer
        effective = 1.0 / conductance
        kind = "faradaico"
        label = "(R_s ∥ R_ct)"
        steps = [
            "C = T^(1/P) · (1/R_s + 1/R_ct)^((P−1)/P)",
            f"1/R_s + 1/R_ct = 1/{r_series:.6g} + "
            f"1/{r_charge_transfer:.6g} = {conductance:.6g} S",
            f"  (equivale a R_s ∥ R_ct = {effective:.6g} Ω)",
        ]
    else:
        conductance = 1.0 / r_series
        effective = r_series
        kind = "bloqueante"
        label = "R_s"
        steps = [
            "C = T^(1/P) · R_s^((1−P)/P)      (electrodo bloqueante)",
        ]
    farads = float(t ** (1.0 / p) * conductance ** ((p - 1.0) / p))
    steps.extend([
        f"T^(1/P) = {t:.6g}^(1/{p:.4f}) = {t ** (1.0 / p):.6g}",
        f"{label}^((1−P)/P) = {effective:.6g}^({(1.0 - p) / p:.4f}) "
        f"= {effective ** ((1.0 - p) / p):.6g}",
        f"C = {farads:.6g} F",
    ])
    return EffectiveCapacitance(
        farads=farads,
        method=f"Brug ({kind})",
        t=t, p=p,
        resistances=resistances,
        steps=steps,
        caution=(
            "vale para una distribución de constantes de tiempo A LO LARGO "
            "de la superficie (rugosidad, corriente no uniforme). Si la "
            "dispersión es a través del espesor de una película, toca "
            "Hsu-Mansfeld. Los dos se aplican indistintamente en la "
            "literatura y no dan lo mismo"
        ),
    )


def cpe_to_capacitance(q: float, n: float, resistance: float) -> float:
    """Effective capacitance of a CPE in parallel with a resistance, in F.

    The Hsu–Mansfeld relation, ``C = (Q · R^(1−n))^(1/n)``. Kept as the
    short form; :func:`hsu_mansfeld_capacitance` returns the same number
    with its arithmetic, and :func:`brug_capacitance` is the other
    relation, which needs the solution resistance as well.

    **This function used to divide that expression by R once more**, and
    the docstring that described it said "Brug's relation" for a formula
    that is Hsu and Mansfeld's. The result was a capacitance too small by
    a factor of exactly R — on a 47 Ω arc, 1.0 µF reported for 49 µF —
    and it reached every effective capacitance the package produced. The
    test that covered it asserted only ``0 < C < Q``, which both the
    right and the wrong value satisfy.

    ``Q`` and ``n`` are ``T`` and ``P`` in ZView and in most commercial
    software. Same parameters, different letters.
    """
    return hsu_mansfeld_capacitance(q, n, resistance).farads


def uncompensated_resistance(spectrum: Impedance) -> tuple[float, str]:
    """The series resistance to use for iR correction, in Ω.

    Taken as the real part of the impedance where it crosses the real axis
    at high frequency — the point where the double layer is a short circuit
    and everything that is left is electrolyte, contacts and current
    collector. That is the resistance the potentiostat did not compensate,
    and it is the number that belongs in
    :attr:`~ramancarbon.echem.curve.Electrode.resistance_ohm`.

    The crossing is found by interpolating Z″ to zero between the two
    highest-frequency points that straddle it. When the spectrum never
    crosses — which happens when the measurement stopped below the
    crossing, or when stray inductance from the cables pushes Z″ positive
    across the whole top end — the highest-frequency real part is returned
    instead, with an explanation, because it is an upper bound and saying
    so is better than refusing outright.

    Returns
    -------
    tuple
        ``(resistance in Ω, how it was obtained)``. The second element goes
        in the report: an iR correction with an unexplained resistance is
        a free parameter.
    """
    # Stored descending in frequency, so index 0 is the fastest point.
    imaginary = spectrum.z.imag
    real = spectrum.z.real
    for index in range(spectrum.n - 1):
        first, second = imaginary[index], imaginary[index + 1]
        if first == 0.0:
            return float(real[index]), (
                f"Z″ = 0 exactamente a {spectrum.frequency[index]:.4g} Hz"
            )
        if first * second < 0.0:
            weight = abs(first) / (abs(first) + abs(second))
            value = float(real[index] + weight * (real[index + 1] - real[index]))
            return value, (
                "cruce con el eje real interpolado entre "
                f"{spectrum.frequency[index]:.4g} y "
                f"{spectrum.frequency[index + 1]:.4g} Hz"
            )
    return float(real[0]), (
        f"el espectro no cruza el eje real; se usa Z′ a la frecuencia más "
        f"alta medida ({spectrum.frequency[0]:.4g} Hz), que es una COTA "
        "SUPERIOR de R_u. Un electrodo puramente capacitivo no cruza nunca "
        "—ahí esta cota es buena— pero si Z″ es POSITIVO arriba, lo que hay "
        "es inductancia de los cables y entonces sobra resistencia: mide más "
        "alto o réstala"
    )


def with_resistance_from(electrode, spectrum: Impedance):
    """A copy of an electrode carrying the R_u measured on this cell.

    The intended use is the one people do by hand and get wrong: measure
    the impedance, read the high-frequency intercept, and apply it to the
    voltammogram and the charge–discharge curve of the **same cell**. It is
    the same cell that matters — a resistance from another day, another
    electrolyte level or another contact is a number with the right units
    and no meaning.
    """
    from dataclasses import replace

    value, how = uncompensated_resistance(spectrum)
    updated = replace(electrode, resistance_ohm=value)
    return updated, how


def fit_circuit(
    spectrum: Impedance,
    circuit: str = "randles_cpe",
    initial: Optional[dict[str, float]] = None,
    max_iterations: int = 4000,
    fixed: Optional[Sequence[str]] = None,
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
    fixed:
        Labels to HOLD at their starting values instead of refining.
        Fixing is not free and it is not neutral: the parameter stops
        contributing a degree of freedom, so every other uncertainty
        comes out smaller, and if the fixed value is wrong that error
        moves into its neighbours rather than showing up as a bad fit.
        It is the right move for a parameter the measurement cannot
        determine — the cable inductance when the sweep stopped at
        100 kHz, a Warburg on a spectrum that never reached the
        diffusion region — and the wrong one everywhere else. Every
        fixed parameter is named in the report for that reason.

    Returns
    -------
    CircuitFit
    """
    text = LIBRARY.get(circuit, user_circuits().get(circuit, circuit))
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

    held = set(fixed or ())
    unknown = held - set(labels)
    if unknown:
        raise CircuitError(
            f"no hay ningún parámetro llamado {', '.join(sorted(unknown))} en "
            f"{text!r}; los que hay son: {', '.join(labels)}"
        )
    free = np.array([label not in held for label in labels])
    if not free.any():
        raise CircuitError(
            "están fijados todos los parámetros: no queda nada que ajustar"
        )

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

    def expand(reduced: np.ndarray) -> np.ndarray:
        """Free parameters back into the full internal vector."""
        full = np.array(to_internal(start_array), dtype=float)
        full[free] = reduced
        return full

    def unpack(internal: np.ndarray) -> None:
        values = to_external(internal)
        position = 0
        for element in elements:
            count = len(element.parameters)
            element.values = [float(v) for v in values[position:position + count]]
            position += count

    def residual(reduced: np.ndarray) -> np.ndarray:
        internal = expand(reduced)
        unpack(internal)
        model = tree.impedance(omega)
        difference = (model - observed) * weight
        return np.concatenate([difference.real, difference.imag])

    internal_lower = to_internal(np.asarray(lower))
    internal_upper = to_internal(np.asarray(upper))

    def run(guess: np.ndarray):
        return least_squares(
            residual,
            np.clip(guess, internal_lower + 1e-9, internal_upper - 1e-9)[free],
            bounds=(internal_lower[free], internal_upper[free]),
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
    scatter = is_log & free
    for _ in range(3):
        perturbed = base.copy()
        perturbed[scatter] += rng.normal(0.0, 0.7, int(scatter.sum()))
        candidate = run(perturbed)
        if candidate.cost < outcome.cost * 0.999:
            outcome = candidate
    unpack(expand(outcome.x))
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
        fixed=sorted(held),
    )
    _errors(outcome, [label for label in labels if label not in held],
            is_log[free], fit, spectrum.n)
    _check_bounds(fit, labels, lower, upper)
    _circuit_warnings(fit, spectrum)
    return fit


#: Relative uncertainty above which a fitted parameter is not a
#: measurement.
#:
#: The package's own rule for whether an extra element earned its place,
#: made into a number: a circuit with more elements always fits at least
#: as well, so a lower residual proves nothing — what decides is whether
#: the new parameter comes back with a usable uncertainty. Fifty per cent
#: is generous, deliberately: it is the threshold for "this number means
#: nothing", not for "this number is good".
PARAMETER_DETERMINED = 0.5


@dataclass
class CircuitCandidate:
    """One circuit tried against one spectrum."""

    circuit: str
    name: str = ""
    fit: Optional[CircuitFit] = None
    concerns: list[str] = field(default_factory=list)
    """Why this circuit may not describe an electrode at all, from
    :func:`circuit_concerns`. Reported, not used to exclude: a candidate
    the user asked about is answered, not dropped."""
    free_parameters: int = 0
    determined: int = 0
    """How many free parameters came back with a usable uncertainty."""
    bic: float = float("nan")
    error: str = ""
    """Set when the circuit could not be fitted at all."""

    @property
    def undetermined(self) -> int:
        return max(self.free_parameters - self.determined, 0)


def circuit_concerns(circuit: str) -> list[str]:
    """Reasons a circuit may not be able to describe an electrode.

    Structural checks only — they look at the circuit, never at the data,
    so they cost nothing and can be run before any fit. Adapted from
    AutoEIS's post-filters, which exist there to prune a generated
    search space; here nothing is pruned, because a circuit somebody
    asked about deserves an answer rather than a silent disappearance.

    What is NOT checked: whether the circuit is appropriate for the
    material. No structural rule can tell that, and
    :class:`CircuitTemplate` carries it in prose instead.
    """
    concerns: list[str] = []
    try:
        elements = parse_circuit(circuit).elements()
    except CircuitError as exc:
        return [f"no se puede leer: {exc}"]
    symbols = [element.symbol for element in elements]

    # An arc needs a parallel branch OR an element that already is one.
    # Checking the string for a "|" alone was wrong on the elements this
    # module adds: a ZARC IS a resistance in parallel with a CPE written
    # as one symbol, and the distributed elements trace curves of their
    # own. The first version of this flagged `R0-Za1` — the circuit the
    # DRT translation produces — as unable to make a semicircle.
    CURVED = {"Za", "T", "G", "W", "Ws", "Wo"}
    if "|" not in circuit and not CURVED.intersection(symbols):
        concerns.append(
            "no tiene ninguna rama en PARALELO ni ningún elemento que sea "
            "un arco por sí mismo, así que no puede producir un "
            "semicírculo: una cadena de R, C, L y Q en serie sólo da "
            "rectas. Si el Nyquist tiene un arco, este circuito no es el "
            "que lo hace"
        )
    if "R" not in symbols:
        concerns.append(
            "no tiene ninguna resistencia, así que no puede reproducir el "
            "corte con el eje real a alta frecuencia"
        )
    if "C" in symbols:
        concerns.append(
            "lleva un condensador IDEAL. Ningún electrodo poroso real lo "
            "es: la dispersión de constantes de tiempo pide un CPE, y un "
            "condensador ahí ajusta peor y además desplaza las "
            "resistencias vecinas para compensar"
        )
    return concerns


@dataclass
class CircuitComparison:
    """Several circuits fitted to the same spectrum, and what separates them."""

    candidates: list[CircuitCandidate]
    points: int = 0

    def ranked(self) -> list[CircuitCandidate]:
        """Best first, by BIC among the ones that fitted."""
        fitted = [c for c in self.candidates if c.fit is not None]
        return sorted(fitted, key=lambda c: c.bic)

    def best(self) -> Optional[CircuitCandidate]:
        order = self.ranked()
        return order[0] if order else None

    def describe(self) -> str:
        lines = [
            "COMPARACIÓN DE CIRCUITOS",
            "─" * 46,
            "",
            "  circuito                    χ²        libres  determinados"
            "      BIC",
        ]
        for candidate in self.ranked():
            fit = candidate.fit
            assert fit is not None
            lines.append(
                f"  {candidate.circuit:<24s} {fit.chi_squared:10.4g} "
                f"{candidate.free_parameters:7d} {candidate.determined:12d} "
                f"{candidate.bic:10.1f}"
            )
        for candidate in self.candidates:
            if candidate.error:
                lines.append(f"  {candidate.circuit:<24s} — {candidate.error}")
        lines.append("")
        lines.append(
            "  El BIC penaliza parámetros, y hace falta porque un circuito "
            "con más\n  elementos ajusta SIEMPRE al menos igual de bien: "
            "que baje el χ² al\n  añadir uno no prueba nada por sí solo."
        )
        lines.append(
            "  Pero el BIC tampoco decide solo. La columna que decide es "
            "«determinados»:\n  un parámetro que vuelve con una "
            f"incertidumbre por encima del "
            f"{100 * PARAMETER_DETERMINED:.0f} %\n  no es una medida, y un "
            "circuito que gana por BIC con la mitad de sus\n  parámetros "
            "indeterminados ha ajustado la forma sin medir nada."
        )
        best = self.best()
        if best is not None:
            lines.append("")
            lines.append(f"  Mejor por BIC: {best.circuit}")
            if best.undetermined:
                lines.append(
                    f"  ⚠ pero {best.undetermined} de sus "
                    f"{best.free_parameters} parámetros libres no están "
                    "determinados. Mira si un circuito más simple de la "
                    "lista los tiene todos."
                )
        for candidate in self.candidates:
            for concern in candidate.concerns:
                lines.append(f"  ⚠ {candidate.circuit}: {concern}")
        lines.append("")
        lines.append(
            "  Y nada de esto dice si el circuito es el APROPIADO para el "
            "material.\n  Ninguna regla estructural puede: un CPE donde va "
            "un condensador sigue\n  ajustando, y entonces su Q no son "
            "faradios."
        )
        return "\n".join(lines)


def compare_circuits(
    spectrum: Impedance,
    circuits: Optional[Sequence[str]] = None,
    initial: Optional[dict[str, dict[str, float]]] = None,
    callback: Optional[Callable[[str], None]] = None,
) -> CircuitComparison:
    """Fit several circuits to one spectrum and rank them honestly.

    ``circuits`` are names from :data:`LIBRARY` or circuit strings;
    without them a short ladder of increasing complexity is used, which
    is the order to try them in anyway.

    **The ranking is by BIC and the ranking is not the answer.** A
    circuit with more elements always fits at least as well, so the
    criterion has to penalise parameters or it is just a complexity
    contest — but a penalised criterion still only measures description,
    and an equivalent circuit is a claim about mechanism. What the
    comparison adds to the BIC column is the count of parameters that
    came back with a usable uncertainty, because that is the package's
    existing rule for whether an element earned its place, and a circuit
    that wins on BIC with half its parameters undetermined has fitted the
    shape without measuring anything.

    This is deliberately not what AutoEIS does. There the circuits are
    GENERATED by an evolutionary search and ranked by Bayesian model
    evidence, which needs MCMC and a Julia dependency; here the
    candidates are a handful somebody chose, the question is which of
    those few to report, and the honest answer fits in a table.
    """
    names = list(circuits) if circuits else [
        "R0-Q1", "R0-(R1|Q1)", "R0-Za1", "R0-(R1|Q1)-Q2",
        "R0-Za1-Za2", "R0-(R1|Q1)-T1",
    ]
    by_circuit = {t.circuit: t.name for t in TEMPLATES}
    candidates: list[CircuitCandidate] = []
    for entry in names:
        text = LIBRARY.get(entry, entry)
        candidate = CircuitCandidate(
            circuit=text,
            name=by_circuit.get(text, entry if entry in LIBRARY else ""),
            concerns=circuit_concerns(text),
        )
        if callback:
            callback(f"ajustando {text}")
        try:
            fit = fit_circuit(
                spectrum, text,
                initial=(initial or {}).get(text) or (initial or {}).get(entry),
            )
        except (CircuitError, ValueError) as exc:
            candidate.error = str(exc)
            candidates.append(candidate)
            continue
        candidate.fit = fit
        labels = [label for element in fit.elements for label in element.labels]
        free = [label for label in labels if label not in fit.fixed]
        candidate.free_parameters = len(free)
        values = fit.values()
        candidate.determined = sum(
            1 for label in free
            if (error := fit.errors.get(label)) is not None
            and values[label] != 0.0
            and abs(error / values[label]) <= PARAMETER_DETERMINED
        )
        # BIC on the modulus-weighted residual, over the real and
        # imaginary parts as separate points, which is what the fit
        # minimises. Same form the Raman side uses to choose between
        # band shapes.
        points = 2 * spectrum.n
        if fit.chi_squared > 0.0:
            candidate.bic = float(
                points * math.log(fit.chi_squared / points)
                + candidate.free_parameters * math.log(points)
            )
        candidates.append(candidate)
    return CircuitComparison(candidates=candidates, points=2 * spectrum.n)


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
    arcs = [e for e in elements if e.symbol == "Za"]
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
    # Several ZARCs in series are several processes, so they are spread
    # over the measured decades rather than stacked on one apex: started
    # identical they are a degenerate pair and the fit has no gradient
    # telling it which way to separate them.
    if len(arcs) > 1:
        slowest = 1.0 / max(float(spectrum.omega[-1]), 1e-12)
        fastest = 1.0 / max(float(spectrum.omega[0]), 1e-12)
        spread = np.geomspace(fastest, slowest, len(arcs) + 2)[1:-1]
        for element, tau in zip(arcs, spread):
            element.values = [polarisation / len(arcs), float(tau), 0.85]

    for element in elements:
        if element.symbol == "W":
            element.values = [polarisation]
        elif element.symbol in ("Ws", "Wo"):
            element.values = [polarisation, 1.0 / omega_peak]
        elif element.symbol == "L":
            element.values = [1e-6]
        elif element.symbol == "La":
            element.values = [1e-6, 1.0]
        elif element.symbol == "Za":
            # The two parameters the apex already measured: the arc's
            # diameter and 1/omega at its top. Seeding a ZARC is the one
            # case where the seed is the measurement, which is the same
            # reason `circuit_from_drt` can build the circuit outright.
            element.values = [polarisation, 1.0 / omega_peak, 0.85]
        elif element.symbol == "G":
            element.values = [polarisation * math.sqrt(omega_peak), omega_peak]
        elif element.symbol == "T":
            # The vertical part of a transmission line is offset from the
            # series resistance by R_ion/3, so the tail's own capacitance
            # and three times the span to the tail are the seeds that put
            # the fit in the right basin. Started from the defaults it
            # converges to a pore a thousand times too resistive.
            offset = max(float(real[-1]) - series, 1e-3)
            element.values = [3.0 * offset, max(tail, capacitance), 0.9]


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
    initial: Optional[dict[str, float]] = None,
    fixed: Optional[Sequence[str]] = None,
) -> EISResult:
    """Impedance analysis: consistency first, then a circuit.

    ``circuit=None`` skips the fit and returns only the model-free
    quantities, which is the right thing to do when the Kramers–Kronig test
    fails.

    ``initial`` and ``fixed`` are passed through to :func:`fit_circuit`;
    see there for why fixing a parameter is not a free action.
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
        result.fit = fit_circuit(spectrum, circuit, initial=initial, fixed=fixed)
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
    "DRT_PROCESSES",
    "circuit_from_drt",
    "describe_drt_peak",
    "drt",
    "drt_starting_values",
    "ELEMENTS",
    "KK_LIMIT",
    "KK_PER_DECADE",
    "KK_RUNS_LIMIT",
    "LIBRARY",
    "TEMPLATES",
    "CircuitError",
    "CircuitFit",
    "CircuitTemplate",
    "EISResult",
    "Element",
    "KKResult",
    "analyse_eis",
    "CircuitCandidate",
    "CircuitComparison",
    "PARAMETER_DETERMINED",
    "circuit_concerns",
    "circuit_for",
    "compare_circuits",
    "draw_circuit",
    "circuit_templates",
    "EffectiveCapacitance",
    "brug_capacitance",
    "cpe_to_capacitance",
    "hsu_mansfeld_capacitance",
    "delete_user_circuit",
    "fit_circuit",
    "kramers_kronig",
    "parse_circuit",
    "save_user_circuit",
    "user_circuit_file",
    "user_circuits",
    "uncompensated_resistance",
    "with_resistance_from",
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
    measured_s: tuple[float, float] = (0.0, 0.0)
    """The range of time constants the frequencies actually covered,
    ``(1/ω_max, 1/ω_min)``. Outside it γ is not determined by data and
    does not sit quietly at zero either — the slow end collects whatever
    the diffusion tail implies, which on the demo spectrum is seven
    hundred times the real features. Anything that reads γ, the plot
    included, has to know where the data stop."""
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

    def interpret(self) -> str:
        """The peaks, what each one is likely to be, and the test for it.

        The table a DRT is read from: τ, the resistance under the peak,
        the capacitance τ/R that identifies the process better than τ
        does, the share of the total polarisation resistance, and the
        experiment that would turn the guess into a measurement.
        """
        lines = [
            "DISTRIBUCIÓN DE CONSTANTES DE TIEMPO",
            "─" * 46,
            f"  R óhmica          {self.ohmic:.4g} Ω",
            f"  regularización λ  {self.regularisation:.4g}",
            f"  residuo           {100 * self.residual:.2f} %",
            f"  τ medidas         {self.measured_s[0]:.3g} – "
            f"{self.measured_s[1]:.3g} s",
            "",
        ]
        if not self.peaks_s:
            lines.append("  Sin picos separados dentro del intervalo medido.")
            lines.append("  Eso NO es «no hay procesos»: puede ser que λ esté")
            lines.append("  alto y los haya fundido, o que no estén resueltos.")
            return "\n".join(lines)
        total = sum(self.peak_resistances) or 1.0
        lines.append(
            "   τ (s)        R (Ω)     C = τ/R      % de R_p   proceso probable")
        for tau, resistance in zip(self.peaks_s, self.peak_resistances):
            process, _, capacitance = describe_drt_peak(tau, resistance)
            lines.append(
                f"  {tau:10.3g} {resistance:10.3g} {capacitance:11.3g} "
                f"{100 * resistance / total:9.1f}   {process}")
        lines.append("")
        lines.append("  Cómo confirmarlo —ninguna de estas asignaciones sale")
        lines.append("  de un solo espectro:")
        for tau, resistance in zip(self.peaks_s, self.peak_resistances):
            _, test, _ = describe_drt_peak(tau, resistance)
            if test:
                lines.append(f"   τ = {tau:.3g} s: {test}")
        for warning in self.warnings:
            lines.append(f"  ⚠ {warning}")
        return "\n".join(lines)


#: What a relaxation time of a given size usually is, in a carbon or
#: oxide electrode in aqueous or organic electrolyte.
#:
#: **Read as a prompt, never as an identification.** A time constant is
#: a number and the processes below overlap in it: an electrode with
#: thick films and one with fast intercalation can put a peak at the same
#: τ. What assigns a peak is varying something and watching it move —
#: potential moves charge transfer, temperature moves everything with an
#: activation energy, electrolyte concentration moves transport, film
#: thickness moves the film arc — and nothing in a single spectrum can
#: substitute for that. Each entry therefore carries the experiment that
#: would confirm it, because a prompt whose test is named is useful and
#: one without is a guess with a citation.
DRT_PROCESSES: tuple[tuple[float, float, str, str], ...] = (
    (0.0, 1e-6,
     "inductancia del montaje o artefacto de alta frecuencia",
     "por debajo de un microsegundo no suele haber electroquímica: acorta "
     "y trenza los cables y mira si se mueve"),
    (1e-6, 1e-4,
     "transporte en el electrolito, contacto, o resistencia del "
     "colector de corriente",
     "cambia la concentración del electrolito: el transporte iónico se "
     "mueve, un contacto no"),
    (1e-4, 1e-2,
     "transferencia de carga en la interfase, o la película superficial "
     "(SEI, óxido nativo)",
     "cambia el potencial: la transferencia de carga se mueve con él de "
     "forma exponencial, una película apenas"),
    (1e-2, 1.0,
     "transferencia de carga lenta, o difusión dentro del poro",
     "cambia la temperatura y saca la energía de activación; y mira el "
     "Bode: una difusión en poro trae su recta de 45°"),
    (1.0, 100.0,
     "difusión en estado sólido o intercalación",
     "cambia el estado de carga: un proceso de intercalación cambia su "
     "resistencia con él, una difusión en el electrolito no"),
    (100.0, float("inf"),
     "acumulación, relajación muy lenta, o la cola de difusión sin medir",
     "mide a frecuencias más bajas: si el pico se mueve al bajar el "
     "extremo, era la cola y no un proceso"),
)


def describe_drt_peak(tau: float, resistance: float) -> tuple[str, str, float]:
    """What a peak at this time constant is likely to be.

    Returns ``(process, the experiment that would confirm it, C = τ/R)``.
    The capacitance is the useful half of the answer and it is exact
    rather than suggested: a peak of γ(τ) IS an arc of resistance ``R``
    and time constant ``τ``, so its capacitance is ``τ/R`` with no
    conversion and no Brug relation — the same identity that makes a
    ZARC's parameters readable. Its ORDER OF MAGNITUDE is what identifies
    the process, far more reliably than τ alone: tens of µF/cm² is a
    double layer, mF is pseudocapacitance or a thick film, and anything
    above that is bulk storage.
    """
    capacitance = tau / resistance if resistance > 0 else float("nan")
    for low, high, process, test in DRT_PROCESSES:
        if low <= tau < high:
            return process, test, capacitance
    return "fuera de los intervalos tabulados", "", capacitance


def circuit_from_drt(result: "DRTResult", max_arcs: int = 4) -> tuple[str, list[str]]:
    """Build an equivalent circuit from the DRT, and say what it assumes.

    One ZARC per resolved peak, in series behind the ohmic resistance:
    ``R0-Za1-Za2-…``. This is a **translation, not a fit**. A peak of
    γ(τ) already measures a time constant and, as the area under it, a
    resistance — which are two of a ZARC's three parameters — so the
    circuit and its starting values come out of the DRT together, and
    the fit afterwards starts in the right basin instead of in whichever
    one the default seeds land in.

    That is the reason this exists. The usual route is to guess a circuit
    from the shape of the Nyquist plot, and the shape of a Nyquist plot
    is exactly what hides overlapping arcs — two processes a factor of
    five apart in τ look like one slightly depressed semicircle. The DRT
    separates them without a model, which is why it has to be read first.

    What it does NOT do:

    * **It cannot see a process the frequencies did not reach.** Only
      peaks inside the measured range are used, for the same reason the
      peak finder is limited to it.
    * **It gives no diffusion element.** A Warburg, a Gerischer and a
      porous transmission line are continuous distributions, not peaks;
      the DRT spreads them over a broad hump that this will chop into
      arcs. If the Bode plot has a 45° stretch, the chain below is the
      wrong model there and the note says so.
    * **It does not know what the processes ARE.**
      :func:`describe_drt_peak` offers a prompt and an experiment, and
      that is as far as one spectrum goes.

    Returns ``(circuit string, notes)``.
    """
    notes: list[str] = []
    peaks = list(zip(result.peaks_s, result.peak_resistances))
    if not peaks:
        return "R0-Q1", [
            "la DRT no resolvió ningún pico dentro del intervalo medido, "
            "así que no hay nada que traducir: lo que se propone es el "
            "circuito más simple que no afirma procesos que no se ven"
        ]
    peaks.sort(key=lambda item: item[0])
    if len(peaks) > max_arcs:
        notes.append(
            f"la DRT dio {len(peaks)} picos y se usan los {max_arcs} de "
            "mayor resistencia: un arco por pico más allá de eso es un "
            "circuito con más parámetros que regiones de frecuencia"
        )
        peaks = sorted(sorted(peaks, key=lambda item: -item[1])[:max_arcs],
                       key=lambda item: item[0])

    # Peaks closer than a factor of three in tau are ONE process. The same
    # threshold the module already applies when reporting them: at that
    # separation the split is decided by the regularisation and not by the
    # data, and two ZARCs there are two names for one arc.
    merged: list[tuple[float, float]] = []
    for tau, resistance in peaks:
        if merged and tau / merged[-1][0] < 3.0:
            previous_tau, previous_r = merged[-1]
            total = previous_r + resistance
            notes.append(
                f"los picos en τ = {previous_tau:.3g} s y {tau:.3g} s están a "
                f"un factor {tau / previous_tau:.1f}, por debajo de tres: se "
                "funden en uno. A esa distancia el reparto lo decide λ y no "
                "los datos"
            )
            merged[-1] = (math.sqrt(previous_tau * tau), total)
        else:
            merged.append((tau, resistance))

    circuit = "R0-" + "-".join(f"Za{index + 1}" for index in range(len(merged)))
    for index, (tau, resistance) in enumerate(merged, start=1):
        process, test, capacitance = describe_drt_peak(tau, resistance)
        notes.append(
            f"Za{index}: τ = {tau:.3g} s, R = {resistance:.3g} Ω, "
            f"C = τ/R = {capacitance:.3g} F — {process}. "
            f"Para comprobarlo: {test}" if test else
            f"Za{index}: τ = {tau:.3g} s, R = {resistance:.3g} Ω, "
            f"C = τ/R = {capacitance:.3g} F — {process}"
        )
    notes.append(
        "es una TRADUCCIÓN de la DRT, no un ajuste: γ(τ) ya midió cada τ y "
        "cada R, así que el circuito y sus valores de partida salen juntos. "
        "Ajústalo después para obtener las incertidumbres"
    )
    notes.append(
        "y si el Bode tiene un tramo a 45°, ahí hay difusión —Warburg, "
        "Gerischer o línea de transmisión— que es una distribución continua "
        "y no un pico: esta cadena la trocearía en arcos. Cámbiale el "
        "último arco por el elemento que corresponda"
    )
    return circuit, notes


def drt_starting_values(result: "DRTResult", circuit: str) -> dict[str, float]:
    """Starting values for the ZARCs of a circuit built from a DRT.

    Pairs the ZARCs in the circuit with the DRT's peaks in order of
    increasing τ, which is the order :func:`circuit_from_drt` writes them
    in. ``R0`` gets the DRT's own ohmic term.
    """
    elements = [e for e in parse_circuit(circuit).elements() if e.symbol == "Za"]
    peaks = sorted(zip(result.peaks_s, result.peak_resistances))
    values: dict[str, float] = {}
    resistors = [e for e in parse_circuit(circuit).elements() if e.symbol == "R"]
    if resistors and result.ohmic > 0:
        values[f"{resistors[0].name}.R"] = float(result.ohmic)
    for element, (tau, resistance) in zip(elements, peaks):
        values[f"{element.name}.R"] = float(resistance)
        values[f"{element.name}.tau"] = float(tau)
        values[f"{element.name}.gamma"] = 0.85
    return values


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

    # Peaks are looked for only inside the MEASURED range of time
    # constants. Outside it the distribution is not determined by data,
    # and it does not sit quietly there either: the unmeasured slow end
    # collects whatever the diffusion tail implies, which on the demo
    # spectrum was a γ two hundred times the real features and hid every
    # one of them under a threshold expressed as a fraction of the
    # maximum. The pile-up is still reported, as a warning.
    measured = (float(1.0 / omega.max()), float(1.0 / omega.min()))
    peaks, resistances = _drt_peaks(tau, gamma, measured=measured)
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
        peak_resistances=resistances, ohmic=ohmic, measured_s=measured,
        warnings=warnings,
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


def _drt_peaks(tau, gamma, min_fraction: float = 0.05, measured=None):
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
    tau = np.asarray(tau, dtype=float)
    if gamma.size < 3 or float(np.max(gamma)) <= 0:
        return [], []
    inside = (np.ones(tau.shape, dtype=bool) if measured is None
              else (tau >= measured[0]) & (tau <= measured[1]))
    if not inside.any():
        return [], []
    threshold = min_fraction * float(np.max(gamma[inside]))
    log_tau = np.log(tau)

    maxima = [
        index for index in range(1, gamma.size - 1)
        if inside[index]
        and gamma[index] > threshold
        and gamma[index] >= gamma[index - 1]
        and gamma[index] >= gamma[index + 1]
    ]
    if not maxima:
        return [], []

    # Basin boundaries: the lowest point between neighbouring peaks, and
    # the edges of the MEASURED range at the two ends. Letting the outer
    # basins run to the edge of the extended grid makes them swallow the
    # pile-up that collects there: on the demo spectrum the slowest peak
    # came back as 3980 ohms, which is the unmeasured diffusion tail, not
    # a process.
    first_inside = int(np.argmax(inside))
    last_inside = int(inside.size - 1 - np.argmax(inside[::-1]))
    bounds = [first_inside]
    for first, second in zip(maxima, maxima[1:]):
        bounds.append(first + int(np.argmin(gamma[first:second + 1])))
    bounds.append(last_inside)

    peaks, resistances = [], []
    for index, position in enumerate(maxima):
        left, right = bounds[index], bounds[index + 1]
        if right <= left:
            continue
        peaks.append(float(tau[position]))
        resistances.append(float(trapezoid(gamma[left:right + 1],
                                           log_tau[left:right + 1])))
    return peaks, resistances
