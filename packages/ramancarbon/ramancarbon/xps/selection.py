"""Which chemical states go into a region's fit, and why each one earned it.

The reference checklist asks it in one line — *"does adding the peak only
improve the residual? Then do not accept it automatically"* — and until
now nothing in this package answered it. The default was to fit every
state the database lists for a region, which for C 1s is fourteen. On a
real PHI Quantera spectrum of a carbon sample that came out as:

    14 componentes, chi2r = 2.49, R2 = 0.99861
    11 pares de parámetros con |r| >= 0.95, seis de ellos en 1.000
       C-O.centre    – C-O-C.centre    -1.000
       O-C=O.height  – anhydride.height -1.000
       C=O.height    – quinone-C.height -0.997

An R² of 0.99861 and four chemical states whose areas are a free choice.
The fit was not wrong about the envelope; it was wrong about having
measured nine chemical states, and it reported per-cent compositions for
several whose uncertainty came back at millions of eV.

So the states are selected instead of assumed, in four filters that each
answer one line of the checklist:

**Does the peak require another element?**  A C–N component needs
nitrogen somewhere in the sample. Given the elements actually present, the
states whose ``requires_any`` is unsatisfied are dropped and named — the
absence is evidence, and a C 1s model with C–S in a sample with no sulfur
is fitting a chemistry that cannot exist.

**Are there two equivalent interpretations?**  Ester, lactone, anhydride
and carboxylic acid all sit between 288 and 290 eV and no fit separates
them. One member per family goes in, under the family's name, and the
others are reported as the ambiguity they are rather than splitting one
peak's area four ways.

**Does adding it only improve the residual?**  The states the literature
calls well established (evidence A and B) start in the model. Every other
one is fitted in and kept only if it pays for its parameters by
:data:`BIC_GAIN` on the Bayesian information criterion — which charges
``ln(n)`` per parameter and is the conservative choice when the failure
mode is over-fitting.

**Is the area stable, and are there two parameters the data do not
separate?**  A candidate that arrives with a negligible area, or that
turns out not to be separable from a component already in the model, is
refused whatever it did to the criterion. A better number bought with an
undetermined parameter is not a better measurement.

What comes out is a model, the fit of it, and a line per candidate saying
what happened to it. The log is the point: a reader who disagrees with a
rejection can put the state back by name, and one who agrees now has the
grounds written down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .acceptance import MIN_SIGNAL_SIGMA
from .elements import ChemicalState
from .fitting import XPSFitResult, XPSModel, fit_region
from .presets import _one_per_family, state_model
from .spectrum import XPSSpectrum
from .tables import XPSDatabase, XPSError, load_xps_database

#: Evidence levels that start in the model without having to earn it.
#:
#: A and B are the levels the master table calls "Alta" and
#: "Alta/Moderada": sp² and sp³ carbon, C=O, pyridinic and graphitic
#: nitrogen, the sulfide doublet. Refusing to fit those until they prove
#: themselves would be the opposite error — they are what the region is
#: made of.
CORE_EVIDENCE = frozenset({"A", "B"})

#: How much BIC a candidate must save to be kept.
#:
#: Six. On the BIC scale a difference of 2 is "worth a glance", 6 is
#: "positive evidence" and 10 is "strong" — the conventional reading of
#: Kass and Raftery, and the reason not to use zero is that at zero every
#: candidate that happens to absorb a little noise gets in. Measured on
#: the real C 1s: with the gate at zero, three states arrive whose areas
#: are under 1 % and whose centres are pinned; at six, none of them does.
BIC_GAIN = 6.0

#: Smallest share of the fitted peak area a new component may have.
#:
#: Below this it is not a chemical state, it is a correction to the
#: background with a name on it.
MIN_AREA_FRACTION = 0.02

#: Correlation at which two components stop being two.
DEGENERATE = 0.95


@dataclass
class Step:
    """What happened to one candidate state."""

    state: str
    label: str
    accepted: bool
    delta_bic: float = float("nan")
    reason: str = ""

    def __str__(self) -> str:
        mark = "✓" if self.accepted else "·"
        gap = ("" if not np.isfinite(self.delta_bic)
               else f"  ΔBIC {self.delta_bic:+.1f}")
        return f"  {mark} {self.label}{gap}{('  — ' + self.reason) if self.reason else ''}"


@dataclass
class Selection:
    """A region's model, its fit, and the grounds for every component."""

    region: str
    model: XPSModel
    result: XPSFitResult
    core: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    missing_element: list[tuple[str, str]] = field(default_factory=list)
    """``(state name, the elements it would need)`` for states dropped
    because the sample does not contain what they are made of."""
    ambiguous: list[tuple[str, list[str]]] = field(default_factory=list)
    """``(family, member names)`` where the fit cannot separate members."""
    elements_known: bool = True
    """False when nobody said which elements are present, so the
    ``requires_any`` filter could not run."""

    @property
    def accepted(self) -> list[str]:
        return self.core + [step.state for step in self.steps if step.accepted]

    def summary(self) -> str:
        lines = [f"SELECCIÓN DE ESTADOS — {self.region}"]
        lines.append(f"  núcleo (evidencia A/B, entra sin condiciones): "
                     f"{', '.join(self.core) or 'ninguno'}")
        if self.steps:
            lines.append(f"  candidatos, aceptados si ganan {BIC_GAIN:.0f} de BIC:")
            lines.extend(str(step) for step in self.steps)
        if self.missing_element:
            lines.append("  descartados por composición (el elemento no está "
                         "en la muestra):")
            lines.extend(f"    {name} — necesita {need}"
                         for name, need in self.missing_element)
        elif not self.elements_known:
            lines.append("  no hay composición completa (hace falta un survey), "
                         "así que no se ha descartado ningún estado por "
                         "composición: medir una región prueba que el elemento "
                         "está, no que los demás no estén")
        if self.ambiguous:
            lines.append("  ambigüedades que el ajuste NO resuelve (entra un "
                         "miembro por familia, con el nombre de la familia):")
            lines.extend(f"    {family}: {', '.join(members)}"
                         for family, members in self.ambiguous)
        result = self.result
        lines.append(f"  resultado: {len(result.components)} componentes, "
                     f"{result.n_parameters} parámetros, "
                     f"χ²_red = {result.reduced_chi2:.3f}, "
                     f"BIC = {result.bic:.1f}")
        return "\n".join(lines)


def _candidate_order(states: Sequence[ChemicalState]) -> list[ChemicalState]:
    """Candidates, best-supported first.

    Priority before evidence before energy: a ``preferred`` state with
    level C is a better bet than a ``conditional`` one with level C, and
    between equals the lower binding energy goes first because that is
    where the intensity is in every region this package fits.
    """
    priority = {"preferred": 0, "conditional": 1, "last_resort": 2}
    evidence = {"A": 0, "B": 1, "C": 2, "D": 3}
    return sorted(states, key=lambda s: (
        priority.get(s.fitting_priority, 1),
        evidence.get(s.evidence_level, 2),
        s.energy_ev,
    ))


def _new_degeneracy(result: XPSFitResult, name: str) -> Optional[str]:
    """A pair involving ``name`` that the data do not separate."""
    for (first, second), value in result.correlations.items():
        if abs(value) < DEGENERATE:
            continue
        owners = (first.split(".")[0], second.split(".")[0])
        if name not in owners:
            continue
        other = owners[1] if owners[0] == name else owners[0]
        return f"sus parámetros no se separan de los de {other} (r = {value:+.2f})"
    return None


def _area_fraction(result: XPSFitResult, name: str) -> float:
    total = sum(abs(item.area) for item in result.components) or 1.0
    for item in result.components:
        if item.name == name:
            return abs(item.area) / total
    return 0.0


def evidence_model(
    spectrum: XPSSpectrum,
    region: str,
    present: Optional[Sequence[str]] = None,
    complete: bool = False,
    database: Optional[XPSDatabase] = None,
    bic_gain: float = BIC_GAIN,
    include_satellites: bool = True,
    **fit_options,
) -> Selection:
    """Fit a region with the states the data actually support.

    Parameters
    ----------
    spectrum:
        The region to fit.
    region:
        Database region key, e.g. ``"C 1s"``.
    present:
        Element symbols known to be in the sample. Used only to confirm
        that a state's ``requires_any`` is satisfied.
    complete:
        Whether ``present`` is the WHOLE composition. Only then may a
        state be dropped for a missing element, and the distinction is not
        pedantic: measuring a region proves an element is there and proves
        nothing about the ones nobody measured. Deriving the composition
        from the list of measured regions — which is the obvious thing to
        do with a file of four narrow scans — turns "not measured" into
        "not present", and on a real spectrum that dropped the metal-oxide
        O 1s component from a sample whose O 1s peaks at 530.05 eV,
        exactly where that component sits. What was left could not reach
        the peak: two components, both pinned against their limits, with a
        reduced chi-squared of 34 against 1.39 for the model that keeps
        it. So the filter runs only when a survey says what is in the
        sample, and otherwise every candidate gets its chance and the
        audit flags the ones resting on an unverified element.
    bic_gain:
        How much BIC a candidate must save. Zero accepts anything that
        improves the fit at all, which is the behaviour this module exists
        to replace.
    include_satellites:
        Attach the shake-up satellites of the accepted states, tied to
        their parents.
    **fit_options:
        Passed to :func:`~ramancarbon.xps.fitting.fit_region`.

    Raises
    ------
    XPSError
        If the region is not in the database, or the core states cannot be
        fitted at all.
    """
    database = database or load_xps_database()
    available = [state for state in database.states_for(region)
                 if not state.is_satellite]
    if not available:
        raise XPSError(
            f"la base de datos no tiene estados para {region!r}; tiene: "
            f"{', '.join(database.region_names())}"
        )

    missing: list[tuple[str, str]] = []
    if present is not None and complete:
        symbols = {str(item).strip() for item in present}
        keep = []
        for state in available:
            if state.requires_any and not symbols.intersection(state.requires_any):
                missing.append((state.name, " o ".join(state.requires_any)))
            else:
                keep.append(state)
        available = keep

    available, ambiguous = _one_per_family(available)

    core = [state for state in available if state.evidence_level in CORE_EVIDENCE]
    candidates = _candidate_order(
        [state for state in available if state not in core])
    if not core:
        # Nothing the literature calls well established. Rather than fit
        # an empty model, the single best-supported candidate becomes the
        # core and the rest still have to earn their place.
        if not candidates:
            raise XPSError(f"no queda ningún estado que ajustar en {region!r}")
        core, candidates = [candidates[0]], candidates[1:]

    def fit(keys: list[str]) -> tuple[XPSModel, XPSFitResult]:
        model = state_model(spectrum, region, keys, database=database,
                            include_satellites=include_satellites)
        return model, fit_region(spectrum, model, database=database,
                                 **fit_options)

    noise = spectrum.noise_estimate()
    if not noise or not np.isfinite(noise) or noise <= 0:
        noise = 0.0

    chosen = [state.key for state in core]
    model, result = fit(chosen)
    steps: list[Step] = []

    for state in candidates:
        trial_keys = chosen + [state.key]
        try:
            trial_model, trial = fit(trial_keys)
        except (XPSError, ValueError) as error:
            steps.append(Step(state.key, state.name, False,
                              reason=f"no se pudo ajustar: {error}"))
            continue
        delta = trial.bic - result.bic
        name = next((c.name for c in trial.components
                     if c.state == state.key), state.key)
        if delta > -bic_gain:
            steps.append(Step(state.key, state.name, False, delta,
                              f"no paga sus parámetros (haría falta "
                              f"{-bic_gain:+.0f})"))
            continue
        fraction = _area_fraction(trial, name)
        if fraction < MIN_AREA_FRACTION:
            steps.append(Step(state.key, state.name, False, delta,
                              f"su área sale el {100.0 * fraction:.1f} % del "
                              "total: es una corrección al fondo con nombre, "
                              "no un estado químico"))
            continue
        # The same test the audit will apply, applied here so the two
        # cannot disagree. They did: the selection accepted an oxidised
        # nitrogen holding 3.7 % of the N 1s area -- past the area gate --
        # whose height was under three times the noise, and the audit then
        # called the region BAJA for it. A gate that admits what the next
        # check throws out is not a gate.
        component = next((c for c in trial.components if c.name == name), None)
        if component is not None and noise and component.peak_height < (
                MIN_SIGNAL_SIGMA * noise):
            steps.append(Step(state.key, state.name, False, delta,
                              f"su altura ({component.peak_height:.0f} cuentas) "
                              f"no llega a {MIN_SIGNAL_SIGMA:.0f} veces el "
                              f"ruido ({noise:.0f})"))
            continue
        clash = _new_degeneracy(trial, name)
        if clash:
            steps.append(Step(state.key, state.name, False, delta, clash))
            continue
        steps.append(Step(state.key, state.name, True, delta))
        chosen, model, result = trial_keys, trial_model, trial

    # Grow, then prune. A candidate is judged against the model as it
    # stood when it arrived, and the model keeps changing after that: on a
    # real N 1s the oxidised-nitrogen component was worth 53 points of BIC
    # when it went in and, once the protonated-pyridinic component had
    # also been accepted, its height had dropped below three times the
    # noise. Checking each arrival is not the same as checking the model
    # that comes out, so the ones that no longer hold up are removed and
    # the rest refitted, until nothing more falls.
    for _ in range(len(steps) + 1):
        doomed = None
        for step in steps:
            if not step.accepted:
                continue
            component = next(
                (c for c in result.components if c.state == step.state), None)
            if component is None or component.satellite:
                continue
            fraction = _area_fraction(result, component.name)
            if noise and component.peak_height < MIN_SIGNAL_SIGMA * noise:
                doomed = (step, f"al final del ajuste su altura "
                                f"({component.peak_height:.0f}) no llega a "
                                f"{MIN_SIGNAL_SIGMA:.0f} veces el ruido "
                                f"({noise:.0f})")
                break
            if fraction < MIN_AREA_FRACTION:
                doomed = (step, f"al final del ajuste su área queda en el "
                                f"{100.0 * fraction:.1f} %")
                break
            clash = _new_degeneracy(result, component.name)
            if clash:
                # Pruning can create a degeneracy that was not there when
                # the candidate arrived: dropping one component moves the
                # others onto each other. A pair the data do not separate
                # is reported as AMBIGUA when both members are core states
                # and nothing can be done, and removed when one of them is
                # a candidate that has an alternative.
                doomed = (step, f"al final del ajuste {clash}")
                break
        if doomed is None:
            break
        step, why = doomed
        step.accepted, step.reason = False, why
        chosen = [key for key in chosen if key != step.state]
        model, result = fit(chosen)

    return Selection(
        region=region, model=model, result=result,
        core=[state.key for state in core], steps=steps,
        missing_element=missing, ambiguous=list(ambiguous),
        elements_known=bool(present is not None and complete),
    )


__all__ = [
    "BIC_GAIN",
    "CORE_EVIDENCE",
    "DEGENERATE",
    "MIN_AREA_FRACTION",
    "Selection",
    "Step",
    "evidence_model",
]
