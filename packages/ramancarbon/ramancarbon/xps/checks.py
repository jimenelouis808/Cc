"""Checks on a finished fit: is it a measurement or a bound?

A least-squares fit always returns numbers. What decides whether they
mean anything is not the residual -- it is whether the parameters landed
where the data put them or where the constraints stopped them, and
whether the residual still has structure in it.

Warnings W02, W03 and W09 of the user's specification. None of them
changes a fit; every one of them is a result.
"""

from __future__ import annotations

from typing import Optional

from .elements import XPSDatabase, load_xps_database
from .fitting import XPSFitResult

#: How close to its bound a value has to be to count as sitting ON it, as
#: a fraction of the allowed range. A parameter the optimiser pushed into
#: the wall stops within numerical noise of it; 2 % of the window is well
#: inside that and well outside anything the data would choose by chance.
BOUND_FRACTION = 0.02

#: Two components closer than this, in units of the narrower one's width,
#: are one component with two labels.
DEGENERATE_FRACTION = 0.25

#: Below this singularity index a Doniach-Šunjić is a Lorentzian with
#: extra steps: alpha enters as |u|^(alpha-1), so at 0.01 the tail is
#: within a per cent of symmetric over any window anyone fits. Measured
#: on synthetic peaks, the area error a symmetric shape makes on a
#: genuinely asymmetric line is 2 % at alpha = 0.05, so this sits an
#: order of magnitude below the smallest asymmetry worth claiming.
MIN_MEANINGFUL_ASYMMETRY = 0.01


def bound_checks(
    result: XPSFitResult,
    region: str,
    database: Optional[XPSDatabase] = None,
) -> list[tuple[str, str]]:
    """``(veredicto, texto)`` for every parameter that hit a wall.

    A component pinned to the edge of its published window is the fit
    saying "the intensity is outside where you let me put it". The number
    it reports is the edge, not a position, and its area is whatever fits
    under a peak held in the wrong place. Reading it as a measurement is
    how a constraint becomes a result.
    """
    database = database or load_xps_database()
    out: list[tuple[str, str]] = []
    pinned: dict[str, list[str]] = {"inferior": [], "superior": []}

    for component in result.components:
        if not component.state:
            continue
        state = next(
            (s for s in database.states_for(region, include_satellites=True)
             if s.key == component.state), None)
        if state is None:
            # A component the user added by hand, or a state from a
            # database version this one does not have. It carries no
            # published bounds, so there is nothing here to check.
            continue

        low, high = state.window
        span = max(high - low, 1e-9)
        margin = BOUND_FRACTION * span
        centre = float(component.peak_position)
        if centre <= low + margin or centre >= high - margin:
            edge = "inferior" if centre <= low + margin else "superior"
            pinned[edge].append(state.name)
            out.append(("aviso", (
                f"{region}: «{state.name}» se ha quedado pegada al límite "
                f"{edge} de su ventana ({low:.1f}–{high:.1f} eV) en "
                f"{centre:.2f} eV. El ajuste está diciendo que la intensidad "
                "cae fuera de donde se le deja ponerla: ese número es el "
                "borde, no una posición, y su área es la que quepa bajo un "
                "pico sujeto donde no va")))

        wlow, whigh = state.fwhm
        wspan = max(whigh - wlow, 1e-9)
        width = float(component.true_fwhm)
        if width <= wlow + BOUND_FRACTION * wspan:
            out.append(("aviso", (
                f"{region}: «{state.name}» ha ido a la anchura mínima "
                f"({wlow:.1f} eV). Una componente que se estrecha hasta el "
                "límite suele estar tapando un residuo, no midiendo un "
                "estado")))
        elif width >= whigh - BOUND_FRACTION * wspan:
            out.append(("aviso", (
                f"{region}: «{state.name}» ha ido a la anchura máxima "
                f"({whigh:.1f} eV). O el estado no es ése, o hay dos "
                "entornos debajo")))

    # Several components pinned to the SAME side is one problem, not
    # several: the whole region wants to sit further along the axis than
    # the reference allows. That is charging or a bad calibration
    # (section 41), and reading it as five separate chemistry problems is
    # how an axis error gets fitted instead of fixed.
    for edge, names in pinned.items():
        if len(names) < 2:
            continue
        direction = ("más baja" if edge == "inferior" else "más alta")
        out.insert(0, ("incoherente", (
            f"{region}: {len(names)} componentes pegadas al MISMO límite "
            f"({edge}): {', '.join(names)}. Eso no son {len(names)} problemas "
            "de química, es uno de eje: la región entera quiere estar a "
            f"energía {direction}. Revisa la referencia de carga antes de "
            "tocar el modelo")))

    out.extend(_symmetric_in_all_but_name(result, region))
    out.extend(_degenerate(result, region))
    return out


def _symmetric_in_all_but_name(result: XPSFitResult,
                               region: str) -> list[tuple[str, str]]:
    """An asymmetric component whose alpha stopped at zero.

    This is the same fault as a position pinned to its window, and it
    hid because alpha is a profile parameter rather than one of the
    three the checks above walk. It matters more than most: in graphite
    and graphene the C 1s is asymmetric because the conduction electrons
    screen the core hole, and fitting it with a symmetric shape forces an
    extra component at about 285.5 eV that is then read as sp3 carbon.
    The database calls that the most common error in carbon XPS, and the
    whole point of giving the state an asymmetric profile is to avoid it.

    A component that says ``ds`` in the model and fits alpha = 0 has
    quietly gone back to it, and its area comes out short by whatever
    the tail was carrying -- which is exactly the direction of "the sp2
    is coming out too low".

    An alpha pinned to the TOP of its range is the mirror image and is
    worth as much: there the line shape is being used to absorb
    something else, usually a background that is not right or a
    component that is missing.
    """
    from .lineshapes import MAX_ASYMMETRY

    out: list[tuple[str, str]] = []
    for component in result.components:
        names = component.extra_names or ()
        if "asymmetry" not in names:
            continue
        alpha = float(component.extra[list(names).index("asymmetry")])
        label = component.label or component.name
        if alpha <= MIN_MEANINGFUL_ASYMMETRY:
            out.append(("aviso", (
                f"{region}: «{label}» se ajustó con forma ASIMÉTRICA y su "
                f"índice de singularidad se ha ido a α = {alpha:.3f}, que es "
                "el límite inferior: sobre estos datos la componente es "
                "simétrica de hecho, no de nombre. Una línea simétrica no "
                "puede llevar la cola, así que su área sale CORTA — y en un "
                "carbono grafítico ese defecto suele reaparecer como una "
                "componente sp³ de más hacia 285.5 eV. Prueba con una "
                "ventana más ancha y fondo lineal, o fija α al valor que "
                "uses en tu grupo")))
        elif alpha >= MAX_ASYMMETRY - BOUND_FRACTION * MAX_ASYMMETRY:
            out.append(("aviso", (
                f"{region}: «{label}» ha llegado al máximo de asimetría "
                f"(α = {alpha:.2f}). Una cola así de larga se está comiendo "
                "algo que no es la componente: casi siempre un fondo que no "
                "es el que toca o una componente que falta a energía de "
                "enlace mayor")))
    return out


def _degenerate(result: XPSFitResult, region: str) -> list[tuple[str, str]]:
    """Components sitting on top of each other: W09, overparameterised.

    Two peaks a quarter of a linewidth apart are not two chemical states
    that the fit resolved. They are one peak that the model had two
    labels for, and their separate areas are a division of one number by
    the optimiser's starting point.
    """
    out: list[tuple[str, str]] = []
    named = [c for c in result.components if c.state]
    for first, second in ((a, b) for i, a in enumerate(named)
                          for b in named[i + 1:]):
        narrower = min(float(first.true_fwhm), float(second.true_fwhm))
        gap = abs(float(first.peak_position) - float(second.peak_position))
        if narrower > 0 and gap < DEGENERATE_FRACTION * narrower:
            out.append(("incoherente", (
                f"{region}: «{first.label}» y «{second.label}» están a "
                f"{gap:.2f} eV, menos de un cuarto de su anchura. Eso no son "
                "dos estados resueltos: es un pico con dos etiquetas, y el "
                "reparto de su área entre los dos lo decide el punto de "
                "partida del ajuste, no la medida")))
    return out


__all__ = ["BOUND_FRACTION", "DEGENERATE_FRACTION",
           "MIN_MEANINGFUL_ASYMMETRY", "bound_checks"]
