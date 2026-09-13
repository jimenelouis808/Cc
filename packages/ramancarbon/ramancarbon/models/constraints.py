"""Linking one fitted parameter to another.

Fixing a parameter is already possible; what specialised fitting software
adds is *linking* — saying that one parameter is a fixed function of
another rather than a fixed number. It is what makes several fits that
should be the same fit actually be the same fit:

``D3.centre = D.centre + 100``
    A shoulder whose position is known relative to a band the data can
    locate, but not on its own.
``G-.fwhm = G.fwhm``
    Two components of a split band that physically share a width.
``pico2.height = 0.5 * pico1.height``
    A doublet with a known branching ratio.

The gain is not convenience. A parameter that the data cannot determine
will be driven by noise if it is free, and will be wrong if it is fixed to
a guess; linking it to something the data *can* determine is the only one
of the three that is both honest and stable. A linked fit has fewer free
parameters than a free one, so its uncertainties are smaller — and that
is only legitimate when the link is physics, which is why every link is
written down in the report.

The syntax is deliberately small: ``target = factor * source + offset``,
with the factor and the offset optional. Anything more would be an
expression language, and an expression language in a fit model is a way
to write constraints nobody can check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


class ConstraintError(ValueError):
    """Raised when a link cannot be parsed, resolved or applied."""


@dataclass(frozen=True)
class Link:
    """``target = factor · source + offset``."""

    target_peak: str
    target_parameter: str
    source_peak: str
    source_parameter: str
    factor: float = 1.0
    offset: float = 0.0

    @property
    def target(self) -> tuple[str, str]:
        return (self.target_peak, self.target_parameter)

    @property
    def source(self) -> tuple[str, str]:
        return (self.source_peak, self.source_parameter)

    def apply(self, source_value: float) -> float:
        return self.factor * float(source_value) + self.offset

    def __str__(self) -> str:
        text = f"{self.target_peak}.{self.target_parameter} = "
        if self.factor != 1.0:
            text += f"{self.factor:g}·"
        text += f"{self.source_peak}.{self.source_parameter}"
        if self.offset:
            text += f" {'+' if self.offset > 0 else '-'} {abs(self.offset):g}"
        return text


_REFERENCE = r"([A-Za-z0-9_+\-⁻¹'′][^\s.=*+]*)\.([A-Za-z0-9_]+)"
_PATTERN = re.compile(
    rf"^\s*{_REFERENCE}\s*=\s*"
    r"(?:([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\*\s*)?"
    rf"{_REFERENCE}"
    r"(?:\s*([-+])\s*(\d*\.?\d+(?:[eE][-+]?\d+)?))?\s*$"
)


def parse_link(text: str) -> Link:
    """Read one link from ``"D3.centre = D.centre + 100"``.

    Raises
    ------
    ConstraintError
        With the offending text, because a mistyped link that silently
        did nothing would be worse than one that fails.
    """
    match = _PATTERN.match(text)
    if not match:
        raise ConstraintError(
            f"no entiendo la ligadura {text!r}. La forma es "
            "«objetivo.parámetro = factor * origen.parámetro + desplazamiento», "
            "con el factor y el desplazamiento opcionales"
        )
    target_peak, target_parameter, factor, source_peak, source_parameter, \
        sign, offset = match.groups()
    value = float(offset) if offset else 0.0
    if sign == "-":
        value = -value
    return Link(
        target_peak=target_peak, target_parameter=target_parameter,
        source_peak=source_peak, source_parameter=source_parameter,
        factor=float(factor) if factor else 1.0, offset=value,
    )


def parse_links(lines: Iterable[str]) -> list[Link]:
    """Several links, ignoring blank lines and ``#`` comments."""
    out: list[Link] = []
    for line in lines:
        stripped = line.split("#", 1)[0].strip()
        if stripped:
            out.append(parse_link(stripped))
    return out


def validate(links: Sequence[Link], available: dict[str, Sequence[str]]) -> None:
    """Check every link against the model it will be applied to.

    ``available`` maps each component's name to its parameter names.
    Everything that can be caught before fitting is caught here: an
    unknown component, an unknown parameter, the same target linked twice,
    a self-link, and a cycle.
    """
    seen: set[tuple[str, str]] = set()
    for link in links:
        for peak, parameter, role in (
            (link.target_peak, link.target_parameter, "objetivo"),
            (link.source_peak, link.source_parameter, "origen"),
        ):
            if peak not in available:
                raise ConstraintError(
                    f"«{link}»: el componente {role} «{peak}» no está en el "
                    f"modelo; hay: {', '.join(sorted(available))}"
                )
            if parameter not in available[peak]:
                raise ConstraintError(
                    f"«{link}»: «{peak}» no tiene el parámetro «{parameter}»; "
                    f"tiene: {', '.join(available[peak])}"
                )
        if link.target == link.source:
            raise ConstraintError(f"«{link}»: un parámetro no se liga a sí mismo")
        if link.target in seen:
            raise ConstraintError(
                f"«{link}»: {link.target_peak}.{link.target_parameter} ya está "
                "ligado; un parámetro solo puede tener una ligadura"
            )
        seen.add(link.target)
    order(links)


def order(links: Sequence[Link]) -> list[Link]:
    """The links in an order that can be applied one after another.

    A link whose source is itself a link target has to be resolved after
    it. Chains are allowed and cycles are not — and a cycle is easy to
    write by accident (``a = b``, ``b = a``), where the effect is not an
    error but a value that depends on the order things happened to be
    applied in.
    """
    targets = {link.target: link for link in links}
    resolved: list[Link] = []
    state: dict[tuple[str, str], int] = {}

    def visit(key: tuple[str, str], trail: list[tuple[str, str]]) -> None:
        if state.get(key) == 2:
            return
        if state.get(key) == 1:
            names = " → ".join(f"{peak}.{parameter}"
                               for peak, parameter in trail + [key])
            raise ConstraintError(f"las ligaduras forman un ciclo: {names}")
        state[key] = 1
        link = targets.get(key)
        if link is not None:
            if link.source in targets:
                visit(link.source, trail + [key])
            resolved.append(link)
        state[key] = 2

    for key in targets:
        visit(key, [])
    return resolved


def describe(links: Sequence[Link]) -> list[str]:
    """One line per link, for the report.

    Every link goes in the report because a linked fit has fewer free
    parameters and therefore smaller uncertainties, and that is only
    legitimate when the link is physics rather than convenience.
    """
    return [str(link) for link in links]


__all__ = [
    "ConstraintError",
    "Link",
    "describe",
    "order",
    "parse_link",
    "parse_links",
    "validate",
]
