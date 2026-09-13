"""Crystallographic symmetry operations, from strings to a closed group.

A CIF says what its symmetry is by listing operations as text —
``'-x+1/2, y, -z+1/2'`` — and that is all this module needs. There is no
space-group table here and that is deliberate: a table is a large piece of
data to get subtly wrong, whereas the operations in the file are the
authority on the file's own contents.

One thing is done beyond parsing. The listed operations are **closed under
composition** before use. For a well-formed CIF from the Crystallography
Open Database the closure changes nothing, because the file already lists
the complete coset decomposition; the check is free and catches a truncated
or hand-edited list. For the reference structures bundled with this
package it does real work, since listing a group's generators is far less
error-prone than typing out its 192 operations.

Closure also gives a cheap correctness test that no table can: the order of
the closed group must equal the multiplicity the space group is known to
have. A wrong operation almost always changes the order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

import numpy as np

#: Positions closer than this (in fractional coordinates) are the same atom.
MERGE_TOLERANCE = 1e-4

#: Refuse to close a group larger than this. A malformed operation can
#: generate an infinite group; without the guard the closure never returns.
MAX_ORDER = 400

_TERM = re.compile(r"([+-]?)\s*(\d+/\d+|\d*\.\d+|\d+)?\s*\*?\s*([xyz])?")


class ClosedGroup(list):
    """A list of operations already closed under composition.

    Marking the closure lets it be skipped on the way back in. A Rietveld
    refinement builds a new :class:`~ramancarbon.xrd.structure.Crystal`
    for every trial cell, and re-closing magnetite's 192 operations each
    time — an O(n²) walk over 37 000 compositions — took two thirds of
    the total refinement time while changing nothing.
    """


class SymmetryError(ValueError):
    """Raised when a symmetry operation cannot be parsed or closed."""


@dataclass(frozen=True)
class SymmetryOperation:
    """One ``(rotation, translation)`` pair acting on fractional coordinates.

    ``rotation`` is a 3×3 integer matrix and ``translation`` a length-3
    vector of fractions in [0, 1). Both are stored as floats for speed but
    the rotation is guaranteed integral: a crystallographic operation maps
    the lattice onto itself.
    """

    rotation: tuple[tuple[float, ...], ...]
    translation: tuple[float, ...]

    @property
    def R(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=float)

    @property
    def t(self) -> np.ndarray:
        return np.asarray(self.translation, dtype=float)

    def apply(self, positions: np.ndarray) -> np.ndarray:
        """Act on an ``(n, 3)`` array of fractional coordinates."""
        return np.mod(np.asarray(positions, dtype=float) @ self.R.T + self.t, 1.0)

    def compose(self, other: "SymmetryOperation") -> "SymmetryOperation":
        """``self ∘ other``: apply ``other`` first."""
        rotation = self.R @ other.R
        translation = np.mod(self.R @ other.t + self.t, 1.0)
        return _make(rotation, translation)

    def to_xyz(self) -> str:
        """Back to CIF text, e.g. ``-x+1/2, y, -z+1/2``."""
        names = ("x", "y", "z")
        parts = []
        for row, shift in zip(self.rotation, self.translation):
            term = ""
            for coefficient, name in zip(row, names):
                if abs(coefficient) < 1e-9:
                    continue
                sign = "-" if coefficient < 0 else ("+" if term else "")
                magnitude = abs(coefficient)
                prefix = "" if abs(magnitude - 1.0) < 1e-9 else f"{magnitude:g}*"
                term += f"{sign}{prefix}{name}"
            fraction = Fraction(shift).limit_denominator(12)
            if fraction:
                term += f"{'+' if fraction > 0 and term else ''}{fraction}"
            parts.append(term or "0")
        return ", ".join(parts)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return self.to_xyz()


def _make(rotation: np.ndarray, translation: np.ndarray) -> SymmetryOperation:
    """Build an operation, snapping to exact integers and twelfths.

    Floating-point drift is the enemy of group closure: two operations that
    differ by 1e-16 hash differently and the group never closes. Rotations
    are integers and translations are always n/12 in the 230 space groups,
    so both can be snapped exactly.
    """
    rotation = np.asarray(rotation, dtype=float)
    rounded = np.round(rotation)
    if float(np.max(np.abs(rounded - rotation))) > 1e-6:
        raise SymmetryError(
            f"la parte rotacional no es entera:\n{np.asarray(rotation)}"
        )
    shift = np.mod(np.round(np.asarray(translation, dtype=float) * 12.0) / 12.0, 1.0)
    shift[np.abs(shift - 1.0) < 1e-9] = 0.0
    return SymmetryOperation(
        rotation=tuple(tuple(float(v) for v in row) for row in rounded),
        translation=tuple(float(v) for v in shift),
    )


def parse_xyz(text: str) -> SymmetryOperation:
    """Parse one CIF-style operation string.

    Accepts the forms found in the wild: ``x,y,z``, ``-x+1/2, y, -z+1/2``,
    ``'1/2-x, 1/2+y, z'``, ``X,Y,Z``, and a leading serial number as some
    files write (``1 x,y,z``).
    """
    cleaned = text.strip().strip("'\"").lower().replace(" ", "")
    if not cleaned:
        raise SymmetryError("operación de simetría vacía")
    # Some files prefix a serial number: "1 x,y,z".
    if cleaned[0].isdigit() and "," in cleaned:
        head, _, rest = cleaned.partition(",")
        if not any(c in head for c in "xyz") and rest.count(",") == 2:
            cleaned = rest

    parts = cleaned.split(",")
    if len(parts) != 3:
        raise SymmetryError(f"la operación {text!r} no tiene tres componentes")

    rotation = np.zeros((3, 3))
    translation = np.zeros(3)
    for row, part in enumerate(parts):
        consumed = 0
        for match in _TERM.finditer(part):
            sign_text, number_text, variable = match.groups()
            if not match.group(0):
                continue
            consumed += len(match.group(0))
            if number_text is None and variable is None:
                continue
            sign = -1.0 if sign_text == "-" else 1.0
            if number_text is None:
                value = 1.0
            elif "/" in number_text:
                numerator, denominator = number_text.split("/")
                value = float(numerator) / float(denominator)
            else:
                value = float(number_text)
            if variable is None:
                translation[row] += sign * value
            else:
                rotation[row]["xyz".index(variable)] += sign * value
        if consumed < len(part):
            raise SymmetryError(f"no se entiende la componente {part!r} de {text!r}")
    return _make(rotation, translation)


def close_group(
    operations: Iterable[SymmetryOperation | str],
    max_order: int = MAX_ORDER,
) -> list[SymmetryOperation]:
    """Close a set of operations under composition.

    The identity is added if absent. Raises :class:`SymmetryError` if the
    group grows past ``max_order``, which in practice means one of the
    operations is wrong: a rotation that is not of crystallographic order,
    or a translation that is not a rational fraction of the cell.

    Returns
    -------
    list[SymmetryOperation]
        The closed group, identity first, otherwise in discovery order.
    """
    if isinstance(operations, ClosedGroup):
        return ClosedGroup(operations)
    identity = parse_xyz("x,y,z")
    group: dict[tuple, SymmetryOperation] = {}

    def key(op: SymmetryOperation) -> tuple:
        return (op.rotation, op.translation)

    group[key(identity)] = identity
    queue: list[SymmetryOperation] = []
    for item in operations:
        op = parse_xyz(item) if isinstance(item, str) else item
        if key(op) not in group:
            group[key(op)] = op
            queue.append(op)

    index = 0
    seeds = list(group.values())
    while index < len(queue) or seeds:
        current = queue[index] if index < len(queue) else None
        if current is None:
            break
        index += 1
        for other in list(group.values()):
            for product in (current.compose(other), other.compose(current)):
                identifier = key(product)
                if identifier in group:
                    continue
                if len(group) >= max_order:
                    raise SymmetryError(
                        f"el grupo supera las {max_order} operaciones al "
                        "cerrarse; alguna operación de simetría está mal "
                        "escrita (una rotación de orden no cristalográfico o "
                        "una traslación que no es fracción de la celda)"
                    )
                group[identifier] = product
                queue.append(product)
    return ClosedGroup(group.values())


def expand_positions(
    positions: Sequence[Sequence[float]],
    operations: Sequence[SymmetryOperation],
    tolerance: float = MERGE_TOLERANCE,
) -> list[list[int]]:
    """Generate every symmetry image of each site, merging coincidences.

    A site sitting on a symmetry element maps onto itself under part of the
    group. Counting those images separately would multiply its scattering
    power by the site symmetry order — a factor of 24 for an atom at the
    origin of a cubic group — so images closer than ``tolerance`` are
    merged.

    Returns
    -------
    list[list[int]]
        For each input site, the indices into the returned coordinate array
        are implicit: the function returns, per site, the list of *unique*
        image indices as offsets into a flat expansion the caller rebuilds
        with :func:`orbit`.
    """
    return [list(range(len(orbit(p, operations, tolerance)))) for p in positions]


def orbit(
    position: Sequence[float],
    operations: Sequence[SymmetryOperation],
    tolerance: float = MERGE_TOLERANCE,
) -> np.ndarray:
    """Unique symmetry images of one site, as an ``(m, 3)`` array.

    ``m`` is the Wyckoff multiplicity of the site: the group order divided
    by the order of the site's own symmetry.
    """
    base = np.mod(np.asarray(position, dtype=float), 1.0)
    images: list[np.ndarray] = []
    for op in operations:
        candidate = np.mod(op.R @ base + op.t, 1.0)
        duplicate = False
        for existing in images:
            difference = np.abs(candidate - existing)
            difference = np.minimum(difference, 1.0 - difference)
            if np.all(difference < tolerance):
                duplicate = True
                break
        if not duplicate:
            images.append(candidate)
    return np.asarray(images, dtype=float)


__all__ = [
    "MAX_ORDER",
    "ClosedGroup",
    "MERGE_TOLERANCE",
    "SymmetryError",
    "SymmetryOperation",
    "close_group",
    "expand_positions",
    "orbit",
    "parse_xyz",
]
