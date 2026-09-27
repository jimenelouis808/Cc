"""Reference peak lists in the ``.dif`` form the card databases export.

Two columns, 2-theta in degrees and relative intensity on an arbitrary
scale, one reflection per line. No structure, no cell, no hkl -- which
is exactly why they are worth reading: a card for a phase nobody has a
CIF for is often the only reference there is, and refusing it because it
is not a structure leaves the user comparing numbers by eye.

What a peak list can and cannot do is worth being clear about. It can be
matched against the peaks found in a pattern, position by position, with
the intensities as corroboration. It cannot be refined: a Rietveld needs
atoms, and no arrangement of them is implied by a list of lines. So this
returns something deliberately narrow, and the matching built on it
reports "compatible" rather than a weight fraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class DIFError(ValueError):
    """A .dif that cannot be read, and why."""


@dataclass(frozen=True)
class ReferenceLines:
    """A card's reflections: where, and how strong relative to each other."""

    name: str
    two_theta: np.ndarray
    intensity: np.ndarray
    source: str = ""

    def __len__(self) -> int:
        return int(self.two_theta.size)

    def within(self, low: float, high: float) -> ReferenceLines:
        """The lines inside a measured range.

        Lines outside it are not evidence either way, and counting them
        as missing is the same mistake as counting an undetectable
        reflection against a phase.
        """
        keep = (self.two_theta >= low) & (self.two_theta <= high)
        return ReferenceLines(self.name, self.two_theta[keep],
                              self.intensity[keep], self.source)

    def strongest(self, n: int = 5) -> list[tuple[float, float]]:
        order = np.argsort(self.intensity)[::-1][:n]
        return [(float(self.two_theta[i]), float(self.intensity[i]))
                for i in order]


def read_dif(path: str | Path, name: str | None = None) -> ReferenceLines:
    """Read one ``.dif`` card."""
    p = Path(path)
    angles: list[float] = []
    heights: list[float] = []
    for line in p.read_bytes().decode("latin-1").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            angle = float(parts[0])
            height = float(parts[1])
        except ValueError:
            continue
        angles.append(angle)
        heights.append(height)
    if len(angles) < 3:
        raise DIFError(
            f"{p.name}: sólo se leyeron {len(angles)} líneas de dos columnas; "
            "un .dif debería traer una reflexión por línea")
    return ReferenceLines(
        name=name or p.stem,
        two_theta=np.asarray(angles, dtype=float),
        intensity=np.asarray(heights, dtype=float),
        source=str(p),
    )


@dataclass(frozen=True)
class CardMatch:
    """How well a card's lines line up with the peaks that were found."""

    card: str
    matched: list[tuple[float, float, float]]
    """``(card angle, observed angle, card intensity)``."""
    missing_strong: list[tuple[float, float]]
    """Strong card lines with no peak: ``(angle, intensity)``."""
    covered_intensity: float
    """Share of the card's intensity, inside the measured range, that
    found a peak."""
    median_offset: float

    @property
    def verdict(self) -> str:
        if not self.matched:
            return "sin coincidencias"
        if self.covered_intensity >= 0.6 and not self.missing_strong:
            return "compatible"
        if self.covered_intensity >= 0.3:
            return "parcial"
        return "sólo líneas débiles"

    def __str__(self) -> str:
        faltan = (f"; faltan {len(self.missing_strong)} líneas fuertes"
                  if self.missing_strong else "")
        return (f"{self.card}: {self.verdict} — "
                f"{len(self.matched)} líneas emparejadas, "
                f"{100.0 * self.covered_intensity:.0f} % de la intensidad de "
                f"la ficha, desplazamiento mediano "
                f"{self.median_offset:+.3f}°{faltan}")


def match_card(card: ReferenceLines, peaks, window: float = 0.35,
               strong_fraction: float = 0.25) -> CardMatch:
    """Line up a card against found peaks.

    `strong_fraction` sets what counts as a line whose absence matters:
    a card's 5 %-of-maximum line missing says nothing, its 50 % line
    missing says a great deal. That asymmetry is the whole point --
    without it a card with a long tail of weak lines can "match" on the
    tail alone, which is how a phase gets accepted on its own noise.
    """
    observed = np.asarray([p.two_theta for p in peaks], dtype=float)
    if observed.size == 0 or len(card) == 0:
        return CardMatch(card.name, [], [], 0.0, 0.0)
    peak_max = float(card.intensity.max()) or 1.0
    matched: list[tuple[float, float, float]] = []
    missing: list[tuple[float, float]] = []
    offsets: list[float] = []
    seen = 0.0
    for angle, height in zip(card.two_theta, card.intensity, strict=True):
        distance = np.abs(observed - angle)
        nearest = int(np.argmin(distance))
        if distance[nearest] <= window:
            matched.append((float(angle), float(observed[nearest]),
                            float(height)))
            offsets.append(float(observed[nearest] - angle))
            seen += float(height)
        elif height >= strong_fraction * peak_max:
            missing.append((float(angle), float(height)))
    total = float(card.intensity.sum()) or 1.0
    return CardMatch(card.name, matched, missing, seen / total,
                     float(np.median(offsets)) if offsets else 0.0)


__all__ = ["CardMatch", "DIFError", "ReferenceLines", "match_card", "read_dif"]
