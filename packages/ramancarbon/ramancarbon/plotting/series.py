"""One curve on a plot, with everything that can be said about it.

A :class:`Series` carries its own data and its own overrides. Anything it
does not override comes from the figure's :class:`~ramancarbon.plotting.
style.PlotStyle`, so setting a palette once styles twenty curves, and
setting ``colour`` on one of them makes that one different without
touching the rest. That inheritance is what makes a plot style reusable.

Series also carry the two transformations people apply while looking at
data and then forget they applied: a vertical **offset** and a
**normalisation**. Both are recorded on the object rather than baked into
the numbers, so the exported data and the axis label can say what was
done — an offset waterfall whose y axis still claims to be counts is a
figure that misleads its own author six months later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional, Sequence

import numpy as np

Kind = Literal["line", "scatter", "step", "bar", "fill", "stick", "errorbar"]


@dataclass
class Series:
    """One curve, its style overrides and its provenance."""

    x: np.ndarray
    y: np.ndarray
    label: str = ""
    kind: Kind = "line"

    # -- per-series style overrides (None means "inherit") -----------
    colour: Optional[str] = None
    line_width: Optional[float] = None
    line_style: Optional[Any] = None
    marker: Optional[str] = None
    marker_size: Optional[float] = None
    marker_every: Optional[int] = None
    alpha: Optional[float] = None
    z_order: Optional[float] = None
    fill_to: Optional[float] = None
    """Fill between the curve and this y value."""
    fill_alpha: float = 0.18

    # -- data-level modifiers ----------------------------------------
    scale: float = 1.0
    """Multiply y before drawing. A ×5 magnified region is normal in a
    Raman figure and the label should say so; :meth:`display_label` does."""
    shift: float = 0.0
    """Add to y before drawing, on top of any automatic offset."""
    x_shift: float = 0.0
    error: Optional[np.ndarray] = None
    """Symmetric y uncertainty, drawn when ``kind='errorbar'``."""

    on_secondary_y: bool = False
    visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"«{self.label or 'serie'}»: x tiene {self.x.shape} e y "
                f"{self.y.shape}"
            )
        if self.error is not None:
            self.error = np.asarray(self.error, dtype=float)
            if self.error.shape != self.y.shape:
                raise ValueError(
                    f"«{self.label or 'serie'}»: el error no tiene la forma de y"
                )

    # -- drawing-time data -------------------------------------------
    def normalised(self, mode: Optional[str]) -> np.ndarray:
        """``y`` after the figure's normalisation, before offset and scale."""
        y = np.asarray(self.y, dtype=float)
        if not mode or y.size == 0:
            return y
        if mode == "max":
            peak = float(np.nanmax(np.abs(y)))
            return y / peak if peak > 0 else y
        if mode == "minmax":
            low, high = float(np.nanmin(y)), float(np.nanmax(y))
            return (y - low) / (high - low) if high > low else y * 0.0
        if mode == "0-100":
            low, high = float(np.nanmin(y)), float(np.nanmax(y))
            return 100.0 * (y - low) / (high - low) if high > low else y * 0.0
        if mode == "area":
            from ..core.compat import trapezoid

            area = float(abs(trapezoid(y, self.x)))
            return y / area if area > 0 else y
        raise ValueError(f"normalización desconocida: {mode!r}")

    def drawn(self, mode: Optional[str], offset: float) -> tuple[np.ndarray, np.ndarray]:
        """The ``(x, y)`` actually plotted, after every modifier."""
        y = self.normalised(mode) * self.scale + self.shift + offset
        return self.x + self.x_shift, y

    def display_label(self) -> str:
        """The legend text, saying so when the curve has been scaled.

        A magnified trace that does not admit it is the most common way a
        figure lies without anyone intending it to.
        """
        if not self.label:
            return ""
        if abs(self.scale - 1.0) < 1e-9:
            return self.label
        factor = f"{self.scale:g}" if self.scale >= 1 else f"1/{1 / self.scale:g}"
        return f"{self.label} (×{factor})"

    def to_dict(self, include_data: bool = False) -> dict[str, Any]:
        """A plain dictionary. Data are excluded unless asked for, so a
        project file stays small and points at the source instead."""
        payload = {
            k: v for k, v in asdict(self).items()
            if k not in ("x", "y", "error")
        }
        if include_data:
            payload["x"] = self.x.tolist()
            payload["y"] = self.y.tolist()
            if self.error is not None:
                payload["error"] = self.error.tolist()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Series":
        known = set(cls.__dataclass_fields__)
        data = {k: v for k, v in payload.items() if k in known}
        data.setdefault("x", [])
        data.setdefault("y", [])
        return cls(**data)


def from_spectrum(spectrum, **overrides) -> Series:
    """A series from a :class:`~ramancarbon.core.spectrum.Spectrum`."""
    return Series(
        x=spectrum.shift, y=spectrum.intensity,
        label=overrides.pop("label", spectrum.name),
        metadata={"laser_nm": spectrum.laser_nm, "kind": "raman"},
        **overrides,
    )


def from_pattern(pattern, **overrides) -> Series:
    """A series from an :class:`~ramancarbon.xrd.pattern.Pattern`."""
    return Series(
        x=pattern.two_theta, y=pattern.intensity,
        label=overrides.pop("label", pattern.name),
        metadata={"wavelength": pattern.wavelength, "kind": "xrd"},
        **overrides,
    )


def stack_offsets(
    series: Sequence[Series],
    fraction: float,
    absolute: Optional[float] = None,
    normalisation: Optional[str] = None,
) -> list[float]:
    """Vertical offsets for a waterfall.

    The step is a fraction of the tallest series *after normalisation*,
    which is what makes a waterfall of raw spectra work at all: without
    it, one intense spectrum sets the step and the other nineteen collapse
    into a line.
    """
    if not series or (abs(fraction) < 1e-12 and absolute is None):
        return [0.0] * len(series)
    if absolute is not None:
        step = float(absolute)
    else:
        spans = []
        for item in series:
            y = item.normalised(normalisation) * item.scale
            if y.size:
                spans.append(float(np.nanmax(y) - np.nanmin(y)))
        step = fraction * (max(spans) if spans else 1.0)
    return [index * step for index in range(len(series))]


__all__ = ["Kind", "Series", "from_pattern", "from_spectrum", "stack_offsets"]
