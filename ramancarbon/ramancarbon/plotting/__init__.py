"""A configurable plot engine shared by all four instruments.

The three objects worth knowing:

:class:`~ramancarbon.plotting.style.PlotStyle`
    Everything about how a figure looks — size, fonts, scales, limits,
    ticks, spines, colours, line widths, markers, legend, offsets — as one
    saveable value. Journal presets set size and font size *together*, so
    the text is the right size on the printed page.
:class:`~ramancarbon.plotting.series.Series`
    One curve, with per-curve overrides of anything in the style, plus the
    two modifiers people apply and then forget: a vertical offset and a
    normalisation. Both are recorded rather than baked in.
:class:`~ramancarbon.plotting.engine.Plot`
    The two together, plus markers, shaded bands, annotations and insets.
    It renders to axes, saves to any image format, and exports the numbers
    actually drawn.

The separation exists so a figure is a *value*: it can be saved in a
project file, reopened, restyled for a different journal, and re-rendered
identically.
"""

from __future__ import annotations

from .engine import Annotation, Band, Inset, Marker, Plot
from .series import Series, from_pattern, from_spectrum
from .style import PRESETS, AxisStyle, LegendStyle, PlotStyle, SecondaryAxis, preset
from .transforms import TRANSFORMS

__all__ = [
    "PRESETS",
    "TRANSFORMS",
    "Annotation",
    "AxisStyle",
    "Band",
    "Inset",
    "LegendStyle",
    "Marker",
    "Plot",
    "PlotStyle",
    "SecondaryAxis",
    "Series",
    "from_pattern",
    "from_spectrum",
    "preset",
]
