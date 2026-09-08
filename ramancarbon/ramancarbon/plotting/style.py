"""Every knob a figure has, in one object.

The point of collecting them here rather than passing keyword arguments
around is that a figure is then a *value*: it can be saved to a project
file, reloaded a year later, applied to a different dataset, or handed to
a colleague, and produce the same picture. Specialised plotting software
works this way for the same reason.

Two decisions worth stating.

**Journal presets carry real column widths.** A single-column figure in
most chemistry journals is 3.25 inches (8.3 cm) and a double-column one is
7 inches (17.8 cm), and a figure drawn at 6 inches and then shrunk to fit
has 8-point labels that arrive as 4-point. The presets set the size and
the font sizes together so the text is the right size *on the printed
page*, which is the only place it matters.

**Colours default to a colour-blind-safe sequence.** Roughly one man in
twelve cannot separate the red and green that a default matplotlib cycle
puts next to each other, and a figure is read by more people than made it.
The sequence here also stays distinguishable in greyscale, because theses
still get printed in black and white.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal, Optional, Sequence

#: Colour-blind-safe categorical sequence (Okabe–Ito), reordered so the
#: first two are also the furthest apart in luminance for greyscale print.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)

#: A high-contrast sequence for dark backgrounds (talks, posters).
BRIGHT: tuple[str, ...] = (
    "#4FC3F7", "#FF8A65", "#81C784", "#BA68C8",
    "#FFD54F", "#4DD0E1", "#F06292", "#E0E0E0",
)

#: Greyscale, for the journals that still charge for colour.
GREYS: tuple[str, ...] = (
    "#000000", "#5a5a5a", "#8c8c8c", "#b4b4b4",
    "#2f2f2f", "#6f6f6f", "#9d9d9d", "#c8c8c8",
)

PALETTES: dict[str, tuple[str, ...]] = {
    "okabe-ito": OKABE_ITO,
    "brillante": BRIGHT,
    "grises": GREYS,
}

#: Marker sequence chosen so that filled and open alternate: adjacent
#: series stay distinguishable when they overlap.
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

#: Line-style sequence, for when colour is not available at all.
DASHES: tuple[str, ...] = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)))

LineStyle = Literal["-", "--", "-.", ":", "none"]
Scale = Literal["linear", "log", "symlog", "sqrt"]


def restore_tuples(klass, payload: dict[str, Any]) -> dict[str, Any]:
    """Turn the lists JSON produced back into the tuples the class declares.

    JSON has no tuple, so ``(10.0, 80.0)`` comes back as ``[10.0, 80.0]``
    and a style read from a project file compares unequal to the one that
    was written — which turns "has anything changed since I saved?" into a
    question nobody can answer. Only fields whose annotation is a tuple are
    converted; a genuine list field stays a list.
    """
    out = dict(payload)
    for name, spec in klass.__dataclass_fields__.items():
        value = out.get(name)
        if isinstance(value, list) and "tuple" in str(spec.type).lower():
            out[name] = tuple(value)
    return out


@dataclass
class AxisStyle:
    """One axis: its scale, its limits, its ticks and its label."""

    label: str = ""
    scale: Scale = "linear"
    limits: Optional[tuple[Optional[float], Optional[float]]] = None
    """``(low, high)``; either may be ``None`` to autoscale that end."""
    invert: bool = False
    """Reverse the direction. Raman shift axes are sometimes drawn this
    way, and infrared ones almost always are."""
    tick_direction: Literal["in", "out", "inout"] = "in"
    """``"in"`` by default: ticks pointing inwards is the convention in
    every physics and chemistry journal, and matplotlib's default is not."""
    major_ticks: Optional[float] = None
    """Spacing between labelled ticks. ``None`` lets matplotlib choose."""
    minor_ticks: Optional[int] = None
    """Number of minor intervals between major ticks."""
    tick_format: Optional[str] = None
    """A format string such as ``"%.2f"`` or ``"%g"``."""
    show_minor: bool = True
    grid: bool = False
    grid_minor: bool = False
    log_base: float = 10.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SecondaryAxis:
    """A second axis showing the same data in different units.

    Real diffraction and spectroscopy figures carry these routinely — 2θ
    with d-spacing above it, Raman shift with absolute wavelength,
    potential against two reference electrodes — and drawing one by hand
    means getting a non-linear transform right every time.
    """

    kind: str
    """One of :data:`~ramancarbon.plotting.transforms.TRANSFORMS`."""
    label: str = ""
    parameter: float = 0.0
    """The constant the transform needs: the wavelength in Å for
    2θ ↔ d, the laser line in nm for shift ↔ wavelength, the reference
    electrode offset in V."""
    ticks: Optional[Sequence[float]] = None
    """Explicit tick positions in the SECONDARY units. Given none, a
    round set is chosen — which matters, because the transform is
    non-linear and evenly spaced ticks on one axis are ugly on the other."""


@dataclass
class LegendStyle:
    """Where the legend goes and what it looks like."""

    show: bool = True
    location: str = "best"
    columns: int = 1
    frame: bool = False
    font_size: Optional[float] = None
    title: str = ""
    outside: bool = False
    """Place it to the right of the axes. For a waterfall of twenty
    spectra this is the only workable option."""


@dataclass
class PlotStyle:
    """The complete description of how a figure should look."""

    # -- canvas ------------------------------------------------------
    width_in: float = 6.4
    height_in: float = 4.2
    dpi: int = 150
    facecolor: Optional[str] = None
    transparent: bool = False

    # -- typography --------------------------------------------------
    font_family: str = "sans-serif"
    font_size: float = 10.0
    label_size: Optional[float] = None
    tick_size: Optional[float] = None
    title_size: Optional[float] = None
    use_mathtext: bool = True

    # -- lines and marks ---------------------------------------------
    palette: str = "okabe-ito"
    line_width: float = 1.4
    line_style: LineStyle = "-"
    marker: str = "none"
    marker_size: float = 4.0
    marker_every: int = 1
    """Draw a marker every N points. Markers on a 3000-point spectrum are
    a solid band; this is how you get a few dozen instead."""
    alpha: float = 1.0
    cycle_markers: bool = False
    cycle_dashes: bool = False

    # -- axes --------------------------------------------------------
    x: AxisStyle = field(default_factory=AxisStyle)
    y: AxisStyle = field(default_factory=AxisStyle)
    secondary_x: Optional[SecondaryAxis] = None
    secondary_y: Optional[SecondaryAxis] = None
    spine_width: float = 0.9
    hide_spines: tuple[str, ...] = ()
    """Which of ``top``, ``right``, ``bottom``, ``left`` to remove."""
    tick_length: float = 3.5
    tick_width: float = 0.9

    # -- composition -------------------------------------------------
    title: str = ""
    legend: LegendStyle = field(default_factory=LegendStyle)
    offset: float = 0.0
    """Vertical offset between successive series, as a fraction of the
    tallest one. Non-zero turns an overlay into a waterfall, which is how
    a series of twenty spectra is actually shown."""
    offset_absolute: Optional[float] = None
    """An offset in data units, overriding :attr:`offset`."""
    normalise: Optional[str] = None
    """Per-series normalisation before plotting: ``"max"``, ``"area"``,
    ``"minmax"``, ``"0-100"``, or ``None``. Applies to the drawing only —
    the underlying data are never modified."""
    annotate_peaks: bool = False
    peak_label_format: str = "{:.0f}"
    tight: bool = True

    def resolved(self, name: str) -> float:
        """A font size, falling back to :attr:`font_size`."""
        value = getattr(self, name)
        return float(self.font_size if value is None else value)

    @property
    def colours(self) -> tuple[str, ...]:
        return PALETTES.get(self.palette, OKABE_ITO)

    def colour(self, index: int) -> str:
        colours = self.colours
        return colours[index % len(colours)]

    def marker_for(self, index: int) -> str:
        if self.cycle_markers:
            return MARKERS[index % len(MARKERS)]
        return self.marker

    def dash_for(self, index: int):
        if self.cycle_dashes:
            return DASHES[index % len(DASHES)]
        return self.line_style

    def replace(self, **changes) -> "PlotStyle":
        """A copy with some fields changed."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """A plain dictionary, for saving in a project file."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlotStyle":
        """Rebuild from :meth:`to_dict`, tolerating missing keys.

        Tolerant on purpose: a project file written by an older version
        must still open, and the fields it does not know about take their
        defaults rather than raising.
        """
        payload = dict(payload)
        nested = {
            "x": AxisStyle,
            "y": AxisStyle,
            "legend": LegendStyle,
            "secondary_x": SecondaryAxis,
            "secondary_y": SecondaryAxis,
        }
        for key, klass in nested.items():
            value = payload.get(key)
            if isinstance(value, dict):
                fields = {f for f in klass.__dataclass_fields__}
                clean = restore_tuples(klass, value)
                payload[key] = klass(**{k: v for k, v in clean.items() if k in fields})
        payload = restore_tuples(cls, payload)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


# -- presets -----------------------------------------------------------

#: Column widths in inches, from the journals' own author guidance.
COLUMN_WIDTHS: dict[str, tuple[float, float]] = {
    "acs": (3.25, 7.00),
    "rsc": (3.26, 6.73),
    "elsevier": (3.54, 7.48),
    "nature": (3.50, 7.20),
    "aps": (3.375, 6.75),
    "wiley": (3.35, 6.90),
}


def journal(
    name: str = "acs", double: bool = False, height_ratio: float = 0.72
) -> PlotStyle:
    """A style sized for a journal's column.

    The font sizes are set so that the text is 7–8 pt *after* the figure is
    placed at its natural size — which is what makes the difference between
    a readable figure and one whose axis labels arrive at 4 pt because it
    was drawn at 6 inches and shrunk to 3.25.
    """
    if name not in COLUMN_WIDTHS:
        raise ValueError(
            f"revista desconocida: {name!r}; disponibles: "
            + ", ".join(sorted(COLUMN_WIDTHS))
        )
    single, wide = COLUMN_WIDTHS[name]
    width = wide if double else single
    return PlotStyle(
        width_in=width,
        height_in=width * height_ratio,
        dpi=600,
        font_size=8.0,
        label_size=8.0,
        tick_size=7.0,
        title_size=8.0,
        line_width=1.0,
        spine_width=0.7,
        tick_width=0.7,
        tick_length=2.8,
        marker_size=3.0,
        legend=LegendStyle(font_size=7.0, frame=False),
    )


def presentation() -> PlotStyle:
    """Large text and thick lines, for a projector."""
    return PlotStyle(
        width_in=10.0,
        height_in=6.0,
        dpi=150,
        font_size=16.0,
        label_size=18.0,
        tick_size=15.0,
        title_size=19.0,
        line_width=2.6,
        spine_width=1.6,
        tick_width=1.6,
        tick_length=6.0,
        marker_size=8.0,
        legend=LegendStyle(font_size=15.0),
    )


def poster() -> PlotStyle:
    """Bigger still, and on the bright palette."""
    style = presentation()
    return style.replace(
        width_in=14.0, height_in=9.0, font_size=22.0, label_size=24.0,
        tick_size=20.0, title_size=26.0, line_width=3.4, palette="brillante",
        legend=LegendStyle(font_size=20.0),
    )


def thesis() -> PlotStyle:
    """A4 text width, 10 pt, the shape most theses want."""
    return PlotStyle(
        width_in=6.3, height_in=4.2, dpi=300, font_size=10.0, label_size=11.0,
        tick_size=9.0, line_width=1.2,
    )


def greyscale() -> PlotStyle:
    """No colour at all: greys, cycled dashes and cycled markers."""
    return PlotStyle(
        palette="grises", cycle_dashes=True, cycle_markers=True,
        marker_every=40, dpi=600,
    )


def waterfall(spacing: float = 0.25) -> PlotStyle:
    """Series stacked with a vertical offset, normalised to their maxima.

    The standard way to show a series of spectra — a temperature run, a
    line scan, a set of samples — and unusable without the normalisation,
    because one intense spectrum flattens the rest.
    """
    return PlotStyle(
        offset=spacing,
        normalise="max",
        legend=LegendStyle(outside=True, frame=False),
        y=AxisStyle(label="Intensidad (u. a., desplazada)", show_minor=False),
    )


#: Named presets offered in the interface and the CLI.
PRESETS: dict[str, str] = {
    "predeterminado": "Pantalla, tamaño medio",
    "acs": "Columna sencilla ACS (3.25 in, 600 ppp)",
    "acs-doble": "Doble columna ACS (7.0 in)",
    "rsc": "Columna sencilla RSC (3.26 in)",
    "elsevier": "Columna sencilla Elsevier (3.54 in)",
    "nature": "Columna sencilla Nature (3.50 in)",
    "aps": "Columna sencilla APS (3.375 in)",
    "wiley": "Columna sencilla Wiley (3.35 in)",
    "tesis": "Ancho de texto A4, 10 pt",
    "presentacion": "Proyector: texto grande, líneas gruesas",
    "poster": "Póster",
    "grises": "Sin color: grises, trazos y símbolos alternados",
    "cascada": "Cascada con desplazamiento vertical",
}


def preset(name: str) -> PlotStyle:
    """One of :data:`PRESETS` by name."""
    if name in ("predeterminado", "", None):
        return PlotStyle()
    if name.endswith("-doble"):
        return journal(name[: -len("-doble")], double=True)
    if name in COLUMN_WIDTHS:
        return journal(name)
    builders = {
        "tesis": thesis,
        "presentacion": presentation,
        "poster": poster,
        "grises": greyscale,
        "cascada": waterfall,
    }
    if name not in builders:
        raise ValueError(
            f"preajuste desconocido: {name!r}; disponibles: "
            + ", ".join(sorted(PRESETS))
        )
    return builders[name]()


__all__ = [
    "BRIGHT",
    "COLUMN_WIDTHS",
    "DASHES",
    "GREYS",
    "MARKERS",
    "OKABE_ITO",
    "PALETTES",
    "PRESETS",
    "AxisStyle",
    "LegendStyle",
    "PlotStyle",
    "SecondaryAxis",
    "greyscale",
    "journal",
    "poster",
    "presentation",
    "preset",
    "restore_tuples",
    "thesis",
    "waterfall",
]
