"""The figure: series plus a style, rendered onto matplotlib axes.

A :class:`Plot` is data and intent; drawing is what happens at the end.
That ordering is what lets the same object be saved to a project file,
re-rendered at a different size for a different journal, exported as a
CSV of exactly the curves that are on screen, or handed to the GUI to be
edited interactively.

Everything the drawing does is derived from the style and the series, and
nothing is hard-coded — which is the difference between a plotting helper
and a plot engine. The one place that judgement is exercised on the user's
behalf is tick placement on secondary axes, and that is delegated to
:mod:`ramancarbon.plotting.transforms` where it can be tested.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .series import Series, stack_offsets
from .style import PlotStyle, restore_tuples
from .transforms import TRANSFORMS, secondary_ticks

#: Formats :meth:`Plot.save` will write.
IMAGE_FORMATS = (".png", ".pdf", ".svg", ".eps", ".ps", ".tif", ".tiff", ".jpg")


@dataclass
class Marker:
    """A vertical or horizontal reference line with an optional label."""

    position: float
    axis: str = "x"
    label: str = ""
    colour: Optional[str] = None
    style: Any = "--"
    width: float = 0.8
    alpha: float = 0.7


@dataclass
class Band:
    """A shaded range, for a fitting window or an excluded region."""

    low: float
    high: float
    axis: str = "x"
    label: str = ""
    colour: Optional[str] = None
    alpha: float = 0.12


@dataclass
class Annotation:
    """A piece of text at a point, optionally with an arrow."""

    x: float
    y: float
    text: str
    arrow_to: Optional[tuple[float, float]] = None
    size: Optional[float] = None
    colour: Optional[str] = None
    rotation: float = 0.0
    ha: str = "center"
    va: str = "bottom"


@dataclass
class Inset:
    """A zoomed copy of the same data in a corner of the axes."""

    x_limits: tuple[float, float]
    y_limits: Optional[tuple[float, float]] = None
    position: tuple[float, float, float, float] = (0.58, 0.55, 0.38, 0.38)
    """``(left, bottom, width, height)`` in axes fractions."""
    frame: bool = True
    ticks: bool = True


@dataclass
class Plot:
    """A complete figure: series, style and decorations."""

    series: list[Series] = field(default_factory=list)
    style: PlotStyle = field(default_factory=PlotStyle)
    markers: list[Marker] = field(default_factory=list)
    bands: list[Band] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    inset: Optional[Inset] = None
    name: str = "figura"

    # -- building ----------------------------------------------------
    def add(self, series: Series | Sequence[Series]) -> "Plot":
        """Append one series or many. Returns self, so calls chain."""
        if isinstance(series, Series):
            self.series.append(series)
        else:
            self.series.extend(series)
        return self

    def mark(self, *positions: float, axis: str = "x", **kwargs) -> "Plot":
        for position in positions:
            self.markers.append(Marker(position=position, axis=axis, **kwargs))
        return self

    def shade(self, low: float, high: float, **kwargs) -> "Plot":
        self.bands.append(Band(low=low, high=high, **kwargs))
        return self

    def annotate(self, x: float, y: float, text: str, **kwargs) -> "Plot":
        self.annotations.append(Annotation(x=x, y=y, text=text, **kwargs))
        return self

    @property
    def visible(self) -> list[Series]:
        return [s for s in self.series if s.visible]

    # -- drawing -----------------------------------------------------
    def offsets(self) -> list[float]:
        """The vertical offset applied to each visible series."""
        return stack_offsets(
            self.visible, self.style.offset, self.style.offset_absolute,
            self.style.normalise,
        )

    def draw(self, ax) -> None:
        """Render onto an existing matplotlib ``Axes``."""
        style = self.style
        offsets = self.offsets()
        twin = None

        for index, (item, offset) in enumerate(zip(self.visible, offsets)):
            target = ax
            if item.on_secondary_y:
                twin = twin or ax.twinx()
                target = twin
            self._draw_series(target, item, index, offset)

        self._draw_bands(ax)
        self._draw_markers(ax)
        self._apply_axes(ax)
        self._draw_annotations(ax)
        if style.annotate_peaks:
            self._label_peaks(ax, offsets)
        self._draw_legend(ax)
        if self.inset is not None:
            # The connector lines drawn by indicate_inset_zoom are artists
            # on the parent axes, and autoscale counts them: without
            # freezing the limits first, a corner inset drags the y axis
            # down to make room for a line that means nothing.
            frozen = (ax.get_xlim(), ax.get_ylim())
            self._draw_inset(ax)
            ax.set_xlim(*frozen[0])
            ax.set_ylim(*frozen[1])
        if style.title:
            ax.set_title(style.title, fontsize=style.resolved("title_size"))

    def _draw_series(self, ax, item: Series, index: int, offset: float) -> None:
        style = self.style
        x, y = item.drawn(style.normalise, offset)
        colour = item.colour or style.colour(index)
        common = {
            "color": colour,
            "alpha": item.alpha if item.alpha is not None else style.alpha,
            "label": item.display_label() or None,
            "zorder": item.z_order if item.z_order is not None else 2 + index * 0.01,
        }
        width = item.line_width if item.line_width is not None else style.line_width
        marker = item.marker if item.marker is not None else style.marker_for(index)
        marker = None if marker in ("none", "", None) else marker
        size = item.marker_size if item.marker_size is not None else style.marker_size
        every = item.marker_every if item.marker_every is not None else style.marker_every
        dash = item.line_style if item.line_style is not None else style.dash_for(index)

        if item.kind == "scatter":
            ax.plot(x, y, linestyle="none", marker=marker or "o",
                    markersize=size, **common)
        elif item.kind == "step":
            ax.step(x, y, where="mid", linewidth=width, linestyle=dash, **common)
        elif item.kind == "bar":
            widths = np.diff(x, prepend=x[0] - (x[1] - x[0]) if x.size > 1 else 1.0)
            ax.bar(x, y, width=np.abs(widths) * 0.9, **common)
        elif item.kind == "stick":
            base = offset
            ax.vlines(x, base, y, linewidth=width, **common)
        elif item.kind == "errorbar":
            ax.errorbar(
                x, y, yerr=item.error, linewidth=width, linestyle=dash,
                marker=marker, markersize=size, capsize=2.0, elinewidth=width * 0.7,
                markevery=every, **common,
            )
        else:
            ax.plot(x, y, linewidth=width, linestyle=dash, marker=marker,
                    markersize=size, markevery=every, **common)

        if item.fill_to is not None:
            ax.fill_between(x, item.fill_to, y, color=colour,
                            alpha=item.fill_alpha, linewidth=0, zorder=1)

    def _draw_bands(self, ax) -> None:
        for index, band in enumerate(self.bands):
            colour = band.colour or self.style.colour(index)
            span = ax.axvspan if band.axis == "x" else ax.axhspan
            span(band.low, band.high, color=colour, alpha=band.alpha,
                 linewidth=0, zorder=0, label=band.label or None)

    def _draw_markers(self, ax) -> None:
        for marker in self.markers:
            colour = marker.colour or "#888888"
            line = ax.axvline if marker.axis == "x" else ax.axhline
            line(marker.position, color=colour, linestyle=marker.style,
                 linewidth=marker.width, alpha=marker.alpha, zorder=1)
            if marker.label:
                if marker.axis == "x":
                    ax.annotate(
                        marker.label, xy=(marker.position, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(2, -4), textcoords="offset points",
                        fontsize=self.style.resolved("tick_size") * 0.85,
                        color=colour, rotation=90, ha="left", va="top",
                    )
                else:
                    ax.annotate(
                        marker.label, xy=(1.0, marker.position),
                        xycoords=("axes fraction", "data"),
                        xytext=(-4, 2), textcoords="offset points",
                        fontsize=self.style.resolved("tick_size") * 0.85,
                        color=colour, ha="right", va="bottom",
                    )

    def _draw_annotations(self, ax) -> None:
        for note in self.annotations:
            arrow = (
                {"arrowstyle": "->", "color": note.colour or "#555555",
                 "linewidth": 0.8}
                if note.arrow_to else None
            )
            ax.annotate(
                note.text,
                xy=note.arrow_to or (note.x, note.y),
                xytext=(note.x, note.y),
                fontsize=note.size or self.style.resolved("tick_size"),
                color=note.colour or "inherit" if note.colour else None,
                rotation=note.rotation, ha=note.ha, va=note.va,
                arrowprops=arrow,
            )

    def _label_peaks(self, ax, offsets: Sequence[float]) -> None:
        """Label the maxima of each visible series.

        Deliberately simple — the prominence threshold is a fraction of
        the range, not a significance test — because this is a *drawing*
        aid. Anything that has to be right about which peaks are real
        comes from the analysis modules, which do the statistics.
        """
        from scipy.signal import find_peaks as _find

        style = self.style
        for index, (item, offset) in enumerate(zip(self.visible, offsets)):
            x, y = item.drawn(style.normalise, offset)
            if y.size < 5:
                continue
            span = float(np.nanmax(y) - np.nanmin(y))
            threshold = span * 0.08 if span > 0 else 0.0
            indices, _ = _find(y, prominence=threshold or None)
            for position in indices:
                ax.annotate(
                    style.peak_label_format.format(x[position]),
                    xy=(x[position], y[position]),
                    xytext=(0, 3), textcoords="offset points",
                    fontsize=style.resolved("tick_size") * 0.8,
                    color=item.colour or style.colour(index),
                    ha="center", va="bottom", rotation=90,
                )

    def _apply_axes(self, ax) -> None:
        style = self.style
        for axis_style, axis, setter, scaler in (
            (style.x, ax.xaxis, ax.set_xlabel, ax.set_xscale),
            (style.y, ax.yaxis, ax.set_ylabel, ax.set_yscale),
        ):
            if axis_style.label:
                setter(axis_style.label, fontsize=style.resolved("label_size"))
            if axis_style.scale == "sqrt":
                # Common on diffraction intensity axes: it shows the weak
                # reflections without a logarithm's exaggeration of the
                # background, and matplotlib has no built-in for it.
                scaler("function", functions=(_sqrt_forward, _sqrt_inverse))
            elif axis_style.scale != "linear":
                scaler(axis_style.scale)
            if axis_style.limits:
                low, high = axis_style.limits
                if axis is ax.xaxis:
                    ax.set_xlim(left=low, right=high)
                else:
                    ax.set_ylim(bottom=low, top=high)
            if axis_style.invert:
                (ax.invert_xaxis if axis is ax.xaxis else ax.invert_yaxis)()

        # After both axes have their scales and limits, so the square-root
        # locator can see the range it has to span.
        for axis_style, axis in ((style.x, ax.xaxis), (style.y, ax.yaxis)):
            _apply_ticks(ax, axis, axis_style, style)

        for name in style.hide_spines:
            if name in ax.spines:
                ax.spines[name].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(style.spine_width)
        ax.tick_params(
            which="major", direction=style.x.tick_direction,
            length=style.tick_length, width=style.tick_width,
            labelsize=style.resolved("tick_size"), top=True, right=True,
        )
        ax.tick_params(
            which="minor", direction=style.x.tick_direction,
            length=style.tick_length * 0.55, width=style.tick_width * 0.8,
            top=True, right=True,
        )
        if style.x.grid or style.y.grid:
            ax.grid(True, which="major", alpha=0.25, linewidth=style.spine_width * 0.7)
        if style.x.grid_minor or style.y.grid_minor:
            ax.grid(True, which="minor", alpha=0.12,
                    linewidth=style.spine_width * 0.5)

        for spec, is_x in ((style.secondary_x, True), (style.secondary_y, False)):
            if spec is not None:
                self._draw_secondary(ax, spec, is_x)

    def _draw_secondary(self, ax, spec, is_x: bool) -> None:
        transform = TRANSFORMS.get(spec.kind)
        if transform is None:
            return
        limits = ax.get_xlim() if is_x else ax.get_ylim()
        positions, labels = secondary_ticks(
            transform, (float(limits[0]), float(limits[1])), spec.parameter,
            explicit=spec.ticks,
        )
        if not positions:
            return
        twin = ax.twiny() if is_x else ax.twinx()
        style = self.style
        if is_x:
            twin.set_xlim(*limits)
            twin.set_xticks(positions)
            twin.set_xticklabels([_tidy(v) for v in labels])
            twin.set_xlabel(spec.label or transform.label,
                            fontsize=style.resolved("label_size"))
        else:
            twin.set_ylim(*limits)
            twin.set_yticks(positions)
            twin.set_yticklabels([_tidy(v) for v in labels])
            twin.set_ylabel(spec.label or transform.label,
                            fontsize=style.resolved("label_size"))
        twin.tick_params(
            direction=style.x.tick_direction, length=style.tick_length,
            width=style.tick_width, labelsize=style.resolved("tick_size"),
        )
        for spine in twin.spines.values():
            spine.set_linewidth(style.spine_width)

    def _draw_legend(self, ax) -> None:
        legend = self.style.legend
        if not legend.show:
            return
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        kwargs: dict[str, Any] = {
            "frameon": legend.frame,
            "ncol": legend.columns,
            "fontsize": legend.font_size or self.style.resolved("tick_size"),
        }
        if legend.title:
            kwargs["title"] = legend.title
        if legend.outside:
            kwargs["loc"] = "upper left"
            kwargs["bbox_to_anchor"] = (1.02, 1.0)
            kwargs["borderaxespad"] = 0.0
        else:
            kwargs["loc"] = legend.location
        ax.legend(handles, labels, **kwargs)

    def _draw_inset(self, ax) -> None:
        spec = self.inset
        inset = ax.inset_axes(spec.position)
        offsets = self.offsets()
        for index, (item, offset) in enumerate(zip(self.visible, offsets)):
            if item.on_secondary_y:
                continue
            self._draw_series(inset, item, index, offset)
        inset.set_xlim(*spec.x_limits)
        if spec.y_limits:
            inset.set_ylim(*spec.y_limits)
        else:
            inset.relim()
            inset.autoscale_view(scalex=False)
        if not spec.ticks:
            inset.set_xticks([])
            inset.set_yticks([])
        inset.tick_params(labelsize=self.style.resolved("tick_size") * 0.8,
                          direction=self.style.x.tick_direction)
        for spine in inset.spines.values():
            spine.set_linewidth(self.style.spine_width * 0.8)
        if spec.frame:
            ax.indicate_inset_zoom(inset, edgecolor="#999999", linewidth=0.7)
        if inset.get_legend():
            inset.get_legend().remove()

    # -- output ------------------------------------------------------
    def figure(self):
        """Render into a fresh matplotlib ``Figure`` and return it."""
        from matplotlib.figure import Figure

        style = self.style
        figure = Figure(figsize=(style.width_in, style.height_in), dpi=style.dpi)
        if style.facecolor:
            figure.patch.set_facecolor(style.facecolor)
        with _rc(style):
            ax = figure.add_subplot(111)
            if style.facecolor:
                ax.set_facecolor(style.facecolor)
            self.draw(ax)
            if style.tight:
                figure.tight_layout()
        return figure

    def save(self, path: str | Path, dpi: Optional[int] = None) -> Path:
        """Write the figure. The format comes from the extension."""
        destination = Path(path)
        if destination.suffix.lower() not in IMAGE_FORMATS:
            raise ValueError(
                f"formato no soportado: {destination.suffix!r}; disponibles: "
                + ", ".join(IMAGE_FORMATS)
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure = self.figure()
        figure.savefig(
            destination,
            dpi=dpi or self.style.dpi,
            transparent=self.style.transparent,
            bbox_inches="tight" if self.style.legend.outside else None,
        )
        return destination

    def data_table(self) -> tuple[list[str], list[list[float]]]:
        """The numbers actually drawn, as columns.

        Exporting the plotted values rather than the source is deliberate.
        A figure with offsets and normalisation applied cannot be
        reproduced from the raw data without knowing exactly what was
        applied, and this is how a reader gets the curve they are looking
        at.
        """
        offsets = self.offsets()
        columns: list[str] = []
        data: list[np.ndarray] = []
        for item, offset in zip(self.visible, offsets):
            x, y = item.drawn(self.style.normalise, offset)
            name = item.label or f"serie{len(columns) // 2 + 1}"
            columns.extend([f"{name}_x", f"{name}_y"])
            data.extend([x, y])
        if not data:
            return [], []
        length = max(len(column) for column in data)
        rows = [
            [float(column[i]) if i < len(column) else math.nan for column in data]
            for i in range(length)
        ]
        return columns, rows

    def save_data(self, path: str | Path, separator: str = ",") -> Path:
        """Write :meth:`data_table` as text."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        columns, rows = self.data_table()
        lines = [f"# {self.name}", separator.join(columns)]
        for row in rows:
            lines.append(separator.join("" if math.isnan(v) else f"{v:.8g}"
                                        for v in row))
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination

    def to_dict(self, include_data: bool = False) -> dict[str, Any]:
        from dataclasses import asdict as _asdict

        return {
            "name": self.name,
            "style": self.style.to_dict(),
            "series": [s.to_dict(include_data) for s in self.series],
            "markers": [_asdict(m) for m in self.markers],
            "bands": [_asdict(b) for b in self.bands],
            "annotations": [_asdict(a) for a in self.annotations],
            "inset": _asdict(self.inset) if self.inset else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Plot":
        plot = cls(name=payload.get("name", "figura"))
        plot.style = PlotStyle.from_dict(payload.get("style", {}))
        plot.series = [Series.from_dict(s) for s in payload.get("series", [])]
        plot.markers = [Marker(**restore_tuples(Marker, m))
                        for m in payload.get("markers", [])]
        plot.bands = [Band(**restore_tuples(Band, b))
                      for b in payload.get("bands", [])]
        plot.annotations = [Annotation(**restore_tuples(Annotation, a))
                            for a in payload.get("annotations", [])]
        inset = payload.get("inset")
        if inset:
            plot.inset = Inset(**restore_tuples(Inset, inset))
        return plot


def _sqrt_forward(values):
    return np.sqrt(np.clip(np.asarray(values, dtype=float), 0.0, None))


def _sqrt_inverse(values):
    return np.square(np.asarray(values, dtype=float))


def _tidy(value: float) -> str:
    """A tick label without trailing zeros."""
    if abs(value) >= 1000 or (value != 0 and abs(value) < 0.01):
        return f"{value:.3g}"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def _sqrt_ticks(low: float, high: float, count: int = 6) -> list[float]:
    """Tick values that are round in the SQUARE ROOT of the data.

    An automatic locator on a square-root axis picks round *values* —
    0, 2000, 4000 — which bunch together at the top of the axis and
    undo the reason for using the scale. Round values of √y, squared,
    spread evenly across the axis, which is what the scale is for: showing
    the weak reflections of a diffraction pattern without a logarithm's
    exaggeration of the background.
    """
    low = max(float(low), 0.0)
    high = max(float(high), low + 1e-9)
    root_low, root_high = math.sqrt(low), math.sqrt(high)
    raw = (root_high - root_low) / max(count, 1)
    if raw <= 0:
        return []
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * magnitude
        if (root_high - root_low) / step <= count * 1.4:
            break
    value = math.ceil(root_low / step) * step
    ticks = []
    while value <= root_high + 1e-9:
        ticks.append(round(value * value, 6))
        value += step
    return ticks


def _apply_ticks(ax, axis, axis_style, style: PlotStyle) -> None:
    from matplotlib.ticker import (
        AutoMinorLocator,
        FixedLocator,
        FormatStrFormatter,
        MultipleLocator,
    )

    if axis_style.major_ticks:
        axis.set_major_locator(MultipleLocator(axis_style.major_ticks))
    elif axis_style.scale == "sqrt":
        low, high = (ax.get_xlim() if axis is ax.xaxis else ax.get_ylim())
        ticks = _sqrt_ticks(low, high)
        if ticks:
            axis.set_major_locator(FixedLocator(ticks))
    if axis_style.show_minor and axis_style.scale == "linear":
        axis.set_minor_locator(AutoMinorLocator(axis_style.minor_ticks or 2))
    if axis_style.tick_format:
        axis.set_major_formatter(FormatStrFormatter(axis_style.tick_format))


def _rc(style: PlotStyle):
    import matplotlib

    return matplotlib.rc_context(
        {
            "font.family": style.font_family,
            "font.size": style.font_size,
            "axes.labelsize": style.resolved("label_size"),
            "xtick.labelsize": style.resolved("tick_size"),
            "ytick.labelsize": style.resolved("tick_size"),
            "axes.titlesize": style.resolved("title_size"),
            "mathtext.default": "regular" if style.use_mathtext else "it",
            "axes.prop_cycle": matplotlib.rcParams["axes.prop_cycle"],
            "savefig.transparent": style.transparent,
        }
    )


__all__ = [
    "IMAGE_FORMATS",
    "Annotation",
    "Band",
    "Inset",
    "Marker",
    "Plot",
]
