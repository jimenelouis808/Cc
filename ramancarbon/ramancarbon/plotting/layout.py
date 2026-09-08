"""Multi-panel figures: a grid of plots with panel letters.

Nearly every figure in a paper is several panels, and assembling them by
hand in a drawing program is where reproducibility goes to die — the
panels end up at different font sizes, the letters at different positions,
and re-running the analysis means redoing the assembly.

:class:`PanelFigure` takes a grid of :class:`~ramancarbon.plotting.engine.
Plot` objects, applies one style to all of them, labels them (a), (b),
(c)… and lays them out at a size that fits a real journal column. Sharing
an axis between panels removes the redundant tick labels, which is what
makes a four-panel figure fit in a column at all.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

from .engine import Plot
from .style import PlotStyle

LabelStyle = Literal["a", "A", "(a)", "(A)", "a)", "none"]


def panel_label(index: int, style: LabelStyle = "(a)") -> str:
    """``0 → "(a)"``, and the other conventions journals ask for."""
    if style == "none":
        return ""
    letter = string.ascii_lowercase[index % 26]
    if style in ("A", "(A)"):
        letter = letter.upper()
    if style.startswith("("):
        return f"({letter})"
    if style.endswith(")"):
        return f"{letter})"
    return letter


@dataclass
class PanelFigure:
    """A grid of plots that share a style and get panel letters."""

    plots: list[Plot] = field(default_factory=list)
    rows: int = 1
    columns: int = 1
    style: Optional[PlotStyle] = None
    """Applied to every panel. ``None`` leaves each plot's own style."""
    share_x: bool = False
    share_y: bool = False
    label_style: LabelStyle = "(a)"
    label_position: tuple[float, float] = (0.02, 0.96)
    label_size: Optional[float] = None
    label_weight: str = "bold"
    width_in: Optional[float] = None
    height_in: Optional[float] = None
    dpi: Optional[int] = None
    wspace: float = 0.22
    hspace: float = 0.22
    name: str = "figura"

    def add(self, plot: Plot) -> "PanelFigure":
        self.plots.append(plot)
        return self

    @property
    def panel_style(self) -> PlotStyle:
        return self.style or (self.plots[0].style if self.plots else PlotStyle())

    def _grid(self) -> tuple[int, int]:
        if self.rows * self.columns >= len(self.plots):
            return self.rows, self.columns
        # Fill out the grid rather than silently dropping panels.
        columns = max(self.columns, 1)
        rows = -(-len(self.plots) // columns)
        return rows, columns

    def figure(self):
        """Render every panel into one matplotlib ``Figure``."""
        from matplotlib.figure import Figure

        from .engine import _rc

        if not self.plots:
            raise ValueError("la figura no tiene ningún panel")
        rows, columns = self._grid()
        base = self.panel_style
        width = self.width_in or base.width_in * columns
        height = self.height_in or base.height_in * rows
        figure = Figure(figsize=(width, height), dpi=self.dpi or base.dpi)
        if base.facecolor:
            figure.patch.set_facecolor(base.facecolor)

        with _rc(base):
            axes: list[Any] = []
            for index, plot in enumerate(self.plots):
                shared_x = axes[0] if (self.share_x and axes) else None
                shared_y = axes[0] if (self.share_y and axes) else None
                ax = figure.add_subplot(
                    rows, columns, index + 1, sharex=shared_x, sharey=shared_y
                )
                if self.style is not None:
                    plot = _restyled(plot, self.style)
                plot.draw(ax)
                axes.append(ax)

                label = panel_label(index, self.label_style)
                if label:
                    ax.annotate(
                        label,
                        xy=self.label_position, xycoords="axes fraction",
                        fontsize=self.label_size or base.resolved("label_size"),
                        fontweight=self.label_weight, ha="left", va="top",
                    )

            # Redundant tick labels are what stop a four-panel figure from
            # fitting in a column. Only the outer panels keep theirs.
            for index, ax in enumerate(axes):
                row, column = divmod(index, columns)
                if self.share_x and row < rows - 1:
                    ax.tick_params(labelbottom=False)
                    ax.set_xlabel("")
                if self.share_y and column > 0:
                    ax.tick_params(labelleft=False)
                    ax.set_ylabel("")

            figure.subplots_adjust(wspace=self.wspace, hspace=self.hspace)
            if base.tight:
                figure.tight_layout()
        return figure

    def save(self, path: str | Path, dpi: Optional[int] = None) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure = self.figure()
        figure.savefig(
            destination,
            dpi=dpi or self.dpi or self.panel_style.dpi,
            transparent=self.panel_style.transparent,
            bbox_inches="tight",
        )
        return destination

    def save_data(self, folder: str | Path, separator: str = ",") -> list[Path]:
        """One data file per panel, named by its letter."""
        directory = Path(folder)
        directory.mkdir(parents=True, exist_ok=True)
        written = []
        for index, plot in enumerate(self.plots):
            letter = string.ascii_lowercase[index % 26]
            written.append(
                plot.save_data(directory / f"{self.name}_{letter}.csv", separator)
            )
        return written

    def to_dict(self, include_data: bool = False) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": self.columns,
            "share_x": self.share_x,
            "share_y": self.share_y,
            "label_style": self.label_style,
            "style": self.style.to_dict() if self.style else None,
            "plots": [p.to_dict(include_data) for p in self.plots],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PanelFigure":
        figure = cls(
            rows=payload.get("rows", 1),
            columns=payload.get("columns", 1),
            share_x=payload.get("share_x", False),
            share_y=payload.get("share_y", False),
            label_style=payload.get("label_style", "(a)"),
            name=payload.get("name", "figura"),
        )
        style = payload.get("style")
        if style:
            figure.style = PlotStyle.from_dict(style)
        figure.plots = [Plot.from_dict(p) for p in payload.get("plots", [])]
        return figure


def _restyled(plot: Plot, style: PlotStyle) -> Plot:
    """A copy of ``plot`` under a shared style, keeping its own labels.

    The axis labels, limits and scales belong to the panel, not to the
    figure: sharing a style must not relabel a diffraction panel with a
    voltammogram's axes.
    """
    merged = style.replace(
        x=plot.style.x, y=plot.style.y,
        secondary_x=plot.style.secondary_x, secondary_y=plot.style.secondary_y,
        title=plot.style.title,
        offset=plot.style.offset, offset_absolute=plot.style.offset_absolute,
        normalise=plot.style.normalise,
    )
    clone = Plot(
        series=plot.series, style=merged, markers=plot.markers,
        bands=plot.bands, annotations=plot.annotations, inset=plot.inset,
        name=plot.name,
    )
    return clone


def grid(
    plots: Sequence[Plot],
    columns: int = 2,
    style: Optional[PlotStyle] = None,
    **kwargs,
) -> PanelFigure:
    """A panel figure from a list of plots, filling rows first."""
    rows = -(-len(plots) // max(columns, 1))
    return PanelFigure(
        plots=list(plots), rows=rows, columns=columns, style=style, **kwargs
    )


__all__ = ["LabelStyle", "PanelFigure", "grid", "panel_label"]
