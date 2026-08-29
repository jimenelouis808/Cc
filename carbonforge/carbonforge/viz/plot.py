"""Minimal matplotlib-based 3D viewer.

The goal is quick QA / debugging plots — not publication-quality renderings.
For that, export to XYZ / CIF and use VESTA, OVITO or ASE's ``ase gui``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms

from ..topology.graph import build_bond_graph


_ELEMENT_COLORS = {
    "C": "#2b2b2b",
    "N": "#3050f8",
    "B": "#ffb5b5",
    "S": "#ffff30",
    "P": "#ff8000",
    "H": "#eeeeee",
    "O": "#ff0d0d",
}
_ELEMENT_SIZES = {"C": 40, "N": 40, "B": 45, "S": 55, "P": 55, "H": 20, "O": 40}


def draw_structure_on_axes(
    atoms: Atoms,
    ax,
    show_bonds: bool = True,
    title: Optional[str] = None,
) -> None:
    """Draw ``atoms`` onto an existing 3D matplotlib axes.

    Shared by :func:`plot_structure` and by the Tkinter GUI, which embeds its
    own canvas. Bonds are drawn as a single ``Line3DCollection`` rather than
    one ``plot`` call per bond — for a few thousand bonds that is the
    difference between an instant redraw and several seconds.

    Parameters
    ----------
    atoms
        Structure to render.
    ax
        A matplotlib axes created with ``projection="3d"``.
    show_bonds
        Draw covalent bonds inferred from covalent radii.
    title
        Overrides the default auto-generated title.
    """
    from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: WPS433

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    colors = [_ELEMENT_COLORS.get(s, "#888888") for s in symbols]
    sizes = [_ELEMENT_SIZES.get(s, 30) for s in symbols]

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=colors,
        s=sizes,
        edgecolors="black",
        linewidths=0.3,
        depthshade=True,
    )

    if show_bonds and len(atoms) > 1:
        g = build_bond_graph(atoms)
        segments = [
            (positions[i], positions[j])
            for i, j in g.edges
            # Skip bonds wrapped across the periodic boundary: drawing them
            # straight would produce long lines slicing through the cell.
            if np.linalg.norm(positions[i] - positions[j]) < 2.0
        ]
        if segments:
            ax.add_collection3d(
                Line3DCollection(segments, colors="#555555", linewidths=0.6)
            )

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_zlabel("z (Å)")
    ax.set_box_aspect(
        (
            float(np.ptp(positions[:, 0])) or 1.0,
            float(np.ptp(positions[:, 1])) or 1.0,
            float(np.ptp(positions[:, 2])) or 1.0,
        )
    )
    ax.set_title(
        title
        if title is not None
        else (
            f"{atoms.info.get('structure_type', 'structure')} "
            f"({len(atoms)} atoms, {atoms.get_chemical_formula()})"
        )
    )


def _setup_figure(atoms: Atoms, figsize: tuple[float, float]):
    # Lazy import so the module stays usable even when matplotlib is missing.
    import matplotlib.pyplot as plt  # noqa: WPS433
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    draw_structure_on_axes(atoms, ax)
    return fig, ax


def plot_structure(atoms: Atoms, figsize: tuple[float, float] = (6.0, 6.0)):
    """Return a matplotlib ``Figure`` with atoms (colour-coded) and bonds.

    Parameters
    ----------
    atoms
        Structure to visualise.
    figsize
        Figure size in inches.
    """
    fig, _ = _setup_figure(atoms, figsize)
    return fig


def save_structure_png(
    atoms: Atoms,
    path: str | Path,
    figsize: tuple[float, float] = (6.0, 6.0),
    dpi: int = 150,
    view: Optional[tuple[float, float]] = None,
) -> Path:
    """Save a PNG rendering of the structure.

    Parameters
    ----------
    atoms
        Structure to render.
    path
        Output file path (``.png``).
    figsize
        Figure size in inches.
    dpi
        Raster resolution.
    view
        Optional ``(elev, azim)`` in degrees to set the camera angle.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    fig, ax = _setup_figure(atoms, figsize)
    if view is not None:
        ax.view_init(elev=view[0], azim=view[1])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out
