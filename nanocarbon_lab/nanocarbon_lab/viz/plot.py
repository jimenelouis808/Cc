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


def _setup_figure(atoms: Atoms, figsize: tuple[float, float]):
    # Lazy import so the module stays usable even when matplotlib is missing.
    import matplotlib.pyplot as plt  # noqa: WPS433
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

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

    g = build_bond_graph(atoms)
    for i, j in g.edges:
        ax.plot(
            [positions[i, 0], positions[j, 0]],
            [positions[i, 1], positions[j, 1]],
            [positions[i, 2], positions[j, 2]],
            color="#555555",
            linewidth=0.6,
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
        f"{atoms.info.get('structure_type', 'structure')} "
        f"({len(atoms)} atoms, {atoms.get_chemical_formula()})"
    )
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
