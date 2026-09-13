"""Minimal matplotlib-based 3D viewer.

The goal is quick QA / debugging plots — not publication-quality renderings.
For that, export to XYZ / CIF and use VESTA, OVITO or ASE's ``ase gui``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

from ..topology.graph import build_bond_graph

#: Element colours, on the CPK convention every structure viewer uses --
#: carbon dark, nitrogen blue, oxygen red, sulphur yellow. Public because
#: the GUI viewer draws its own scene and must agree with this one: two
#: pictures of the same structure in the same program that disagree about
#: what colour nitrogen is are worse than either alone.
#:
#: Covers every element with a covalent radius, so a dichalcogenide or a
#: transition-metal decoration is coloured rather than falling back to
#: grey along with everything else.
ELEMENT_COLOURS: dict[str, str] = {
    # sp2 carbon and its substitutional dopants
    "C": "#2b2b2b", "N": "#3050f8", "B": "#ffb5b5", "S": "#ffff30",
    "P": "#ff8000", "H": "#eeeeee", "O": "#ff0d0d",
    # halogens, for fluorinated and halogenated surfaces
    "F": "#90e050", "Cl": "#1ff01f", "Br": "#a62929", "I": "#940094",
    # group 14, reached by substitution from carbon
    "Si": "#f0c8a0", "Ge": "#668f8f", "Sn": "#668080",
    # chalcogens and pnictogens, reached from oxygen and nitrogen
    "Se": "#ffa100", "Te": "#d47a00", "As": "#bd80e3",
    # dichalcogenide metals
    "Mo": "#54b5b5", "W": "#2194d6", "Nb": "#73c2c9", "Ta": "#4da6ff",
    "V": "#a6a6ab", "Ti": "#bfc2c7", "Zr": "#94e0e0", "Hf": "#4dc2ff",
    # decoration and catalysis metals
    "Pt": "#d0d0e0", "Fe": "#e06633", "Co": "#f090a0", "Ni": "#50d050",
    "Cu": "#c88033", "Zn": "#7d80b0", "Mn": "#9c7ac7", "Al": "#bfa6a6",
}

#: Backwards-compatible alias for the private name this table had.
_ELEMENT_COLORS = ELEMENT_COLOURS
_ELEMENT_SIZES = {"C": 40, "N": 40, "B": 45, "S": 55, "P": 55, "H": 20, "O": 40}


def _setup_figure(atoms: Atoms, figsize: tuple[float, float]):
    # Lazy import so the module stays usable even when matplotlib is missing.
    import matplotlib.pyplot as plt  # noqa: WPS433
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    colors = [ELEMENT_COLOURS.get(s, "#888888") for s in symbols]
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
    view: tuple[float, float] | None = None,
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
