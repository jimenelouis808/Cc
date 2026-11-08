"""A configuration of adsorbed molecules, as a file and as a picture.

The figures these papers carry — Kowalczyk's "snapshot of hydrogen
adsorbed in the idealized hexagonal bundle", Levesque's equivalent — are
one instantaneous configuration from the production run, drawn with the
framework. This module turns what
:func:`~carbonforge.adsorption.gcmc.run_gcmc` keeps into that.

**Each adsorbed site is one molecule, not one atom**, and the file says
so rather than leaving it to be assumed. The model has a single
Lennard-Jones centre per H2 (see
:mod:`~carbonforge.adsorption.potentials`), so there is no second
hydrogen to place and inventing one would be drawing a bond length the
simulation never had. The default symbol is ``H`` because that is what
makes a viewer render it as a small pale sphere, which is the picture
everyone expects; ``X`` is offered for anyone who would rather the file
could not be mistaken for atomic coordinates.

Framework and adsorbate are told apart by **ASE tag**: 0 for the
framework, 1 for the molecules. Tags survive an extxyz round trip, so
the distinction reaches VESTA, OVITO or a Blender importer without a
sidecar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms

#: Tag values written onto the combined structure.
FRAMEWORK_TAG: int = 0
ADSORBATE_TAG: int = 1

#: Projections, as (azimuth, elevation) in degrees for
#: ``Axes3D.view_init``. Named after the **axis looked down**, not after
#: what that axis means: "the tube axis" is a fact about the builder and
#: not about the file, and a schwarzite has no axis at all. An earlier
#: version called these "eje", "lado" and "esquina" and had them wrong —
#: in :func:`~carbonforge.adsorption.bundle.levesque_bundle` the tubes
#: run along **y**, so the view that shows the hexagonal packing and the
#: hydrogen inside each tube is ``"y"``, which that naming had labelled
#: as the side view.
#:
#: Three because one projection of a three-dimensional network hides
#: whichever channel runs along the line of sight, which is why the
#: papers print more than one.
VIEWS: dict[str, tuple[float, float]] = {
    "x": (0.0, 0.0),
    "y": (-90.0, 0.0),
    "z": (-90.0, 90.0),
    "general": (-60.0, 25.0),
}

#: What ``view="todas"`` draws: the three axes, without the oblique one,
#: which is the layout a paper uses.
AXIS_VIEWS: tuple[str, ...] = ("x", "y", "z")


def adsorbed_atoms(
    framework: Atoms, positions: np.ndarray, symbol: str = "H",
) -> Atoms:
    """Framework plus one site per adsorbed molecule, in one ``Atoms``.

    Parameters
    ----------
    framework
        The rigid adsorbent. Its cell and periodicity are carried over.
    positions
        ``(m, 3)`` molecule centres in Å, as
        :attr:`~carbonforge.adsorption.gcmc.GCMCResult.snapshots` holds
        them.
    symbol
        What to call an adsorbed site. ``H`` draws as expected in every
        viewer; ``X`` is ASE's dummy species and cannot be mistaken for
        a real atom. Either way one site is one **molecule**.

    Returns
    -------
    ase.Atoms
        Framework first, molecules after, tagged
        :data:`FRAMEWORK_TAG` and :data:`ADSORBATE_TAG`, with
        ``info["adsorbate_sites"]`` and a note saying a site is a
        molecule.
    """
    sites = np.asarray(positions, dtype=float).reshape(-1, 3)
    combined = framework.copy()
    combined.set_tags([FRAMEWORK_TAG] * len(combined))
    if len(sites):
        guests = Atoms(symbol * len(sites) if len(symbol) == 1
                       else [symbol] * len(sites),
                       positions=sites,
                       cell=framework.cell, pbc=framework.pbc)
        guests.set_tags([ADSORBATE_TAG] * len(guests))
        combined += guests
    combined.info["adsorbate_sites"] = int(len(sites))
    combined.info["adsorbate_note"] = (
        "one site per MOLECULE, not per atom: this model has a single "
        "Lennard-Jones centre per H2")
    return combined


def write_snapshot(
    framework: Atoms, positions: np.ndarray, path: str | Path,
    symbol: str = "H",
) -> Path:
    """Write one configuration as extxyz, tags and all."""
    from ase.io import write

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atoms = adsorbed_atoms(framework, positions, symbol=symbol)
    note = atoms.info.pop("adsorbate_note")
    # The note is prose and extxyz's comment line is key=value pairs;
    # ASE writes it as a quoted string, which some readers choke on. The
    # count is a number and survives.
    write(str(path), atoms)
    atoms.info["adsorbate_note"] = note
    return path


def plot_snapshot(
    framework: Atoms, positions: np.ndarray, view: str = "esquina",
    figsize: tuple[float, float] = (5.0, 5.0),
):
    """Draw one configuration: framework as a mesh, molecules as spheres.

    The framework is drawn small and pale and the adsorbate large and
    saturated, which is the opposite of how a structure viewer weights
    them by default and is what makes the picture legible: the question
    the figure answers is *where the hydrogen went*, and a 2600-molecule
    cloud inside 8704 carbons is invisible if both are drawn alike.

    Parameters
    ----------
    framework, positions
        As :func:`adsorbed_atoms`.
    view
        A key of :data:`VIEWS` — ``"x"``, ``"y"`` or ``"z"`` for the
        projection down that axis, ``"general"`` for an oblique one —
        or ``"todas"`` for the three axes side by side. Which axis
        shows what depends on the builder: for the Levesque bundle the
        tubes run along y.
    figsize
        Of a single panel; the three-view figure is three times wide.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    sites = np.asarray(positions, dtype=float).reshape(-1, 3)
    wanted = list(AXIS_VIEWS) if view == "todas" else [view]
    for name in wanted:
        if name not in VIEWS:
            raise ValueError(
                f"unknown view {name!r}; this module offers "
                f"{', '.join(VIEWS)} or 'todas'.")

    width = figsize[0] * len(wanted)
    figure = plt.figure(figsize=(width, figsize[1]), facecolor="#fcfcfb")
    # mplot3d reserves a wide margin for axes nobody is drawing here,
    # which left the structure occupying about a third of each panel.
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.94,
                           wspace=0.0)
    carbon = framework.get_positions()
    for index, name in enumerate(wanted):
        axis = figure.add_subplot(1, len(wanted), index + 1,
                                  projection="3d")
        axis.set_facecolor("#fcfcfb")
        axis.scatter(carbon[:, 0], carbon[:, 1], carbon[:, 2],
                     s=3, c="#898781", alpha=0.35, linewidths=0,
                     depthshade=False)
        if len(sites):
            axis.scatter(sites[:, 0], sites[:, 1], sites[:, 2],
                         s=26, c="#2a78d6", alpha=0.9, linewidths=0,
                         depthshade=True)
        azimuth, elevation = VIEWS[name]
        axis.view_init(elev=elevation, azim=azimuth)
        axis.set_axis_off()
        # Equal aspect, or a cubic cell renders as a slab: matplotlib's
        # 3D box defaults to (4, 4, 3) and squashes z by a quarter.
        axis.set_box_aspect((1, 1, 1))
        edges = np.diag(np.asarray(framework.cell, dtype=float))
        span = 0.5 * float(edges.max())
        middle = 0.5 * edges
        for setter, centre in zip((axis.set_xlim, axis.set_ylim,
                                   axis.set_zlim), middle):
            setter(centre - span, centre + span)
        axis.set_title(f"por {name}" if name in AXIS_VIEWS else name,
                       fontsize=9, color="#52514e", pad=-4)
    return figure
