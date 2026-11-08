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

#: Colour per adsorbate, from the validated categorical order. Keyed by
#: the name a caller uses, so a second adsorbate gets its own slot
#: rather than the next hue somebody liked: these are checked for
#: colour-vision separation as a set, and inventing a sixth by eye
#: would break that for everyone who reads the figure.
#:
#: Pink for H2 because the framework is grey and the question the
#: figure answers is *where the hydrogen went* — the adsorbate has to
#: be the thing the eye lands on. It sits at 2.62:1 against the page,
#: under the 3:1 a thin mark would need, so the figure always carries a
#: label naming the species and the count: the marks are large filled
#: circles rather than hairlines, and the label is the relief.
ADSORBATE_COLOURS: dict[str, str] = {
    "H2": "#e87ba4",
    "CH4": "#2a78d6",
    "CO2": "#1baf7a",
    "N2": "#eda100",
    "O2": "#6250d6",
}

#: What an adsorbate with no entry above gets.
DEFAULT_COLOUR: str = ADSORBATE_COLOURS["H2"]

#: The framework is context, so muted ink and thin.
FRAMEWORK_COLOUR: str = "#898781"

#: Longest distance drawn as a bond, in Å. Graphitic C-C is 1.42 and
#: this workspace's relaxers accept up to 1.80, so 1.9 draws every real
#: bond and no second neighbour -- the nearest of those is 2.46.
BOND_CUTOFF: float = 1.9


def framework_bonds(atoms: Atoms, cutoff: float = BOND_CUTOFF) -> np.ndarray:
    """Bond segments to draw, as ``(n, 2, 3)``, none crossing the cell.

    A bond whose partner lies in the next image is drawn as **two
    stubs**, one leaving each side, rather than as a single straight
    line between the two stored positions -- which would slice a
    cell-length line through the picture. This is the fact that bit the
    structure builders in this workspace again and again: a periodic
    bond list names the pair and not the image, so anything that draws
    one has to take the minimum image itself.

    Dropping boundary bonds instead, which is what the quick viewer in
    :mod:`carbonforge.viz` does, leaves every tube in a bundle looking
    cut off at its ends. The stubs say "this continues", which is true.
    """
    from scipy.spatial import cKDTree

    edges = np.diag(np.asarray(atoms.cell, dtype=float))
    if len(atoms) < 2 or not np.all(edges > 0):
        return np.empty((0, 2, 3))
    positions = np.asarray(atoms.get_positions(), dtype=float) % edges
    tree = cKDTree(positions, boxsize=edges)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    if not len(pairs):
        return np.empty((0, 2, 3))

    first = positions[pairs[:, 0]]
    second = positions[pairs[:, 1]]
    delta = second - first
    delta -= edges * np.round(delta / edges)
    crossing = np.any(np.abs(second - first - delta) > 1e-9, axis=1)

    segments = [np.stack([first, first + delta], axis=1)]
    if crossing.any():
        segments.append(np.stack([second[crossing],
                                  second[crossing] - delta[crossing]],
                                 axis=1))
    return np.concatenate(segments, axis=0)


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
    framework: Atoms, positions: np.ndarray, view: str = "todas",
    figsize: tuple[float, float] = (5.0, 5.0),
    species: str = "H2", colour: str | None = None,
    bonds: bool = True, bond_cutoff: float = BOND_CUTOFF,
):
    """Draw one configuration: the framework bonded, the adsorbate on top.

    The framework is drawn as a thin bonded mesh in muted ink and the
    adsorbate as large saturated spheres. That is the opposite of how a
    structure viewer weights them by default, and it is what makes the
    picture legible: the question the figure answers is *where the
    adsorbate went*, and a 2600-molecule cloud inside 8704 carbons is
    invisible if both are drawn alike.

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
    species
        What the adsorbate is called. Picks the colour from
        :data:`ADSORBATE_COLOURS` and names it on the figure.
    colour
        Overrides that. Any matplotlib colour.
    bonds
        Draw the framework's bonds. Off gives the bare point cloud,
        which is faster on a very large cell and much harder to read.
    bond_cutoff
        Longest distance drawn as a bond, in Å.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    sites = np.asarray(positions, dtype=float).reshape(-1, 3)
    wanted = list(AXIS_VIEWS) if view == "todas" else [view]
    for name in wanted:
        if name not in VIEWS:
            raise ValueError(
                f"unknown view {name!r}; this module offers "
                f"{', '.join(VIEWS)} or 'todas'.")
    paint = colour or ADSORBATE_COLOURS.get(species, DEFAULT_COLOUR)
    segments = framework_bonds(framework, bond_cutoff) if bonds else None

    width = figsize[0] * len(wanted)
    figure = plt.figure(figsize=(width, figsize[1]), facecolor="#fcfcfb")
    # mplot3d reserves a wide margin for axes nobody is drawing here,
    # which left the structure occupying about a third of each panel.
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.90,
                           wspace=0.0)
    carbon = framework.get_positions()
    for index, name in enumerate(wanted):
        axis = figure.add_subplot(1, len(wanted), index + 1,
                                  projection="3d")
        axis.set_facecolor("#fcfcfb")
        if segments is not None and len(segments):
            # One collection rather than a call per bond: for the
            # thirteen thousand of a validation bundle that is the
            # difference between a redraw and a coffee.
            axis.add_collection3d(Line3DCollection(
                segments, colors=FRAMEWORK_COLOUR, linewidths=0.5,
                alpha=0.55))
        else:
            axis.scatter(carbon[:, 0], carbon[:, 1], carbon[:, 2],
                         s=3, c=FRAMEWORK_COLOUR, alpha=0.35,
                         linewidths=0, depthshade=False)
        if len(sites):
            axis.scatter(sites[:, 0], sites[:, 1], sites[:, 2],
                         s=26, c=paint, alpha=0.9, linewidths=0,
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

    # The label is not decoration: the adsorbate colour sits under 3:1
    # against the page, and a visible label naming what it is is the
    # relief that makes that legible rather than merely pretty.
    figure.text(0.5, 0.975,
                f"{len(sites)} moléculas de {species} sobre "
                f"{len(framework)} átomos de armazón",
                ha="center", va="top", fontsize=9, color="#52514e")
    figure.text(0.5, 0.012,
                f"cada punto {'rosa' if paint == DEFAULT_COLOUR else 'de color'}"
                f" es UNA MOLÉCULA, no un átomo",
                ha="center", va="bottom", fontsize=7.5, color="#898781")
    return figure
