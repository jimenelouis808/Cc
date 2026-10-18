"""Read a render bundle and decide what each atom's colour means.

The pure half of the Blender pipeline: **no ``bpy`` import**, so it runs
and is tested in an ordinary interpreter. :mod:`mesh_builder` needs
Blender and imports from here; :mod:`styles` is the other pure module and
holds the palettes this one looks colours up in.

The split exists because the colour decision is where the mistakes are.
Which slot an atom takes, whether the slot order is stable between runs,
which radius a gold atom gets beside a hydrogen, and what an atom with no
ring metadata falls back to are all answerable from the bundle alone --
and none of them was checkable while the code sat in a module that
cannot be imported without Blender.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Loaded by path rather than imported as a package, so the sibling
# import has to be made to work rather than assumed.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import styles as style_defs

#: How long a bond may be, in Angstrom, before it is treated as one
#: that left the cell. Above any real chemical bond and far below a
#: cell edge, so the test never has to be close.
MAX_REAL_BOND = 3.4

#: Fraction of a seam bond's true length the stub is drawn at. Half
#: puts the cut where the cell face is, which is where the structure
#: really does stop.
SEAM_STUB_FRACTION = 0.5


def split_seam_bonds(positions, bonds, cell, mode: str = "stub"):
    """Separate the bonds that leave the cell from the ones that do not.

    **This is the single worst-looking defect the pipeline had on a
    periodic structure, and it is not a rendering fault.** A bond list
    built under the minimum-image convention records the pair ``(i, j)``
    and throws away *which image* j was in, so drawn from the raw
    coordinates a bond across a cell face is a straight line as long as
    the cell. Measured on a 920-atom super-graphene cell: **70 of its
    1380 bonds, up to 41.1 A**, against a real bond length of 1.41 --
    five per cent of the bonds, drawn as girders across the whole
    picture. That is what "saturated with bonds" looks like.

    The cell is in the sidecar, so the image each bond reaches into can
    be recovered here: for each one, find the integer combination of
    lattice vectors that makes it shortest. Zero means an ordinary
    interior bond and anything else means it crossed a face.

    ``mode`` is ``"stub"`` (draw a half-bond pointing the way the
    structure continues, which is what a crystallographic viewer does at
    a cell face), ``"drop"`` (leave them out) or ``"keep"`` (draw them
    as they are -- the old behaviour, kept only so the defect can be
    reproduced).

    Returns ``(interior, stubs)``, the stubs as ``[atom, dx, dy, dz]``
    with a single index and an offset.
    """
    if cell is None or len(bonds) == 0:
        return list(bonds), []
    lattice = np.asarray(cell, dtype=float)
    if lattice.shape != (3, 3):                 # pragma: no cover - guard
        return list(bonds), []
    live = [axis for axis in range(3)
            if float(np.linalg.norm(lattice[axis])) > 1e-6]
    if not live:
        return list(bonds), []

    points = np.asarray(positions, dtype=float)
    pairs = np.asarray(bonds, dtype=int)
    raw = points[pairs[:, 1]] - points[pairs[:, 0]]

    # Every shift of -1, 0 or +1 along each live axis. One cell is
    # always enough: a bond longer than that is not a bond.
    offsets = [np.zeros(3)]
    for axis in live:
        offsets = [existing + sign * lattice[axis]
                   for existing in offsets for sign in (-1, 0, 1)]
    candidates = np.stack(offsets)                     # (k, 3)
    deltas = raw[:, None, :] - candidates[None, :, :]  # (n, k, 3)
    best = np.argmin(np.linalg.norm(deltas, axis=2), axis=1)
    short = deltas[np.arange(len(pairs)), best]
    crossed = np.linalg.norm(candidates[best], axis=1) > 1e-6

    if mode == "keep":
        return [list(map(int, pair)) for pair in pairs], []
    interior = [list(map(int, pair))
                for pair, is_seam in zip(pairs, crossed, strict=True)
                if not is_seam]
    if mode == "drop":
        return interior, []

    stubs: list[list[float]] = []
    for pair, is_seam, vector in zip(pairs, crossed, short, strict=True):
        if not is_seam:
            continue
        reach = vector * SEAM_STUB_FRACTION
        # Both ends are in the picture and both have a neighbour that is
        # not, so both get a stub.
        stubs.append([float(pair[0]), *(+reach)])
        stubs.append([float(pair[1]), *(-reach)])
    return interior, stubs


def load_bundle(xyz_path: str, json_path: str | None = None,
                seams: str = "stub"):
    """Read the structure and whatever metadata came with it.

    Parameters
    ----------
    xyz_path
        A plain ``.xyz``, as written by
        :func:`nanocarbon_lab.exports.xyz.write_xyz`.
    json_path
        The companion ``.json``, as written by
        :func:`nanocarbon_lab.exports.xyz.write_render_bundle` or
        :func:`nanocarbon_lab.scene.write_scene`. Without it bonds are
        guessed from a distance cutoff and there is no ring or item
        metadata, so only ``element`` and ``uniform`` colouring mean
        anything.

    Returns
    -------
    dict
        ``positions``, ``symbols``, ``bonds``, ``ring_sizes_per_atom``,
        ``item_per_atom``, ``labels`` and ``seam_stubs``. A dict rather
        than a tuple because this grew from three fields to seven and a
        positional return had already been unpacked in two places.
    """
    lines = Path(xyz_path).read_text().splitlines()
    n = int(lines[0])
    positions: list[tuple[float, float, float]] = []
    symbols: list[str] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        positions.append(tuple(float(value) for value in parts[1:4]))

    bonds: list[tuple[int, int]] = []
    ring_sizes_per_atom: list[list[int]] = [[] for _ in range(n)]
    item_per_atom = [0] * n
    labels: list[str] = []
    seam_stubs: list[list[float]] = []
    cell = None
    if json_path and Path(json_path).exists():
        bundle = json.loads(Path(json_path).read_text())
        bonds = [tuple(bond) for bond in bundle.get("bonds", [])]
        loaded = bundle.get("ring_sizes_per_atom")
        if loaded and len(loaded) == n:
            ring_sizes_per_atom = loaded
        items = bundle.get("item_per_atom")
        if items and len(items) == n:
            item_per_atom = [int(value) for value in items]
        labels = list(bundle.get("scene", {}).get("labels", []))
        seam_stubs = [list(stub) for stub in bundle.get("seam_stubs", [])]
        cell = bundle.get("cell")
    if not bonds:
        bonds = _bonds_from_distance(positions)
    elif not seam_stubs:
        # A periodic structure's own bonds cross its cell faces, and the
        # sidecar carries the cell, so they can be found and dealt with
        # here rather than drawn as girders across the picture.
        bonds, seam_stubs = split_seam_bonds(positions, bonds, cell, seams)
    return {
        "positions": positions,
        "symbols": symbols,
        "bonds": bonds,
        "ring_sizes_per_atom": ring_sizes_per_atom,
        "item_per_atom": item_per_atom,
        "labels": labels,
        # Half-bonds at a periodic seam: ``[atom, dx, dy, dz]``. One atom
        # each, not two, so they are kept apart from "bonds" -- anything
        # that reads a bond as a pair of indices would otherwise have to
        # learn about them.
        "seam_stubs": seam_stubs,
    }


def _bonds_from_distance(positions, cutoff: float = 1.8):
    """Fallback bond guesser for a bare ``.xyz`` with no sidecar.

    A fixed carbon-shaped cutoff, and it is the fallback for a reason:
    it is wrong for a 2.4 Å Mo-S bond and **actively dangerous on a
    composition**, where it would invent bonds between two structures
    that merely sit close together. The sidecar carries the real bond
    list precisely so this does not run.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree([tuple(p) for p in positions])
    return sorted(tree.query_pairs(r=cutoff))


def _dominant_ring(ring_sizes) -> int:
    """Which ring an atom should be coloured by.

    An atom on a trivalent surface sits on three rings; if any is
    non-hexagonal, the smallest non-hexagon wins, pentagons being the
    most visually informative (they mark the convex points and the caps).
    """
    for size in (5, 7, 8):
        if size in ring_sizes:
            return size
    return 6


#: Below this ratio of the first two singular values, a structure has
#: no longest axis to speak of and the principal one is decided by
#: floating-point noise. Measured: a C240 cage reads **1.0000** and a
#: coated spherical nanoparticle 1.0123, against 1.73 for a sheet, 3.01
#: for a row of three structures and 6.29 for a long net.
DEGENERATE_AXIS = 1.05


def depth_axis_quality(positions) -> float:
    """How well defined the structure's longest axis is.

    The ratio of the first two singular values. At 1.0 the two
    directions are equally long, so which one comes out first is
    numerical noise: a sphere has no longest axis, and grading a cage
    along "its own" one grades it along an arbitrary direction that a
    tiny perturbation rotates. Compare against :data:`DEGENERATE_AXIS`.
    """
    points = np.asarray(positions, dtype=float)
    if len(points) < 3:                         # pragma: no cover - tiny
        return 1.0
    centred = points - points.mean(axis=0)
    values = np.linalg.svd(centred, full_matrices=False)[1]
    if len(values) < 2 or values[1] <= 1e-12:   # pragma: no cover - flat
        return float("inf")
    return float(values[0] / values[1])


def depth_bands(positions, axis=None) -> list[int]:
    """Which depth band each atom falls in, along the longest axis.

    The structure's **own** longest axis rather than a fixed one: a
    sheet lying in xy and a tube standing in z should both get a
    gradient that runs across the whole thing, and picking z for both
    would give the sheet a single colour.

    Measured by principal axis rather than by bounding box, so a
    structure lying on a diagonal is still graded along its length.

    ``axis`` overrides that with an explicit direction, which is the
    answer for a round subject: a cage or a spherical particle has no
    longest axis (see :func:`depth_axis_quality`), so the principal one
    is arbitrary there and grading along the view direction -- which is
    what "depth" means to the eye anyway -- is both meaningful and
    reproducible. Left ``None`` the behaviour is unchanged.
    """
    points = np.asarray(positions, dtype=float)
    if len(points) < 2:                         # pragma: no cover - tiny
        return [0] * len(points)
    centred = points - points.mean(axis=0)
    if axis is not None:
        direction = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            raise ValueError("depth axis must not be the zero vector.")
        along = centred @ (direction / norm)
    else:
        # The first principal component is the direction of most extent.
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        along = centred @ axes[0]
    low, high = float(along.min()), float(along.max())
    span = max(high - low, 1e-9)
    bands = style_defs.DEPTH_BANDS
    scaled = (along - low) / span * (bands - 1)
    return [int(round(value)) for value in scaled]


def colour_keys(bundle: dict, colour_by: str,
                depth_axis=None) -> tuple[list, list[int]]:
    """The material keys for a colour mode, and one index per atom.

    Returns ``(keys, index_per_atom)`` where ``keys`` is the ordered list
    of distinct values the mode colours by -- ring sizes, element
    symbols, item indices, or a single placeholder -- and
    ``index_per_atom`` says which slot each atom uses. That is exactly
    the shape a Blender material list plus per-face ``material_index``
    needs, and keeping it a pure function of the bundle is what lets it
    be tested without Blender.
    """
    n = len(bundle["positions"])
    if colour_by == "ring":
        per_atom = [_dominant_ring(bundle["ring_sizes_per_atom"][i]
                                   if i < len(bundle["ring_sizes_per_atom"])
                                   else [])
                    for i in range(n)]
        keys = [6, 5, 7, 8]
    elif colour_by == "element":
        per_atom = list(bundle["symbols"])
        # Sorted so the slot order is stable between runs; a material
        # list whose order depends on which atom came first is a
        # difference nobody can see and everybody has to debug.
        keys = sorted(set(per_atom))
    elif colour_by == "structure":
        per_atom = list(bundle["item_per_atom"])
        keys = sorted(set(per_atom))
    elif colour_by == "depth":
        per_atom = depth_bands(bundle["positions"], depth_axis)
        # Every band, not only the occupied ones, so the ramp means the
        # same thing on a structure with a gap in it as on one without.
        keys = list(range(style_defs.DEPTH_BANDS))
    elif colour_by == "uniform":
        return [0], [0] * n
    else:
        raise ValueError(f"unknown colour mode {colour_by!r}.")
    lookup = {key: index for index, key in enumerate(keys)}
    return keys, [lookup[value] for value in per_atom]


def atom_radii(bundle: dict, colour_by: str, scale: float = 1.0,
               model: str = "ballstick") -> list[float]:
    """One drawn radius per atom, in Angstrom.

    The **model** decides first: a space-filling atom is drawn at its van
    der Waals radius whatever the colour mode, because that is what a
    space-filling model *is*. ``auto`` asks per atom, so a gold core
    comes out solid inside an open carbon shell.

    Within ball-and-stick the colour mode still matters: element radii
    when colouring by element -- a hydrogen and a gold atom drawn the
    same size is a picture that misleads about which is which -- and one
    radius throughout otherwise, which is right for an all-carbon
    lattice and keeps a ring-coloured render reading as a lattice rather
    than as a chemistry diagram.
    """
    symbols = bundle["symbols"]
    if model != "ballstick":
        return [style_defs.model_radius(symbol, model, scale)
                for symbol in symbols]
    if colour_by == "element":
        return [style_defs.element_radius(symbol) * scale
                for symbol in symbols]
    return [style_defs.DEFAULT_ELEMENT_RADIUS * scale
            for _ in bundle["positions"]]
