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

# Loaded by path rather than imported as a package, so the sibling
# import has to be made to work rather than assumed.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import styles as style_defs


def load_bundle(xyz_path: str, json_path: str | None = None):
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
    if not bonds:
        bonds = _bonds_from_distance(positions)
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


def colour_keys(bundle: dict, colour_by: str) -> tuple[list, list[int]]:
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
    elif colour_by == "uniform":
        return [0], [0] * n
    else:
        raise ValueError(f"unknown colour mode {colour_by!r}.")
    lookup = {key: index for index, key in enumerate(keys)}
    return keys, [lookup[value] for value in per_atom]


def atom_radii(bundle: dict, colour_by: str, scale: float = 1.0
               ) -> list[float]:
    """One display radius per atom, in Å.

    Element radii when the colouring is by element -- a hydrogen and a
    gold atom drawn the same size is a picture that misleads about which
    is which -- and one radius throughout otherwise, which is right for
    an all-carbon lattice and keeps a ring-coloured render reading as a
    lattice rather than as a chemistry diagram.
    """
    if colour_by == "element":
        return [style_defs.element_radius(symbol) * scale
                for symbol in bundle["symbols"]]
    return [style_defs.DEFAULT_ELEMENT_RADIUS * scale
            for _ in bundle["positions"]]
