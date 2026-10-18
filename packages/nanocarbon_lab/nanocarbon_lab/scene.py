"""Several finished structures placed in one space, for a figure.

Every other module here returns **one** structure and takes care that it
is physical. This one puts two or more of them side by side so they can
be rendered together -- a nanotube beside the particle it grew on, three
sizes of a cage in a row, a core-shell particle next to the bare core --
and the result is a **picture, not a structure**.

That distinction is the whole module, and it is marked everywhere rather
than left to be inferred:

* A composition carries ``info["is_illustration"] = True`` and a
  ``structure_type`` of ``"composition"``. Nothing downstream should
  quantify it, export it to a DFT code or quote a density for it: the
  gap between two items is whatever the user dragged it to, and there is
  no relaxed interface anywhere in it. :func:`compose` says so in the
  report and the exporters repeat it.
* **Scaling an item is a lie about its bond lengths**, and it is offered
  anyway because an illustration sometimes needs a 10 Å cage to read
  beside a 200 Å particle. Any scale other than 1 is recorded per item,
  counted in the report, and warned about. What is never done is
  silently normalising sizes to make a layout look tidy.
* **Clearance is measured, not assumed.** The report gives the closest
  approach between every pair of items, so "they look separate" is
  replaced by a number. Two items interpenetrating is a legitimate
  thing to draw (a particle embedded in a matrix) and an easy thing to
  do by accident, so it is reported either way.

**Bonds and rings are re-indexed, never re-guessed.** Each item's own
``info["bonds"]`` and ``info["rings"]`` are shifted by that item's atom
offset, exactly as :func:`nanocarbon_lab.cell.supercell` does for a
repeat -- for the same reason, which that function's docstring states at
length: a renderer handed one item's connectivity for a whole scene draws
a fraction of the bonds and nobody notices from the atom count. Re-running
a bond guesser over the composition would be worse than either, because
it would **invent bonds between items** that are merely close.

Each atom's item index goes into ``Atoms.set_tags``, so a renderer can
colour by structure -- which is the colouring a composition actually
wants, and the reason this module and the Blender colour modes landed
together.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

import numpy as np
from ase import Atoms

#: Closest approach, in Å, below which two items are reported as touching
#: rather than merely close. A van der Waals contact between two separate
#: objects is about 3.4 Å; below 2 Å their atoms are interpenetrating,
#: whatever the layout was meant to be.
TOUCHING = 2.0

#: And below this they are close enough that a viewer will read them as
#: joined, which may well be the intention.
CLOSE = 4.0


@dataclass
class Item:
    """One structure placed in a scene.

    Attributes
    ----------
    atoms
        The structure. Never modified; :func:`compose` works on copies.
    label
        What to call it in the report and in the per-structure colouring.
    translation
        Where to put its centre, in Å. Applied after the rotation, and
        measured from the item's own **centre**, not from its first atom
        -- so the number means the same thing for every structure.
    rotation
        Intrinsic x-y-z Euler angles in **degrees**. Degrees because a
        figure is adjusted by eye and nobody nudges a structure by
        0.0873 radians.
    scale
        Uniform scale. ``1.0`` leaves the structure alone; anything else
        changes every bond length in it and is recorded as such. See the
        module docstring.
    """

    atoms: Atoms
    label: str = ""
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = 1.0

    def placed(self) -> Atoms:
        """A copy of the structure, rotated, scaled and moved.

        Order matters and is fixed here so every consumer agrees: scale,
        then rotate about the item's own centre, then translate. Rotating
        about the centre rather than the origin is what makes the
        rotation sliders behave -- about the origin, turning an item that
        sits 100 Å out swings it across the scene.
        """
        copy = self.atoms.copy()
        copy.set_positions(self.oriented()
                           + np.asarray(self.translation, dtype=float))
        # A composition has no lattice: the items were placed by hand and
        # whatever cell each carried describes a box around that item
        # alone. Keeping one would claim a periodicity the scene does not
        # have.
        copy.set_cell(None)
        copy.pbc = False
        return copy

    def oriented(self) -> np.ndarray:
        """The item's positions scaled and rotated about its own centre.

        What :meth:`placed` produces before the translation, and what the
        arrangement helpers have to measure: an item's footprint along x
        is a property of how it is *turned*, so measuring the unrotated
        structure would space a rod laid along the layout axis as though
        it were still standing up.
        """
        positions = np.asarray(self.atoms.get_positions(), dtype=float)
        if len(positions) == 0:                 # pragma: no cover - empty
            return positions
        centre = positions.mean(axis=0)
        return ((positions - centre) * float(self.scale)
                @ euler_matrix(self.rotation).T)

    def half_widths(self) -> np.ndarray:
        """Half the oriented bounding box along x, y and z, in Å."""
        oriented = self.oriented()
        if len(oriented) == 0:                  # pragma: no cover - empty
            return np.zeros(3)
        return 0.5 * (oriented.max(axis=0) - oriented.min(axis=0))

    def radius(self) -> float:
        """Half the longest diagonal of the oriented bounding box, Å.

        The right measure when neighbours may sit at any angle to each
        other -- a grid or a ring -- and the wrong one for a row, where
        the footprint along the row's own axis is what matters. See
        :func:`arrange`.
        """
        return float(np.linalg.norm(self.half_widths()))


def euler_matrix(angles_deg: Sequence[float]) -> np.ndarray:
    """Rotation matrix for intrinsic x-y-z Euler angles, in degrees.

    Written out rather than taken from scipy so the convention is
    visible: ``R = Rz @ Ry @ Rx``, applied to column vectors, which means
    a point is turned about x first. Any convention would do; an
    *unstated* one would not, because the GUI's three sliders and the
    command line's three numbers have to mean the same thing.
    """
    alpha, beta, gamma = (math.radians(float(a)) for a in angles_deg)
    cx, sx = math.cos(alpha), math.sin(alpha)
    cy, sy = math.cos(beta), math.sin(beta)
    cz, sz = math.cos(gamma), math.sin(gamma)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


@dataclass
class SceneReport:
    """What the composition came out as, measured.

    Attributes
    ----------
    n_items, n_atoms
        How many structures and how many atoms in total.
    labels
        One per item, in order.
    counts
        Atoms per item.
    spans
        The whole scene's bounding box, Å.
    clearances
        ``(i, j, distance)`` for every pair of items, closest first. The
        number is the shortest atom-to-atom distance between the two, so
        it is the one a viewer sees.
    scaled
        Items whose scale is not 1, as ``(label, scale)``. A scaled item
        has non-physical bond lengths.
    bonds, rings
        How many of each were carried over. A structure with no recorded
        bonds contributes none, and the report says so rather than the
        renderer silently guessing.
    warnings
        Everything worth saying before this is used for anything but a
        picture.
    """

    n_items: int
    n_atoms: int
    labels: list[str]
    counts: list[int]
    spans: tuple[float, float, float]
    clearances: list[tuple[int, int, float]]
    scaled: list[tuple[str, float]] = dataclass_field(default_factory=list)
    bonds: int = 0
    rings: int = 0
    warnings: list[str] = dataclass_field(default_factory=list)

    def summary(self) -> str:
        """The report as text, for the window and the command line."""
        lines = [
            f"composition of {self.n_items} structure(s), "
            f"{self.n_atoms} atoms",
            "  " + ", ".join(f"{label} ({count})" for label, count
                             in zip(self.labels, self.counts, strict=True)),
            f"  bounding box {self.spans[0]:.1f} x {self.spans[1]:.1f} x "
            f"{self.spans[2]:.1f} A",
            f"  carried {self.bonds} bonds and {self.rings} rings",
        ]
        if self.clearances:
            worst = self.clearances[0]
            lines.append(
                f"  closest pair: {self.labels[worst[0]]} and "
                f"{self.labels[worst[1]]} at {worst[2]:.2f} A")
        if self.scaled:
            lines.append("  scaled: " + ", ".join(
                f"{label} x{scale:g}" for label, scale in self.scaled))
        lines += [f"  ! {text}" for text in self.warnings]
        return "\n".join(lines)


def compose(items: Sequence[Item], vacuum: float = 10.0,
            ) -> tuple[Atoms, SceneReport]:
    """Put the items into one :class:`ase.Atoms`, and measure the result.

    Parameters
    ----------
    items
        The placed structures, in the order they should be numbered.
    vacuum
        Padding around the whole scene in the cell that is written, Å.
        The cell exists so a viewer that insists on one has something
        sensible; it is a bounding box and not a lattice, and ``pbc``
        stays off.

    Returns
    -------
    (atoms, report)
        ``atoms`` carries ``set_tags`` = the item index per atom,
        re-indexed ``info["bonds"]`` and ``info["rings"]``, and
        ``info["scene"]`` describing what went in.

    Raises
    ------
    ValueError
        If there is nothing to compose, or an item has no atoms.
    """
    if not items:
        raise ValueError("a composition needs at least one structure.")
    for index, item in enumerate(items):
        if len(item.atoms) == 0:
            raise ValueError(
                f"item {index} ({item.label or 'unnamed'}) has no atoms.")

    placed = [item.placed() for item in items]
    labels = [item.label or f"item {index + 1}"
              for index, item in enumerate(items)]

    combined = Atoms()
    tags: list[int] = []
    bonds: list[list[int]] = []
    rings: list[list[int]] = []
    dropped_bonds = [0] * len(items)
    dropped_rings = [0] * len(items)
    offset = 0
    offsets: list[int] = []
    for index, (item, piece) in enumerate(zip(items, placed, strict=True)):
        offsets.append(offset)
        combined += piece
        tags += [index] * len(piece)
        # Shifted, not re-guessed. Re-running a bond search over the
        # whole scene would invent bonds between items that merely sit
        # close together, which is the one thing a composition must not
        # claim.
        crossing = _crossing_pairs(item.atoms)
        for bond in item.atoms.info.get("bonds", []) or []:
            i, j = int(bond[0]), int(bond[1])
            if frozenset((i, j)) in crossing:
                dropped_bonds[index] += 1
                continue
            bonds.append([i + offset, j + offset])
        for ring in item.atoms.info.get("rings", []) or []:
            members = [int(atom) for atom in ring]
            if crossing and any(
                    frozenset(pair) in crossing
                    for pair in zip(members, members[1:] + members[:1],
                                    strict=True)):
                dropped_rings[index] += 1
                continue
            rings.append([atom + offset for atom in members])
        offset += len(piece)

    combined.set_tags(tags)
    positions = np.asarray(combined.get_positions(), dtype=float)
    spans = positions.max(axis=0) - positions.min(axis=0)
    combined.set_cell(np.diag(spans + float(vacuum)))
    combined.pbc = False
    combined.center()

    clearances = _clearances(placed)
    scaled = [(labels[index], float(item.scale))
              for index, item in enumerate(items)
              if abs(item.scale - 1.0) > 1e-9]
    warnings = _warnings(items, labels, placed, clearances, scaled,
                         dropped_bonds, dropped_rings)

    combined.info.update({
        "structure_type": "composition",
        # The flag every consumer should branch on. A composition is not
        # a structure: there is no relaxed interface between any two
        # items and the gaps are whatever the layout chose.
        "is_illustration": True,
        "quality_family": None,
        "bonds": bonds,
        "rings": rings,
        "ring_counts": _ring_counts(rings),
        "scene": {
            "labels": labels,
            "counts": [len(piece) for piece in placed],
            "offsets": offsets,
            "translations": [list(map(float, item.translation))
                             for item in items],
            "rotations": [list(map(float, item.rotation)) for item in items],
            "scales": [float(item.scale) for item in items],
            "sources": [str(item.atoms.info.get("source", ""))
                        for item in items],
            "structure_types": [str(item.atoms.info.get("structure_type", ""))
                                for item in items],
        },
    })

    report = SceneReport(
        n_items=len(items),
        n_atoms=len(combined),
        labels=labels,
        counts=[len(piece) for piece in placed],
        spans=tuple(float(value) for value in spans),  # type: ignore
        clearances=clearances,
        scaled=scaled,
        bonds=len(bonds),
        rings=len(rings),
        warnings=warnings,
    )
    return combined, report


def _clearances(placed: Sequence[Atoms]) -> list[tuple[int, int, float]]:
    """Closest atom-to-atom distance for every pair of items.

    The distance a viewer sees, not a centre-to-centre one: two long rods
    crossing at their ends are far apart by their centres and touching by
    their atoms.
    """
    from scipy.spatial import cKDTree

    out: list[tuple[int, int, float]] = []
    trees = [cKDTree(np.asarray(piece.get_positions(), dtype=float))
             for piece in placed]
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            distance, _ = trees[j].query(trees[i].data, workers=-1)
            out.append((i, j, float(np.min(distance))))
    out.sort(key=lambda entry: entry[2])
    return out


def _ring_counts(rings: Sequence[Sequence[int]]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for ring in rings:
        counts[len(ring)] = counts.get(len(ring), 0) + 1
    return counts


def _crossing_pairs(atoms: Atoms) -> set[frozenset[int]]:
    """Which of an item's recorded bonds leave through a cell face.

    A composition is **finite**: the cell is cleared, so the image the
    far end of such a bond actually lives in does not exist in the scene.
    Drawn from the raw coordinates the bond is then a straight line as
    long as the item's cell -- up to 12 Å across a (5,5) tube's own
    period, which is most of what ends up on screen. The pair is right
    and the *cell* is the information a minimum-image bond list throws
    away, so there is nothing to shift it by and it has to go.

    Returned as unordered pairs, so a ring can be tested against it
    without assuming which way round its atoms were recorded.
    """
    if not any(atoms.get_pbc()):
        return set()
    recorded = atoms.info.get("bonds") or []
    if len(recorded) == 0:
        return set()
    from .utils.geometry import bond_shifts

    shifts = bond_shifts(atoms, recorded)
    return {frozenset((int(bond[0]), int(bond[1])))
            for bond, shift in zip(recorded, shifts, strict=True)
            if np.any(np.asarray(shift) != 0)}


def _warnings(items: Sequence[Item], labels: Sequence[str],
              placed: Sequence[Atoms],
              clearances: Sequence[tuple[int, int, float]],
              scaled: Sequence[tuple[str, float]],
              dropped_bonds: Sequence[int] = (),
              dropped_rings: Sequence[int] = ()) -> list[str]:
    out = [
        "this is an ILLUSTRATION, not a structure: the items were placed "
        "by hand, there is no relaxed interface between any two of them, "
        "and nothing about the gaps is physics. Do not quantify it, "
        "relax it or hand it to a DFT code."
    ]
    if scaled:
        out.append(
            "scaled items have non-physical bond lengths: "
            + ", ".join(f"{label} x{scale:g}" for label, scale in scaled)
            + ". That is a legitimate choice for a figure and it is not a "
            "structure any more."
        )
    for i, j, distance in clearances:
        if distance < TOUCHING:
            out.append(
                f"{labels[i]} and {labels[j]} come within {distance:.2f} A, "
                "so their atoms interpenetrate. That may be what you want "
                "(a particle embedded in a matrix) and it is easy to do by "
                "accident."
            )
        elif distance < CLOSE:
            out.append(
                f"{labels[i]} and {labels[j]} are {distance:.2f} A apart, "
                "which a viewer will read as joined. There is no bond "
                "there -- the connectivity of each item is carried "
                "separately and nothing is drawn between them."
            )
    missing = [labels[index] for index, item in enumerate(items)
               if not item.atoms.info.get("bonds")]
    if missing:
        out.append(
            "no recorded bonds for: " + ", ".join(missing)
            + ". Those items will be drawn as atoms alone unless the "
            "renderer guesses -- which it does per scene, and a guess "
            "over a composition invents bonds between items."
        )
    crossed = [(labels[index], count)
               for index, count in enumerate(dropped_bonds) if count]
    if crossed:
        out.append(
            "dropped bonds that left a periodic cell: "
            + ", ".join(f"{label} ({count})" for label, count in crossed)
            + ". A composition is finite, so the image those bonds reached "
            "into is not in the scene; drawn anyway each one is a line as "
            "long as the item's own cell. Repeat a periodic item before "
            "composing it if you want its seam filled in."
        )
    ringless = [(labels[index], count)
                for index, count in enumerate(dropped_rings) if count]
    if ringless:
        out.append(
            "dropped rings that straddled a periodic cell: "
            + ", ".join(f"{label} ({count})" for label, count in ringless)
            + ". The atoms are correct and their coordinates are on "
            "opposite sides of the cell, so the ring cannot be filled "
            "without drawing a polygon across the figure."
        )
    return out


# ----------------------------------------------------------------------
# arrangements
# ----------------------------------------------------------------------
def arrange(items: Sequence[Item], layout: str = "row", gap: float = 8.0,
            axis: str = "x", columns: int = 0) -> list[Item]:
    """Place the items automatically, with a measured gap between them.

    Parameters
    ----------
    items
        The structures, with whatever rotations and scales they already
        carry -- those are respected, since the spacing is computed from
        each item's size *after* its own scale.
    layout
        ``"row"``, ``"grid"``, ``"circle"`` or ``"stack"``.
    gap
        Free space between neighbouring items, Å. It is the **gap**, not
        the pitch: asking for 8 Å leaves 8 Å of space whatever the items'
        sizes, which is what somebody laying out a figure means.
    axis
        For ``"row"`` and ``"stack"``, which Cartesian axis to lay them
        along. A row is horizontal by convention (x) and a stack
        vertical (z), and both are overridable.
    columns
        For ``"grid"``, how many per row. ``0`` picks a near-square
        arrangement.

    Returns
    -------
    list of Item
        New items with their translations set. The input is not modified.

    Notes
    -----
    **Which size is measured depends on the layout**, and using one for
    both was wrong by more than twice. A row's neighbours only ever meet
    along the row's own axis, so what has to clear is each item's
    footprint along *that* axis; a grid's or a ring's neighbours sit at
    an angle to each other, so there the bounding sphere is the honest
    bound. Measured: a C60, an 18 Å (6,6) tube and a C60 at ``gap=8``
    came out **17.9 Å** apart in a row and **23.3 Å** in a grid under the
    bounding-sphere rule -- the tube's diagonal, not its waist -- against
    8.0 Å for both under the per-axis one. A ring keeps the sphere and so
    keeps its margin, which is correct rather than generous: two items
    adjacent round a circle are turned relative to each other and their
    footprints along the line between them are not their x or y extents
    at all.

    Both measures are taken **after** each item's own rotation and
    scale, which is the other half of the same mistake: a rod turned to
    lie along the row has a quite different footprint from the same rod
    standing up, and the arrangement has to see the one it will be drawn
    with.
    """
    if not items:
        return []
    directions = {"x": np.array([1.0, 0.0, 0.0]),
                  "y": np.array([0.0, 1.0, 0.0]),
                  "z": np.array([0.0, 0.0, 1.0])}
    if axis not in directions:
        raise ValueError(f"axis must be x, y or z; got {axis!r}.")
    if layout in ("row", "stack"):
        # A stack is a row along z unless told otherwise, which is the
        # only difference between the two.
        name = axis if layout == "row" or axis != "x" else "z"
        along = "xyz".index(name)
        reach = [float(item.half_widths()[along]) for item in items]
        return [_moved(item, directions[name] * offset)
                for item, offset in zip(items, _line(reach, gap),
                                        strict=True)]
    if layout == "grid":
        return _grid(items, gap, columns)
    if layout == "circle":
        return _circle(items, [item.radius() for item in items], gap)
    raise ValueError(
        f"layout must be row, grid, circle or stack; got {layout!r}.")


def _line(reach: Sequence[float], gap: float) -> list[float]:
    """Centre offsets along a line, centred on zero.

    ``reach`` is each item's half-width along the line, so consecutive
    centres are ``reach[i] + gap + reach[i+1]`` apart and the free space
    between any two neighbours is exactly ``gap``.
    """
    offsets: list[float] = []
    cursor = 0.0
    for index, value in enumerate(reach):
        if index:
            cursor += reach[index - 1] + gap + value
        offsets.append(cursor)
    span = offsets[-1] if offsets else 0.0
    return [offset - 0.5 * span for offset in offsets]


def _moved(item: Item, vector: np.ndarray) -> Item:
    return Item(atoms=item.atoms, label=item.label,
                translation=tuple(float(value) for value in vector),
                rotation=item.rotation, scale=item.scale)


def _grid(items: Sequence[Item], gap: float, columns: int) -> list[Item]:
    """A grid on one pitch per axis, taken from the widest item.

    One pitch for the whole grid rather than per-row spacing, because a
    figure with ragged columns reads as a mistake even when every
    individual gap is right. But **one pitch per axis**, not one for
    both: a grid's neighbours are axis-aligned -- along a row they are x
    apart and down a column y apart -- so the footprint that has to
    clear is the per-axis one, exactly as for a row. Using the bounding
    sphere for a grid of three C60s and an 18 Å tube spaced them 23 Å
    apart for a requested 8.
    """
    count = len(items)
    columns = columns if columns > 0 else max(1, int(round(math.sqrt(count))))
    rows = math.ceil(count / columns)
    widths = np.asarray([item.half_widths() for item in items])
    pitch_x = 2.0 * float(widths[:, 0].max()) + gap
    pitch_y = 2.0 * float(widths[:, 1].max()) + gap
    out: list[Item] = []
    for index, item in enumerate(items):
        row, column = divmod(index, columns)
        x = (column - 0.5 * (columns - 1)) * pitch_x
        y = (0.5 * (rows - 1) - row) * pitch_y
        out.append(_moved(item, np.array([x, y, 0.0])))
    return out


def _circle(items: Sequence[Item], radii: Sequence[float],
            gap: float) -> list[Item]:
    count = len(items)
    if count == 1:
        return [_moved(items[0], np.zeros(3))]
    # The ring has to be wide enough that neighbours clear each other.
    # Two neighbours on a circle of radius R subtend 2*pi/n, so their
    # centres are 2 R sin(pi/n) apart; that has to cover both radii plus
    # the gap.
    needed = 2.0 * max(radii) + gap
    ring = needed / (2.0 * math.sin(math.pi / count))
    out: list[Item] = []
    for index, item in enumerate(items):
        angle = 2.0 * math.pi * index / count
        out.append(_moved(item, np.array([ring * math.cos(angle),
                                          ring * math.sin(angle), 0.0])))
    return out


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------
def write_scene(atoms: Atoms, path: str | Path,
                report: SceneReport | None = None,
                ) -> dict[str, Path]:
    """Write the composition in the formats its two destinations read.

    Parameters
    ----------
    atoms
        The composed structure, from :func:`compose`.
    path
        Output path **without** extension.
    report
        Written alongside as a text file when given, so the warnings
        travel with the files rather than living only in a terminal that
        has scrolled.

    Returns
    -------
    dict
        ``{"xyz": ..., "cif": ..., "json": ..., "notes": ...}`` --
        whichever were written.

    Notes
    -----
    Three formats because the two destinations want different things.
    **VESTA** reads the ``.xyz`` directly and the ``.cif`` when a box is
    wanted; the CIF is written as P1 with the bounding box as its cell,
    which is a container and not a lattice -- VESTA will offer to show
    periodic images of it and they would be meaningless.
    **Blender** reads the ``.xyz`` plus the ``.json`` sidecar, which is
    where the carried bonds, the rings and the **per-atom item index**
    live; that index is what makes per-structure colouring possible.
    """
    from .exports.xyz import write_render_bundle

    path = Path(path)
    xyz_path, json_path = write_render_bundle(atoms, path, cif=True)
    written = {"xyz": xyz_path, "json": json_path}
    cif_path = path.with_suffix(".cif")
    if cif_path.exists():
        written["cif"] = cif_path

    # The sidecar gains what a composition has and a single structure
    # does not: which item each atom belongs to.
    bundle = json.loads(json_path.read_text())
    bundle["scene"] = atoms.info.get("scene", {})
    bundle["item_per_atom"] = [int(tag) for tag in atoms.get_tags()]
    bundle["is_illustration"] = True
    json_path.write_text(json.dumps(bundle, indent=2))

    if report is not None:
        notes = path.with_suffix(".txt")
        notes.write_text(report.summary() + "\n")
        written["notes"] = notes
    return written


#: Keys a render bundle's sidecar carries that are worth restoring. Every
#: one of them is atom-indexed or derived from atom indices, which is why
#: they cannot be recovered from the coordinates and must travel in the
#: file beside them.
SIDECAR_KEYS = ("bonds", "rings", "ring_sizes_per_atom", "ring_counts",
                "structure_type", "terminal_atoms", "grafted_atoms",
                "defect_log", "quality_family")


def _restore_sidecar(atoms: Atoms, path: Path) -> None:
    """Put a render bundle's recorded metadata back onto ``atoms``.

    Silent when there is no sidecar, when it will not parse, or when its
    atom count disagrees with the file's: a bond list indexed for a
    different structure is worse than none, and this is a convenience
    rather than a contract. ``info`` already present wins, so reading a
    file that carried its own metadata (an ``extxyz``, a trajectory)
    never has it overwritten by a stale neighbour.
    """
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return
    try:
        recorded = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(recorded, dict):
        return
    if recorded.get("n_atoms") not in (None, len(atoms)):
        return
    for key in SIDECAR_KEYS:
        if key in recorded and key not in atoms.info:
            atoms.info[key] = recorded[key]


def items_from_files(paths: Sequence[str | Path],
                     labels: Sequence[str] | None = None) -> list[Item]:
    """Read structures from files into unplaced items.

    Anything ASE reads. ``info["source"]`` records the path, which the
    composition carries into its own ``info`` so a figure can say where
    each piece came from -- which matters more here than anywhere else,
    a composition being the one thing in this package assembled from
    several earlier results.

    **A render bundle's ``.json`` sidecar beside the file is read too**,
    and that is not a convenience. An ``.xyz`` carries no connectivity,
    so without it a composition has no bonds at all, and the renderer
    falls back to guessing them from a distance cutoff -- which is
    exactly the thing this module exists not to do, since two structures
    placed 4 A apart would come out bonded to each other. The sidecar is
    how the recorded bonds survive the trip through a file, so it is
    looked for rather than left to the caller to pass.
    """
    from ase import io as ase_io

    out: list[Item] = []
    for index, path in enumerate(paths):
        path = Path(path)
        atoms = ase_io.read(str(path), index=-1)
        if isinstance(atoms, list):             # pragma: no cover - ase API
            atoms = atoms[-1]
        _restore_sidecar(atoms, path)
        atoms.info.setdefault("source", str(path))
        label = (labels[index] if labels and index < len(labels)
                 else path.stem)
        out.append(Item(atoms=atoms, label=label))
    return out


def describe_layouts() -> Mapping[str, str]:
    """The arrangements, for a dropdown and for ``--help``."""
    return {
        "row": "side by side along one axis",
        "grid": "a near-square grid, one pitch for every cell",
        "circle": "evenly round a ring, wide enough that neighbours clear",
        "stack": "one above another, for a layered figure",
    }


__all__: list[str] = [
    "CLOSE",
    "TOUCHING",
    "Item",
    "SceneReport",
    "arrange",
    "compose",
    "describe_layouts",
    "euler_matrix",
    "items_from_files",
    "write_scene",
]
