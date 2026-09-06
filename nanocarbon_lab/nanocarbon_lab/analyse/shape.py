"""What shape a structure is, when nothing recorded it.

A file this package wrote says what it is. A file someone else wrote says
only where the atoms are, and `pbc` is no help: an XYZ has none at all,
and a CIF written by a plane-wave code has `pbc=(True, True, True)` even
when the cell is a slab with 15 Å of vacuum over it, because that is the
only way those codes can express a slab.

So the shape is measured. Two questions, in this order.

**Which directions actually repeat.** A periodic axis whose atoms leave a
gap wider than a bond is not periodic in any useful sense -- it is vacuum
that the file had to declare periodic. Counting those gaps turns
`pbc=(True, True, True)` on a slab into the 2D it really is.

**What the finite part looks like.** Principal axes of the atom cloud
give three spans. A sheet has one span near zero; a chain has two. That
much is easy, and it is not enough: a nanotube's three spans are all
large, and so are a cage's. What separates them is that both are
**hollow** -- the atoms sit on a shell, at nearly constant distance from
an axis (a tube) or from a point (a cage) -- while a cluster or a piece
of bulk fills its volume. The radial spread is what says which, and it is
a ratio, so it does not need a length scale to compare against.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

#: A non-periodic direction must show a gap at least this wide, in Å, for
#: the axis to count as vacuum rather than as a real repeat. Wider than
#: any bond this package deals with (the longest is a 2.76 Å Te-Te) and
#: far narrower than the 10-15 Å a slab is padded with, so nothing sits
#: near the boundary.
VACUUM_GAP: float = 4.0

#: A principal span below this is "no real extent": a single layer.
#: Graphene is exactly 0 and an MX2 sandwich 3.13 Å, and both are one
#: layer, so the threshold has to clear the sandwich or every
#: dichalcogenide monolayer reads as three-dimensional. The narrowest
#: thing that must stay *above* it is a nanotube's diameter, and the
#: smallest tube that exists is about 6.8 Å across, so 4.0 sits with
#: room on both sides.
FLAT_SPAN: float = 4.0

#: Thickness up to which a 2D structure is one layer -- a "sheet" rather
#: than a "slab". An MX2 monolayer is 3.13 Å and a bilayer 9.3, so
#: anything between separates them; calling a monolayer a slab would
#: suggest a thickness it does not have.
LAYER_THICKNESS: float = 5.0

#: Relative spread of the radial distances below which the atoms lie on a
#: shell rather than filling a volume. Measured: a (6,6) nanotube 0.000,
#: a C60 0.000, a capped tube 0.09 (its caps are at a smaller radius than
#: its wall), a solid cluster 0.3 and up.
SHELL_SPREAD: float = 0.20


def vacuum_axes(atoms: Atoms) -> list[int]:
    """Cell axes along which the atoms leave a real gap.

    An axis is vacuum when, sorting the atoms by their fractional
    coordinate along it, some consecutive pair is further apart than
    :data:`VACUUM_GAP` -- including the wrap-around pair, since the gap
    in a slab is usually split across the cell boundary.

    This is what lets a slab written as fully periodic be recognised as
    2D. Trusting `pbc` alone would call it a bulk crystal and then
    compute a density for it, which is the number that would be quoted.
    """
    if not len(atoms) or not atoms.cell.rank:
        return []
    cell = np.asarray(atoms.cell)
    fractional = atoms.get_scaled_positions(wrap=True)
    axes = []
    for axis in range(3):
        length = float(np.linalg.norm(cell[axis]))
        if length < 1e-8:
            continue
        ordered = np.sort(fractional[:, axis]) * length
        if len(ordered) < 2:
            axes.append(axis)
            continue
        gaps = np.diff(ordered)
        wrap = (ordered[0] + length) - ordered[-1]
        if max(float(gaps.max()) if gaps.size else 0.0, float(wrap)) >= VACUUM_GAP:
            axes.append(axis)
    return axes


def periodic_axes(atoms: Atoms) -> list[int]:
    """Axes that are declared periodic *and* have no vacuum gap."""
    gaps = set(vacuum_axes(atoms))
    return [axis for axis in range(3)
            if atoms.pbc[axis] and axis not in gaps]


def principal_spans(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    """Extent of the atom cloud along its own principal axes.

    Returns
    -------
    (spans, axes)
        ``spans`` descending, in Å; ``axes`` the matching unit vectors as
        rows, so ``axes[0]`` is the long direction of a tube.
    """
    positions = atoms.get_positions()
    if len(positions) < 2:
        return np.zeros(3), np.eye(3)
    centred = positions - positions.mean(axis=0)
    _, _, directions = np.linalg.svd(centred, full_matrices=True)
    projected = centred @ directions.T
    spans = projected.max(axis=0) - projected.min(axis=0)
    order = np.argsort(spans)[::-1]
    return spans[order], directions[order]


def shell_spread(distances: np.ndarray) -> float:
    """Relative spread of a set of radii: ``std / mean``.

    Near zero when the atoms lie on a shell and large when they fill a
    volume. A ratio on purpose -- a 4 Å tube and a 40 Å one are equally
    hollow, and a threshold in Ångströms would have to be different for
    each.
    """
    if not distances.size or distances.mean() < 1e-8:
        return float("inf")
    return float(distances.std() / distances.mean())


def _axis_spreads(centred: np.ndarray, axes: np.ndarray,
                  extra: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Smallest radial spread about any candidate axis, and that axis.

    Every principal axis is tried, plus the periodic direction when there
    is one. Taking the longest principal axis alone is wrong for a short
    fat tube: a one-cell MoS2 nanotube is 33 Å across and 4.6 Å long, so
    its longest principal axis is a *diameter*, the atoms read as filling
    it, and a textbook nanotube was classified as a chain.
    """
    candidates = list(axes)
    if extra is not None:
        candidates.append(extra)
    best, best_axis = float("inf"), axes[0]
    for axis in candidates:
        norm = float(np.linalg.norm(axis))
        if norm < 1e-8:
            continue
        unit = axis / norm
        along = centred @ unit
        radial = np.linalg.norm(centred - np.outer(along, unit), axis=1)
        spread = shell_spread(radial)
        if spread < best:
            best, best_axis = spread, unit
    return best, best_axis


def _thickness_across(atoms: Atoms, subset: Atoms,
                      periodic: list[int]) -> float:
    """Extent perpendicular to the periodic directions, in Å.

    Measured along the **cell's** non-periodic direction, not along the
    smallest principal axis. On a minimal cell those are not the same
    thing: a 1x1 MoS2 bilayer spans 9.4 Å between its two layers and
    almost nothing in plane, so its smallest principal span is an
    *in-plane* direction and the bilayer read as 0.0 Å thick -- a slab
    reported as a single sheet.
    """
    cell = np.asarray(atoms.cell)
    positions = subset.get_positions()
    if not len(positions):
        return 0.0
    if len(periodic) == 2:
        normal = np.cross(cell[periodic[0]], cell[periodic[1]])
    elif len(periodic) == 1:
        spans, axes = principal_spans(subset)
        along = cell[periodic[0]]
        along = along / np.linalg.norm(along)
        residual = positions - np.outer(positions @ along, along)
        _, _, directions = np.linalg.svd(residual - residual.mean(axis=0),
                                         full_matrices=True)
        projected = (residual - residual.mean(axis=0)) @ directions.T
        widths = projected.max(axis=0) - projected.min(axis=0)
        return float(np.sort(widths)[0])
    else:
        return 0.0
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return 0.0
    normal = normal / norm
    projected = positions @ normal
    return float(projected.max() - projected.min())


def _classify(atoms: Atoms, indices: list[int] | None = None) -> dict:
    """Shape of one connected piece, or of the whole structure."""
    subset = atoms if indices is None else atoms[indices]
    periodic = periodic_axes(atoms)
    vacuum = vacuum_axes(atoms)
    spans, axes = principal_spans(subset)

    positions = subset.get_positions()
    centred = positions - (positions.mean(axis=0) if len(positions)
                           else np.zeros(3))
    cell = np.asarray(atoms.cell)
    extra = cell[periodic[0]] if len(periodic) == 1 else None
    axis_spread, tube_axis = _axis_spreads(centred, axes, extra)
    point_spread = shell_spread(np.linalg.norm(centred, axis=1))

    along = centred @ tube_axis
    radial = np.linalg.norm(centred - np.outer(along, tube_axis), axis=1)

    report = {
        "n_periodic_axes": len(periodic),
        "periodic_axes": periodic,
        "vacuum_axes": vacuum,
        "declared_pbc": [bool(flag) for flag in atoms.pbc],
        "spans": [round(float(value), 3) for value in spans],
        "shell_spread_about_point": round(point_spread, 3),
        "shell_spread_about_axis": round(axis_spread, 3),
        "radius": round(float(radial.mean()), 3),
    }

    if len(periodic) == 3:
        report.update(dimensionality=3, shape="bulk", basis="cell",
                      reason="periodic in all three directions, with no "
                             "vacuum gap along any of them.")
        return report

    if len(periodic) == 2:
        thickness = _thickness_across(atoms, subset, periodic)
        report.update(dimensionality=2,
                      shape="sheet" if thickness <= LAYER_THICKNESS else "slab",
                      basis="cell",
                      reason=f"periodic in two directions and {thickness:.2f} Å "
                             "thick across the third.")
        return report

    if len(periodic) == 1:
        if axis_spread <= SHELL_SPREAD:
            report.update(dimensionality=1, shape="tube", basis="cell",
                          reason=f"periodic along one direction, with the "
                                 f"atoms on a shell {radial.mean():.1f} Å from "
                                 "it.")
        elif _thickness_across(atoms, subset, periodic) <= FLAT_SPAN:
            thickness = _thickness_across(atoms, subset, periodic)
            report.update(dimensionality=1, shape="ribbon", basis="cell",
                          reason=f"periodic along one direction and flat "
                                 f"across it ({thickness:.2f} Å thick).")
        else:
            report.update(dimensionality=1, shape="wire", basis="cell",
                          reason="periodic along one direction, and solid "
                                 "rather than hollow across it.")
        return report

    # Nothing repeats, so the shape is whatever the atom cloud looks like.
    if len(subset) < 3:
        report.update(dimensionality=0, shape="molecule", basis="atoms",
                      reason="too few atoms to have a shape.")
        return report

    if spans[1] < FLAT_SPAN:
        report.update(dimensionality=1, shape="chain", basis="atoms",
                      reason=f"extended along one direction only "
                             f"({spans[0]:.1f} Å), under {FLAT_SPAN} Å across "
                             "the other two.")
        return report

    if spans[2] < FLAT_SPAN:
        report.update(dimensionality=2, shape="flake", basis="atoms",
                      reason=f"a single layer {spans[0]:.1f} x {spans[1]:.1f} Å, "
                             f"{spans[2]:.2f} Å thick, with no direction "
                             "repeating.")
        return report

    if axis_spread <= SHELL_SPREAD:
        report.update(dimensionality=1, shape="tube", basis="atoms",
                      reason=f"hollow about an axis: the atoms sit "
                             f"{radial.mean():.1f} Å from it with "
                             f"{100 * axis_spread:.0f}% spread.")
        return report

    if point_spread <= SHELL_SPREAD:
        report.update(dimensionality=0, shape="cage", basis="atoms",
                      reason=f"a closed shell: every atom "
                             f"{np.linalg.norm(centred, axis=1).mean():.1f} Å "
                             f"from the centre, {100 * point_spread:.0f}% "
                             "spread.")
        return report

    from ..functionalize.attach import bond_pairs
    from .rings import is_surface_net

    if is_surface_net(subset, bond_pairs(subset)):
        report.update(dimensionality=0, shape="branched shell", basis="atoms",
                      reason="a finite trivalent net -- a surface, but not "
                             "hollow about a single point or axis, which is "
                             "what a branched junction looks like.")
        return report

    report.update(dimensionality=3, shape="cluster", basis="atoms",
                  reason="fills its volume in all three directions and is "
                         "neither a shell nor a trivalent net.")
    return report


def describe_shape(atoms: Atoms) -> dict:
    """Classify a structure's shape from its geometry alone.

    Returns
    -------
    dict
        ``dimensionality`` 0-3 and ``shape`` a word for it: ``"molecule"``,
        ``"cage"``, ``"branched shell"``, ``"chain"``, ``"tube"``,
        ``"ribbon"``, ``"wire"``, ``"sheet"``, ``"flake"``, ``"slab"``,
        ``"bulk"`` or ``"cluster"``. ``basis`` says whether the verdict
        came from the cell or from the atom cloud, and the measured
        numbers behind it come too, so a reader can disagree with the
        thresholds rather than having to trust them.

    Notes
    -----
    **Each disjoint piece is classified separately** and the verdict is
    theirs when they agree. A multi-wall nanotube is two nested shells:
    taken together the atoms fill an annulus and read as a solid, and the
    structure came out "cluster" until the pieces were separated. When
    they agree, ``n_components`` says how many there are -- which is the
    difference between a nanotube and a multi-wall one.
    """
    from ..functionalize.attach import bond_pairs
    from ..topology import connected_components

    whole = _classify(atoms)
    if len(atoms) < 2 or not len(bond_pairs(atoms)):
        whole["n_components"] = len(atoms)
        return whole

    components = connected_components(atoms)
    whole["n_components"] = len(components)
    if len(components) < 2:
        return whole

    verdicts = [_classify(atoms, part) for part in components if len(part) >= 3]
    shapes = {verdict["shape"] for verdict in verdicts}
    if len(shapes) == 1 and verdicts:
        agreed = verdicts[0]
        # The pieces win only when they disagree with the whole about
        # *dimensionality*. Two stacked MX2 layers are two disjoint
        # sheets and also one slab, both 2D, and the slab is the more
        # useful of the two true statements -- it accounts for every
        # atom. Two nested tubes are 1D each while their union reads as
        # a 0D branched shell, and there the union is simply wrong: the
        # annulus between the walls is empty space, not material.
        if agreed["dimensionality"] != whole["dimensionality"]:
            whole.update(dimensionality=agreed["dimensionality"],
                         shape=agreed["shape"],
                         basis=agreed["basis"] + "/parts",
                         reason=f"{len(components)} disjoint pieces, each one "
                                f"a {agreed['shape']}: " + agreed["reason"])
        else:
            whole["reason"] += (f" Made of {len(components)} disjoint pieces, "
                                f"each a {agreed['shape']}.")
        whole["component_shapes"] = [verdict["shape"] for verdict in verdicts]
        whole["component_radii"] = [verdict["radius"] for verdict in verdicts]
        return whole

    whole["component_shapes"] = [verdict["shape"] for verdict in verdicts]
    return whole


__all__ = [
    "FLAT_SPAN",
    "LAYER_THICKNESS",
    "SHELL_SPREAD",
    "VACUUM_GAP",
    "describe_shape",
    "periodic_axes",
    "principal_spans",
    "shell_spread",
    "vacuum_axes",
]
