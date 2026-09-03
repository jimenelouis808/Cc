"""Nanoribbons cut from a TMD monolayer.

A ribbon is a monolayer made finite in one direction, and what makes it
interesting is entirely the edge. Unlike graphene, where a zigzag edge
is a zigzag edge, an MX2 zigzag edge comes in two chemically distinct
forms -- terminated by the metal row or by the chalcogen row -- and they
differ in almost everything that matters: the metal-terminated edge is
metallic and magnetic, the chalcogen-terminated one is not, and MoS2
triangles grown by CVD take their shape from which edge is cheaper under
the growth conditions.

So the termination is an explicit argument here, not something the cut
happens to decide.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from ase import Atoms

from ..utils.constants import DEFAULT_VACUUM_1D, DEFAULT_VACUUM_2D
from ..utils.geometry import center_in_cell
from .materials import Phase, TMDMaterial, coordination_geometry, get_material
from .slab import build_tmd_layers

Edge = Literal["zigzag", "armchair"]
Termination = Literal["metal", "chalcogen", "mixed"]


def build_tmd_ribbon(
    material: str | TMDMaterial = "MoS2",
    width: int = 6,
    length: int = 1,
    edge: Edge = "zigzag",
    termination: Termination = "mixed",
    phase: Phase = "2H",
    vacuum: float = DEFAULT_VACUUM_1D,
    vacuum_z: float = DEFAULT_VACUUM_2D,
) -> Atoms:
    """Cut a nanoribbon from a monolayer.

    Parameters
    ----------
    material
        Formula or :class:`~nanocarbon_lab.tmd.materials.TMDMaterial`.
    width
        Number of lattice rows across the ribbon (>= 2).
    length
        Repeats along the periodic direction (>= 1).
    edge
        ``"zigzag"`` or ``"armchair"``. A zigzag ribbon runs along a
        lattice vector; an armchair one runs along the perpendicular
        direction, which needs a rectangular supercell.
    termination
        Only meaningful for a zigzag edge, where the two sides are
        chemically different rows.

        ``"mixed"``
            The natural cut: one edge metal, the other chalcogen. This
            is what a plain slice of the lattice gives.
        ``"metal"``
            Both edges metal-terminated, made by stripping the outermost
            chalcogen row. Metallic and magnetic in DFT; the edge that
            dominates Mo-rich CVD growth.
        ``"chalcogen"``
            Both edges chalcogen-terminated, made by stripping the
            outermost metal row.
    phase
        Layer phase, passed through to the monolayer builder.
    vacuum
        Vacuum across the ribbon width (Å).
    vacuum_z
        Vacuum perpendicular to the layer (Å).

    Returns
    -------
    ase.Atoms
        Periodic along the ribbon axis only, with the edge description
        and measured width recorded in ``info``.

    Raises
    ------
    ValueError
        For a width below 2, a non-positive length, or an unknown edge.
    """
    if isinstance(material, str):
        material = get_material(material)
    if width < 2:
        raise ValueError("width must be >= 2 rows.")
    if length < 1:
        raise ValueError("length must be >= 1.")
    if edge not in ("zigzag", "armchair"):
        raise ValueError(f"Unknown edge {edge!r}; expected 'zigzag' or 'armchair'.")

    # Build an oversized sheet, then cut. Cutting a supercell is more
    # robust than trying to write down the ribbon's own basis: the phase
    # and stacking logic stays in one place.
    # The two edge types are perpendicular cuts of the same lattice, so
    # they differ in which axis is periodic, not just in where the cut
    # falls. In the a x a*sqrt(3) rectangle, x is the zigzag direction and
    # y the armchair one: a zigzag-*edged* ribbon runs along x, an
    # armchair-edged ribbon along y. Getting this backwards produces a
    # ribbon that is simply the other type in a different cell.
    if edge == "zigzag":
        sheet = build_tmd_layers(material, n_layers=1, phase=phase,
                                 nx=length, ny=width + 2, vacuum=vacuum_z)
        axis, cut_axis = 0, 1
        row_spacing = material.a * np.sqrt(3.0) / 2.0
    else:
        sheet = _rectangular_sheet(material, phase, width + 2, length, vacuum_z)
        axis, cut_axis = 1, 0
        row_spacing = material.a / 2.0

    positions = sheet.get_positions()
    symbols = np.array(sheet.get_chemical_symbols())
    coords = positions[:, cut_axis]
    lo = coords.min() + row_spacing * 0.5
    keep = (coords >= lo - 1e-6) & (coords < lo + width * row_spacing - 1e-6)

    if edge == "zigzag" and termination != "mixed":
        keep = _retermine_zigzag(
            keep, coords, symbols, material, termination, row_spacing
        )

    ribbon = sheet[keep]
    cell = np.array(sheet.cell)
    span = ribbon.get_positions()[:, cut_axis]
    cell[cut_axis] = np.zeros(3)
    cell[cut_axis][cut_axis] = (span.max() - span.min()) + vacuum
    ribbon.set_cell(cell)
    ribbon.set_pbc([axis == 0, axis == 1, False])
    center_in_cell(ribbon, axes=(cut_axis, 2))

    measured = float(span.max() - span.min())
    ribbon.info.update(
        {
            "structure_type": "tmd_ribbon",
            "material": material.formula,
            "metal": material.metal,
            "chalcogen": material.chalcogen,
            "phase": phase,
            "coordination": coordination_geometry(phase),
            "edge": edge,
            "termination": termination if edge == "zigzag" else "n/a",
            "width_rows": width,
            "width_angstrom": measured,
            "length_cells": length,
            "a": material.a,
            "bond_length": material.bond_length,
        }
    )
    return ribbon


def _retermine_zigzag(keep, coords, symbols, material, termination, row_spacing):
    """Strip the outermost row so both edges carry the same element.

    A plain cut leaves one metal edge and one chalcogen edge. Removing
    the outermost row of the unwanted element on the side that has it
    makes both edges alike, at the cost of one row of width.
    """
    kept = np.flatnonzero(keep)
    if kept.size == 0:
        return keep
    unwanted = material.chalcogen if termination == "metal" else material.metal
    lo, hi = coords[kept].min(), coords[kept].max()
    # Half the *smallest* gap between adjacent rows. Metal and chalcogen
    # rows alternate unevenly -- a/sqrt(3) then a/(2*sqrt(3)) -- so a
    # tolerance scaled to the lattice period `row_spacing` is nearly three
    # times too wide and sweeps up the neighbouring row, which is what
    # made every termination come out identical.
    tol = material.a / (4.0 * np.sqrt(3.0))
    for edge_coord in (lo, hi):
        on_edge = kept[np.abs(coords[kept] - edge_coord) < tol]
        if on_edge.size and np.all(symbols[on_edge] == unwanted):
            keep[on_edge] = False
    return keep


def _rectangular_sheet(material: TMDMaterial, phase: Phase, nx: int, ny: int,
                       vacuum_z: float) -> Atoms:
    """An orthogonal supercell, needed for armchair ribbons.

    The hexagonal primitive cell cannot express an armchair edge as a
    plain slice: the edge runs along a direction that is not a lattice
    vector. A rectangular cell of a x a*sqrt(3) can, and repeating it
    keeps the cut a simple coordinate range.
    """
    primitive = build_tmd_layers(material, n_layers=1, phase=phase,
                                 nx=1, ny=1, vacuum=vacuum_z)
    # Two primitive cells stacked along b give the a x a*sqrt(3) rectangle.
    doubled = primitive.repeat((1, 2, 1))
    positions = doubled.get_positions()
    cell = np.array(doubled.cell)
    rect = np.array([
        [material.a, 0.0, 0.0],
        [0.0, material.a * np.sqrt(3.0), 0.0],
        cell[2],
    ])
    # Wrap into the rectangle: the doubled hexagonal cell contains exactly
    # the same atoms, just expressed on a sheared basis.
    inverse = np.linalg.inv(rect)
    fractional = positions @ inverse
    fractional[:, :2] = np.mod(fractional[:, :2], 1.0)
    wrapped = Atoms(symbols=doubled.get_chemical_symbols(),
                    positions=fractional @ rect, cell=rect,
                    pbc=(True, True, False))
    return wrapped.repeat((nx, ny, 1))


__all__ = ["Edge", "Termination", "build_tmd_ribbon"]
