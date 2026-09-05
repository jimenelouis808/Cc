"""Twisted bilayers and van der Waals heterostructures.

Two hexagonal sheets stacked with a relative twist share a periodic cell
only at special angles. For integers ``m > n >= 0`` the superlattice
vector ``m*a1 + n*a2`` in one layer has exactly the same length as
``n*a1 + m*a2`` in the other, so rotating by the angle between them maps
one lattice onto a common superlattice:

    cos(theta) = (m^2 + n^2 + 4mn) / (2 (m^2 + mn + n^2))

and the cell holds ``m^2 + mn + n^2`` primitive cells **per layer**. That
series is not a curiosity: ``(2,1)`` is 21.79 deg, and ``(31,30)`` is
1.0845 deg -- the magic angle of twisted bilayer graphene, where the
moire bands flatten and the system superconducts. It is also why a twist
is not a free parameter: ask for 1.1 deg and you get 1.0845, with 11 164
atoms, because nothing periodic exists in between.

Stacking two *different* materials is a separate problem. Their lattice
constants differ (graphene 2.46 Å, hBN 2.504 Å, MoS2 3.16 Å), so a common
cell needs one of them strained. The builder reports that strain rather
than hiding it, and refuses a stack that would need more than a few per
cent -- at which point the model says more about the strain than about
the material.

Layers are described uniformly (:class:`Layer2D`): an in-plane lattice
constant plus a basis of ``(symbol, u, v, z)``. Graphene and hBN are two
sites at z = 0; a dichalcogenide is a metal at z = 0 and two chalcogens
at +-h/2, which is the same X-M-X sandwich :mod:`nanocarbon_lab.tmd`
builds, expressed once more so that stacking code needs to know nothing
about which family a layer came from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..tmd.materials import MATERIALS, get_material
from ..utils.constants import CC_BOND, DEFAULT_VACUUM_2D
from ..utils.geometry import center_in_cell

#: Interlayer separation (Å) for a van der Waals gap between two sheets,
#: measured between the facing atomic planes rather than between layer
#: centres -- a TMD's outer chalcogen plane is what touches its neighbour,
#: not its metal plane.
VDW_GAP = 3.35

#: Largest lattice mismatch a heterostructure may be strained through.
#: Graphene on hBN is 1.8% and routinely modelled; beyond a few per cent
#: the strain dominates whatever the model was built to show.
MAX_MISMATCH = 0.05


@dataclass(frozen=True)
class Layer2D:
    """One hexagonal 2D layer: lattice constant plus a basis.

    Attributes
    ----------
    name
        Label carried into ``atoms.info``.
    a
        In-plane lattice constant (Å) of the primitive hexagonal cell.
    basis
        ``(symbol, u, v, z)`` per site: ``u, v`` fractional in the
        hexagonal cell, ``z`` in Å from the layer's own mid-plane.
    thickness
        Distance from the lowest to the highest atomic plane (Å). Zero for
        graphene and hBN; ~3.1 Å for a dichalcogenide sandwich.
    """

    name: str
    a: float
    basis: tuple[tuple[str, float, float, float], ...]
    thickness: float = 0.0

    @property
    def n_sites(self) -> int:
        return len(self.basis)


def graphene_layer(bond: float = CC_BOND) -> Layer2D:
    """Graphene as a two-site hexagonal cell."""
    # (1/3, 1/3), not (1/3, 2/3): with a2 = a(1/2, sqrt3/2) -- the
    # 60-degree convention used throughout this module -- the honeycomb's
    # second site is at (1/3)(a1 + a2), which is a/sqrt(3) = 1.42 Å from
    # the first. The (1/3, 2/3) form belongs to the 120-degree cell and
    # here puts sites 0.82 Å apart.
    return Layer2D("graphene", math.sqrt(3.0) * bond,
                   (("C", 0.0, 0.0, 0.0), ("C", 1 / 3, 1 / 3, 0.0)))


def hbn_layer(a: float = 2.504) -> Layer2D:
    """Hexagonal boron nitride -- graphene's lattice, two species.

    The same honeycomb with B and N on the two sublattices, which makes
    it the bipartite case: every ring alternates, exactly as M and X do
    in a dichalcogenide.
    """
    return Layer2D("hBN", a, (("B", 0.0, 0.0, 0.0), ("N", 1 / 3, 1 / 3, 0.0)))


def tmd_layer(material: str = "MoS2") -> Layer2D:
    """A dichalcogenide monolayer as an X-M-X sandwich."""
    mat = get_material(material)
    half = mat.h / 2.0
    return Layer2D(
        mat.formula, mat.a,
        (
            (mat.metal, 0.0, 0.0, 0.0),
            (mat.chalcogen, 1 / 3, 1 / 3, +half),
            (mat.chalcogen, 1 / 3, 1 / 3, -half),
        ),
        thickness=mat.h,
    )


def available_layers() -> tuple[str, ...]:
    """Every layer name :func:`get_layer` accepts."""
    return ("graphene", "hBN", *sorted(MATERIALS))


def get_layer(name: str) -> Layer2D:
    """Look up a layer by name (``"graphene"``, ``"hBN"``, or a formula)."""
    if name == "graphene":
        return graphene_layer()
    if name.lower() in ("hbn", "h-bn"):
        return hbn_layer()
    if name in MATERIALS:
        return tmd_layer(name)
    raise KeyError(
        f"Unknown layer {name!r}. Available: {', '.join(available_layers())}."
    )


# --------------------------------------------------------------- commensurate
def twist_angle(m: int, n: int) -> float:
    """Commensurate twist angle in degrees for the pair ``(m, n)``."""
    if m <= 0 or n < 0 or n > m:
        raise ValueError(f"Need m > 0 and 0 <= n <= m; got ({m}, {n}).")
    denominator = 2.0 * (m * m + m * n + n * n)
    cosine = (m * m + n * n + 4 * m * n) / denominator
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def cells_per_layer(m: int, n: int) -> int:
    """Primitive cells per layer in the ``(m, n)`` moire supercell."""
    return m * m + m * n + n * n


def commensurate_series(max_index: int = 40) -> list[tuple[int, int, float, int]]:
    """Every ``(m, n, angle, cells)`` up to ``max_index``, angle ascending.

    Only coprime pairs: ``(2, 1)`` and ``(4, 2)`` describe the same twist,
    and the second merely repeats the first's cell four times.
    """
    seen: list[tuple[int, int, float, int]] = []
    for m in range(1, max_index + 1):
        for n in range(m + 1):
            if math.gcd(m, n) != 1:
                continue
            angle = twist_angle(m, n)
            if angle <= 0.0 or angle >= 60.0:
                continue
            seen.append((m, n, angle, cells_per_layer(m, n)))
    seen.sort(key=lambda item: item[2])
    return seen


def nearest_commensurate(target_deg: float,
                         max_index: int = 40) -> tuple[int, int, float, int]:
    """The commensurate twist closest to ``target_deg``.

    Returns ``(m, n, angle, cells)``. Angles are not free: between two
    entries of the series there is no periodic cell at all, so a request
    is snapped and the achieved angle reported.
    """
    series = commensurate_series(max_index)
    if not series:
        raise ValueError("max_index too small to produce any twist angle.")
    return min(series, key=lambda item: abs(item[2] - target_deg))


# -------------------------------------------------------------------- filling
def _hex_vectors(a: float) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([a, 0.0]),
            np.array([a * 0.5, a * math.sqrt(3.0) / 2.0]))


def _rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])


def _fill_supercell(layer: Layer2D, rotation_deg: float,
                    super_vectors: np.ndarray, z_shift: float,
                    scale: float = 1.0, expected: int | None = None):
    """Place one layer's atoms inside the supercell, wrapped into it.

    Every candidate lattice point is wrapped into ``[0, 1)`` in the
    supercell basis and then deduplicated, rather than being tested
    against the cell boundary. On a rotated lattice a site can land at a
    fractional coordinate of 1 - 1e-13, and a half-open interval test
    then drops or double-counts it depending on rounding; wrapping and
    deduplicating cannot.
    """
    a1, a2 = _hex_vectors(layer.a * scale)
    rotation = _rotation(rotation_deg)
    basis_matrix = np.linalg.inv(super_vectors.T)

    longest = max(np.linalg.norm(super_vectors[0]),
                  np.linalg.norm(super_vectors[1]))
    reach = int(math.ceil(longest / (layer.a * scale))) + 2

    seen: dict[tuple[int, int, int], tuple[str, np.ndarray]] = {}
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            origin = rotation @ (i * a1 + j * a2)
            for symbol, u, v, z in layer.basis:
                point = origin + rotation @ (u * a1 + v * a2)
                frac = basis_matrix @ point
                frac -= np.floor(frac + 1e-9)
                # Round at 1e-3 of a cell, not finer. Two (i, j) pairs
                # that generate the same site differ by floating-point
                # noise, so a 1e-5 key fails to collide and the site is
                # kept twice -- which showed up as a 484-atom cell where
                # 28 were expected, with atoms 0.12 Å apart. Distinct
                # sites are ~1/sqrt(cells) apart in fractional terms,
                # far coarser than this even for the magic angle.
                key = (int(round(frac[0] * 1e3)) % 1000,
                       int(round(frac[1] * 1e3)) % 1000,
                       int(round(z * 100.0)))
                if key in seen:
                    continue
                seen[key] = (symbol, np.array([frac[0], frac[1],
                                               z * scale + z_shift]))

    symbols = [item[0] for item in seen.values()]
    fractional = np.array([item[1] for item in seen.values()])
    if expected is not None and len(symbols) != expected:
        # A wrong count here is silent otherwise: the cell still builds
        # and still looks plausible, it just has the wrong number of
        # atoms in it. Commensurability fixes this number exactly, so
        # check it rather than trusting the wrap-and-deduplicate.
        raise RuntimeError(
            f"{layer.name} filled the supercell with {len(symbols)} atoms "
            f"where commensurability requires {expected}. This is a bug in "
            "the lattice fill, not a parameter problem."
        )
    cartesian = np.zeros((len(fractional), 3))
    cartesian[:, :2] = fractional[:, :2] @ super_vectors
    cartesian[:, 2] = fractional[:, 2]
    return symbols, cartesian


def _supercell_vectors(a: float, m: int, n: int) -> np.ndarray:
    """The two moire lattice vectors, as rows."""
    a1, a2 = _hex_vectors(a)
    first = m * a1 + n * a2
    # 60 degrees on from the first, which keeps the supercell hexagonal.
    second = _rotation(60.0) @ first
    return np.array([first, second])


# ------------------------------------------------------------------- builders
def build_twisted_bilayer(
    layer: str = "graphene",
    target_angle: float = 5.0,
    max_index: int = 40,
    gap: float = VDW_GAP,
    vacuum: float = DEFAULT_VACUUM_2D,
    top_layer: str | None = None,
    max_mismatch: float = MAX_MISMATCH,
) -> Atoms:
    """Stack two hexagonal layers with a commensurate relative twist.

    Parameters
    ----------
    layer
        Bottom layer: ``"graphene"``, ``"hBN"`` or a dichalcogenide
        formula. See :func:`available_layers`.
    target_angle
        Wanted twist in degrees. Snapped to the nearest commensurate
        angle, because no periodic cell exists in between; the achieved
        angle is reported in ``info``.
    max_index
        Largest ``m`` searched. Small angles need large indices -- the
        magic angle is ``(31, 30)`` -- and the cell grows as
        ``m^2 + mn + n^2``, so this also bounds the atom count.
    gap
        Separation between the facing atomic planes (Å), not between
        layer mid-planes.
    vacuum
        Total vacuum along z.
    top_layer
        A different material for the upper layer, making this a
        heterostructure rather than a twisted homobilayer. Its lattice is
        strained onto the lower one's supercell.
    max_mismatch
        Largest lattice mismatch accepted for a heterostructure.

    Returns
    -------
    ase.Atoms
        Periodic in-plane, with the achieved angle, the moire period, the
        cell count and any imposed strain in ``info``.

    Raises
    ------
    ValueError
        For an unknown layer, a non-positive gap, or a heterostructure
        whose lattice mismatch exceeds ``max_mismatch``.
    """
    bottom = get_layer(layer)
    top = get_layer(top_layer) if top_layer else bottom
    if gap <= 0:
        raise ValueError("gap must be positive.")

    mismatch = (top.a - bottom.a) / bottom.a
    if abs(mismatch) > max_mismatch:
        raise ValueError(
            f"{bottom.name} (a = {bottom.a:.3f} Å) and {top.name} "
            f"(a = {top.a:.3f} Å) differ by {abs(mismatch):.1%}, over the "
            f"{max_mismatch:.0%} limit. A common cell would have to strain one "
            "of them so hard that the model would describe the strain rather "
            "than the materials."
        )

    m, n, angle, cells = nearest_commensurate(target_angle, max_index)
    super_vectors = _supercell_vectors(bottom.a, m, n)

    # One layer unrotated and the other turned by the full angle -- *not*
    # a symmetric +-theta/2, tempting as that looks. The supercell
    # m*a1 + n*a2 is a lattice vector of the unrotated layer, and
    # R(-theta) maps it to n*a1 + m*a2, which is a lattice vector too;
    # that pair is what commensurability means here. Rotating both layers
    # by half the angle leaves the supercell commensurate with neither,
    # and the fill then produced 242 atoms where 14 were required.
    separation = gap + 0.5 * (bottom.thickness + top.thickness)
    bottom_symbols, bottom_xyz = _fill_supercell(
        bottom, 0.0, super_vectors, 0.0,
        expected=cells * bottom.n_sites)
    # Minus, not plus. V = m*a1 + n*a2 is a lattice vector of a layer
    # turned by phi exactly when R(-phi)*V is one of the unrotated
    # lattice, and it is R(+theta)*V that lands on n*a1 + m*a2. Getting
    # this backwards leaves the top layer incommensurate and it filled
    # with 98 atoms where 14 were required.
    top_symbols, top_xyz = _fill_supercell(
        top, -angle, super_vectors, separation,
        scale=bottom.a / top.a, expected=cells * top.n_sites)

    symbols = bottom_symbols + top_symbols
    positions = np.vstack([bottom_xyz, top_xyz])
    height = separation + 0.5 * top.thickness + vacuum
    cell = np.zeros((3, 3))
    cell[:2, :2] = super_vectors
    cell[2, 2] = height

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell,
                  pbc=(True, True, False))
    center_in_cell(atoms, axes=(2,))

    period = float(np.linalg.norm(super_vectors[0]))
    atoms.info.update(
        {
            "structure_type": "twisted_bilayer",
            "bottom_layer": bottom.name,
            "top_layer": top.name,
            "twist_angle": angle,
            "requested_angle": target_angle,
            "commensurate_index": (m, n),
            "cells_per_layer": cells,
            "moire_period": period,
            "interlayer_gap": gap,
            "lattice_mismatch": mismatch,
            "imposed_strain": -mismatch if top_layer else 0.0,
            "n_bottom": len(bottom_symbols),
            "n_top": len(top_symbols),
        }
    )
    return atoms


def build_vdw_stack(
    layers: list[str],
    gap: float = VDW_GAP,
    nx: int = 1,
    ny: int = 1,
    vacuum: float = DEFAULT_VACUUM_2D,
    max_mismatch: float = MAX_MISMATCH,
) -> Atoms:
    """Stack layers vertically with no twist, on a common in-plane cell.

    The untwisted counterpart of :func:`build_twisted_bilayer`, and the
    cheap one: with every layer aligned there is no moire, so the cell
    stays primitive and an arbitrary number of layers costs nothing. Use
    it for graphene/hBN/MoS2 sandwiches where the stacking order is the
    point and the twist is not.

    Every layer is strained onto the first one's lattice; the largest
    strain imposed is reported and bounded by ``max_mismatch``.
    """
    if len(layers) < 2:
        raise ValueError("A stack needs at least two layers.")
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1.")

    resolved = [get_layer(name) for name in layers]
    base = resolved[0]
    strains = [(entry.a - base.a) / base.a for entry in resolved]
    worst = max(abs(value) for value in strains)
    if worst > max_mismatch:
        culprit = resolved[max(range(len(strains)),
                               key=lambda k: abs(strains[k]))]
        raise ValueError(
            f"{culprit.name} differs from {base.name} by {worst:.1%}, over the "
            f"{max_mismatch:.0%} limit; the stack would be a strain study."
        )

    super_vectors = np.array([nx * _hex_vectors(base.a)[0],
                              ny * _hex_vectors(base.a)[1]])
    symbols: list[str] = []
    positions: list[np.ndarray] = []
    height = 0.0
    for index, entry in enumerate(resolved):
        if index:
            height += gap + 0.5 * (resolved[index - 1].thickness
                                   + entry.thickness)
        part_symbols, part_xyz = _fill_supercell(
            entry, 0.0, super_vectors, height, scale=base.a / entry.a,
            expected=nx * ny * entry.n_sites)
        symbols += part_symbols
        positions.append(part_xyz)

    stacked = np.vstack(positions)
    total = height + 0.5 * resolved[-1].thickness + vacuum
    cell = np.zeros((3, 3))
    cell[:2, :2] = super_vectors
    cell[2, 2] = total

    atoms = Atoms(symbols=symbols, positions=stacked, cell=cell,
                  pbc=(True, True, False))
    center_in_cell(atoms, axes=(2,))
    atoms.info.update(
        {
            "structure_type": "vdw_stack",
            "layers": list(layers),
            "n_layers": len(resolved),
            "interlayer_gap": gap,
            "imposed_strain": strains,
            "worst_strain": worst,
            "supercell": (nx, ny),
        }
    )
    return atoms


__all__ = [
    "MAX_MISMATCH",
    "VDW_GAP",
    "Layer2D",
    "available_layers",
    "build_twisted_bilayer",
    "build_vdw_stack",
    "cells_per_layer",
    "commensurate_series",
    "get_layer",
    "graphene_layer",
    "hbn_layer",
    "nearest_commensurate",
    "tmd_layer",
    "twist_angle",
]
