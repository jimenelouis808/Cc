"""Finding attachment sites and building their local frames.

Attaching a group correctly means knowing *where* the free valence points.
That direction differs by site type:

* **Edge carbon** (coordination 2): the dangling bond lies in the sheet
  plane, opposite the sum of the two existing bond directions.
* **Basal carbon** (coordination 3): there is no free valence. Attaching
  here forces the carbon from sp2 to sp3, and the group goes along the local
  surface normal. This is what makes graphene oxide, and it genuinely
  puckers the sheet — the geometry produced here is a starting point that
  must be relaxed.
* **Bridge site**: two adjacent carbons sharing an epoxide, whose oxygen
  sits above their midpoint along the average normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
from ase import Atoms

from ..topology.graph import build_bond_graph

SiteKind = Literal["edge", "basal"]


@dataclass
class AttachmentSite:
    """A place where a functional group can be attached.

    Attributes
    ----------
    index
        Index of the anchor carbon.
    kind
        ``"edge"`` (coordination 2) or ``"basal"`` (coordination 3).
    origin
        Cartesian position of the anchor carbon.
    direction
        Unit vector along which the group extends.
    coordination
        The anchor's coordination number.
    """

    index: int
    kind: SiteKind
    origin: np.ndarray
    direction: np.ndarray
    coordination: int


def _neighbour_vectors(
    atoms: Atoms, index: int, neighbours: Sequence[int]
) -> np.ndarray:
    """Unit vectors from ``index`` to each neighbour, minimum-image aware."""
    positions = atoms.get_positions()
    cell = np.array(atoms.cell)
    pbc = atoms.get_pbc()
    vectors = []
    for neighbour in neighbours:
        delta = positions[neighbour] - positions[index]
        # Wrap across periodic boundaries so a neighbour on the far side of
        # the cell does not produce a direction pointing across the box.
        if any(pbc):
            fractional = np.linalg.solve(cell.T, delta)
            fractional -= np.round(fractional) * np.array(pbc, dtype=float)
            delta = cell.T @ fractional
        norm = np.linalg.norm(delta)
        if norm > 1e-8:
            vectors.append(delta / norm)
    return np.array(vectors) if vectors else np.zeros((0, 3))


def _edge_direction(vectors: np.ndarray) -> np.ndarray:
    """Dangling-bond direction for an under-coordinated atom."""
    total = vectors.sum(axis=0)
    norm = np.linalg.norm(total)
    if norm < 1e-6:
        # Two collinear neighbours leave the direction undetermined in-plane;
        # fall back to a perpendicular so the group at least does not overlap.
        fallback = np.cross(vectors[0], [0.0, 0.0, 1.0])
        if np.linalg.norm(fallback) < 1e-6:
            fallback = np.cross(vectors[0], [0.0, 1.0, 0.0])
        return fallback / np.linalg.norm(fallback)
    return -total / norm


def _basal_normal(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Local surface normal for a three-coordinated atom.

    ``reference`` breaks the up/down ambiguity: the normal is flipped to
    point away from it (used to keep groups on the outside of a tube, or on a
    chosen face of a sheet).
    """
    if len(vectors) < 2:
        return np.array([0.0, 0.0, 1.0])
    normal = np.cross(vectors[0], vectors[1])
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / norm
    if np.dot(normal, reference) < 0:
        normal = -normal
    return normal


def find_sites(
    atoms: Atoms,
    kind: Optional[SiteKind] = None,
    element: str = "C",
    outward_from_center: bool = True,
) -> list[AttachmentSite]:
    """Enumerate attachment sites.

    Parameters
    ----------
    atoms
        Structure to inspect.
    kind
        Restrict to ``"edge"`` or ``"basal"``. ``None`` returns both.
    element
        Only consider anchors of this element.
    outward_from_center
        For basal sites, orient the normal away from the structure's centre
        of geometry. That puts groups on the outside of a nanotube and on a
        consistent face of a sheet. Set ``False`` to keep the raw normal,
        whose sign then depends on neighbour ordering.

    Returns
    -------
    list[AttachmentSite]
        Ordered by atom index.
    """
    graph = build_bond_graph(atoms)
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    centre = positions.mean(axis=0)

    sites: list[AttachmentSite] = []
    for index in range(len(atoms)):
        if symbols[index] != element:
            continue
        neighbours = list(graph.neighbors(index)) if index in graph else []
        coordination = len(neighbours)
        if coordination == 2:
            site_kind: SiteKind = "edge"
        elif coordination == 3:
            site_kind = "basal"
        else:
            # Coordination 0, 1 or >=4: not a sane anchor. A lone or dangling
            # atom needs fixing, not decorating.
            continue
        if kind is not None and site_kind != kind:
            continue

        vectors = _neighbour_vectors(atoms, index, neighbours)
        if len(vectors) == 0:
            continue
        if site_kind == "edge":
            direction = _edge_direction(vectors)
        else:
            reference = (
                positions[index] - centre
                if outward_from_center
                else np.array([0.0, 0.0, 1.0])
            )
            if np.linalg.norm(reference) < 1e-6:
                reference = np.array([0.0, 0.0, 1.0])
            direction = _basal_normal(vectors, reference)

        sites.append(
            AttachmentSite(
                index=index,
                kind=site_kind,
                origin=positions[index].copy(),
                direction=direction,
                coordination=coordination,
            )
        )

    _make_normals_consistent(sites, positions, centre)
    return sites


def _make_normals_consistent(
    sites: list[AttachmentSite],
    positions: np.ndarray,
    centre: np.ndarray,
) -> None:
    """Give neighbouring basal sites normals pointing the same way.

    On a **curved** surface (a nanotube) the radial vector from the centre
    settles the up/down ambiguity cleanly. On a **flat** sheet it does not:
    the radial vector lies in the plane, so its dot product with the normal
    is near zero and the sign it produces is effectively arbitrary. Adjacent
    carbons then get opposite normals, and an epoxide averaging two of them
    ends up with a near-zero direction and lands inside the plane, on top of
    a carbon.

    So: keep the radial sign only where it is decisive, and elsewhere align
    every normal with the consensus of those that were.
    """
    basal = [site for site in sites if site.kind == "basal"]
    if len(basal) < 2:
        return

    decisive: list[np.ndarray] = []
    ambiguous: list[AttachmentSite] = []
    for site in basal:
        radial = site.origin - centre
        norm = np.linalg.norm(radial)
        if norm > 1e-6 and abs(np.dot(site.direction, radial / norm)) > 0.3:
            decisive.append(site.direction)
        else:
            ambiguous.append(site)

    if not ambiguous:
        return

    # Consensus direction: the mean of the decisive normals when the surface
    # is curved, otherwise simply the first normal, which fixes a face.
    if decisive:
        consensus = np.mean(decisive, axis=0)
        if np.linalg.norm(consensus) < 1e-6:
            consensus = ambiguous[0].direction
    else:
        consensus = ambiguous[0].direction
    consensus = consensus / np.linalg.norm(consensus)

    for site in ambiguous:
        if np.dot(site.direction, consensus) < 0:
            site.direction = -site.direction


def find_bridge_sites(
    atoms: Atoms,
    element: str = "C",
    outward_from_center: bool = True,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    """Enumerate adjacent basal pairs suitable for a bridging epoxide.

    Returns
    -------
    list of (i, j, midpoint, normal)
        One entry per bonded pair of three-coordinated carbons, with the
        midpoint of the bond and the averaged outward normal.
    """
    graph = build_bond_graph(atoms)
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    centre = positions.mean(axis=0)

    basal = {
        site.index: site
        for site in find_sites(
            atoms, kind="basal", element=element,
            outward_from_center=outward_from_center,
        )
    }
    pairs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for i, j in graph.edges:
        if i not in basal or j not in basal:
            continue
        if symbols[i] != element or symbols[j] != element:
            continue
        midpoint = 0.5 * (positions[i] + positions[j])
        normal = basal[i].direction + basal[j].direction
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            reference = midpoint - centre
            norm_ref = np.linalg.norm(reference)
            normal = (
                reference / norm_ref if norm_ref > 1e-6
                else np.array([0.0, 0.0, 1.0])
            )
        else:
            normal = normal / norm
        pairs.append((int(i), int(j), midpoint, normal))
    return pairs


def local_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an orthonormal frame whose z axis is ``direction``.

    The x and y axes are arbitrary but deterministic, which matters for
    reproducibility: the same structure always produces the same group
    orientation.
    """
    z_hat = np.asarray(direction, dtype=float)
    z_hat = z_hat / np.linalg.norm(z_hat)
    # Pick the global axis least aligned with z, so the cross product is
    # numerically well conditioned.
    helper = np.zeros(3)
    helper[int(np.argmin(np.abs(z_hat)))] = 1.0
    x_hat = np.cross(helper, z_hat)
    x_hat = x_hat / np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    return x_hat, y_hat, z_hat
