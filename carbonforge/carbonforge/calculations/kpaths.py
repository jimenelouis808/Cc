"""High-symmetry k-paths for electronic band structures.

Choosing a band path by hand is error-prone: the correct special points
depend on the Bravais lattice, and a graphene *orthogonal* supercell has a
different path (Γ-X-S-Y-Γ) from the hexagonal primitive cell (Γ-M-K-Γ).
Getting this wrong silently produces a plausible-looking but meaningless
band plot.

So we delegate lattice classification to ASE's :meth:`ase.cell.Cell.bandpath`,
which implements the Setyawan-Curtarolo conventions and is aware of ``pbc``,
and fall back to a simple Γ-to-zone-boundary path only when ASE cannot
classify the cell (low-symmetry or disordered systems, where no standard
path exists anyway).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ase import Atoms


@dataclass
class BandPathSpec:
    """A resolved band path ready to be written into an input file.

    Attributes
    ----------
    labels
        Ordered special-point labels, e.g. ``["G", "M", "K", "G"]``.
        ``"G"`` is used for Γ because neither QE nor SIESTA accept UTF-8 here.
    points
        Fractional (crystal) coordinates of each label, shape ``(n, 3)``.
    npoints_per_segment
        How many k-points to sample along each segment.
    dimensionality
        Number of periodic directions (0-3).
    source
        ``"ase"`` when ASE classified the lattice, ``"fallback"`` when we
        generated a generic path. Surfaced to the user, because a fallback
        path is not a standard one and should be reported rather than
        silently trusted.
    note
        Human-readable explanation shown in the GUI / CLI.
    """

    labels: list[str]
    points: np.ndarray
    npoints_per_segment: int = 30
    dimensionality: int = 3
    source: str = "ase"
    note: str = ""

    @property
    def path_string(self) -> str:
        """Return the path as a compact string, e.g. ``"G-M-K-G"``."""
        return "-".join(self.labels)

    def total_points(self) -> int:
        """Total number of k-points the path will generate."""
        return max(0, len(self.labels) - 1) * self.npoints_per_segment + 1


def _fallback_path(atoms: Atoms, npoints: int) -> BandPathSpec:
    """Generic Γ → zone-boundary path along each periodic direction.

    Used when ASE cannot classify the lattice. Not a standard path — the
    ``source="fallback"`` flag exists so callers can say so out loud.
    """
    pbc = list(atoms.get_pbc())
    dim = int(sum(pbc))
    if dim == 0:
        return BandPathSpec(
            labels=["G"],
            points=np.zeros((1, 3)),
            npoints_per_segment=npoints,
            dimensionality=0,
            source="fallback",
            note=(
                "Sistema aislado (0D): no hay dispersión de bandas. "
                "Solo se calcula el punto Γ."
            ),
        )

    labels = ["G"]
    points = [np.zeros(3)]
    axis_names = {0: "X", 1: "Y", 2: "Z"}
    for axis, periodic in enumerate(pbc):
        if not periodic:
            continue
        edge = np.zeros(3)
        edge[axis] = 0.5
        labels.append(axis_names[axis])
        points.append(edge)
        # Return to Γ between axes so segments stay physically meaningful.
        if axis != max(i for i, p in enumerate(pbc) if p):
            labels.append("G")
            points.append(np.zeros(3))

    return BandPathSpec(
        labels=labels,
        points=np.array(points),
        npoints_per_segment=npoints,
        dimensionality=dim,
        source="fallback",
        note=(
            "ASE no pudo clasificar la red de Bravais, así que se usa un "
            "camino genérico Γ → borde de zona. No es un camino estándar: "
            "revísalo antes de publicar."
        ),
    )


def suggest_band_path(
    atoms: Atoms,
    npoints_per_segment: int = 30,
    path: Optional[str] = None,
) -> BandPathSpec:
    """Propose a band path appropriate to the structure's lattice.

    Parameters
    ----------
    atoms
        Structure whose cell and ``pbc`` determine the path.
    npoints_per_segment
        Sampling density along each segment.
    path
        Explicit path string (e.g. ``"GMKG"``) to override the automatic
        choice. Passed straight to ASE, which validates the labels.

    Returns
    -------
    BandPathSpec

    Notes
    -----
    For a 1D system (a CNT periodic along z) the meaningful path is Γ → Z:
    the transverse directions carry vacuum, so dispersion there is an
    artefact of the periodic images, not physics.
    """
    pbc = list(atoms.get_pbc())
    dim = int(sum(pbc))

    if dim == 0:
        return _fallback_path(atoms, npoints_per_segment)

    try:
        bandpath = atoms.cell.bandpath(
            path=path, npoints=0, pbc=atoms.get_pbc()
        )
        special = bandpath.special_points
        labels = [c for c in bandpath.path if c not in ", "]
        if not labels:
            return _fallback_path(atoms, npoints_per_segment)
        points = np.array([special[label] for label in labels])
    except Exception:
        # ASE raises a variety of exceptions for cells it cannot classify
        # (degenerate, extremely low symmetry, disordered supercells).
        return _fallback_path(atoms, npoints_per_segment)

    return BandPathSpec(
        labels=[("G" if lab == "G" else lab) for lab in labels],
        points=points,
        npoints_per_segment=npoints_per_segment,
        dimensionality=dim,
        source="ase",
        note=(
            f"Camino estándar para la red detectada por ASE "
            f"({dim}D, {len(labels)} puntos de alta simetría)."
        ),
    )


def format_qe_kpath(spec: BandPathSpec) -> str:
    """Render a :class:`BandPathSpec` as a QE ``K_POINTS crystal_b`` card.

    In ``crystal_b`` format each line carries a special point in crystal
    coordinates plus the number of points used to reach the *next* one; the
    final point takes a weight of 0.
    """
    lines = ["K_POINTS crystal_b", f"  {len(spec.labels)}"]
    for index, (label, point) in enumerate(zip(spec.labels, spec.points)):
        is_last = index == len(spec.labels) - 1
        count = 0 if is_last else spec.npoints_per_segment
        lines.append(
            f"  {point[0]:12.8f} {point[1]:12.8f} {point[2]:12.8f} {count:4d}  ! {label}"
        )
    return "\n".join(lines)


def format_siesta_bandlines(spec: BandPathSpec) -> str:
    """Render a :class:`BandPathSpec` as a SIESTA ``BandLines`` block.

    SIESTA's convention is the mirror image of QE's: each line gives the
    number of points used to *arrive* at that point, so the first entry is 1
    and the counts are shifted by one relative to :func:`format_qe_kpath`.
    Mixing the two conventions up shifts every label on the plot, so the
    shift is applied here rather than left to the caller.
    """
    lines = ["%block BandLines"]
    for index, (label, point) in enumerate(zip(spec.labels, spec.points)):
        count = 1 if index == 0 else spec.npoints_per_segment
        lines.append(
            f"  {count:4d} {point[0]:12.8f} {point[1]:12.8f} {point[2]:12.8f}  {label}"
        )
    lines.append("%endblock BandLines")
    return "\n".join(lines)
