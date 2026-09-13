"""Assemblies of tubes: multi-wall nanotubes and hexagonally packed bundles.

Neither is a new *topology* -- each shell or tube is an ordinary capped
nanotube from :func:`nanocarbon_lab.builders.capped_cnt.build_capped_cnt`.
What makes them their own objects is the **van der Waals spacing** between
walls, which is not something the covalent relaxation knows about: shells
of a MWCNT are held ~3.4 Å apart (the graphite interlayer distance) by
dispersion alone, and tubes in a rope pack on a triangular lattice at the
same gap.

So these builders place independently-relaxed shells, and the geometry
check confirms the walls stay apart rather than interpenetrating. There
is deliberately no attempt to relax the assembly as a whole: doing that
properly needs a dispersion term the valence force field does not have,
and a bare covalent relaxation would happily collapse the shells into
each other.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from ..utils.geometry import center_in_cell
from .capped_cnt import build_capped_cnt

# Graphite interlayer / inter-tube van der Waals gap.
GRAPHITIC_GAP = 3.4


def build_multiwall_cnt(
    n_shells: int = 2,
    inner_freq: int = 3,
    freq_step: int = 2,
    n_body_rings: int = 10,
    bond: float = CC_BOND,
    roughness: float = 0.0,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
    **shell_kwargs,
) -> Atoms:
    """Build a multi-wall carbon nanotube from concentric capped shells.

    Each shell is a complete capped tube, so every one satisfies Euler's
    theorem independently (12 pentagons apiece) and the assembly's ring
    counts are simply their sum.

    Parameters
    ----------
    n_shells
        Number of concentric walls (>= 1).
    inner_freq
        Subdivision frequency of the innermost shell, which fixes its
        radius at ``~1.96 * inner_freq`` Å.
    freq_step
        Frequency increment between shells. Because the lattice quantises
        radius in steps of ``sqrt(3) * bond * 5 / (2*pi)`` (~1.96 Å),
        the wall spacing is ``freq_step * 1.96`` Å. ``freq_step=2`` gives
        ~3.9 Å, the closest realisable value to graphite's 3.4 Å -- the
        radius simply cannot land on 3.4 exactly, and the achieved
        spacing is reported in ``atoms.info["wall_spacing"]``.
    n_body_rings
        Body length of every shell. Shells share a length, so the caps
        nest.
    bond, roughness, vacuum, seed
        As for :func:`build_capped_cnt`; passed to every shell.
    **shell_kwargs
        Any further :func:`build_capped_cnt` arguments (``defects``,
        ``shape``, ...) applied to every shell.

    Returns
    -------
    ase.Atoms
        The assembled tube. ``atoms.info`` carries per-shell radii,
        ``wall_spacing``, summed ``ring_counts``, and the combined
        ``bonds``/``rings`` with shell offsets already applied.

    Raises
    ------
    ValueError
        For ``n_shells < 1`` or a non-positive ``freq_step``.
    """
    if n_shells < 1:
        raise ValueError("n_shells must be >= 1.")
    if freq_step < 1:
        raise ValueError("freq_step must be >= 1.")

    positions: list[np.ndarray] = []
    bonds: list[list[int]] = []
    rings: list[list[int]] = []
    ring_counts: dict[int, int] = {}
    radii: list[float] = []
    shell_ranges: list[tuple[int, int]] = []
    offset = 0

    for shell in range(n_shells):
        freq = inner_freq + shell * freq_step
        tube = build_capped_cnt(
            n_body_rings=n_body_rings, freq=freq, bond=bond,
            roughness=roughness, seed=seed, **shell_kwargs,
        )
        shell_pos = tube.get_positions()
        shell_pos = shell_pos - shell_pos.mean(axis=0)
        positions.append(shell_pos)
        bonds += [[a + offset, b + offset] for a, b in tube.info["bonds"]]
        rings += [[a + offset for a in ring] for ring in tube.info["rings"]]
        for size, count in tube.info["ring_counts"].items():
            ring_counts[size] = ring_counts.get(size, 0) + count
        radii.append(float(tube.info["radius"]))
        shell_ranges.append((offset, offset + len(tube)))
        offset += len(tube)

    merged = np.vstack(positions)
    atoms = Atoms(symbols=["C"] * len(merged), positions=merged, pbc=False)
    extents = merged.max(axis=0) - merged.min(axis=0)
    atoms.set_cell(np.diag(extents + vacuum))
    center_in_cell(atoms, axes=(0, 1, 2))

    spacings = [radii[i + 1] - radii[i] for i in range(len(radii) - 1)]
    atoms.info.update(
        {
            "structure_type": "multiwall_cnt",
            "n_shells": n_shells,
            "shell_radii": radii,
            "wall_spacing": float(np.mean(spacings)) if spacings else 0.0,
            "n_body_rings": n_body_rings,
            "bond": bond,
            "roughness": roughness,
            "ring_counts": {int(k): int(v) for k, v in ring_counts.items()},
            "rings": rings,
            "bonds": bonds,
            "geometry": _assembly_geometry(merged, bonds, shell_ranges),
        }
    )
    return atoms


def build_bundle(
    n_rings_across: int = 1,
    freq: int = 3,
    n_body_rings: int = 10,
    gap: float = GRAPHITIC_GAP,
    bond: float = CC_BOND,
    roughness: float = 0.0,
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int | None = 0,
    **tube_kwargs,
) -> Atoms:
    """Build a hexagonally packed rope of parallel nanotubes.

    Single-wall tubes grown by most routes self-assemble into ropes, packed
    on a triangular lattice at a van der Waals gap. Tubes are placed on
    complete hexagonal shells around a central one, so ``n_rings_across``
    of 0, 1, 2 gives 1, 7, 19 tubes.

    Parameters
    ----------
    n_rings_across
        Hexagonal shells around the central tube (0 = a single tube).
    freq, n_body_rings, bond, roughness, seed
        Passed to :func:`build_capped_cnt` for every tube.
    gap
        Wall-to-wall separation in Å; the lattice constant is
        ``2 * radius + gap``. Defaults to graphite's 3.4 Å.
    vacuum
        Vacuum padding around the bundle.
    **tube_kwargs
        Further :func:`build_capped_cnt` arguments applied to each tube.

    Returns
    -------
    ase.Atoms
        The rope, with ``n_tubes``, ``lattice_constant`` and summed ring
        statistics in ``atoms.info``.
    """
    if n_rings_across < 0:
        raise ValueError("n_rings_across must be >= 0.")
    if gap <= 0:
        raise ValueError("gap must be positive.")

    template = build_capped_cnt(
        n_body_rings=n_body_rings, freq=freq, bond=bond,
        roughness=roughness, seed=seed, **tube_kwargs,
    )
    base = template.get_positions()
    base = base - base.mean(axis=0)
    radius = float(template.info["radius"])
    lattice = 2.0 * radius + gap

    # Axial coordinates of a triangular lattice, filtered to complete
    # hexagonal shells (|q|, |r|, |q+r| all within n_rings_across).
    centres: list[np.ndarray] = []
    span = n_rings_across
    for q in range(-span, span + 1):
        for r in range(-span, span + 1):
            if abs(q + r) > span:
                continue
            x = lattice * (q + r / 2.0)
            y = lattice * (np.sqrt(3.0) / 2.0) * r
            centres.append(np.array([x, y, 0.0]))

    positions: list[np.ndarray] = []
    bonds: list[list[int]] = []
    rings: list[list[int]] = []
    ring_counts: dict[int, int] = {}
    tube_ranges: list[tuple[int, int]] = []
    offset = 0
    for centre in centres:
        positions.append(base + centre)
        tube_ranges.append((offset, offset + len(base)))
        bonds += [[a + offset, b + offset] for a, b in template.info["bonds"]]
        rings += [[a + offset for a in ring] for ring in template.info["rings"]]
        for size, count in template.info["ring_counts"].items():
            ring_counts[size] = ring_counts.get(size, 0) + count
        offset += len(base)

    merged = np.vstack(positions)
    atoms = Atoms(symbols=["C"] * len(merged), positions=merged, pbc=False)
    extents = merged.max(axis=0) - merged.min(axis=0)
    atoms.set_cell(np.diag(extents + vacuum))
    center_in_cell(atoms, axes=(0, 1, 2))

    atoms.info.update(
        {
            "structure_type": "bundle",
            "n_tubes": len(centres),
            "n_rings_across": n_rings_across,
            "tube_radius": radius,
            "lattice_constant": lattice,
            "gap": gap,
            "n_body_rings": n_body_rings,
            "bond": bond,
            "roughness": roughness,
            "ring_counts": {int(k): int(v) for k, v in ring_counts.items()},
            "rings": rings,
            "bonds": bonds,
            "geometry": _assembly_geometry(merged, bonds, tube_ranges),
        }
    )
    return atoms


def _assembly_geometry(
    positions: np.ndarray,
    bonds: list[list[int]],
    shell_ranges: list[tuple[int, int]],
) -> dict[str, float | int]:
    """Geometry report for an assembly, plus the closest inter-wall approach.

    That second number is the whole point of these structures: walls must
    sit near the van der Waals gap, and the covalent relaxation has no
    dispersion term that would keep them there, so it is measured.

    It is computed **between shells**, using the index ranges the builder
    already knows, rather than by excluding bonded neighbours. Exclusion
    does not work here: every wall is full of non-bonded intra-wall
    distances shorter than the gap -- 2.30 Å across a pentagon, 2.84 Å
    across a hexagon -- so a lone tube would report a "wall separation" of
    2.8 Å and the number would mean nothing at all.
    """
    from scipy.spatial import cKDTree

    from .capped_cnt import geometry_report

    report = dict(geometry_report(positions, [tuple(b) for b in bonds]))

    closest = float("inf")
    for i, (start_a, end_a) in enumerate(shell_ranges):
        others = [
            index
            for j, (start_b, end_b) in enumerate(shell_ranges)
            if j != i
            for index in range(start_b, end_b)
        ]
        if not others:
            continue
        tree = cKDTree(positions[others])
        distances, _ = tree.query(positions[start_a:end_a], k=1)
        closest = min(closest, float(distances.min()))

    report["min_wall_separation"] = closest if np.isfinite(closest) else float("nan")
    return report


def stack_rings(atoms: Atoms) -> dict[int, int]:
    """Ring-size histogram of an assembly (already summed in ``info``)."""
    return dict(atoms.info.get("ring_counts", {}))


__all__ = [
    "GRAPHITIC_GAP",
    "build_bundle",
    "build_multiwall_cnt",
    "stack_rings",
]
