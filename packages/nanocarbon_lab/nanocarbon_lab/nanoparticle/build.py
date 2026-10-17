"""One call from a file path to a finished particle.

The pipeline is four steps -- read, shape, carve, coat -- and three
callers want all four: the command line, the window, and
:class:`nanocarbon_lab.jobs.Job`. Writing it out three times is how the
three drift, so it is written here once and they all go through it.

The only thing this adds over calling the four modules in order is the
**assembly of the reports**. A finished particle carries three of them --
what the crystal was, what the cut did, what the coating came out as --
and they go into ``atoms.info`` as plain dictionaries, because that is
what survives being written to a file and read back by something that has
never heard of this package.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ase import Atoms

from ..utils.constants import CC_BOND, DEFAULT_VACUUM_1D
from . import shapes as shape_module
from .carve import CarveReport, carve_nanoparticle
from .coat import GRAPHITIC_GAP, CoatReport, encapsulate
from .crystal import CrystalReport, read_crystal
from .quality import particle_quality

#: Area one carbon occupies in a graphitic sheet, Å^2. Half of the
#: two-atom cell, ``(sqrt(3)/2) a^2`` with ``a = sqrt(3) * 1.42``. Used
#: only to estimate how many atoms a shell will have before building it.
AREA_PER_CARBON = 0.5 * 1.5 * math.sqrt(3.0) * CC_BOND ** 2


def build_nanoparticle(
    cif: str | Path,
    shape: str = "sphere",
    shape_params: Mapping[str, Any] | None = None,
    hollow: float = 0.0,
    centre: str = "cell",
    trim_loose: bool = True,
    stoichiometric: bool = False,
    shells: int = 0,
    gap: float = GRAPHITIC_GAP,
    interlayer: float = GRAPHITIC_GAP,
    dopant: str | None = None,
    dopant_conc: float = 0.0,
    dopant_site: str = "random",
    codope_spec: Sequence[tuple[str, float]] = (),
    codope_affinity: str = "random",
    graft: str | None = None,
    graft_coverage: float = 0.0,
    graft_where: str = "all",
    vacuum: float = DEFAULT_VACUUM_1D,
    seed: int = 0,
    max_atoms: int = 4_000_000,
    **coat_options: Any,
) -> tuple[Atoms, CrystalReport, CarveReport, CoatReport | None]:
    """Read a crystal, cut a particle out of it, and coat it.

    Parameters
    ----------
    cif
        The crystal file. Anything ASE reads; it has to describe a
        periodic crystal.
    shape, shape_params
        A name from :data:`~nanocarbon_lab.nanoparticle.shapes.SHAPES`
        and its parameters. Missing parameters take the catalogue's
        defaults, which are a size that builds quickly rather than a
        recommendation.
    hollow
        Non-zero makes the shape a cage with that wall thickness, Å.
    centre, trim_loose, stoichiometric, max_atoms
        Passed to
        :func:`~nanocarbon_lab.nanoparticle.carve.carve_nanoparticle`.
    codope_spec, codope_affinity
        Several heteroatoms in the shell at once; see
        :func:`~nanocarbon_lab.nanoparticle.coat.encapsulate`.
    shells
        ``0`` returns the bare core. More than that wraps it; see
        :func:`~nanocarbon_lab.nanoparticle.coat.encapsulate`.
    gap, interlayer, dopant, dopant_conc, dopant_site, graft,
    graft_coverage, graft_where, seed, **coat_options
        Passed to the coating. The dopant and the graft apply to the
        **shell**, never to the core.
    vacuum
        Padding in the finished cell, Å.

    Returns
    -------
    (atoms, crystal_report, carve_report, coat_report)
        ``coat_report`` is ``None`` when ``shells`` is 0. ``atoms.info``
        carries all three as plain dictionaries plus a ``verdict``.

    Raises
    ------
    ValueError
        From any of the four steps: no usable cell, an unknown shape, a
        cut that keeps nothing, a gap inside the core. Each says which.
    """
    crystal, crystal_report = read_crystal(cif)
    if hollow:
        cut = shape_module.hollowed(shape, hollow, **dict(shape_params or {}))
    else:
        cut = shape_module.make_shape(shape, **dict(shape_params or {}))

    core, carve_report = carve_nanoparticle(
        crystal, cut, centre=centre, trim_loose=trim_loose,
        stoichiometric=stoichiometric, vacuum=vacuum, max_atoms=max_atoms,
    )

    coat_report: CoatReport | None = None
    atoms = core
    if shells > 0:
        atoms, coat_report = encapsulate(
            core, gap=gap, shells=shells, interlayer=interlayer,
            dopant=dopant, dopant_conc=dopant_conc, dopant_site=dopant_site,
            codope_spec=codope_spec, codope_affinity=codope_affinity,
            graft=graft, graft_coverage=graft_coverage,
            graft_where=graft_where, seed=seed, vacuum=vacuum,
            **coat_options,
        )

    verdict, explanation = particle_quality(carve_report)
    atoms.info["crystal"] = {
        "source": str(crystal_report.source),
        "formula": crystal_report.formula,
        "cell_lengths": list(crystal_report.cell_lengths),
        "cell_angles": list(crystal_report.cell_angles),
        "density": crystal_report.density,
        "point_group_order": crystal_report.point_group_order,
        "lattice_system": crystal_report.lattice_system,
        "warnings": list(crystal_report.warnings),
    }
    atoms.info["particle_verdict"] = verdict
    atoms.info["particle_verdict_why"] = explanation
    return atoms, crystal_report, carve_report, coat_report


def estimate_nanoparticle_atoms(
    cif: str | Path,
    shape: str = "sphere",
    shape_params: Mapping[str, Any] | None = None,
    hollow: float = 0.0,
    shells: int = 0,
    gap: float = GRAPHITIC_GAP,
    interlayer: float = GRAPHITIC_GAP,
) -> int:
    """How many atoms the build will have, without building it.

    The core follows from the crystal's own density: the shape's measured
    volume over the unit-cell volume, times the atoms in that cell. That
    is exact up to the atoms the surface cuts in half, which is where the
    error lives, and the error is not symmetric between shapes. Measured
    against real builds on fcc gold: a 15 Å sphere estimates 2.7 % low, a
    24 Å cube **17 % high**. The cube is worse for the reason the carver
    warns about -- its faces land on lattice planes, so the cut excludes
    a whole plane of atoms per face, and a volume argument cannot know
    that. Read this as a size, not as a count.

    Each shell follows from its **area**, not its volume: a graphitic
    sheet is one atom thick, so a shell is its area over
    :data:`AREA_PER_CARBON`. The area is taken from the offset shape
    rather than from the core's envelope, which is the same thing to
    within the facet rounding and does not need the core built first.

    Returns ``0`` when the file cannot be read, rather than raising: an
    estimate is shown while the user is still typing the path.
    """
    try:
        crystal, _ = read_crystal(cif)
    except Exception:                           # noqa: BLE001 - a guess
        return 0
    import numpy as np

    try:
        if hollow:
            cut = shape_module.hollowed(shape, hollow,
                                        **dict(shape_params or {}))
        else:
            cut = shape_module.make_shape(shape, **dict(shape_params or {}))
    except ValueError:
        return 0
    cell_volume = float(abs(np.linalg.det(crystal.cell.array)))
    if cell_volume <= 0:                        # pragma: no cover - guarded
        return 0
    total = int(cut.volume() / cell_volume * len(crystal))
    for index in range(shells):
        grown = shape_module.offset(cut, gap + index * interlayer)
        total += int(_surface_area(grown) / AREA_PER_CARBON)
    return total


def _surface_area(shape: shape_module.Shape, resolution: int = 72) -> float:
    """Surface area of a shape's zero level set, Å^2.

    Counted the cheap way -- the volume of a thin shell around the
    surface, divided by its thickness -- because the only consumer is an
    atom-count estimate and meshing the surface properly costs more than
    the build it is estimating. Accurate to a few per cent, which is
    better than the estimate it feeds.
    """
    import numpy as np

    thickness = 2.0 * shape.extent / (resolution - 1) * 2.0
    axis = np.linspace(-shape.extent, shape.extent, int(resolution))
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    value = shape.field(grid)
    band = int(np.count_nonzero(np.abs(value) <= 0.5 * thickness))
    voxel = (2.0 * shape.extent / (resolution - 1)) ** 3
    return band * voxel / thickness


__all__ = [
    "AREA_PER_CARBON",
    "build_nanoparticle",
    "estimate_nanoparticle_atoms",
]
