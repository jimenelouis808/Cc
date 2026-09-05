"""Periodic 3D networks of interconnected carbon nanotubes.

A schwarzite gets its negative curvature from a minimal surface, which is
a smooth sponge with no straight sections anywhere. A **nanotube network**
is the other way round: straight tubes joined at discrete nodes, which is
what a 3D carbon architecture grown from a template or assembled from
junctions actually looks like. Both are periodic, and both are built by
the same route -- implicit field, periodic marching cubes, isotropic
remesh, dual -- so the topology is derived rather than prescribed.

What that derivation gives here is worth stating, because it is the whole
argument for building these implicitly instead of gluing junctions
together by hand:

* The straight sections come out **all-hexagon**, exactly as a real tube
  wall is.
* The nodes come out with **heptagons and octagons**, because a node is a
  saddle and a saddle carries negative Gaussian curvature. Nobody puts
  them there.
* The count is not a matter of taste: ``sum(6 - n) = 6*chi``, and for a
  periodic cell chi is fixed by the net's own topology. A cubic cell of
  6-coordinate nodes and a diamond cell of 4-coordinate ones have
  different genus and therefore different ring budgets, and the builder
  checks the one against the other.

The two nets are not interchangeable. The cubic net joins three tubes at
90 degrees, which is the sharpest branch a graphitic sheet is ever asked
to cover; the diamond net joins four at 109.47 degrees, which is the
angle sp2 carbon wants anyway. The diamond nodes are correspondingly
gentler, at the cost of eight nodes per cell instead of one -- so it
needs a much larger cell to hold the same tube length, and many more
atoms.
"""

from __future__ import annotations

import math

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND
from ..utils.rng import make_rng
from . import implicit as im
from . import remesh as rm
from .junction import _finish

#: Strut length as a fraction of the cell edge, per net. Cubic struts run
#: edge to edge; diamond struts are the tetrahedral quarter-diagonal.
STRUT_FRACTION = {"cubic": 1.0, "diamond": math.sqrt(3.0) / 4.0}


def minimum_cell(kind: str, tube_radius: float, blend: float) -> float:
    """Smallest cell that leaves a real tube between two nodes.

    Each node consumes about ``tube_radius + blend`` of the strut at
    either end. Below the length where what remains is shorter than the
    tube is wide, there is no tube -- only two nodes touching, and the
    "network of nanotubes" is a sponge with a different name. This is the
    honest floor, and the builder refuses below it rather than returning
    something that photographs well and is not what was asked for.
    """
    try:
        fraction = STRUT_FRACTION[kind]
    except KeyError:
        raise ValueError(
            f"Unknown network {kind!r}; expected one of {list(STRUT_FRACTION)}."
        ) from None
    return (3.0 * tube_radius + 2.0 * blend) / fraction


def build_nanotube_network(
    kind: str = "cubic",
    cell: float = 40.0,
    tube_radius: float = 6.0,
    blend: float = 5.0,
    bond: float = CC_BOND,
    grid_resolution: int = 72,
    remesh_iterations: int = 25,
    anneal_sweeps: int = 0,
    roughness: float = 0.0,
    relax_iterations: int = 3000,
    seed: int | None = 0,
) -> Atoms:
    """Build a periodic unit cell of interconnected carbon nanotubes.

    Parameters
    ----------
    kind
        ``"cubic"`` -- one 6-coordinate node per cell, tubes along the
        three axes. The classic 3D nanotube scaffold, and the harsher of
        the two: three tubes crossing at right angles.

        ``"diamond"`` -- eight 4-coordinate nodes, tubes along the
        tetrahedral directions at 109.47 degrees, which is the angle sp2
        carbon adopts at a branch anyway. Gentler nodes, but eight of
        them per cell, so it needs a much larger cell and many more atoms
        for the same tube width.
    cell
        Cubic cell edge in Å. Must leave a real tube between nodes; see
        :func:`minimum_cell`.
    tube_radius
        Radius of each tube. Unlike a rolled ``(n, m)`` tube this is a
        free parameter, because the wall is meshed rather than wrapped --
        the lattice adapts to the radius instead of quantising it.
    blend
        Smooth-union radius at the nodes. Too small leaves a crease with
        curvature no hexagonal net can tile; too large rounds the node
        into a sphere and the tubes stop being tubes.
    bond, remesh_iterations, relax_iterations, roughness, seed
        As for :func:`~nanocarbon_lab.builders.junction.build_schwarzite`.
    anneal_sweeps
        Defaults to 0, and for the schwarzite's reason: at a node the
        5-7 pairs *are* how a hexagonal net covers the curvature, so
        annealing them away only makes the remaining bonds stretch.
    grid_resolution
        Grid points across the cell. Higher than the schwarzite default
        because a network has thin necks between wide tubes, and a coarse
        grid welds them shut.

    Returns
    -------
    ase.Atoms
        A genuinely periodic cell -- ``pbc=(True, True, True)``, cubic --
        ready for a periodic DFT or MD code, with the ring census, the
        Euler budget and the node geometry in ``atoms.info``.

    Raises
    ------
    ValueError
        For an unknown net, or a cell too small to hold a tube.
    RuntimeError
        If the periodic weld leaves the surface open at every resolution
        tried.
    """
    if tube_radius <= 0 or blend <= 0:
        raise ValueError("tube_radius and blend must be positive.")
    floor = minimum_cell(kind, tube_radius, blend)
    if cell < floor:
        strut = STRUT_FRACTION[kind] * cell
        raise ValueError(
            f"cell={cell:.1f} Å is too small for a {kind!r} network of "
            f"{tube_radius:.1f} Å tubes: the struts are {strut:.1f} Å long "
            f"and each node eats about {tube_radius + blend:.1f} Å of either "
            f"end, so nothing recognisable as a tube is left between them. "
            f"Use cell >= {floor:.0f} Å, or a narrower tube."
        )

    field, _ = im.network_field(kind, cell=cell, tube_radius=tube_radius,
                                blend=blend)

    # Same grid retry as the schwarzite: whether a neck is resolved
    # depends on how the surface falls between sample points, so one
    # (cell, resolution) pair can tear where both its neighbours are fine.
    failures: list[str] = []
    for attempt, resolution in enumerate(
        (grid_resolution, grid_resolution + 8, grid_resolution + 16)
    ):
        mesh = rm.periodic_marching_cubes_mesh(field, cell, resolution=resolution)
        stats = rm.mesh_statistics(mesh)
        if stats["boundary_edges"]:
            failures.append(
                f"resolution {resolution}: periodic weld left "
                f"{stats['boundary_edges']} boundary edges"
            )
            continue
        try:
            nodes, bonds = im._NETWORKS[kind]
            return _finish(
                rm.isotropic_remesh(
                    mesh, field, target_edge=np.sqrt(3.0) * bond,
                    iterations=remesh_iterations, box=cell,
                    anneal_sweeps=anneal_sweeps, rng=make_rng(seed),
                ),
                bond=bond,
                relax_iterations=relax_iterations,
                vacuum=0.0,
                box=cell,
                roughness=roughness,
                rng=make_rng(seed),
                info={
                    "structure_type": "nanotube_network",
                    "network_kind": kind,
                    "cell": cell,
                    "tube_radius": tube_radius,
                    "blend": blend,
                    "bond": bond,
                    "n_nodes": len(nodes),
                    "n_struts": len(bonds),
                    "node_coordination": 2 * len(bonds) // len(nodes),
                    "strut_length": STRUT_FRACTION[kind] * cell,
                    "anneal_sweeps": anneal_sweeps,
                    "roughness": roughness,
                    "grid_resolution": resolution,
                    "grid_retries": attempt,
                },
            )
        except RuntimeError as exc:
            failures.append(f"resolution {resolution}: {exc}")

    raise RuntimeError(
        f"Could not build a valid {kind!r} network at cell={cell:.1f} Å, "
        f"tube_radius={tube_radius:.1f} Å after {len(failures)} grid "
        "resolutions:\n  " + "\n  ".join(failures)
        + "\nA wider cell or a narrower tube gives the necks more room."
    )


__all__ = ["STRUT_FRACTION", "build_nanotube_network", "minimum_cell"]
