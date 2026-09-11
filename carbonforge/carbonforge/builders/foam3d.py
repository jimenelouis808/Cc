"""3D disordered carbon foam / aerogel-like builders.

The construction is intentionally stochastic and **pre-relaxation**: we
tile small graphitic flakes (hexagonal patches cut from graphene) at random
positions and orientations inside a cubic box, rejecting placements that
bring atoms of different flakes closer than ``min_distance``. The result is
a low-density, topologically disordered 3D carbon network that is a good
starting point for MD annealing (LAMMPS AIREBO/Tersoff) or ML-driven
structure optimisation.

This is **not** a fully bonded schwarzite: no attempt is made to saturate
dangling bonds. Downstream relaxation is assumed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms

from ..utils.constants import CC_BOND, HARD_MIN_DISTANCE
from ..utils.rng import make_rng


def _hex_flake(radius: float, bond: float) -> np.ndarray:
    """Return positions of a hexagonal graphene flake in the xy plane.

    Parameters
    ----------
    radius
        Flake radius in Å (atoms further than this are discarded).
    bond
        C-C bond length.

    Returns
    -------
    numpy.ndarray of shape (N, 3)
    """
    a = np.sqrt(3.0) * bond
    # Build a generous parallelogram of lattice points, then filter by radius.
    n_uc = int(np.ceil(radius / a)) + 2
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([a / 2.0, a * np.sqrt(3.0) / 2.0, 0.0])
    basis = np.array([[1.0 / 3.0, 1.0 / 3.0, 0.0], [2.0 / 3.0, 2.0 / 3.0, 0.0]])
    positions = []
    for i in range(-n_uc, n_uc + 1):
        for j in range(-n_uc, n_uc + 1):
            shift = i * a1 + j * a2
            for b in basis:
                p = shift + b[0] * a1 + b[1] * a2
                if p[0] ** 2 + p[1] ** 2 <= radius ** 2:
                    positions.append(p)
    if not positions:
        raise ValueError(f"Flake radius {radius} Å is too small for bond {bond} Å.")
    return np.array(positions)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix (Shoemake's quaternion method)."""
    u1, u2, u3 = rng.random(3)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
    )
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def build_carbon_foam(
    box_size: float = 30.0,
    n_flakes: int = 20,
    flake_radius: float = 4.0,
    min_distance: float = HARD_MIN_DISTANCE + 0.3,
    bond: float = CC_BOND,
    seed: Optional[int] = None,
    max_placement_tries: int = 2000,
) -> Atoms:
    """Build a 3D disordered carbon foam by random placement of graphene flakes.

    Parameters
    ----------
    box_size
        Edge of the cubic simulation box (Å). PBC is applied.
    n_flakes
        Target number of flakes. The actual number may be smaller if the
        placement solver runs out of retries.
    flake_radius
        Radius of each hexagonal flake (Å).
    min_distance
        Minimum allowed distance between atoms of *different* flakes (Å).
        Must be above :data:`HARD_MIN_DISTANCE` to keep the result physical.
    bond
        C-C bond length (Å).
    seed
        RNG seed for reproducibility.
    max_placement_tries
        Total number of rejection-sampling tries before giving up.

    Returns
    -------
    ase.Atoms
        Cubic, fully periodic structure. Metadata in ``atoms.info`` records
        the realised number of flakes and the final density (g/cm³).

    Raises
    ------
    ValueError
        If parameters are inconsistent (e.g. ``min_distance`` too small).
    """
    if min_distance < HARD_MIN_DISTANCE:
        raise ValueError(
            f"min_distance={min_distance} Å is below the hard limit {HARD_MIN_DISTANCE} Å."
        )
    if box_size < 3 * flake_radius:
        raise ValueError("box_size should be at least 3 * flake_radius.")

    rng = make_rng(seed)
    cell = np.eye(3) * box_size
    placed: list[np.ndarray] = []

    base_flake = _hex_flake(flake_radius, bond)
    tries = 0
    while len(placed) < n_flakes and tries < max_placement_tries:
        tries += 1
        R = _random_rotation(rng)
        t = rng.random(3) * box_size
        candidate = (base_flake @ R.T) + t

        # Minimum-image check against already placed atoms.
        ok = True
        for existing in placed:
            delta = candidate[:, None, :] - existing[None, :, :]
            # Apply minimum-image convention on the cubic box.
            delta -= box_size * np.round(delta / box_size)
            dmin = np.linalg.norm(delta, axis=-1).min()
            if dmin < min_distance:
                ok = False
                break
        if ok:
            placed.append(candidate)

    if not placed:
        raise RuntimeError(
            "Could not place any flake. Increase box_size or lower min_distance."
        )

    positions = np.concatenate(placed, axis=0) % box_size
    atoms = Atoms(
        symbols=["C"] * len(positions),
        positions=positions,
        cell=cell,
        pbc=True,
    )

    # Density in g/cm^3 (carbon mass 12.011 u, 1 u/Å^3 = 1.66054 g/cm^3).
    mass_g_per_cm3 = len(atoms) * 12.011 / (box_size ** 3) * 1.66054
    atoms.info.update(
        {
            "structure_type": "carbon_foam",
            "n_flakes_requested": n_flakes,
            "n_flakes_placed": len(placed),
            "flake_radius": flake_radius,
            "box_size": box_size,
            "density_g_cm3": mass_g_per_cm3,
            "seed": seed,
            "bond": bond,
        }
    )
    return atoms
