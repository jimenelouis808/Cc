"""Small-amplitude local distortion to break perfect symmetries.

Used as a pre-relaxation step so DFT / MD relaxations do not get stuck in
unstable saddle points.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from ase import Atoms

from ..utils.rng import make_rng


def apply_random_distortion(
    atoms: Atoms,
    amplitude: float = 0.05,
    seed: Optional[int] = None,
) -> Atoms:
    """Add isotropic Gaussian displacements to every atom.

    Parameters
    ----------
    atoms
        Input structure (not mutated).
    amplitude
        Standard deviation of the Gaussian displacement (Å). Must be small
        compared to the bond length (rule of thumb: <= 0.1 Å).
    seed
        RNG seed.

    Returns
    -------
    ase.Atoms
        Perturbed structure.
    """
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative.")
    if amplitude > 0.2:
        raise ValueError(
            f"amplitude={amplitude} Å is too large for nanocarbon (>0.2 Å)."
        )
    rng = make_rng(seed)
    out = atoms.copy()
    out.info = {**atoms.info}
    positions = out.get_positions()
    positions += rng.normal(0.0, amplitude, size=positions.shape)
    out.set_positions(positions)
    out.info.setdefault("defects", []).append(
        {"type": "random_distortion", "amplitude": amplitude, "seed": seed}
    )
    return out
