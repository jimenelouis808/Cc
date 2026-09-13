"""Reproducible random number generation."""

from __future__ import annotations

import numpy as np


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Build a NumPy random generator with an explicit, reproducible seed.

    Parameters
    ----------
    seed
        Integer seed. ``None`` draws a fresh seed from the OS entropy pool; in
        that case the seed is **not** recorded and the run is not reproducible.

    Returns
    -------
    numpy.random.Generator
        A PCG64-backed generator.
    """
    return np.random.default_rng(seed)
