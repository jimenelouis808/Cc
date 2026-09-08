"""Raman maps: a cube of spectra, and what to do with forty thousand of them.

A map is not a list of spectra. It is two spatial axes and one spectral
one, and treating it as a cube is what makes the useful operations cheap
enough to be interactive.

:class:`~ramancarbon.mapping.cube.RamanMap`
    The cube, its axes, its missing pixels and its sampling.
:mod:`~ramancarbon.mapping.io`
    Reading the two text layouts instruments export, long and wide.
:mod:`~ramancarbon.mapping.images`
    One number per pixel — a band's area, a ratio, a position, a width —
    each with the guard that keeps it from being a beautiful picture of
    the background.
:mod:`~ramancarbon.mapping.chemometrics`
    Finding the components without being told: PCA, k-means, MCR-ALS,
    and the cosmic-ray removal that has to come before all three.
"""

from __future__ import annotations

from .chemometrics import (
    Clustering,
    Decomposition,
    despike_map,
    kmeans,
    mcr_als,
    pca,
    suggested_components,
)
from .cube import MapError, RamanMap, from_spectra
from .images import (
    PropertyMap,
    band_intensity,
    band_position,
    band_ratio,
    band_width,
    coverage,
    noise_level,
    signal_to_noise,
)
from .io import read_map, write_map

__all__ = [
    "Clustering",
    "Decomposition",
    "MapError",
    "PropertyMap",
    "RamanMap",
    "band_intensity",
    "band_position",
    "band_ratio",
    "band_width",
    "coverage",
    "despike_map",
    "from_spectra",
    "kmeans",
    "mcr_als",
    "noise_level",
    "pca",
    "read_map",
    "signal_to_noise",
    "suggested_components",
    "write_map",
]
