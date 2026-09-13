"""Twisted bilayers and van der Waals heterostructures.

Sits above both :mod:`nanocarbon_lab.builders` and
:mod:`nanocarbon_lab.tmd` because it composes them: a stack may pair
graphene with hBN, or MoS2 with WS2, and the stacking code should not
have to know which family a layer came from. Layers are therefore
described uniformly by :class:`~nanocarbon_lab.hetero.moire.Layer2D`.
"""

from .moire import (
    MAX_MISMATCH,
    VDW_GAP,
    Layer2D,
    available_layers,
    build_twisted_bilayer,
    build_vdw_stack,
    cells_per_layer,
    commensurate_series,
    get_layer,
    nearest_commensurate,
    twist_angle,
)

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
    "nearest_commensurate",
    "twist_angle",
]
