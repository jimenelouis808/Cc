"""Substitutional dopants for nanocarbons (N, B, S, P, co-doping)."""

from .substitutional import (
    codope,
    dope_directed,
    dope_random,
    substitute_atoms,
)

__all__ = ["codope", "dope_directed", "dope_random", "substitute_atoms"]
