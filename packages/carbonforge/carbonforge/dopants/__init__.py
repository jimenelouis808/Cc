"""Substitutional dopants for nanocarbons (N, B, S, P, co-doping)."""

from .substitutional import (
    substitute_atoms,
    dope_random,
    dope_directed,
    codope,
    dope_at_sites,
)

__all__ = ["substitute_atoms", "dope_random", "dope_directed", "codope", "dope_at_sites"]
