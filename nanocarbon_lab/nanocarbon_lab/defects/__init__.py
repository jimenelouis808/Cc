"""Structural defects: vacancies, Stone-Wales, local distortion."""

from .distortion import apply_random_distortion
from .stone_wales import stone_wales_defect
from .vacancies import introduce_vacancies

__all__ = [
    "apply_random_distortion",
    "introduce_vacancies",
    "stone_wales_defect",
]
