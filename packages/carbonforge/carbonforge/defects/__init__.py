"""Structural defects: vacancies, Stone-Wales, local distortion."""

from .vacancies import introduce_vacancies
from .stone_wales import stone_wales_defect
from .distortion import apply_random_distortion

__all__ = [
    "introduce_vacancies",
    "stone_wales_defect",
    "apply_random_distortion",
]
