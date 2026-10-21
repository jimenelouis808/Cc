"""Shared setup for the two samples."""
import numpy as np
from ramancarbon.xrd.io import read_pattern
from ramancarbon.xrd.reference import load_library

UP = "/root/.claude/uploads/0495242d-7cd2-5e51-b6e4-d3296f40ec00"
FILES = {
    "S1_900": f"{UP}/cf4cc304-3_S1_I_Se_900_2theta_10-90_BB_Dtex_110626.ras",
    "S2_1020": f"{UP}/e08df46a-2_S2_1_Se_2theta_10-90_BB_Dtex_110626.ras",
}
# What the user declares in each sample.
DECLARED = {
    "S1_900": ["FeSe_tetragonal", "FeSe_hexagonal", "Fe3C_cementita",
               "Fe_alfa", "grafito_2H"],
    "S2_1020": ["C_turbostratico_3.44", "Fe3C_cementita", "Fe_alfa",
                "FeSe_tetragonal"],
}


def library_map():
    return {e.crystal.name: e.crystal for e in load_library()}


def load(name):
    p = read_pattern(FILES[name])
    p.name = name
    return p
