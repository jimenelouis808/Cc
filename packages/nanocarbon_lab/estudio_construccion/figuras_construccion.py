"""Write the clean crystalline cells and draw every superstructure with its
defects coloured (pentagons red, heptagons blue, other non-hexagons amber).

    python figuras_construccion.py <carbonforge validacion_rendijas folder>
        -> perfectas/*.extxyz, figuras/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
from ase.io import read, write
from ase.neighborlist import neighbor_list

from nanocarbon_lab.builders.knee import build_knee_schwarzite, build_knee_supernetwork

from auditoria import FILES, rings

HERE = Path(__file__).resolve().parent
#: The crystalline cells that come out CLEAN (barrido_perfectas.jsonl, auditoria.json).
PERFECT = {
    "schwarzite-P_k20_n7": lambda: build_knee_schwarzite(circumference=20, arm_rows=7, kind="primitive"),
    "schwarzite-D_k18_n5": lambda: build_knee_schwarzite(kind="diamond"),
    "gyroid_k20_n5": lambda: build_knee_schwarzite(kind="gyroid"),
    "super-graphene_k22_n5": lambda: build_knee_supernetwork("super-graphene"),
    "super-square_k12_n5": lambda: build_knee_supernetwork("super-square", circumference=12, arm_rows=5),
}
COLOURS = {5: (0.84, 0.19, 0.15), 7: (0.16, 0.38, 0.75)}
OTHER, HEX = (0.93, 0.62, 0.10), (0.78, 0.80, 0.82)


def colours(atoms):
    i, j = neighbor_list("ij", atoms, 1.8)
    g = nx.Graph([(a, b) for a, b in zip(i, j) if a < b])
    col = [HEX] * len(atoms)
    for ring in rings(g):
        c = COLOURS.get(len(ring), OTHER if len(ring) != 6 else None)
        if c is None:
            continue
        for a in ring:
            if col[a] == HEX or c == OTHER:
                col[a] = c
    return col


def draw(atoms, path: Path) -> None:
    atoms = atoms.copy()
    if atoms.pbc.all():
        rotation = "-25x,30y"
    elif atoms.pbc.any():
        # A sheet: 2x2 cells (3x3 if small), seen 35 deg off its normal.
        n = 3 if len(atoms) < 500 else 2
        atoms = atoms.repeat((n, n, 1))
        rotation = "-35x"
    else:
        rotation = "-25x,30y"
    write(str(path), atoms, rotation=rotation, colors=colours(atoms),
          radii=0.75, scale=12 if len(atoms) > 3000 else 18, show_unit_cell=0)


def main(source: Path) -> None:
    (HERE / "perfectas").mkdir(exist_ok=True)
    (HERE / "figuras").mkdir(exist_ok=True)
    for name, build in PERFECT.items():
        target = HERE / "perfectas" / f"{name}.extxyz"
        if not target.exists():
            atoms = build()
            for key in ("rings", "bonds", "disclination_check", "geometry"):
                atoms.info.pop(key, None)
            atoms.info = {k: v for k, v in atoms.info.items()
                          if isinstance(v, (int, float, str, bool))}
            write(str(target), atoms)
        draw(read(str(target)), HERE / "figuras" / f"perfecta_{name}.png")
        print("perfect", name, flush=True)
    for label, rel in FILES.items():
        stem = label.split(" (")[0].replace(" ", "-") + ("_mallada" if "meshed" in label else "_cristalina")
        draw(read(str(source / rel)), HERE / "figuras" / f"{stem}.png")
        print("drawn", label, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
