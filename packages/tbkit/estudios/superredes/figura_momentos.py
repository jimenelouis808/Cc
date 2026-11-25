"""Draw the mean-field moments on the structure: red up, blue down,
intensity by |m| (scaled to each panel's maximum).

    python figura_momentos.py   -> figuras/momentos_*.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ase.io import write

from magnetismo import structure

HERE = Path(__file__).resolve().parent
CASES = {
    "momentos_super-graphene_2t_k8": ("knee-super-graphene", "magnetismo_k8.json", None),
    "momentos_super-graphene_2.5t": ("knee-super-graphene", "magnetismo_U.json", "knee-super-graphene | U=2.5|t|"),
    "momentos_super-square_2.5t": ("knee-super-square", "magnetismo_U.json", "knee-super-square | U=2.5|t|"),
}


def colour(m, top):
    x = min(abs(m) / top, 1.0)
    base = np.array([0.80, 0.82, 0.84])
    tint = np.array([0.80, 0.12, 0.10]) if m > 0 else np.array([0.10, 0.30, 0.78])
    return tuple(base + (tint - base) * x)


def main() -> None:
    for out, (name, source, key) in CASES.items():
        data = json.loads((HERE / source).read_text())
        moments = np.array((data[key] if key else data)["moments"])
        atoms = structure(name)
        top = float(np.abs(moments).max())
        rep = atoms.repeat((2, 2, 1))
        m = np.tile(moments, 4)
        write(str(HERE / "figuras" / f"{out}.png"), rep, rotation="-30x",
              colors=[colour(x, top) for x in m], radii=0.8, scale=14, show_unit_cell=0)
        print(out, "max", top)


if __name__ == "__main__":
    main()
