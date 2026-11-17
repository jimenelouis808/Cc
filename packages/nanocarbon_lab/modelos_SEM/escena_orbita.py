"""The anemone ringed by FeSe plates and rods, like a planet and its belt.

    uv run python modelos_SEM/escena_orbita.py
    uv run python modelos_SEM/escena_orbita.py --piezas 14 --radio 290

The second combined composition (`escena_conjunta.py` is the first). The
anemone (`xyz/anemona_conjunta.extxyz`) is the planet; round it runs one
tilted ring of alternating pieces:

* **plates** (`xyz/placa_H_mixta.xyz`, the evenly cut plate in its full
  shell), each facing out from the centre, so the ring shows them
  face-on at its sides and edge-on where it passes in front and behind;
* **rods** (from `escena_fese.make_rod`), lying along the ring, so they
  read as the dashes of an orbit.

Made for `portada_blend.py --style revista`, whose mirror floor gives
the ring back underneath. Nothing is relaxed together; every piece is
checked against all the rest and pushed outward along its radius until
it clears `holgura`, and any push is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from anemona_tubos import rotation_to
from ase import Atoms
from ase.io import read, write
from escena_fese import RODS, make_rod, picture
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent

OPT: dict[str, float] = {
    "piezas": 12,       # pieces in the ring, plates and rods alternating
    "radio": 240.0,     # A, ring radius
    "inclinacion": 16.0,  # degrees the ring is tipped toward the camera
    "holgura": 6.0,     # A, closest any two pieces may come
    "semilla": 0,
}


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])
    rng = np.random.default_rng(int(OPT["semilla"]))
    up = np.array([0.0, 0.0, 1.0])
    tip = np.radians(OPT["inclinacion"])
    # The ring's plane, tipped about x toward a camera on -y.
    tilt = np.array([[1.0, 0.0, 0.0],
                     [0.0, np.cos(tip), -np.sin(tip)],
                     [0.0, np.sin(tip), np.cos(tip)]])

    scene, groups = Atoms(), []
    anemone = read(str(HERE / "xyz" / "anemona_conjunta.extxyz"))
    a_pos = anemone.positions - anemone.positions.mean(axis=0)
    scene += Atoms(anemone.get_chemical_symbols(), a_pos)
    groups.extend([0] * len(anemone))
    everything = a_pos.copy()
    print(f"anémona: {len(anemone)} átomos")

    plate = read(str(HERE / "xyz" / "placa_H_mixta.xyz"))
    p_pos = plate.positions - plate.positions.mean(axis=0)
    rods = [make_rod(*shape) for shape in RODS[:2]]   # the two shorter

    count = int(OPT["piezas"])
    phase = rng.uniform(0.0, 2.0 * np.pi / count)
    for k in range(count):
        angle = phase + 2.0 * np.pi * k / count
        radial = tilt @ np.array([np.cos(angle), np.sin(angle), 0.0])
        tangent = tilt @ np.array([-np.sin(angle), np.cos(angle), 0.0])
        if k % 2 == 0:
            # A plate, facing out, turned a little about its normal.
            local = p_pos @ rotation_to(up, radial, 0.4 * k).T
            symbols, label = plate.get_chemical_symbols(), 1
        else:
            piece = rods[(k // 2) % len(rods)]
            local = piece.positions @ rotation_to(up, tangent,
                                                  rng.uniform(0, 6.3)).T
            symbols, label = piece.get_chemical_symbols(), 2
        tree = cKDTree(everything)
        push = 0.0
        while float(tree.query(local + radial * (OPT["radio"] + push))[0]
                    .min()) < OPT["holgura"]:
            push += 3.0
        if push:
            print(f"  pieza {k}: empujada {push:.0f} Å hacia fuera")
        placed = local + radial * (OPT["radio"] + push)
        scene += Atoms(symbols, placed)
        groups.extend([label] * len(placed))
        everything = np.vstack((everything, placed))

    scene.set_array("grupo", np.asarray(groups, dtype=int))
    scene.positions -= 0.5 * (scene.positions.max(axis=0)
                              + scene.positions.min(axis=0))
    closest = float(cKDTree(scene.positions).query(scene.positions,
                                                   k=2)[0][:, 1].min())
    name = "escena_orbita"
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"anémona con un anillo de placas y barras de FeSe: "
                  f"{scene.get_chemical_formula()}; piezas COLOCADAS")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene, format="extxyz")
    picture(scene, HERE / "img" / f"{name}.png", 20.0, 0.0)
    print(f"\n{name}: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, caja "
          f"{np.ptp(scene.positions, axis=0).round(0)} Å")
    print("  (.extxyz lleva `grupo`: 0 anémona, 1 placas, 2 barras)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
