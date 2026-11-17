"""The anemone, the FeSe plates and the FeSe rods in one composed frame.

    uv run python modelos_SEM/escena_conjunta.py
    uv run python modelos_SEM/escena_conjunta.py --placas 5 --barras 9

Laid out for a PORTRAIT camera looking along +y (the `front` view of
`portada_blend.py`): x across the page, z up it, y into it.

* **The anemone** (`xyz/anemona_conjunta.extxyz`, a compact 18-arm one
  from `anemona_tubos.py`) sits upper right and a little back -- the
  subject, on the upper third.
* **The plates** (`xyz/placa_H_C.xyz`, wrapped all the way round) rise
  in an arc up the left, each turned a step further toward the viewer,
  so the column reads from edge to face as the eye climbs.
* **The rods** fan across the foreground at the bottom, lying down and
  pointing into the page, their angles spread evenly about the depth
  axis so the fan is ordered rather than spilled.

Nothing is relaxed together. Every piece is checked against everything
already placed and, if it comes closer than `holgura`, slid away from
the centre of the frame until it clears; the slide is printed.
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
    "placas": 4,
    "barras": 7,
    "holgura": 6.0,     # A, closest any two pieces may come
    "semilla": 0,
}


def place(piece: np.ndarray, everything: np.ndarray, away: np.ndarray,
          gap: float) -> tuple[np.ndarray, float]:
    """`piece` slid along unit `away` until it clears `everything`."""
    shift = 0.0
    if len(everything):
        tree = cKDTree(everything)
        while float(tree.query(piece + away * shift)[0].min()) < gap:
            shift += 2.0
    return piece + away * shift, shift


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])
    rng = np.random.default_rng(int(OPT["semilla"]))
    gap = OPT["holgura"]
    up = np.array([0.0, 0.0, 1.0])

    scene, groups = Atoms(), []
    everything = np.empty((0, 3))

    def add(symbols, positions, label, name):
        nonlocal scene, everything
        centre = positions.mean(axis=0)
        away = centre - np.array([0.0, 0.0, 40.0])
        away[1] *= 0.3
        away /= max(np.linalg.norm(away), 1e-9)
        moved, shift = place(positions, everything, away, gap)
        if shift:
            print(f"  {name}: apartada {shift:.0f} Å para no tocar")
        scene += Atoms(symbols, moved)
        groups.extend([label] * len(moved))
        everything = np.vstack((everything, moved))

    # --- the anemone: upper right, a little back.
    anemone = read(str(HERE / "xyz" / "anemona_conjunta.extxyz"))
    a_pos = anemone.positions - anemone.positions.mean(axis=0)
    a_pos = a_pos @ rotation_to(up, np.array([0.2, -0.3, 1.0]), 0.6).T
    add(anemone.get_chemical_symbols(), a_pos + [85.0, 40.0, 95.0],
        0, "anémona")
    print(f"anémona: {len(anemone)} átomos")

    # --- the plates: an arc up the left, edge-on at the bottom and
    # turning to face the viewer as it climbs.
    plate = read(str(HERE / "xyz" / "placa_H_C.xyz"))
    p_pos = plate.positions - plate.positions.mean(axis=0)
    count = int(OPT["placas"])
    for k in range(count):
        t = k / max(count - 1, 1)
        normal = np.array([0.25, -np.sin(np.radians(15 + 70 * t)),
                           np.cos(np.radians(15 + 70 * t))])
        turned = p_pos @ rotation_to(up, normal / np.linalg.norm(normal),
                                     np.radians(17.0 * k)).T
        where = np.array([-150.0 + 35.0 * np.sin(np.pi * t),
                          -30.0 * t, -45.0 + 78.0 * k])
        add(plate.get_chemical_symbols(), turned + where, 1, f"placa {k}")
    print(f"placas: {count} de {len(plate)} átomos")

    # --- the rods: a fan in the foreground, lying down.
    rods = [make_rod(*shape) for shape in RODS]
    count = int(OPT["barras"])
    for k in range(count):
        piece = rods[(k * 2) % len(rods)]
        t = k / max(count - 1, 1)
        # Lying on the "table" and pointing into the page, fanned about
        # the depth axis: a rod seen nearly end-on takes little width,
        # so seven of them fit across the foreground without touching.
        angle = np.radians(-34.0 + 68.0 * t)
        heading = np.array([np.sin(angle), np.cos(angle), 0.18])
        turned = piece.positions @ rotation_to(
            up, heading / np.linalg.norm(heading),
            rng.uniform(0, 2 * np.pi)).T
        where = np.array([-150.0 + 300.0 * t,
                          -60.0 + 30.0 * abs(t - 0.5),
                          -150.0 + 45.0 * (2.0 * t - 1.0) ** 2])
        add(piece.get_chemical_symbols(), turned + where, 2, f"barra {k}")
    print(f"barras: {count}")

    scene.set_array("grupo", np.asarray(groups, dtype=int))
    scene.positions -= 0.5 * (scene.positions.max(axis=0)
                              + scene.positions.min(axis=0))
    closest = float(cKDTree(scene.positions).query(scene.positions,
                                                   k=2)[0][:, 1].min())
    name = "escena_conjunta"
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"anémona, placas y barras de FeSe: "
                  f"{scene.get_chemical_formula()}; piezas COLOCADAS")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene, format="extxyz")
    picture(scene, HERE / "img" / f"{name}.png", 0.0, 0.0)
    span = np.ptp(scene.positions, axis=0)
    print(f"\n{name}: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, caja {span.round(0)} Å")
    print("  (.extxyz lleva `grupo`: 0 anémona, 1 placas, 2 barras)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
