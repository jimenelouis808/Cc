"""A field of anemones: one hero in front, a few smaller ones behind,
and FeSe rods and plates lying about on the ground between them.

    uv run python modelos_SEM/escena_campo.py
    uv run python modelos_SEM/escena_campo.py --fondo 3 --barras 40

Laid out for a camera on -y looking along +y, low (`portada_blend.py
--style revista --elev 10`), with the mirror floor under everything:

* **the hero** (`xyz/anemona_conjunta.extxyz`, 18 arms) stands well in
  front, so a short lens (`--lente 24`) makes it loom over the rest;
* **the background** is `xyz/anemona_mini.extxyz` (12 short arms, about
  half the atoms) placed two to three hero radii behind, each turned
  differently so the copies do not read as one;
* **the ground** carries rods lying down at random headings and a few
  plates, flat or leaning, scattered between and around them.

Pieces on the ground sit on one plane, so the floor the renderer puts
under the lowest atom is the ground they lie on. Every piece is checked
against everything already placed and is re-drawn if it comes closer
than `holgura`; how many made it is printed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import construir
import numpy as np
from anemona_tubos import rotation_to
from ase import Atoms
from ase.io import read, write
from escena_fese import RODS, make_rod, picture
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent

OPT: dict[str, float] = {
    "fondo": 3,          # background anemones
    "barras": 36,
    "placas": 5,
    "holgura": 5.0,      # A, closest any two pieces may come
    "semilla": 2,
}

#: Where the background anemones stand: (x, y, lift), y into the page.
#: Lift is above the ground, so they rest at different heights.
#: Wide to the sides on purpose: seen through a short lens from close
#: in, anything behind the hero within about two of its radii of the
#: axis is hidden by it.
BACK = ((-430.0, 260.0, 10.0), (470.0, 380.0, 0.0), (-140.0, 760.0, 30.0),
        (720.0, 820.0, 20.0))

TRIES = 300


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])
    rng = np.random.default_rng(int(OPT["semilla"]))
    up = np.array([0.0, 0.0, 1.0])
    gap = OPT["holgura"]

    scene, groups = Atoms(), []
    placed = np.empty((0, 3))

    def add(symbols, positions, label):
        nonlocal scene, placed
        scene += Atoms(symbols, positions)
        groups.extend([label] * len(positions))
        placed = np.vstack((placed, positions))

    def clear(positions) -> bool:
        return not len(placed) or float(
            cKDTree(placed).query(positions)[0].min()) >= gap

    hero = read(str(HERE / "xyz" / "anemona_conjunta.extxyz"))
    h_pos = hero.positions - hero.positions.mean(axis=0)
    h_pos = h_pos @ rotation_to(up, np.array([0.15, -0.2, 1.0]), 0.3).T
    ground = float(h_pos[:, 2].min()) - 2.0
    add(hero.get_chemical_symbols(), h_pos + [30.0, -420.0, 0.0], 0)
    print(f"protagonista: {len(hero)} átomos, suelo a z = {ground:.0f} Å")

    mini = read(str(HERE / "xyz" / "anemona_mini.extxyz"))
    m_pos = mini.positions - mini.positions.mean(axis=0)
    made = 0
    for x, y, lift in BACK[:int(OPT["fondo"])]:
        for _ in range(TRIES):
            turn = construir._random_rotation(rng)
            local = m_pos @ turn.T
            where = np.array([x, y, ground + lift - local[:, 2].min()])
            trial = local + where + rng.normal(scale=12.0, size=3) * [1, 1, 0]
            if clear(trial):
                add(mini.get_chemical_symbols(), trial, 1)
                made += 1
                break
    print(f"fondo: {made} anémonas de {len(mini)} átomos")

    # --- the ground: rods lying down, plates flat or leaning.
    rods = [make_rod(*shape) for shape in RODS[:2]]
    plate = read(str(HERE / "xyz" / "placa_H_mixta.xyz"))
    p_pos = plate.positions - plate.positions.mean(axis=0)
    span_x, span_y = (-640.0, 640.0), (-700.0, 700.0)
    rods_made = plates_made = 0
    for k in range(int(OPT["barras"]) + int(OPT["placas"])):
        is_plate = k % max(1, round((OPT["barras"] + OPT["placas"])
                                    / max(OPT["placas"], 1))) == 0 \
            and plates_made < OPT["placas"]
        for _ in range(TRIES):
            if is_plate:
                tilt = rng.normal(scale=0.35, size=3) * [1, 1, 0] + up
                local = p_pos @ rotation_to(up, tilt / np.linalg.norm(tilt),
                                            rng.uniform(0, 6.3)).T
                symbols = plate.get_chemical_symbols()
            else:
                piece = rods[int(rng.integers(len(rods)))]
                azim = rng.uniform(0, 2 * np.pi)
                heading = np.array([np.cos(azim), np.sin(azim),
                                    rng.normal(scale=0.08)])
                local = piece.positions @ rotation_to(
                    up, heading / np.linalg.norm(heading),
                    rng.uniform(0, 6.3)).T
                symbols = piece.get_chemical_symbols()
            where = np.array([rng.uniform(*span_x), rng.uniform(*span_y),
                              ground - local[:, 2].min()])
            trial = local + where
            if clear(trial):
                add(symbols, trial, 3 if is_plate else 2)
                if is_plate:
                    plates_made += 1
                else:
                    rods_made += 1
                break
    print(f"suelo: {rods_made} barras y {plates_made} placas")

    scene.set_array("grupo", np.asarray(groups, dtype=int))
    scene.positions -= 0.5 * (scene.positions.max(axis=0)
                              + scene.positions.min(axis=0))
    closest = float(cKDTree(scene.positions).query(scene.positions,
                                                   k=2)[0][:, 1].min())
    name = "escena_campo"
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"campo de anémonas con barras y placas de FeSe: "
                  f"{scene.get_chemical_formula()}; piezas COLOCADAS")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene, format="extxyz")
    picture(scene, HERE / "img" / f"{name}.png", 12.0, 0.0)
    print(f"\n{name}: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, caja "
          f"{np.ptp(scene.positions, axis=0).round(0)} Å")
    print("  (.extxyz lleva `grupo`: 0 protagonista, 1 fondo, 2 barras, "
          "3 placas)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
