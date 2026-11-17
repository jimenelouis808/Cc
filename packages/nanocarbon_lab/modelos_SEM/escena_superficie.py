"""A close-up of a bare FeSe surface, with the precursors landing on it.

    uv run python modelos_SEM/escena_superficie.py
    uv run python modelos_SEM/escena_superficie.py --moleculas 60

The macro shot `escena_sintesis.py` is the wide shot of. No carbon
shell: the subject is the FeSe itself.

* **The plate** is the evenly cut core of `xyz/placa_H_mixta.xyz`, lying
  face up, so its broad face -- one flat {10-10} plane of Fe and Se at
  the same height -- is what the camera looks down on. Centred, so the
  hexagon sits symmetric in the frame.
* **Two bare rods** (`xyz/barra_T_nucleo_desnudo.xyz`) stand behind it
  in a V, mirror images of each other about the centre line.
* **The precursors** at TRUE scale (in a close-up they need no
  magnifying): acetonitrile, ferrocene and selenourea whole higher up,
  in fragments lower down, and single Fe, Se, C and N atoms resting on
  the surface at a bonding distance. All placed at random -- the
  symmetry is the composition's, not the molecules'.

Every piece is checked against everything already placed and re-drawn
if it comes closer than `holgura`. Made for `portada_blend.py --style
revista --calor --bola 4 5 6 --encuadrar 0`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import construir
import numpy as np
from anemona_tubos import rotation_to
from ase import Atoms
from ase.io import read, write
from escena_fese import picture
from escena_sintesis import PRECURSORS
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent

OPT: dict[str, float] = {
    "moleculas": 46,
    "adatomos": 34,      # single atoms resting on the surface
    "holgura": 2.6,      # A, between molecules and from the solids
    "semilla": 3,
}

#: A, an adatom's height above the nearest surface atom: a bond, not a
#: van der Waals contact, since it has landed and stuck.
ADATOM_HEIGHT = 2.4


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])
    rng = np.random.default_rng(int(OPT["semilla"]))
    scene, groups = Atoms(), []
    placed = np.empty((0, 3))

    def add(symbols, positions, label):
        nonlocal scene, placed
        scene += Atoms(symbols, positions)
        groups.extend([label] * len(positions))
        placed = np.vstack((placed, positions))

    def clear(positions, gap) -> bool:
        return not len(placed) or float(
            cKDTree(placed).query(positions)[0].min()) >= gap

    # --- the plate, face up and centred.
    mixed = read(str(HERE / "xyz" / "placa_H_mixta.xyz"))
    plate = mixed[np.asarray(mixed.get_chemical_symbols()) != "C"]
    p_pos = plate.positions - plate.positions.mean(axis=0)
    add(plate.get_chemical_symbols(), p_pos, 0)
    top = float(p_pos[:, 2].max())
    radius = float(np.linalg.norm(p_pos[:, :2], axis=1).max())
    print(f"placa desnuda: {plate.get_chemical_formula()}, "
          f"{2 * radius:.0f} Å de ancho")

    # --- two bare rods behind it, a V mirrored about x = 0.
    rod = read(str(HERE / "xyz" / "barra_T_nucleo_desnudo.xyz"))
    r_pos = rod.positions - rod.positions.mean(axis=0)
    long = np.linalg.svd(r_pos, full_matrices=False)[2][0]
    axis = np.array([0.42, 0.0, 1.0])
    leaning = r_pos @ rotation_to(long, axis / np.linalg.norm(axis), 0.0).T
    for sign in (1.0, -1.0):
        # The left rod is the right one's mirror image, not a turned copy.
        local = leaning * [sign, 1.0, 1.0]
        add(rod.get_chemical_symbols(),
            local + [sign * (radius + 4.0), radius + 14.0,
                     top + 0.45 * np.ptp(local[:, 2])], 2)
    print(f"barras desnudas: 2 × {rod.get_chemical_formula()}")

    # --- adatoms resting on the surface.
    surface = p_pos[p_pos[:, 2] > top - 0.8]
    made = {4: 0, 5: 0, 6: 0}
    for _ in range(int(OPT["adatomos"])):
        symbol = str(rng.choice(["Fe", "Se", "C", "N"], p=[0.3, 0.3, 0.25,
                                                           0.15]))
        for _try in range(100):
            seat = surface[int(rng.integers(len(surface)))]
            point = (seat + [0.0, 0.0, ADATOM_HEIGHT]
                     + rng.normal(scale=0.4, size=3) * [1, 1, 0])[None]
            if clear(point, 2.2) and float(np.linalg.norm(
                    point[0, :2])) < radius - 3.0:
                add([symbol], point, 6)
                made[6] += 1
                break

    # --- precursors and fragments above, at true scale.
    floor_z, top_z = top + 7.0, top + 70.0
    for _ in range(int(OPT["moleculas"])):
        builder = PRECURSORS[int(rng.integers(len(PRECURSORS)))]
        mol, fragments = builder()
        local = mol.positions - mol.positions.mean(axis=0)
        for _try in range(200):
            z = rng.uniform(floor_z, top_z)
            depth = (top_z - z) / (top_z - floor_z)
            angle, reach = rng.uniform(0, 2 * np.pi), radius * np.sqrt(
                rng.uniform(0.0, 1.3))
            where = np.array([reach * np.cos(angle), reach * np.sin(angle),
                              z])
            positions = local.copy()
            label = 4
            if depth > 0.45:
                label = 5
                spread = 1.5 + 6.0 * (depth - 0.45)
                for piece in (np.asarray(f) for f in fragments):
                    out = local[piece].mean(axis=0)
                    norm = np.linalg.norm(out)
                    out = (out / norm if norm > 1e-6
                           else rng.normal(size=3))
                    out /= np.linalg.norm(out)
                    positions[piece] = local[piece] + out * spread
            positions = positions @ construir._random_rotation(rng).T + where
            if clear(positions, OPT["holgura"]):
                add(mol.get_chemical_symbols(), positions, label)
                made[label] += 1
                break
    print(f"precursores enteros: {made[4]}, descompuestos: {made[5]}, "
          f"adátomos: {made[6]}")

    scene.set_array("grupo", np.asarray(groups, dtype=int))
    scene.positions -= 0.5 * (scene.positions.max(axis=0)
                              + scene.positions.min(axis=0))
    closest = float(cKDTree(scene.positions).query(scene.positions,
                                                   k=2)[0][:, 1].min())
    name = "escena_superficie"
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"superficie de FeSe con precursores: "
                  f"{scene.get_chemical_formula()}; piezas COLOCADAS")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene, format="extxyz")
    picture(scene, HERE / "img" / f"{name}.png", 30.0, 0.0)
    print(f"\n{name}: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, caja "
          f"{np.ptp(scene.positions, axis=0).round(0)} Å")
    print("  (.extxyz lleva `grupo`: 0 placa, 2 barras, 4 precursores, "
          "5 fragmentos, 6 adátomos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
