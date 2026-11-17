"""A vortex: FeSe plates and rods spiralling up out of an anemone.

    uv run python modelos_SEM/escena_vortice.py
    uv run python modelos_SEM/escena_vortice.py --piezas 30 --vueltas 2.5

The anemone (`xyz/anemona_conjunta.extxyz`) sits low, over the mirror
floor; out of it rises a widening helix of pieces, as if the particle
were throwing them off -- tight and close above it, opening as it
climbs. Made for a tall portrait frame (`portada_blend.py --style
revista --elev 8`).

* **rods** lie along the helix's own tangent, so they read as the
  streamlines of the swirl;
* **plates** (`xyz/placa_H_mixta.xyz`) face outward and tip with the
  climb, so they show their faces as they pass the sides.

Round its base, `campo` small anemones (`anemona_mini`, `anemona_mini_b`)
rest on the ground, over the back and the sides so none hides the
source.

Every piece is checked against everything already placed and pushed
outward along its radius until it clears `holgura`; any push is
printed.
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
    "piezas": 40,
    "vueltas": 2.2,      # turns of the helix
    "alto": 560.0,       # A the helix climbs
    "radio_base": 130.0, # A, helix radius just above the anemone
    "radio_cima": 300.0, # A, helix radius at the top
    "holgura": 6.0,
    "campo": 7,          # small anemones on the ground round the base
    "semilla": 0,
}

#: The small anemones of the field, taken in turn.
MINIS = ("anemona_mini", "anemona_mini_b")


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])
    rng = np.random.default_rng(int(OPT["semilla"]))
    up = np.array([0.0, 0.0, 1.0])

    anemone = read(str(HERE / "xyz" / "anemona_conjunta.extxyz"))
    a_pos = anemone.positions - anemone.positions.mean(axis=0)
    scene = Atoms(anemone.get_chemical_symbols(), a_pos)
    groups = [0] * len(anemone)
    placed = a_pos.copy()
    top = float(a_pos[:, 2].max())

    rods = [make_rod(*shape) for shape in RODS]
    plate = read(str(HERE / "xyz" / "placa_H_mixta.xyz"))
    p_pos = plate.positions - plate.positions.mean(axis=0)

    count = int(OPT["piezas"])
    turns, rise = OPT["vueltas"], OPT["alto"]
    r0, r1 = OPT["radio_base"], OPT["radio_cima"]
    for k in range(count):
        t = (k + 0.5) / count
        angle = 2.0 * np.pi * turns * t
        radius = r0 + (r1 - r0) * t ** 0.8
        height = 0.55 * top + rise * t
        radial = np.array([np.cos(angle), np.sin(angle), 0.0])
        # d/dt of the helix: round the axis, and up it.
        tangent = (2.0 * np.pi * turns * radius
                   * np.array([-np.sin(angle), np.cos(angle), 0.0])
                   + np.array([0.0, 0.0, rise]))
        tangent /= np.linalg.norm(tangent)
        if k % 3 == 1:
            normal = radial + 0.5 * tangent
            local = p_pos @ rotation_to(up, normal / np.linalg.norm(normal),
                                        rng.uniform(0, 6.3)).T
            symbols, label = plate.get_chemical_symbols(), 1
        else:
            # Shorter rods low in the swirl, longer ones as it opens.
            piece = rods[min(int(t * len(rods)), len(rods) - 1)]
            local = piece.positions @ rotation_to(up, tangent,
                                                  rng.uniform(0, 6.3)).T
            symbols, label = piece.get_chemical_symbols(), 2
        tree = cKDTree(placed)
        push = 0.0
        while float(tree.query(local + radial * (radius + push)
                               + [0.0, 0.0, height])[0].min()) \
                < OPT["holgura"]:
            push += 3.0
        if push:
            print(f"  pieza {k}: empujada {push:.0f} Å hacia fuera")
        trial = local + radial * (radius + push) + [0.0, 0.0, height]
        scene += Atoms(symbols, trial)
        groups.extend([label] * len(trial))
        placed = np.vstack((placed, trial))

    # --- the field: small anemones resting on the ground round the
    # base, more of them behind than in front so none hides the source.
    ground = float(a_pos[:, 2].min())
    minis = [read(str(HERE / "xyz" / f"{name}.extxyz")) for name in MINIS
             if (HERE / "xyz" / f"{name}.extxyz").exists()]
    made = 0
    for k in range(int(OPT["campo"])):
        mini = minis[k % len(minis)]
        m_pos = mini.positions - mini.positions.mean(axis=0)
        # Spread over the back and the sides: -150 to +150 degrees from
        # straight behind (+y), alternating, at 380 to 640 A.
        side = -1.0 if k % 2 else 1.0
        angle = np.radians(90.0 + side * (25.0 + 125.0 * (k / max(
            OPT["campo"] - 1, 1))))
        for _ in range(200):
            spin = rotation_to(up, rng.normal(size=3), rng.uniform(0, 6.3))
            local = m_pos @ spin.T
            dist = rng.uniform(380.0, 640.0)
            where = np.array([dist * np.cos(angle), dist * np.sin(angle),
                              ground - local[:, 2].min()])
            trial = local + where
            if float(cKDTree(placed).query(trial)[0].min()) \
                    >= OPT["holgura"]:
                scene += Atoms(mini.get_chemical_symbols(), trial)
                groups.extend([3] * len(trial))
                placed = np.vstack((placed, trial))
                made += 1
                break
    print(f"campo: {made} anémonas pequeñas")

    scene.set_array("grupo", np.asarray(groups, dtype=int))
    scene.positions -= 0.5 * (scene.positions.max(axis=0)
                              + scene.positions.min(axis=0))
    closest = float(cKDTree(scene.positions).query(scene.positions,
                                                   k=2)[0][:, 1].min())
    name = "escena_vortice"
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"vórtice de placas y barras de FeSe sobre una anémona: "
                  f"{scene.get_chemical_formula()}; piezas COLOCADAS")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene, format="extxyz")
    picture(scene, HERE / "img" / f"{name}.png", 8.0, 0.0)
    print(f"\n{name}: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, caja "
          f"{np.ptp(scene.positions, axis=0).round(0)} Å")
    print("  (.extxyz lleva `grupo`: 0 anémona, 1 placas, 2 barras, "
          "3 campo)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
