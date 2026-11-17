"""The synthesis as one picture: precursors fall, break up, and land as
the products each temperature makes.

    uv run python modelos_SEM/escena_sintesis.py
    uv run python modelos_SEM/escena_sintesis.py --moleculas 90

Atomistic, top to bottom, for a camera on -y (`portada_blend.py --style
revista --elev 6 --bola 4 5 6 --ampliar 5`). The molecules, fragments
and loose atoms are MAGNIFIED (`ampliar`, 5 by default) about their own
centres, and a caption says so: at true scale a 5 A molecule beside a
70 A plate is a speck. `--ampliar` in `portada_blend.py` must be given
the same factor, so their bonds are found at true length:

* **precursors**, whole: acetonitrile CH3CN, ferrocene Fe(C5H5)2 and
  selenourea SeC(NH2)2, built atom by atom from their bond lengths and
  falling at random orientations;
* **decomposition**: lower down the same molecules come apart into
  their fragments -- CH3 + CN, Fe + two C5H5 rings, Se + C(NH2)2 --
  drifting further apart the lower they are;
* **atoms** of Fe and Se raining onto the products;
* **products**, left to right: at **900 C** a coated FeSe plate with a
  window cut in its doped, functionalised shell so the FeSe surface
  shows, and two coated FeSe rods; at **1020 C** carbon nanotubes doped
  with N and Se and carrying oxygen groups.

The fragments are a drawing of the chemistry, not a simulation of it:
which bonds break first, and in what order, is not computed. The
products are this package's own checked models. Captions go into
`xyz/escena_sintesis.rotulos.json`, which `portada_blend.py` turns into
3D text. Every piece is checked against everything already placed and
re-drawn if it comes closer than `holgura`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import construir
import numpy as np
from anemona_tubos import rotation_to
from ase import Atoms
from ase.io import read, write
from escena_fese import picture
from scipy.spatial import cKDTree

from nanocarbon_lab.functionalize.attach import functionalize

HERE = Path(__file__).resolve().parent

OPT: dict[str, float] = {
    "moleculas": 34,
    # Precursors, fragments and loose atoms are drawn this many times
    # their size, each about its own centre: at true scale a 5 A
    # molecule beside a 70 A plate is a speck. The caption says so.
    "ampliar": 5.0,
    "holgura": 2.6,      # A, true-scale contact, multiplied by `ampliar`
    "ventana": 1.0,      # 1 cuts a window in the plate's shell, 0 not
    "semilla": 4,
}

#: Plate core in `placa_H_NSeO.xyz`: written first, Fe813 Se813.
PLATE_CORE = 1626

#: Extra oxygen groups grafted on the tubes, so they show at this scale.
GRAFTS = (("hydroxyl", 0.05), ("carboxyl", 0.04), ("carbonyl", 0.03))


def acetonitrile() -> tuple[Atoms, list[list[int]]]:
    """CH3-C#N, axis along z, and its two fragments (CH3, CN)."""
    h = []
    for k in range(3):
        a = 2.0 * np.pi * k / 3.0
        h.append([1.03 * np.cos(a), 1.03 * np.sin(a), -0.36])
    positions = [[0, 0, 0], [0, 0, 1.46], [0, 0, 2.62], *h]
    mol = Atoms("CCNHHH", positions=positions)
    return mol, [[0, 3, 4, 5], [1, 2]]


def ferrocene() -> tuple[Atoms, list[list[int]]]:
    """Fe(C5H5)2, eclipsed, and its three fragments (Fe, ring, ring)."""
    symbols, positions = ["Fe"], [[0.0, 0.0, 0.0]]
    rings = []
    for z in (1.66, -1.66):
        ring = []
        for k in range(5):
            a = 2.0 * np.pi * k / 5.0
            ring.append(len(symbols))
            symbols.append("C")
            positions.append([1.216 * np.cos(a), 1.216 * np.sin(a), z])
        for k in range(5):
            a = 2.0 * np.pi * k / 5.0
            ring.append(len(symbols))
            symbols.append("H")
            positions.append([2.296 * np.cos(a), 2.296 * np.sin(a), z])
        rings.append(ring)
    return Atoms(symbols, positions=positions), [[0], *rings]


def selenourea() -> tuple[Atoms, list[list[int]]]:
    """SeC(NH2)2, planar, and its fragments (Se, C(NH2)2)."""
    symbols = ["C", "Se", "N", "N"]
    positions = [[0, 0, 0], [0, 1.86, 0], [1.16, -0.67, 0], [-1.16, -0.67, 0]]
    for n in (2, 3):
        base = np.asarray(positions[n], dtype=float)
        out = base / np.linalg.norm(base)
        for turn in (np.radians(60.0), -np.radians(60.0)):
            c, s = np.cos(turn), np.sin(turn)
            d = np.array([c * out[0] - s * out[1], s * out[0] + c * out[1], 0])
            symbols.append("H")
            positions.append(list(base + 1.01 * d))
    return Atoms(symbols, positions=positions), [[1], [0, 2, 3, 4, 5, 6, 7]]


PRECURSORS = (acetonitrile, ferrocene, selenourea)


def coated_plate(rng: np.random.Generator, window: bool) -> tuple:
    """The doped, functionalised plate around the evenly cut core.

    The shell is `placa_H_NSeO`'s (its first 1626 atoms are the old
    basal core, dropped); the core is `placa_H_mixta`'s. With `window`,
    the shell is opened over one half of the front face so the FeSe
    surface shows, and any group left without a shell atom to hang on
    is removed with it.
    """
    dressed = read(str(HERE / "xyz" / "placa_H_NSeO.xyz"))
    shell = dressed[PLATE_CORE:]
    mixed = read(str(HERE / "xyz" / "placa_H_mixta.xyz"))
    core = mixed[np.asarray(mixed.get_chemical_symbols()) != "C"]
    gaps = cKDTree(shell.positions).query(core.positions)[0]
    core = core[gaps >= 2.2]
    centre = core.positions.mean(axis=0)
    if window:
        rel = shell.positions - centre
        cut = (rel[:, 2] > 0.0) & (rel[:, 0] > -8.0)
        shell = shell[~cut]
        symbols = np.asarray(shell.get_chemical_symbols())
        for _ in range(4):
            tree = cKDTree(shell.positions)
            near = tree.query(shell.positions, k=2)[0][:, 1]
            keep = (symbols == "C") | (near < 1.9)
            shell, symbols = shell[keep], symbols[keep]
    return core, shell, centre


def place(local: np.ndarray, where: np.ndarray, turn: np.ndarray):
    """Positions of a piece turned by `turn` and set at `where`."""
    return (local - local.mean(axis=0)) @ turn.T + where


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])
    rng = np.random.default_rng(int(OPT["semilla"]))
    up = np.array([0.0, 0.0, 1.0])
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

    # --- 900 C, left: the plate, facing the camera and tipped up.
    core, shell, centre = coated_plate(rng, bool(OPT["ventana"]))
    normal = np.array([-0.15, -0.80, 0.58])
    turn = rotation_to(up, normal / np.linalg.norm(normal), 0.5)
    where = np.array([-95.0, 0.0, -70.0])
    add(core.get_chemical_symbols(), (core.positions - centre) @ turn.T
        + where, 0)
    add(shell.get_chemical_symbols(), (shell.positions - centre) @ turn.T
        + where, 1)
    print(f"placa: núcleo {core.get_chemical_formula()}, coraza "
          f"{shell.get_chemical_formula()}"
          + (" con ventana" if OPT["ventana"] else ""))

    rod = read(str(HERE / "xyz" / "barra_T_NSeO.xyz"))
    for axis, at in (((0.85, 0.35, 0.25), (-120.0, 75.0, -150.0)),
                     ((0.25, 0.45, 1.0), (-195.0, 40.0, -55.0))):
        a = np.asarray(axis) / np.linalg.norm(axis)
        r_pos = rod.positions - rod.positions.mean(axis=0)
        long = np.linalg.svd(r_pos, full_matrices=False)[2][0]
        r_pos = r_pos @ rotation_to(long, a, rng.uniform(0, 6.3)).T
        add(rod.get_chemical_symbols(), r_pos + at, 2)
    print(f"barras: 2 × {rod.get_chemical_formula()}")

    # --- 1020 C, right: doped tubes, with more oxygen groups grafted.
    tube = read(str(HERE / "xyz" / "nanotubo_NSeO.xyz"))
    for index, (name, coverage) in enumerate(GRAFTS):
        try:
            tube = functionalize(tube, construir.group_for(name),
                                 coverage=coverage, face="outer",
                                 seed=int(OPT["semilla"]) + index)
        except ValueError as exc:
            print(f"  {name}: no se injertó ({exc})")
    print(f"tubo: {tube.get_chemical_formula()}")
    t_pos = tube.positions - tube.positions.mean(axis=0)
    long = np.linalg.svd(t_pos, full_matrices=False)[2][0]
    for axis, at, spin in (((-0.30, -0.20, 1.0), (75.0, -10.0, -75.0), 0.0),
                           ((0.35, 0.30, 1.0), (175.0, 70.0, -85.0), 2.0)):
        a = np.asarray(axis) / np.linalg.norm(axis)
        local = t_pos @ rotation_to(long, a, spin).T
        add(tube.get_chemical_symbols(), local + at, 3)
    products_top = float(placed[:, 2].max())

    # --- precursors, fragments and atoms, above the products.
    count = int(OPT["moleculas"])
    zoom = float(OPT["ampliar"])
    gap = OPT["holgura"] * zoom
    floor_z, top_z = products_top + 30.0, products_top + 330.0
    made = {4: 0, 5: 0, 6: 0}
    for _ in range(count):
        builder = PRECURSORS[int(rng.integers(len(PRECURSORS)))]
        mol, fragments = builder()
        for _try in range(200):
            z = rng.uniform(floor_z, top_z)
            depth = (top_z - z) / (top_z - floor_z)      # 0 top, 1 bottom
            where = np.array([rng.uniform(-210.0, 210.0),
                              rng.uniform(-50.0, 60.0), z])
            turn = construir._random_rotation(rng)
            local = (mol.positions - mol.positions.mean(axis=0)) * zoom
            if depth < 0.40:
                label, pieces = 4, [np.arange(len(mol))]
                spread = 0.0
            else:
                label, pieces = 5, [np.asarray(f) for f in fragments]
                spread = zoom * (2.0 + 5.0 * (depth - 0.40))
            positions = local.copy()
            for piece in pieces:
                if spread:
                    out = local[piece].mean(axis=0)
                    norm = np.linalg.norm(out)
                    out = out / norm if norm > 1e-6 else rng.normal(size=3)
                    out = out / np.linalg.norm(out)
                    jitter = construir._random_rotation(rng) if depth > 0.7 \
                        else np.eye(3)
                    centre_piece = local[piece].mean(axis=0)
                    positions[piece] = ((local[piece] - centre_piece)
                                        @ jitter.T + centre_piece
                                        + out * spread)
            positions = positions @ turn.T + where
            if clear(positions, gap):
                add(mol.get_chemical_symbols(), positions, label)
                made[label] += 1
                break
    # Single atoms nearest the products: Fe over both, Se over the FeSe.
    for _ in range(int(count * 0.7)):
        symbol = "Fe" if rng.random() < 0.55 else "Se"
        x = rng.uniform(-210.0, 0.0 if symbol == "Se" else 210.0)
        for _try in range(100):
            point = np.array([[x, rng.uniform(-50.0, 60.0),
                               rng.uniform(products_top - 10.0,
                                           floor_z + 25.0)]])
            if clear(point, gap):
                add([symbol], point, 6)
                made[6] += 1
                break
    print(f"precursores enteros: {made[4]}, descompuestos: {made[5]}, "
          f"átomos sueltos: {made[6]}")

    scene.set_array("grupo", np.asarray(groups, dtype=int))
    shift = 0.5 * (scene.positions.max(axis=0) + scene.positions.min(axis=0))
    scene.positions -= shift
    closest = float(cKDTree(scene.positions).query(scene.positions,
                                                   k=2)[0][:, 1].min())
    name = "escena_sintesis"
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"síntesis: precursores, descomposición y productos a "
                  f"900 y 1020 C: {scene.get_chemical_formula()}")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene, format="extxyz")
    low = float(scene.positions[:, 2].min())
    high = float(scene.positions[:, 2].max())
    # The temperatures stand ON the floor, in front of the products: one
    # placed below the lowest atom sank into the mirror, which cut the
    # foot off every glyph -- "1020" read as "1030".
    labels = [
        {"texto": "900 °C", "pos": [-110.0 - shift[0], -150.0, low + 9.0],
         "tam": 20.0, "color": [1.0, 0.62, 0.30], "brillo": 4.0},
        {"texto": "1020 °C", "pos": [125.0 - shift[0], -150.0, low + 9.0],
         "tam": 20.0, "color": [0.45, 0.80, 1.00], "brillo": 4.0},
        {"texto": "CH3CN  ·  Fe(C5H5)2  ·  SeC(NH2)2",
         "pos": [0.0, -60.0, high + 22.0], "tam": 13.0,
         "color": [0.92, 0.94, 1.0], "brillo": 2.5},
        {"texto": f"moléculas y átomos ×{zoom:g}",
         "pos": [0.0, -60.0, high + 6.0], "tam": 7.0,
         "color": [0.60, 0.66, 0.80], "brillo": 1.6},
    ]
    (HERE / "xyz" / f"{name}.rotulos.json").write_text(
        json.dumps(labels, indent=1, ensure_ascii=False))
    picture(scene, HERE / "img" / f"{name}.png", 0.0, 0.0)
    print(f"\n{name}: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, caja "
          f"{np.ptp(scene.positions, axis=0).round(0)} Å")
    print("  (.extxyz lleva `grupo`: 0 núcleo de la placa, 1 su coraza, "
          "2 barras, 3 tubos, 4 precursores, 5 fragmentos, 6 átomos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
