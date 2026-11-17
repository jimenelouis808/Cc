"""The synthesis as one picture: precursors fall, break up, and land as
the products each temperature makes.

    uv run python modelos_SEM/escena_sintesis.py
    uv run python modelos_SEM/escena_sintesis.py --moleculas 90

Atomistic, top to bottom, for a camera on -y (`portada_blend.py --style
revista --calor --bola 4 5 6 --ampliar 5 --encuadrar 0 1 2 3`). The
molecules, fragments and loose atoms are MAGNIFIED (`ampliar`, 5 by
default) about their own centres: at true scale a 5 A molecule beside a
70 A plate is a speck. `--ampliar` in `portada_blend.py` must be given
the same factor, so their bonds are found at true length:

* **precursors**, whole: acetonitrile CH3CN, ferrocene Fe(C5H5)2 and
  selenourea SeC(NH2)2, built atom by atom from their bond lengths and
  falling at random orientations;
* **decomposition**: lower down the same molecules come apart into
  their fragments -- CH3 + CN, Fe + two C5H5 rings, Se + C(NH2)2 --
  drifting further apart the lower they are;
* **atoms** of Fe and Se raining onto the products;
* **products**, in mirror symmetry about the centre line: on the left
  (**900 C**) a coated FeSe plate facing the camera, with a window cut
  in its doped, functionalised shell so the FeSe surface shows, and two
  coated rods crossed in an X behind it; on the right (**1020 C**) a
  carbon nanotube doped with N and Se and carrying oxygen groups, with
  two more crossed behind it. The molecules above are mirrored too.

No captions: the two temperatures are carried by the light --
`portada_blend.py --calor` washes the left half amber and the right
half white-hot blue.

The fragments are a drawing of the chemistry, not a simulation of it:
which bonds break first, and in what order, is not computed. The
products are this package's own checked models. Every piece is
checked against everything already placed and re-drawn if it comes
closer than `holgura`.
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

    # --- the products, in mirror symmetry about x = 0: on the left
    # (900 C) the plate facing the camera with two rods crossed in an X
    # behind it; on the right (1020 C) a tube standing in front with
    # two more crossed behind it.
    core, shell, centre = coated_plate(rng, bool(OPT["ventana"]))
    normal = np.array([0.14, -0.94, 0.30])
    turn = rotation_to(up, normal / np.linalg.norm(normal), 0.52)
    left, right = np.array([-88.0, 0.0, -60.0]), np.array([88.0, 0.0, -60.0])
    add(core.get_chemical_symbols(), (core.positions - centre) @ turn.T
        + left, 0)
    add(shell.get_chemical_symbols(), (shell.positions - centre) @ turn.T
        + left, 1)
    print(f"placa: núcleo {core.get_chemical_formula()}, coraza "
          f"{shell.get_chemical_formula()}"
          + (" con ventana" if OPT["ventana"] else ""))

    def stand(piece: Atoms, axis, at, spin: float, label: int) -> None:
        local = piece.positions - piece.positions.mean(axis=0)
        long = np.linalg.svd(local, full_matrices=False)[2][0]
        a = np.asarray(axis, dtype=float)
        local = local @ rotation_to(long, a / np.linalg.norm(a), spin).T
        add(piece.get_chemical_symbols(), local + at, label)

    rod = read(str(HERE / "xyz" / "barra_T_NSeO.xyz"))
    for sign in (1.0, -1.0):
        stand(rod, (0.55 * sign, 0.0, 1.0), left + [0.0, 58.0 + 20 * sign, 0.0],
              0.0, 2)
    print(f"barras: 2 × {rod.get_chemical_formula()}")

    tube = read(str(HERE / "xyz" / "nanotubo_NSeO.xyz"))
    for index, (name, coverage) in enumerate(GRAFTS):
        try:
            tube = functionalize(tube, construir.group_for(name),
                                 coverage=coverage, face="outer",
                                 seed=int(OPT["semilla"]) + index)
        except ValueError as exc:
            print(f"  {name}: no se injertó ({exc})")
    print(f"tubo: {tube.get_chemical_formula()}")
    stand(tube, (0.0, 0.0, 1.0), right + [0.0, -6.0, 0.0], 0.0, 3)
    for sign in (1.0, -1.0):
        stand(tube, (0.55 * sign, 0.0, 1.0),
              right + [0.0, 70.0 + 25 * sign, 0.0], 1.0, 3)
    products_top = float(placed[:, 2].max())
    tree = cKDTree(placed)
    print(f"productos: par más cercano entre átomos a "
          f"{float(tree.query(placed, k=2)[0][:, 1].min()):.2f} Å")

    # --- precursors, fragments and atoms, above the products.
    count = int(OPT["moleculas"])
    zoom = float(OPT["ampliar"])
    gap = OPT["holgura"] * zoom
    floor_z, top_z = products_top + 30.0, products_top + 260.0
    made = {4: 0, 5: 0, 6: 0}
    for _ in range(count // 2):
        builder = PRECURSORS[int(rng.integers(len(PRECURSORS)))]
        mol, fragments = builder()
        for _try in range(200):
            z = rng.uniform(floor_z, top_z)
            depth = (top_z - z) / (top_z - floor_z)      # 0 top, 1 bottom
            # The left half only; its mirror image fills the right.
            where = np.array([rng.uniform(-200.0, -14.0),
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
            mirror = positions * [-1.0, 1.0, 1.0]
            if clear(positions, gap) and clear(mirror, gap) and float(
                    cKDTree(positions).query(mirror)[0].min()) >= gap:
                add(mol.get_chemical_symbols(), positions, label)
                add(mol.get_chemical_symbols(), mirror, label)
                made[label] += 2
                break
    # Single atoms nearest the products, mirrored like the molecules.
    for _ in range(int(count * 0.35)):
        symbol = "Fe" if rng.random() < 0.55 else "Se"
        for _try in range(100):
            point = np.array([[rng.uniform(-200.0, -14.0),
                               rng.uniform(-50.0, 60.0),
                               rng.uniform(products_top - 10.0,
                                           floor_z + 25.0)]])
            twin = point * [-1.0, 1.0, 1.0]
            if clear(point, gap) and clear(twin, gap):
                add([symbol], point, 6)
                add([symbol], twin, 6)
                made[6] += 2
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
    # No captions: the temperatures are told by the light
    # (`portada_blend.py --calor`), warm on the left and white-hot on
    # the right. A caption file from an older run would still be drawn,
    # so it is removed.
    (HERE / "xyz" / f"{name}.rotulos.json").unlink(missing_ok=True)
    picture(scene, HERE / "img" / f"{name}.png", 0.0, 0.0)
    print(f"\n{name}: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, caja "
          f"{np.ptp(scene.positions, axis=0).round(0)} Å")
    print("  (.extxyz lleva `grupo`: 0 núcleo de la placa, 1 su coraza, "
          "2 barras, 3 tubos, 4 precursores, 5 fragmentos, 6 átomos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
