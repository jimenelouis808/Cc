"""FeSe plates and rods in CRYSTALLINE carbon: the 900 C field, redone.

    uv run python modelos_SEM/escena_fese.py
    uv run python modelos_SEM/escena_fese.py --placas 8 --barras 16

`construir.py` coats the hexagonal plate and the tetragonal rod with a
meshed graphitic shell. That shell is sound and amorphous-walled, and
in a picture it reads as a lumpy sack. This scene applies what the
anemone taught:

* **The carbon is built on its lattice, never meshed.** A rod's carbon
  is a capped nanotube (`build_capped_cnt`: hexagons along the barrel,
  six pentagons a cap). A plate's carbon is flat graphene -- a plane is
  developable, so the flakes are exact by construction.
* **The crystal is cut by its container.** A rod is tetragonal FeSe
  carved to the tube's own cavity: everything inside the wall and no
  closer to it than the contact. A sphere or a capsule dropped in
  leaves gaps a picture shows at once.
* **Sizes vary.** Three plates and three rods, drawn from at random.
* **Every placement is checked** against a measured clearance, and a
  piece that does not clear is re-drawn or left out. The count that
  made it is printed.

What this is NOT: a plate's rim is bare. A graphene flake covers each
basal face at the van der Waals distance, and the six side faces are
open, because closing them on a lattice needs a prismatic cage this
package does not build. This is a SCENE: each piece is rigid, the
pieces are placed and nothing is relaxed together.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import construir
import numpy as np
from anemona_tubos import rotation_to
from ase import Atoms
from ase.io import write
from scipy.spatial import cKDTree

from nanocarbon_lab.builders.capped_cnt import build_capped_cnt

HERE = Path(__file__).resolve().parent
CIF = HERE / "cif"
BOND = 1.42

OPT: dict[str, float] = {
    "placas": 7,
    "barras": 14,
    "contacto": 3.0,     # A, FeSe atom to wall carbon inside a tube
    "hueco": 3.4,        # A, graphene to the plate's basal face
    "holgura": 3.3,      # A, between two pieces of the scene
    "campo": 120.0,      # A, half-width of the field the pieces fall in
    "alto": 34.0,        # A, half-height of that field
    "semilla": 0,
}

#: (circumradius, half thickness) of the three plates, in A. 3.0 is the
#: floor the crystal allows (see `construir.py`).
PLATES = ((26.0, 3.0), (34.0, 4.5), (44.0, 6.0))

#: (lattice frequency, body rings) of the three tubes: 77, 96 and
#: 138 A long, a body ring being 2.13 * freq A. Radius is
#: 1.96 * freq A: 7.9 and 9.8 A, the narrowest that hold a crystal.
RODS = ((4, 9), (5, 9), (5, 13))

LOOSE = 3.2
TRIES = 600


def graphene_hexagon(circumradius: float) -> np.ndarray:
    """A flat hexagonal graphene flake, zigzag-edged, in the xy plane.

    Honeycomb sites inside a regular hexagon, with atoms left holding
    fewer than two bonds removed. Bonds are 1.42 A exactly.
    """
    a = BOND * np.sqrt(3.0)
    a1 = np.array([a, 0.0])
    a2 = np.array([0.5 * a, 0.5 * np.sqrt(3.0) * a])
    reach = int(circumradius / a) + 3
    cells = np.array([(i, j) for i in range(-reach, reach + 1)
                      for j in range(-reach, reach + 1)], dtype=float)
    base = cells[:, :1] * a1 + cells[:, 1:] * a2
    sites = np.vstack((base, base + np.array([0.0, BOND])))
    sites -= np.array([0.5 * a, 0.5 * BOND])       # a hexagon centre
    # Inside the hexagon: under every one of its six edge lines.
    apothem = circumradius * np.sqrt(3.0) / 2.0
    angles = np.radians(np.arange(6) * 60.0 + 30.0)
    inside = (sites @ np.column_stack((np.cos(angles), np.sin(angles))).T
              <= apothem + 1e-6).all(axis=1)
    sites = sites[inside]
    for _ in range(3):
        count = cKDTree(sites).query_ball_point(sites, BOND * 1.1,
                                                return_length=True) - 1
        sites = sites[count >= 2]
    return np.column_stack((sites, np.zeros(len(sites))))


def make_plate(circumradius: float, half: float) -> Atoms:
    """Hexagonal FeSe between two graphene flakes, c axis along z."""
    core, _, _, (verdict, _why) = construir.core_from(
        CIF / "FeSe_hexagonal.cif", "prism",
        {"radius": circumradius, "half_height": half, "sides": 6})
    core.positions -= core.positions.mean(axis=0)
    flake = graphene_hexagon(circumradius + 2.0)
    top = float(core.positions[:, 2].max()) + OPT["hueco"]
    bottom = float(core.positions[:, 2].min()) - OPT["hueco"]
    carbon = np.vstack((flake + [0.0, 0.0, top], flake + [0.0, 0.0, bottom]))
    piece = Atoms(f"C{len(carbon)}", positions=carbon) + Atoms(
        core.get_chemical_symbols(), core.positions)
    piece.info = {"nucleo": core.get_chemical_formula(), "veredicto": verdict}
    return piece


def make_rod(freq: int, rings: int) -> Atoms:
    """Tetragonal FeSe filling a straight capped nanotube, axis along z."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tube = build_capped_cnt(n_body_rings=rings, freq=freq,
                                shape="straight", vacuum=10.0, seed=0)
    wall = np.asarray(tube.positions, dtype=float)
    wall -= wall.mean(axis=0)
    axis = np.linalg.svd(wall, full_matrices=False)[2][0]
    wall = wall @ rotation_to(axis, np.array([0.0, 0.0, 1.0]), 0.0).T
    radius = float(np.median(np.hypot(wall[:, 0], wall[:, 1])))
    half = float(np.abs(wall[:, 2]).max())
    block, _, _, _ = construir.core_from(
        CIF / "FeSe_tetragonal.cif", "rod",
        {"radius": radius, "half_length": half})
    inner = block.positions - block.positions.mean(axis=0)
    # The crystal's long axis onto the tube's.
    long_axis = np.linalg.svd(inner, full_matrices=False)[2][0]
    inner = inner @ rotation_to(long_axis, np.array([0.0, 0.0, 1.0]), 0.0).T
    contact = OPT["contacto"]
    keep = ((cKDTree(wall).query(inner)[0] >= contact)
            & (np.hypot(inner[:, 0], inner[:, 1]) < radius - 1.0)
            & (np.abs(inner[:, 2]) < half - 1.0))
    core = block[keep]
    core.positions = inner[keep]
    for _ in range(3):
        near = cKDTree(core.positions).query(core.positions, k=2)[0][:, 1]
        core = core[near <= LOOSE]
    piece = Atoms(f"C{len(wall)}", positions=wall) + Atoms(
        core.get_chemical_symbols(), core.positions)
    piece.info = {"nucleo": core.get_chemical_formula(),
                  "radio": radius, "largo": 2.0 * half,
                  "censo": tube.info.get("ring_counts")}
    return piece


def picture(atoms: Atoms, path: Path, elevation: float,
            azimuth: float) -> None:
    """A depth-shaded orthographic view: C grey, Fe orange, Se green."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = atoms.positions - atoms.positions.mean(axis=0)
    a, b = np.radians(elevation), np.radians(azimuth)
    turn = (np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a), np.cos(a)]])
            @ np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0],
                        [-np.sin(b), 0, np.cos(b)]]))
    view = positions @ turn.T
    order = np.argsort(view[:, 2])
    depth = (view[:, 2] - view[:, 2].min()) / np.ptp(view[:, 2])
    symbols = np.array(atoms.get_chemical_symbols())
    fade = (0.55 + 0.45 * depth)[:, None]
    colour = np.repeat((0.68 - 0.5 * depth)[:, None], 3, axis=1)
    for symbol, rgb in (("Fe", (0.93, 0.49, 0.13)), ("Se", (0.55, 0.72, 0.16))):
        mask = symbols == symbol
        colour[mask] = 1.0 - (1.0 - np.asarray(rgb)) * fade[mask]
    size = np.where(symbols == "C", 1.3, 5.0)
    fig, axis = plt.subplots(figsize=(12, 10), dpi=170)
    axis.scatter(view[order, 0], view[order, 1], s=size[order],
                 c=colour[order], linewidths=0)
    axis.set_aspect("equal")
    axis.axis("off")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])
    rng = np.random.default_rng(int(OPT["semilla"]))
    started = time.time()

    plates = [make_plate(*shape) for shape in PLATES]
    for shape, piece in zip(PLATES, plates, strict=True):
        print(f"placa R={shape[0]:.0f} Å, grosor {2 * shape[1]:.0f} Å: "
              f"{piece.info['nucleo']} ({piece.info['veredicto']}) entre "
              f"dos grafenos, {len(piece)} átomos")
    rods = [make_rod(*shape) for shape in RODS]
    for piece in rods:
        info = piece.info
        print(f"barra: tubo de radio {info['radio']:.1f} Å y "
              f"{info['largo']:.0f} Å, anillos {info['censo']}, núcleo "
              f"{info['nucleo']}, {len(piece)} átomos")

    scene = Atoms()
    groups: list[int] = []
    everything = np.empty((0, 3))
    field, height, gap = OPT["campo"], OPT["alto"], OPT["holgura"]
    up = np.array([0.0, 0.0, 1.0])

    def drop(piece: Atoms, heading_of, label: int) -> bool:
        nonlocal scene, everything
        tree = cKDTree(everything) if len(everything) else None
        for _ in range(TRIES):
            turn = rotation_to(up, heading_of(), rng.uniform(0, 2 * np.pi))
            where = np.array([rng.uniform(-field, field),
                              rng.uniform(-field, field),
                              rng.uniform(-height, height)])
            trial = piece.positions @ turn.T + where
            if tree is None or float(tree.query(trial)[0].min()) >= gap:
                scene += Atoms(piece.get_chemical_symbols(), trial)
                groups.extend([label] * len(piece))
                everything = np.vstack((everything, trial))
                return True
        return False

    def plate_heading() -> np.ndarray:
        # Lying down, more or less: the normal within ~35 deg of z.
        tilt = rng.normal(scale=0.35, size=3) * [1.0, 1.0, 0.0] + up
        return tilt / np.linalg.norm(tilt)

    def rod_heading() -> np.ndarray:
        # Any way round, but mostly lying in the field.
        heading = rng.normal(size=3) * [1.0, 1.0, 0.45]
        return heading / np.linalg.norm(heading)

    placed_plates = sum(
        drop(plates[int(rng.integers(len(plates)))], plate_heading, 0)
        for _ in range(int(OPT["placas"])))
    placed_rods = sum(
        drop(rods[int(rng.integers(len(rods)))], rod_heading, 1)
        for _ in range(int(OPT["barras"])))
    print(f"colocadas: {placed_plates} de {int(OPT['placas'])} placas, "
          f"{placed_rods} de {int(OPT['barras'])} barras")

    scene.set_array("grupo", np.asarray(groups, dtype=int))
    scene.positions -= scene.positions.mean(axis=0)
    closest = float(cKDTree(scene.positions).query(scene.positions,
                                                   k=2)[0][:, 1].min())
    (HERE / "xyz").mkdir(exist_ok=True)
    (HERE / "img").mkdir(exist_ok=True)
    write(str(HERE / "xyz" / "escena_fese.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"placas y barras de FeSe en carbono cristalino: "
                  f"{scene.get_chemical_formula()}; piezas COLOCADAS")
    write(str(HERE / "xyz" / "escena_fese.extxyz"), scene, format="extxyz")
    picture(scene, HERE / "img" / "escena_fese.png", 58.0, 20.0)
    picture(scene, HERE / "img" / "escena_fese_b.png", 20.0, -40.0)
    print(f"\nescena: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, {time.time() - started:.0f} s")
    print("  xyz/escena_fese.xyz  (.extxyz lleva `grupo`: 0 placas, 1 barras)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
