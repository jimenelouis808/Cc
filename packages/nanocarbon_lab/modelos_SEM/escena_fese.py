"""FeSe plates and FeSe rods as two clean scenes.

    uv run python modelos_SEM/escena_fese.py
    uv run python modelos_SEM/escena_fese.py --placas 7 --barras 12

Two scenes, because a plate and a rod in one frame compete:

* **`escena_placas`** -- the hexagonal FeSe plate, wrapped all the way
  round in its graphitic shell, shown as a column of plates each turned
  a step further from face-on to edge-on.
* **`escena_barras_tubo`** -- tetragonal FeSe rods, each carved to the
  cavity of a capped nanotube built on its lattice, as a sparse heap.

Every piece is rigid and placed; nothing is relaxed together. The
closest pair of atoms in each scene is printed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import construir
import numpy as np
from anemona_tubos import rotation_to
from ase import Atoms
from ase.io import read, write
from scipy.spatial import cKDTree

from nanocarbon_lab.builders.capped_cnt import build_capped_cnt

HERE = Path(__file__).resolve().parent
CIF = HERE / "cif"
BOND = 1.42

OPT: dict[str, float] = {
    "placas": 5,
    "barras": 9,
    "contacto": 3.0,     # A, FeSe atom to wall carbon inside a tube
    "piso": 9.0,         # A, one plate's shell to the next one's
    "holgura": 9.0,      # A, between two rods
    "campo": 85.0,       # A, half-width of the field the rods fall in
    "semilla": 0,
}

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


def wrapped_plate() -> Atoms:
    """The hexagonal FeSe plate in its CLOSED graphitic shell.

    `construir.py` builds and checks it (`xyz/placa_H_C.xyz`): the shell
    is the meshed one, because a lattice cannot close round a prism's
    rim. Wrapped all the way round was the request, and this is the one
    piece of the package that does it.
    """
    source = HERE / "xyz" / "placa_H_C.xyz"
    if not source.exists():
        raise SystemExit("falta xyz/placa_H_C.xyz: corre antes "
                         "`construir.py --solo placa_H`.")
    plate = read(str(source))
    plate.positions -= plate.positions.mean(axis=0)
    return plate


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


def save(scene: Atoms, groups: list[int], name: str, note: str,
         views: tuple) -> None:
    """Write one scene's xyz, extxyz and two pictures; print its check."""
    scene.set_array("grupo", np.asarray(groups, dtype=int))
    scene.positions -= scene.positions.mean(axis=0)
    closest = float(cKDTree(scene.positions).query(scene.positions,
                                                   k=2)[0][:, 1].min())
    (HERE / "xyz").mkdir(exist_ok=True)
    (HERE / "img").mkdir(exist_ok=True)
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"{note}: {scene.get_chemical_formula()}; piezas COLOCADAS")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene, format="extxyz")
    for suffix, view in zip(("", "_b"), views, strict=True):
        picture(scene, HERE / "img" / f"{name}{suffix}.png", *view)
    print(f"{name}: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å")


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])
    rng = np.random.default_rng(int(OPT["semilla"]))
    up = np.array([0.0, 0.0, 1.0])

    # --- scene 1, the plates: ONE plate seen five ways. A column of
    # them, each turned a step further from face-on to edge-on, so the
    # hexagon and the thinness are both on the page and nothing
    # overlaps. A heap of plates shows neither.
    plate = wrapped_plate()
    count = int(OPT["placas"])
    scene, groups = Atoms(), []
    placed = np.empty((0, 3))
    level = 0.0
    for k in range(count):
        tilt = np.radians(-70.0 + 140.0 * k / max(count - 1, 1))
        normal = np.array([0.0, np.sin(tilt), np.cos(tilt)])
        turned = plate.positions @ rotation_to(up, normal,
                                               np.radians(11.0 * k)).T
        # Rise until this plate clears the one under it by the gap.
        while len(placed) and float(cKDTree(placed).query(
                turned + [0.0, 0.0, level])[0].min()) < OPT["piso"]:
            level += 2.0
        trial = turned + [0.0, 0.0, level]
        scene += Atoms(plate.get_chemical_symbols(), trial)
        groups.extend([k] * len(plate))
        placed = trial
    save(scene, groups, "escena_placas",
         f"{count} placas de FeSe hexagonal envueltas en carbono",
         ((8.0, 25.0), (30.0, -50.0)))

    # --- scene 2, the rods: a sparse heap, as the 900 C field is, but
    # few of them and none touching.
    rods = [make_rod(*shape) for shape in RODS]
    for piece in rods:
        info = piece.info
        print(f"barra: tubo de radio {info['radio']:.1f} Å y "
              f"{info['largo']:.0f} Å, anillos {info['censo']}, núcleo "
              f"{info['nucleo']}, {len(piece)} átomos")
    scene, groups = Atoms(), []
    placed = np.empty((0, 3))
    made = 0
    for k in range(int(OPT["barras"])):
        piece = rods[k % len(rods)]
        for _ in range(400):
            heading = rng.normal(size=3) * [1.0, 1.0, 0.35]
            turn = rotation_to(up, heading / np.linalg.norm(heading), 0.0)
            where = rng.uniform(-1.0, 1.0, size=3) * [OPT["campo"],
                                                      OPT["campo"], 22.0]
            trial = piece.positions @ turn.T + where
            if not len(placed) or float(cKDTree(placed).query(
                    trial)[0].min()) >= OPT["holgura"]:
                scene += Atoms(piece.get_chemical_symbols(), trial)
                groups.extend([k] * len(piece))
                placed = np.vstack((placed, trial))
                made += 1
                break
    print(f"barras colocadas: {made} de {int(OPT['barras'])}")
    save(scene, groups, "escena_barras_tubo",
         f"{made} barras de FeSe tetragonal en nanotubos",
         ((55.0, 15.0), (20.0, -40.0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
