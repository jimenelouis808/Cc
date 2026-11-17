"""The anemone as CRYSTALLINE nanotubes, with no atom budget.

    uv run python modelos_SEM/anemona_tubos.py
    uv run python modelos_SEM/anemona_tubos.py --brazos 20 --semilla 4

`anemona.py` builds the anemone as one meshed surface, and for all its
topological virtue it looks wrong: the wall is the implicit route's --
lumpy, with 5-7 pairs everywhere -- and nothing like the clean
hexagonal tubes of a micrograph schematic or of `construir.py`'s own
arms. This goes back to those arms and takes them further:

* **Every arm is a capped nanotube built on its lattice**
  (`build_capped_cnt`): all hexagons along the barrel, six pentagons a
  cap, bonds graphitic by construction. Arms differ in diameter (the
  lattice quantises it: 5.9, 7.9 or 9.8 A radius), in length and in how
  far and which way they curve.
* **They grow from a central carbide particle** in its graphitic shell
  (`xyz/particula_Fe3C_C.xyz`), which is what an anemone of
  catalytically grown tubes radiates from.
* **Thin branches leave some arms**, base against the arm's wall at the
  graphitic contact, leaning toward the tip.
* **Fe3C fills the tip of every tube.** Not a sphere dropped in: a plug
  of cementite cut by the tube itself -- everything inside the tip and
  no closer to the wall than the contact -- which is how a tip-growth
  catalyst particle sits.
* **Fe beads decorate the walls**, two sizes, blue-noise seated along
  each tube's own outward normal.

This is a SCENE. Each tube is a relaxed, checked structure; where they
meet they are placed at van der Waals contact, not fused, and the metal
is placed and not relaxed. Every placement is CHECKED against a
measured clearance, and a piece that does not clear is re-drawn or left
out -- the count that made it is printed.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import construir
import numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.spatial import cKDTree

from nanocarbon_lab.builders.capped_cnt import build_capped_cnt

HERE = Path(__file__).resolve().parent
CIF = HERE / "cif"

OPT: dict[str, float] = {
    "brazos": 20,
    "ramas": 14,
    "anillos_min": 10, "anillos_max": 26,   # body rings of an arm
    "curva_max": 0.9,                       # `waviness` of the arc
    "contacto": 2.8,                        # A, metal atom to wall atom
    "hueco": 3.4,                           # A, carbon to carbon
    # A^2 of wall per Fe bead. SPARSE on purpose: at 60 the first run
    # seated 1077 beads and the tubes disappeared under them. The
    # decoration in a micrograph is a sprinkle, not a coat.
    "fe_cada": 420.0,
    "semilla": 0,
}

#: Lattice frequency -> how often an arm takes it. The radius is
#: 1.96 * freq A, so these are 5.9, 7.9 and 9.8 A.
ARM_FREQS = ((3, 0.45), (4, 0.35), (5, 0.20))

#: Branches are thinner than what they leave.
BRANCH_FREQS = (2, 3)

#: Radii of the two decorating bead sizes, in A.
BEADS = (2.6, 3.6)

#: Tries at seating one arm or one branch before giving it up.
TRIES = 80

#: A plug is this many tube radii long (drawn between the two).
PLUG_LENGTH = (1.6, 2.8)

LOOSE = 3.0


def rotation_to(source: np.ndarray, target: np.ndarray,
                roll: float) -> np.ndarray:
    """The rotation taking unit `source` onto unit `target`, then rolled.

    `roll` turns about `target`, which is what makes two arms of the
    same curvature bend different ways.
    """
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine, cosine = np.linalg.norm(cross), float(source @ target)
    if sine < 1e-9:
        base = np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = cross / sine
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        base = np.eye(3) + sine * K + (1.0 - cosine) * (K @ K)
    t = target
    T = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    spin = np.eye(3) + np.sin(roll) * T + (1.0 - np.cos(roll)) * (T @ T)
    return spin @ base


def centreline(positions: np.ndarray) -> np.ndarray:
    """A tube's axis as a polyline, base first.

    Slices along the principal axis and takes each slice's centroid.
    A principal axis is a poor parameterisation of a bent tube and a
    perfectly good way to order its slices.
    """
    centred = positions - positions.mean(axis=0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    along = centred @ axis
    bins = max(6, int(np.ptp(along) / 4.0))
    edges = np.linspace(along.min(), along.max(), bins + 1)
    points = [positions[(along >= lo) & (along <= hi)].mean(axis=0)
              for lo, hi in zip(edges[:-1], edges[1:], strict=True)
              if ((along >= lo) & (along <= hi)).any()]
    return np.asarray(points)


def make_tube(freq: int, rings: int, waviness: float, seed: int) -> dict:
    """One capped tube in its own frame: base at the origin, pointing +z."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tube = build_capped_cnt(n_body_rings=rings, freq=freq, shape="arc",
                                waviness=waviness, vacuum=10.0, seed=seed)
    positions = np.asarray(tube.positions, dtype=float)
    line = centreline(positions)
    if line[-1][2] < line[0][2]:
        line = line[::-1]
    # The base is the END of the wall, not the first slice's centroid:
    # step back along the first tangent by the cap's own reach.
    tangent = line[1] - line[0]
    tangent /= np.linalg.norm(tangent)
    reach = float(((positions - line[0]) @ -tangent).max())
    base = line[0] - tangent * reach
    turn = rotation_to(tangent, np.array([0.0, 0.0, 1.0]), 0.0)
    return {"positions": (positions - base) @ turn.T,
            "radius": float(tube.info.get("radius", 1.96 * freq)),
            "freq": freq, "rings": rings}


def placed_tube(piece: dict, origin: np.ndarray, direction: np.ndarray,
                roll: float) -> dict:
    """The tube moved so its base sits at `origin` heading along `direction`."""
    turn = rotation_to(np.array([0.0, 0.0, 1.0]), direction, roll)
    positions = piece["positions"] @ turn.T + origin
    line = centreline(positions)
    if np.linalg.norm(line[-1] - origin) < np.linalg.norm(line[0] - origin):
        line = line[::-1]
    # Outward normal and fractional position of every atom, off the
    # tube's own centreline -- so "outside" never has to be guessed
    # from a local patch, which gets the sign wrong near a neighbour.
    dense = np.concatenate([np.linspace(a, b, 12, endpoint=False)
                            for a, b in zip(line[:-1], line[1:], strict=True)]
                           + [line[-1:]])
    index = cKDTree(dense).query(positions)[1]
    outward = positions - dense[index]
    outward /= np.maximum(np.linalg.norm(outward, axis=1)[:, None], 1e-9)
    tip_tangent = line[-1] - line[-2]
    tip_tangent /= np.linalg.norm(tip_tangent)
    return {"positions": positions, "outward": outward,
            "fraction": index / max(len(dense) - 1, 1), "line": dense,
            "radius": piece["radius"], "tip_tangent": tip_tangent}


def clear_of(positions: np.ndarray, tree: cKDTree | None,
             gap: float) -> bool:
    """Whether nothing already placed comes within `gap`."""
    return tree is None or float(tree.query(positions)[0].min()) >= gap


def plug(tube: dict, block: Atoms, contact: float,
         rng: np.random.Generator) -> Atoms:
    """Cementite filling the tube's tip, cut by the tube itself.

    Kept: atoms inside the barrel near the tip, no closer to any wall
    atom than `contact`, and not left without a neighbour by that cut.
    The result is a plug the shape of the cavity, cap included, which
    is how a tip-growth catalyst sits -- and it is no longer exactly
    3:1, which the scene's summary states.
    """
    positions = tube["positions"]
    wall = cKDTree(positions)
    tangent = tube["tip_tangent"]
    apex = positions[np.argmax(positions @ tangent)]
    length = tube["radius"] * rng.uniform(*PLUG_LENGTH)
    centre = apex - tangent * (0.5 * length + contact)
    local = block.positions - block.positions.mean(axis=0)
    local = local @ construir._random_rotation(rng).T + centre
    along = (local - apex) @ tangent
    across = np.linalg.norm((local - apex) - np.outer(along, tangent), axis=1)
    keep = ((along < 0.0) & (along > -(length + contact))
            & (across < tube["radius"] - 1.0)
            & (wall.query(local)[0] >= contact))
    out = block[keep]
    out.positions = local[keep]
    if len(out) > 1:
        near = cKDTree(out.positions).query(out.positions, k=2)[0][:, 1]
        out = out[near <= LOOSE]
    return out


def picture(atoms: Atoms, path: Path, elevation: float,
            azimuth: float) -> None:
    """A depth-shaded orthographic view."""
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
    iron = np.array(atoms.get_chemical_symbols()) == "Fe"
    shade = 0.62 - 0.5 * depth                      # far is pale
    colour = np.repeat(shade[:, None], 3, axis=1)
    colour[iron] = np.column_stack((0.80 + 0.15 * depth[iron],
                                    0.42 + 0.12 * depth[iron],
                                    0.10 + 0.08 * depth[iron]))
    size = np.where(iron, 9.0, 1.6)
    fig, axis = plt.subplots(figsize=(11, 11), dpi=170)
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
    gap, contact = OPT["hueco"], OPT["contacto"]
    started = time.time()

    # --- the centre: a carbide particle in its graphitic shell.
    pieces: list[tuple[str, Atoms]] = []
    core_file = HERE / "xyz" / "particula_Fe3C_C.xyz"
    everything = np.empty((0, 3))
    start_radius = 12.0
    if core_file.exists():
        core = read(str(core_file))
        core.positions -= core.positions.mean(axis=0)
        start_radius = float(np.linalg.norm(core.positions, axis=1).max())
        pieces.append(("nucleo", core))
        everything = core.positions.copy()
        print(f"núcleo: {core.get_chemical_formula()}, radio "
              f"{start_radius:.1f} Å")

    # --- the arms.
    arms = int(OPT["brazos"])
    index = np.arange(arms, dtype=float)
    height = 1.0 - 2.0 * (index + 0.5) / arms
    radial = np.sqrt(np.maximum(0.0, 1.0 - height**2))
    angle = index * np.pi * (3.0 - np.sqrt(5.0))
    directions = np.column_stack((radial * np.cos(angle),
                                  radial * np.sin(angle), height))
    freqs = np.array([f for f, _ in ARM_FREQS])
    weights = np.array([w for _, w in ARM_FREQS])
    tubes: list[dict] = []
    for arm in range(arms):
        freq = int(rng.choice(freqs, p=weights / weights.sum()))
        rings = int(rng.integers(OPT["anillos_min"], OPT["anillos_max"] + 1))
        piece = make_tube(freq, rings, rng.uniform(0.15, OPT["curva_max"]),
                          int(rng.integers(1 << 30)))
        tree = cKDTree(everything) if len(everything) else None
        seated = None
        for attempt in range(TRIES):
            # Lean a little off the exact Fibonacci direction, and stand
            # further out with each failed try.
            lean = directions[arm] + rng.normal(scale=0.10, size=3)
            lean /= np.linalg.norm(lean)
            origin = lean * (start_radius + gap + 0.6 * attempt)
            trial = placed_tube(piece, origin, lean,
                                rng.uniform(0.0, 2.0 * np.pi))
            if clear_of(trial["positions"], tree, gap - 0.2):
                seated = trial
                break
        if seated is None:
            continue
        tubes.append(seated)
        everything = np.vstack((everything, seated["positions"]))
    print(f"brazos: {len(tubes)} de {arms}, "
          f"{sum(len(t['positions']) for t in tubes)} C")

    # --- branches: thin tubes leaving an arm's wall.
    branches = 0
    for _ in range(int(OPT["ramas"])):
        tree = cKDTree(everything)
        for _try in range(TRIES):
            host = tubes[int(rng.integers(len(tubes)))]
            eligible = np.flatnonzero((host["fraction"] > 0.3)
                                      & (host["fraction"] < 0.75))
            seat = int(rng.choice(eligible))
            normal = host["outward"][seat]
            line = host["line"]
            near = int(host["fraction"][seat] * (len(line) - 1))
            tangent = line[min(near + 3, len(line) - 1)] - line[max(near - 3, 0)]
            tangent /= np.linalg.norm(tangent)
            theta = np.radians(rng.uniform(40.0, 65.0))
            heading = np.cos(theta) * tangent + np.sin(theta) * normal
            piece = make_tube(int(rng.choice(BRANCH_FREQS)),
                              int(rng.integers(5, 11)),
                              rng.uniform(0.1, 0.4),
                              int(rng.integers(1 << 30)))
            origin = host["positions"][seat] + normal * gap
            trial = placed_tube(piece, origin, heading,
                                rng.uniform(0.0, 2.0 * np.pi))
            if clear_of(trial["positions"], tree, gap - 0.4):
                tubes.append(trial)
                everything = np.vstack((everything, trial["positions"]))
                branches += 1
                break
    print(f"ramas: {branches} de {int(OPT['ramas'])}")

    # --- cementite in every tip.
    block, _, _, _ = construir.core_from(
        CIF / "Fe3C_cementita.cif", "sphere", {"radius": 16.0})
    carbide = Atoms()
    filled = 0
    for tube in tubes:
        particle = plug(tube, block, contact, rng)
        if len(particle) >= 6:
            carbide += particle
            filled += 1
    print(f"Fe3C: {filled} de {len(tubes)} puntas, {len(carbide)} átomos "
          f"({carbide.get_chemical_formula()})")

    # --- iron on the walls.
    solid = np.vstack((everything, carbide.positions))
    tree = cKDTree(solid)
    beads = [construir.core_from(CIF / "Fe_alfa.cif", "sphere",
                                 {"radius": r})[0] for r in BEADS]
    iron = Atoms()
    centres: list[np.ndarray] = []
    seats: list[np.ndarray] = []
    for tube in tubes:
        area = 2.0 * np.pi * tube["radius"] * float(
            np.linalg.norm(np.diff(tube["line"], axis=0), axis=1).sum())
        eligible = np.flatnonzero((tube["fraction"] > 0.08)
                                  & (tube["fraction"] < 0.86))
        for _ in range(int(round(area / OPT["fe_cada"]))):
            draw = rng.choice(eligible, size=24, replace=False)
            if seats:
                far = np.linalg.norm(
                    tube["positions"][draw][:, None, :]
                    - np.asarray(seats)[None], axis=2).min(axis=1)
                pick = int(draw[int(np.argmax(far))])
            else:
                pick = int(draw[0])
            bead = beads[int(rng.integers(len(beads)))]
            local = ((bead.positions - bead.positions.mean(axis=0))
                     @ construir._random_rotation(rng).T)
            reach = float(np.linalg.norm(local, axis=1).max())
            seat, normal = tube["positions"][pick], tube["outward"][pick]
            where = seat + normal * construir._seating(local, seat, normal,
                                                       tree, contact)
            # Checked, both against every wall and against other beads.
            if float(tree.query(local + where)[0].min()) < contact - 0.05:
                continue
            if any(np.linalg.norm(where - c) < reach + r + 2.0
                   for c, r in centres):
                continue
            placed = bead.copy()
            placed.positions = local + where
            iron += placed
            centres.append((where, reach))
            seats.append(seat)
    print(f"Fe: {len(centres)} partículas, {len(iron)} átomos; la más "
          f"hundida a {tree.query(iron.positions)[0].min():.2f} Å "
          f"(contacto {contact:.1f} Å)")

    # --- assemble, with a `grupo` column: 0 tubes, 1 core particle,
    # 2 the cementite in the tips, 3 the decorating iron.
    carbon = Atoms(f"C{sum(len(t['positions']) for t in tubes)}",
                   positions=np.vstack([t["positions"] for t in tubes]))
    scene = carbon.copy()
    groups = [0] * len(carbon)
    for _name, piece in pieces:
        scene += Atoms(piece.get_chemical_symbols(), piece.positions)
        groups += [1] * len(piece)
    scene += Atoms(carbide.get_chemical_symbols(), carbide.positions)
    groups += [2] * len(carbide)
    scene += Atoms(iron.get_chemical_symbols(), iron.positions)
    groups += [3] * len(iron)
    scene.set_array("grupo", np.asarray(groups, dtype=int))
    scene.positions -= scene.positions.mean(axis=0)

    closest = float(cKDTree(scene.positions).query(scene.positions,
                                                   k=2)[0][:, 1].min())
    (HERE / "xyz").mkdir(exist_ok=True)
    (HERE / "img").mkdir(exist_ok=True)
    write(str(HERE / "xyz" / "escena_anemona_tubos.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"anemona de nanotubos: {scene.get_chemical_formula()}; "
                  "tubos relajados, metal COLOCADO (no relajado)")
    write(str(HERE / "xyz" / "escena_anemona_tubos.extxyz"), scene,
          format="extxyz")
    picture(scene, HERE / "img" / "escena_anemona_tubos.png", 22.0, 35.0)
    picture(scene, HERE / "img" / "escena_anemona_tubos_b.png", -30.0, 120.0)
    print(f"\nescena: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, {time.time() - started:.0f} s")
    print("  xyz/escena_anemona_tubos.xyz  (.extxyz lleva `grupo`: 0 tubos, "
          "1 núcleo, 2 Fe3C en las puntas, 3 Fe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
