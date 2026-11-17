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
* **Branches are FUSED into the arm's wall** (`build_branched_cnt`):
  one carbon network, a ring of six heptagons round each junction and
  nothing else out of place. They were placed at van der Waals contact
  before, and read as loose sticks lying on the arms.
* **Fe3C fills the tip of every tube it fits in**, arm or branch. Not a
  sphere dropped in: a plug of cementite cut by the tube itself.
* **Fe decorates the walls as caps that WET the tube.** Each bead is
  sunk toward the wall and cut by it at the Fe-C contact, so it sits on
  the carbon with a footprint instead of touching it at one atom.

This is a SCENE. Each arm is a relaxed, checked structure; the arms
stand off the central particle at van der Waals contact, and the metal
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

from nanocarbon_lab.builders.branched_cnt import build_branched_cnt

HERE = Path(__file__).resolve().parent
CIF = HERE / "cif"

OPT: dict[str, float] = {
    "brazos": 11,
    "ramas_max": 2,                         # fused branches per arm, at most
    "anillos_min": 22, "anillos_max": 44,   # body rings of an arm
    "curva_max": 0.9,                       # `waviness` of the arc
    "contacto": 2.8,                        # A, carbide atom to wall atom
    # A, decorating iron to wall. An Fe-C contact, not a van der Waals
    # one: at 2.8 the beads read as hovering over the tubes.
    "fe_contacto": 2.1,
    # Fraction of its radius a bead is sunk toward the wall before the
    # wall cuts it. 0 is a sphere touching at a point.
    "fe_hundido": 0.45,
    "hueco": 3.4,                           # A, carbon to carbon
    # A^2 of wall per Fe bead. SPARSE on purpose: at 60 the first run
    # seated 1077 beads and the tubes disappeared under them.
    "fe_cada": 420.0,
    "semilla": 0,
    # Fraction of the arms that take the thinnest tube (3.9 A radius);
    # the rest take 5.9 A, the narrowest that still holds a carbide.
    "finos": 0.3,
    "nombre": "escena_anemona_tubos",       # stem of the files written
    # A, how far each arm's base sinks below the central particle's
    # outer shell. 0 stands the arms off it at van der Waals contact;
    # above 0 the base goes INTO the shell and the shell is opened where
    # each tube enters, so the arms look grown out of the particle. An
    # appearance, not a weld: the two networks are not bonded.
    "hundir": 0.0,
}

#: A, shell carbons closer than this to an arm are removed when the
#: arms are sunk, which is what opens a mouth round each tube.
MOUTH = 2.6

#: A, the closest a sunk arm may come to the particle's metal.
METAL_CLEARANCE = 2.2

#: Lattice frequency -> how often an arm takes it. The radius is
#: 1.96 * freq A, so these are 3.9 and 5.9 A: thin tubes.
ARM_FREQS = ((2, 0.3), (3, 0.7))      # the split is OPT["finos"]

#: Branches are thinner than what they leave, so only the freq-3 arms
#: carry them.
BRANCH_FREQ = 2

#: Radii of the two decorating bead sizes BEFORE the wall cuts them.
BEADS = (3.4, 4.6)

#: Tries at seating one arm or one branch before giving it up.
TRIES = 240

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


def make_arm(freq: int, rings: int, branches: int, waviness: float,
             seed: int) -> dict:
    """One arm in its own frame: base at the origin, pointing +z.

    A capped tube with `branches` thinner tubes fused into its wall.
    Carries each atom's outward normal and its `part` (0 trunk, 1.. a
    branch), both from the builder rather than guessed from positions.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tube = build_branched_cnt(
            freq=freq, n_body_rings=rings, branches=branches,
            branch_freq=min(BRANCH_FREQ, freq), branch_rings=(8, 16),
            branch_tilt=38.0, shape="arc", waviness=waviness, seed=seed)
    positions = np.asarray(tube.positions, dtype=float)
    part = np.asarray(tube.info["part"], dtype=int)
    line = centreline(positions[part == 0])
    if line[-1][2] < line[0][2]:
        line = line[::-1]
    # The base is the END of the wall, not the first slice's centroid:
    # step back along the first tangent by the cap's own reach.
    tangent = line[1] - line[0]
    tangent /= np.linalg.norm(tangent)
    reach = float(((positions[part == 0] - line[0]) @ -tangent).max())
    base = line[0] - tangent * reach
    turn = rotation_to(tangent, np.array([0.0, 0.0, 1.0]), 0.0)
    return {"positions": (positions - base) @ turn.T,
            "normals": np.asarray(tube.info["normals"]) @ turn.T,
            "part": part, "radius": float(tube.info["radius"]),
            "branch_radius": 1.96 * min(BRANCH_FREQ, freq),
            "n_branches": int(tube.info["n_branches"]),
            "census": tube.info["ring_counts"],
            "geometry": tube.info["geometry"]}


def placed_arm(piece: dict, origin: np.ndarray, direction: np.ndarray,
               roll: float) -> dict:
    """The arm moved so its base sits at `origin` heading along `direction`.

    Also lists its `tips` -- the trunk's and each branch's -- as
    (atoms of that part, unit tangent out of the tip, tube radius),
    which is what a plug of carbide is cut against.
    """
    turn = rotation_to(np.array([0.0, 0.0, 1.0]), direction, roll)
    positions = piece["positions"] @ turn.T + origin
    part = piece["part"]
    trunk = positions[part == 0]
    line = centreline(trunk)
    if np.linalg.norm(line[-1] - origin) < np.linalg.norm(line[0] - origin):
        line = line[::-1]
    dense = np.concatenate([np.linspace(a, b, 12, endpoint=False)
                            for a, b in zip(line[:-1], line[1:], strict=True)]
                           + [line[-1:]])
    distance, index = cKDTree(dense).query(positions)
    tangent = line[-1] - line[-2]
    tips = [(trunk, tangent / np.linalg.norm(tangent), piece["radius"])]
    for label in range(1, int(part.max()) + 1):
        atoms = positions[part == label]
        if len(atoms) < 30:
            continue
        # Root: where the branch meets the trunk. Tip: the far end.
        away = distance[part == label]
        root = atoms[np.argsort(away)[:10]].mean(axis=0)
        far = atoms[np.argsort(away)[-10:]].mean(axis=0)
        heading = far - root
        tips.append((atoms, heading / np.linalg.norm(heading),
                     piece["branch_radius"]))
    return {"positions": positions, "outward": piece["normals"] @ turn.T,
            "fraction": index / max(len(dense) - 1, 1), "part": part,
            "radius": piece["radius"], "tips": tips}


def clear_of(positions: np.ndarray, tree: cKDTree | None,
             gap: float) -> bool:
    """Whether nothing already placed comes within `gap`."""
    return tree is None or float(tree.query(positions)[0].min()) >= gap


def plug(wall: cKDTree, tip: tuple, block: Atoms, contact: float,
         rng: np.random.Generator) -> Atoms:
    """Cementite filling one tube tip, cut by the tube itself.

    Kept: atoms inside the barrel near the tip, no closer to any wall
    atom than `contact`, and not left without a neighbour by that cut.
    The result is a plug the shape of the cavity, cap included, which
    is how a tip-growth catalyst sits -- and it is no longer exactly
    3:1, which the scene's summary states.
    """
    atoms, tangent, radius = tip
    apex = atoms[np.argmax(atoms @ tangent)]
    length = radius * rng.uniform(*PLUG_LENGTH)
    centre = apex - tangent * (0.5 * length + contact)
    local = block.positions - block.positions.mean(axis=0)
    local = local @ construir._random_rotation(rng).T + centre
    along = (local - apex) @ tangent
    across = np.linalg.norm((local - apex) - np.outer(along, tangent), axis=1)
    keep = ((along < 0.0) & (along > -(length + contact))
            & (across < radius - 1.0)
            & (wall.query(local)[0] >= contact))
    out = block[keep]
    out.positions = local[keep]
    if len(out) > 1:
        near = cKDTree(out.positions).query(out.positions, k=2)[0][:, 1]
        out = out[near <= LOOSE]
    return out


def wetting_bead(block: Atoms, seat: np.ndarray, normal: np.ndarray,
                 solid: cKDTree, contact: float, sink: float,
                 rng: np.random.Generator) -> Atoms:
    """An iron bead sunk onto the wall and cut by it.

    The block's centre goes `contact + (1 - sink) * radius` out along
    the normal -- so a fraction `sink` of its radius would be inside
    the contact layer -- and every atom closer to anything solid than
    `contact` is removed. What is left is a cap whose flat side follows
    the tube: it touches the carbon over a footprint, not at one atom.
    Atoms the cut leaves without an iron neighbour go too.
    """
    local = ((block.positions - block.positions.mean(axis=0))
             @ construir._random_rotation(rng).T)
    radius = float(np.linalg.norm(local, axis=1).max())
    where = local + seat + normal * (contact + (1.0 - sink) * radius)
    # Outside the wall only: the bead may not reach through a hexagon.
    keep = ((solid.query(where)[0] >= contact)
            & ((where - seat) @ normal > 0.5 * contact))
    out = block[keep]
    out.positions = where[keep]
    if len(out) > 1:
        near = cKDTree(out.positions).query(out.positions, k=2)[0][:, 1]
        out = out[near <= LOOSE]
    # The cut is on a crystal, so its lowest surviving plane can sit up
    # to a layer above the contact. Settle the cap down until its
    # nearest atom is AT the contact -- a few passes, since a step
    # along the normal does not close an oblique gap one for one.
    for _ in range(4):
        if not len(out):
            break
        slack = float(solid.query(out.positions)[0].min()) - contact
        if slack < 0.02:
            break
        out.positions -= normal * slack
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
    weights = np.array([OPT["finos"], 1.0 - OPT["finos"]])
    tubes: list[dict] = []
    fused = heptagons = 0
    bond_low, bond_high = np.inf, 0.0
    for arm in range(arms):
        freq = int(rng.choice(freqs, p=weights / weights.sum()))
        rings = int(rng.integers(OPT["anillos_min"], OPT["anillos_max"] + 1))
        # Body rings are counted in the lattice's own rows, so a thin
        # tube needs more of them for the same length.
        rings = int(round(rings * 3.0 / freq))
        wanted = (int(rng.integers(1, int(OPT["ramas_max"]) + 1))
                  if freq > BRANCH_FREQ and OPT["ramas_max"] >= 1 else 0)
        piece = make_arm(freq, rings, wanted,
                         rng.uniform(0.15, OPT["curva_max"]),
                         int(rng.integers(1 << 30)))
        sink = float(OPT["hundir"])
        if sink > 0.0 and len(pieces):
            # Arms against arms only; the shell is opened afterwards.
            others = everything[len(pieces[0][1]):]
            tree = cKDTree(others) if len(others) else None
            metal = cKDTree(pieces[0][1].positions[
                np.asarray(pieces[0][1].get_chemical_symbols()) != "C"])
            stand = start_radius - sink
        else:
            tree = cKDTree(everything) if len(everything) else None
            metal, stand = None, start_radius + gap
        seated = None
        for attempt in range(TRIES):
            # Lean a little off the exact Fibonacci direction, and stand
            # further out with each failed try.
            lean = directions[arm] + rng.normal(scale=0.10, size=3)
            lean /= np.linalg.norm(lean)
            origin = lean * (stand + 0.6 * (attempt // 6))
            trial = placed_arm(piece, origin, lean,
                               rng.uniform(0.0, 2.0 * np.pi))
            if clear_of(trial["positions"], tree, gap - 0.2) and (
                    metal is None
                    or clear_of(trial["positions"], metal, METAL_CLEARANCE)):
                seated = trial
                break
        if seated is None:
            continue
        tubes.append(seated)
        everything = np.vstack((everything, seated["positions"]))
        fused += piece["n_branches"]
        heptagons += piece["census"].get(7, 0)
        bond_low = min(bond_low, piece["geometry"]["bond_min"])
        bond_high = max(bond_high, piece["geometry"]["bond_max"])
        print(f"  brazo {arm + 1}: radio {piece['radius']:.1f} Å, "
              f"{len(piece['positions'])} C, {piece['n_branches']} ramas "
              f"fundidas, {time.time() - started:.0f} s", flush=True)
    if float(OPT["hundir"]) > 0.0 and len(pieces):
        core = pieces[0][1]
        arm_tree = cKDTree(np.vstack([t["positions"] for t in tubes]))
        is_carbon = np.asarray(core.get_chemical_symbols()) == "C"
        mouth = is_carbon & (arm_tree.query(core.positions)[0] < MOUTH)
        core = core[~mouth]
        pieces[0] = ("nucleo", core)
        everything = np.vstack([core.positions]
                               + [t["positions"] for t in tubes])
        print(f"coraza abierta: {int(mouth.sum())} C quitados donde "
              f"entran los brazos (hundidos {OPT['hundir']:.1f} Å)")
    print(f"brazos: {len(tubes)} de {arms}, "
          f"{sum(len(t['positions']) for t in tubes)} C; {fused} ramas "
          f"fundidas con {heptagons} heptágonos (6 por unión); enlaces "
          f"{bond_low:.3f}-{bond_high:.3f} Å")

    # --- cementite in every tip it fits in, trunk and branch alike.
    block, _, _, _ = construir.core_from(
        CIF / "Fe3C_cementita.cif", "sphere", {"radius": 16.0})
    carbide = Atoms()
    filled = tips = 0
    wall = cKDTree(everything)
    for tube in tubes:
        for tip in tube["tips"]:
            tips += 1
            particle = plug(wall, tip, block, contact, rng)
            if len(particle) >= 6:
                carbide += particle
                filled += 1
    print(f"Fe3C: {filled} de {tips} puntas, {len(carbide)} átomos "
          f"({carbide.get_chemical_formula()})")

    # --- iron wetting the walls.
    fe_contact, sink = OPT["fe_contacto"], OPT["fe_hundido"]
    solid = np.vstack((everything, carbide.positions))
    tree = cKDTree(solid)
    beads = [construir.core_from(CIF / "Fe_alfa.cif", "sphere",
                                 {"radius": r})[0] for r in BEADS]
    iron = Atoms()
    seats: list[np.ndarray] = []
    count = 0
    for tube in tubes:
        area = 2.62 * len(tube["positions"])        # A^2 per sp2 carbon
        eligible = np.flatnonzero((tube["part"] > 0)
                                  | ((tube["fraction"] > 0.08)
                                     & (tube["fraction"] < 0.9)))
        for _ in range(int(round(area / OPT["fe_cada"]))):
            draw = rng.choice(eligible, size=24, replace=False)
            if seats:
                far = np.linalg.norm(
                    tube["positions"][draw][:, None, :]
                    - np.asarray(seats)[None], axis=2).min(axis=1)
                pick = int(draw[int(np.argmax(far))])
            else:
                pick = int(draw[0])
            seat, normal = tube["positions"][pick], tube["outward"][pick]
            if seats and float(np.linalg.norm(
                    np.asarray(seats) - seat, axis=1).min()) < 2.4 * BEADS[1]:
                continue
            bead = wetting_bead(beads[int(rng.integers(len(beads)))], seat,
                                normal, tree, fe_contact, sink, rng)
            if len(bead) < 8:
                continue
            iron += bead
            seats.append(seat)
            count += 1
    gaps = tree.query(iron.positions)[0]
    touching = int((gaps < fe_contact + 0.6).sum())
    print(f"Fe: {count} partículas, {len(iron)} átomos; el más cercano al "
          f"carbono a {gaps.min():.2f} Å (contacto {fe_contact:.1f} Å), "
          f"{touching / max(count, 1):.1f} átomos de Fe por partícula en "
          "contacto con la pared")

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
    name = str(OPT["nombre"])
    (HERE / "xyz").mkdir(exist_ok=True)
    (HERE / "img").mkdir(exist_ok=True)
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"anemona de nanotubos: {scene.get_chemical_formula()}; "
                  "tubos relajados, metal COLOCADO (no relajado)")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene,
          format="extxyz")
    picture(scene, HERE / "img" / f"{name}.png", 22.0, 35.0)
    picture(scene, HERE / "img" / f"{name}_b.png", -30.0, 120.0)
    print(f"\nescena: {len(scene)} átomos, {scene.get_chemical_formula()}, "
          f"par más cercano a {closest:.2f} Å, {time.time() - started:.0f} s")
    print(f"  xyz/{name}.xyz  (.extxyz lleva `grupo`: 0 tubos, "
          "1 núcleo, 2 Fe3C en las puntas, 3 Fe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
