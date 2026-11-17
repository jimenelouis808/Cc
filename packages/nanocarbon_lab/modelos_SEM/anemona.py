"""A light anemone scene: one carbon surface, Fe3C in the bulbs, Fe on the wall.

    uv run python modelos_SEM/anemona.py
    uv run python modelos_SEM/anemona.py --brazos 6 --semilla 3

The earlier anemone (`construir.py`, `escena_anemona_decorada`) placed
eight finished capped tubes round a point: 28 520 atoms, arms that touch
nothing, and the carbide hidden inside a tube tip no wider than the tube.
This one is built the other way round and comes out at about a fifth of
that:

* **The carbon is ONE closed surface** (`builders.anemone`): a core,
  arms of unequal length, side branches, and a bulb at each arm's tip.
  It is a tree, so its ring budget is +12 whatever the arm count, and
  the builder checks that.
* **A Fe3C particle sits INSIDE each bulb.** Encapsulated in the literal
  sense -- the wall closes round it. Each particle is carved from the
  cementite CIF to fit ITS bulb: the bulb's free radius is measured off
  the relaxed wall, and the particle is the largest sphere that clears
  it by the contact distance.
* **Fe beads decorate the outside of the arms**, seated along the local
  outward normal with the same seating rule `construir.decorate` uses,
  kept off the bulbs (where they would read as the tip particle) and
  out of the crowded root of the arms.

**What this is and is not.** The carbon wall is a relaxed, checked
structure. The metal is PLACED: nothing here computes whether cementite
fills a bulb or iron wets a tube, and the 2.8 A contact is the drawing
convention `construir.py` already uses for iron on a wall, well inside
the 3.4 A graphitic gap. It is a model to look at and to start from,
not an interface anybody relaxed.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import construir
import numpy as np
from ase import Atoms
from ase.io import write
from scipy.spatial import cKDTree

from nanocarbon_lab.builders.anemone import build_anemone

HERE = Path(__file__).resolve().parent
CIF = HERE / "cif"

#: Defaults, each overridable as ``--name value``.
OPT: dict[str, float] = {
    "brazos": 7,
    # Core to bulb centre, in A. LONG on purpose: at 28-44 A with 10 A
    # bulbs the first version spent 3800 of its 5400 carbons on the
    # bulbs and came out as a clump of blobs on stubs. An anemone is
    # mostly arm, so the atoms go into length and the bulbs stay small.
    "largo_min": 36.0, "largo_max": 62.0,
    "ramas": 5,
    "radio_tubo": 3.3,                      # A, the meshed floor is 3.0
    # 9 A and not smaller: the relaxed wall leaves about 2 A less than
    # the request, and at 7.5 A the carbide that fits is 4 to 16 atoms.
    "radio_bulbo": 9.0,                     # A
    "radio_nucleo": 6.0,                    # A
    "contacto": 2.8,                        # A, metal atom to wall atom
    "fe_radio": 2.6,                        # A, one decorating bead
    "fe_particulas": 34,
    "semilla": 0,
}

#: Beads keep this many bulb radii clear of a bulb's centre.
BULB_CLEAR = 1.35

#: And this far, in A, beyond the core's own radius: the arms' roots
#: are too crowded for a bead to seat without bridging two arms.
CORE_CLEAR = 7.0

#: Wall atoms drawn per seat; the one furthest from every seat already
#: taken is kept (best-candidate sampling, as `construir.decorate`).
CANDIDATES = 24


def carbide_for(radius: float) -> Atoms | None:
    """The largest stoichiometric Fe3C sphere of at most `radius`, or None."""
    while radius >= 2.0:
        try:
            core, _, _, _ = construir.core_from(
                CIF / "Fe3C_cementita.cif", "sphere", {"radius": radius})
        except Exception:                            # noqa: BLE001
            core = None
        if core is not None and len(core) >= 4:
            return core
        radius -= 0.25
    return None


#: Two metal atoms further apart than this are not neighbours, in A:
#: an atom left with none after the cavity cut is dropped as loose.
LOOSE = 3.0


def fill_bulbs(carbon: Atoms, contact: float,
               bulb_radius: float) -> tuple[Atoms, list[dict]]:
    """Seat one carbide in each bulb, cut to that bulb's own cavity.

    A rigid sphere sized to the wall's NEAREST approach wastes the
    cavity: a relaxed bulb is dented, the deepest dent sets that
    radius, and the carbide that fits came out at 16 atoms in a bulb
    with room for several times that. An encapsulated particle fills
    its shell, so the particle here is cut the way the shell would cut
    it: a cementite sphere reaching the wall's MEAN radius less the
    contact, minus every atom closer to the wall than the contact,
    minus any atom that cut leaves without a neighbour.

    The price is stated in the table: the cut is no longer exactly
    3:1, and the formula printed is the one in the file.
    """
    wall = cKDTree(carbon.positions)
    positions = np.asarray(carbon.positions, dtype=float)
    metal = Atoms()
    rows = []
    for centre in np.asarray(carbon.info["bulb_centres"]):
        distance = np.linalg.norm(positions - centre, axis=1)
        shell = distance[distance < 1.2 * bulb_radius]
        mean_radius = float(shell.mean()) if shell.size else 0.0
        particle = carbide_for(mean_radius - contact)
        row = {"radio_medio": mean_radius, "atomos": 0, "formula": "—",
               "holgura": float("nan")}
        if particle is not None:
            placed = particle.copy()
            placed.positions = (particle.positions
                                - particle.positions.mean(axis=0) + centre)
            keep = wall.query(placed.positions)[0] >= contact
            placed = placed[keep]
            if len(placed) > 1:
                near = cKDTree(placed.positions).query(placed.positions,
                                                       k=2)[0][:, 1]
                placed = placed[near <= LOOSE]
            if len(placed):
                metal += placed
                row.update(atomos=len(placed),
                           formula=placed.get_chemical_formula(),
                           holgura=float(wall.query(placed.positions)[0].min()))
        rows.append(row)
    return metal, rows


def decorate(carbon: Atoms, bead: Atoms, count: int, seed: int,
             contact: float, core_radius: float,
             bulb_radius: float) -> tuple[Atoms, int]:
    """Seat `count` beads on the arms, evenly but not in rows.

    Seats are wall atoms outside the bulbs and outside the arms' roots.
    Each bead is turned at random and pushed out along the local normal
    until no atom of it is closer than `contact` to ANY wall atom -- so
    a bead between two arms rests on both rather than inside one.
    """
    positions = np.asarray(carbon.positions, dtype=float)
    centre = np.asarray(carbon.info["core_centre"], dtype=float)
    bulbs = np.asarray(carbon.info["bulb_centres"], dtype=float)
    keep = np.linalg.norm(positions - centre, axis=1) > core_radius + CORE_CLEAR
    for bulb in bulbs:
        keep &= np.linalg.norm(positions - bulb, axis=1) > BULB_CLEAR * bulb_radius
    free = np.flatnonzero(keep)
    if len(free) < CANDIDATES:
        return Atoms(), 0

    rng = np.random.default_rng(seed)
    shape = bead.positions - bead.positions.mean(axis=0)
    reach = float(np.linalg.norm(shape, axis=1).max())
    tree = cKDTree(positions)
    seats: list[int] = []
    out = Atoms()
    centres: list[np.ndarray] = []
    for _ in range(count):
        draw = rng.choice(free, size=CANDIDATES, replace=False)
        if seats:
            taken = positions[np.asarray(seats)]
            far = np.linalg.norm(positions[draw][:, None, :] - taken[None],
                                 axis=2).min(axis=1)
            pick = int(draw[int(np.argmax(far))])
        else:
            pick = int(draw[0])
        local = shape @ construir._random_rotation(rng).T
        patch = positions[tree.query(positions[pick], k=30)[1]]
        normal = np.linalg.svd(patch - patch.mean(axis=0),
                               full_matrices=False)[2][2]
        # Outward is away from the tube's own axis, found as the
        # centroid of the nearest stretch of wall.
        stretch = positions[tree.query(positions[pick], k=120)[1]]
        if float(normal @ (positions[pick] - stretch.mean(axis=0))) < 0.0:
            normal = -normal
        where = positions[pick] + normal * construir._seating(
            local, positions[pick], normal, tree, contact)
        # The seating is CHECKED. Near a branch the local normal's sign
        # can come out inward, and a bead pushed into a 3.5 A tube can
        # never clear its wall: one was left 1.1 A from it. A bead that
        # does not rest at the contact is not drawn.
        if float(tree.query(local + where)[0].min()) < contact - 0.05:
            where = positions[pick] - normal * construir._seating(
                local, positions[pick], -normal, tree, contact)
            if float(tree.query(local + where)[0].min()) < contact - 0.05:
                continue
        # Two beads must not overlap each other either.
        if any(np.linalg.norm(where - other) < 2.0 * reach + contact
               for other in centres):
            continue
        placed = bead.copy()
        placed.positions = local + where
        out += placed
        centres.append(where)
        seats.append(pick)
    return out, len(centres)


def picture(atoms: Atoms, path: Path) -> None:
    """A depth-sorted orthographic view, enough to see what was built."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions = atoms.positions - atoms.positions.mean(axis=0)
    # A fixed oblique view, so two runs can be compared.
    a, b = np.radians(25.0), np.radians(35.0)
    turn = (np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)],
                      [0, np.sin(a), np.cos(a)]])
            @ np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0],
                        [-np.sin(b), 0, np.cos(b)]]))
    view = positions @ turn.T
    order = np.argsort(view[:, 2])
    symbols = np.array(atoms.get_chemical_symbols())
    carbide = np.asarray(atoms.arrays.get("grupo", np.zeros(len(atoms))))
    colour = np.where(symbols == "C", "#3a3a3a", "#e8802f")
    colour = np.where((symbols == "C") & (carbide == 1), "#2a78d6", colour)
    size = np.where(symbols == "Fe", 16.0, 2.5)
    fig, axis = plt.subplots(figsize=(8, 8), dpi=150)
    axis.scatter(view[order, 0], view[order, 1], s=size[order],
                 c=colour[order], linewidths=0)
    axis.set_aspect("equal")
    axis.axis("off")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


#: What the carbon depends on. The metal's own knobs are not here, so
#: changing a contact or a bead count reuses the wall instead of
#: spending five minutes rebuilding the same one.
CARBON_KEYS = ("brazos", "largo_min", "largo_max", "ramas", "radio_tubo",
               "radio_bulbo", "radio_nucleo", "semilla")

#: Carried from the builder's `info` into the cache.
CARRIED = ("bulb_centres", "bulb_radii", "core_centre", "n_branches",
           "ring_counts", "ring_budget")


def carbon_wall() -> Atoms:
    """Build the anemone's wall, or reload it if nothing it depends on moved."""
    cache = HERE / "xyz" / "escena_anemona_ligera_carbono.json"
    key = {name: OPT[name] for name in CARBON_KEYS}
    if cache.exists():
        stored = json.loads(cache.read_text())
        if stored.get("clave") == key:
            carbon = Atoms(f"C{len(stored['posiciones'])}",
                           positions=stored["posiciones"])
            carbon.info.update(stored["info"])
            carbon.info["ring_counts"] = {
                int(size): count
                for size, count in stored["info"]["ring_counts"].items()}
            print("carbono: reutilizado de la caché "
                  "(borra el .json para reconstruirlo)")
            return carbon
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        carbon = build_anemone(
            arms=int(OPT["brazos"]),
            arm_length=(OPT["largo_min"], OPT["largo_max"]),
            branches=int(OPT["ramas"]),
            core_radius=OPT["radio_nucleo"], tube_radius=OPT["radio_tubo"],
            bulb_radius=OPT["radio_bulbo"], seed=int(OPT["semilla"]))
    geometry = carbon.info.get("geometry", {})
    info = {name: carbon.info[name] for name in CARRIED}
    info["geometry"] = {name: float(geometry[name])
                        for name in ("bond_min", "bond_max")
                        if name in geometry}
    info["geometry"]["n_close_contacts"] = int(
        geometry.get("n_close_contacts", 0))
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({
        "clave": key, "posiciones": carbon.positions.tolist(),
        "info": info}))
    return carbon


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])

    started = time.time()
    carbon = carbon_wall()
    geometry = carbon.info.get("geometry", {})
    census = dict(sorted(carbon.info["ring_counts"].items()))
    print(f"carbono: {len(carbon)} C en una sola superficie, "
          f"{int(OPT['brazos'])} brazos y {carbon.info['n_branches']} ramas, "
          f"{time.time() - started:.0f} s")
    print(f"  censo {census}, presupuesto {carbon.info['ring_budget']:+d}")
    print(f"  enlaces {geometry.get('bond_min', 0):.3f}–"
          f"{geometry.get('bond_max', 0):.3f} Å, "
          f"{geometry.get('n_close_contacts', 0)} contactos cercanos")

    metal, rows = fill_bulbs(carbon, OPT["contacto"], OPT["radio_bulbo"])
    print(f"\nFe3C encapsulado: {sum(1 for r in rows if r['atomos'])} de "
          f"{len(rows)} bulbos")
    for row in rows:
        print(f"  bulbo de radio medio {row['radio_medio']:.1f} Å -> "
              f"{row['formula']} ({row['atomos']} át.), a "
              f"{row['holgura']:.2f} Å de la pared")

    bead, _, _, _ = construir.core_from(
        CIF / "Fe_alfa.cif", "sphere", {"radius": OPT["fe_radio"]})
    beads, seated = decorate(carbon, bead, int(OPT["fe_particulas"]),
                             int(OPT["semilla"]), OPT["contacto"],
                             OPT["radio_nucleo"], OPT["radio_bulbo"])
    print(f"\nFe decorando la pared: {seated} partículas de {len(bead)} "
          f"átomos ({len(beads)} Fe)")
    if len(beads):
        nearest = cKDTree(carbon.positions).query(beads.positions)[0].min()
        print(f"  la más hundida a {nearest:.2f} Å de la pared "
              f"(contacto {OPT['contacto']:.1f} Å)")

    scene = Atoms(carbon.get_chemical_symbols(), carbon.positions)
    # `grupo`: 0 the carbon wall, 1 the encapsulated carbide, 2 the
    # decorating iron -- so a viewer can colour or hide each.
    groups = [0] * len(carbon) + [1] * len(metal) + [2] * len(beads)
    scene += metal
    scene += beads
    scene.set_array("grupo", np.asarray(groups, dtype=int))
    scene.positions -= scene.positions.mean(axis=0)

    (HERE / "xyz").mkdir(exist_ok=True)
    (HERE / "img").mkdir(exist_ok=True)
    plain = Atoms(scene.get_chemical_symbols(), scene.positions)
    write(str(HERE / "xyz" / "escena_anemona_ligera.xyz"), plain,
          comment=f"anemona: {scene.get_chemical_formula()}; pared de C "
                  "relajada, Fe3C y Fe COLOCADOS (no relajados)")
    write(str(HERE / "xyz" / "escena_anemona_ligera.extxyz"), scene,
          format="extxyz")
    picture(scene, HERE / "img" / "escena_anemona_ligera.png")
    print(f"\nescena: {len(scene)} átomos, {scene.get_chemical_formula()}")
    print("  xyz/escena_anemona_ligera.xyz  (.extxyz lleva la columna "
          "`grupo`: 0 pared, 1 Fe3C, 2 Fe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
