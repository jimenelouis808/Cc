"""A close-up of a functionalised nanotube with a carbide particle inside.

    uv run python modelos_SEM/escena_tubo_particula.py
    uv run python modelos_SEM/escena_tubo_particula.py --freq 6 --anillos 18

* **The tube** is a capped nanotube built on its lattice
  (`build_capped_cnt`): hexagons along the barrel, six pentagons a cap.
  Wide (`freq` 5 is a 9.8 A radius) so a particle fits inside.
* **Its wall** is doped with N and Se and carries hydroxyl, carboxyl
  and carbonyl groups on the outside -- the same chemistry as
  `construir.py`'s NSeO version, at a higher coverage so it shows up
  close.
* **The particle** is cementite, Fe3C, carved as a sphere and cut by
  the tube's cavity: every atom no closer to the wall than `contacto`,
  which is how a particle trapped in a tube sits.

Made for `portada_blend.py --style revista --dopantes 1.8`, drawn
ball-and-stick so the wall is an open honeycomb and the particle shows
through it. Nothing is relaxed together; the clearance is printed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import construir
import numpy as np
from anemona_tubos import rotation_to
from ase import Atoms
from ase.io import write
from escena_fese import picture
from scipy.spatial import cKDTree

from nanocarbon_lab.builders.capped_cnt import build_capped_cnt
from nanocarbon_lab.dopants.codoping import codope
from nanocarbon_lab.functionalize.attach import functionalize

HERE = Path(__file__).resolve().parent

OPT: dict[str, float] = {
    "freq": 5,           # lattice frequency: radius 1.96 * freq A
    "anillos": 14,       # body rings
    "n": 0.06, "se": 0.025,
    "contacto": 3.0,     # A, particle atom to wall carbon
    "particula": 8.5,    # A, radius of the cementite sphere before the cut
    "semilla": 1,
}

#: Oxygen groups on the outer wall, (name, coverage).
GRAFTS = (("hydroxyl", 0.06), ("carboxyl", 0.05), ("carbonyl", 0.04))


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])
    seed = int(OPT["semilla"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tube = build_capped_cnt(n_body_rings=int(OPT["anillos"]),
                                freq=int(OPT["freq"]), shape="straight",
                                vacuum=10.0, seed=seed)
    print(f"tubo: {len(tube)} C, anillos {tube.info.get('ring_counts')}")
    # Groups FIRST, then the dopants: grafted after codoping, the three
    # passes found sites for 17 oxygens in total against 210 hydroxyls
    # alone on the undoped tube -- the doped wall no longer carries the
    # surface description the site search reads.
    wall = tube
    for k, (name, coverage) in enumerate(GRAFTS):
        try:
            wall = functionalize(wall, construir.group_for(name),
                                 coverage=coverage, face="outer",
                                 seed=seed + k)
        except ValueError as exc:
            print(f"  {name}: no se injertó ({exc})")
    wall = codope(wall, [("N", OPT["n"]), ("Se", OPT["se"])],
                  affinity=construir.AFFINITY, seed=seed)
    print(f"pared: {wall.get_chemical_formula()}")

    # The particle at the tube's middle, cut by the cavity.
    w_pos = wall.positions - wall.positions[
        np.asarray(wall.get_chemical_symbols()) == "C"].mean(axis=0)
    block, _, _, _ = construir.core_from(
        HERE / "cif" / "Fe3C_cementita.cif", "sphere",
        {"radius": OPT["particula"]})
    local = block.positions - block.positions.mean(axis=0)
    carbon_wall = w_pos[np.asarray(wall.get_chemical_symbols()) != "H"]
    keep = cKDTree(carbon_wall).query(local)[0] >= OPT["contacto"]
    particle = block[keep]
    particle.positions = local[keep]
    near = cKDTree(particle.positions).query(particle.positions, k=2)[0][:, 1]
    particle = particle[near <= 3.0]
    gap = float(cKDTree(w_pos).query(particle.positions)[0].min())
    print(f"partícula: {particle.get_chemical_formula()}, a {gap:.2f} Å "
          "de la pared como mínimo")

    # Lying across the frame on a rising diagonal.
    long = np.linalg.svd(carbon_wall, full_matrices=False)[2][0]
    axis = np.array([1.0, -0.3, 0.55])
    turn = rotation_to(long, axis / np.linalg.norm(axis), 0.3)
    scene = Atoms(wall.get_chemical_symbols(), w_pos @ turn.T) + Atoms(
        particle.get_chemical_symbols(), particle.positions @ turn.T)
    scene.set_array("grupo", np.asarray([0] * len(wall)
                                        + [1] * len(particle)))
    scene.positions -= 0.5 * (scene.positions.max(axis=0)
                              + scene.positions.min(axis=0))
    name = "escena_tubo_particula"
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"nanotubo dopado y funcionalizado con Fe3C dentro: "
                  f"{scene.get_chemical_formula()}")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene, format="extxyz")
    picture(scene, HERE / "img" / f"{name}.png", 12.0, 0.0)
    print(f"{name}: {len(scene)} átomos, {scene.get_chemical_formula()}"
          "  (`grupo`: 0 tubo, 1 partícula)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
