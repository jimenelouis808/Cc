"""A close-up of a bare FeSe rod, with the plate standing behind it.

    uv run python modelos_SEM/escena_barra.py
    uv run python modelos_SEM/escena_barra.py --radio 12 --semilargo 60

The rod's version of `escena_superficie.py`, and quieter: no precursors.

* **The rod** is tetragonal FeSe carved fresh as a capsule (`radio`,
  `semilargo`), fatter than `barra_T_nucleo_desnudo` so its surface
  fills a close-up; it lies across the frame on a rising diagonal.
* **The plate** (the evenly cut core of `xyz/placa_H_mixta.xyz`)
  stands behind it, face to the camera, centred: far enough back that
  the depth of field softens it.

Made for `portada_blend.py --style revista --calor --invertir-calor
--encuadrar 2`. Nothing here is relaxed; the clearance between the two
is measured and printed.
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

HERE = Path(__file__).resolve().parent

OPT: dict[str, float] = {
    "radio": 11.0,       # A, the rod's radius
    "semilargo": 55.0,   # A, half its length
    "fondo": 95.0,       # A, how far behind the rod the plate stands
}


def main() -> int:
    for index, word in enumerate(sys.argv[1:], start=1):
        if word.startswith("--") and word[2:] in OPT:
            OPT[word[2:]] = type(OPT[word[2:]])(sys.argv[index + 1])

    rod, _, _, (verdict, why) = construir.core_from(
        HERE / "cif" / "FeSe_tetragonal.cif", "rod",
        {"radius": OPT["radio"], "half_length": OPT["semilargo"]})
    r_pos = rod.positions - rod.positions.mean(axis=0)
    long = np.linalg.svd(r_pos, full_matrices=False)[2][0]
    axis = np.array([1.0, -0.25, 0.62])
    r_pos = r_pos @ rotation_to(long, axis / np.linalg.norm(axis), 0.4).T
    print(f"barra: {rod.get_chemical_formula()} -> {verdict} ({why})")

    mixed = read(str(HERE / "xyz" / "placa_H_mixta.xyz"))
    plate = mixed[np.asarray(mixed.get_chemical_symbols()) != "C"]
    p_pos = plate.positions - plate.positions.mean(axis=0)
    p_pos = p_pos @ rotation_to(np.array([0.0, 0.0, 1.0]),
                                np.array([0.0, -1.0, 0.12]), 0.0).T
    p_pos += [0.0, OPT["fondo"], 6.0]

    gap = float(cKDTree(p_pos).query(r_pos)[0].min())
    print(f"placa: {plate.get_chemical_formula()}, a {gap:.0f} Å de la barra")
    scene = Atoms(rod.get_chemical_symbols(), r_pos) + Atoms(
        plate.get_chemical_symbols(), p_pos)
    scene.set_array("grupo", np.asarray([2] * len(rod) + [0] * len(plate)))
    scene.positions -= 0.5 * (scene.positions.max(axis=0)
                              + scene.positions.min(axis=0))
    name = "escena_barra"
    write(str(HERE / "xyz" / f"{name}.xyz"),
          Atoms(scene.get_chemical_symbols(), scene.positions),
          comment=f"barra de FeSe con la placa al fondo: "
                  f"{scene.get_chemical_formula()}")
    write(str(HERE / "xyz" / f"{name}.extxyz"), scene, format="extxyz")
    picture(scene, HERE / "img" / f"{name}.png", 12.0, 0.0)
    print(f"{name}: {len(scene)} átomos, {scene.get_chemical_formula()}"
          "  (`grupo`: 2 barra, 0 placa)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
