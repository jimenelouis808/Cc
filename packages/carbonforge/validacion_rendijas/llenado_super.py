"""The same zero-coverage limit, for the superlattices themselves.

    uv run python validacion_rendijas/llenado_super.py <carpeta> [muestras]

`llenado.py` compares a flat-walled adsorbent with an idealised
groove-bearing one -- a slit against a tube bundle -- and finds the
groove offers the deeper well and the slit twice the slope per gram.
Neither is a superlattice: a superlattice carries NODES, and a node is
neither groove nor slit. This script measures the real thing.

The structures come from `nanocarbon_lab`, through files. Neither
package imports the other, so they are built by a script run inside
that package and handed over as extxyz; `<carpeta>` is where they
landed. Nothing here builds a structure, and nothing here imports
across.

Two things must happen before a superlattice can be sampled at all,
and they are not the same kind of decision -- the same split
`empotra.py` makes for the earlier series:

* **A periodic axis shorter than twice the 12 A cutoff is
  REPLICATED.** That decides nothing: a replicated cell is the same
  crystal, and the bond count under the periodic boundary is checked
  against the replica count to prove it.
* **A free axis has to be closed, and that is a choice.** A sheet is
  periodic in two directions and finite in the third, so the third is
  given a gap between images. 3.4 A, the graphitic contact, is the
  densest solid these sheets form without touching, and it is the
  same value the earlier series used so the widths stay comparable.

What the gap costs is worth knowing before reading any number: it adds
void that nobody adsorbs in, so it lowers the void fraction and the
pore volume of a sheet while leaving its channel and its area alone.
The Henry coefficient is per gram of framework and is not affected by
how much empty space the box carries -- which is the reason this
script reports `K_H` and `q_st` and not an excess density.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read
from scipy.spatial import cKDTree

from carbonforge.adsorption.henry import henry_constant
from carbonforge.adsorption.porosity import analyse_porosity

#: As everywhere else in this folder.
TEMPERATURE = 77.0

#: Twice the 12 A cutoff.
MIN_EDGE = 24.0

#: Between images along a free axis. The graphitic contact.
GAP = 3.4

#: Insertion attempts, and two seeds, for the reason `llenado.py`
#: gives: a wide cell's Widom average is noisier than its own standard
#: error admits, so the spread between seeds is the honest bar.
SAMPLES = 200_000
SEEDS = (0, 1)


def periodic_bonds(atoms: Atoms, lo: float = 1.1, hi: float = 1.8) -> int:
    """Bonds under the cell's own periodic boundary."""
    tree = cKDTree(atoms.get_positions(), boxsize=np.diag(atoms.cell))
    pairs = tree.query_pairs(hi, output_type="ndarray")
    if not len(pairs):
        return 0
    distances = np.linalg.norm(
        atoms.get_distances(pairs[:, 0], pairs[:, 1], mic=True, vector=True),
        axis=1)
    return int(((distances > lo) & (distances < hi)).sum())


def embed(atoms: Atoms) -> tuple[Atoms, list[int], float]:
    """``(cell, replica counts, gap used)``, legal for the GCMC.

    The lattice and the periodicity are read off the extxyz rather
    than assumed, which is why the handoff is extxyz and not a plain
    XYZ: a plain one carries no cell at all and ASE reads it as zero,
    so embedding into it would silently produce a structure with no
    periodicity and a report about nothing.
    """
    cell = np.asarray(atoms.cell, dtype=float)
    off = np.abs(cell - np.diag(np.diag(cell))).max()
    if off > 1e-8:
        raise ValueError(
            f"this cell is triclinic (largest off-diagonal {off:.3g} A) "
            "and every neighbour search downstream is orthorhombic-only")

    edges = np.diag(cell).astype(float).copy()
    positions = atoms.get_positions()
    free = [axis for axis, periodic in enumerate(atoms.pbc) if not periodic]
    for axis in free:
        span = positions[:, axis].max() - positions[:, axis].min()
        edges[axis] = span + GAP
        positions[:, axis] -= positions[:, axis].min() - GAP / 2.0

    atoms = atoms.copy()
    atoms.set_positions(positions)
    atoms.set_cell(np.diag(edges))
    atoms.set_pbc(True)
    atoms.wrap()

    repeat = [max(1, int(np.ceil(MIN_EDGE / edge))) for edge in edges]
    big = atoms * tuple(repeat)
    if min(np.diag(big.cell)) < MIN_EDGE:
        raise ValueError(
            f"shortest edge {min(np.diag(big.cell)):.1f} A after "
            f"replication by {repeat}, still under {MIN_EDGE:.0f}")

    # The replication is the same crystal or it is a bug: one cell's
    # bonds times the replica count, exactly.
    one = periodic_bonds(atoms)
    many = periodic_bonds(big)
    expected = one * int(np.prod(repeat))
    if many != expected:
        raise ValueError(
            f"replicating by {repeat} gave {many} bonds under the "
            f"periodic boundary against {expected} expected. A "
            "replicated cell is the same crystal, so this is a torn "
            "seam rather than a modelling choice.")
    return big, repeat, GAP if free else 0.0


def soundness(source: Path) -> tuple[bool | None, float | None]:
    """``(collapsed, angle_sum_min)`` from the builder's json, if any.

    The check itself lives in `nanocarbon_lab` and the packages do not
    import each other, so the verdict travels in the json beside the
    extxyz and this reads it. Without it this script measures whatever
    it is handed and cannot tell a tuned cell from a pinched one --
    which matters because a pinched tube IS a narrow slit and offers
    spuriously deep sites. Measured on the super-cubic grid, the cell
    with the HIGHEST coefficient of ten (23128 against the best slit's
    8660) had an angle sum of 325.2 deg, past tetrahedral, so it is
    not a hybridisation at all.

    `None` means the json says nothing about it -- an older handoff --
    and the row is reported with that stated rather than assumed
    sound.
    """
    record = source.with_suffix(".json")
    if not record.exists():
        return None, None
    try:
        data = json.loads(record.read_text())
    except (OSError, ValueError):
        return None, None
    return data.get("collapsed"), data.get("angle_sum_min")


def measure(atoms: Atoms, label: str, samples: int = SAMPLES) -> dict:
    """Henry coefficient, heat and channel, as `llenado.py` reports them."""
    runs = [henry_constant(atoms, temperature=TEMPERATURE, samples=samples,
                           seed=seed) for seed in SEEDS]
    coefficients = [run.henry_coefficient for run in runs]
    porosity = analyse_porosity(atoms, samples=20_000, area_points=200,
                               temperature=298.0, seed=0)
    return {
        "nombre": label,
        "atomos": len(atoms),
        "canal_a": porosity.largest_free_sphere,
        "area_m2_g": porosity.accessible_area_m2_g,
        "k_h": float(np.mean(coefficients)),
        "k_h_error": 0.5 * (max(coefficients) - min(coefficients)),
        "q_st": runs[0].isosteric_heat,
        "avisos": sorted({w for run in runs for w in run.warnings}),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1])
    # These cells are mostly vacuum, so their Widom average converges
    # more slowly than a slit's and the default trips the report's own
    # convergence warning. The count is an argument rather than a
    # constant because the answer to that warning is to raise it.
    samples = int(sys.argv[2]) if len(sys.argv) > 2 else SAMPLES
    sources = sorted(folder.glob("*.extxyz"))
    if not sources:
        print(f"no hay .extxyz en {folder}")
        return 1

    rows = []
    for source in sources:
        atoms = read(str(source))
        big, repeat, gap = embed(atoms)
        collapsed, angle_sum = soundness(source)
        row = measure(big, source.stem, samples=samples)
        row["collapsed"] = collapsed
        row["angle_sum_min"] = angle_sum
        row["repeticion"] = repeat
        row["hueco_a"] = gap
        row["atomos_uno"] = len(atoms)
        rows.append(row)
        print(f"{source.stem}: {len(atoms)} -> {row['atomos']} átomos "
              f"(x{repeat}), hueco {gap:.1f} Å", flush=True)

    print(f"\nH2 a {TEMPERATURE:.0f} K, límite de cobertura cero, "
          f"{samples} inserciones x {len(SEEDS)} semillas\n")
    # Sound and collapsed are listed SEPARATELY rather than ranked
    # together, because a collapsed wall's coefficient is not a worse
    # answer to the same question -- it is an answer about a different
    # structure, and ranking the two in one table invites quoting the
    # top row.
    for heading, wanted in (("sanas", False), ("colapsadas", True),
                            ("sin veredicto", None)):
        family = [r for r in rows if r["collapsed"] is wanted]
        if not family:
            continue
        print(f"#### {heading}")
        if wanted:
            print("La pared está doblada más que un carbono tetraédrico "
                  "(328.4°), así que es un tubo pinzado — una rendija "
                  "estrecha con otro nombre. **Estos K_H no son del "
                  "diseño que se pedía.**")
        print("| estructura | átomos | canal (Å) | área (m²/g) | "
              "K_H (mol/kg/MPa) | q_st (kJ/mol) | ángulo mín (°) |")
        print("|---|---|---|---|---|---|---|")
        for row in sorted(family, key=lambda r: -r["k_h"]):
            angle = ("—" if row["angle_sum_min"] is None
                     else f"{row['angle_sum_min']:.1f}")
            print(f"| {row['nombre']} | {row['atomos']} | "
                  f"{row['canal_a']:.2f} | {row['area_m2_g']:.0f} | "
                  f"{row['k_h']:.0f} ± {row['k_h_error']:.0f} | "
                  f"{row['q_st']:.2f} | {angle} |")
        print()

    for row in rows:
        for warning in row["avisos"]:
            print(f"  aviso ({row['nombre']}): {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
