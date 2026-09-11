"""Example 09 — EDLC on a doped carbon electrode.

Builds the constant-potential simulation used to study the electric double
layer: electrode | electrolyte | electrode, with the electrodes held at fixed
potential while their charge responds.

Two things decide whether the result means anything, and both are set up
here rather than left to be discovered:

* **Constant potential, not constant charge.** A plain MD run fixes the
  electrode charge, which is the wrong ensemble for a capacitor and gives a
  capacitance that can be off by tens of percent.
* **The force field must carry charges.** AIREBO and Tersoff — what
  carbonforge uses for neutral carbon — have none, so they cannot describe a
  double layer at all.

Run: ``python -m carbonforge.examples.ex09_edlc``
"""

from pathlib import Path

from carbonforge.builders import build_graphene_supercell
from carbonforge.exports.lammps_edlc import EDLCSettings, write_edlc
from carbonforge.forcefields import (
    assign_types,
    build_edlc_cell,
    check_edlc_setup,
    describe_provenance,
)
from carbonforge.functionalization import (
    functionalize_random,
    make_graphitic_n,
    make_pyridinic_n,
)

OUT = Path("out/edlc")


def compare_electrodes() -> None:
    """Type three electrodes and show how the doping changes the charges."""
    print("=== Cómo cambia el tipado con el dopaje ===")
    pristine = build_graphene_supercell(6, 6)
    graphitic = make_graphitic_n(pristine, n_sites=4, seed=0)
    pyridinic = make_pyridinic_n(pristine, n_defects=2, seed=0)
    hydroxylated = functionalize_random(
        pristine, "OH", n_groups=4, site_kind="basal", seed=0
    )

    for label, electrode in [
        ("prístino", pristine),
        ("N grafítico", graphitic),
        ("N piridínico", pyridinic),
        ("hidroxilado", hydroxylated),
    ]:
        counts = assign_types(electrode).counts()
        summary = ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
        print(f"  {label:14s} {summary}")

    print(
        "\n  El carbono prístino es neutro. Al dopar, el N atrae densidad y "
        "sus\n  carbonos vecinos quedan positivos: eso es lo que cambia la "
        "doble capa."
    )


def build_cell() -> None:
    """Assemble and export an aqueous EDLC on an N-doped electrode."""
    print("\n=== Celda EDLC con electrodo dopado ===")
    electrode = make_graphitic_n(build_graphene_supercell(6, 6), n_sites=4, seed=0)

    cell = build_edlc_cell(
        electrode,
        separation=40.0,
        electrolyte="aqueous",
        potential_v=1.0,
        electrolyte_kwargs={"salt": "NaCl", "molarity": 1.0, "seed": 0},
    )
    print(cell.summary())

    warnings = check_edlc_setup(cell)
    if warnings:
        print("\nAvisos:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print("\n✅ El montaje no presenta problemas conocidos.")

    settings = EDLCSettings(
        temperature_k=298.0,
        equilibration_steps=500000,
        production_steps=2000000,
    )
    written = write_edlc(cell, OUT / "aqueous", settings=settings)
    print(f"\n{len(written)} archivos en {OUT / 'aqueous'}")


def ionic_liquid_variant() -> None:
    """The same electrode with an ionic liquid instead of brine."""
    print("\n=== Variante con líquido iónico ===")
    electrode = make_graphitic_n(build_graphene_supercell(6, 6), n_sites=4, seed=0)
    cell = build_edlc_cell(
        electrode,
        separation=50.0,
        electrolyte="ionic_liquid",
        potential_v=2.0,
        electrolyte_kwargs={"seed": 0},
    )
    print(cell.summary())
    print(
        "\n  Separación mayor a propósito: las capas ordenadas de un líquido "
        "iónico\n  son más gruesas que la doble capa acuosa, y a 40 Å podrían "
        "solaparse."
    )
    write_edlc(cell, OUT / "ionic_liquid")


def show_provenance() -> None:
    """Where the parameters come from — and which ones to distrust."""
    print("\n=== Procedencia de los parámetros ===")
    print(describe_provenance(["C_sp2", "C_N", "N_graph", "OW", "Na"]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    compare_electrodes()
    build_cell()
    ionic_liquid_variant()
    show_provenance()
    print(
        "\nPara medir la capacitancia: ejecuta run, y de "
        "electrode_charge.dat\nsaca <Q>; entonces C = <Q> / V. Usa solo la "
        "etapa de producción."
    )


if __name__ == "__main__":
    main()
