"""Example 08 — functional groups and nitrogen configurations.

Focused on nanoribbons and graphene rather than nanotubes, and on the
distinction that matters most when studying nitrogen in carbon materials:

* an **attached** nitrogen group (-NH2, -NO2) hangs off a carbon;
* a **lattice** nitrogen (graphitic, pyridinic, pyrrolic) sits inside the
  ring system.

They are different chemistry, they show up at different N 1s binding
energies in XPS, and they dope the material differently. carbonforge keeps
them in separate modules for that reason.

Run: ``python -m carbonforge.examples.ex08_functional_groups``
"""

from functools import partial
from pathlib import Path

from carbonforge.builders import build_graphene_supercell, build_nanoribbon
from carbonforge.functionalization import (
    XPS_BINDING_ENERGY_EV,
    coverage,
    describe_groups,
    functionalize_bridges,
    functionalize_random,
    make_graphitic_n,
    make_pyridinic_n,
    make_pyridinic_n_oxide,
    make_pyrrolic_like,
    nitrogen_report,
)
from carbonforge.topology import ring_statistics
from carbonforge.validation import run_basic_checks
from carbonforge.exports.qe import write_qe_dos
from carbonforge.workflows import batch_structure_sweep, write_dataset

OUT = Path("out/functional")


def catalogue() -> None:
    print("=== Grupos disponibles ===")
    print(describe_groups())


def edge_groups() -> None:
    """Attach nitrogen-bearing groups to a zigzag ribbon edge."""
    print("\n=== Grupos nitrogenados en el borde de una nanocinta ===")
    ribbon = build_nanoribbon(6, 3, edge="zigzag")

    for key in ("NH2", "NO2", "CN", "CONH2"):
        decorated = functionalize_random(ribbon, key, n_groups=2, seed=0)
        report = run_basic_checks(decorated)
        info = coverage(decorated)
        print(
            f"  {key:6s} {decorated.get_chemical_formula():18s} "
            f"{info['n_groups']} grupos  válida={report.ok}"
        )


def oxidised_graphene() -> None:
    """Graphene oxide: hydroxyls and epoxides on the basal plane."""
    print("\n=== Óxido de grafeno (plano basal) ===")
    sheet = build_graphene_supercell(5, 5)

    hydroxylated = functionalize_random(
        sheet, "OH", n_groups=3, site_kind="basal", seed=1
    )
    print(f"  hidroxilado: {hydroxylated.get_chemical_formula()}")

    epoxidised = functionalize_bridges(sheet, n_groups=3, seed=1)
    print(f"  epoxidado:   {epoxidised.get_chemical_formula()}")
    print(
        "  Nota: ambos convierten carbonos sp2 en sp3 y arrugan la lámina. "
        "Relaja antes de sacar conclusiones."
    )


def nitrogen_configurations() -> None:
    """The four lattice configurations, and why they are not interchangeable."""
    print("\n=== Configuraciones de nitrógeno en la red ===")
    sheet = build_graphene_supercell(6, 6)

    variants = {
        "grafítico": make_graphitic_n(sheet, n_sites=2, seed=0),
        "piridínico": make_pyridinic_n(sheet, n_defects=1, n_per_vacancy=3, seed=0),
        "pirrólico (precursor)": make_pyrrolic_like(sheet, seed=0),
        "N-óxido": make_pyridinic_n_oxide(sheet, seed=0),
    }
    for label, structure in variants.items():
        rings = ring_statistics(structure, max_ring=8)
        print(
            f"  {label:24s} {structure.get_chemical_formula():14s} "
            f"anillos 5/6/7 = {rings[5]}/{rings[6]}/{rings[7]}"
        )

    print("\n  Energías de enlace N 1s en XPS (eV), para identificarlas:")
    for name, (low, high) in XPS_BINDING_ENERGY_EV.items():
        print(f"    {name:12s} {low:.1f} – {high:.1f}")

    print("\n--- Informe detallado del caso piridínico ---")
    print(nitrogen_report(variants["piridínico"]))


def where_does_nitrogen_contribute() -> None:
    """Set up the projected DOS, which is the question N doping really asks.

    A band structure says whether there is a gap. The PDOS says *which atoms*
    put states at the Fermi level — and that is what distinguishes graphitic
    from pyridinic nitrogen.
    """
    print("\n=== Densidad de estados proyectada ===")
    sheet = build_graphene_supercell(5, 5)
    doped = make_graphitic_n(sheet, n_sites=2, seed=0)

    written = write_qe_dos(doped, OUT / "pdos", force=True)
    print(f"  {len(written)} archivos en {OUT / 'pdos'}")
    print("  Ejecuta ./run_dos.sh y después:")
    print("    carbonforge plot-dos <carpeta> --fermi <E_F de pw.scf.out>")
    print(
        "  Te dirá qué porcentaje de los estados en E_F viene del nitrógeno."
    )


def sweep_and_export() -> None:
    """A sweep over ribbon widths and edges, each aminated, exported to QE."""
    print("\n=== Barrido de nanocintas aminadas ===")
    jobs = batch_structure_sweep(
        build_nanoribbon,
        {"width": [4, 6], "edge": ["zigzag", "armchair"], "length": [3]},
        name_prefix="gnr_nh2",
        post_factory=lambda params, seed: [
            partial(functionalize_random, group_key="NH2",
                    n_groups=2, seed=seed)
        ],
        export="qe",
    )
    manifest = write_dataset(jobs, OUT / "sweep")
    print(f"  {len(jobs)} estructuras generadas. Manifiesto: {manifest}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    catalogue()
    edge_groups()
    oxidised_graphene()
    nitrogen_configurations()
    where_does_nitrogen_contribute()
    sweep_and_export()


if __name__ == "__main__":
    main()
