"""Example 05 — build a realistic carbon nanocoil, pre-relax it with the
harmonic spring model, save a PNG and export to both QE and LAMMPS.

Run: ``python -m carbonforge.examples.ex05_nanocoil``
"""

from pathlib import Path

from carbonforge.builders import build_nanocoil
from carbonforge.exports.lammps import write_lammps
from carbonforge.exports.qe import QESettings, write_qe_input
from carbonforge.relax import harmonic_pre_relax
from carbonforge.validation import run_basic_checks
from carbonforge.viz import save_structure_png


def main() -> None:
    coil = build_nanocoil(
        n=6, m=6,
        coil_radius=25.0,
        pitch=12.0,
        n_turns=1.5,
        stone_wales_density=0.0,
    )
    print(coil)
    print("tube radius (Å):", coil.info["tube_radius"])
    print("arc length (Å):", coil.info["arc_length"])

    harmonic_pre_relax(coil, steps=200, step_size=0.03)
    print("after pre-relax:", coil.info["harmonic_relax"])
    print(run_basic_checks(coil).summary())

    out = Path("out/nanocoil")
    save_structure_png(coil, out / "nanocoil.png", view=(20, 45))
    write_qe_input(coil, out / "qe",
                   settings=QESettings(calculation="relax"), force=True)
    write_lammps(coil, out / "lammps", force=True)
    print("exported to", out)


if __name__ == "__main__":
    main()
