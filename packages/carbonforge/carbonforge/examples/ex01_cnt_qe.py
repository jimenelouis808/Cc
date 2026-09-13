"""Example 01 — build a (6,6) armchair CNT and export a QE relax input.

Run: ``python -m carbonforge.examples.ex01_cnt_qe``
"""

from pathlib import Path

from carbonforge.builders import build_cnt
from carbonforge.exports.qe import write_qe_input, QESettings
from carbonforge.validation import run_basic_checks


def main() -> None:
    atoms = build_cnt(n=6, m=6, length=12.0)
    print(atoms)
    print(run_basic_checks(atoms).summary())

    outdir = Path("out/cnt_6_6_relax")
    settings = QESettings(calculation="relax", ecutwfc=60.0)
    path = write_qe_input(atoms, outdir, settings=settings)
    print(f"QE input written to {path}")


if __name__ == "__main__":
    main()
