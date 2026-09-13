"""Example 06 — band structure, Raman/IR and spin-orbit setups.

Shows the two things that most often go wrong when preparing these
calculations, and how carbonforge catches them before you queue a job:

1. Raman on a **metallic** nanotube cannot work (QE needs a band gap for the
   dielectric response), and the PAW defaults cannot do Raman at all.
2. Spin-orbit coupling with scalar-relativistic pseudopotentials completes
   without error and returns zero splitting.

Run: ``python -m carbonforge.examples.ex06_spectra_and_bands``
"""

from pathlib import Path

from carbonforge.builders import build_cnt
from carbonforge.calculations import (
    phonon_setup,
    raman_setup,
    soc_setup,
    suggest_band_path,
)
from carbonforge.exports.qe import (
    QESettings,
    write_qe_bands,
    write_qe_spectroscopy,
)
from carbonforge.exports.siesta import SiestaSettings, write_siesta
from carbonforge.validation.calculations import check_full_setup

OUT = Path("out/spectra")

# Norm-conserving pseudopotentials: required for DFPT Raman.
NC_PSEUDOS = {"C": "C_ONCV_PBE-1.2.upf"}


def band_structure() -> None:
    """Band structure of a semiconducting (10,0) zigzag tube."""
    tube = build_cnt(10, 0, length=8.0)
    path = suggest_band_path(tube)
    print(f"Camino de bandas: {path.path_string}  ({path.total_points()} puntos k)")
    print(f"  origen: {path.source} — {path.note}")

    written = write_qe_bands(tube, OUT / "bands", force=True)
    print("  archivos:", ", ".join(p.name for p in written.values()))


def raman_the_wrong_way() -> None:
    """A request that cannot work, and the report that explains why."""
    armchair = build_cnt(6, 6, length=8.0)  # metallic
    report = check_full_setup(
        armchair,
        spectroscopy=raman_setup(),
        pseudopotentials={"C": "C.pbe-n-kjpaw_psl.1.0.0.UPF"},  # PAW
    )
    print("\n--- Raman sobre un CNT armchair con pseudos PAW ---")
    print(report.summary())


def raman_the_right_way() -> None:
    """The same request on a semiconductor with the correct pseudopotentials."""
    tube = build_cnt(10, 0, length=8.0)
    spec = raman_setup(laser_wavelength_nm=532.0)
    report = check_full_setup(
        tube, spectroscopy=spec, pseudopotentials=NC_PSEUDOS
    )
    print("\n--- Raman sobre un CNT (10,0) con pseudos norm-conserving ---")
    print(report.summary())

    settings = QESettings(pseudopotentials=NC_PSEUDOS)
    written = write_qe_spectroscopy(
        tube, OUT / "raman", spec, settings=settings, force=True
    )
    print("  archivos:", ", ".join(p.name for p in written.values()))


def phonons_on_a_metal() -> None:
    """Frequencies alone have none of the restrictions."""
    armchair = build_cnt(6, 6, length=8.0)
    report = check_full_setup(
        armchair,
        spectroscopy=phonon_setup(),
        pseudopotentials={"C": "C.pbe-n-kjpaw_psl.1.0.0.UPF"},
    )
    print("\n--- Solo fonones sobre el mismo CNT metálico ---")
    print(report.summary())


def spin_orbit() -> None:
    """SOC on pure carbon: allowed, but the warning explains the catch."""
    tube = build_cnt(10, 0, length=8.0)
    report = check_full_setup(
        tube, spinorbit=soc_setup(), pseudopotentials={"C": "C.rel-pbe-n-nc.UPF"}
    )
    print("\n--- Espín-órbita en carbono puro ---")
    print(report.summary())


def siesta_equivalent() -> None:
    """The same band structure prepared for SIESTA."""
    tube = build_cnt(10, 0, length=8.0)
    path = write_siesta(
        tube, OUT / "siesta",
        settings=SiestaSettings(run_type="bands"),
        force=True,
    )
    print(f"\nSIESTA: {path}")


def main() -> None:
    band_structure()
    raman_the_wrong_way()
    raman_the_right_way()
    phonons_on_a_metal()
    spin_orbit()
    siesta_equivalent()


if __name__ == "__main__":
    main()
