"""Example 07 — the full loop: prepare, (run), analyse.

Steps 1 and 3 run here. Step 2 is yours: carbonforge never executes
Quantum ESPRESSO or SIESTA, it only writes their inputs and reads their
outputs. To keep the example self-contained, it fabricates the output files
that a real run would have produced, so the parsing and plotting can be
demonstrated without a DFT installation.

Run: ``python -m carbonforge.examples.ex07_analyse_results``
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files rather than open windows

from carbonforge.builders import build_cnt
from carbonforge.calculations import raman_setup
from carbonforge.exports.qe import QESettings, write_qe_bands, write_qe_spectroscopy
from carbonforge.results.bands import attach_path_labels, plot_bands, read_qe_bands
from carbonforge.results.spectra import plot_spectrum, read_dynmat
from carbonforge.workflows.convergence import (
    convergence_table,
    cutoff_sweep,
    read_total_energies,
)

OUT = Path("out/analysis")
NC_PSEUDOS = {"C": "C_ONCV_PBE-1.2.upf"}


def step1_prepare() -> None:
    """Write the inputs you would submit to a cluster."""
    tube = build_cnt(10, 0, length=8.0)

    bands = write_qe_bands(tube, OUT / "bands", force=True)
    print(f"Bandas:       {len(bands)} archivos en {OUT / 'bands'}")

    spectro = write_qe_spectroscopy(
        tube, OUT / "raman", raman_setup(),
        settings=QESettings(pseudopotentials=NC_PSEUDOS), force=True,
    )
    print(f"Raman:        {len(spectro)} archivos en {OUT / 'raman'}")

    sweep = cutoff_sweep(tube, OUT / "converge", cutoffs=(40, 60, 80), force=True)
    print(f"Convergencia: {len(sweep) - 1} entradas en {OUT / 'converge'}")


def _fake_outputs() -> None:
    """Stand in for step 2 — running the codes."""
    # A bands.dat as bands.x would write it: 5 k-points, 4 bands.
    rows = []
    for index in range(5):
        kz = 0.5 * index / 4
        rows.append(f"           0.000000  0.000000  {kz:.6f}")
        base = -20.0 + index * 0.3
        rows.append(f"   {base:.3f}   {base + 11.0:.3f}   "
                    f"{base + 22.0:.3f}   {base + 26.0:.3f}")
    (OUT / "bands" / "bands.dat").write_text(
        " &plot nbnd=   4, nks=   5 /\n" + "\n".join(rows) + "\n"
    )

    (OUT / "raman" / "dynmat.out").write_text(
        "     Raman activities are in A^4/amu units\n\n"
        "# mode   [cm-1]    [THz]      IR          Raman   depol.fact\n"
        "    1       0.00    0.0000    0.0000      0.0000    0.0000\n"
        "    2       0.00    0.0000    0.0000      0.0000    0.0000\n"
        "    3       0.00    0.0000    0.0000      0.0000    0.0000\n"
        "    4     165.40    4.9585    0.0000     55.2000    0.1000\n"
        "    5    1342.10   40.2350    0.0210     28.7000    0.7500\n"
        "    6    1591.80   47.7210    0.0000    210.4000    0.0500\n"
    )

    header = ("     number of atoms/cell      =           40\n\n"
              "!    total energy              =    {:.8f} Ry\n\n     JOB DONE.\n")
    for cutoff, energy in [(40, -212.30), (60, -213.85), (80, -213.8520)]:
        (OUT / "converge" / f"ecut_{cutoff}.out").write_text(header.format(energy))


def step3_analyse() -> None:
    """Read the outputs and produce figures."""
    bands = read_qe_bands(OUT / "bands" / "bands.dat")
    attach_path_labels(bands, ["G", "X"])
    print(f"\nBandas: {bands.n_kpoints} puntos k x {bands.n_bands} bandas")
    gap = bands.band_gap(fermi=-5.0)
    print(f"  gap muestreado con E_F = -5 eV: "
          f"{'metálico' if gap is None else f'{gap:.3f} eV'}")
    plot_bands(bands, reference=-5.0).savefig(OUT / "bands.png", dpi=130)

    spectrum = read_dynmat(OUT / "raman" / "dynmat.out")
    print(f"\n{spectrum.summary()}")
    plot_spectrum(
        spectrum, "raman", laser_wavelength_nm=532.0, temperature_k=300.0
    ).savefig(OUT / "raman.png", dpi=130)

    points = read_total_energies(OUT / "converge")
    print()
    print(convergence_table(points, tolerance_mev_per_atom=1.0,
                            parameter_name="ecutwfc (Ry)"))

    print(f"\nFiguras en {OUT}/bands.png y {OUT}/raman.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("--- 1. Preparar las entradas ---")
    step1_prepare()
    print("\n--- 2. (aquí ejecutarías pw.x / ph.x — se simula) ---")
    _fake_outputs()
    print("\n--- 3. Analizar los resultados ---")
    step3_analyse()


if __name__ == "__main__":
    main()
