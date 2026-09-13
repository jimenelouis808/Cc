"""Example 03 — batch generation of N-doped CNTs with vacancies, exported
to Quantum ESPRESSO. A dataset.json metadata file is also written.
"""

from pathlib import Path

from nanocarbon_lab.workflows import batch_cnt_sweep, write_dataset


def main() -> None:
    jobs = batch_cnt_sweep(
        chiralities=[(6, 6), (8, 0), (6, 3)],
        lengths=[8.0, 12.0],
        dopant="N",
        dopant_concentrations=[0.0, 0.05],
        vacancies=[0, 1],
        seed=0,
        export="qe",
    )
    meta = write_dataset(jobs, Path("out/cnt_dataset"))
    print(f"Generated {len(jobs)} structures. Manifest: {meta}")


if __name__ == "__main__":
    main()
