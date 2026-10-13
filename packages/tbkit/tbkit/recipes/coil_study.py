"""One structure file to Raman, IR and electronic structure, with tbkit alone, in one command.

What was done by hand for the user's 480-atom helix, for any wire or finite structure (a coil,
a tube, a ribbon, a molecule)::

    python -m tbkit.recipes.coil_study coil.xtl --nucleos 8

from ``packages/tbkit``. Files in ``out/estudios/<nombre>/``; every step resumes where it
stopped, so the same command continues after an interruption. The steps:

1. **preparar** -- read the file (``.xtl`` included), periodicity automatic (bonds that cross
   the cell; ≥ 5 Å of vacuum is not periodic), the periodic axis turned onto z in an
   orthorhombic box (:func:`tbkit.structure_io.align_wire`). Writes ``estructura.extxyz``
   and ``estudio.json`` (the :mod:`tbkit.recipes.study` file the other steps use).
2. **relajar** -- BFGS on the atoms, cell fixed (the period is the file's: a coil's period
   is a spring with no minimum in a model without van der Waals; see the note printed).
3. **electronica** -- bands from Γ to Z, DOS and gap of the relaxed structure.
4. **born** -- Born charges as ∂F/∂E (:func:`tbkit.infrared.born_by_field`), with an SCC set.
5. **hessiana** -- finite-difference Γ Hessian, split over ``--nucleos`` processes.
6. **raman** -- resonant Raman at ``--laseres`` (η ``--eta``), split the same way.
7. **informe** -- ``report.json`` and ``spectra.npz`` (``tbkit abrir`` / «Abrir resultado…»),
   and a summary on screen.

``--paso X`` runs the steps up to X (the ones already done return at once). Model: ``sp3`` (Xu carbon; equal to xu_chn
without SCC on pure carbon) for a carbon-only structure, else the smallest SCC set that
covers its elements; the Born charges always use an SCC set (``xu_chn`` for carbon).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

STEPS = ("preparar", "relajar", "electronica", "born", "hessiana", "raman", "informe")
NK, SIGMA = 21, 0.1


def _work(name: str) -> Path:
    return Path("out") / "estudios" / name


def _models(atoms) -> tuple[str, str]:
    """(model of energies/forces/Raman, SCC model of the Born charges)."""
    from ..gui.actions import load_model, suggest_model

    elements = set(atoms.get_chemical_symbols())
    if elements <= {"C"}:
        return "sp3", "xu_chn"
    name = suggest_model(atoms)
    if name == "pi":
        raise ValueError(f"Ningún conjunto de tbkit cubre {sorted(elements)} con repulsión: "
                         "no hay fuerzas ni modos.")
    return name, name if load_model(name).scc else "xu_chn"


def prepare(source: Path, work: Path, args) -> Path:
    """``estructura.extxyz`` and ``estudio.json`` (kept if they exist: delete to redo)."""
    from ase.io import write

    from ..structure_io import align_wire, dimensionality, read_structure

    study = work / "estudio.json"
    if study.exists():
        print(f"preparar: ya está ({study})")
        return study
    work.mkdir(parents=True, exist_ok=True)
    atoms = read_structure(source)
    if atoms.info.get("periodicity"):
        print(atoms.info["periodicity"])
    if int(atoms.pbc.sum()) > 1:
        print(f"⚠ {dimensionality(atoms.pbc)}: el IR solo tendrá los ejes abiertos que haya.")
    aligned = align_wire(atoms, vacuum=args.vacio)
    write(work / "estructura.extxyz", aligned, format="extxyz")
    model, born_model = _models(aligned)
    config = {
        "nombre": work.name, "carpeta": str(work), "modelo": model,
        "kpts": "auto" if args.kpts == "auto" else [int(k) for k in args.kpts.split(",")],
        "kT": args.kT,
        "sistemas": {"sistema": {"estructura": str((work / "estructura.extxyz").resolve())}},
        "relajar": {"fmax": args.fmax, "pasos": 2000},
        "hessiana": {"delta": 0.01, "radio": 6.0},
        "raman": {"laseres_ev": args.laseres, "eta": args.eta, "min_cm1": args.min_raman,
                  "delta": 0.005},
        "ir": {"min_cm1": 50, "delta": 0.005, "born": "campo", "modelo": born_model,
               "campo": 0.002},
    }
    study.write_text(json.dumps(config, indent=1, ensure_ascii=False))
    print(f"preparar: {len(aligned)} átomos · {aligned.get_chemical_formula()} · "
          f"{dimensionality(aligned.pbc)} · celda {np.round(np.diag(aligned.cell), 3).tolist()} Å"
          f" · modelo {model}, Born con {born_model}")
    return study


def _study(study_path: Path):
    from .study import Study

    return Study(study_path)


def _run(study_path: Path, step: str, cores: int, parts: int = 1) -> None:
    """``tbkit.recipes.study`` in child processes: ``parts`` of them with one thread each,
    or one with every core; then once more whole, to assemble the parts."""
    base = [sys.executable, "-m", "tbkit.recipes.study", str(study_path), step, "sistema"]
    if parts > 1:
        env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1"}
        logs = [open(study_path.parent / f"{step}_{r}.log", "a") for r in range(parts)]
        procs = [subprocess.Popen(base + ["--part", f"{r}/{parts}"], env=env,
                                  stdout=logs[r], stderr=subprocess.STDOUT)
                 for r in range(parts)]
        codes = [p.wait() for p in procs]
        for log in logs:
            log.close()
        if any(codes):
            raise SystemExit(f"{step}: una parte falló (mira {step}_*.log en "
                             f"{study_path.parent}).")
    env = {**os.environ, "OMP_NUM_THREADS": str(cores)}
    subprocess.run(base, env=env, check=True)


def _estimate(study_path: Path, cores: int) -> None:
    """Time one force evaluation and say what the Hessian and the Raman will take."""
    s = _study(study_path)
    atoms = s.relaxed("sistema").copy()
    atoms.calc = s.calc_factory(atoms)()
    start = time.time()
    atoms.get_forces()
    one = time.time() - start
    evaluations = 6 * len(atoms)
    hours = evaluations * one * 1.3 / cores / 3600       # one thread per part is slower
    print(f"una evaluación de fuerzas: {one:.1f} s ({cores} hilos) → hessiana ≈ "
          f"{hours:.1f} h con {cores} procesos ({evaluations} desplazamientos)")


def electronic(study_path: Path) -> dict:
    """Bands Γ–Z (or the levels of a finite structure), DOS and gap, with the study model."""
    from ..analysis import bands
    from ..hamiltonian import System
    from ..kpoints import mesh
    from ..solver import solve

    s = _study(study_path)
    out_json = s.work / "electronica.json"
    if out_json.exists():
        return json.loads(out_json.read_text())
    atoms = s.relaxed("sistema")
    system = System.build(atoms, s.model())
    kT = float(s.config["kT"])
    if system.periodic:
        solution = solve(system, *mesh(system, max(s.kpts(atoms))), kT=kT)
        line = np.c_[np.zeros(NK), np.zeros(NK), np.linspace(0, 0.5, NK)]
        eps = bands(system, line)[0] - solution.fermi
    else:
        solution = solve(system, kT=kT)
        eps = solution.energies[0] - solution.fermi
    occupied, empty = eps[eps <= 0].max(), eps[eps > 0].min()
    grid = np.linspace(-6, 6, 1201)
    weights = np.full(len(eps), 1.0 / len(eps))
    if system.periodic:
        weights = np.full(NK, 1.0)
        weights[[0, -1]] = 0.5
        weights /= weights.sum()
    dos = sum(2 * w * np.exp(-0.5 * ((grid[:, None] - e[None]) / SIGMA) ** 2).sum(1)
              for w, e in zip(weights, eps, strict=True)) / (SIGMA * np.sqrt(2 * np.pi))
    np.savez(s.work / "electronica.npz", bands=eps, grid=grid, dos=dos)
    result = {"modelo": s.config["modelo"], "fermi_eV": float(solution.fermi),
              "gap_eV": float(empty - occupied), "homo_eV": float(occupied),
              "lumo_eV": float(empty),
              "nota": ("bandas en 21 puntos de Γ a Z con la malla k del estudio; energías "
                       "respecto a E_F") if system.periodic else "niveles en Γ (finita)"}
    out_json.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def summary(study_path: Path) -> None:
    s = _study(study_path)
    report = json.loads((s.work / "report.json").read_text())
    entry = report["sistema"]
    elec = json.loads((s.work / "electronica.json").read_text())
    print(f"\n== {s.name}: {entry['atoms']} átomos, k {entry['kpts']}, modelo "
          f"{report['modelo']}")
    print(f"gap {elec['gap_eV']:.3f} eV (E_F {elec['fermi_eV']:.3f} eV)")
    for laser in report["lasers_eV"]:
        top = entry[f"strongest_{laser:.2f}"][:5]
        print(f"Raman {laser:.2f} eV: " + ", ".join(f"{r['frequency_cm1']:.0f}" for r in top)
              + " cm⁻¹ (las 5 más fuertes)")
    if "ir" in entry:
        top = entry["ir"]["strongest"][:5]
        print("IR: " + ", ".join(f"{r['frequency_cm1']:.0f} ({r['intensity_km_mol']:.1f})"
                                 for r in top) + " cm⁻¹ (km/mol)")
    print(f"archivos: {s.work}/report.json, spectra.npz, electronica.json/npz "
          "(tbkit abrir report.json)")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("archivo", type=Path, help="estructura (.xtl, .xyz extendido, .cif…)")
    parser.add_argument("--nombre", default=None, help="carpeta en out/estudios (por defecto, "
                        "el nombre del archivo)")
    parser.add_argument("--nucleos", type=int, default=os.cpu_count() or 1,
                        help="procesos para la hessiana y el Raman (uno por núcleo)")
    parser.add_argument("--paso", choices=STEPS + ("todo",), default="todo",
                        help="un solo paso (con lo que necesite antes); por defecto todos")
    parser.add_argument("--vacio", type=float, default=7.0,
                        help="Å de vacío a cada lado de los ejes abiertos")
    parser.add_argument("--kpts", default="auto", help="'auto' (≥ 60 Å por eje periódico, "
                        "como la coil) o 'n1,n2,n3'")
    parser.add_argument("--kT", type=float, default=0.05, help="ensanchamiento de Fermi, eV")
    parser.add_argument("--fmax", type=float, default=0.02, help="fuerza máxima de la "
                        "relajación, eV/Å")
    parser.add_argument("--laseres", type=float, nargs="+", default=[1.96, 2.33, 2.54],
                        help="energías del láser, eV")
    parser.add_argument("--eta", type=float, default=0.1, help="anchura η del Raman "
                        "resonante, eV")
    parser.add_argument("--min-raman", type=float, default=900.0,
                        help="Raman solo de modos por encima de esto, cm⁻¹")
    args = parser.parse_args(argv)

    work = _work(args.nombre or args.archivo.stem)
    study = prepare(args.archivo, work, args)
    order = STEPS if args.paso == "todo" else STEPS[:STEPS.index(args.paso) + 1]
    if "relajar" in order:
        _run(study, "relax", args.nucleos)
        print("nota: el período quedó fijo en el del archivo. En una coil el período es un "
              "resorte; sin van der Waals (tbkit, PBE) no tiene un mínimo fiable.")
    if "electronica" in order:
        e = electronic(study)
        print(f"electrónica: gap {e['gap_eV']:.3f} eV, E_F {e['fermi_eV']:.3f} eV")
    if "born" in order:
        _run(study, "born", args.nucleos)
    if "hessiana" in order:
        if not (work / "sistema" / "hessian.npy").exists():
            _estimate(study, args.nucleos)
        _run(study, "hessian", args.nucleos, parts=args.nucleos)
    if "raman" in order:
        _run(study, "raman", args.nucleos, parts=args.nucleos)
    if "informe" in order:
        _run(study, "report", args.nucleos)
        summary(study)

if __name__ == "__main__":
    main()
