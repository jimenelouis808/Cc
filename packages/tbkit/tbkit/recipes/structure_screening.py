"""Rank prebuilt structures of one composition by TB energy (functional groups, isomers).

Run (resumable; one file per structure; ``--part r/n`` shares the work)::

    python -m tbkit.recipes.structure_screening WORKDIR a.extxyz b.extxyz ... \\
        --model chn --kmesh 4 --host-atoms 204 --free-radius 6

The structures come in as files (built by nanocarbon_lab: the packages do not
import each other) and must share one composition, so their energies compare
directly. Each is relaxed with the model; with ``--free-radius R`` only the atoms
within R Å of the added atoms (index ≥ ``--host-atoms``) and of the sites named
in ``info["anchor"]``/``info["h_on"]`` move, which ranks local chemistry at a
fraction of the cost of a full relaxation (the host is already relaxed). With
``--prefilter K`` every structure first gets one single point (``*_sp.json``) and
only the K lowest are relaxed: for tens of candidates (a vacancy motif placed on
every inequivalent bond) relaxing all of them would take a day. The ranking is the
model's: confirm the order of the best ones with DFT.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

MAX_STEPS = 400                   # BFGS steps per relaxation (--max-steps)


def _free_mask(atoms, host_atoms: int | None, radius: float | None) -> np.ndarray | None:
    if radius is None:
        return None
    centres = list(range(host_atoms, len(atoms))) if host_atoms is not None else []
    for key in ("anchor", "h_on"):
        if key in atoms.info:
            centres.append(int(atoms.info[key]))
    if not centres:
        raise ValueError("--free-radius necesita átomos añadidos (--host-atoms) o info['anchor'].")
    d = atoms.get_all_distances(mic=True)[centres].min(axis=0)
    return d <= radius


def relax_one(path: Path, model, workdir: Path, kmesh: int, kT: float, fmax: float,
              host_atoms, radius) -> dict:
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS

    from ..calculator import TBCalculator

    out = workdir / f"{path.stem}.json"
    if out.exists():
        return json.loads(out.read_text())
    atoms = read(path)
    free = _free_mask(atoms, host_atoms, radius)
    if free is not None:
        atoms.set_constraint(FixAtoms(mask=~free))
    atoms.calc = TBCalculator(model, kpts=kmesh, kT=kT)
    opt = BFGS(atoms, logfile=str(workdir / f"{path.stem}.log"),
               trajectory=str(workdir / f"{path.stem}.traj"))
    from ..progress import Progress, watch_optimizer

    with Progress(None, f"relajando {path.stem}",
                  status=workdir / f"progreso_{path.stem}.json") as bar:
        watch_optimizer(opt, bar)
        converged = bool(opt.run(fmax=fmax, steps=MAX_STEPS))
    energy = float(atoms.get_potential_energy())
    atoms.calc = None
    atoms.set_constraint()
    write(workdir / f"{path.stem}_relaxed.extxyz", atoms)
    row = {"name": path.stem, "energy_eV": energy, "converged": converged,
           "steps": opt.get_number_of_steps(), "free_atoms": int(free.sum()) if free is not None
           else len(atoms), "formula": atoms.get_chemical_formula(),
           "info": {k: v for k, v in atoms.info.items() if isinstance(v, (int, float, str))}}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row, indent=1))
    tmp.replace(out)
    return row


def single_point(path: Path, model, workdir: Path, kmesh: int, kT: float) -> dict:
    from ..calculator import TBCalculator

    out = workdir / f"{path.stem}_sp.json"
    if out.exists():
        return json.loads(out.read_text())
    atoms = read(path)
    atoms.calc = TBCalculator(model, kpts=kmesh, kT=kT)
    row = {"name": path.stem, "energy_eV": float(atoms.get_potential_energy())}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(row))
    tmp.replace(out)
    return row


def main(argv=None) -> None:
    from ..gui.actions import load_model

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workdir", type=Path,
                        help="carpeta de trabajo (un archivo por estructura, reanudable)")
    parser.add_argument("structures", nargs="+", type=Path,
                        help="estructuras a comparar (misma composición)")
    parser.add_argument("--model", default="chn",
                        help="conjunto TB (nombre o archivo)")
    parser.add_argument("--kmesh", type=int, default=8,
                        help="puntos k por eje periódico")
    parser.add_argument("--kT", type=float, default=0.05,
                        help="temperatura electrónica (eV)")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="criterio de convergencia: fuerza máxima (eV/Å)")
    parser.add_argument("--host-atoms", type=int, default=None,
                        help="átomos del anfitrión: los añadidos después son el grupo")
    parser.add_argument("--free-radius", type=float, default=None,
                        help="relajar solo los átomos a menos de R Å del grupo (vacío: todos)")
    parser.add_argument("--part", default="0/1")
    parser.add_argument("--max-steps", type=int, default=400, help="pasos máximos de BFGS")
    parser.add_argument("--prefilter", type=int, default=None,
                        help="un punto simple por estructura y relajar solo las K más bajas")
    args = parser.parse_args(argv)
    global MAX_STEPS
    MAX_STEPS = args.max_steps
    args.workdir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)
    r, n = (int(x) for x in args.part.split("/"))
    paths = sorted(args.structures)
    formulas = {read(p).get_chemical_formula() for p in paths}
    if len(formulas) != 1:
        raise ValueError(f"Composiciones distintas {sorted(formulas)}: sus energías no se comparan.")
    single = None
    if args.prefilter:
        single = [single_point(p, model, args.workdir, args.kmesh, args.kT)
                  for k, p in enumerate(paths) if k % n == r]
        if n > 1 and not all((args.workdir / f"{p.stem}_sp.json").exists() for p in paths):
            print(f"parte {r}/{n}: {len(single)} puntos simples")
            return
        energies = {p.stem: json.loads((args.workdir / f"{p.stem}_sp.json").read_text())["energy_eV"]
                    for p in paths}
        best = sorted(paths, key=lambda p: energies[p.stem])[:args.prefilter]
        paths = sorted(best)
    from ..progress import Progress

    mine = [p for k, p in enumerate(paths) if k % n == r]
    bar = Progress(len(mine), f"cribado parte {r + 1}/{n} (estructuras relajadas)",
                   status=args.workdir / f"progreso_cribado_{r}.json")
    rows = []
    for p in mine:
        rows.append(relax_one(p, model, args.workdir, args.kmesh, args.kT, args.fmax,
                              args.host_atoms, args.free_radius))
        bar.step(note=p.stem)
    bar.close()
    if n > 1:
        print(f"parte {r}/{n}: {len(rows)} estructuras")
        return
    lowest = min(row["energy_eV"] for row in rows)
    for row in rows:
        row["relative_eV"] = row["energy_eV"] - lowest
    rows.sort(key=lambda row: row["energy_eV"])
    report = {"model": model.name, "kmesh": args.kmesh, "free_radius": args.free_radius,
              "formula": formulas.pop(), "structures": rows}
    if args.prefilter:
        sp = sorted((json.loads(f.read_text()) for f in args.workdir.glob("*_sp.json")),
                    key=lambda row: row["energy_eV"])
        report["prefilter"] = {"relaxed": args.prefilter, "single_points": [
            {"name": row["name"], "relative_eV": row["energy_eV"] - sp[0]["energy_eV"]}
            for row in sp]}
    (args.workdir / "report.json").write_text(json.dumps(report, indent=1))
    for row in rows:
        print(f"{row['name']:16s} ΔE {row['relative_eV']:+.3f} eV  "
              f"{'ok' if row['converged'] else 'SIN CONVERGER'}  {row['info']}")


if __name__ == "__main__":
    main()
