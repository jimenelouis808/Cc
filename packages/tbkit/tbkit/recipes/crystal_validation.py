"""Crystals against GPAW: how far the molecule-fitted sets carry into solids.

Every ``xu_ch*`` set was fitted on molecules; a periodic structure is an
extrapolation of it. This recipe measures that extrapolation on the doped
and functionalised crystals the package is meant for (substitutional N, B,
S, P, Se and pyridinic N in graphene, an epoxide and a hydroxyl on it,
graphane, h-BN, N in diamond, N in a nanotube), each with the smallest set
that covers its elements.

Two stages, so the comparison needs no GPAW once the references exist::

    # 1. GPAW (hours in serial). Resumable: every finished point is a file.
    python -m tbkit.recipes.crystal_validation gpaw WORKDIR [--systems a b ...]
    python -m tbkit.recipes.crystal_validation collect WORKDIR gpaw_crystals.json
    # 2. The model against them (minutes, no GPAW):
    python -m tbkit.recipes.crystal_validation compare gpaw_crystals.json OUT.json

GPAW stage, per crystal: relax the positions (cell fixed, BFGS, fmax
0.05 eV/Å) from the built geometry, then single points on three random
distortions (σ = 0.05 Å) and two uniform in-plane (or bulk) strains of ±2 %.
Same settings as the molecular references (PBE, LCAO dzp, h = 0.2 Å),
spin-paired, Fermi-Dirac kT = 0.1 eV and the same Γ-centred k mesh in both
codes, so what differs is the model and nothing else.

Checkpoints: the relaxation writes its trajectory and BFGS Hessian after
every step, a single point writes ``WORKDIR/<crystal>/<label>.json`` when it
ends (atomically), and a relaunch skips what is on disk. A kill costs at most
the SCF in progress. ``tbkit/recipes/run_crystals.sh`` relaunches the stage
until every crystal has its ``done`` marker.

Compare stage: forces (RMS error, and RMS of the DFT forces as the scale),
energies relative to the relaxed structure, and the model relaxed from the
GPAW minimum (RMS and largest displacement, bonds of the dopant, its height
above the sheet). Each set's error on crystals is set against its error on
the random distortions of its own molecular references, computed the same
way: the ratio is the size of the extrapolation.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import numpy as np
from ase import Atoms

SETTINGS = {"code": "GPAW", "mode": "lcao", "basis": "dzp", "xc": "PBE", "h": 0.2,
            "smearing_ev": 0.1, "spin": "spin-paired", "fmax": 0.05, "max_steps": 80,
            "sigma": 0.05, "n_random": 3, "strains": [0.98, 1.02], "seed": 7}

A_GRAPHENE = 2.46
A_HBN = 2.50
A_GRAPHANE = 2.54
A_DIAMOND = 3.567


def _sheet(a: float = A_GRAPHENE, n: int = 4, symbols: str = "C2") -> Atoms:
    from ase.build import graphene

    return graphene(symbols, a=a, vacuum=7.5).repeat((n, n, 1))


def _neighbours(atoms: Atoms, index: int, cutoff: float = 1.8) -> list[int]:
    d = atoms.get_distances(index, range(len(atoms)), mic=True)
    return [int(j) for j in np.argsort(d) if j != index and d[j] < cutoff]


def _substituted(element: str, lift: float = 0.0) -> Callable[[], Atoms]:
    def build():
        atoms = _sheet()
        atoms[0].symbol = element
        atoms.positions[0, 2] += lift        # S, P, Se leave the plane: let them choose
        return atoms
    return build


def _pyridinic() -> Atoms:
    atoms = _sheet()
    ring = _neighbours(atoms, 0)
    for j in ring:
        atoms[j].symbol = "N"
    del atoms[0]
    return atoms


def _epoxide() -> Atoms:
    atoms = _sheet()
    j = _neighbours(atoms, 0)[0]
    middle = atoms.positions[0] + 0.5 * atoms.get_distance(0, j, mic=True, vector=True)
    atoms.positions[[0, j], 2] += 0.15
    return atoms + Atoms("O", positions=[middle + [0.0, 0.0, 1.35]])


def _hydroxyl() -> Atoms:
    atoms = _sheet()
    atoms.positions[0, 2] += 0.35
    top = atoms.positions[0]
    return atoms + Atoms("OH", positions=[top + [0.0, 0.0, 1.46], top + [0.93, 0.0, 1.78]])


def _graphane() -> Atoms:
    """Chair graphane: C at ±0.23 Å, each with its H on its own side."""
    cell = [[A_GRAPHANE, 0, 0], [-A_GRAPHANE / 2, A_GRAPHANE * np.sqrt(3) / 2, 0], [0, 0, 15.0]]
    frac = np.array([[0.0, 0.0], [1 / 3, 2 / 3]])
    xy = frac @ np.array(cell)[:2, :2]
    z0 = 7.5
    positions = [[*xy[0], z0 - 0.23], [*xy[1], z0 + 0.23],
                 [*xy[0], z0 - 0.23 - 1.10], [*xy[1], z0 + 0.23 + 1.10]]
    atoms = Atoms("C2H2", positions=positions, cell=cell, pbc=[True, True, False])
    return atoms.repeat((2, 2, 1))


def _hbn() -> Atoms:
    return _sheet(A_HBN, 3, "BN")


def _diamond_n() -> Atoms:
    from ase.build import bulk

    atoms = bulk("C", "diamond", a=A_DIAMOND, cubic=True).repeat((2, 2, 2))
    atoms[0].symbol = "N"
    return atoms


def _nanotube_n() -> Atoms:
    from ase.build import nanotube

    atoms = nanotube(8, 0, length=2, bond=1.42, vacuum=6.0)
    atoms.pbc = [False, False, True]
    atoms[0].symbol = "N"
    return atoms


#: name -> (builder, parameter set, k mesh, index of the atom the checks follow).
#: The index is the dopant (or the carbon under the adsorbate); None for none.
CRYSTALS: dict[str, tuple] = {
    "graphene": (_sheet, "xu_chno", (3, 3, 1), None),
    "graphene_N": (_substituted("N"), "xu_chn", (3, 3, 1), 0),
    "graphene_N3V": (_pyridinic, "xu_chn", (3, 3, 1), None),
    "graphene_B": (_substituted("B"), "xu_chnob", (3, 3, 1), 0),
    "graphene_S": (_substituted("S", 0.8), "xu_chnos", (3, 3, 1), 0),
    "graphene_P": (_substituted("P", 0.8), "xu_chnop", (3, 3, 1), 0),
    "graphene_Se": (_substituted("Se", 0.9), "xu_chnose", (3, 3, 1), 0),
    "graphene_epoxide": (_epoxide, "xu_chno", (3, 3, 1), 32),
    "graphene_OH": (_hydroxyl, "xu_chno", (3, 3, 1), 32),
    "graphane": (_graphane, "xu_chn", (6, 6, 1), None),
    "hBN": (_hbn, "xu_chnob", (4, 4, 1), None),
    "cnt80_N": (_nanotube_n, "xu_chn", (1, 1, 4), 0),
    "diamond_N": (_diamond_n, "xu_chn", (2, 2, 2), 0),
}

#: The molecular references each set was fitted on that hold its own element.
MOLECULAR_BASELINE = {"xu_chn": "gpaw_chn.json", "xu_chno": "gpaw_chno.json",
                      "xu_chnob": "gpaw_b.json", "xu_chnos": "gpaw_s.json",
                      "xu_chnop": "gpaw_p.json", "xu_chnose": "gpaw_se.json"}


# --------------------------------------------------------------------------- GPAW stage

def gpaw_factory(kpts) -> Callable:
    def make():
        from gpaw import GPAW, FermiDirac

        return GPAW(mode=SETTINGS["mode"], basis=SETTINGS["basis"], xc=SETTINGS["xc"],
                    h=SETTINGS["h"], kpts=tuple(kpts), symmetry="off",
                    occupations=FermiDirac(SETTINGS["smearing_ev"]), txt=None)
    return make


def _write_atomic(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=1), encoding="utf-8")
    temporary.replace(path)


def _point(atoms: Atoms, label: str, extra: dict | None = None) -> dict:
    data = {"label": label, "symbols": atoms.get_chemical_symbols(),
            "positions": atoms.get_positions().tolist(), "cell": atoms.cell.array.tolist(),
            "pbc": [bool(p) for p in atoms.pbc], "energy": float(atoms.get_potential_energy()),
            "forces": atoms.get_forces().tolist()}
    data.update(extra or {})
    return data


def _atoms_of(data: dict) -> Atoms:
    return Atoms(data["symbols"], positions=data["positions"], cell=data["cell"],
                 pbc=data["pbc"])


def strained(atoms: Atoms, scale: float) -> Atoms:
    """Cell and positions scaled along the periodic axes only (vacuum untouched)."""
    out = atoms.copy()
    factors = np.where(atoms.pbc, scale, 1.0)
    frac = atoms.get_scaled_positions(wrap=False)
    out.set_cell(atoms.cell.array * factors[:, None], scale_atoms=False)
    # Non-periodic axes keep their Cartesian coordinates.
    new = frac @ out.cell.array
    for axis in np.flatnonzero(~atoms.pbc):
        new[:, axis] = atoms.positions[:, axis]
    out.positions = new
    return out


def distorted(atoms: Atoms) -> list[tuple[str, Atoms]]:
    rng = np.random.default_rng(SETTINGS["seed"])
    out = []
    for k in range(SETTINGS["n_random"]):
        moved = atoms.copy()
        moved.positions += rng.normal(0.0, SETTINGS["sigma"], size=moved.positions.shape)
        out.append((f"rnd{k}", moved))
    for scale in SETTINGS["strains"]:
        out.append((f"x{scale:.2f}", strained(atoms, scale)))
    return out


def _relax(name: str, folder: Path, make_calc: Callable, log) -> dict:
    """GPAW relaxation that survives a kill: resumes from its trajectory and Hessian."""
    from ase.io import Trajectory, read
    from ase.optimize import BFGS

    done = folder / "relaxed.json"
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    trajectory = folder / "relax.traj"
    hessian = folder / "relax_bfgs.json"
    atoms = CRYSTALS[name][0]()
    steps_before = 0
    if trajectory.exists() and trajectory.stat().st_size > 0:
        try:
            images = read(trajectory, index=":")
            # Each launch writes its starting image again: count moves, not images.
            steps_before = sum(not np.allclose(a.positions, b.positions)
                               for a, b in pairwise(images))
            atoms.positions = images[-1].get_positions()
            log(f"{name}: relajación retomada en el paso {steps_before}")
        except (OSError, ValueError, EOFError) as error:    # cut mid-write
            log(f"{name}: trayectoria ilegible ({error}); se empieza de nuevo")
            trajectory.unlink()
            hessian.unlink(missing_ok=True)
            steps_before = 0
    atoms.calc = make_calc()
    with Trajectory(str(trajectory), "a", atoms) as traj:
        optimizer = BFGS(atoms, restart=str(hessian), logfile=None)
        optimizer.attach(traj.write, interval=1)
        optimizer.attach(lambda: log(f"{name}: paso {steps_before + optimizer.nsteps} "
                                     f"fmax {np.linalg.norm(atoms.get_forces(), axis=1).max():.3f}"),
                         interval=1)
        converged = bool(optimizer.run(fmax=SETTINGS["fmax"],
                                       steps=max(SETTINGS["max_steps"] - steps_before, 0)))
    result = _point(atoms, "relaxed", {"converged": converged,
                                       "steps": steps_before + optimizer.nsteps})
    _write_atomic(done, result)
    return result


def run_crystal(name: str, workdir: Path, make_calc: Callable | None = None,
                log=print) -> Path:
    """The GPAW stage for one crystal; returns its ``done`` marker. Idempotent."""
    folder = Path(workdir) / name
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "done"
    if marker.exists():
        return marker
    make_calc = make_calc or gpaw_factory(CRYSTALS[name][2])
    relaxed = _relax(name, folder, make_calc, log)
    for label, atoms in distorted(_atoms_of(relaxed)):
        path = folder / f"{label}.json"
        if path.exists():
            continue
        atoms.calc = make_calc()
        _write_atomic(path, _point(atoms, label))
        log(f"{name}: {label} listo")
    marker.write_text("ok\n", encoding="utf-8")
    return marker


def collect(workdir: Path, out: Path) -> Path:
    """Finished crystals of ``workdir`` as one reference file (with the settings)."""
    import ase

    from ..references import ReferenceStructure, save_references

    structures = []
    for name, (_, model, kpts, _) in CRYSTALS.items():
        folder = Path(workdir) / name
        if not (folder / "done").exists():
            continue
        points = [json.loads((folder / "relaxed.json").read_text(encoding="utf-8"))]
        points += [json.loads((folder / f"{label}.json").read_text(encoding="utf-8"))
                   for label, _ in distorted(_atoms_of(points[0]))]
        for p in points:
            extra = {"model": model, "kpts": list(kpts)}
            if p["label"] == "relaxed":
                extra.update(converged=p["converged"], steps=p["steps"])
            structures.append(ReferenceStructure(
                f"{name}/{p['label']}", name, _atoms_of(p), p["energy"], np.array(p["forces"]),
                np.zeros(0), 0, "test", extra))
    try:
        import gpaw
        version = gpaw.__version__
    except ImportError:
        version = None
    settings = dict(SETTINGS, gpaw_version=version, ase_version=ase.__version__,
                    boundary="periodic along pbc, zero elsewhere")
    return save_references(out, structures, settings)


# --------------------------------------------------------------------------- comparison

def _model_forces(atoms: Atoms, model, kpts=None, kT: float = SETTINGS["smearing_ev"]):
    from ..calculator import TBCalculator

    atoms = atoms.copy()
    atoms.calc = TBCalculator(model, kpts=tuple(kpts) if kpts else None, kT=kT)
    return float(atoms.get_potential_energy()), atoms.get_forces()


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def molecular_baseline(set_name: str) -> dict:
    """Force errors of a set on its own molecules, measured as on the crystals.

    Two numbers, because they answer different questions: at the GPAW
    minima (``eq``: how far the model's minimum is from DFT's) and on the
    random distortions (``rnd``: the shape of the energy surface around it).
    Per structure, so the median is not ruled by one strained ring.
    """
    from ..params import load_parameters
    from ..references import load_references

    root = Path(__file__).resolve().parents[1] / "parameters" / "references"
    refs, _ = load_references(root / MOLECULAR_BASELINE[set_name])
    model = load_parameters(set_name)
    out = {"file": MOLECULAR_BASELINE[set_name]}
    for kind in ("eq", "rnd"):
        chosen = [r for r in refs if r.label.split("/")[-1].startswith(kind)]
        errors, scale, per = [], [], []
        for ref in chosen:
            _, forces = _model_forces(ref.atoms, model, kT=0.02)
            errors.append(forces - ref.forces)
            scale.append(ref.forces)
            per.append(_rms(forces - ref.forces))
        out[kind] = {"structures": len(chosen),
                     "force_rmse": _rms(np.concatenate(errors)),
                     "force_rmse_median": float(np.median(per)),
                     "force_rms_dft": _rms(np.concatenate(scale)),
                     "worst": chosen[int(np.argmax(per))].label, "worst_rmse": float(max(per))}
    return out


def lattice_minimum(points: list[dict], key: str) -> float | None:
    """Scale of the energy minimum from a parabola through x0.98, 1, x1.02."""
    energy = {p["label"]: p[key] for p in points}
    labels = ["x0.98", "relaxed", "x1.02"]
    if not all(label in energy for label in labels):
        return None
    a, b, _ = np.polyfit([0.98, 1.0, 1.02], [energy[label] for label in labels], 2)
    return float(-b / (2 * a)) if a > 0 else None


def _local(atoms: Atoms, index: int | None) -> dict:
    if index is None:
        return {}
    bonds = sorted(float(atoms.get_distance(index, j, mic=True))
                   for j in _neighbours(atoms, index, 2.1))
    out = {"bonds": [round(b, 4) for b in bonds]}
    if atoms.pbc.sum() == 2:
        plane = np.median(atoms.positions[:, 2])
        out["height"] = round(float(atoms.positions[index, 2] - plane), 4)
    return out


def compare_crystal(name: str, refs: list, kT: float = SETTINGS["smearing_ev"],
                    relax: bool = True) -> dict:
    """One crystal: the model against its GPAW points."""
    from ..params import load_parameters
    from ..tasks import relax as tb_relax

    _, set_name, kpts, index = CRYSTALS[name]
    model = load_parameters(set_name)
    by_label = {r.label.split("/", 1)[1]: r for r in refs}
    base = by_label["relaxed"]
    e0, _ = _model_forces(base.atoms, model, kpts, kT)
    points, errors, scale = [], [], []
    for label, ref in by_label.items():
        energy, forces = _model_forces(ref.atoms, model, kpts, kT)
        diff = forces - ref.forces
        errors.append(diff)
        scale.append(ref.forces)
        points.append({"label": label, "force_rmse": _rms(diff),
                       "force_max_error": float(np.linalg.norm(diff, axis=1).max()),
                       "force_rms_dft": _rms(ref.forces),
                       "dE_dft": ref.energy - base.energy, "dE_model": energy - e0})
    errors, scale = np.concatenate(errors), np.concatenate(scale)
    distortion = [p for p in points if p["label"].startswith("rnd")]
    result = {"crystal": name, "set": set_name, "kpts": list(kpts), "atoms": len(base.atoms),
              "formula": base.atoms.get_chemical_formula(), "points": points,
              "force_rmse": _rms(errors), "force_rms_dft": _rms(scale),
              "force_relative": _rms(errors) / _rms(scale),
              "force_rmse_at_minimum": next(p["force_rmse"] for p in points
                                            if p["label"] == "relaxed"),
              "force_rmse_distorted_median": float(np.median([p["force_rmse"]
                                                              for p in distortion])),
              "energy_mae": float(np.mean([abs(p["dE_model"] - p["dE_dft"])
                                           for p in distortion])),
              "lattice_scale_dft": lattice_minimum(points, "dE_dft"),
              "lattice_scale_model": lattice_minimum(points, "dE_model"),
              "gpaw_converged": base.extra.get("converged")}
    if relax:
        summary, relaxed = tb_relax(base.atoms, model, kmesh=tuple(kpts), kT=kT, fmax=0.01)
        shift = relaxed.positions - base.atoms.positions
        result["relax"] = {"converged": summary["converged"], "steps": summary["steps"],
                           "rms_displacement": _rms(np.linalg.norm(shift, axis=1)),
                           "max_displacement": float(np.linalg.norm(shift, axis=1).max()),
                           "local_dft": _local(base.atoms, index),
                           "local_model": _local(relaxed, index)}
    return result


def compare(reference_file: Path, crystals: list[str] | None = None,
            relax: bool = True) -> dict:
    from ..references import load_references

    refs, settings = load_references(reference_file)
    names = [n for n in CRYSTALS if any(r.group == n for r in refs)]
    names = [n for n in names if crystals is None or n in crystals]
    results = [compare_crystal(n, [r for r in refs if r.group == n], relax=relax)
               for n in names]
    baselines = {s: molecular_baseline(s) for s in sorted({r["set"] for r in results})}
    for r in results:
        base = baselines[r["set"]]
        r["ratio_to_molecules"] = {
            "at_minimum": r["force_rmse_at_minimum"] / base["eq"]["force_rmse_median"],
            "distorted": r["force_rmse_distorted_median"] / base["rnd"]["force_rmse_median"]}
    return {"references": Path(reference_file).name, "settings": settings,
            "molecular_baseline": baselines, "crystals": results}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("gpaw")
    g.add_argument("workdir", type=Path)
    g.add_argument("--systems", nargs="*", default=list(CRYSTALS))
    c = sub.add_parser("collect")
    c.add_argument("workdir", type=Path)
    c.add_argument("out", type=Path)
    m = sub.add_parser("compare")
    m.add_argument("references", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--systems", nargs="*")
    args = parser.parse_args(argv)

    if args.stage == "gpaw":
        log_path = args.workdir / "progress.log"
        args.workdir.mkdir(parents=True, exist_ok=True)

        def log(text):
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{os.getpid()}] {text}\n")
            print(text, flush=True)

        for name in args.systems:
            run_crystal(name, args.workdir, log=log)
            log(f"{name}: terminado")
    elif args.stage == "collect":
        print(collect(args.workdir, args.out))
    else:
        data = compare(args.references, args.systems)
        args.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        for r in data["crystals"]:
            ratio = r["ratio_to_molecules"]
            print(f"{r['crystal']:17s} {r['set']:9s} F(min) {r['force_rmse_at_minimum']:.3f} "
                  f"(x{ratio['at_minimum']:.2f} mol.)  F(rnd) {r['force_rmse_distorted_median']:.3f} "
                  f"(x{ratio['distorted']:.2f} mol.)  dE(rnd) {r['energy_mae']:.3f} eV  "
                  f"a/a0 DFT {r['lattice_scale_dft'] or float('nan'):.4f} "
                  f"TB {r['lattice_scale_model'] or float('nan'):.4f}  relax "
                  f"{r.get('relax', {}).get('max_displacement', float('nan')):.3f} Å")

if __name__ == "__main__":
    main()
