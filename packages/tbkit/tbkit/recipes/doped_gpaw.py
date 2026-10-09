"""GPAW check of the doped-coil results (tbkit.recipes.doped_raman) on the modes that matter.

Run from packages/tbkit (resumable; files in out/doped_gpaw/<name>/)::

    python -m tbkit.recipes.doped_gpaw select NAME            # tbkit: which modes (no GPAW)
    python -m tbkit.recipes.doped_gpaw relax  NAME            # GPAW-PBE relaxation
    python -m tbkit.recipes.doped_gpaw modes  NAME [--part r/n]   # GPAW ± along each mode
    python -m tbkit.recipes.doped_gpaw report                 # tbkit vs GPAW tables

NAME is ``pristine``, ``N`` or ``amine``. GPAW exactly as in the nanocoil recipe
(LCAO dzp, PBE, h = 0.2 Å, Fermi-Dirac 0.05 eV, 1×1×2 k: PBE has a 0.22 eV gap and
forces at kz = 2 agree with kz = 4 to 6e-4 eV/Å). The pristine coil keeps its PBE
geometry from that recipe; N and the amine are relaxed here from their xu_chn
geometries (fmax 0.05 eV/Å).

For each chosen xu_chn mode L (same atom order in both methods) GPAW is run at
x_PBE ± s·L, s moving the most displaced atom by ``MAX_DISP``. From the two
calculations:

* frequency: Rayleigh quotient ω² = L·K·L / L·M·L with K·L from the forces, the
  PBE frequency of that displacement pattern. Exact if L is a PBE eigenmode, an upper
  bound otherwise (a TB mode that mixes PBE modes comes out higher).
* IR: dμ/dQ along L from GPAW's dipole, components across the axis only (x, y), as
  in the tbkit IR; intensity in km/mol.

Modes chosen per structure: the 3 strongest Raman lines at 2.33 eV in the G region,
the 2 strongest breathing-weighted lines in the D window, the 3 strongest IR lines,
the highest carbon mode, and the dopant's own: graphitic N's 3 modes with most
weight on N, the amine's group modes above 900 cm⁻¹ (> 40 % on N, its H and the C
bonded to them).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

from ..settings import Param, add_arguments, apply

WORK = Path("out/doped_gpaw")
KZ, KT, VACUUM, MAX_DISP = 2, 0.05, 6.0, 0.01
GPAW_SETTINGS = {"mode": "lcao", "basis": "dzp", "xc": "PBE", "h": 0.2}
FMAX, MAX_STEPS = 0.05, 300
PRISTINE_PBE = Path("out/nanocoil/gpaw_relax/relaxed.extxyz")
NAMES = ("pristine", "N", "amine")


def folder(name: str) -> Path:
    path = WORK / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def _gpaw():
    from gpaw import GPAW, FermiDirac

    from ..progress import watch_gpaw_scf

    calc = GPAW(**GPAW_SETTINGS, kpts=(1, 1, KZ), symmetry="off", occupations=FermiDirac(KT), txt=None)
    watch_gpaw_scf(calc, "GPAW SCF (coil)")
    return calc


# ---------------------------------------------------------------- tbkit side
def select(name: str) -> list[dict]:
    """Pick the modes and store their xu_chn vectors, frequencies and intensities."""
    from ..modes import participation
    from ..sites import bond_graph, ring_breathing
    from . import doped_raman as tb

    vib = tb.vibrations(name)
    raman = tb.raman(name)
    ir = tb.infrared(name)
    atoms = vib.atoms
    f = vib.frequencies
    symbols = np.array(atoms.get_chemical_symbols())
    hetero = set(np.flatnonzero(symbols != "C").tolist())
    near = {j for i, j in bond_graph(atoms) if i in hetero} | \
           {i for i, j in bond_graph(atoms) if j in hetero}
    groups = {"local": sorted(hetero | near)} if hetero else {}
    if "N" in symbols:
        groups["N"] = np.flatnonzero(symbols == "N").tolist()
    shares = participation(vib, groups) if groups else {}

    nu_l = 2.33 * 8065.544
    w = raman.frequencies
    li = list(tb.LASERS_EV).index(2.33)
    intensity = raman.activities[li] * (nu_l - w) ** 4 / w / (1 - np.exp(-1.4387769 * w / 300))
    breathing = ring_breathing(atoms, vib.modes[raman.mode_indices])["B"]
    ir_of = dict(zip((int(k) for k in ir["mode_indices"]), ir["intensities"], strict=True))
    raman_of = dict(zip((int(k) for k in raman.mode_indices), intensity / intensity.max(),
                        strict=True))

    picks: dict[int, list[str]] = {}

    def add(indices, label):
        for k in indices:
            picks.setdefault(int(k), []).append(label)

    g = np.flatnonzero((w > 1500) & (w < 1700))
    add(raman.mode_indices[g[np.argsort(-intensity[g])[:3]]], "G")
    dwin = np.flatnonzero((w > 1100) & (w < 1450))
    add(raman.mode_indices[dwin[np.argsort(-(intensity * breathing)[dwin])[:2]]], "D")
    add(ir["mode_indices"][np.argsort(-ir["intensities"])[:3]], "IR")
    carbon_like = np.flatnonzero(f < 2000)
    add([carbon_like[np.argmax(f[carbon_like])]], "más alto")
    if "N" in shares and name == "N":
        add(np.argsort(-shares["N"])[:3], "N")
    if name == "amine":
        add(np.flatnonzero((shares["local"] > 0.4) & (f > 900)), "grupo")

    rows, vectors = [], []
    for k in sorted(picks):
        rows.append({"mode": k, "why": picks[k], "tb_cm1": float(f[k]),
                     "tb_ir_km_mol": float(ir_of.get(k, np.nan)),
                     "tb_raman_2.33_rel": float(raman_of[k]) if k in raman_of else None,
                     **{f"on_{g}": float(v[k]) for g, v in shares.items()}})
        vectors.append(vib.modes[k])
    np.save(folder(name) / "vectors.npy", np.array(vectors))
    np.save(folder(name) / "masses.npy", atoms.get_masses())
    _write(folder(name) / "picks.json", rows)
    return rows


# ---------------------------------------------------------------- GPAW side
def _trimmed(atoms):
    out = atoms.copy()
    xy = out.positions[:, :2]
    cell = out.cell.array.copy()
    cell[0, 0], cell[1, 1] = xy.max(axis=0) - xy.min(axis=0) + 2 * VACUUM
    out.set_cell(cell, scale_atoms=False)
    out.positions[:, :2] += -xy.min(axis=0) + VACUUM
    return out


def relaxed(name: str):
    """PBE geometry: the nanocoil recipe's for the pristine coil, relaxed here otherwise."""
    if name == "pristine":
        return read(PRISTINE_PBE)
    path = folder(name) / "relaxed.extxyz"
    if path.exists():
        return read(path)
    from ase.io import Trajectory
    from ase.optimize import BFGS

    from . import doped_raman as tb

    atoms = _trimmed(read(tb.WORK / name / "relaxed.extxyz"))
    traj = folder(name) / "relax.traj"
    if traj.exists() and traj.stat().st_size > 0:
        try:
            atoms.positions = read(traj, index=-1).positions
        except Exception:                           # noqa: BLE001 - cut mid-write
            traj.unlink()
    atoms.calc = _gpaw()
    with Trajectory(str(traj), "a", atoms) as t:
        opt = BFGS(atoms, restart=str(folder(name) / "relax_bfgs.json"),
                   logfile=str(folder(name) / "relax.log"))
        opt.attach(t.write, interval=1)
        converged = bool(opt.run(fmax=FMAX, steps=MAX_STEPS))
    atoms.calc = None
    write(path, atoms)
    _write(folder(name) / "relax.json", {"converged": converged, "fmax": FMAX})
    return atoms


def mode_jobs(name: str) -> list[Path]:
    """The output file of every ± calculation of ``name``, in job order."""
    picks = json.loads((folder(name) / "picks.json").read_text())
    return [folder(name) / f"mode_{p['mode']:04d}_{t}.json" for p in picks for t in ("p", "m")]


def modes(name: str, part=(0, 1), job: int | None = None) -> None:
    """``job``: only that calculation. One GPAW per process is the safe way to run
    them: a process that ran several grew to 6.5 GB (the calculators are not freed)
    and was killed for memory with three running on 15 GB."""
    base = relaxed(name)
    vectors = np.load(folder(name) / "vectors.npy")
    picks = json.loads((folder(name) / "picks.json").read_text())
    jobs = [(j, s) for j in range(len(picks)) for s in (1, -1)]
    for n, (j, sign) in enumerate(jobs):
        if (job is not None and n != job) or (job is None and n % part[1] != part[0]):
            continue
        path = folder(name) / f"mode_{picks[j]['mode']:04d}_{'p' if sign > 0 else 'm'}.json"
        if path.exists():
            continue
        step = MAX_DISP / np.linalg.norm(vectors[j], axis=1).max()
        atoms = base.copy()
        atoms.positions += sign * step * vectors[j]
        atoms.calc = _gpaw()
        forces = atoms.get_forces()
        _write(path, {"step": float(step), "forces": forces.tolist(),
                      "dipole": atoms.get_dipole_moment().tolist(),
                      "energy": float(atoms.get_potential_energy())})


# ---------------------------------------------------------------- comparison
def report() -> dict:
    from ase.units import _amu, _e, _hbar, invcm

    from ..infrared import DEBYE_PER_EA, KM_PER_MOL
    from . import doped_raman as tb

    scale = _hbar * 1e10 / np.sqrt(_e * _amu) / invcm
    out = {"gpaw": "LCAO dzp, PBE, h 0.2, FD 0.05 eV, kz 2", "max_disp_A": MAX_DISP}
    for name in NAMES:
        path = folder(name) / "picks.json"
        if not path.exists():
            continue
        picks = json.loads(path.read_text())
        vectors = np.load(folder(name) / "vectors.npy")
        masses = np.load(folder(name) / "masses.npy")
        rows = []
        for j, pick in enumerate(picks):
            files = [folder(name) / f"mode_{pick['mode']:04d}_{t}.json" for t in ("p", "m")]
            if not all(p.exists() for p in files):
                continue
            plus, minus = (json.loads(p.read_text()) for p in files)
            s = plus["step"]
            u = vectors[j]
            k_u = -(np.array(plus["forces"]) - np.array(minus["forces"])) / (2 * s)
            curvature = float(np.sum(u * k_u))
            inertia = float(np.sum(masses[:, None] * u ** 2))
            pbe = float(np.sign(curvature) * scale * np.sqrt(abs(curvature) / inertia))
            dmu = (np.array(plus["dipole"]) - np.array(minus["dipole"]))[:2] / (2 * s)
            # u = L = e/√m (1/√amu), so dμ along s·L is dμ/dQ in e·Å/(Å·√amu) = e/√amu
            ir = float(np.sum(dmu ** 2) * DEBYE_PER_EA ** 2 * KM_PER_MOL)
            rows.append({**pick, "pbe_cm1": pbe, "pbe_over_tb": pbe / pick["tb_cm1"],
                         "pbe_ir_km_mol": ir})
        entry = {"modes": rows}
        if rows:
            ratio = np.array([r["pbe_over_tb"] for r in rows])
            entry["pbe_over_tb_mean"] = float(ratio.mean())
            entry["pbe_over_tb_range"] = [float(ratio.min()), float(ratio.max())]
        if name != "pristine" and (folder(name) / "relaxed.extxyz").exists():
            entry["geometry_tb_vs_pbe"] = _geometry(read(tb.WORK / name / "relaxed.extxyz"),
                                                    read(folder(name) / "relaxed.extxyz"))
        out[name] = entry
    _write(WORK / "report.json", out)
    return out


def _geometry(tb_atoms, pbe_atoms) -> dict:
    from ..sites import bond_graph

    bonds = sorted(bond_graph(pbe_atoms))
    d = np.array([tb_atoms.get_distance(a, b, mic=True) - pbe_atoms.get_distance(a, b, mic=True)
                  for a, b in bonds])
    symbols = pbe_atoms.get_chemical_symbols()
    hetero = [k for k, (a, b) in enumerate(bonds) if symbols[a] != "C" or symbols[b] != "C"]
    return {"bond_rms_A": float(np.sqrt(np.mean(d ** 2))), "bond_mean_A": float(d.mean()),
            "bond_max_abs_A": float(np.abs(d).max()),
            "hetero_bonds": [{"atoms": list(bonds[k]), "elements": [symbols[i] for i in bonds[k]],
                              "tb_minus_pbe_A": float(d[k])} for k in hetero]}


#: Adjustable with --ajuste NOMBRE=VALOR (tbkit recetas doped_gpaw).
PARAMS = [
    Param("GPAW_SETTINGS", "ajustes de GPAW: modo, base, funcional, malla h (Å)", "",
          "LCAO dzp PBE h 0.2: la referencia de todo tbkit; sz no sirve (es la base mínima "
          "de tbkit y reproduce sus errores)", "cálculo"),
    Param("KZ", "puntos k a lo largo del eje", "",
          "2: PBE tiene gap de 0.22 eV y las fuerzas con 2 y 4 k difieren 6e-4 eV/Å", "convergencia"),
    Param("KT", "ensanchamiento de Fermi-Dirac", "eV", "", "convergencia"),
    Param("VACUUM", "vacío transversal", "Å", "", "convergencia"),
    Param("MAX_DISP", "desplazamiento del átomo que más se mueve a lo largo de cada modo",
          "Å", "", "convergencia"),
    Param("FMAX", "fuerza máxima al relajar con GPAW", "eV/Å", "", "convergencia"),
    Param("MAX_STEPS", "pasos máximos de la relajación GPAW", "", "", "convergencia"),
    Param("PRISTINE_PBE", "geometría PBE de la coil sin dopar", "", "", "rutas"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("select", "relax", "modes", "report"))
    parser.add_argument("name", nargs="?", choices=NAMES)
    parser.add_argument("--part", default="0/1")
    parser.add_argument("--job", type=int, default=None, help="modes: solo ese cálculo")
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    part = tuple(int(x) for x in args.part.split("/"))
    if args.step == "select":
        for row in select(args.name):
            print(f"{row['tb_cm1']:8.1f}  {','.join(row['why'])}")
    elif args.step == "relax":
        relaxed(args.name)
    elif args.step == "modes":
        modes(args.name, part, args.job)
    else:
        print(json.dumps({k: v.get("pbe_over_tb_mean") for k, v in report().items()
                          if isinstance(v, dict)}))


if __name__ == "__main__":
    main()
