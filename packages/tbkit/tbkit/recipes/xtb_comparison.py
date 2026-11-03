"""tbkit against GFN1-xTB and GFN2-xTB (tblite) on the same GPAW references.

Run from packages/tbkit (needs ``tblite``, optional: ``uv pip install tblite``)::

    python -m tbkit.recipes.xtb_comparison molecules     # forces, relative energies, gaps
    python -m tbkit.recipes.xtb_comparison frequencies   # relax + harmonic frequencies
    python -m tbkit.recipes.xtb_comparison crystals      # periodic: doped sheets, tubes, solids
    python -m tbkit.recipes.xtb_comparison layers        # bilayer/graphite binding vs PBE+D3
    python -m tbkit.recipes.xtb_comparison spin          # triplet/singlet, radicals
    python -m tbkit.recipes.xtb_comparison timing        # wall time on growing flakes
    python -m tbkit.recipes.xtb_comparison report        # validation/xtb_vs_tbkit.json

Every stage caches its result in ``out/xtb/<stage>.json`` (crystals and layers also one file
per crystal or point, so a restart resumes); ``report`` gathers them.

What is compared, and what it means. The references are GPAW PBE (LCAO dzp for molecules
and crystals, plane waves for layers and spin). tbkit's sets were fitted to PBE; GFN1 and
GFN2 were fitted to hybrid-DFT and coupled-cluster data and carry their own dispersion
(D3 and D4), so a difference from PBE is partly PBE's own error. Where PBE is known to be
the weaker reference (dispersion, singlet-triplet gaps) the comparison says so. The point
is the cost-accuracy position of a parametrised TB fitted to one's own DFT against the
standard general-purpose TB, on the systems this package is for.

xTB in tblite samples only Γ: periodic structures are repeated until every periodic axis
is at least ``XTB_MIN_LENGTH`` Å (and a multiple of 3 cells for hexagonal sheets, so K
folds to Γ); forces are read on the atoms of the first copy. tbkit uses each crystal's own
k mesh. Electronic temperature: 0.02 eV in tbkit, 300 K in xTB (its default).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms

from ..settings import Param, add_arguments, apply

WORK = Path("out/xtb")
OUT = Path("validation/xtb_vs_tbkit.json")
REFERENCES = Path(__file__).resolve().parents[1] / "parameters" / "references"
MOLECULE_FILES = {"gpaw_chn.json": "xu_chn", "gpaw_chno.json": "xu_chno",
                  "gpaw_b.json": "xu_chnob", "gpaw_p.json": "xu_chnop",
                  "gpaw_s.json": "xu_chnos", "gpaw_se.json": "xu_chnose"}
HESSIAN_FILES = {"gpaw_chno_hessians.json": "xu_chno", "gpaw_b_hessians.json": "xu_chnob",
                 "gpaw_p_hessians.json": "xu_chnop", "gpaw_s_hessians.json": "xu_chnos",
                 "gpaw_se_hessians.json": "xu_chnose"}
METHODS = ("GFN1-xTB", "GFN2-xTB")
KT_TB = 0.02
XTB_MIN_LENGTH = 14.0
FMAX = 1e-3
DELTA = 0.01
HARTREE, BOHR = 27.211386245988, 0.52917721067


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def _write(name: str, data) -> Path:
    path = WORK / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)
    return path


# ---------------------------------------------------------------- calculators
def xtb_calculator(method: str, **kw):
    from tblite.ase import TBLite

    return TBLite(method=method, verbosity=0, **kw)


def tb_calculator(set_name: str, kpts=None):
    from ..calculator import TBCalculator
    from ..params import load_parameters

    return TBCalculator(load_parameters(set_name), kpts=tuple(kpts) if kpts else None, kT=KT_TB)


def xtb_singlepoint(method: str, atoms: Atoms, charge: float = 0.0, uhf: int = 0,
                    spin_polarization: bool = False):
    """Energy (eV), forces (eV/Å), orbital energies (eV) and occupations of a molecule.
    ``spin_polarization``: tblite's spin-polarised xTB (spGFN); without it GFN is
    spin-restricted and a triplet differs from a singlet only by its occupations."""
    from tblite.interface import Calculator

    calc = Calculator(method, atoms.numbers, atoms.positions / BOHR, charge=charge, uhf=uhf)
    calc.set("verbosity", 0)
    if spin_polarization:
        calc.add("spin-polarization", 1.0)
    res = calc.singlepoint()
    return (float(res.get("energy")) * HARTREE, -np.asarray(res.get("gradient")) * HARTREE / BOHR,
            np.asarray(res.get("orbital-energies")) * HARTREE,
            np.asarray(res.get("orbital-occupations")))


def tb_singlepoint(set_name: str, atoms: Atoms):
    """Energy, forces and the HOMO-LUMO gap of a molecule with a tbkit set."""
    from ..forces import energy_and_forces
    from ..hamiltonian import System
    from ..params import load_parameters
    from ..scc import energy_and_forces as scc_energy, self_consistent
    from ..solver import solve

    model = load_parameters(set_name)
    system = System.build(atoms, model)
    if model.scc:
        result = self_consistent(system, kT=KT_TB, tol=1e-10)
        energy, forces, _ = scc_energy(result)
        solution = result.solution
    else:
        solution = solve(system, kT=KT_TB)
        energy, forces, _ = energy_and_forces(solution)
    homo, lumo = solution.homo_lumo()
    return float(energy), forces, float(lumo - homo)


def _xtb_gap(levels, occupations) -> float:
    occupied = levels[occupations > 1.0]
    empty = levels[occupations < 1.0]
    return float(empty.min() - occupied.max())


# ---------------------------------------------------------------- molecules
def molecules() -> dict:
    from ..references import load_references

    out = {}
    for filename, set_name in MOLECULE_FILES.items():
        refs, _ = load_references(REFERENCES / filename)
        rows = []
        for ref in refs:
            row = {"label": ref.label, "group": ref.group, "kind": ref.label.split("/")[1][:3],
                   "energy_dft": ref.energy, "force_rms_dft": _rms(ref.forces)}
            e, f, gap = tb_singlepoint(set_name, ref.atoms)
            row["tb"] = {"energy": e, "force_rmse": _rms(f - ref.forces), "gap": gap}
            for method in METHODS:
                e, f, levels, occ = xtb_singlepoint(method, ref.atoms)
                row[method] = {"energy": e, "force_rmse": _rms(f - ref.forces),
                               "gap": _xtb_gap(levels, occ)}
            if ref.levels is not None and len(ref.levels) > ref.n_occupied:
                row["gap_dft"] = float(ref.levels[ref.n_occupied] - ref.levels[ref.n_occupied - 1])
            rows.append(row)
        out[filename] = {"set": set_name, "rows": rows, "summary": _summarise(rows)}
        print(filename, json.dumps(out[filename]["summary"]), flush=True)
    _write("molecules", out)
    return out


def _summarise(rows: list[dict]) -> dict:
    codes = ("tb",) + METHODS
    summary = {}
    eq = {r["group"]: r for r in rows if r["kind"] == "eq"}
    for code in codes:
        s = {}
        for kind in ("eq", "rnd"):
            chosen = [r[code]["force_rmse"] for r in rows if r["kind"] == kind]
            if chosen:
                s[f"force_rmse_{kind}_median"] = float(np.median(chosen))
        errors = []
        for r in rows:
            base = eq.get(r["group"])
            if base is None or r is base:
                continue
            errors.append((r[code]["energy"] - base[code]["energy"])
                          - (r["energy_dft"] - base["energy_dft"]))
        s["relative_energy_mae"] = float(np.mean(np.abs(errors))) if errors else None
        gaps = [r[code]["gap"] - r["gap_dft"] for r in rows if r["kind"] == "eq" and "gap_dft" in r]
        s["gap_mae"] = float(np.mean(np.abs(gaps))) if gaps else None
        s["gap_mean_signed"] = float(np.mean(gaps)) if gaps else None
        summary[code] = s
    return summary


# ---------------------------------------------------------------- frequencies
def _frequencies(atoms: Atoms, calc) -> tuple[np.ndarray, float, Atoms]:
    import tempfile

    from ase.optimize import BFGS
    from ase.vibrations import Vibrations

    atoms = atoms.copy()
    atoms.calc = calc
    start = atoms.positions.copy()
    BFGS(atoms, logfile=None).run(fmax=FMAX, steps=500)
    drift = float(np.sqrt(np.mean(np.sum((atoms.positions - start) ** 2, axis=1))))
    with tempfile.TemporaryDirectory() as tmp:
        vib = Vibrations(atoms, delta=DELTA, name=str(Path(tmp) / "vib"))
        vib.run()
        energies = vib.get_energies()
    freqs = np.real(energies) * 8065.544 - np.imag(energies) * 8065.544
    return np.sort(freqs), drift, atoms


def _internal(freqs: np.ndarray, atoms: Atoms) -> np.ndarray:
    n = 3 * len(atoms) - (5 if _linear(atoms) else 6)
    return np.sort(freqs)[-n:]


def _linear(atoms: Atoms) -> bool:
    if len(atoms) < 3:
        return True
    x = atoms.positions - atoms.positions.mean(axis=0)
    return np.linalg.svd(x, compute_uv=False)[1] < 1e-3


def _frequency_scale(set_name: str) -> float:
    """The set's own scale factor against GPAW (``frequency_scale``, recipe
    ``frequency_scaling``), reported beside the raw numbers, never in their place."""
    from ..params import read_parameter_file

    entry = read_parameter_file(set_name).get("frequency_scale")
    return float(entry["value"]) if entry else 1.0


def frequencies() -> dict:
    from ..references import load_references

    out = {}
    jobs = []
    for filename, set_name in HESSIAN_FILES.items():
        data = json.loads((REFERENCES / filename).read_text())
        for name, entry in data["molecules"].items():
            atoms = Atoms(entry["symbols"], positions=entry["positions"])
            jobs.append((filename, set_name, name, atoms, np.asarray(entry["frequencies"])))
    chn = json.loads((REFERENCES / "gpaw_chn_frequencies.json").read_text())["frequencies"]
    refs, _ = load_references(REFERENCES / "gpaw_chn.json")
    eq = {r.group: r for r in refs if r.label.endswith("/eq")}
    for name, freqs in chn.items():
        if name in eq:
            jobs.append(("gpaw_chn_frequencies.json", "xu_chn", name, eq[name].atoms,
                         np.asarray(freqs)))
    for filename, set_name, name, atoms, ref in jobs:
        ref_int = _internal(ref, atoms)
        row = {"set": set_name, "gpaw": ref_int.tolist()}
        for code in ("tb",) + METHODS:
            calc = tb_calculator(set_name) if code == "tb" else xtb_calculator(code)
            try:
                f, drift, _ = _frequencies(atoms, calc)
                mine = _internal(f, atoms)
                row[code] = {"rms_cm1": _rms(mine - ref_int), "drift_A": drift,
                             "mean_signed_cm1": float(np.mean(mine - ref_int)),
                             "frequencies": mine.tolist()}
                if code == "tb":
                    scale = _frequency_scale(set_name)
                    row["tb_scaled"] = {"rms_cm1": _rms(scale * mine - ref_int), "scale": scale,
                                        "mean_signed_cm1": float(np.mean(scale * mine - ref_int))}
            except Exception as error:          # a method that cannot treat the molecule
                row[code] = {"error": str(error)[:200]}
        out.setdefault(filename, {})[name] = row
        print(filename, name, {c: round(row[c].get("rms_cm1", -1), 1) for c in ("tb",) + METHODS},
              flush=True)
    summary = {}
    for code in ("tb", "tb_scaled") + METHODS:
        values = [r[code]["rms_cm1"] for f in out.values() for r in f.values()
                  if code in r and "rms_cm1" in r[code]]
        summary[code] = {"median_rms_cm1": float(np.median(values)), "molecules": len(values),
                         "mean_rms_cm1": float(np.mean(values))}
    result = {"molecules": out, "summary": summary}
    _write("frequencies", result)
    return result


# ---------------------------------------------------------------- crystals
def _repeats(atoms: Atoms) -> tuple[int, int, int]:
    reps = []
    lengths = atoms.cell.lengths()
    hexagonal = abs(atoms.cell.angles()[2] - 120) < 1 or abs(atoms.cell.angles()[2] - 60) < 1
    for axis in range(3):
        if not atoms.pbc[axis]:
            reps.append(1)
            continue
        n = int(np.ceil(XTB_MIN_LENGTH / lengths[axis]))
        if hexagonal and axis < 2:
            n = 3 * int(np.ceil(n / 3))
        reps.append(max(n, 1))
    return tuple(reps)


def crystals() -> dict:
    from ..references import load_references
    from .crystal_validation import CRYSTALS

    refs, _ = load_references(REFERENCES / "gpaw_crystals.json")
    out = {}
    for name, (_, set_name, kpts, _) in CRYSTALS.items():
        group = [r for r in refs if r.group == name]
        if not group:
            continue
        cached = WORK / "crystals" / f"{name}.json"          # resumable: one file per crystal
        if cached.exists():
            out[name] = json.loads(cached.read_text())
            continue
        base = next(r for r in group if r.label.endswith("/relaxed"))
        rows = []
        energies = {}
        for ref in group:
            row = {"label": ref.label, "force_rms_dft": _rms(ref.forces)}
            atoms = ref.atoms.copy()
            atoms.calc = tb_calculator(set_name, kpts)
            e = float(atoms.get_potential_energy())
            row["tb"] = {"force_rmse": _rms(atoms.get_forces() - ref.forces), "energy": e}
            reps = _repeats(base.atoms)        # the relaxed cell's: a strained point that
            big = ref.atoms.repeat(reps)       # dips under the length must not grow (x27 once)
            for method in METHODS:
                try:
                    big.calc = xtb_calculator(method)
                    forces = big.get_forces()[: len(ref.atoms)]
                    e = float(big.get_potential_energy()) / np.prod(reps)
                    row[method] = {"force_rmse": _rms(forces - ref.forces), "energy": e}
                except Exception as error:
                    row[method] = {"error": str(error)[:200]}
            row["xtb_repeats"] = list(reps)
            energies[ref.label] = ref.energy
            rows.append(row)
            print(name, ref.label, {c: round(row[c].get("force_rmse", -1), 3)
                                    for c in ("tb",) + METHODS}, flush=True)
        summary = {}
        for code in ("tb",) + METHODS:
            ok = [r for r in rows if "force_rmse" in r[code]]
            b = next((r for r in ok if r["label"] == base.label), None)
            dist = [r for r in ok if "/rnd" in r["label"]]
            summary[code] = {
                "force_rmse_median": float(np.median([r[code]["force_rmse"] for r in ok]))
                if ok else None,
                "energy_mae_distorted": float(np.mean([abs((r[code]["energy"] - b[code]["energy"])
                                                           - (energies[r["label"]] - base.energy))
                                                       for r in dist])) if b and dist else None}
        out[name] = {"set": set_name, "atoms": len(base.atoms), "rows": rows, "summary": summary}
        _write(f"crystals/{name}", out[name])
    _write("crystals", out)
    return out


# ---------------------------------------------------------------- layers
def layers() -> dict:
    from . import interlayer as il

    names = ("mono",) + tuple(n for n in il.POINTS if n != "mono")
    energies = {}
    for name in names:
        try:
            ref = il.load(name)
        except FileNotFoundError:
            continue
        cached = WORK / "layers" / f"{il._slug(name)}.json"   # resumable: one file per point
        if cached.exists():
            energies[name] = json.loads(cached.read_text())
            continue
        atoms = il.geometry(name)
        atoms.pbc = True          # slabs: the same 25 Å cell as GPAW
        reps = (6, 6, 1)
        big = atoms.repeat(reps)
        row = {"n_atoms": ref["n_atoms"]}
        for method in METHODS:
            big.calc = xtb_calculator(method)
            row[method] = float(big.get_potential_energy()) / np.prod(reps)
        energies[name] = row
        _write(f"layers/{il._slug(name)}", row)
        print(name, row, flush=True)
    out = {}
    mono = energies["mono"]
    tb = json.loads(Path("validation/interlayer_tb_vs_gpaw.json").read_text())["binding"] \
        if Path("validation/interlayer_tb_vs_gpaw.json").exists() else {}
    for name, row in energies.items():
        if name == "mono":
            continue
        ref = il.binding(name)
        n = row["n_atoms"]
        entry = {"pbe": ref["pbe"], "pbe_d3": ref["pbe_d3"]}
        for method in METHODS:
            entry[method] = 1000 * (row[method] - n / 2 * mono[method]) / n
        if name in tb:
            entry["xu_capas"] = tb[name]["xu_capas_meV"]
        out[name] = entry
    _write("layers", out)
    return out


# ---------------------------------------------------------------- spin
SPIN_SYSTEMS = {"O2": 2, "CH2": 2, "triangulene": 2}


def spin() -> dict:
    from . import spin_validation as sv

    out = {}
    gpaw = {}
    for name, unpaired in SPIN_SYSTEMS.items():
        atoms = sv.geometry(name)
        high, low = ("T", "S") if name in ("O2", "CH2") else ("T", "NM")
        try:
            e_high = json.loads(sv._slug(name, high).read_text())["energy_eV"]
            e_low = json.loads(sv._slug(name, low).read_text())["energy_eV"]
            gpaw[name] = 1000 * (e_low - e_high)
        except FileNotFoundError:
            gpaw[name] = None
        row = {"gpaw_meV": gpaw[name]}
        for method in METHODS:
            high_e = xtb_singlepoint(method, atoms, uhf=unpaired, spin_polarization=True)[0]
            low_e = xtb_singlepoint(method, atoms, uhf=0, spin_polarization=True)[0]
            row[method] = 1000 * (low_e - high_e)
        tb = json.loads(Path("validation/spin_tb_vs_gpaw.json").read_text())["systems"].get(name) \
            if Path("validation/spin_tb_vs_gpaw.json").exists() else None
        if tb:
            for code in ("tb", "tb_scaled"):
                row[code] = 1000 * tb["states"][low][code]["relative_eV"]
        out[name] = row
        print(name, row, flush=True)
    note = ("E(M = 0) − E(M = 2), meV. xTB con polarización de espín (spGFN de tblite), uhf = 0 "
            "frente a uhf = 2; tbkit: SCC con espín (W del átomo y W·λ); GPAW: PBE con espín, "
            "M fija. Sin la polarización de espín GFN no separa triplete de singlete (≈ −70 meV "
            "en los tres sistemas). Fenalenilo fuera: su estado sin espín de GPAW (SOMO "
            "semiocupado) no tiene análogo directo en xTB.")
    result = {"note": note, "systems": out}
    _write("spin", result)
    return result


# ---------------------------------------------------------------- timing
def timing() -> dict:
    from .spin_validation import _flake, _saturate

    dx, dy = np.sqrt(3) * 1.42, 1.5 * 1.42
    out = []
    for size in (3, 5, 8, 11):
        centres = [(c * dx + r * dx / 2, r * dy, 0) for r in range(size) for c in range(size - r)]
        atoms = _saturate(_flake(centres))
        row = {"atoms": len(atoms), "formula": atoms.get_chemical_formula()}
        for code in ("tb",) + METHODS:
            a = atoms.copy()
            if code == "tb":
                from ..calculator import TBCalculator
                from ..params import load_parameters

                # kT 0.1 eV: the zigzag triangles have many near-zero levels and the SCC of
                # the larger ones does not settle at 0.02 eV (what is timed is the cost)
                a.calc = TBCalculator(load_parameters("xu_chn"), kT=0.1)
            else:
                a.calc = xtb_calculator(code)
            start = time.perf_counter()
            a.get_forces()
            row[code] = time.perf_counter() - start
        out.append(row)
        print(row, flush=True)
    result = {"note": "segundos de reloj por energía + fuerzas de un flake triangular "
                      "H-saturado (OMP_NUM_THREADS=2), tbkit xu_chn con SCC (kT 0,1 eV) "
                      "frente a tblite (300 K)", "rows": out}
    _write("timing", result)
    return result


# ---------------------------------------------------------------- report
STAGES = ("molecules", "frequencies", "crystals", "layers", "spin", "timing")


def report() -> dict:
    out = {"description": "tbkit (sus conjuntos ajustados a PBE) frente a GFN1-xTB y GFN2-xTB "
                          "(tblite) sobre las mismas referencias GPAW PBE. xTB no se ajustó a "
                          "PBE: parte de su diferencia es el error del propio PBE."}
    for stage in STAGES:
        path = WORK / f"{stage}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if stage == "molecules":
            data = {f: v["summary"] | {"set": v["set"], "structures": len(v["rows"])}
                    for f, v in data.items()}
        elif stage == "frequencies":
            data = {"summary": data["summary"],
                    "per_file": {f: {m: {c: r[c].get("rms_cm1")
                                         for c in ("tb", "tb_scaled") + METHODS if c in r}
                                     for m, r in mols.items()}
                                 for f, mols in data["molecules"].items()}}
        elif stage == "crystals":
            data = {c: v["summary"] | {"set": v["set"], "atoms": v["atoms"]}
                    for c, v in data.items()}
        out[stage] = data
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


PARAMS = [
    Param("KT_TB", "ensanchamiento de Fermi-Dirac de tbkit", "eV", "", "cálculo"),
    Param("XTB_MIN_LENGTH", "longitud mínima de cada eje periódico en xTB (solo Γ)", "Å", "",
          "convergencia"),
    Param("FMAX", "fuerza máxima de la relajación antes de las frecuencias", "eV/Å", "",
          "convergencia"),
    Param("DELTA", "desplazamiento de las diferencias finitas de las frecuencias", "Å", "",
          "convergencia"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES + ("report",))
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    result = globals()[args.stage]()
    if args.stage == "report":
        print(json.dumps(result, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
