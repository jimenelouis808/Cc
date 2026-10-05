"""Nanocoil phase 2: the periodic 204-atom knee coil with TB (Xu, Tang) and GPAW prepared.

Run (each stage resumable: every result is a file in WORKDIR)::

    python -m tbkit.recipes.nanocoil tb WORKDIR --model tang_carbon   # relax, topology, gap, Γ modes
    python -m tbkit.recipes.nanocoil tb WORKDIR --model xu_carbon
    python -m tbkit.recipes.nanocoil gpaw-k WORKDIR                   # k convergence (GPAW)
    python -m tbkit.recipes.nanocoil gpaw-relax WORKDIR               # PBE relaxation (GPAW)
    python -m tbkit.recipes.nanocoil gpaw-phase3 WORKDIR              # PBE gap, geometry, projected modes
    python -m tbkit.recipes.nanocoil raman WORKDIR [--part r/n]       # resonant Raman, every mode

Input: ``recipes/data/coil204_knee.extxyz``, built by nanocarbon_lab
(``build_knee_periodic_coil(coil_radius=7.75, pitch=15.0, sides_per_turn=6,
circumference=6, relax=True)``, preset "Nanocoil (knees, periodic, smallest)":
{5: 12, 6: 78, 7: 12}, periodic along z, Lz = 15 Å) and exchanged as a file
(the packages do not import each other). See docs/PLAN_NANOCOIL_RAMAN.md.

Checks that decide whether a result is kept (docs/PLAN, phase 2):
* topology: the bond graph (C-C < 1.85 Å) after relaxation must equal the
  input's; any bond made or broken is a TOPOLOGY_WARNING, never corrected;
* Γ modes: the three acoustic ones near zero, imaginary modes counted;
* Raman (non-resonant) only if the relaxed coil has a gap; a gapless coil
  is recorded as such (its Raman is resonant, a different calculation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

from .. import sites

INPUT = Path(__file__).with_name("data") / "coil204_knee.extxyz"
KMESH = (1, 1, 4)
KT = 0.05
BOND = 1.85


def coil() -> Atoms:  # noqa: F821
    return read(INPUT)


def bond_graph(atoms) -> set:
    return sites.bond_graph(atoms, BOND)


def rings(atoms, largest: int = 7) -> list[tuple]:
    return sites.rings(atoms, largest, BOND)


def ring_census(atoms) -> dict:
    return sites.ring_census(atoms, cutoff=BOND)


def ring_atoms(atoms, sizes=(5, 7)) -> dict:
    return sites.ring_atoms(atoms, sizes, cutoff=BOND)


def _write(path: Path, data) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def _relax_resumable(atoms, calc, folder: Path, fmax: float, steps: int = 600):
    from ase.io import Trajectory
    from ase.optimize import BFGS

    traj = folder / "relax.traj"
    if traj.exists() and traj.stat().st_size > 0:
        try:
            atoms.positions = read(traj, index=-1).positions
        except Exception:                           # noqa: BLE001 - cut mid-write
            traj.unlink()
    atoms.calc = calc
    with Trajectory(str(traj), "a", atoms) as t:
        opt = BFGS(atoms, restart=str(folder / "relax_bfgs.json"), logfile=str(folder / "relax.log"))
        opt.attach(t.write, interval=1)
        converged = bool(opt.run(fmax=fmax, steps=steps))
    return atoms, converged


def stage_tb(workdir: Path, model_name: str) -> dict:
    from ase.vibrations import Vibrations

    from ..calculator import TBCalculator
    from ..params import load_parameters

    folder = Path(workdir) / f"tb_{model_name}"
    folder.mkdir(parents=True, exist_ok=True)
    model = load_parameters(model_name)
    start = coil()
    relaxed_path = folder / "relaxed.extxyz"
    if relaxed_path.exists():
        atoms = read(relaxed_path)
        converged = json.loads((folder / "relax.json").read_text())["converged"]
    else:
        atoms, converged = _relax_resumable(start.copy(), TBCalculator(model, kpts=KMESH, kT=KT),
                                            folder, fmax=0.02)
        atoms.calc = None
        write(relaxed_path, atoms)
        _write(folder / "relax.json", {"converged": converged})
    made = bond_graph(atoms) - bond_graph(start)
    broken = bond_graph(start) - bond_graph(atoms)
    calc = TBCalculator(model, kpts=KMESH, kT=KT)
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    gap = float(calc.last_solution.gap())
    bonds = np.array([atoms.get_distance(a, b, mic=True) for a, b in bond_graph(atoms)])
    report = {"model": model_name, "atoms": len(atoms), "relax_converged": converged,
              "energy_eV": energy, "gap_eV": gap,
              "bond_min_max": [float(bonds.min()), float(bonds.max())],
              "ring_census": ring_census(atoms),
              "topology": "OK" if not made and not broken else "TOPOLOGY_WARNING",
              "bonds_made": sorted(made), "bonds_broken": sorted(broken)}
    _write(folder / "report.json", report)
    # Γ modes: ase.vibrations caches every displacement in its directory (resumable)
    vib = Vibrations(atoms, name=str(folder / "vib"), delta=0.01)
    vib.run()
    data = vib.get_vibrations()
    energies, modes = data.get_energies_and_modes(all_atoms=True)
    from ase.units import invcm

    freq = np.where(np.abs(energies.imag) > np.abs(energies.real),
                    -np.abs(energies.imag), np.abs(energies.real)) / invcm
    rings = ring_atoms(atoms)
    on5, on7 = np.array(rings[5]), np.array(rings[7])
    weight = np.sum(np.abs(modes) ** 2, axis=2)                 # (modes, atoms)
    weight /= weight.sum(axis=1, keepdims=True)
    participation = 1 / np.sum(weight ** 2, axis=1) / len(atoms)
    table = [{"frequency_cm1": float(f), "participation": float(p),
              "on_pentagons": float(w[on5].sum()), "on_heptagons": float(w[on7].sum())}
             for f, p, w in zip(freq, participation, weight, strict=True)]
    np.savez(folder / "modes.npz", frequencies=freq, modes=modes)
    report.update({"modes": len(freq), "imaginary_below_-20": int(np.sum(freq < -20)),
                   "lowest_three": [float(v) for v in np.sort(np.abs(freq))[:3]],
                   "ring_atoms": {5: len(on5), 7: len(on7)},
                   "share_of_atoms_on_5_and_7": [len(on5) / len(atoms), len(on7) / len(atoms)],
                   "raman": ("non-resonant applicable" if gap > 0.1 else
                             "gapless: non-resonant Raman does not apply (resonant only)")})
    _write(folder / "report.json", report)
    _write(folder / "modes.json", table)
    return report


def _trimmed(atoms, vacuum: float):
    """Transverse cell = coil extent + 2 vacuum (z untouched)."""
    out = atoms.copy()
    xy = out.positions[:, :2]
    span = xy.max(axis=0) - xy.min(axis=0)
    cell = out.cell.array.copy()
    cell[0, 0], cell[1, 1] = span + 2 * vacuum
    out.set_cell(cell, scale_atoms=False)
    out.positions[:, :2] += -xy.min(axis=0) + vacuum
    return out


def _gpaw(kpts):
    from gpaw import GPAW, FermiDirac

    return GPAW(mode="lcao", basis="dzp", xc="PBE", h=0.2, kpts=kpts, symmetry="off",
                occupations=FermiDirac(KT), txt=None)


def stage_gpaw_k(workdir: Path, vacuum: float = 6.0) -> dict:
    """Single points at the Tang-relaxed geometry: Γ, 1x1x2, 1x1x4 (energy, forces)."""
    folder = Path(workdir) / "gpaw_k"
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(workdir) / "tb_tang_carbon" / "relaxed.extxyz"
    base = _trimmed(read(source) if source.exists() else coil(), vacuum)
    write(folder / "geometry.extxyz", base)
    out = {}
    for kz in (1, 2, 4):
        path = folder / f"k{kz}.json"
        if not path.exists():
            atoms = base.copy()
            atoms.calc = _gpaw((1, 1, kz))
            _write(path, {"kz": kz, "energy": float(atoms.get_potential_energy()),
                          "forces": atoms.get_forces().tolist()})
        out[kz] = json.loads(path.read_text())
    e4 = out[4]["energy"]
    f4 = np.array(out[4]["forces"])
    summary = {k: {"dE_per_atom_vs_k4": (v["energy"] - e4) / len(base),
                   "dF_rms_vs_k4": float(np.sqrt(np.mean((np.array(v["forces"]) - f4) ** 2)))}
               for k, v in out.items()}
    _write(folder / "summary.json", summary)
    return summary


def stage_gpaw_relax(workdir: Path, kz: int = 2, vacuum: float = 6.0) -> dict:
    folder = Path(workdir) / "gpaw_relax"
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(workdir) / "tb_tang_carbon" / "relaxed.extxyz"
    atoms = _trimmed(read(source) if source.exists() else coil(), vacuum)
    atoms, converged = _relax_resumable(atoms, _gpaw((1, 1, kz)), folder, fmax=0.05, steps=300)
    atoms.calc = None
    write(folder / "relaxed.extxyz", atoms)
    report = {"converged": converged, "kz": kz, "vacuum": vacuum,
              "topology": "OK" if bond_graph(atoms) == bond_graph(coil()) else "TOPOLOGY_WARNING"}
    _write(folder / "report.json", report)
    return report


def _bond_lengths(atoms, bonds) -> np.ndarray:
    return np.array([atoms.get_distance(a, b, mic=True) for a, b in bonds])


def compare_geometry(reference, other) -> dict:
    """Bond by bond on the input's bond graph, split by the rings a bond belongs to."""
    bonds = sorted(bond_graph(coil()))
    d = _bond_lengths(other, bonds) - _bond_lengths(reference, bonds)
    faces = rings(reference)
    kind = {}
    for ring in faces:
        members = set(ring)
        for b in bonds:
            if b[0] in members and b[1] in members:
                kind.setdefault(b, set()).add(len(ring))
    out = {"bond_rms_A": float(np.sqrt(np.mean(d ** 2))), "bond_max_abs_A": float(np.abs(d).max()),
           "bond_mean_A": float(d.mean())}
    for size in (5, 6, 7):
        sel = np.array([size in kind.get(b, ()) for b in bonds])
        out[f"bond_mean_in_{size}_rings_A"] = float(d[sel].mean())
    return out


def selected_modes(workdir: Path) -> list[dict]:
    """Per TB set: the mode most on pentagons and the highest mode (on heptagons)."""
    picks = []
    for model in ("xu_carbon", "tang_carbon"):
        folder = Path(workdir) / f"tb_{model}"
        table = json.loads((folder / "modes.json").read_text())
        data = np.load(folder / "modes.npz")
        for label, index in (("most_pentagon", int(np.argmax([m["on_pentagons"] for m in table]))),
                             ("highest", int(np.argmax(data["frequencies"])))):
            picks.append({"model": model, "label": label, "index": index,
                          "tb_frequency_cm1": float(data["frequencies"][index]),
                          "vector": data["modes"][index].reshape(-1)})
    return picks


def _projected_frequency(f0, f1, vector, step: float) -> float:
    """ω from k = -v·(F(x0 + s v) - F(x0))/s with |v| = 1 (all atoms carbon)."""
    from ase.data import atomic_masses
    from ase.units import _amu, _e, _hbar, invcm

    k = -float(vector @ (f1 - f0).ravel()) / step
    scale = _hbar * 1e10 / np.sqrt(_e * _amu)               # eV for k in eV/Å², m in amu
    omega = scale * np.sqrt(abs(k) / atomic_masses[6]) / invcm
    return float(np.sign(k) * omega)


def stage_gpaw_phase3(workdir: Path, kz: int = 2, max_disp: float = 0.02) -> dict:
    """PBE at the PBE geometry: gap (SCF + Γ–Z bands), geometry vs TB, projected frequencies."""
    workdir = Path(workdir)
    folder = workdir / "gpaw_phase3"
    folder.mkdir(parents=True, exist_ok=True)
    pbe = read(workdir / "gpaw_relax" / "relaxed.extxyz")
    report = {"geometry_vs_pbe": {m: compare_geometry(pbe, read(workdir / f"tb_{m}" / "relaxed.extxyz"))
                                  for m in ("xu_carbon", "tang_carbon")}}
    ground = folder / "ground.json"
    if not ground.exists():
        atoms = pbe.copy()
        atoms.calc = _gpaw((1, 1, kz))
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        calc = atoms.calc
        fermi = float(calc.get_fermi_level())
        path = np.linspace(0, 0.5, 11)
        bands = calc.fixed_density(kpts=[(0, 0, kzp) for kzp in path], symmetry="off", txt=None)
        eps = np.array([bands.get_eigenvalues(kpt=i) for i in range(len(path))]) - fermi
        gap = float(max(0.0, eps[eps > 0].min() - eps[eps <= 0].max()))
        _write(ground, {"energy": energy, "forces": forces.tolist(), "fermi_eV": fermi,
                        "kz_path": path.tolist(), "bands_near_fermi_eV":
                        [sorted(row[np.abs(row) < 1.0].tolist()) for row in eps],
                        "gap_eV": gap})
    g = json.loads(ground.read_text())
    report["pbe_gap_eV"] = g["gap_eV"]
    report["pbe_bands_within_1eV_of_fermi"] = g["bands_near_fermi_eV"]
    f0 = np.array(g["forces"])
    modes = []
    for pick in selected_modes(workdir):
        vector = pick.pop("vector")
        vector = vector / np.linalg.norm(vector)
        step = max_disp / np.abs(vector.reshape(-1, 3)).sum(axis=1).max()
        path = folder / f"{pick['model']}_{pick['label']}.json"
        if not path.exists():
            atoms = pbe.copy()
            atoms.positions += step * vector.reshape(-1, 3)
            atoms.calc = _gpaw((1, 1, kz))
            _write(path, {"step": step, "forces": atoms.get_forces().tolist()})
        disp = json.loads(path.read_text())
        pick["pbe_projected_cm1"] = _projected_frequency(f0, np.array(disp["forces"]), vector, disp["step"])
        modes.append(pick)
    report["projected_modes"] = modes
    _write(folder / "report.json", report)
    return report


LASERS_EV = (1.58, 1.96, 2.33, 2.41, 2.54)          # 785, 633, 532, 514, 488 nm


def tb_vibrations(workdir: Path, model_name: str):
    """The Γ modes of stage ``tb`` as a tbkit Vibrations (L = e/√m), from ase's cache."""
    from ase.vibrations import Vibrations as AseVibrations

    from ..modes import Vibrations

    folder = Path(workdir) / f"tb_{model_name}"
    atoms = read(folder / "relaxed.extxyz")
    hessian = AseVibrations(atoms, name=str(folder / "vib"), delta=0.01).get_vibrations() \
        .get_hessian_2d()
    return Vibrations.from_hessian(atoms, hessian, source=model_name)


def stage_raman(workdir: Path, model_name: str = "tang_carbon", part: tuple = (0, 1),
                kmesh: int = 4, eta: float = 0.1, delta: float = 0.005) -> dict:
    """Resonant Raman of every internal mode, α differentiated along each mode.

    ``part=(r, n)`` computes the modes with index ≡ r (mod n): n processes share
    the cache folder and the last call (n = 1) assembles. Independent-particle α
    of the crystal (no local fields), interband only; the coil is gapless in TB,
    so every laser is resonant: the activities are not comparable to a
    non-resonant calculation."""
    from ..params import load_parameters
    from ..raman import RamanResult, internal_modes, spectrum
    from ..resonance import resonant_raman

    workdir = Path(workdir)
    folder = workdir / f"raman_{model_name}"
    vib = tb_vibrations(workdir, model_name)
    atoms = vib.atoms
    model = load_parameters(model_name)
    modes = [int(k) for k in internal_modes(atoms, vib.frequencies, [])]
    r, n = part
    chosen = [k for k in modes if k % n == r]
    result = resonant_raman(atoms, model, LASERS_EV, eta=eta, kmesh=kmesh, kT=KT,
                            phonons=(vib.frequencies, vib.modes), select=chosen,
                            cache_dir=folder / "tensors", delta=delta)
    if n > 1:
        return {"part": list(part), "modes": len(chosen)}
    groups = ring_atoms(atoms, (5, 6, 7))
    from ..modes import participation

    shares = participation(vib, {f"en_{s}": groups[s] for s in (5, 6, 7)})
    out = {"model": model_name, "lasers_eV": list(LASERS_EV), "eta_eV": eta, "kmesh": kmesh,
           "delta_A": delta, "modes": len(chosen), "warnings": result.warnings,
           "fraction_of_atoms": {s: len(groups[s]) / len(atoms) for s in (5, 6, 7)}}
    spectra = {}
    for li, laser in enumerate(LASERS_EV):
        at = RamanResult(result.frequencies, result.activities[li], result.depolarization[li],
                         None, None, "", result.frequencies)
        grid, curve = spectrum(at, np.linspace(100, 1900, 3601), fwhm=10.0,
                               laser_nm=1239.84193 / laser)
        spectra[f"{laser:.2f}"] = curve
        order = np.argsort(-result.activities[li])[:15]
        out[f"strongest_{laser:.2f}eV"] = [
            {"mode": int(result.mode_indices[j]), "frequency_cm1": float(result.frequencies[j]),
             "activity": float(result.activities[li, j]),
             "share_of_total": float(result.activities[li, j] / result.activities[li].sum()),
             **{k: float(v[result.mode_indices[j]]) for k, v in shares.items()}}
            for j in order]
    np.savez(folder / "spectra.npz", grid=grid, **spectra, frequencies=result.frequencies,
             activities=result.activities, mode_indices=result.mode_indices)
    _write(folder / "report.json", out)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("tb", "gpaw-k", "gpaw-relax", "gpaw-phase3",
                                                 "raman"))
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--model", default="tang_carbon")
    parser.add_argument("--kz", type=int, default=2)
    parser.add_argument("--part", default="0/1", help="r/n: modos con índice ≡ r (mod n)")
    args = parser.parse_args(argv)
    if args.stage == "tb":
        print(json.dumps({k: v for k, v in stage_tb(args.workdir, args.model).items()
                          if k not in ("bonds_made", "bonds_broken")}, indent=1))
    elif args.stage == "gpaw-k":
        print(json.dumps(stage_gpaw_k(args.workdir), indent=1))
    elif args.stage == "raman":
        r, n = (int(x) for x in args.part.split("/"))
        out = stage_raman(args.workdir, args.model, (r, n))
        print(json.dumps({k: v for k, v in out.items() if not k.startswith("strongest")},
                         indent=1))
    elif args.stage == "gpaw-relax":
        print(json.dumps(stage_gpaw_relax(args.workdir, args.kz), indent=1))
    else:
        print(json.dumps(stage_gpaw_phase3(args.workdir, args.kz), indent=1))


if __name__ == "__main__":
    main()
