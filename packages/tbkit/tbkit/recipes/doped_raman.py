"""Resonant Raman of the 5-7 coil: pristine, graphitic N and -NH2 + H, one model (xu_chn).

Run from packages/tbkit (each step resumable, files in out/doped_raman/<name>/)::

    python -m tbkit.recipes.doped_raman relax   NAME
    python -m tbkit.recipes.doped_raman hessian NAME [--part r/n]
    python -m tbkit.recipes.doped_raman check   NAME      # embedding vs full forces
    python -m tbkit.recipes.doped_raman raman   NAME [--part r/n]
    python -m tbkit.recipes.doped_raman born    NAME [--part r/n]   # IR: Born charges
    python -m tbkit.recipes.doped_raman report

NAME is ``pristine``, ``N`` or ``amine``. All three use ``xu_chn`` with SCC (on the
pristine coil SCC moves forces by up to 0.37 eV/Å against Xu: charge goes between
pentagons and heptagons, so the reference must be computed the same way), kz = 4
(converged for the coil's phonons), kT = 0.05 eV.

The pristine Hessian is computed in full (1224 force calls). For N and the amine
only the rows of atoms within ``RADIUS`` Å of the change are computed and the
pristine Hessian is kept elsewhere (``modes.embedded_hessian``); ``check`` tests
that by the Rayleigh quotient of the highest-weight modes on the change with the
full model. Raman: α(ω_L + iη) with the SCC ground state, differentiated along each
mode above ``MIN_CM1`` (the D/G region and the C-N, N-H stretches; the low modes,
whose heights depend most on η, are left out).

IR: Born charges of the same SCC ground state (``infrared.born_rows``), only the
components across the coil axis (x, y): along the periodic axis the dipole needs
the Berry phase, and the model is gapless there. The amine's are embedded like the
Hessian (checked against the full model along 9 modes: within 5 %); graphitic N's
are computed in full, because its extra electron spreads over the gapless coil and
the embedded ones were off by up to 26 %. The translational sum rule is imposed
and its residual reported. Every mode above
``IR_MIN_CM1``.

D band: the first-order Γ spectrum of this cell has no double resonance, so the D
band is estimated by what it is made of: the Raman intensity of each mode weighted
by its ring-breathing character B (``sites.ring_breathing``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read, write

WORK = Path("out/doped_raman")
KZ, KT, DELTA = 4, 0.05, 0.01
RADIUS = 6.0
MIN_CM1 = 900.0
LASERS_EV = (1.96, 2.33, 2.54)
IR_MIN_CM1 = 50.0
BORN_DELTA = 0.005
STRUCTURES = {
    "pristine": {"start": "out/nanocoil/tb_xu_carbon/relaxed.extxyz", "centres": None},
    # Born charges of N in full: its extra electron spreads over the gapless coil and the
    # embedded ones missed the full-model IR by up to 26 % (out/doped_raman/N/ir_check_*).
    "N": {"start": "out/coil_N/site_0000_relaxed.extxyz", "centres": [0], "born_full": True},
    "amine": {"start": "out/coil_NH2/screen/a048_h059_relaxed.extxyz", "centres": [48, 59]},
}
N_HOST = 204


def _model():
    from ..params import load_parameters

    return load_parameters("xu_chn")


def _calc():
    from ..calculator import TBCalculator

    return TBCalculator(_model(), kpts=(1, 1, KZ), kT=KT)


def folder(name: str) -> Path:
    path = WORK / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def relaxed(name: str):
    path = folder(name) / "relaxed.extxyz"
    if path.exists():
        return read(path)
    from ase.optimize import BFGS

    atoms = read(STRUCTURES[name]["start"])
    atoms.calc = _calc()
    opt = BFGS(atoms, logfile=str(folder(name) / "relax.log"),
               trajectory=str(folder(name) / "relax.traj"))
    converged = bool(opt.run(fmax=0.02, steps=500))
    atoms.calc = None
    write(path, atoms)
    (folder(name) / "relax.json").write_text(json.dumps({"converged": converged,
                                                         "steps": opt.get_number_of_steps()}))
    return atoms


def region(name: str, atoms) -> list[int]:
    spec = STRUCTURES[name]["centres"]
    if spec is None:
        return list(range(len(atoms)))
    centres = list(spec) + list(range(N_HOST, len(atoms)))
    d = atoms.get_all_distances(mic=True)[centres].min(axis=0)
    return sorted(int(i) for i in np.flatnonzero(d <= RADIUS))


def hessian(name: str, part=(0, 1)):
    from ..modes import embedded_hessian, hessian_rows

    atoms = relaxed(name)
    rows = hessian_rows(atoms, _calc, folder(name) / "fd", region(name, atoms), DELTA, part)
    if rows is None:
        return None
    if STRUCTURES[name]["centres"] is None:
        return 0.5 * (rows + rows.T)
    reference = hessian("pristine")
    if reference is None:
        raise RuntimeError("Falta la hessiana de la coil sin dopar.")
    return embedded_hessian(reference, N_HOST, len(atoms), region(name, atoms), rows)


def vibrations(name: str):
    from ..modes import Vibrations

    path = folder(name) / "hessian.npy"
    atoms = relaxed(name)
    if not path.exists():
        h = hessian(name)
        if h is None:
            raise RuntimeError(f"{name}: hessiana incompleta.")
        np.save(path, h)
    return Vibrations.from_hessian(atoms, np.load(path), source=f"xu_chn {name}")


def check(name: str) -> dict:
    """Embedded modes against the full model: the 4 modes with most weight on the
    change, Rayleigh quotient with all forces from xu_chn."""
    from ..modes import participation
    from ..sites import projected_frequency

    vib = vibrations(name)
    reg = region(name, vib.atoms)
    core = [i for i in reg if i >= N_HOST] + list(STRUCTURES[name]["centres"] or [])
    share = participation(vib, {"core": core})["core"]
    picks = [int(k) for k in np.argsort(-share)[:4]]
    out = []
    for k in picks:
        r = projected_frequency(vib.atoms, vib.modes[k], _calc, max_disp=0.01)
        out.append({"mode": k, "embedded_cm1": float(vib.frequencies[k]),
                    "full_cm1": r["frequency_cm1"], "share_on_core": float(share[k])})
    (folder(name) / "check.json").write_text(json.dumps(out, indent=1))
    return out


def raman(name: str, part=(0, 1)):
    from ..resonance import resonant_raman

    vib = vibrations(name)
    chosen = [int(k) for k in np.flatnonzero(vib.frequencies > MIN_CM1)]
    chosen = [k for k in chosen if k % part[1] == part[0]]
    return resonant_raman(vib.atoms, _model(), LASERS_EV, eta=0.1, kmesh=KZ, kT=KT,
                          phonons=(vib.frequencies, vib.modes), select=chosen,
                          cache_dir=folder(name) / "tensors", delta=0.005)


def strongest_modes(name: str, top: int = 40) -> list[int]:
    """The ``top`` modes with the largest activity at any laser (η = 0.1 eV)."""
    result = raman(name)
    best = result.activities.max(axis=0)
    order = np.argsort(-best)[:top]
    return sorted(int(result.mode_indices[k]) for k in order)


def raman_eta(name: str, eta: float, part=(0, 1), top: int = 40):
    """The same resonant Raman with another η, on the ``top`` strongest modes: how much the
    heights depend on the broadening (positions do not)."""
    from ..resonance import resonant_raman

    vib = vibrations(name)
    chosen = [k for k in strongest_modes(name, top) if k % part[1] == part[0]]
    return resonant_raman(vib.atoms, _model(), LASERS_EV, eta=eta, kmesh=KZ, kT=KT,
                          phonons=(vib.frequencies, vib.modes), select=chosen,
                          cache_dir=folder(name) / f"tensors_eta{eta:g}", delta=0.005)


def born_region(name: str, atoms) -> list[int]:
    if STRUCTURES[name].get("born_full"):
        return list(range(len(atoms)))
    return region(name, atoms)


def born(name: str, part=(0, 1)):
    """Born charges (N, 3, 3), transverse columns only; None while incomplete."""
    from ..infrared import born_rows

    atoms = relaxed(name)
    reg = born_region(name, atoms)
    rows = born_rows(atoms, _calc, folder(name) / "born", reg, BORN_DELTA, part)
    if rows is None or len(reg) == len(atoms):
        return rows
    reference = born("pristine")
    if reference is None:
        raise RuntimeError("Faltan las cargas de Born de la coil sin dopar.")
    out = np.zeros((len(atoms), 3, 3))
    out[:N_HOST] = reference
    out[reg] = rows
    return out


def born_ready(name: str) -> bool:
    """Every displacement of ``name`` (and of the pristine host) cached: ``report``
    must never start computing them in one process."""
    names = [name] if STRUCTURES[name]["centres"] is None else ["pristine", name]
    for n in names:
        atoms = relaxed(n)
        if not all((folder(n) / "born" / f"z_{a:04d}_{c}_{t}.npy").exists()
                   for a in born_region(n, atoms) for c in range(3) for t in ("p", "m")):
            return False
    return True


def infrared(name: str) -> dict:
    from ..infrared import mode_intensities

    vib = vibrations(name)
    z = born(name)
    if z is None:
        raise RuntimeError(f"{name}: cargas de Born incompletas.")
    keep = np.flatnonzero(vib.frequencies > IR_MIN_CM1)
    intensities, info = mode_intensities(z, vib.modes[keep])
    return {"frequencies": vib.frequencies[keep], "intensities": intensities,
            "mode_indices": keep, **info}


def report() -> dict:
    from ..modes import participation
    from ..raman import RamanResult, spectrum
    from ..sites import ring_breathing

    out = {"lasers_eV": list(LASERS_EV), "min_cm1": MIN_CM1, "radius_A": RADIUS}
    grid = np.linspace(900, 3700, 2801)
    ir_grid = np.linspace(0, 3700, 3701)
    spectra = {}
    for name in STRUCTURES:
        vib = vibrations(name)
        result = raman(name)
        atoms = vib.atoms
        groups = {"N": [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "N"],
                  "H": [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "H"]}
        groups = {k: v for k, v in groups.items() if v}
        shares = participation(vib, groups) if groups else {}
        entry = {"atoms": len(atoms), "modes_computed": len(result.frequencies)}
        breathing = ring_breathing(atoms, vib.modes[result.mode_indices])["B"]
        entry["breathing_B"] = breathing.tolist()
        entry["raman_frequencies_cm1"] = result.frequencies.tolist()
        for li, laser in enumerate(LASERS_EV):
            r = RamanResult(result.frequencies, result.activities[li], result.depolarization[li],
                            None, None, "", result.frequencies)
            _, curve = spectrum(r, grid, fwhm=10.0, laser_nm=1239.84193 / laser)
            spectra[f"{name}_{laser:.2f}"] = curve
            rb = RamanResult(result.frequencies, result.activities[li] * breathing,
                             result.depolarization[li], None, None, "", result.frequencies)
            _, d_curve = spectrum(rb, grid, fwhm=10.0, laser_nm=1239.84193 / laser)
            spectra[f"{name}_{laser:.2f}_breathing"] = d_curve
            entry[f"activities_{laser:.2f}"] = result.activities[li].tolist()
            order = np.argsort(-result.activities[li])[:12]
            entry[f"strongest_{laser:.2f}"] = [
                {"frequency_cm1": float(result.frequencies[j]),
                 "activity": float(result.activities[li, j]),
                 **{f"on_{g}": float(v[result.mode_indices[j]]) for g, v in shares.items()}}
                for j in order]
        if born_ready(name):
            ir = infrared(name)
            half = 5.0
            shape = half / np.pi / ((ir_grid[:, None] - ir["frequencies"][None]) ** 2 + half ** 2)
            spectra[f"{name}_ir"] = shape @ ir["intensities"]
            order = np.argsort(-ir["intensities"])[:12]
            entry["ir"] = {"axes": ir["axes"], "sum_rule_residual_e": ir["sum_rule_residual_e"],
                           "frequencies_cm1": ir["frequencies"].tolist(),
                           "intensities_km_mol": ir["intensities"].tolist(),
                           "strongest": [{"frequency_cm1": float(ir["frequencies"][j]),
                                          "intensity_km_mol": float(ir["intensities"][j]),
                                          **{f"on_{g}": float(v[ir["mode_indices"][j]])
                                             for g, v in shares.items()}} for j in order]}
        out[name] = entry
    np.savez(WORK / "spectra.npz", grid=grid, ir_grid=ir_grid, **spectra)
    (WORK / "report.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("relax", "hessian", "check", "raman", "born",
                                         "eta", "report"))
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("name", nargs="?", choices=tuple(STRUCTURES))
    parser.add_argument("--part", default="0/1")
    args = parser.parse_args(argv)
    part = tuple(int(x) for x in args.part.split("/"))
    if args.step == "relax":
        relaxed(args.name)
    elif args.step == "hessian":
        done = hessian(args.name, part)
        print("hessiana completa" if done is not None else f"parte {args.part} hecha")
    elif args.step == "check":
        print(json.dumps(check(args.name), indent=1))
    elif args.step == "eta":
        raman_eta(args.name, args.eta, part)
        print(f"raman η={args.eta:g} {args.name} parte {args.part} hecha")
    elif args.step == "born":
        done = born(args.name, part)
        print("cargas de Born completas" if done is not None else f"parte {args.part} hecha")
    elif args.step == "raman":
        raman(args.name, part)
        print(f"raman {args.name} parte {args.part} hecha")
    else:
        print(json.dumps({k: v for k, v in report().items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
