"""Double-resonance Raman (D, 2D) of the 204-atom 5-7 coil with xu_chn.

Run from packages/tbkit (resumable; files in out/coil_dr/)::

    python -m tbkit.recipes.coil_double_resonance fc [--part r/n]   # force constants
    python -m tbkit.recipes.coil_double_resonance check             # D(q=0) vs the Γ modes
    python -m tbkit.recipes.coil_double_resonance twod LASER [--part r/n]   # 2D-type band
    python -m tbkit.recipes.coil_double_resonance dband LASER [--part r/n]  # defect D band
    python -m tbkit.recipes.coil_double_resonance gband LASER               # G reference
    python -m tbkit.recipes.coil_double_resonance report                    # spectra, tables

Stage 3 (``twod``, ``dband``, ``gband``) puts these phonons into
:mod:`tbkit.double_resonance` with the coil's own electrons (the same model, all bands
within ``WINDOW`` of E_F, ``NK`` k points along the axis): two phonons (q, ν) and
(−q, ν) for the second-order band near 2×1350 cm⁻¹ (graphene's 2D) and its relatives,
and one phonon plus elastic scattering by a defect for the defect-activated band near
1350 (graphene's D, and D′). The defect is an N-like substitution reduced to its
on-site part: the xu_chn N–C on-site difference, screened by the SCC shift that the
N actually gets in the N-doped coil (``recipes/doped_raman``). Intensities are summed
over the x, y, z polarisations (a sample of randomly oriented coils, roughly); the
D band's is per defect per cell. Laser energies are the model's: the TB coil has a
0.19 eV gap where PBE has 0.22 (``recipes/electronic_compare``).

Stage 2 of the double-resonance work: the phonons of the coil at any q along its
axis. The coil is periodic along z only, with a 15 Å period; force constants Φ(R)
between the cell and its neighbours R = −1, 0, +1 come from finite displacements of
the central cell's atoms in a 1×1×3 supercell (xu_chn with SCC, 1×1×2 k — the 1×1×4
of the primitive cell's Raman work, in the longer cell — kT 0.05 eV, ±0.01 Å). The
dynamical matrix D(q) is then their Fourier sum, exact at q = 0, ±1/3 (in 2π/c) and
smooth in between, as for graphene in ``tbkit.graphene``. ``check`` compares D(0)
with the Γ Hessian of ``recipes/doped_raman`` (1×1×4 k in the primitive cell).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

WORK = Path("out/coil_dr")
#: xu_carbon (Xu's sp³ carbon, no SCC): a 612-atom supercell force call takes ~14 min
#: with xu_chn's self-consistent charges and ~1-2 with xu_carbon; for the undoped coil
#: xu_chn's C-C is Xu's, so the difference is the charge transfer between rings only.
MODEL = "xu_carbon"
SOURCE = Path("out/nanocoil/tb_xu_carbon/relaxed.extxyz")
GAMMA_MODES = Path("out/nanocoil/tb_xu_carbon/modes.npz")
REPEAT, KZ, KT, DELTA = 3, 2, 0.05, 0.01
NK, WINDOW, GAMMA = 24, 3.2, 0.1            # k points along the axis, eV, eV
BAND_RANGE = (1100.0, 1750.0)               # cm⁻¹: the phonons that enter D, D', 2D, 2D'
POLARISATIONS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def primitive():
    return read(SOURCE)


def supercell():
    return primitive().repeat((1, 1, REPEAT))


def _calculator():
    from ..calculator import TBCalculator
    from ..params import load_parameters

    return TBCalculator(load_parameters(MODEL), kpts=(1, 1, KZ), kT=KT)


def fc(part: tuple[int, int] = (0, 1)) -> None:
    """Forces of the supercell with each atom of the central cell displaced ±DELTA along
    x, y, z: one file per atom (``fc/a{atom}.npy``, shape (3, 2, n_super, 3))."""
    folder = WORK / "fc"
    folder.mkdir(parents=True, exist_ok=True)
    n = len(primitive())
    big = supercell()
    centre = n * (REPEAT // 2)                    # atoms of cell 1 (of 0, 1, 2)
    calc = _calculator()
    r, total = part
    for a in range(n):
        if a % total != r:
            continue
        path = folder / f"a{a:03d}.npy"
        if path.exists():
            continue
        forces = np.zeros((3, 2, len(big), 3))
        for alpha in range(3):
            for s, step in enumerate((DELTA, -DELTA)):
                moved = big.copy()
                moved.positions[centre + a, alpha] += step
                moved.calc = calc
                forces[alpha, s] = moved.get_forces()
                calc.reset()
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, forces)
        tmp.replace(path)


def force_constants() -> np.ndarray:
    """Φ[R, a, α, b, β] (eV/Å²), R = −1, 0, +1, from the central-cell displacements;
    symmetrised (Φ_ab(R) = Φ_ba(−R)) and with the acoustic sum rule imposed."""
    n = len(primitive())
    phi = np.zeros((3, n, 3, n, 3))
    for a in range(n):
        f = np.load(WORK / "fc" / f"a{a:03d}.npy")
        d = -(f[:, 0] - f[:, 1]) / (2 * DELTA)          # (α, n_super, 3) = Φ[a α, j β]
        for cell in range(REPEAT):
            phi[cell, a] = d[:, cell * n:(cell + 1) * n, :]
    # cells 0, 1, 2 of the supercell are R = −1, 0, +1 relative to the centre
    phi = 0.5 * (phi + np.transpose(phi[::-1], (0, 3, 4, 1, 2)))
    for a in range(n):
        for alpha in range(3):
            for beta in range(3):
                phi[1, a, alpha, a, beta] -= phi[:, a, alpha, :, beta].sum()
    return phi


def dynamical_matrix(phi: np.ndarray, masses: np.ndarray, q: float) -> np.ndarray:
    """D(q) for q in units of 2π/c (lattice convention: phase e^{2πi q R})."""
    n = len(masses)
    d = sum(phi[i] * np.exp(2j * np.pi * q * r) for i, r in enumerate((-1, 0, 1)))
    d = d.reshape(3 * n, 3 * n)
    m = np.repeat(masses, 3)
    return d / np.sqrt(np.outer(m, m))


def check() -> dict:
    from ase.units import invcm

    atoms = primitive()
    masses = atoms.get_masses()
    phi = force_constants()
    w2 = np.linalg.eigvalsh(0.5 * (dynamical_matrix(phi, masses, 0.0)
                                   + dynamical_matrix(phi, masses, 0.0).conj().T)).real
    from ase.units import _amu, _e, _hbar
    to_cm1 = np.sqrt(_e / _amu) * 1e10 * _hbar / _e / invcm
    ours = np.sign(w2) * np.sqrt(np.abs(w2)) * to_cm1
    ref = np.sort(np.load(GAMMA_MODES)["frequencies"]) if "frequencies" in \
        np.load(GAMMA_MODES).files else np.sort(np.load(GAMMA_MODES)["f"])
    diff = ours[6:] - ref[6:]
    out = {"modes": len(ours), "max_abs_diff_cm1": float(np.abs(diff).max()),
           "rms_diff_cm1": float(np.sqrt(np.mean(diff ** 2))),
           "highest_cm1": [float(ours[-1]), float(ref[-1])]}
    (WORK / "check.json").write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- stage 3
def _setup():
    from ..double_resonance import Coupling, mesh_bands
    from ..hamiltonian import System
    from ..params import load_parameters

    atoms = primitive()
    system = System.build(atoms, load_parameters(MODEL))
    bands, _ = mesh_bands(system, (1, 1, NK), WINDOW, POLARISATIONS)
    return atoms, system, bands, Coupling(system)


def phonons_at(m: int):
    """Frequencies (cm⁻¹) and lattice-convention eigenvectors of D(q = m/NK)."""
    from ase.units import _amu, _e, _hbar, invcm

    atoms = primitive()
    masses = atoms.get_masses()
    path = WORK / "phonons" / f"q{m:02d}.npz"
    if path.exists():
        data = np.load(path)
        return data["f"], data["e"]
    path.parent.mkdir(parents=True, exist_ok=True)
    d = dynamical_matrix(force_constants(), masses, m / NK)
    w2, vec = np.linalg.eigh(0.5 * (d + d.conj().T))
    to_cm1 = np.sqrt(_e / _amu) * 1e10 * _hbar / _e / invcm
    f = np.sign(w2) * np.sqrt(np.abs(w2)) * to_cm1
    e = vec.T.reshape(len(f), len(masses), 3)
    np.savez(path, f=f, e=e)
    return f, e


def _jobs():
    """(m, ν) with ν in BAND_RANGE, m = 0..NK/2 (−q gives the same, counted twice)."""
    out = []
    for m in range(NK // 2 + 1):
        f, _ = phonons_at(m)
        out += [(m, int(nu)) for nu in np.flatnonzero((f >= BAND_RANGE[0]) & (f <= BAND_RANGE[1]))]
    return out


def defect_vertex(system):
    """N-like substitution at one atom: on-site difference (xu_chn N − C) plus the SCC
    shift the N gets in the N-doped coil relative to its carbons."""
    from ..double_resonance import DefectVertex
    from ..params import load_parameters

    chn = load_parameters("xu_chn")
    ds = chn.onsite["N"]["s"] - chn.onsite["C"]["s"]
    dp = chn.onsite["N"]["p"] - chn.onsite["C"]["p"]
    shift = json.loads((WORK / "n_shift.json").read_text())["n_minus_c_eV"] \
        if (WORK / "n_shift.json").exists() else 0.0
    site = int(json.loads((WORK / "n_shift.json").read_text()).get("site", 0)) \
        if (WORK / "n_shift.json").exists() else 0
    block = np.diag([ds + shift] + [dp + shift] * 3)
    return DefectVertex(system, {(site, site, (0, 0, 0)): (block, None)}), site


def n_shift() -> dict:
    """SCC shift of the N against the mean carbon shift in the N-doped coil (xu_chn)."""
    from ..hamiltonian import System
    from ..params import load_parameters
    from ..scc import self_consistent
    from . import doped_raman as tb

    atoms = read(tb.WORK / "N" / "relaxed.extxyz")
    system = System.build(atoms, load_parameters("xu_chn"))
    k = np.array([[0.0, 0.0, (i + 0.5) / 4 - 0.5] for i in range(4)])
    result = self_consistent(system, kT=KT, tol=1e-8, kpts=k, weights=np.full(4, 0.25))
    per_atom = np.array([result.shift[system.basis.of_atom(a).start] for a in range(len(atoms))])
    n = atoms.get_chemical_symbols().index("N")
    carbons = [a for a, s in enumerate(atoms.get_chemical_symbols()) if s == "C"]
    out = {"n_minus_c_eV": float(per_atom[n] - per_atom[carbons].mean()), "site": n}
    (WORK / "n_shift.json").write_text(json.dumps(out, indent=1))
    return out


def run_band(kind: str, laser: float, part: tuple[int, int]) -> Path:
    from ..double_resonance import PhononVertex, two_vertices_q

    atoms, system, bands, coupling = _setup()
    masses = atoms.get_masses()
    defect = defect_vertex(system)[0] if kind == "dband" else None
    out = WORK / f"{kind}_{laser:.2f}" / f"part{part[0]}of{part[1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else {}
    for n, (m, nu) in enumerate(_jobs()):
        key = f"{m}:{nu}"
        if n % part[1] != part[0] or key in done:
            continue
        f, e = phonons_at(m)
        first = PhononVertex(coupling, e[nu], masses, f[nu])
        second = defect if kind == "dband" else PhononVertex(coupling, e[nu], masses, f[nu],
                                                               conjugate=True)
        a = two_vertices_q(bands, (1, 1, NK), (0, 0, m), first, second, laser, GAMMA)
        power = float(np.sum(np.abs(a) ** 2))
        done[key] = {"q": m / NK, "frequency_cm1": float(f[nu]),
                     "shift_cm1": float(f[nu] if kind == "dband" else 2 * f[nu]),
                     "intensity": power / 2 if kind == "twod" else power,
                     "multiplicity": 1 if m in (0, NK // 2) else 2}
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(done))
        tmp.replace(out)
    return out


def g_band(laser: float) -> dict:
    """First-order Γ modes in BAND_RANGE (the G band and its neighbours), same units."""
    from ..double_resonance import PhononVertex, one_vertex

    atoms, _, bands, coupling = _setup()
    f, e = phonons_at(0)
    rows = {}
    for nu in np.flatnonzero((f >= BAND_RANGE[0]) & (f <= BAND_RANGE[1])):
        a = one_vertex(bands, PhononVertex(coupling, e[nu], atoms.get_masses(), f[nu]),
                       laser, GAMMA)
        rows[str(int(nu))] = {"frequency_cm1": float(f[nu]),
                              "intensity": float(np.sum(np.abs(a) ** 2))}
    path = WORK / f"gband_{laser:.2f}.json"
    path.write_text(json.dumps(rows))
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("fc", "check", "nshift", "twod", "dband", "gband"))
    parser.add_argument("laser", nargs="?", type=float)
    parser.add_argument("--part", default="0/1")
    args = parser.parse_args(argv)
    part = tuple(int(x) for x in args.part.split("/"))
    if args.step == "fc":
        fc(part)
    elif args.step == "check":
        print(json.dumps(check(), indent=1))
    elif args.step == "nshift":
        print(json.dumps(n_shift(), indent=1))
    elif args.step == "gband":
        print(len(g_band(args.laser)))
    else:
        print(run_band(args.step, args.laser, part))


if __name__ == "__main__":
    main()
