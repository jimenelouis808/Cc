"""Double-resonance Raman (D, 2D) of the 204-atom 5-7 coil (Xu carbon; N-like defect for D).

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
import sys
from pathlib import Path

import numpy as np
from ase.io import read

from ..progress import Progress
from ..settings import Param, add_arguments, apply

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
    mine = [a for a in range(n) if a % total == r]
    bar = Progress(len(mine), f"constantes de fuerza, parte {r + 1}/{total}",
                   status=folder / f"progreso_{r}.json",
                   done=sum((folder / f"a{a:03d}.npy").exists() for a in mine))
    for a in mine:
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
        bar.step()
    bar.close()


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
    mine = [(m, nu) for n, (m, nu) in enumerate(_jobs()) if n % part[1] == part[0]]
    bar = Progress(len(mine), f"{kind} {laser:.2f} eV, parte {part[0] + 1}/{part[1]}",
                   status=out.parent / f"progreso_{part[0]}.json",
                   done=sum(f"{m}:{nu}" in done for m, nu in mine))
    for m, nu in mine:
        key = f"{m}:{nu}"
        if key in done:
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
        bar.step()
    bar.close()
    return out


#: Branch windows for the two-phonon bands: every pair inside one window (the 2D region
#: from the D-type branches, the 2D'/2G region from the G-type ones); pairs across the
#: two windows (D + D'-type combinations near 2900 cm⁻¹) are left out.
PAIR_WINDOWS = {"2D": (1150.0, 1480.0), "2G": (1480.0, 1750.0)}


def run_pairs(laser: float, part: tuple[int, int]) -> Path:
    """W[ν, ν'] of every pair (q, ν), (−q, ν') within each window, per q (one file each)."""
    from ..double_resonance import phonon_pairs_q_fast

    atoms, _, bands, coupling = _setup()
    folder = WORK / f"pairs_{laser:.2f}"
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [(m, name) for m in range(NK // 2 + 1) for name in PAIR_WINDOWS]
    mine = [(m, name) for n, (m, name) in enumerate(jobs) if n % part[1] == part[0]]
    done = sum((folder / f"q{m:02d}_{name}.npz").exists() for m, name in mine)
    bar = Progress(len(mine), f"pares {laser:.2f} eV, parte {part[0] + 1}/{part[1]}",
                   status=folder / f"progreso_{part[0]}.json", done=done)
    for m, name in mine:
        path = folder / f"q{m:02d}_{name}.npz"
        if path.exists():
            continue
        f, e = phonons_at(m)
        lo, hi = PAIR_WINDOWS[name]
        sel = np.flatnonzero((f >= lo) & (f <= hi))
        w = phonon_pairs_q_fast(bands, (1, 1, NK), coupling, (0, 0, m), f[sel], e[sel],
                                atoms.get_masses(), laser, GAMMA)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, frequencies=f[sel], w=w, m=m)
        tmp.replace(path)
        bar.step(note=f"q = {m}/{NK}, {name}")
    bar.close()
    return folder


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


LASERS = (1.96, 2.33, 2.54)
FWHM = 30.0                                  # cm⁻¹, Lorentzian width of the drawn spectra
#: PBE/xu_carbon frequency ratio on the coil (recipes/doped_gpaw, xu_chn = Xu's C-C):
#: 0.950 for the G modes, 0.960 ± 0.020 for the lattice 1000-1800 cm⁻¹.
SCALE_G, SCALE_LATTICE = 0.950, 0.960
WINDOWS = {"G": (1500, 1700), "D": (1200, 1460), "D'": (1560, 1720),
           "2D": (2400, 2920), "2D'+2G": (3000, 3480)}


def _load(kind: str, laser: float) -> list[dict]:
    rows = []
    for path in sorted((WORK / f"{kind}_{laser:.2f}").glob("part*.json")):
        rows += list(json.loads(path.read_text()).values())
    return rows


def _pairs_rows(laser: float) -> list[dict]:
    """Two-phonon lines from ``run_pairs``: every ordered pair (q, ν), (−q, ν') over the
    whole q mesh (q and −q both counted, all ordered pairs, as ``tbkit.graphene`` counts its
    2D: the same convention, so I(2D)/I(G) compares with graphene's)."""
    rows = []
    for path in sorted((WORK / f"pairs_{laser:.2f}").glob("q*.npz")):
        data = np.load(path)
        f, w, m = data["frequencies"], data["w"], int(data["m"])
        multiplicity = 1 if m in (0, NK // 2) else 2
        shifts = f[:, None] + f[None, :]
        for i in range(len(f)):
            for j in range(len(f)):
                rows.append({"q": m / NK, "frequency_cm1": float(0.5 * shifts[i, j]),
                             "shift_cm1": float(shifts[i, j]), "intensity": float(w[i, j]),
                             "multiplicity": multiplicity})
    return rows


def _lorentz(grid, shifts, weights, fwhm=None):
    half = (FWHM if fwhm is None else fwhm) / 2
    return (weights[None, :] * half / np.pi / ((grid[:, None] - shifts[None, :]) ** 2
                                                + half ** 2)).sum(axis=1)


def report() -> dict:
    """Spectra, band positions, intensity ratios and their laser dispersion."""
    out = {"model": MODEL, "nk": NK, "gamma_eV": GAMMA, "fwhm_cm1": FWHM,
           "intensity_note": "Σ over x, y, z polarisations; q-sums are means over the NK points "
                             "(−q counted with q); the D band is per defect per cell",
           "lasers": {}}
    grid1, grid2 = np.arange(1000.0, 1800.0, 1.0), np.arange(2000.0, 3600.0, 1.0)
    spectra = {"grid1": grid1, "grid2": grid2}
    for laser in LASERS:
        path = WORK / f"gband_{laser:.2f}.json"
        if not path.exists():
            continue
        g = list(json.loads(path.read_text()).values())
        gf = np.array([r["frequency_cm1"] for r in g])
        gi = np.array([r["intensity"] for r in g])
        entry = {"I_G": float(gi[(gf >= WINDOWS["G"][0]) & (gf <= WINDOWS["G"][1])].sum()),
                 "G_position_cm1": float(grid1[np.argmax(_lorentz(grid1, gf, gi))])}
        spectra[f"G_{laser:.2f}"] = _lorentz(grid1, gf, gi)
        for kind, grid in (("dband", grid1), ("twod", grid2)):
            rows = _load(kind, laser) if kind == "dband" else _pairs_rows(laser)
            if not rows:
                continue
            shifts = np.array([r["shift_cm1"] for r in rows])
            weights = np.array([r["intensity"] * r["multiplicity"] for r in rows]) / NK
            curve = _lorentz(grid, shifts, weights)
            spectra[f"{kind}_{laser:.2f}"] = curve
            names = ("D", "D'") if kind == "dband" else ("2D", "2D'+2G")
            for name in names:
                lo, hi = WINDOWS[name]
                mask = (shifts >= lo) & (shifts <= hi)
                gm = (grid >= lo) & (grid <= hi)
                entry[f"I_{name}"] = float(weights[mask].sum())
                entry[f"{name}_position_cm1"] = float(grid[gm][np.argmax(curve[gm])])
                entry[f"I_{name}/I_G"] = float(weights[mask].sum() / entry["I_G"])
            top = np.argsort(weights)[::-1][:8]
            entry[f"top_{kind}"] = [{"q": rows[i]["q"], "phonon_cm1": rows[i]["frequency_cm1"],
                                     "shift_cm1": rows[i]["shift_cm1"],
                                     "share": float(weights[i] / weights.sum())} for i in top]
        out["lasers"][f"{laser:.2f}"] = entry
    done = [float(k) for k, e in out["lasers"].items() if "D_position_cm1" in e]
    for name in ("D", "2D"):
        pts = [(float(k), e[f"{name}_position_cm1"]) for k, e in out["lasers"].items()
               if f"{name}_position_cm1" in e]
        if len(pts) >= 2:
            x, y = np.array(pts).T
            out[f"{name}_dispersion_cm1_per_eV"] = float(np.polyfit(x, y, 1)[0])
    out["scaled_note"] = (f"posiciones crudas de {MODEL}; escaladas a PBE: G ×{SCALE_G}, resto "
                          f"×{SCALE_LATTICE} (recipes/doped_gpaw)")
    out["lasers_with_d"] = done
    (WORK / "report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez(WORK / "spectra.npz", **spectra)
    return out


#: Adjustable with --ajuste NOMBRE=VALOR (tbkit recetas coil_double_resonance).
PARAMS = [
    Param("MODEL", "modelo de TB para electrones y fonones", "",
          "xu_carbon: la coil sin dopar es solo carbono y su fuerza cuesta 38 s en la supercelda "
          "frente a ~14 min con las cargas autoconsistentes de xu_chn", "cálculo"),
    Param("SOURCE", "geometría de partida (relajada con MODEL)", "", "", "rutas"),
    Param("REPEAT", "periodos de la supercelda para las constantes de fuerza", "",
          "3: Φ(R) a ±1 celda; D(q=0) reproduce los modos Γ a 0.013 cm⁻¹", "convergencia"),
    Param("KZ", "puntos k a lo largo del eje en la supercelda de fuerzas", "",
          "2 en 3 periodos equivale a 6 en la celda: los modos Γ convergen con 4", "convergencia"),
    Param("KT", "ensanchamiento de Fermi-Dirac", "eV", "el de todo el trabajo de la coil",
          "convergencia"),
    Param("DELTA", "desplazamiento de las diferencias finitas de las fuerzas", "Å", "",
          "convergencia"),
    Param("NK", "puntos k (y q) a lo largo del eje en la doble resonancia", "",
          "24: paso de 0.017 Å⁻¹, la energía cambia menos que γ entre puntos", "convergencia"),
    Param("WINDOW", "estados a ± esta energía del nivel de Fermi", "eV",
          "cubre el láser más alto (2.54 eV) con margen", "convergencia"),
    Param("GAMMA", "ancho de los estados intermedios (γ)", "eV",
          "0.1, como el η del Raman resonante; las alturas dependen de él", "cálculo"),
    Param("BAND_RANGE", "fonones que entran en D (un fonón)", "cm⁻¹", "", "cálculo"),
    Param("PAIR_WINDOWS", "ventanas de ramas para los pares de dos fonones (2D, 2G)", "cm⁻¹",
          "los pares entre ventanas (D+D') se dejan fuera", "cálculo"),
    Param("POLARISATIONS", "polarizaciones de luz sumadas", "",
          "x, y, z: muestra de coils orientadas al azar, aproximadamente", "cálculo"),
    Param("LASERS", "energías de láser del reporte", "eV", "633, 532, 488 nm", "salida"),
    Param("FWHM", "ancho lorentziano de los espectros dibujados", "cm⁻¹", "", "salida"),
    Param("SCALE_G", "factor PBE/TB para la banda G", "", "medido en recipes/doped_gpaw",
          "salida"),
    Param("SCALE_LATTICE", "factor PBE/TB para el resto de la red", "",
          "0.960 ± 0.020 medido en recipes/doped_gpaw", "salida"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("fc", "check", "nshift", "twod", "dband", "gband",
                                         "pairs", "report"))
    parser.add_argument("laser", nargs="?", type=float)
    parser.add_argument("--part", default="0/1")
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    part = tuple(int(x) for x in args.part.split("/"))
    if args.step == "fc":
        fc(part)
    elif args.step == "check":
        print(json.dumps(check(), indent=1))
    elif args.step == "nshift":
        print(json.dumps(n_shift(), indent=1))
    elif args.step == "pairs":
        print(run_pairs(args.laser, part))
    elif args.step == "report":
        print(json.dumps(report(), indent=1, ensure_ascii=False)[:3000])
    elif args.step == "gband":
        print(len(g_band(args.laser)))
    else:
        print(run_band(args.step, args.laser, part))


if __name__ == "__main__":
    main()
