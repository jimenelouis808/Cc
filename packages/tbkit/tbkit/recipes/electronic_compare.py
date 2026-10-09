"""Electronic structure, tbkit against GPAW: the coil's bands and DOS, graphene, molecules.

Run from packages/tbkit (resumable; files in out/electronic/)::

    python -m tbkit.recipes.electronic_compare gpaw NAME   # GPAW side (needs gpaw)
    python -m tbkit.recipes.electronic_compare tb NAME     # tbkit side
    python -m tbkit.recipes.electronic_compare report      # comparison + validation file

NAME is ``pristine``, ``N`` or ``amine`` (the 204-atom coil at its PBE geometry, from
``recipes/doped_gpaw``) or ``graphene``. Both methods see the same geometry; nothing
is fitted here.

* Coil: GPAW LCAO dzp, PBE, h 0.2 Å, Fermi-Dirac 0.05 eV, SCF with 1×1×2 k (as in
  ``doped_gpaw``), then non-self-consistent bands at ``NK`` points from Γ to Z. tbkit
  ``xu_chn`` with self-consistent charges on 1×1×4 k (the setting of the Raman work),
  then its bands on the same points with the converged charge shifts. The DOS of each
  is the band energies on that path (half the 1D zone, which time reversal makes
  enough), weighted by the trapezoid rule and broadened by ``SIGMA``. Energies are
  relative to each method's own Fermi level.
* The dopant's own states: ΔDOS = DOS(doped) − DOS(pristine), same method; the two
  coils have the same number of atoms (N) or two more (amine), so ΔDOS integrates to
  the extra states.
* Graphene: bands along Γ-M-K-Γ at a = 2.467 Å (GPAW SCF 24×24 k). ``xu_chn`` C-C is
  Xu's, so this tests the carbon part that every coil state is made of.
* Molecules: the Kohn-Sham levels stored in ``parameters/references/gpaw_chn.json``
  (GPAW, same settings) against xu_chn's at the same geometry. Those levels were fit
  targets for 16 of the 22 molecules (``role``); the 6 held out are reported apart.
  Occupied levels are compared relative to the HOMO, so no common shift enters.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read

from ..settings import Param, add_arguments, apply

WORK = Path("out/electronic")
COILS = ("pristine", "N", "amine")
NK, SIGMA, KT_TB, KZ_TB = 21, 0.1, 0.05, 4
MODEL = "xu_chn"
GPAW_SETTINGS = {"mode": "lcao", "basis": "dzp", "xc": "PBE", "h": 0.2}
GPAW_KZ, GPAW_KT, K_PER_CALL, EXTRA_BANDS = 2, 0.05, 4, 250
GRAPHENE_KMESH = (24, 48)                   # GPAW, tbkit
GRAPHENE_A = 2.467
WINDOW = (-25.0, 12.0)            # eV around the Fermi level kept in the files


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def geometry(name: str) -> Atoms:
    if name == "graphene":
        from ase.build import graphene

        atoms = graphene(a=GRAPHENE_A, vacuum=7.5)
        atoms.pbc = (True, True, False)
        return atoms
    from . import doped_gpaw as g

    return g.relaxed(name) if name == "pristine" else read(g.folder(name) / "relaxed.extxyz")


def kpath(name: str, atoms: Atoms):
    """Fractional k points and the x axis (1/Å) of the band plot, with labels."""
    if name == "graphene":
        bp = atoms.cell.bandpath("GMKG", npoints=90, pbc=atoms.pbc)
        x, sx, labels = bp.get_linear_kpoint_axis()
        return bp.kpts, x, list(zip(sx.tolist(), labels))
    kz = np.linspace(0.0, 0.5, NK)
    return np.array([[0.0, 0.0, k] for k in kz]), kz * 2 * np.pi / atoms.cell[2, 2], \
        [(0.0, "Γ"), (float(np.pi / atoms.cell[2, 2]), "Z")]


def _keep(eps: np.ndarray) -> list:
    return [[float(e) for e in row if WINDOW[0] <= e <= WINDOW[1]] for row in eps]


# ---------------------------------------------------------------- GPAW side
def run_gpaw(name: str) -> Path:
    from gpaw import GPAW, FermiDirac

    out = WORK / name / "gpaw.json"
    if out.exists():
        return out
    atoms = geometry(name)
    from ..progress import watch_gpaw_scf

    kpts = (GRAPHENE_KMESH[0],) * 2 + (1,) if name == "graphene" else (1, 1, GPAW_KZ)
    atoms.calc = GPAW(**GPAW_SETTINGS, kpts=kpts, symmetry="off",
                      occupations=FermiDirac(GPAW_KT), txt=str(WORK / name / "gpaw_scf.txt"))
    watch_gpaw_scf(atoms.calc, f"GPAW SCF ({name})")
    (WORK / name).mkdir(parents=True, exist_ok=True)
    atoms.get_potential_energy()
    fermi = float(atoms.calc.get_fermi_level())
    k, _, _ = kpath(name, atoms)
    # A few k points per non-SCF call and only the bands the window needs: all 21 at
    # once with every band took 14 GB for the amine coil (killed twice).
    nbands = int(atoms.calc.get_number_of_electrons() // 2) + EXTRA_BANDS
    eps = []
    for chunk in np.array_split(k, max(1, len(k) // K_PER_CALL)):
        bands = atoms.calc.fixed_density(kpts=chunk, symmetry="off", txt=None,
                                         nbands=min(nbands, atoms.calc.get_number_of_bands()))
        eps += [bands.get_eigenvalues(kpt=i) for i in range(len(chunk))]
        del bands
        gc.collect()
    eps = np.array(eps) - fermi
    _write(out, {"name": name, "fermi_eV": fermi, "kpts": k.tolist(), "bands_minus_fermi": _keep(eps),
                 "settings": f"GPAW LCAO dzp PBE h 0.2, FD 0.05 eV, SCF k {kpts}, "
                             f"bands non-SCF on {len(k)} k"})
    return out


# ---------------------------------------------------------------- tbkit side
def run_tb(name: str) -> Path:
    from ..hamiltonian import System
    from ..kpoints import mesh
    from ..params import load_parameters
    from ..scc import _solve_shifted_k, self_consistent

    out = WORK / name / "tb.json"
    if out.exists():
        return out
    atoms = geometry(name)
    system = System.build(atoms, load_parameters(MODEL))
    if name == "graphene":
        k0, w0 = mesh(atoms, GRAPHENE_KMESH[1])
    else:
        k0 = np.array([[0.0, 0.0, (i + 0.5) / KZ_TB - 0.5] for i in range(KZ_TB)])
        w0 = np.full(KZ_TB, 1.0 / KZ_TB)
    result = self_consistent(system, kT=KT_TB, tol=1e-8, kpts=k0, weights=w0)
    fermi = float(result.solution.fermi)
    k, _, _ = kpath(name, atoms)
    path = _solve_shifted_k(system, result.shift, 0.0, KT_TB, k, None)
    eps = path.energies[0] - fermi
    _write(out, {"name": name, "fermi_eV": fermi, "kpts": k.tolist(), "bands_minus_fermi": _keep(eps),
                 "scc_converged": bool(result.converged),
                 "settings": f"xu_chn SCC, kT {KT_TB} eV, SCF k {'48x48' if name == 'graphene' else f'1x1x{KZ_TB}'}"})
    return out


# ---------------------------------------------------------------- comparison
def dos(bands: list, grid: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """States per eV per cell (spin included), trapezoid weights along the path."""
    sigma = SIGMA if sigma is None else sigma
    n = len(bands)
    w = np.full(n, 1.0 / (n - 1))
    w[0] = w[-1] = 0.5 / (n - 1)
    out = np.zeros_like(grid)
    for wk, row in zip(w, bands):
        e = np.asarray(row)[:, None]
        out += 2 * wk * np.exp(-0.5 * ((grid[None] - e) / sigma) ** 2).sum(axis=0) / (sigma * np.sqrt(2 * np.pi))
    return out


def gap(bands: list, at_fermi: float = 0.02) -> float:
    """Smallest gap around the Fermi level along the path; 0 if a state lies within
    ``at_fermi`` of it (a semimetal's Dirac point sits there, on either side)."""
    if any(abs(e) < at_fermi for row in bands for e in row):
        return 0.0
    occ = max(max([e for e in row if e < 0.0], default=-np.inf) for row in bands)
    emp = min(min([e for e in row if e > 0.0], default=np.inf) for row in bands)
    return float(max(0.0, emp - occ))


def graphene_points(bands: list, x: np.ndarray, labels: list) -> dict:
    """Levels at Γ and M, the π van Hove pair at M, and ħv_F (eV·Å) from the π* band one
    path step before K (a chord: it underestimates the slope by the band's curvature)."""
    index = {name: int(np.argmin(np.abs(x - pos))) for pos, name in labels[:3]}
    m = np.asarray(bands[index["M"]])
    k = index["K"]
    pi_star = min(e for e in bands[k - 1] if e > 0)      # one step before the Dirac point
    return {"gamma_eV": sorted(bands[index["G"]]), "M_eV": sorted(bands[index["M"]]),
            "pi_M_below_eV": float(m[m < 0].max()), "pi_M_above_eV": float(m[m > 0].min()),
            "fermi_velocity_eV_A": float(pi_star / (x[k] - x[k - 1]))}


def _pearson(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def molecules() -> dict:
    from ..params import load_parameters
    from ..references import load_references
    from ..tasks import levels

    refs, _ = load_references(Path(__file__).parents[1] / "parameters" / "references" / "gpaw_chn.json")
    model = load_parameters(MODEL)
    rows = []
    for ref in refs:
        if not ref.label.endswith("/eq"):
            continue
        tb, _ = levels(ref.atoms, model)
        lt = np.asarray(tb["levels"])
        n = ref.n_occupied
        ld = np.asarray(ref.levels)
        occ_t, occ_d = lt[:n] - lt[n - 1], ld[:n] - ld[n - 1]
        rows.append({"molecule": ref.group, "role": ref.role,
                     "gap_tb_eV": float(lt[n] - lt[n - 1]), "gap_gpaw_eV": float(ld[n] - ld[n - 1]),
                     "occupied_rms_vs_homo_eV": float(np.sqrt(np.mean((occ_t - occ_d) ** 2))),
                     "occupied_width_tb_eV": float(-occ_t[0]), "occupied_width_gpaw_eV": float(-occ_d[0])})
    out = {"molecules": rows}
    for role in ("train", "test"):
        sel = [r for r in rows if r["role"] == role]
        if sel:
            dg = np.array([r["gap_tb_eV"] - r["gap_gpaw_eV"] for r in sel])
            out[f"summary_{role}"] = {
                "n": len(sel), "gap_mean_error_eV": float(dg.mean()),
                "gap_rms_error_eV": float(np.sqrt(np.mean(dg ** 2))),
                "gap_correlation": _pearson([r["gap_tb_eV"] for r in sel], [r["gap_gpaw_eV"] for r in sel]),
                "occupied_rms_vs_homo_median_eV": float(np.median([r["occupied_rms_vs_homo_eV"] for r in sel]))}
    return out


VALENCE = {"H": 1, "C": 4, "N": 5, "O": 6}


def half_filled_band(bands: list) -> list[float]:
    """Energy range of the band at the Fermi level when the cell has an odd electron
    count: per k, the level nearest E_F (spin-paired, that band holds one electron)."""
    nearest = [min(row, key=abs) for row in bands]
    return [float(min(nearest)), float(max(nearest))]


def report() -> dict:
    grid = np.linspace(-8.0, 6.0, 1401)
    out = {"what": "Electronic structure, tbkit xu_chn against GPAW PBE dzp at the same geometry "
                   "(recipes/electronic_compare)", "sigma_eV": SIGMA, "systems": {}}
    curves = {}
    for name in (*COILS, "graphene"):
        g, t = WORK / name / "gpaw.json", WORK / name / "tb.json"
        if not (g.exists() and t.exists()):
            continue
        g, t = json.loads(g.read_text()), json.loads(t.read_text())
        dg, dt = dos(g["bands_minus_fermi"], grid), dos(t["bands_minus_fermi"], grid)
        curves[name] = (dg, dt)
        electrons = sum(VALENCE[s] for s in geometry(name).get_chemical_symbols())
        entry = {"electrons_per_cell": electrons,
                 "gap_gpaw_eV": gap(g["bands_minus_fermi"]), "gap_tb_eV": gap(t["bands_minus_fermi"]),
                 "settings": {"gpaw": g["settings"], "tb": t["settings"]}}
        if electrons % 2:
            # One band holds a single electron: a metal in both methods, whatever the
            # level spacing on the path says. Its width is what tells them apart.
            entry["gap_gpaw_eV"] = entry["gap_tb_eV"] = 0.0
            entry["half_filled_band_gpaw_eV"] = half_filled_band(g["bands_minus_fermi"])
            entry["half_filled_band_tb_eV"] = half_filled_band(t["bands_minus_fermi"])
        if name == "graphene":
            atoms = geometry(name)
            _, x, labels = kpath(name, atoms)
            entry["gpaw"] = graphene_points(g["bands_minus_fermi"], x, labels)
            entry["tb"] = graphene_points(t["bands_minus_fermi"], x, labels)
            out["systems"][name] = entry
            continue                      # a path DOS is not graphene's 2D DOS
        for lo, hi in ((-2.0, 2.0), (-6.0, 4.0)):
            m = (grid >= lo) & (grid <= hi)
            entry[f"dos_correlation_{lo:+.0f}_{hi:+.0f}"] = _pearson(dg[m], dt[m])
        entry["states_within_1eV"] = {
            "gpaw": float(np.trapezoid(dg[np.abs(grid) <= 1], grid[np.abs(grid) <= 1])),
            "tb": float(np.trapezoid(dt[np.abs(grid) <= 1], grid[np.abs(grid) <= 1]))}
        out["systems"][name] = entry
    for name in ("N", "amine"):
        if name in curves and "pristine" in curves:
            ddg = curves[name][0] - curves["pristine"][0]
            ddt = curves[name][1] - curves["pristine"][1]
            m = (grid >= -3) & (grid <= 3)
            out["systems"][name]["delta_dos_correlation_-3_+3"] = _pearson(ddg[m], ddt[m])
    out["molecules"] = molecules()
    _write(WORK / "report.json", out)
    np.savez(WORK / "dos.npz", grid=grid, **{f"{n}_{s}": c[i] for n, c in curves.items()
                                            for i, s in enumerate(("gpaw", "tb"))})
    Path("validation/electronic_tb_vs_gpaw.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


#: Adjustable with --ajuste NOMBRE=VALOR (tbkit recetas electronic_compare).
PARAMS = [
    Param("GPAW_SETTINGS", "ajustes de GPAW: modo, base, funcional, malla h (Å)", "",
          "los de doped_gpaw, para comparar con las mismas geometrías", "cálculo"),
    Param("GPAW_KZ", "puntos k del SCF de GPAW en la coil", "", "", "convergencia"),
    Param("GPAW_KT", "ensanchamiento de Fermi-Dirac de GPAW", "eV", "", "convergencia"),
    Param("NK", "puntos k del camino Γ–Z de las bandas", "", "", "convergencia"),
    Param("K_PER_CALL", "puntos k por llamada no autoconsistente de GPAW", "",
          "4: los 21 a la vez con todas las bandas usaron 14 GB (la amina se cayó dos veces)",
          "recursos"),
    Param("EXTRA_BANDS", "bandas vacías calculadas por encima de las ocupadas", "",
          "250 cubren la ventana del reporte", "convergencia"),
    Param("MODEL", "modelo de TB", "", "", "cálculo"),
    Param("KZ_TB", "puntos k del SCC de tbkit", "", "4, como el trabajo de la coil", "convergencia"),
    Param("KT_TB", "ensanchamiento de Fermi-Dirac de tbkit", "eV", "", "convergencia"),
    Param("GRAPHENE_A", "constante de red del grafeno", "Å", "", "cálculo"),
    Param("GRAPHENE_KMESH", "malla k del SCF del grafeno (GPAW y tbkit)", "", "", "convergencia"),
    Param("SIGMA", "ensanchamiento gaussiano de la DOS", "eV", "", "salida"),
    Param("WINDOW", "niveles guardados alrededor del nivel de Fermi", "eV", "", "salida"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("gpaw", "tb", "report"))
    parser.add_argument("name", nargs="?", choices=(*COILS, "graphene"))
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    if args.step == "report":
        print(json.dumps(report()["systems"], indent=1, ensure_ascii=False))
    elif args.step == "gpaw":
        print(run_gpaw(args.name))
    else:
        print(run_tb(args.name))


if __name__ == "__main__":
    main()
