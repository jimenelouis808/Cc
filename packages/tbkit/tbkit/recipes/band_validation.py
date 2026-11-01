"""Band structures of graphene and of semiconducting and metallic nanotubes, tbkit against GPAW.

Run from packages/tbkit (resumable; GPAW results in out/bands/, the comparison in
``validation/bands_tb_vs_gpaw.json``)::

    python -m tbkit.recipes.band_validation gpaw SYSTEM [--code pw|dzp]   # GPAW reference
    python -m tbkit.recipes.band_validation tb                            # every TB model
    python -m tbkit.recipes.band_validation report                        # validation file

Systems: graphene (Γ-M-K-Γ), zigzag (8,0) and (10,0) (semiconducting), (9,0) (metallic
in the π model, a small curvature gap in DFT), armchair (5,5) and (6,6) (metallic). Every
method sees the same geometry: graphene at a = 2.467 Å (PBE), tubes rolled from it with
ASE's builder (C-C 1.424 Å), nothing relaxed, so the bands differ only by the Hamiltonian.

* Reference: GPAW PBE with plane waves (``pw``, 400 eV), Fermi-Dirac 0.05 eV, SCF on a
  k mesh, bands non-self-consistent on the path. ``dzp`` runs the LCAO dzp basis used
  everywhere else in this package, with the same settings: it measures that basis's own
  error against plane waves, which is the error bar of every other GPAW comparison here.
* TB: ``pi`` (Hückel, t = −2.7 eV), ``sp3`` (Xu), ``tang`` (tang_carbon) and ``chn``
  (xu_chn with its self-consistent charges; on pure carbon its C-C is Xu's).

Compared, each method relative to its own Fermi level (mid-gap when there is a gap):
the gap and where it is; the optical transitions E11, E22 (M11 for metals) from the band
extrema, the quantities a Kataura plot uses; the Fermi velocity of the Dirac bands; the
bottom of graphene's π band at Γ; and the band RMS error of the 4 highest valence and 4
lowest conduction bands (band index counted from the HOMO, which is the same in every
method: 4 valence electrons per carbon in TB s+p and in GPAW's PAW setups; the π model
has one per carbon, so it is indexed from its own HOMO).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ase import Atoms

from ..settings import Param, add_arguments, apply

WORK = Path("out/bands")
OUT = Path("validation/bands_tb_vs_gpaw.json")
SYSTEMS = ("graphene", "8,0", "10,0", "9,0", "5,5", "6,6")
MODELS = ("pi", "sp3", "tang", "chn")
GRAPHENE_A = 2.467
VACUUM = 7.0
NK_TUBE, NK_GRAPHENE = 31, 90   # 31: the path contains kz = 1/3, where armchair tubes cross
KMESH_TUBE, KMESH_GRAPHENE = 12, 24
PW_CUTOFF = 400.0
KT = 0.05
BANDS_EACH_SIDE = 4
WINDOW = (-12.0, 8.0)                     # eV around the Fermi level kept in the files


def geometry(name: str) -> Atoms:
    """``graphene``, ``graphene@A`` (lattice constant A Å: strained, for the fit of
    distance laws in ``recipes/band_fit``) or a tube ``n,m``."""
    if name.startswith("graphene"):
        from ase.build import graphene

        a = float(name.split("@")[1]) if "@" in name else GRAPHENE_A
        atoms = graphene(a=a, vacuum=VACUUM)
        atoms.pbc = (True, True, False)
        return atoms
    from ase.build import nanotube

    n, m = (int(x) for x in name.split(","))
    bond = GRAPHENE_A / np.sqrt(3)
    atoms = nanotube(n, m, length=1, bond=bond, vacuum=VACUUM)
    atoms.pbc = (False, False, True)
    return atoms


def kpath(name: str, atoms: Atoms):
    """Fractional k points, the x axis of the plot (1/Å) and its labels."""
    if name.startswith("graphene"):
        bp = atoms.cell.bandpath("GMKG", npoints=NK_GRAPHENE, pbc=atoms.pbc)
        x, sx, labels = bp.get_linear_kpoint_axis()
        return bp.kpts, x, list(zip(sx.tolist(), labels))
    kz = np.linspace(0.0, 0.5, NK_TUBE)
    c = atoms.cell[2, 2]
    return np.c_[np.zeros(NK_TUBE), np.zeros(NK_TUBE), kz], kz * 2 * np.pi / c, \
        [(0.0, "Γ"), (float(np.pi / c), "Z")]


def scf_mesh(name: str) -> tuple[int, int, int]:
    return (KMESH_GRAPHENE, KMESH_GRAPHENE, 1) if name.startswith("graphene") \
        else (1, 1, KMESH_TUBE)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def _slug(name: str) -> str:
    return name.replace(",", "_").replace("@", "_a")


# ---------------------------------------------------------------- GPAW
def run_gpaw(name: str, code: str = "pw") -> Path:
    from gpaw import GPAW, PW, FermiDirac

    out = WORK / _slug(name) / f"gpaw_{code}.json"
    if out.exists():
        return out
    atoms = geometry(name)
    settings = ({"mode": PW(PW_CUTOFF)} if code == "pw"
                else {"mode": "lcao", "basis": "dzp", "h": 0.2})
    out.parent.mkdir(parents=True, exist_ok=True)
    # time reversal only: no point group (an unrelaxed tube's symmetry is not needed and the
    # k points of the band path stay as given)
    atoms.calc = GPAW(**settings, xc="PBE", kpts=scf_mesh(name),
                      symmetry={"point_group": False, "time_reversal": True},
                      occupations=FermiDirac(KT), txt=str(out.with_suffix(".txt")))
    atoms.get_potential_energy()
    fermi = float(atoms.calc.get_fermi_level())
    k, _, _ = kpath(name, atoms)
    n_occ = len(atoms) * 2
    band = atoms.calc.fixed_density(kpts=k, symmetry="off", txt=None,
                                    nbands=n_occ + 2 * len(atoms),
                                    convergence={"bands": n_occ + len(atoms)})
    eps = np.array([band.get_eigenvalues(kpt=i) for i in range(len(k))])
    _write(out, {"system": name, "code": f"gpaw_{code}", "fermi_eV": fermi, "n_occ": n_occ,
                 "kpts": k.tolist(), "bands": eps[:, : n_occ + len(atoms)].tolist(),
                 "settings": f"GPAW PBE {'PW ' + str(PW_CUTOFF) + ' eV' if code == 'pw' else 'LCAO dzp h 0.2'}, "
                             f"FD {KT} eV, SCF k {scf_mesh(name)}, bands non-SCF on {len(k)} k"})
    return out


# ---------------------------------------------------------------- tbkit
def run_tb(name: str, model_name: str) -> Path:
    from ..gui.actions import load_model
    from ..hamiltonian import System
    from ..kpoints import mesh
    from ..solver import solve

    out = WORK / _slug(name) / f"tb_{model_name}.json"
    if out.exists():
        return out
    atoms = geometry(name)
    model = load_model(model_name)
    system = System.build(atoms, model)
    k0, w0 = mesh(system, KMESH_GRAPHENE if name.startswith("graphene") else KMESH_TUBE)
    k, _, _ = kpath(name, atoms)
    if getattr(model, "scc", False):
        from ..scc import _solve_shifted_k, self_consistent

        result = self_consistent(system, kT=KT, tol=1e-8, kpts=k0, weights=w0)
        fermi = float(result.solution.fermi)
        eps = _solve_shifted_k(system, result.shift, 0.0, KT, k, None).energies[0]
    else:
        from ..analysis import bands

        fermi = float(solve(system, k0, w0, kT=KT).fermi)
        eps = bands(system, k)[0]
    n_occ = int(round(system.electrons)) // 2
    _write(out, {"system": name, "code": f"tb_{model_name}", "fermi_eV": fermi, "n_occ": n_occ,
                 "kpts": k.tolist(), "bands": np.asarray(eps).tolist(),
                 "settings": f"tbkit {model.name}, kT {KT} eV, SCF/Fermi k mesh "
                             f"{KMESH_GRAPHENE if name.startswith('graphene') else KMESH_TUBE}"})
    return out


# ---------------------------------------------------------------- comparison
def reference_level(eps: np.ndarray, n_occ: int, fermi: float) -> float:
    """Mid-gap when the bands have a gap, else the Fermi level."""
    vbm, cbm = eps[:, n_occ - 1].max(), eps[:, n_occ].min()
    return 0.5 * (vbm + cbm) if cbm - vbm > 0.01 else fermi


def _distinct(values, tol=0.005):
    out = []
    for v in values:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def band_crossing(eps: np.ndarray, n: int, x: np.ndarray, tolerance: float = 0.3):
    """A crossing of the highest valence and lowest conduction band between two k points.

    Bands are sorted by energy at each k, so where two bands cross between path points the
    sorted bands show a narrow V and a gap that is only the sampling: the curvature of a
    real tube moves the Fermi point of an armchair tube off kz = 1/3. Between points i and
    i+1, the bands continue as the swapped lines (E_{n-1}(i) -> E_n(i+1)) and
    (E_n(i) -> E_{n-1}(i+1)); it is a crossing when those two lines have opposite slopes and
    each matches, within ``tolerance``, the slope of its band on the neighbouring interval.
    Returns the two slopes (eV per unit of ``x``) or None."""
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
        if h <= 0:
            continue
        s1 = (eps[i + 1, n] - eps[i, n - 1]) / h
        s2 = (eps[i + 1, n - 1] - eps[i, n]) / h
        if s1 * s2 >= 0:
            continue
        checks = []
        if i > 0 and x[i] > x[i - 1]:
            hl = x[i] - x[i - 1]
            checks += [((eps[i, n - 1] - eps[i - 1, n - 1]) / hl, s1),
                       ((eps[i, n] - eps[i - 1, n]) / hl, s2)]
        if i + 2 < len(x) and x[i + 2] > x[i + 1]:
            hr = x[i + 2] - x[i + 1]
            checks += [((eps[i + 2, n] - eps[i + 1, n]) / hr, s1),
                       ((eps[i + 2, n - 1] - eps[i + 1, n - 1]) / hr, s2)]
        if checks and all(abs(a - b) <= tolerance * abs(b) for a, b in checks):
            return abs(s1), abs(s2)
    return None


def metrics(name: str, data: dict) -> dict:
    eps = np.asarray(data["bands"], dtype=float)
    n = int(data["n_occ"])
    zero = reference_level(eps, n, data["fermi_eV"])
    eps = eps - zero
    vbm, cbm = eps[:, n - 1].max(), eps[:, n].min()
    gap = float(max(0.0, cbm - vbm))
    x = _xaxis(name, data)
    crossing = band_crossing(eps, n, x) if gap > 0 else None
    if crossing is not None:
        gap = 0.0
    k_gap = float(np.asarray(data["kpts"])[int(np.argmax(eps[:, n - 1]))][2]) if not name.startswith("graphene") else None
    valence = _distinct(sorted((eps[:, j].max() for j in range(max(0, n - 12), n)), reverse=True))
    conduction = _distinct(sorted(eps[:, j].min() for j in range(n, min(eps.shape[1], n + 12))))
    transitions = [float(c - v) for c, v in zip(conduction, valence)]
    if gap == 0.0:
        # A metal: the crossing bands give 0 (and, sampled, a spurious small value); M11 is
        # the first pair of extrema further out.
        transitions = [0.0] + [t for t in transitions if t > 0.25]
    transitions = transitions[:3] if not name.startswith("graphene") else []
    out = {"gap_eV": gap, "gap_at_kz": k_gap, "transitions_eV": transitions}
    # Fermi velocity: steepest slope of the two bands at the Fermi level where they meet it
    if crossing is not None:
        out["fermi_velocity_eV_A"] = float(max(crossing))
    elif gap < 0.02:
        slopes = []
        for j in (n - 1, n):
            e = eps[:, j]
            for i in range(len(e) - 1):
                if e[i] * e[i + 1] <= 0 and x[i + 1] > x[i]:
                    slopes.append(abs(e[i + 1] - e[i]) / (x[i + 1] - x[i]))
        if slopes:
            out["fermi_velocity_eV_A"] = float(np.max(slopes))
    if name.startswith("graphene"):
        out["pi_bottom_gamma_eV"] = float(_pi_bottom(eps, n))
    out["valence_rel"] = eps[:, max(0, n - BANDS_EACH_SIDE): n][:, ::-1].tolist()   # HOMO first
    out["conduction_rel"] = eps[:, n: n + BANDS_EACH_SIDE].tolist()                 # LUMO first
    return out


def _xaxis(name: str, data: dict) -> np.ndarray:
    atoms = geometry(name)
    _, x, _ = kpath(name, atoms)
    return x


def _pi_bottom(eps: np.ndarray, n: int) -> float:
    """The π band at Γ. With s+p (4 occupied bands per 2 atoms) the occupied levels at Γ are
    the lowest σ, then π, then the degenerate σ pair: π is the second lowest. The π model
    has one occupied band, π itself."""
    levels = np.sort(eps[0, :n])
    return levels[1] if n == 4 else levels[0]


def compare(ref: dict, other: dict) -> dict:
    """Band by band from the HOMO down and from the LUMO up, as many as both have; the RMS
    only where the reference band lies within 1 or 2 eV of the reference level. Further out
    the empty bands of a plane-wave calculation include σ* and nearly-free-electron states
    that no minimal basis has, so a whole-band RMS measures the basis, not the model."""
    diffs, refs = [], []
    for side in ("valence_rel", "conduction_rel"):
        a, b = np.asarray(ref[side]), np.asarray(other[side])
        m = min(a.shape[1], b.shape[1])
        diffs.append((b[:, :m] - a[:, :m]).ravel())
        refs.append(a[:, :m].ravel())
    d, e = np.concatenate(diffs), np.abs(np.concatenate(refs))
    out = {}
    for window in (1.0, 2.0):
        inside = e <= window
        out[f"band_rms_{window:.0f}eV"] = float(np.sqrt((d[inside] ** 2).mean())) if inside.any() else None
    out["gap_error_eV"] = other["gap_eV"] - ref["gap_eV"]
    if "fermi_velocity_eV_A" in ref and "fermi_velocity_eV_A" in other:
        out["fermi_velocity_ratio"] = other["fermi_velocity_eV_A"] / ref["fermi_velocity_eV_A"]
    out["transition_errors_eV"] = [o - r for o, r in zip(other["transitions_eV"],
                                                         ref["transitions_eV"])]
    return out


def report() -> dict:
    systems = {}
    for name in SYSTEMS:
        folder = WORK / _slug(name)
        files = {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")} \
            if folder.exists() else {}
        if "gpaw_pw" not in files:
            continue
        entry = {"reference": "gpaw_pw", "methods": {}}
        ref = metrics(name, files["gpaw_pw"])
        for code, data in sorted(files.items()):
            mm = metrics(name, data)
            row = {k: v for k, v in mm.items() if not k.endswith("_rel")}
            if code != "gpaw_pw":
                row.update(compare(ref, mm))
            row["settings"] = data["settings"]
            entry["methods"][code] = row
        systems[name] = entry
    out = {"description": "Bands of graphene and nanotubes at one geometry per system "
                          "(graphene a = 2.467 Å, tubes rolled from it, unrelaxed), every method "
                          "relative to its own Fermi level or mid-gap. Reference: GPAW PBE plane "
                          "waves. RMS over the 4 highest valence and 4 lowest conduction bands "
                          "(index from the HOMO), at the k points where the reference band is within 1 or 2 eV.",
           "systems": systems}
    _write(OUT, out)
    return out


#: Adjustable with --ajuste NOMBRE=VALOR.
PARAMS = [
    Param("PW_CUTOFF", "energía de corte de las ondas planas de GPAW", "eV", "", "convergencia"),
    Param("KMESH_TUBE", "puntos k del SCF a lo largo del eje de los tubos", "", "", "convergencia"),
    Param("KMESH_GRAPHENE", "malla k del SCF del grafeno (n×n)", "", "", "convergencia"),
    Param("NK_TUBE", "puntos del camino Γ–Z de los tubos", "", "", "salida"),
    Param("VACUUM", "vacío alrededor de cada estructura", "Å", "", "convergencia"),
    Param("KT", "ensanchamiento de Fermi-Dirac", "eV", "", "cálculo"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("gpaw", "tb", "report"))
    parser.add_argument("system", nargs="?",
                        help="sistema del paso gpaw (vacío en tb y report: todos)")
    parser.add_argument("--code", default="pw", choices=("pw", "dzp"),
                        help="GPAW con ondas planas (referencia) o LCAO dzp")
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=WORK)
    if args.step == "gpaw":
        for name in ([args.system] if args.system else SYSTEMS):
            print(run_gpaw(name, args.code), flush=True)
    elif args.step == "tb":
        for name in ([args.system] if args.system else SYSTEMS):
            for model in MODELS:
                print(run_tb(name, model), flush=True)
    else:
        print(json.dumps({n: {c: {k: v for k, v in r.items() if k in ("gap_eV", "band_rms_1eV",
                                                                          "gap_error_eV")}
                              for c, r in e["methods"].items()}
                          for n, e in report()["systems"].items()}, indent=1))


if __name__ == "__main__":
    main()
