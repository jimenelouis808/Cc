"""Coupling between graphene layers fitted to GPAW: two sets, one for bands, one for energies.

Run from packages/tbkit, after ``recipes/interlayer gpaw`` (and ``band_validation`` for
graphene)::

    python -m tbkit.recipes.interlayer_fit bands      # parameters/c_capas.json
    python -m tbkit.recipes.interlayer_fit energy     # parameters/xu_capas.json
    python -m tbkit.recipes.interlayer_fit report     # validation/interlayer_tb_vs_gpaw.json

Both sets add ``far_hopping`` (``TBModel``): V_ppσ and V_ppπ between atoms more than three
bonds apart, used for those pairs instead of the base set's laws (V_ssσ and V_spσ are zero
there), ``v0 exp(-β (d/3.35 - 1))`` cut between 5.0 and 5.5 Å. The bonds the base set was
fitted with are untouched and a compressed bilayer still couples.

* ``c_capas`` = ``c_bandas`` (held fixed) + far hopping fitted to GPAW PBE bands of AB
  bilayers at 3.0, 3.35 and 3.7 Å, the AA bilayer and AB graphite, with monolayer graphene
  in the fit so the far law cannot spoil it (its first far pairs are in-plane, 4.27 Å).
  Electronic only, like ``c_bandas``.
* ``xu_capas`` = Xu (held fixed) + far hopping fitted to the levels at K of the same systems
  (Xu's in-plane bands are 25 % slow, so whole bands would make the far law compensate them;
  at K a monolayer is exactly degenerate and every splitting is interlayer) + a
  ``LayerDispersionTerm`` fitted to the GPAW PBE+D3(BJ) binding curves of the AB and AA
  bilayers (9 distances each). For energies, forces and relaxations of stacked systems.

Held out, never in a fit: the ABC trilayer (bands), graphite at three c and the sliding
points (energies).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..settings import Param, add_arguments, apply
from . import band_fit
from . import band_validation as bv
from . import interlayer as il

OUT_BANDS = Path("tbkit/parameters/c_capas.json")
OUT_ENERGY = Path("tbkit/parameters/xu_capas.json")
REPORT = Path("validation/interlayer_tb_vs_gpaw.json")
BAND_FIT = ("AB@3.0", "AB@3.35", "AB@3.7", "AA@3.35", "graphite@3.35")
BAND_HELD = ("ABC@3.35",)
ENERGY_FIT = tuple(f"{s}@{d}" for s in ("AB", "AA") for d in il.CURVE_D)
ENERGY_HELD = ("graphite@3.2", "graphite@3.35", "graphite@3.5", "slide@0.5", "slide@1.5",
               "ABC@3.35")
LAYER_WINDOW = (-3.0, 3.0)        # eV around E_F, bands of the layered systems
K_WINDOW = 1.5                    # eV, levels at K used for xu_capas
TAIL = (5.0, 5.5)                 # Å
D_FAR = 3.35                      # Å, d0 of the far laws
KMESH_TB = 24
KT = 0.05
SOURCE = "ajuste a GPAW PBE entre capas de grafeno (tbkit.recipes.interlayer_fit)"
FAR_START = (0.35, 7.0, -0.05, 7.0)               # V_ppσ, β_ppσ, V_ppπ, β_ppπ
FAR_LOWER = (0.0, 1.0, -1.0, 1.0)
FAR_UPPER = (2.0, 20.0, 0.5, 20.0)
DISP_START = (0.02, 3.0, 15.0, 3.0)              # a (eV), α (1/Å), c6 (eV Å⁶), r0 (Å)
DISP_LOWER = (0.0, 0.5, 0.0, 1.0)
DISP_UPPER = (5.0, 10.0, 200.0, 6.0)


# ---------------------------------------------------------------- models
def far_entries(p) -> list[dict]:
    names = {"pps": "ppσ", "ppp": "ppπ"}
    return [{"pair": ["C", "C"], "bond": bond,
             "law": {"type": "Exponential", "v0": float(v), "d0": D_FAR, "beta": float(b),
                     "cutoff": TAIL[1]},
             "tail": list(TAIL), "unit": "eV",
             "description": f"V {names[bond]} entre átomos a más de 3 enlaces, en d0 = {D_FAR} Å",
             "source": SOURCE}
            for bond, v, b in (("pps", p[0], p[1]), ("ppp", p[2], p[3]))]


def _read(name: str) -> dict:
    from ..params import read_parameter_file

    return json.loads(json.dumps(read_parameter_file(name)))


def bands_dict(p, info=None) -> dict:
    data = _read("c_bandas")
    data.pop("fit", None)
    data.update({
        "name": "carbono s+p ajustado a bandas, con acoplamiento entre capas (c_capas)",
        "reference": "este trabajo: c_bandas (tbkit.recipes.band_fit) sin cambios + "
                     "acoplamiento entre capas de tbkit.recipes.interlayer_fit (bandas GPAW PBE "
                     "de bicapas AB y AA y de grafito AB)",
        "system": "carbono sp2 apilado: multicapas, grafito, tubos de varias paredes, coils; "
                  "estructura electrónica a geometría dada",
        "validity": data["validity"] + "; entre capas: 3.0 a 3.7 Å ajustado, más allá "
                    "extrapolado; el acoplamiento lejano no lleva solapamiento",
        "far_hopping": far_entries(p),
        "far": {"bonds": 3, "bond_cut": 1.85},
    })
    if info:
        data["fit"] = info
    return data


def energy_dict(p_far, p_disp=None, info=None) -> dict:
    data = _read("xu_carbon")
    data.update({
        "name": "Xu con acoplamiento y dispersión entre capas (xu_capas)",
        "reference": "Xu et al. 1992 sin cambios + acoplamiento lejano y dispersión de "
                     "tbkit.recipes.interlayer_fit (GPAW PBE y PBE+D3(BJ))",
        "system": "carbono sp2/sp3 apilado: multicapas, grafito, tubos de varias paredes, coils",
        "validity": "lo de Xu, más: separación entre capas 2.8 a 5.5 Å (curvas AB y AA); la "
                    "corrugación AB/AA es la de un término de pares; el término de dispersión "
                    "usa el grafo de enlaces de cada geometría (fijo dentro de una relajación "
                    "sin rupturas)",
        "far_hopping": far_entries(p_far),
        "far": {"bonds": 3, "bond_cut": 1.85},
    })
    if p_disp is not None:
        a, alpha, c6, r0 = (float(x) for x in p_disp)
        data["repulsive"] = {
            "type": "sum", "unit": "eV", "source": SOURCE,
            "description": "repulsión de Xu + pared y dispersión entre átomos a más de 3 enlaces",
            "terms": [data["repulsive"],
                      {"type": "layer_dispersion", "a": a, "alpha": alpha, "c6": c6, "r0": r0,
                       "tail": [9.0, 11.0], "bond_cut": 1.85, "exclude": 3,
                       "units": {"a": "eV", "alpha": "1/Å", "c6": "eV Å⁶", "r0": "Å"},
                       "source": SOURCE}]}
    if info:
        data["fit"] = info
    return data


def _model(data: dict):
    from ..params import model_from_dict

    return model_from_dict(data)


# ---------------------------------------------------------------- TB on the layer systems
def _reference(name: str) -> dict:
    return il.load(name)


def tb_bands(name: str, model) -> dict:
    from ..analysis import bands
    from ..hamiltonian import System
    from ..kpoints import mesh
    from ..solver import solve

    atoms = il.geometry(name)
    system = System.build(atoms, model)
    k0, w0 = mesh(system, (KMESH_TB, KMESH_TB, 8))
    k, _, _ = il.kpath(atoms)
    fermi = float(solve(system, k0, w0, kT=KT).fermi)
    return {"bands": np.asarray(bands(system, k)[0]),
            "n_occ": int(round(system.electrons)) // 2, "fermi_eV": fermi}


def _aligned(data: dict):
    eps = np.asarray(data["bands"], dtype=float)
    n = int(data["n_occ"])
    return eps - bv.reference_level(eps, n, data["fermi_eV"]), n


def band_residuals(ref: dict, tb: dict, window) -> np.ndarray:
    a, n_a = _aligned(ref)
    b, n_b = _aligned(tb)
    lo, hi = window
    out = []
    for side in (-1, 1):
        m = 0
        while True:
            ia = n_a - 1 - m if side < 0 else n_a + m
            ib = n_b - 1 - m if side < 0 else n_b + m
            if ia < 0 or ib < 0 or ia >= a.shape[1] or ib >= b.shape[1]:
                break
            inside = (a[:, ia] >= lo) & (a[:, ia] <= hi)
            if not inside.any():
                break
            out.append((b[:, ib] - a[:, ia])[inside])
            m += 1
    return np.concatenate(out)


def k_index(name: str) -> int:
    _, x, labels = il.kpath(il.geometry(name))
    xk = next(pos for pos, lab in labels if lab == "K")
    return int(np.argmin(np.abs(np.asarray(x) - xk)))


def k_levels(data: dict, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Levels at K within ``K_WINDOW`` of the reference level, below and above it, each
    ordered away from it."""
    eps, n = _aligned(data)
    at_k = eps[index]
    below = at_k[:n][::-1]
    above = at_k[n:]
    return below[below > -K_WINDOW], above[above < K_WINDOW]


def k_residuals(ref: dict, tb: dict, index: int) -> np.ndarray:
    rb, ra = k_levels(ref, index)
    eps, n = _aligned(tb)
    full_b, full_a = eps[index][:n][::-1], eps[index][n:]
    return np.r_[full_b[:len(rb)] - rb, full_a[:len(ra)] - ra]


# ---------------------------------------------------------------- energies
def tb_energy(name: str, model) -> float:
    from ..forces import energy_and_forces
    from ..hamiltonian import System
    from ..kpoints import mesh
    from ..solver import solve

    system = System.build(il.geometry(name), model)
    k, w = mesh(system, (KMESH_TB, KMESH_TB, 8))
    return float(energy_and_forces(solve(system, k, w, kT=KT), need_forces=False)[0])


def far_pairs(name: str) -> np.ndarray:
    """Distances of the ordered far pairs within the dispersion cutoff."""
    from ..neighbors import far_mask, neighbor_list

    atoms = il.geometry(name)
    ii, jj, dd, ss = neighbor_list("ijdS", atoms, 11.0)
    return dd[far_mask(atoms, ii, jj, ss, 1.85, 3)]


def dispersion_energy(p, distances) -> float:
    from ..repulsive import LayerDispersionTerm

    return 0.5 * float(LayerDispersionTerm(*p).pair(distances)[0].sum())


def target(name: str, mono: dict) -> float:
    return il.binding(name, mono=mono)["pbe_d3"]


# ---------------------------------------------------------------- fits
def fit_bands() -> dict:
    from scipy.optimize import least_squares

    refs = {n: _reference(n) for n in BAND_FIT}
    graphene_ref = band_fit._reference("graphene")

    def residuals(p):
        model = _model(bands_dict(p))
        parts = [band_residuals(refs[n], tb_bands(n, model), LAYER_WINDOW) for n in BAND_FIT]
        parts.append(band_fit.residuals_of("graphene", graphene_ref, model))
        return np.concatenate([r / np.sqrt(len(r)) for r in parts])

    result = least_squares(residuals, np.array(FAR_START), x_scale="jac", max_nfev=200,
                           bounds=(FAR_LOWER, FAR_UPPER))
    info = {"free": ["V_ppσ", "β_ppσ", "V_ppπ", "β_ppπ"],
            "bounds": {"lower": list(FAR_LOWER), "upper": list(FAR_UPPER)},
            "at_bound": [int(i) for i in np.flatnonzero(result.active_mask)],
            "cost_start": float(np.sum(residuals(np.array(FAR_START)) ** 2)),
            "cost_end": float(np.sum(result.fun ** 2)), "nfev": int(result.nfev),
            "fit_systems": list(BAND_FIT) + ["graphene"], "held_out": list(BAND_HELD),
            "window_eV": list(LAYER_WINDOW)}
    OUT_BANDS.write_text(json.dumps(bands_dict(result.x, info), indent=1, ensure_ascii=False))
    return info


def fit_energy() -> dict:
    from scipy.optimize import least_squares

    refs = {n: _reference(n) for n in BAND_FIT}
    index = {n: k_index(n) for n in BAND_FIT}

    def k_res(p):
        model = _model(energy_dict(p))
        return np.concatenate([k_residuals(refs[n], tb_bands(n, model), index[n])
                               for n in BAND_FIT])

    far = least_squares(k_res, np.array(FAR_START), x_scale="jac", max_nfev=200,
                        bounds=(FAR_LOWER, FAR_UPPER))
    # dispersion: the electronic part of the binding is fixed by the far law above
    mono_ref = il.load("mono")
    e_mono = tb_energy("mono", _model(energy_dict(far.x)))
    electronic, pairs, targets, sizes = {}, {}, {}, {}
    for name in ENERGY_FIT:
        n = il.load(name)["n_atoms"]
        electronic[name] = 1000 * (tb_energy(name, _model(energy_dict(far.x))) - n / 2 * e_mono) / n
        pairs[name] = far_pairs(name)
        targets[name] = target(name, mono_ref)
        sizes[name] = n
    mono_pairs = far_pairs("mono")

    def e_res(q):
        out = []
        e1 = dispersion_energy(q, mono_pairs)
        for name in ENERGY_FIT:
            n = sizes[name]
            disp = 1000 * (dispersion_energy(q, pairs[name]) - n / 2 * e1) / n
            out.append(electronic[name] + disp - targets[name])
        return np.array(out)

    disp = least_squares(e_res, np.array(DISP_START), x_scale="jac", max_nfev=2000,
                         bounds=(DISP_LOWER, DISP_UPPER))
    info = {"far": {"free": ["V_ppσ", "β_ppσ", "V_ppπ", "β_ppπ"],
                    "at_bound": [int(i) for i in np.flatnonzero(far.active_mask)],
                    "rms_k_levels_eV": float(np.sqrt(np.mean(far.fun ** 2))),
                    "fit_systems": list(BAND_FIT), "k_window_eV": K_WINDOW},
            "dispersion": {"free": ["a", "alpha", "c6", "r0"],
                           "at_bound": [int(i) for i in np.flatnonzero(disp.active_mask)],
                           "rms_meV_per_atom": float(np.sqrt(np.mean(disp.fun ** 2))),
                           "fit_systems": list(ENERGY_FIT),
                           "target": "GPAW PBE + D3(BJ) binding per atom"},
            "held_out": list(BAND_HELD + ENERGY_HELD)}
    OUT_ENERGY.write_text(json.dumps(energy_dict(far.x, disp.x, info), indent=1,
                                     ensure_ascii=False))
    return info


# ---------------------------------------------------------------- report
def _equilibrium(ds, es) -> tuple[float, float]:
    """Minimum of a cubic spline through the curve (Å, meV/atom)."""
    from scipy.interpolate import CubicSpline

    spline = CubicSpline(ds, es)
    fine = np.linspace(min(ds), max(ds), 2001)
    i = int(np.argmin(spline(fine)))
    return float(fine[i]), float(spline(fine[i]))


def report() -> dict:
    from ..params import load_parameters

    models = {"c_bandas": load_parameters("c_bandas"), "c_capas": load_parameters(str(OUT_BANDS)),
              "xu_carbon": load_parameters("xu_carbon"),
              "xu_capas": load_parameters(str(OUT_ENERGY))}
    out = {"description": "Bandas y enlace entre capas de grafeno frente a GPAW PBE (ondas "
                          "planas; energías PBE+D3(BJ)). RMS de bandas en ±3 eV de E_F; niveles "
                          f"en K a ±{K_WINDOW} eV de E_F.", "bands": {}, "binding": {}}
    graphene_ref = band_fit._reference("graphene")
    for name in BAND_FIT + BAND_HELD + ("graphene",):
        row = {"role": "ajuste" if name in BAND_FIT + ("graphene",) else "fuera"}
        if name == "graphene":
            for label, model in models.items():
                r = band_fit.residuals_of(name, graphene_ref, model)
                row[label] = {"band_rms_eV": float(np.sqrt(np.mean(r ** 2)))}
        else:
            ref, index = _reference(name), k_index(name)
            rb, ra = k_levels(ref, index)
            row["k_levels_gpaw_eV"] = sorted(rb.tolist() + ra.tolist())
            for label, model in models.items():
                tb = tb_bands(name, model)
                r = band_residuals(ref, tb, LAYER_WINDOW)
                kr = k_residuals(ref, tb, index)
                eps, n = _aligned(tb)
                row[label] = {"band_rms_eV": float(np.sqrt(np.mean(r ** 2))),
                              "k_levels_rms_eV": float(np.sqrt(np.mean(kr ** 2))),
                              "k_levels_eV": sorted((kr + np.r_[rb, ra]).tolist())}
        out["bands"][name] = row
    mono = il.load("mono")
    xu = models["xu_capas"]
    e_mono = tb_energy("mono", xu)
    curves = {}
    for name in ENERGY_FIT + ENERGY_HELD:
        b = il.binding(name, mono=mono)
        n = il.load(name)["n_atoms"]
        tb = 1000 * (tb_energy(name, xu) - n / 2 * e_mono) / n
        out["binding"][name] = {"role": "ajuste" if name in ENERGY_FIT else "fuera",
                                "pbe_meV": b["pbe"], "pbe_d3_meV": b["pbe_d3"],
                                "xu_capas_meV": tb}
    for stacking in ("AB", "AA"):
        ds = list(il.CURVE_D)
        for label, key in (("pbe_d3", "pbe_d3_meV"), ("xu_capas", "xu_capas_meV")):
            es = [out["binding"][f"{stacking}@{d}"][key] for d in ds]
            d0, e0 = _equilibrium(ds, es)
            curves.setdefault(stacking, {})[label] = {"d_eq_A": d0, "binding_meV": e0}
    out["equilibrium"] = curves
    REPORT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


PARAMS = [
    Param("LAYER_WINDOW", "ventana de energías de las bandas de bicapas y grafito", "eV", "",
          "cálculo"),
    Param("K_WINDOW", "niveles en K usados para xu_capas (± respecto a E_F)", "eV", "",
          "cálculo"),
    Param("KMESH_TB", "malla k del TB en el plano (n×n)", "", "", "convergencia"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("bands", "energy", "report"))
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=il.WORK)
    result = {"bands": fit_bands, "energy": fit_energy, "report": report}[args.step]()
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
