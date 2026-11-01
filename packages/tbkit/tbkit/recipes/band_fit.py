"""An s+p carbon set fitted to GPAW plane-wave bands: electronic structure at a given geometry.

Run from packages/tbkit (after ``recipes/band_validation gpaw`` for the systems below)::

    python -m tbkit.recipes.band_validation gpaw graphene@2.42     # strained graphene, -1.9 %
    python -m tbkit.recipes.band_validation gpaw graphene@2.52     # +2.1 %
    python -m tbkit.recipes.band_fit fit                           # writes parameters/c_bandas.json
    python -m tbkit.recipes.band_fit report                        # fit and held-out errors

Why: against GPAW PBE plane waves, Xu's carbon gives graphene's Fermi velocity 25 % low and
the (8,0) tube a 0.98 eV gap where PBE has 0.55 eV (``validation/bands_tb_vs_gpaw.json``).
Xu was fitted to total energies and structures, not to bands, and its hoppings stop at
2.6 Å. This set is fitted to bands only and is meant for electronic properties (bands, DOS,
optical response, resonance energies) at a geometry that comes from elsewhere (Xu, DFT):
it has **no repulsion**, so no energies or forces. ``tbkit.hybrid`` already takes modes from
one calculator and the response from another.

Form: s and p on carbon, non-orthogonal. Each of V_ssσ, V_spσ, V_ppσ, V_ppπ and the four
overlaps is ``v0 exp(-β (d/d0 - 1))`` with d0 = 1.42 Å (one β per hopping, one shared by the
overlaps), out to third neighbours of graphene (2.84 Å) and switched off by a cubic tail
between 3.0 and 3.3 Å. Free: the s-p on-site splitting, 4 hopping amplitudes, 4 hopping β,
4 overlap amplitudes and the overlap β (14 numbers). The p on-site energy is fixed at 0
(bands are compared relative to the Fermi level or mid-gap, so only the splitting matters).

Targets: GPAW PBE plane-wave bands, each method relative to its own Fermi level (mid-gap with
a gap), band by band from the HOMO down and the LUMO up, at every k of the path:

* graphene at a = 2.42, 2.467 and 2.52 Å, valence bands within 9 eV below E_F (σ and π) and
  the π* band up to 3 eV above: the distance dependence comes from the strain;
* tubes (8,0) and (5,5), bands within 2.5 eV of the reference level (curvature).

Held out, never seen by the fit: tubes (10,0), (9,0) and (6,6). The report gives both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..settings import Param, add_arguments, apply
from . import band_validation as bv

OUT = Path("tbkit/parameters/c_bandas.json")
REPORT = Path("validation/c_bandas_fit.json")
FIT_SYSTEMS = ("graphene@2.42", "graphene", "graphene@2.52", "8,0", "5,5")
HELD_OUT = ("10,0", "9,0", "6,6")
GRAPHENE_WINDOW = (-9.0, 3.0)
TUBE_WINDOW = (-2.5, 2.5)
D0, TAIL = 1.42, (3.0, 3.3)
BONDS = ("sss", "sps", "pps", "ppp")
#: Starting point: Xu's amplitudes at 1.42 Å and a β of strained graphene (Pereira 2009).
START = {"es": -6.5, "v": (-5.4, 5.9, 6.4, -2.7), "beta": (3.0, 3.0, 3.0, 3.4),
         "s": (0.13, -0.15, -0.15, 0.12), "beta_s": 3.0}
SOURCE = "ajuste a bandas GPAW PBE de ondas planas (tbkit.recipes.band_fit)"


def unpack(p: np.ndarray) -> dict:
    return {"es": p[0], "v": p[1:5], "beta": p[5:9], "s": p[9:13], "beta_s": p[13]}


def pack(d: dict) -> np.ndarray:
    return np.r_[d["es"], d["v"], d["beta"], d["s"], d["beta_s"]]


def parameter_dict(p: np.ndarray, fit_info: dict | None = None) -> dict:
    """The parameter file for the vector ``p`` (every number with its unit and source)."""
    q = unpack(p)
    names = {"sss": "ssσ", "sps": "spσ", "pps": "ppσ", "ppp": "ppπ"}

    def entry(kind, bond, v0, beta, unit):
        return {"pair": ["C", "C"], "bond": bond,
                "law": {"type": "Exponential", "v0": float(v0), "d0": D0, "beta": float(beta),
                        "cutoff": TAIL[1]},
                "tail": list(TAIL), "unit": unit,
                "description": f"{kind} {names[bond]} en d0 = {D0} Å", "source": SOURCE}

    data = {
        "name": "carbono s+p ajustado a bandas (c_bandas)",
        "reference": "este trabajo: tbkit.recipes.band_fit, bandas GPAW PBE de ondas planas "
                     "de grafeno (tres constantes de red) y de los tubos (8,0) y (5,5)",
        "system": "carbono sp2: grafeno, nanotubos, coils; estructura electrónica a geometría dada",
        "validity": "solo bandas, DOS y respuesta óptica a una geometría dada (de Xu o de DFT): "
                    "sin repulsión, sin energías ni fuerzas; C-C de 1.37 a 1.48 Å; "
                    "electrones a ±3 eV del nivel de Fermi (más lejos, las bandas vacías de "
                    "GPAW tienen estados σ* y casi libres que una base mínima no describe)",
        "units": {"energy": "eV", "length": "Å"},
        "orbitals": {"C": ["s", "px", "py", "pz"]},
        "valence": {"C": 4},
        "onsite": {"C": {
            "s": {"value": float(q["es"]), "unit": "eV", "description": "energía on-site 2s",
                  "source": SOURCE},
            "p": {"value": 0.0, "unit": "eV",
                  "description": "energía on-site 2p (fija: solo importa la diferencia s-p)",
                  "source": "convención de este ajuste"}}},
        "hopping": [entry("V", b, v, be, "eV") for b, v, be in zip(BONDS, q["v"], q["beta"],
                                                                    strict=True)],
        "overlap": [entry("S", b, s, q["beta_s"], "1") for b, s in zip(BONDS, q["s"],
                                                                         strict=True)],
        "notes": "Juego electrónico: úsese para bandas y respuesta con la geometría de otro "
                 "modelo (tbkit.hybrid). Las colas entre 3.0 y 3.3 Å son cúbicas de valor y "
                 "pendiente continuos.",
    }
    if fit_info:
        data["fit"] = fit_info
    return data


def model_of(p: np.ndarray):
    from ..params import model_from_dict

    return model_from_dict(parameter_dict(p))


def tb_bands(name: str, model) -> dict:
    """Bands of ``model`` on the validation path of ``name`` (relative to E_F or mid-gap)."""
    from ..analysis import bands
    from ..hamiltonian import System
    from ..kpoints import mesh
    from ..solver import solve

    atoms = bv.geometry(name)
    system = System.build(atoms, model)
    k0, w0 = mesh(system, bv.KMESH_GRAPHENE if name.startswith("graphene") else bv.KMESH_TUBE)
    k, _, _ = bv.kpath(name, atoms)
    fermi = float(solve(system, k0, w0, kT=bv.KT).fermi)
    eps = bands(system, k)[0]
    n = int(round(system.electrons)) // 2
    return {"bands": np.asarray(eps), "n_occ": n, "fermi_eV": fermi}


def _aligned(data: dict) -> tuple[np.ndarray, int]:
    eps = np.asarray(data["bands"], dtype=float)
    n = int(data["n_occ"])
    return eps - bv.reference_level(eps, n, data["fermi_eV"]), n


def _reference(name: str) -> dict:
    path = bv.WORK / bv._slug(name) / "gpaw_pw.json"
    if not path.exists():
        raise FileNotFoundError(f"Falta la referencia GPAW de {name}: corre "
                                f"python -m tbkit.recipes.band_validation gpaw {name}")
    return json.loads(path.read_text())


def residuals_of(name: str, ref: dict, model) -> np.ndarray:
    """TB − GPAW, band by band from the HOMO/LUMO, where the GPAW band is inside the window."""
    a, n_a = _aligned(ref)
    b, n_b = _aligned(tb_bands(name, model))
    lo, hi = GRAPHENE_WINDOW if name.startswith("graphene") else TUBE_WINDOW
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


def fit() -> dict:
    from scipy.optimize import least_squares

    refs = {name: _reference(name) for name in FIT_SYSTEMS}

    def residuals(p):
        try:
            model = model_of(p)
            parts = [residuals_of(name, refs[name], model) for name in FIT_SYSTEMS]
        except (ValueError, np.linalg.LinAlgError):
            return np.full(sum(len(residuals_of(n, refs[n], model_of(pack(START))))
                               for n in FIT_SYSTEMS), 10.0)
        # each system weighs the same whatever its number of bands and k points
        return np.concatenate([r / np.sqrt(len(r)) for r in parts])

    p0 = pack(START)
    result = least_squares(residuals, p0, x_scale="jac", max_nfev=400)
    info = {"cost_start": float(np.sum(residuals(p0) ** 2)),
            "cost_end": float(np.sum(result.fun ** 2)), "nfev": int(result.nfev),
            "status": int(result.status), "fit_systems": list(FIT_SYSTEMS),
            "held_out": list(HELD_OUT)}
    OUT.write_text(json.dumps(parameter_dict(result.x, info), indent=1, ensure_ascii=False))
    return info


def report() -> dict:
    from ..params import load_parameters

    rows = {}
    for label, model in (("c_bandas", load_parameters(str(OUT))),
                         ("xu_carbon", load_parameters("xu_carbon"))):
        for name in (*FIT_SYSTEMS, *HELD_OUT):
            try:
                ref = _reference(name)
            except FileNotFoundError:
                continue
            r = residuals_of(name, ref, model)
            m_ref = bv.metrics(name, ref)
            tb = tb_bands(name, model)
            tb["bands"] = tb["bands"].tolist()
            tb["kpts"] = ref["kpts"]
            m_tb = bv.metrics(name, tb)
            rows.setdefault(name, {"role": "ajuste" if name in FIT_SYSTEMS else "apartado"})
            rows[name][label] = {"band_rms_eV": float(np.sqrt(np.mean(r ** 2))),
                                 "gap_eV": m_tb["gap_eV"], "gap_gpaw_eV": m_ref["gap_eV"],
                                 "fermi_velocity_eV_A": m_tb.get("fermi_velocity_eV_A"),
                                 "fermi_velocity_gpaw_eV_A": m_ref.get("fermi_velocity_eV_A")}
    out = {"description": "Error de bandas frente a GPAW PBE (ondas planas) en las ventanas del "
                          "ajuste: grafeno −9…+3 eV, tubos ±2.5 eV del nivel de referencia.",
           "systems": rows}
    REPORT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


#: Adjustable with --ajuste NOMBRE=VALOR.
PARAMS = [
    Param("GRAPHENE_WINDOW", "ventana de energías del grafeno en el ajuste", "eV", "", "cálculo"),
    Param("TUBE_WINDOW", "ventana de energías de los tubos en el ajuste", "eV", "", "cálculo"),
]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("fit", "report"))
    add_arguments(parser)
    args = parser.parse_args(argv)
    apply(sys.modules[__name__], args, record=bv.WORK)
    print(json.dumps(fit() if args.step == "fit" else report(), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
