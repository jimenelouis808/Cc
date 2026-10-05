"""docs/VALIDACION.md, written from the stored results (never by hand).

Run::

    python -m tbkit.recipes.validation_report            # rewrites docs/VALIDACION.md

Every number in the document comes from a file of ``validation/`` or of
``tbkit/parameters/``; a test regenerates it and fails if the committed copy
differs, so the document cannot drift from the data. To change a number,
rerun the recipe that produced the file, then this one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "validation"
PARAMETERS = ROOT / "tbkit" / "parameters"
TARGET = ROOT / "docs" / "VALIDACION.md"

SETS = ("pi_huckel", "xu_carbon", "tang_carbon", "xu_chn", "xu_chno", "xu_chnob", "xu_chnos",
        "xu_chnop", "xu_chnose")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value, digits=2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def _table(header, rows) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _sets() -> list[str]:
    out = ["## Conjuntos de parámetros", "",
           "Lo que cada archivo declara (`system`, `validity`) y su factor de escala de "
           "frecuencias frente a GPAW (`frequency_scale`, receta `frequency_scaling`). "
           "Fuente: `tbkit/parameters/*.json`.", ""]
    rows = []
    for name in SETS:
        data = _load(PARAMETERS / f"{name}.json")
        scale = data.get("frequency_scale")
        rows.append((f"`{name}`", "sí" if data.get("scc") else "no",
                     _f(scale["value"], 4) if scale else "—",
                     f"{_f(scale['rms_before_cm1'], 1)} → {_f(scale['rms_after_cm1'], 1)}"
                     if scale else "—",
                     f"{scale['modes']} / {scale['molecules']}" if scale else "—"))
    out += _table(("conjunto", "SCC", "λ frecuencias", "RMS (cm⁻¹) sin → con λ",
                   "modos / moléculas"), rows)
    out += ["", "Validez declarada por cada archivo:", ""]
    for name in SETS:
        data = _load(PARAMETERS / f"{name}.json")
        out += [f"- **`{name}`** ({data.get('system', '')}): {data.get('validity', '')}"]
    return out + [""]


def _infrared() -> list[str]:
    data = _load(VALIDATION / "ir_tb_vs_gpaw.json")
    out = ["## Infrarrojo frente a GPAW (8 moléculas C/H/N, modos de GPAW)", "",
           "Fuente: `validation/ir_tb_vs_gpaw.json`. log₁₀(TB/GPAW) de las intensidades, "
           "modos internos (ω > 100 cm⁻¹: fuera las rotaciones y traslaciones, cuya intensidad "
           "en GPAW es ruido numérico) con más del 5 % del más intenso de su molécula.", ""]
    stats = {}
    for key in ("charges", "onsite"):
        logs = []
        for mol in data.values():
            gpaw, tb = np.array(mol["gpaw"]), np.array(mol[key])
            keep = (gpaw > 0.05 * gpaw.max()) & (np.array(mol["freq"]) > 100)
            logs += list(np.log10(tb[keep] / gpaw[keep]))
        logs = np.array(logs)
        stats[key] = (logs.mean(), np.median(np.abs(logs)), len(logs))
    out += _table(("", "solo cargas", "cargas + dipolos intraatómicos"),
                  [("media de log₁₀", _f(stats["charges"][0]), _f(stats["onsite"][0])),
                   ("mediana de \\|log₁₀\\|", _f(stats["charges"][1]), _f(stats["onsite"][1])),
                   ("modos", stats["charges"][2], stats["onsite"][2])])
    return out + [""]


def _crystals() -> list[str]:
    data = _load(VALIDATION / "crystals_tb_vs_gpaw.json")
    s = data["settings"]
    out = ["## Cristales frente a GPAW (extrapolación de conjuntos ajustados en moléculas)", "",
           f"Fuente: `validation/crystals_tb_vs_gpaw.json` (GPAW {s['xc']} {s['mode']} "
           f"{s['basis']}, h = {s['h']} Å). Errores de fuerza RMS en eV/Å; «razón» = error "
           "en el cristal / mediana del error molecular del mismo conjunto medido igual.", ""]
    rows = []
    for c in data["crystals"]:
        ratio = c.get("ratio_to_molecules", {})
        rows.append((c["crystal"], f"`{c['set']}`", c["atoms"],
                     _f(c.get("force_rmse_at_minimum"), 3),
                     _f(c.get("force_rmse_distorted_median"), 3),
                     _f(ratio.get("at_minimum")), _f(ratio.get("distorted")),
                     _f(100 * (c["lattice_scale_model"] - 1), 2) if "lattice_scale_model" in c
                     else "—",
                     _f(100 * (c["lattice_scale_dft"] - 1), 2) if "lattice_scale_dft" in c
                     else "—"))
    out += _table(("cristal", "conjunto", "átomos", "F mínimo", "F distorsionado (mediana)",
                   "razón mínimo", "razón distorsionado", "red TB (%)", "red GPAW (%)"), rows)
    out += ["", "Error molecular de referencia (medianas por estructura):", ""]
    rows = [(f"`{name}`", _f(b["eq"]["force_rmse_median"], 3), _f(b["rnd"]["force_rmse_median"], 3))
            for name, b in data["molecular_baseline"].items()]
    out += _table(("conjunto", "F en el mínimo", "F distorsionado"), rows)
    return out + [""]


def _tang() -> list[str]:
    data = _load(PARAMETERS / "tang_carbon.json")
    fit = data["fit"]
    out = ["## Carbono dependiente del entorno (`tang_carbon`) frente a GPAW", "",
           f"Fuente: `tbkit/parameters/tang_carbon.json` (`fit`). Fuera del ajuste: "
           f"{', '.join(fit['held_out'])}.", ""]
    rows = [(g, _f(e["force_rmse"]), _f(e["force_rms_dft"]), _f(e["energy_mae_per_atom"], 3),
             "fuera" if g in fit["held_out"] else "ajuste")
            for g, e in fit["errors"].items()]
    out += _table(("estructura", "F RMSE (eV/Å)", "F RMS GPAW", "\\|ΔE\\| (eV/átomo)", ""), rows)
    return out + [""]


def _nanotubes() -> list[str]:
    data = _load(VALIDATION / "cnt_xu_carbon.json")
    out = ["## Nanotubos prístinos con Xu", "",
           "Fuente: `validation/cnt_xu_carbon.json`. RBM frente a las relaciones empíricas "
           "de Araujo y de Jorio; G⁺/G⁻ y su desdoblamiento frente a la referencia guardada. "
           "Malla k: la de la receta (`cnt_validation --kmesh`, 16 por defecto, kT 0,03 eV); el "
           "archivo no la guarda. Los tubos sin gap no se han comprobado en k después del "
           "hallazgo del paso 8 (`modes.GAPLESS_WARNING`): sus G pueden estar ablandadas por "
           "la malla, además de faltarles la anomalía de Kohn física.", ""]
    rows = [(t["tube"], t["kind"], _f(t["diameter_nm"], 3), _f(t["rbm"], 0),
             _f(t["rbm_araujo"], 0), _f(t["rbm_jorio"], 0), _f(t["g_plus"], 0),
             _f(t["g_minus"], 0), _f(t["g_split"], 0), _f(t.get("g_split_reference"), 0))
            for t in data]
    out += _table(("tubo", "tipo", "d (nm)", "RBM", "Araujo", "Jorio", "G⁺", "G⁻", "ΔG",
                   "ΔG ref."), rows)
    return out + [""]


def _graphene() -> list[str]:
    data = _load(VALIDATION / "graphene_2d_gpaw.json")
    out = ["## Grafeno: G, 2D y 2D′ por doble resonancia", "",
           f"Fuente: `validation/graphene_2d_gpaw.json` (fonones: {data['phonons']}; "
           f"electrones: {data['electrons']}; γ = {data['gamma_ev']} eV).", ""]
    rows = [(_f(r["laser_ev"]), _f(r["g_frequency"], 0), _f(r["2D_position"], 0),
             _f(r["2D'_position"], 0), _f(r["2D_intensity"] / r["g_intensity"], 1))
            for r in data["results"]]
    out += _table(("láser (eV)", "G", "2D", "2D′", "I(2D)/I(G)"), rows)
    first, last = data["results"][0], data["results"][-1]
    slope = (last["2D_position"] - first["2D_position"]) / (last["laser_ev"] - first["laser_ev"])
    out += ["", f"Dispersión de la 2D: {_f(slope, 0)} cm⁻¹/eV."]
    return out + [""]


def _nanocoil() -> list[str]:
    data = _load(VALIDATION / "nanocoil204_tb.json")
    p3 = data["gpaw_phase3"]
    out = ["## Nanocoil periódica de 204 átomos (anillos 5–7)", "",
           "Fuente: `validation/nanocoil204_tb.json`.", ""]
    rows = []
    for name in ("xu_carbon", "tang_carbon"):
        g = p3["geometry_vs_pbe"][name]
        census = ", ".join(f"{k}: {v}" for k, v in data[name]["ring_census"].items())
        rows.append((f"`{name}`", data[name]["topology"], census,
                     _f(data[name]["gap_eV"]), _f(g["bond_rms_A"], 3), _f(g["bond_mean_A"], 3)))
    out += _table(("conjunto", "topología", "anillos", "gap TB (eV)", "enlaces vs PBE RMS (Å)",
                   "media (Å)"), rows)
    out += ["", f"Gap PBE: {_f(p3['pbe_gap_eV'])} eV ({p3['pbe_gap_note']}).", "",
            "Frecuencias de modos TB con las constantes de fuerza PBE (cociente de Rayleigh, "
            "cota superior):", ""]
    rows = [(f"`{m['model']}`", m["label"], _f(m["tb_frequency_cm1"], 0),
             _f(m["pbe_projected_cm1"], 0)) for m in p3["projected_modes"]]
    out += _table(("conjunto", "modo", "TB (cm⁻¹)", "PBE (cm⁻¹)"), rows)
    raman = data.get("raman_resonant_tang")
    if raman:
        conv = raman["convergence_top20_modes"]
        out += ["", "Raman resonante (Tang): picos a 532 nm (ensanchamiento 60 cm⁻¹): " +
                ", ".join(str(p) for p in raman["peaks_532nm_fwhm60_cm1"]) + " cm⁻¹. "
                f"Actividades de los 20 modos más intensos: ×{_f(conv['kmesh_4_to_8_activity_ratio'][0])}"
                f"–{_f(conv['kmesh_4_to_8_activity_ratio'][1])} al pasar de 4 a 8 k; "
                f"×{_f(conv['eta_0.1_to_0.2_activity_ratio'][0])}–"
                f"{_f(conv['eta_0.1_to_0.2_activity_ratio'][1])} al pasar η de 0,1 a 0,2 eV "
                "(posiciones robustas, alturas relativas no)."]
    return out + [""]


def render() -> str:
    lines = ["# Validación de tbkit", "",
             "Generado por `python -m tbkit.recipes.validation_report` a partir de los "
             "archivos de `validation/` y `tbkit/parameters/`. No editar a mano: un test "
             "comprueba que coincide con los datos.", ""]
    for part in (_sets, _infrared, _crystals, _tang, _nanotubes, _graphene, _nanocoil):
        lines += part()
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(render(), encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
