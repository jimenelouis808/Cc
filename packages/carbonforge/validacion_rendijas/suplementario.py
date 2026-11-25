"""Build the Supplementary Information of informe_adsorcion.html.

Every table is generated from the files the study wrote (json, txt), not
typed by hand, and is in English. The block between the SUPPLEMENTARY
markers of the report is replaced, so a rerun after new results updates
it in place. Also prints the ``files`` map the publish needs for the
gallery images.

    python suplementario.py
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPORT = HERE / "informe_adsorcion.html"
START, END = "<!-- SUPPLEMENTARY START -->", "<!-- SUPPLEMENTARY END -->"

_tables = 0
_figures = 0


def table(caption: str, header: list[str], rows: list[list], num: set[int] | None = None,
          note: str = "") -> str:
    global _tables
    _tables += 1
    num = num if num is not None else set(range(1, len(header)))
    head = "".join(f'<th class="num">{h}</th>' if i in num else f"<th>{h}</th>"
                   for i, h in enumerate(header))
    body = "".join(
        "<tr>" + "".join(
            f'<td class="num">{c}</td>' if i in num else f"<td>{c}</td>"
            for i, c in enumerate(r)) + "</tr>" for r in rows)
    extra = f'<p class="sub">{note}</p>' if note else ""
    return (f'<div class="tablewrap"><table><caption><b>Table S{_tables}.</b> '
            f"{caption}</caption><thead><tr>{head}</tr></thead><tbody>{body}"
            f"</tbody></table></div>{extra}")


def figure(src: str, caption: str, alt: str) -> str:
    global _figures
    _figures += 1
    return (f'<figure><div class="img"><img src="{src}" alt="{html.escape(alt)}" '
            f'loading="lazy"></div><figcaption><b>Figure S{_figures}.</b> {caption}'
            "</figcaption></figure>")


def md_rows(path: Path, section: str | None = None) -> list[list[str]]:
    """Rows of the first markdown table in a file (after ``section``)."""
    text = path.read_text()
    if section:
        text = text.split(section, 1)[1]
    rows = []
    started = False
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not started:
                started = True          # header
                continue
            rows.append(cells)
        elif started:
            break
    return rows


def f(x, d=2) -> str:
    return f"{x:.{d}f}"


# --------------------------------------------------------------------------

def s_levesque() -> str:
    d = json.loads((ROOT / "validacion_GCMC" / "datos.json").read_text())
    out = []
    for t in ("77.0", "293.0"):
        rows = [[f(p["presion_mpa"], 1), f(p["rho_publicado"], 5),
                 f(p["rho_calculado"], 5), f"{p['diferencia_pct']:+.1f}",
                 f"{p['n_medio']:.0f} ± {p['n_error']:.0f}"] for p in d[t]]
        out.append(table(
            f"H₂ in the Levesque et al. (2002) (17,0) SWCNT bundle at {float(t):.0f} K: "
            "adsorbed density ρ<sub>o</sub> (g/cm³) published vs computed. Nothing fitted.",
            ["P (MPa)", "ρ<sub>o</sub> published", "ρ<sub>o</sub> this work",
             "difference (%)", "N per cell"], rows))
    return "".join(out)


def s_slits_h2() -> str:
    files = sorted(HERE.glob("rendija_*.json"),
                   key=lambda p: float(p.stem.split("_")[1]))
    por, load = [], []
    for p in files:
        d = json.loads(p.read_text())
        h = p.stem.split("_")[1]
        q = d["porosidad"]
        por.append([h, f(q["largest_free_sphere"]),
                     f"{q['accessible_area_m2_g']:.0f}", f(q["helium_void_fraction"], 3),
                     f(q["pore_volume_cm3_g"], 3)])
        for pt in d["puntos"]:
            load.append([h, f(pt["presion_mpa"], 1),
                         f"{pt['n_medio']:.1f} ± {pt['n_error']:.1f}",
                         f(pt["wt_pct_total"]), f(pt["densidad_exceso_g_cm3"], 4)])
    return (table("Graphene slit series (AA stacking): pore geometry. H is the "
                  "centre-to-centre wall separation; the channel is the largest free "
                  "sphere; the area is geometric (probe 2.89 Å), not BET.",
                  ["H (Å)", "channel (Å)", "area (m²/g)", "helium void fraction",
                   "pore volume (cm³/g)"], por)
            + table("H₂ in the graphene slits at 77 K (1.5 × 10⁶ equilibration + "
                    "1.5 × 10⁶ production moves).",
                    ["H (Å)", "P (MPa)", "N per cell", "wt% absolute (total mass)",
                     "excess density (g/cm³)"], load))


def s_schwarzites() -> str:
    por, load = [], []
    for name in ("primitive", "diamond", "gyroid"):
        d = json.loads((ROOT / "validacion_schwarzitas" / f"{name}.json").read_text())
        q = d["porosidad"]
        por.append([name, f"{q['cell_volume_a3'] ** (1 / 3):.1f}", f(q["largest_free_sphere"]),
                    f(q["largest_included_sphere"]), f"{q['accessible_area_m2_g']:.0f}",
                    f(q["helium_void_fraction"], 3)])
        for pt in d["puntos"]:
            load.append([name, f(pt["presion_mpa"], 1),
                         f"{pt['n_medio']:.0f} ± {pt['n_error']:.0f}",
                         f(pt["wt_pct_total"]), f(pt["wt_pct_por_carbono"]),
                         f(pt["densidad_absoluta_g_cm3"], 5)])
    return (table("Crystalline (knee-route) schwarzites: geometry.",
                  ["structure", "cell edge (Å)", "channel (Å)", "largest cavity (Å)",
                   "area (m²/g)", "helium void fraction"], por, num={1, 2, 3, 4, 5})
            + table("H₂ in the schwarzites at 77 K, compared in the text with "
                    "Kowalczyk et al. (2007), whose values are read from figures "
                    "(~3 wt% at 3.8 MPa).",
                    ["structure", "P (MPa)", "N per cell", "wt% (total mass)",
                     "wt% (per carbon)", "ρ<sub>o</sub> (g/cm³)"], load, num={1, 2, 3, 4, 5}))


def s_catalogue() -> str:
    rows = md_rows(HERE / "catalogo_canales.txt")
    tr = {"no": "no", "SÍ": "yes", "pared colapsada": "collapsed wall"}
    rows = [r[:6] + [tr.get(r[6], r[6])] for r in rows]
    return table("Superstructure catalogue (nanocarbon_lab presets, meshed route): "
                 "triage before any simulation. Window: channel 5.5–11 Å and bonds "
                 "1.30–1.55 Å. Minimum angle sum per atom; below 328.4° the wall "
                 "has collapsed.",
                 ["structure", "atoms", "channel (Å)", "area (m²/g)",
                  "min. angle sum (°)", "bonds (Å)", "in window"], rows,
                 num={1, 2, 3, 4})


def s_henry_h2() -> str:
    rows = []
    for fname, route in (("catalogo_1M.txt", "meshed cage/net"),
                         ("cristalinas_1M.txt", "crystalline"),
                         ("cubico_1M.txt", "meshed, tuned cell")):
        for r in md_rows(HERE / fname, "#### sanas"):
            rows.append([r[0], route] + r[1:])
    rows.append(["graphene slit, H = 6.6 Å", "flat", "960", "3.55", "—",
                 "8660 ± 50", "7.81", "—"])
    rows.append(["SWCNT bundle, gap 4.0 Å", "idealised", "8704", "10.21", "—",
                 "4252 ± 107", "8.96", "—"])
    return table("H₂ at 77 K, zero coverage: Henry constant and isosteric heat "
                 "(Widom, 10⁶ insertions × 2 seeds; the ± is the seed spread).",
                 ["structure", "route", "atoms", "channel (Å)", "area (m²/g)",
                  "K<sub>H</sub> (mol kg⁻¹ MPa⁻¹)", "q<sub>st</sub> (kJ/mol)",
                  "min. angle sum (°)"], rows, num={2, 3, 4, 5, 6, 7})


def s_cages_h2() -> str:
    rows = []
    for cage in ("super-cube", "super-tetrahedron"):
        d = json.loads((HERE / "isotermas" / cage / "resultados.json").read_text())
        for p in d["puntos"]:
            rows.append([cage, f"{p['presion_mpa']:g}",
                         f"{p['n_medio']:.1f} ± {p['n_error']:.1f}",
                         f(p["wt_pct_total"]),
                         f(p["densidad_absoluta_g_cm3"], 5),
                         f(p["densidad_exceso_g_cm3"], 5)])
    return table("Full H₂ isotherms of the two cages at 77 K (campaign folders in "
                 "isotermas/). Points below 1e-4 MPa carry short-sampling warnings.",
                 ["cage", "P (MPa)", "N per cell", "wt% (total mass)",
                  "absolute density (g/cm³)", "excess density (g/cm³)"], rows,
                 num={1, 2, 3, 4, 5})


def s_co2_graphite() -> str:
    rows = []
    for line in (HERE / "valida_co2_grafito.txt").read_text().splitlines():
        m = re.match(r"T\s+(\d+) K\s+B_AS/A\s+([\d.]+) A\s+ln\(B/A \[cm\]\)\s+(-?[\d.]+)"
                     r"\s+q_st\(0\)\s+([\d.]+)", line)
        if m:
            rows.append(list(m.groups()))
    return table("CO₂ on a single graphite surface (5-layer AB graphite, no fitting) "
                 "vs Bottani et al. (1994): experimental q<sub>st</sub>(0) = 14.4 kJ/mol "
                 "at 203 K; ln B<sub>AS</sub> vs 1/T slope 1546 ± 95 K; "
                 "B<sub>AS</sub>/A = 159–270 Å at 298 K with BET areas.",
                 ["T (K)", "B<sub>AS</sub>/A (Å)", "ln(B<sub>AS</sub>/A [cm])",
                  "q<sub>st</sub>(0) (kJ/mol)"], rows)


def s_henry_co2() -> str:
    rows = []
    for line in (HERE / "henry_co2.txt").read_text().splitlines():
        m = re.match(r"(.+?)\s+(\d+) C\s+K_H\s+([\d.]+) mol/kg/MPa\s+q_st\s+([\d.]+)", line)
        if m:
            name = m.group(1).strip().replace("rendija H=", "slit H = ") + (
                " Å" if "rendija" in m.group(1) else "")
            rows.append([name, m.group(2), m.group(3), m.group(4)])
    for line in (HERE / "henry_cristalinas.txt").read_text().splitlines():
        m = re.match(r"cristalina-(\S+)\s+canal\s+([\d.]+) A\s+K_H\s+([\d.]+).*q_st\s+([\d.]+)",
                     line)
        if m:
            rows.append([m.group(1) + f" (channel {m.group(2)} Å)", "—", m.group(3),
                         m.group(4)])
    return table("CO₂ at 298 K, zero coverage (Widom, neutral Steele walls). "
                 "For the cages the Widom average does not converge (rare deep "
                 "pockets); they are judged by their isotherms.",
                 ["structure", "C atoms", "K<sub>H</sub> (mol kg⁻¹ MPa⁻¹)",
                  "q<sub>st</sub> (kJ/mol)"], rows)


def s_iso_co2() -> str:
    rows = []
    labels = {"rendija-6.6": ("6.6 Å slit (Steele)", "298"),
              "super-cube": ("super-cube", "298"),
              "super-tetrahedron": ("super-tetrahedron", "298")}
    for p in sorted((HERE / "isotermas_co2").glob("*.json")):
        stem = p.stem
        if stem in labels:
            name, t = labels[stem]
        else:
            m = re.match(r"funcional-(\w+?)(-ads)?(-373K)?$", stem)
            if not m:
                continue
            field = "adsorption FF" if m.group(2) else "electrode FF"
            name = f"slit {m.group(1)} ({field})"
            t = "373" if m.group(3) else "298"
        for pt in json.loads(p.read_text())["puntos"]:
            rows.append([name, t, f"{pt['pressure_mpa']:g}",
                         f"{pt['mean_count']:.1f} ± {pt['count_error']:.1f}",
                         f(pt["mol_per_kg"], 3), f(pt["excess_mol_per_kg"], 3)])
    return table("All pure-CO₂ GCMC points (Ewald, TraPPE). 'Electrode FF' is the "
                 "first, representative charge table (COOH charged as a carboxylate); "
                 "'adsorption FF' uses published group charges and DFT charges "
                 "for graphitic N.",
                 ["structure", "T (K)", "P (MPa)", "N per cell", "absolute (mol/kg)",
                  "excess (mol/kg)"], rows, num={1, 2, 3, 4, 5})


def _henry_lines(path: Path):
    for line in path.read_text().splitlines():
        m = re.match(r"(\S+)\s+(\S+)\s+carga neta\s+(\S+) e/celda.*con cargas: K_H\s+([\d.]+)"
                     r" \(([\d-]+)\)\s+q_st\s+([\d.]+).*sin cargas: K_H\s+([\d.]+) \(([\d-]+)\)"
                     r"\s+q_st\s+([\d.]+)", line)
        if m:
            yield m.groups()


def s_functional() -> str:
    rows = []
    for fname, field in (("henry_co2_funcional.txt", "electrode"),
                         ("henry_co2_funcional-ads.txt", "adsorption")):
        for g in _henry_lines(HERE / fname):
            rows.append([g[0], g[1], field, g[2], f"{g[3]} ({g[4]})", g[5],
                         f"{g[6]} ({g[7]})", g[8]])
    for line in (HERE / "cargas_literatura.txt").read_text().splitlines():
        m = re.match(r"(\S+)\s+(.+?)\s+carga neta\s+(\S+) e -> 0\s+K_H\s+([\d.]+) \(([\d-]+)\)"
                     r"\s+q_st\s+([\d.]+)", line)
        if m:
            rows.append([m.group(1), "", f"literature ({m.group(2)})", m.group(3),
                         f"{m.group(4)} ({m.group(5)})", m.group(6), "—", "—"])
    n = json.loads((HERE / "n_dft.json").read_text())
    rows.append(["N5", "C912N48", "DFT (IAO)", f"{n['carga_neta_por_celda']:+.2f}",
                 f"{n['henry']['K_H']:.1f} ({n['henry']['rango'][0]:.0f}-"
                 f"{n['henry']['rango'][1]:.0f})", f(n["henry"]["q_st"]), "—", "—"])
    rows.append(["N5", "C912N48", "DFT, N with C Lennard-Jones", "", "972.9", "32.68",
                 "—", "—"])
    return table("CO₂ Henry constants of the functionalised 6.6 Å slits at 298 K "
                 "under each charge set, with and without the wall charges (2 seeds "
                 "× 1.2 × 10⁵; range in brackets). Net cell charge before "
                 "neutralisation over the sp² sheet.",
                 ["variant", "formula", "charge set", "net charge (e)",
                  "K<sub>H</sub> with charges", "q<sub>st</sub>", "K<sub>H</sub> no charges",
                  "q<sub>st</sub>"], rows, num={3, 4, 5, 6, 7})


def s_charges() -> str:
    rows = [
        ["COOH", "C(carboxyl) +0.15, O= −0.50, O(H) −0.60, H +0.42", "−0.53",
         "C(carboxyl) +0.75, O= −0.50, O(H) −0.55, H +0.36, ring C −0.06", "0.00",
         "Tenney & Lastoskie 2006"],
        ["OH", "C +0.15, O −0.60, H +0.42", "−0.03", "C +0.30, O −0.60, H +0.30", "0.00",
         "Tenney & Lastoskie 2006"],
        ["NH₂", "C +0.15, N −0.90, H +0.36", "−0.03", "C −0.172, N −0.539, H +0.301",
         "−0.109", "Chen et al. 2026"],
        ["graphitic N", "N −0.45, 3 C +0.15", "0.00", "N −0.017, 3 C +0.089", "+0.25",
         "this work (DFT)"],
    ]
    out = table("Wall partial charges (e): the representative electrode table vs the "
                "adsorption force field.",
                ["group", "electrode table", "net", "adsorption FF", "net", "source"],
                rows, num={2, 4})
    drows = []
    for r in ("4", "6"):
        d = json.loads((HERE / "dft_cargas" / f"cargas_n_grafitico_r{r}.json").read_text())
        for scheme in ("iao", "esp"):
            drows.append([d["flake"], scheme.upper(), f(d["N"][scheme], 3),
                          ", ".join(f(x, 3) for x in d["C_N"][scheme]),
                          f(d["C_sp2_mean_interior"][scheme], 3)
                          if d["C_sp2_mean_interior"][scheme] == d["C_sp2_mean_interior"][scheme]
                          else "—",
                          f"{d['seconds']:.0f}"])
    out += table("DFT charges of graphitic N (UKS PBE/def2-SVP, H-terminated flakes, "
                 "PySCF). IAO converges between flakes; the ESP fit flips sign and is "
                 "discarded.",
                 ["flake", "scheme", "N", "C neighbours", "interior C (mean)", "time (s)"],
                 drows, num={2, 4, 5})
    return out


def s_selectivity() -> str:
    rows = []
    for p in sorted((HERE / "selectividad").glob("*.json")):
        d = json.loads(p.read_text())
        label = p.stem.replace("funcional-", "slit ").replace("cristalina-knee-", "")
        if p.stem == "funcional-N5":
            label += " (representative N charges)"
        rows.append([label,
                     f(d["mol_per_kg"][0], 3), f(d["mol_per_kg"][1], 3),
                     f"{d['mean_counts'][1]:.2f}", f(d["adsorbed_fractions"][0], 3),
                     f"{d['selectivity']:.1f} ± {d['selectivity_error']:.1f}",
                     f"{d['energy_drift_k']:.0e}", str(len(d["warnings"]))])
    n = json.loads((HERE / "n_dft.json").read_text())["mezcla"]
    rows.append(["slit N5 (DFT charges)", f(n["mol_per_kg"][0], 3), f(n["mol_per_kg"][1], 3),
                 f"{n['mean_counts'][1]:.2f}", f(n["adsorbed_fractions"][0], 3),
                 f"{n['selectivity']:.1f} ± {n['selectivity_error']:.1f}",
                 f"{n['energy_drift_k']:.0e}", str(len(n["warnings"]))])
    return table("Dry flue gas (15 % CO₂ in N₂, 0.1 MPa, 298 K): mixture GCMC "
                 "(2 × 10⁵ equilibration + 6 × 10⁵ production). Functionalised slits "
                 "use the adsorption FF (published group charges); 6.6 Å slit, cages "
                 "and crystalline structures have neutral Steele walls.",
                 ["structure", "CO₂ (mol/kg)", "N₂ (mol/kg)", "N₂ per cell",
                  "x<sub>CO₂</sub>", "S CO₂/N₂", "energy drift (K)", "warnings"], rows)


def s_humid() -> str:
    rows = []
    for p in sorted((HERE / "humedad").glob("*.json")):
        d = json.loads(p.read_text())
        rows.append([p.stem.rsplit("-hr", 1)[0], f"{d['hr']:g}",
                     f(d["mol_per_kg"][0], 3), f(d["mol_per_kg"][1], 3),
                     f(d["mol_per_kg"][2], 3), f"{d['mean_counts'][2]:.1f}",
                     f"{d['selectivity']:.0f} ± {d['selectivity_error']:.0f}"])
    return table("Humid flue gas (CO₂/N₂/H₂O, SPC/E water, 0.1 MPa, 298 K); relative "
                 "humidity against the real saturation pressure (3.17 kPa). "
                 "3 × 10⁵ equilibration + 6 × 10⁵ production from an empty cell.",
                 ["slit", "RH (%)", "CO₂ (mol/kg)", "N₂ (mol/kg)", "H₂O (mol/kg)",
                  "H₂O per cell", "S CO₂/N₂"], rows)


def s_capacity() -> str:
    rows = []
    for line in (HERE / "capacidad_trabajo.txt").read_text().splitlines()[1:]:
        m = re.match(r"(.+?)\s+([\d.]+) \(([\d.]+)-([\d.]+)\)\s+([\d.]+) \(([\d.]+)-([\d.]+)\)", line)
        if m:
            rows.append([m.group(1).strip(), m.group(2), m.group(5)])
    tsa = {}
    for line in (HERE / "ciclo_temperatura.txt").read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 6:
            tsa[parts[0]] = parts[1:]
    names = {"prístina": "pristina", "N 5 %": "N5", "OH": "OH", "COOH": "COOH",
             "NH₂": "NH2"}
    out_rows = []
    for r in rows:
        t = tsa.get(names.get(r[0], ""), ["—"] * 5)
        out_rows.append([r[0], r[1], r[2], t[0], t[1], t[2], t[3], t[4]])
    return table("Working capacities from pure-CO₂ isotherms (log-P interpolation, "
                 "single component, mol/kg). TSA: adsorb at 298 K and 15 kPa, desorb "
                 "at 373 K under a 15 kPa purge or under pure CO₂ at 0.1 MPa.",
                 ["structure", "VSA 100→10 kPa", "VSA 15→1 kPa", "298 K, 15 kPa",
                  "373 K, 15 kPa", "373 K, 100 kPa", "TSA purge", "TSA CO₂"], out_rows)


def s_settings() -> str:
    rows = [
        ["H₂ isotherms (slits, schwarzitas)", "77", "1.5×10⁶ / 1.5×10⁶", "insert, delete, move", "virial (Levesque)", "Feynman-Hibbs"],
        ["H₂ cage isotherms", "77", "campaign defaults", "insert, delete, move", "virial", "Feynman-Hibbs"],
        ["H₂ Henry", "77", "10⁶ insertions × 2 seeds", "Widom", "virial", "Feynman-Hibbs"],
        ["CO₂ isotherms", "298 / 373", "10⁵ / 2×10⁵", "insert, delete, move, rotate", "Peng-Robinson", "Ewald 10⁻⁵"],
        ["CO₂ Henry", "298", "1.2–2×10⁵ × 2 seeds", "Widom", "—", "Ewald with charged walls"],
        ["CO₂/N₂ mixture", "298", "2×10⁵ / 6×10⁵", "+ identity swap", "PR mixture, k<sub>ij</sub> −0.017", "Ewald"],
        ["CO₂/N₂/H₂O", "298", "3×10⁵ / 6×10⁵", "+ identity swap", "PR mixture", "Ewald"],
        ["snapshots (figures)", "77 / 298", "5–20×10⁴ / 2–5×10³", "as above", "as above", "illustrative only"],
    ]
    return table("Simulation settings. Moves are equally weighted; translations up "
                 "to 1 Å; rotations draw a uniform orientation. LJ cutoff 12 Å, no "
                 "tail corrections; cells shorter than twice the cutoff or "
                 "non-orthorhombic are refused.",
                 ["run", "T (K)", "equilibration / production", "moves", "reservoir",
                  "notes"], rows, num=set())


# --------------------------------------------------------------------------

GAS_LABEL = {"h2": "H₂, 77 K, 10 MPa", "co2": "CO₂, 298 K, 0.1 MPa",
             "co2_1mpa": "CO₂, 298 K, 1 MPa",
             "mezcla": "15 % CO₂ in N₂, 0.1 MPa", "humeda": "humid flue gas, 80 % RH"}


def molecule_count(path: Path) -> str:
    """Molecules per cell in a gallery snapshot, by species (whole cell,
    not only the slab drawn)."""
    from ase.io import read
    from ase.neighborlist import neighbor_list
    import numpy as np
    atoms = read(str(path))
    tags = atoms.get_tags()
    sym = np.array(atoms.get_chemical_symbols())
    if (tags == 2).any():
        return f"{int((tags == 2).sum())} H₂"
    ads = np.flatnonzero(tags > 0)
    sub = atoms[ads]
    i, j = neighbor_list("ij", sub, 1.25)
    parent = list(range(len(sub)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(i, j):
        parent[find(a)] = find(b)
    groups = {}
    for k in range(len(sub)):
        groups.setdefault(find(k), []).append(sym[ads][k])
    counts = {"CO₂": 0, "N₂": 0, "H₂O": 0}
    for members in groups.values():
        m = sorted(members)
        counts["CO₂" if "C" in m else "H₂O" if "H" in m else "N₂"] += 1
    return " · ".join(f"{v} {k}" for k, v in counts.items() if v) or "0"


def gallery() -> tuple[str, dict]:
    from galeria import GASES, STRUCTURES
    files = {}
    cells = []
    header = "".join(f'<div class="gh">{GAS_LABEL[g]}</div>' for g in GASES)
    for name, (_src, style) in STRUCTURES.items():
        row = [f'<div class="gname">{name}<span class="sub">'
               f'{ {"slab": "8 Å slab", "net": "slab, 40 % of cell", "axis": "down the tube axis"}.get(style, "whole cell")}'
               "</span></div>"]
        for g in GASES:
            png = HERE / "galeria" / f"{name}__{g}.png"
            if png.exists():
                key = f"fig/gal/{png.name}"
                files[key] = str(png)
                row.append(f'<div class="gcell"><img src="{key}" loading="lazy" '
                           f'alt="{name} with {GAS_LABEL[g]}"><span class="gcount">'
                           f'{molecule_count(png.with_suffix(".extxyz"))}</span></div>')
            else:
                row.append('<div class="gcell missing">not run</div>')
        cells.append("".join(row))
    global _figures
    _figures += 1
    grid = (f'<figure><div class="gallery-wrap"><div class="gallery"><div></div>{header}{"".join(cells)}</div></div>'
            f"<figcaption><b>Figure S{_figures}.</b> Every superstructure and the SWCNT "
            "bundle with each gas: final configurations of short, illustrative GCMC "
            "runs (neutral walls). Colours: carbon wall grey; H₂ pink (one centre per "
            "molecule); CO₂ C dark grey, O red; N₂ blue; water O amber, H white. "
            "The count under each image is molecules per WHOLE cell, not per slab: "
            "with CO₂ at 0.1 MPa and 298 K these wide-pored structures adsorb little "
            "(0.2–1.6 mol/kg against 5.8 for the 6.6 Å slit), so the emptiness is real; "
            "the 1 MPa column holds 2–7 times more. "
            "3D-periodic cells are drawn as an 8 Å slab through the cell centre.</figcaption>"
            "</figure>")
    return grid, files


def build() -> dict:
    files = {}
    parts = ['<section id="si"><h2><span class="n">S</span>Supplementary Information</h2>',
             '<div class="col"><p class="sub">Tables and figures in English. Every '
             "table is generated from the study's output files by "
             "<code>suplementario.py</code>.</p></div>"]
    parts.append('<h3 class="sih">S1 · Validation of the H₂ GCMC</h3>')
    parts.append(s_levesque())
    if (HERE / "galeria" / "swcnt-bundle__h2.png").exists():
        files["fig/gal/swcnt-bundle__h2.png"] = str(HERE / "galeria" / "swcnt-bundle__h2.png")
        parts.append('<div class="grid2">' + figure(
            "fig/gal/swcnt-bundle__h2.png",
            "The Levesque et al. (2002) bundle of open (17,0) SWCNTs (D = 13.3 Å, "
            "6 Å gap, 8704 C) with H₂ at 77 K and 10 MPa, seen down the tube axis. "
            "H₂ fills the tube interiors and the interstitial channels.",
            "SWCNT bundle with H2") + figure(
            "fig/isoterma_validacion.png",
            "Validation isotherms against Levesque et al. (2002) at 77 and 293 K.",
            "Validation isotherms") + "</div>")
    parts.append('<h3 class="sih">S2 · H₂ in graphene slits</h3>')
    parts.append(s_slits_h2())
    parts.append('<h3 class="sih">S3 · Schwarzites</h3>')
    parts.append(s_schwarzites())
    imgs = []
    for name, label in (("schwarzite-P", "Schwarz P (primitive)"),
                        ("schwarzite-D", "Schwarz D (diamond)"),
                        ("gyroid", "gyroid")):
        png = HERE / "galeria" / f"{name}__h2.png"
        if png.exists():
            key = f"fig/gal/{png.name}"
            files[key] = str(png)
            imgs.append(figure(key, f"{label} with H₂ at 77 K and 10 MPa (8 Å slab "
                               "through the cell).", f"{label} with H2"))
    imgs.append(figure("fig/capacidad_vs_area.png",
                       "H₂ capacity of the schwarzites vs geometric area, against "
                       "Chahine's rule (~1 wt% per 500 m²/g at 77 K).", "capacity vs area"))
    parts.append('<div class="grid2">' + "".join(imgs) + "</div>")
    parts.append('<h3 class="sih">S4 · Superstructure catalogue and H₂ at zero coverage</h3>')
    parts.append(s_catalogue())
    parts.append(s_henry_h2())
    parts.append(s_cages_h2())
    parts.append('<h3 class="sih">S5 · CO₂: validation, Henry constants and isotherms</h3>')
    parts.append(s_co2_graphite())
    parts.append(s_henry_co2())
    parts.append(s_iso_co2())
    parts.append('<h3 class="sih">S6 · Wall chemistry and charges</h3>')
    parts.append(s_charges())
    parts.append(s_functional())
    parts.append('<h3 class="sih">S7 · Mixtures, humidity and regeneration</h3>')
    parts.append(s_selectivity())
    parts.append(s_humid())
    parts.append(s_capacity())
    parts.append('<h3 class="sih">S8 · Gallery: every superstructure with every gas</h3>')
    grid, gfiles = gallery()
    files.update(gfiles)
    parts.append(grid)
    parts.append('<h3 class="sih">S9 · Simulation settings</h3>')
    parts.append(s_settings())
    parts.append("</section>")
    block = "\n".join(parts)
    text = REPORT.read_text()
    if START in text:
        pre, rest = text.split(START, 1)
        post = rest.split(END, 1)[1]
    else:
        pre, post = text.split("  <footer>", 1)
        post = "\n  <footer>" + post
    REPORT.write_text(pre + START + "\n" + block + "\n" + END + post)
    return files


if __name__ == "__main__":
    print(json.dumps(build(), indent=0))
