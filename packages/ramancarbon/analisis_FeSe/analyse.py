"""Full Rietveld analysis of the two FeSe/C samples, with tables.

Writes everything into OUT so the figure script can redraw without
re-refining: a refinement is a minute, a figure is a second, and the two
should not be tied together.
"""
import json
import sys
import time

sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import load, library_map
from ramancarbon.xrd.heights import height_agreement
from ramancarbon.xrd.rietveld import (
    PhaseModel,
    auto_refine,
    background_order_sensitivity,
    fit_columns,
    reflection_ticks,
)

OUT = "/home/user/Cc/packages/ramancarbon/analisis_FeSe"

# The phases the user declares, with the carbon chosen by the comparison
# in carbon.py rather than by what the label says: a graphite 2H cell has
# to stretch 2.3 % to reach this material's 002, which is the signature of
# a turbostratic carbon and not of graphite.
PHASES = {
    "S1_900": ["FeSe_tetragonal", "FeSe_hexagonal", "Fe3C_cementita",
               "Fe_alfa", sys.argv[2] if len(sys.argv) > 2 else "C_turbostratico_3.44"],
    "S2_1020": ["C_turbostratico_3.44", "Fe3C_cementita", "Fe_alfa",
                "FeSe_tetragonal"],
}
PRETTY = {
    "FeSe_tetragonal": "FeSe tetragonal (T-FeSe, P4/nmm)",
    "FeSe_hexagonal": "FeSe hexagonal (H-FeSe, P6₃/mmc)",
    "Fe3C_cementita": "Fe₃C cementita (Pnma)",
    "Fe_alfa": "α-Fe (Im-3m)",
    "grafito_2H": "Grafito 2H (P6₃/mmc)",
    "C_turbostratico_3.36": "C turbostrático d₀₀₂=3.36 Å",
    "C_turbostratico_3.44": "C turbostrático d₀₀₂=3.44 Å",
    "C_turbostratico_3.50": "C turbostrático d₀₀₂=3.50 Å",
}

sample = sys.argv[1]
# The package default of 6 cannot follow the fluorescence hump these
# patterns sit on: at order 6 the S2 refinement returns 99 % carbon with
# GOF 2.44. The order is passed in, and the sweep below is what justifies
# whichever one was chosen.
ORDER = int(sys.argv[3]) if len(sys.argv) > 3 else 12
lib = library_map()
pattern = load(sample)
models = [PhaseModel(crystal=lib[n]) for n in PHASES[sample]]

start = time.perf_counter()
result = auto_refine(pattern, models, background_order=ORDER,
                     callback=lambda s: print("  .", s, flush=True))
elapsed = time.perf_counter() - start
print(f"\n### {sample}: {elapsed:.0f} s", flush=True)

weights = result.weight_fractions()
shares = result.phase_contributions()
sizes = result.crystallite_sizes()
errors = {p.name: p.error for p in result.parameters}

# Scale uncertainty propagates into the weight fraction; the fraction is
# a ratio so the scale's own relative error is what carries over, and
# that is a lower bound on the real uncertainty -- it says nothing about
# the background order, which the sweep below measures separately.
rows = []
for phase in result.phases:
    name = phase.crystal.name
    scale_error = errors.get(f"escala[{name}]")
    relative = (scale_error / phase.scale
                if scale_error and phase.scale > 0 else None)
    cell = phase.current_crystal().lattice
    rows.append({
        "phase": name,
        "pretty": PRETTY.get(name, name),
        "weight": weights.get(name),
        "weight_sigma": (relative * weights[name]
                         if relative is not None and weights.get(name) else None),
        "share": shares.get(name),
        "size_nm": sizes.get(name),
        "scale": phase.scale,
        "a": cell.a, "b": cell.b, "c": cell.c,
        "lattice_factors": list(phase.lattice_factors),
        "site_shifts": list(phase.site_shifts),
        "free_coordinates": phase.free_coordinates,
        "profile": [phase.profile.u, phase.profile.v, phase.profile.w,
                    phase.profile.eta0],
        "sites": [
            {"label": s.label or s.element, "element": s.element,
             "fract": list(s.fract),
             "published": list(lib[name].sites[i].fract)}
            for i, s in enumerate(phase.current_crystal().sites)
        ],
    })

agreement = height_agreement(result)
print(agreement.describe(), flush=True)

print("\n--- barrido del orden del fondo ---", flush=True)
sweep = background_order_sensitivity(
    pattern, [PhaseModel(crystal=lib[n]) for n in PHASES[sample]],
    orders=(8, 10, 12, 14, 16, 20))
print(sweep.summary(), flush=True)

columns, table = fit_columns(result)
np.savetxt(f"{OUT}/{sample}_ajuste.csv", table, delimiter=",",
           header=",".join(columns), comments="")
ticks = reflection_ticks(result)
np.savez(f"{OUT}/{sample}_ticks.npz", **ticks)

summary = {
    "sample": sample,
    "background_order": ORDER,
    "seconds": elapsed,
    "r_wp": result.r_wp, "r_p": result.r_p,
    "r_expected": result.r_expected, "gof": result.gof,
    "chi2": result.chi_squared, "converged": result.converged,
    "background_share": result.background_share,
    "free_parameters": result.free_parameters,
    "n_points": int(pattern.two_theta.size),
    "stages": result.stages,
    "zero": result.zero, "displacement": result.displacement,
    "wavelength": pattern.wavelength,
    "phases": rows,
    "height_raw": agreement.median_raw_height_ratio,
    "height_envelope": agreement.median_height_ratio,
    "height_area": agreement.median_area_ratio,
    "height_verdict": agreement.verdict,
    "background_spread": sweep.spread(),
    "background_fractions": {str(k): v for k, v in sweep.fractions.items()},
    "background_rwp": {str(k): v for k, v in sweep.r_wp.items()},
    "background_best_order": sweep.best_order(),
    "warnings": result.warnings,
}
with open(f"{OUT}/{sample}_resumen.json", "w") as handle:
    json.dump(summary, handle, indent=2, ensure_ascii=False, default=float)
print(f"\nescrito {OUT}/{sample}_*", flush=True)
