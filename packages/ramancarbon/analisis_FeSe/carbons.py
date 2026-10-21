"""One carbon, two carbons, or none?

S2's figure shows a calculated 002 that is too narrow and too tall while
the observed band has wide wings -- which is the signature of TWO
populations, a well stacked one and a disordered one, not of one
population the model cannot widen enough.  So: try them.

Run at background order 10, not the package default of 6, so that the
carbon model is being compared and not the background's inability to
follow a fluorescence hump.
"""
import sys
from dataclasses import replace

sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import load, library_map
from ramancarbon.xrd.rietveld import PhaseModel, auto_refine

ORDER = 10
lib = library_map()
sample = sys.argv[1] if len(sys.argv) > 1 else "S2_1020"
IRON = {
    "S2_1020": ["Fe3C_cementita", "Fe_alfa", "FeSe_tetragonal"],
    "S1_900": ["FeSe_tetragonal", "FeSe_hexagonal", "Fe3C_cementita",
               "Fe_alfa"],
}[sample]

def renamed(name, tag):
    return replace(lib[name], name=tag)

MODELS = {
    "sin carbono": [],
    "1 turbostratico 3.44": ["C_turbostratico_3.44"],
    "turbostratico + grafito 2H": ["C_turbostratico_3.44", "grafito_2H"],
    "turbostratico 3.50 + 3.36": ["C_turbostratico_3.50",
                                  "C_turbostratico_3.36"],
}
pattern = load(sample)
print(f"== {sample}, fondo de orden {ORDER}\n")
for label, carbons in MODELS.items():
    crystals = [lib[n] for n in IRON] + [lib[n] for n in carbons]
    res = auto_refine(pattern, [PhaseModel(crystal=c) for c in crystals],
                      background_order=ORDER)
    w = res.weight_fractions()
    sizes = res.crystallite_sizes()
    print("%-28s Rwp=%6.3f%% GOF=%.3f fondo=%.1f%%"
          % (label, 100*res.r_wp, res.gof, 100*res.background_share))
    for ph in res.phases:
        n = ph.crystal.name
        size = sizes.get(n)
        pinned = [q.name.split("[")[0] for q in res.parameters
                  if q.free and q.name.endswith(f"[{n}]")
                  and q.kind in ("profile_w", "profile_uv")
                  and abs(q.value - q.upper) < 1e-6 * max(1, abs(q.upper))]
        print("      %-22s %6.1f %%  L=%s nm  FWHM(26°)=%.2f°%s"
              % (n, 100*(w.get(n) or 0),
                 "  — " if size is None else "%4.1f" % size,
                 ph.profile.fwhm_at(26.0),
                 "  CLAVADO: " + ",".join(pinned) if pinned else ""))
    print()

# Two populations of the SAME structure: the package supports it, but
# weight_fractions is keyed by crystal name, so a copy needs its own.
crystals = ([lib[n] for n in IRON]
            + [renamed("C_turbostratico_3.44", "C_apilado"),
               renamed("C_turbostratico_3.44", "C_desordenado")])
res = auto_refine(pattern, [PhaseModel(crystal=c) for c in crystals],
                  background_order=ORDER)
w = res.weight_fractions()
sizes = res.crystallite_sizes()
print("%-28s Rwp=%6.3f%% GOF=%.3f fondo=%.1f%%"
      % ("2 poblaciones del mismo C", 100*res.r_wp, res.gof,
         100*res.background_share))
for ph in res.phases:
    n = ph.crystal.name
    size = sizes.get(n)
    print("      %-22s %6.1f %%  L=%s nm  FWHM(26°)=%.2f°  d002=%.4f"
          % (n, 100*(w.get(n) or 0),
             "  — " if size is None else "%4.1f" % size,
             ph.profile.fwhm_at(26.0),
             ph.current_crystal().lattice.c / 2))
