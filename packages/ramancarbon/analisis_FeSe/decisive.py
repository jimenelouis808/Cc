"""Quantify the iron phases with the carbon band excluded.

Every carbon model tried ends with W, U and V pinned at the ceiling, so
the carbon's fraction and its "size" are the bound and not a
measurement, and the iron fractions move 18 points depending on which
carbon model is chosen.  Taking the band out of the fit stops the carbon
from deciding the iron numbers.  What it costs -- the carbon is then not
quantified at all -- is the honest price and is stated.

gamma-Fe is tried because S2 has a peak at d = 2.0897 A that none of the
declared phases produces, and austenite with carbon dissolved (a ~ 3.62)
puts its 111 exactly there.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import load, library_map
from ramancarbon.xrd.rietveld import PhaseModel, auto_refine

ORDER = 10
BAND = {"S2_1020": [(16.0, 37.0)], "S1_900": [(16.0, 37.0)]}
IRON = {
    "S2_1020": ["Fe3C_cementita", "Fe_alfa", "FeSe_tetragonal"],
    "S1_900": ["FeSe_tetragonal", "FeSe_hexagonal", "Fe3C_cementita",
               "Fe_alfa"],
}
lib = library_map()
for sample in sys.argv[1:]:
    pattern = load(sample)
    for label, names in (("declaradas", IRON[sample]),
                         ("+ gamma-Fe", IRON[sample] + ["Fe_gamma"])):
        res = auto_refine(pattern, [PhaseModel(crystal=lib[n]) for n in names],
                          background_order=ORDER, excluded=BAND[sample])
        w = res.weight_fractions()
        sizes = res.crystallite_sizes()
        print("\n%s / %s  Rwp=%6.3f%% GOF=%.3f fondo=%.1f%%"
              % (sample, label, 100*res.r_wp, res.gof,
                 100*res.background_share))
        for ph in res.phases:
            n = ph.crystal.name
            cell = ph.current_crystal().lattice
            pinned = [q.name.split("[")[0] for q in res.parameters
                      if q.free and q.name.endswith(f"[{n}]")
                      and q.kind in ("profile_w", "profile_uv")
                      and abs(q.value - q.upper) < 1e-6 * max(1, abs(q.upper))]
            size = sizes.get(n)
            print("    %-20s %6.2f %%  a=%.4f c=%.4f  L=%s%s"
                  % (n, 100*(w.get(n) or 0), cell.a, cell.c,
                     " — " if size is None else "%.1f nm" % size,
                     "  CLAVADO:" + ",".join(pinned) if pinned else ""))
