"""Is S1's carbon graphite, or a turbostratic carbon?"""
import sys, time
sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import load, library_map
from ramancarbon.xrd.rietveld import auto_refine, PhaseModel

lib = library_map()
p = load("S1_900")
base = ["FeSe_tetragonal", "FeSe_hexagonal", "Fe3C_cementita", "Fe_alfa"]
for carbon in ("grafito_2H", "C_turbostratico_3.36", "C_turbostratico_3.44",
               "C_turbostratico_3.50"):
    models = [PhaseModel(crystal=lib[n]) for n in base + [carbon]]
    t = time.perf_counter()
    res = auto_refine(p, models)
    ph = res.phases[-1]
    c0 = ph.crystal.lattice.c
    print("%-22s Rwp=%.4f%% GOF=%.4f  c: %.3f -> %.3f (%+.2f%%)  d002=%.4f  "
          "W=%.3f U=%.3f  %.0f s"
          % (carbon, 100*res.r_wp, res.gof, c0, c0*ph.lattice_factors[2],
             100*(ph.lattice_factors[2]-1), c0*ph.lattice_factors[2]/2,
             ph.profile.w, ph.profile.u, time.perf_counter()-t))
