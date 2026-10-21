"""A GOF of 2.44 says something is missing. What sits at 42.71 and 51.47?

The unexplained peaks ARE the result, not a leftover.  Two steps: ask
the library which bundled phase puts a strong line at those angles, then
actually refine the candidates in and see whether the fit moves.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import load, library_map
from ramancarbon.xrd.powder import reflections
from ramancarbon.xrd.rietveld import PhaseModel, auto_refine

WANTED = (42.712, 51.468)
lib = library_map()
print("fases de la biblioteca con una línea fuerte en 42.71 o 51.47 deg:")
for name, crystal in sorted(lib.items()):
    hits = []
    for r in reflections(crystal, wavelength=1.540593,
                         two_theta_range=(10.0, 90.0), normalise=True):
        for target in WANTED:
            if abs(r.two_theta - target) < 0.35 and r.intensity >= 25.0:
                hits.append("%s %.2f I=%.0f" % (str(r.hkl), r.two_theta,
                                               r.intensity))
    if hits:
        print("  %-24s %s" % (name, "; ".join(hits)))

base = ["C_turbostratico_3.44", "Fe3C_cementita", "Fe_alfa", "FeSe_tetragonal"]
pattern = load("S2_1020")
print("\nrefinando con cada candidato añadido:")
reference = auto_refine(pattern, [PhaseModel(crystal=lib[n]) for n in base])
print("  %-24s Rwp=%.3f%% GOF=%.3f" % ("(sin añadir)", 100*reference.r_wp,
                                       reference.gof))
for extra in sys.argv[1:]:
    res = auto_refine(pattern,
                      [PhaseModel(crystal=lib[n]) for n in base + [extra]])
    w = res.weight_fractions().get(extra)
    print("  %-24s Rwp=%.3f%% GOF=%.3f  su fracción=%.1f %%  (ΔRwp=%+.3f)"
          % (extra, 100*res.r_wp, res.gof, 100*(w or 0),
             100*(res.r_wp - reference.r_wp)))
