"""What does the Caglioti ceiling do to the answer?"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import load, library_map
from ramancarbon.xrd import rietveld as R
from ramancarbon.xrd.rietveld import PhaseModel, auto_refine

lib = library_map()
p = load(sys.argv[1])
names = (["FeSe_tetragonal", "FeSe_hexagonal", "Fe3C_cementita", "Fe_alfa",
          "C_turbostratico_3.44"] if sys.argv[1] == "S1_900" else
         ["C_turbostratico_3.44", "Fe3C_cementita", "Fe_alfa",
          "FeSe_tetragonal"])
carbon = "C_turbostratico_3.44"
print("fondo de orden 12\nW_max  Rwp%    GOF    fondo%  C_peso%  C_señal%  W_C     FWHM_C  L_C(nm) clavado")
for cap in (2.0, 4.0, 6.0, 8.0, 12.0, 25.0):
    R.PROFILE_W_LIMIT = cap
    R.PROFILE_U_LIMIT = cap
    R.PROFILE_V_LIMIT = cap / 2.5
    models = [PhaseModel(crystal=lib[n]) for n in names]
    res = auto_refine(p, models, background_order=12)
    ph = next(x for x in res.phases if x.crystal.name == carbon)
    w = res.weight_fractions()[carbon]
    sh = res.phase_contributions()[carbon]
    size = res.crystallite_sizes().get(carbon)
    fwhm = ph.profile.fwhm_at(26.0)
    pinned = sum(1 for q in res.parameters
                 if q.free and q.name.endswith(f"[{carbon}]")
                 and (abs(q.value - q.upper) < 1e-6 * max(1, abs(q.upper))
                      or abs(q.value - q.lower) < 1e-6 * max(1, abs(q.lower))))
    print("%5.1f  %6.3f %6.3f  %5.1f   %6.1f   %6.1f   %7.3f %6.2f  %s      %d"
          % (cap, 100*res.r_wp, res.gof, 100*res.background_share,
             100*w, 100*sh, ph.profile.w, fwhm,
             "None " if size is None else "%5.1f" % size, pinned))
