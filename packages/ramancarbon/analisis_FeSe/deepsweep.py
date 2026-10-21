"""Where does the answer stop moving with the background order?

Cu Ka on an iron-rich sample fluoresces, which raises a strongly curved
background, and a sixth-order Chebyshev cannot follow it -- so the
carbon phase absorbs it.  The package default is 6 and on these two
patterns that is plainly the wrong choice; the question is which order
is the right one, and the only honest answer is the one where the
fractions stop moving.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import load, library_map
from ramancarbon.xrd.rietveld import PhaseModel, background_order_sensitivity

NAMES = {
    "S1_900": ["FeSe_tetragonal", "FeSe_hexagonal", "Fe3C_cementita",
               "Fe_alfa", "C_turbostratico_3.44"],
    "S2_1020": ["C_turbostratico_3.44", "Fe3C_cementita", "Fe_alfa",
                "FeSe_tetragonal"],
}
lib = library_map()
ORDERS = (8, 10, 12, 14, 16, 20)
for sample in sys.argv[1:]:
    sweep = background_order_sensitivity(
        load(sample), [PhaseModel(crystal=lib[n]) for n in NAMES[sample]],
        orders=ORDERS)
    print(f"\n===== {sample}")
    phases = NAMES[sample]
    print("orden   Rwp %   GOF   " + " ".join("%-11s" % n[:11] for n in phases))
    for order in ORDERS:
        fr = sweep.fractions[order]
        print("%4d  %7.3f %6.3f   " % (order, 100*sweep.r_wp[order],
                                       sweep.gof[order])
              + " ".join("%9.1f  " % (100*(fr.get(n) or 0)) for n in phases))
    print("  dispersión:", {k: round(100*v, 1)
                            for k, v in sorted(sweep.spread().items(),
                                               key=lambda kv: -kv[1])})
