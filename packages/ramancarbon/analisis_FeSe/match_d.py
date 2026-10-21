"""Which bundled phase has a line at each unexplained d-spacing?

Done against the whole library, including weak reflections, because the
question is not "which phase wins" -- `identify_phases` already answers
that -- but "is this peak one the declared phases can produce at all, or
is it something else".
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import library_map
from ramancarbon.xrd.powder import reflections

UNEXPLAINED = {
    "S2_1020": [(25.851, 3.4437, 1365.5), (42.712, 2.1152, 201.6),
                (43.260, 2.0897, 170.3), (51.468, 1.7741, 117.6),
                (79.150, 1.2091, 56.4)],
    "S1_900": [(43.890, 2.0612, 235.0), (45.988, 1.9719, 184.6),
               (28.730, 3.1048, 170.9), (49.237, 1.8491, 164.6),
               (50.499, 1.8058, 122.0), (39.870, 2.2592, 103.5)],
}
TOL = 0.012  # Angstrom: about 0.25 deg at these angles.

lib = library_map()
cache = {name: reflections(c, wavelength=1.540593,
                           two_theta_range=(10.0, 90.0), normalise=True,
                           min_relative=0.01)
         for name, c in lib.items()}

for sample, peaks in UNEXPLAINED.items():
    print(f"\n===== {sample}")
    for two_theta, d, height in peaks:
        hits = []
        for name, refl in cache.items():
            for r in refl:
                if abs(r.d - d) <= TOL:
                    hits.append((r.intensity, name, str(r.hkl), r.d))
        hits.sort(reverse=True)
        print("  2th=%7.3f d=%.4f I=%7.1f" % (two_theta, d, height))
        if not hits:
            print("        SIN CANDIDATO en la biblioteca")
        for intensity, name, hkl, dd in hits[:5]:
            print("        %-24s %-12s d=%.4f  I_rel=%5.1f"
                  % (name, hkl, dd, intensity))
