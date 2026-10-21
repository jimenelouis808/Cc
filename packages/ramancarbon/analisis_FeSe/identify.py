"""What does the library say is there, and which peaks stay unexplained?

Independent of every Rietveld choice: this works off peak POSITIONS,
which depend only on the lattice and not on the background model, the
profile ceiling or the carbon's scale.  So it is the one part of this
analysis the fluorescence background cannot corrupt.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import load
from ramancarbon.xrd.search import find_peaks, identify_phases

for sample in sys.argv[1:]:
    pattern = load(sample)
    peaks = find_peaks(pattern)
    print(f"\n{'='*70}\n{sample}: {len(peaks)} picos detectados")
    for p in sorted(peaks, key=lambda q: -q.height)[:14]:
        print("   2th=%7.3f  altura=%8.1f  d=%.4f" % (p.two_theta, p.height,
                                                      pattern.wavelength /
                                                      (2*np.sin(np.radians(p.two_theta/2)))))
    result = identify_phases(pattern, peaks=peaks, max_phases=6)
    print()
    print(result.describe() if hasattr(result, "describe") else result.summary())
