"""How wide is the carbon 002 band, and can the profile bounds reach it?"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import load

for name in ("S1_900", "S2_1020"):
    p = load(name)
    tt, y = p.two_theta, p.intensity
    win = (tt > 18.0) & (tt < 34.0)
    x, v = tt[win], np.convolve(y[win], np.ones(41)/41, mode="same")
    # A straight baseline between the window edges.
    base = np.interp(x, [x[0], x[-1]], [v[:60].mean(), v[-60:].mean()])
    band = v - base
    peak = band.max()
    i = band.argmax()
    half = np.flatnonzero(band > peak / 2.0)
    fwhm = x[half[-1]] - x[half[0]] if half.size else float("nan")
    print("%-9s 002 at 2th=%.2f  altura=%.0f sobre base=%.0f  FWHM=%.2f deg"
          % (name, x[i], peak, base[i], fwhm))

# What the Caglioti bounds can reach at that angle.
for two_theta in (25.9,):
    t = np.tan(np.radians(two_theta / 2.0))
    print("\nen 2th=%.1f (tan(th)=%.3f):" % (two_theta, t))
    print("  FWHM maxima con las cotas actuales (U=2, V=1, W=2): %.2f deg"
          % np.sqrt(2*t*t + 1*t + 2))
    for cap in (8.0, 20.0, 50.0):
        print("  con W<=%4.0f (U,V igual):                        %.2f deg"
              % (cap, np.sqrt(2*t*t + 1*t + cap)))
