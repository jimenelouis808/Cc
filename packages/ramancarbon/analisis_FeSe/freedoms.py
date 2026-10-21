"""Free positional coordinates per phase, done properly.

A displacement d of a site is allowed only if every operation of the
site's stabiliser maps d onto d, i.e. (R - I) d = 0 for all of them.
That is the null space of the stacked (3|S|, 3) matrix -- basis
independent and exact.  The earlier version averaged the rotations into
a projector and then symmetrised it, which is wrong in a fractional
(non-orthonormal) basis: it invented an eigenvector of eigenvalue 1 for
the hexagonal 2c site, which is in fact completely fixed.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-Cc/0495242d-7cd2-5e51-b6e4-d3296f40ec00/scratchpad/xrd")
import numpy as np
from common import library_map, DECLARED


def stabiliser(crystal, fract):
    x = np.asarray(fract, dtype=float)
    keep = []
    for op in crystal.operations:
        delta = np.mod(op.R @ x + op.t - x + 0.5, 1.0) - 0.5
        if np.all(np.abs(delta) < 1e-3):
            keep.append(op)
    return keep


def free_directions(crystal, fract):
    ops = stabiliser(crystal, fract)
    if not ops:
        return np.eye(3)
    stack = np.vstack([op.R - np.eye(3) for op in ops])
    _, s, vt = np.linalg.svd(stack)
    s = np.concatenate([s, np.zeros(3 - s.size)])
    null = vt[s < 1e-8]
    return null.T if null.size else np.zeros((3, 0))


lib = library_map()
names = []
for group in DECLARED.values():
    for n in group:
        if n not in names:
            names.append(n)
grand = 0
for name in names:
    c = lib[name]
    print(f"\n{name}  [{c.space_group}]  order={c.order}")
    total = 0
    for site in c.sites:
        basis = free_directions(c, site.fract)
        n = basis.shape[1]
        total += n
        dirs = "  ".join("(" + ", ".join(f"{v:+.3f}" for v in basis[:, k]) + ")"
                         for k in range(n)) if n else "fijo"
        print(f"   {site.label or site.element:6s} {site.element:3s} "
              f"{np.round(site.fract, 4)}  libres={n}  {dirs}")
    grand += total
    print(f"   -> {total} coordenada(s) posicional(es) libre(s)")
print(f"\nTOTAL sobre las fases declaradas: {grand}")
