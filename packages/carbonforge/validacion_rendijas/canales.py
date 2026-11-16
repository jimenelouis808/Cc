"""The free channel of every handed-over cell, as triage.

    uv run python validacion_rendijas/canales.py <carpeta>

Measuring a Henry coefficient costs minutes per structure; measuring
the channel the probe can cross costs seconds, and across everything
in this folder the channel is what predicts uptake. Measured against
H2 at 77 K: 6.72 A gives 12861 mol/kg/MPa, 10.03 gives 9425, 14.61
gives 352 and 15.80 gives 153 -- two orders of magnitude lost to a
channel 5 A too wide. So a shape sweep is triaged on the channel and
only its survivors are measured properly.

The window is read off those measurements rather than assumed, and it
has a floor as well as a ceiling: a crystalline super-square with a
4.74 A channel reads 3137, losing the other way because sigma for
H2 against carbon is 3.18 A and a channel under about twice that
cannot hold a molecule with both walls in reach.

Soundness comes from the json beside each file, for the reason
`llenado_super.py` gives: the check lives in the other package and the
verdict travels in the file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ase.io import read

from carbonforge.adsorption.porosity import analyse_porosity

from llenado_super import embed

#: Free-channel window, in Å, where uptake per gram is worth
#: measuring. Derived from the measurements in this folder's README,
#: not assumed.
WINDOW = (5.5, 11.0)

#: The sp2 bond window, in Å, which this repository's own validation
#: uses. It is a SECOND gate and not a restatement of the angle sum:
#: measured on this sweep, `knee-super-graphene_k18_n5` reads a
#: perfectly sound 350.7 deg with a 1.596 Å bond, and
#: `knee-gyroid_k16_n7` reads 347.8 deg at 1.634 Å. An angle sum is
#: about how a carbon is bent; a bond length is about whether it is
#: still bonded, and a cell can pass one and fail the other.
BOND_WINDOW = (1.30, 1.55)

#: Samples for the channel. It comes off a grid rather than the random
#: cloud, so this only sets the void fraction's error and the channel
#: is a lower bound in the grid spacing either way.
SAMPLES = 20_000


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1])
    sources = sorted(folder.glob("*.extxyz"))
    if not sources:
        print(f"no hay .extxyz en {folder}")
        return 1

    rows = []
    for source in sources:
        record = source.with_suffix(".json")
        data = json.loads(record.read_text()) if record.exists() else {}
        if data.get("collapsed"):
            rows.append((source.stem, None, data, "pared colapsada"))
            continue
        try:
            big, _repeat, _gap = embed(read(str(source)))
        except ValueError as error:
            rows.append((source.stem, None, data, str(error)[:60]))
            continue
        porosity = analyse_porosity(big, samples=SAMPLES, area_points=200,
                                    temperature=298.0, seed=0)
        rows.append((source.stem, porosity, data, ""))

    print("| estructura | átomos | canal (Å) | área (m²/g) | ángulo mín | "
          "enlaces (Å) | en ventana |")
    print("|---|---|---|---|---|---|---|")
    chosen = []
    for name, porosity, data, note in sorted(rows, key=lambda r: r[0]):
        if porosity is None:
            print(f"| {name} | {data.get('atomos', '—')} | — | — | "
                  f"{data.get('angle_sum_min', float('nan')):.1f} | — | "
                  f"{note} |")
            continue
        channel = porosity.largest_free_sphere
        inside = WINDOW[0] <= channel <= WINDOW[1]
        low, high = data.get("bond_min"), data.get("bond_max")
        bonded = (low is None or
                  (BOND_WINDOW[0] <= low and high <= BOND_WINDOW[1]))
        if inside and bonded:
            chosen.append((name, channel))
        bonds = f"{low:.3f}–{high:.3f}" if low else "—"
        verdict = ("SÍ" if inside and bonded else
                   "enlace fuera" if inside else "no")
        print(f"| {name} | {data.get('atomos', '—')} | {channel:.2f} | "
              f"{porosity.accessible_area_m2_g:.0f} | "
              f"{data.get('angle_sum_min', float('nan')):.1f} | {bonds} | "
              f"{verdict} |")

    print(f"\nEn la ventana de canal {WINDOW[0]}–{WINDOW[1]} Å y con "
          f"enlaces en {BOND_WINDOW[0]}–{BOND_WINDOW[1]} Å: "
          f"{len(chosen)} de {len(rows)}")
    for name, channel in sorted(chosen, key=lambda c: c[1]):
        print(f"  {name}  canal {channel:.2f} Å")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
