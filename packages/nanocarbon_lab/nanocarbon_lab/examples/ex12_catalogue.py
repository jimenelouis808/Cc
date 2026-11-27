#!/usr/bin/env python3
"""A catalogue image of the clean superstructures, rings coloured by size.

Every entry is built by this package at the size the measured tables in
CLAUDE.md give as clean, cached as JSON so a second run only re-renders,
and drawn with each ring filled by its size -- pentagons red, hexagons
grey, heptagons blue, octagons green -- because the point of these
structures is *where* the non-hexagons sit, which an atoms-and-bonds view
does not show. Labels are in Spanish, like the window.

    python -m nanocarbon_lab.examples.ex12_catalogue --out catalogo/
    python -m nanocarbon_lab.examples.ex12_catalogue --out catalogo/ --quick

``--quick`` keeps the entries that build in under about two minutes.
Builds run in ``--jobs`` processes.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

#: (key, Spanish label, builder, kwargs, quick). The builder is a name
#: resolved in `_build`, so the table pickles for the process pool.
ENTRIES: list[tuple[str, str, str, dict, bool]] = [
    ("cube", "Jaula cúbica", "geodesic",
     {"graph": "super-cube", "m": 1, "freq": 6, "rows": 1,
      "window_radius": 4}, True),
    ("tetra", "Jaula tetraédrica", "geodesic",
     {"graph": "super-tetrahedron", "m": 1, "freq": 8, "rows": 1,
      "window_radius": 3}, True),
    ("octa", "Jaula octaédrica", "geodesic",
     {"graph": "super-octahedron", "m": 1, "freq": 8, "rows": 1,
      "window_radius": 3}, True),
    ("ico", "Jaula icosaédrica", "geodesic",
     {"graph": "super-icosahedron", "m": 2, "freq": 2, "rows": 1}, True),
    ("dodeca", "Jaula dodecaédrica", "geodesic",
     {"graph": "super-dodecahedron", "m": 1, "freq": 6, "rows": 1,
      "window_radius": 4}, True),
    ("cubocta", "Cuboctaedro", "geodesic",
     {"graph": "super-cuboctahedron", "m": 1, "freq": 4, "rows": 2}, True),
    ("icosido", "Icosidodecaedro", "geodesic",
     {"graph": "super-icosidodecahedron", "m": 2, "freq": 2, "rows": 1}, True),
    ("ttetra", "Tetraedro truncado", "geodesic",
     {"graph": "super-truncated-tetrahedron", "m": 1, "freq": 4, "rows": 2},
     True),
    ("tocta", "Octaedro truncado", "geodesic",
     {"graph": "super-truncated-octahedron", "m": 1, "freq": 4, "rows": 2},
     True),
    ("rhombi", "Rombicuboctaedro", "geodesic",
     {"graph": "super-rhombicuboctahedron", "m": 1, "freq": 6, "rows": 2},
     False),
    ("hyper", "Hipercubo (teseracto)", "geodesic",
     {"graph": "super-hypercube", "m": 1, "freq": 6, "rows": 1,
      "window_radius": 2}, False),
    ("wtico", "Icosidodecaedro truncado (Wythoff)", "geodesic",
     {"graph": "wythoff-truncated-icosidodecahedron", "m": 1, "freq": 2,
      "rows": 1}, True),
    ("wtcubo", "Cuboctaedro truncado (Wythoff)", "geodesic",
     {"graph": "wythoff-truncated-cuboctahedron", "m": 1, "freq": 4,
      "rows": 2}, True),
    ("wrico", "Rombicosidodecaedro (Wythoff)", "geodesic",
     {"graph": "wythoff-rhombicosidodecahedron", "m": 2, "freq": 2,
      "rows": 1}, False),
    ("wtdod", "Dodecaedro truncado (Wythoff)", "geodesic",
     {"graph": "wythoff-truncated-dodecahedron", "m": 2, "freq": 2,
      "rows": 1}, False),
    ("c60", "C60 de tubos", "geodesic",
     {"graph": "superfullerene-C60", "m": 2, "freq": 2, "rows": 1}, False),
    ("scubic", "Súper-cúbica", "geodesic",
     {"graph": "super-cubic", "m": 1, "freq": 6, "rows": 1,
      "window_radius": 4}, True),
    ("ssquare", "Súper-cuadrada", "geodesic",
     {"graph": "super-square", "m": 1, "freq": 6, "rows": 1,
      "window_radius": 4}, True),
    ("sbcc", "Súper-bcc", "geodesic",
     {"graph": "super-bcc", "m": 2, "freq": 3, "rows": 1}, True),
    ("sgraph", "Súper-grafeno", "geodesic",
     {"graph": "super-graphene", "m": 1, "freq": 6, "rows": 1,
      "window_radius": 5}, False),
    ("skagome", "Súper-kagome", "geodesic",
     {"graph": "super-kagome", "m": 2, "freq": 2, "rows": 1}, False),
    ("ssrs", "Súper-srs (giroide)", "geodesic",
     {"graph": "super-srs", "m": 1, "freq": 4, "rows": 1,
      "window_radius": 3}, False),
    ("sdia", "Súper-diamante", "geodesic",
     {"graph": "super-diamond", "m": 2, "freq": 3, "rows": 1}, False),
    ("sfcc", "Súper-fcc", "geodesic",
     {"graph": "super-fcc", "m": 2, "freq": 4, "rows": 1}, False),
    ("slon", "Súper-lonsdaleíta", "geodesic",
     {"graph": "super-lonsdaleite", "m": 2, "freq": 3, "rows": 1}, False),
    ("stube", "Supertubo (4,4)", "geodesic",
     {"graph": "supertube-(4,4)", "m": 1, "freq": 4, "rows": 2}, False),
    ("pchiral", "Schwarz P quiral GC(3,1)", "goldberg",
     {"kind": "primitive", "m": 3, "n": 1}, True),
    ("fau", "Schwarz D GC(2,0), conmensurable con FAU", "goldberg",
     {"kind": "diamond", "m": 2, "n": 0}, True),
    ("dchiral", "Schwarz D quiral GC(2,1)", "goldberg",
     {"kind": "diamond", "m": 2, "n": 1}, True),
]

#: Ring fill by size; anything else is drawn black so it cannot hide.
RING_COLOURS = {5: "#d62728", 6: "#bdbdbd", 7: "#1f5fbf", 8: "#2ca02c"}


def _build(entry: tuple, cache: Path) -> Path:
    """Build one entry (or reuse its cache) and return the JSON path."""
    key, _label, builder, kwargs, _quick = entry
    path = cache / f"{key}.json"
    if path.exists():
        return path
    started = time.time()
    if builder == "geodesic":
        from nanocarbon_lab.builders.geodesic import (
            GEODESIC_CAGES,
            build_geodesic_superstructure,
        )

        tolerance = GEODESIC_CAGES.get(kwargs["graph"], {}).get(
            "tolerance_deg", 2.5)
        atoms = build_geodesic_superstructure(tolerance_deg=tolerance,
                                              **kwargs)
    else:
        from nanocarbon_lab.builders.goldberg import build_goldberg

        atoms = build_goldberg(**kwargs)
    from nanocarbon_lab.validation.quality import sp2_quality

    geometry = atoms.info["geometry"]
    record = {
        "positions": atoms.get_positions().tolist(),
        "cell": np.asarray(atoms.cell.lengths()).tolist(),
        "pbc": [bool(p) for p in atoms.pbc],
        "rings": [list(map(int, r)) for r in atoms.info["rings"]],
        "census": {str(k): int(v)
                   for k, v in atoms.info["ring_counts"].items()},
        "verdict": str(sp2_quality(geometry)[0]),
        "placed": float(atoms.info.get("disclinations_placed", 0.0)),
        "bonds": [float(geometry["bond_min"]), float(geometry["bond_max"])],
        "seconds": round(time.time() - started, 1),
    }
    path.write_text(json.dumps(record))
    return path


def _unwrap(points: np.ndarray, cell: np.ndarray, pbc: list[bool]
            ) -> np.ndarray:
    """A ring's corners joined across the cell seam, not split by it."""
    out = points.copy()
    live = np.array(pbc) & (cell > 0)
    for k in range(1, len(out)):
        d = out[k] - out[k - 1]
        d[live] -= cell[live] * np.round(d[live] / cell[live])
        out[k] = out[k - 1] + d
    return out


def _panel(ax, record: dict, label: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    positions = np.asarray(record["positions"])
    cell = np.asarray(record["cell"])
    live = np.array(record["pbc"]) & (cell > 0)
    # A periodic cell drawn alone looks torn at its faces, so it is tiled:
    # each ring is moved whole until its centre lies in the cell, then
    # repeated over the images. 2x2x2 for a crystal, 3x3 for a sheet.
    reps = [1, 1, 1]
    if live.any():
        reps = [(3 if live.sum() == 2 else 2) if on else 1 for on in live]
    images = [np.array([i, j, k]) * cell
              for i in range(reps[0]) for j in range(reps[1])
              for k in range(reps[2])]
    polygons, colours = [], []
    for ring in record["rings"]:
        corners = _unwrap(positions[ring], cell, record["pbc"])
        if live.any():
            centre = corners.mean(axis=0)
            shift = np.zeros(3)
            shift[live] = np.floor(centre[live] / cell[live]) * cell[live]
            corners = corners - shift
        for image in images:
            polygons.append(corners + image)
            colours.append(RING_COLOURS.get(len(ring), "#000000"))
    ax.add_collection3d(Poly3DCollection(
        polygons, facecolors=colours, edgecolors="#333333",
        linewidths=0.05, alpha=1.0))
    every = np.vstack(polygons)
    centre = every.mean(axis=0)
    half = float(np.ptp(every, axis=0).max()) / 2.0
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre,
                         strict=True):
        setter(c - half, c + half)
    ax.set_box_aspect((1, 1, 1))
    sheet = int(live.sum()) == 2
    ax.view_init(elev=70 if sheet else 22, azim=35)
    ax.set_axis_off()
    census = " ".join(f"{k}:{v}" for k, v in sorted(
        record["census"].items(), key=lambda kv: int(kv[0])) if k != "6")
    lo, hi = record["bonds"]
    tiled = "" if not live.any() else " (celda ×" + "×".join(
        str(r) for r, on in zip(reps, live, strict=True) if on) + ")"
    ax.set_title(f"{label}{tiled}\n{len(positions)} átomos · {census}\n"
                 f"{lo:.3f}–{hi:.3f} Å · {record['verdict']} · "
                 f"{record['placed']:.0%}", fontsize=7)


def render(entries: list[tuple], cache: Path, out: Path,
           columns: int = 5) -> Path:
    """Draw every cached entry into one grid image; returns its path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(entries) / columns))
    fig = plt.figure(figsize=(3.2 * columns, 3.4 * rows))
    for k, entry in enumerate(entries):
        record = json.loads((cache / f"{entry[0]}.json").read_text())
        ax = fig.add_subplot(rows, columns, k + 1, projection="3d")
        _panel(ax, record, entry[1])
    fig.suptitle("Superestructuras de carbono cristalinas — nanocarbon_lab "
                 "(pentágonos rojo, heptágonos azul, octágonos verde)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--quick", action="store_true",
                        help="Only the entries that build in minutes.")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args(argv)
    out = Path(args.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    entries = [e for e in ENTRIES if e[4] or not args.quick]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for path in pool.map(_build, entries, [cache] * len(entries)):
            print(f"  built {path.name}", flush=True)
    image = render(entries, cache, out / "catalogo.png")
    print(f"catalogue: {image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
