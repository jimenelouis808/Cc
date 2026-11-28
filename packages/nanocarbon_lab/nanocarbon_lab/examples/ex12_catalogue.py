#!/usr/bin/env python3
"""A catalogue image of the clean superstructures, rings coloured by size.

Every entry is built by this package at the size the measured tables in
CLAUDE.md give as clean, cached as JSON so a second run only re-renders,
and drawn with each ring filled by its size -- pentagons red, hexagons
grey, heptagons blue, octagons green -- because the point of these
structures is *where* the non-hexagons sit, which an atoms-and-bonds view
does not show. Labels are in Spanish by default (``--lang en`` for
English).

    python -m nanocarbon_lab.examples.ex12_catalogue --out catalogo/
    python -m nanocarbon_lab.examples.ex12_catalogue --out catalogo/ --quick
    python -m nanocarbon_lab.examples.ex12_catalogue --out catalogo/ \\
        --lang en --dpi 400 --panels

``--dpi`` sets the grid's resolution (170 gives about 2700 px across, 400
about 6400). ``--panels`` also writes one 3840 x 2160 (4K) image per
structure into ``panels_<lang>/``, which is what to open to zoom in on a
node.

The knee and Goldberg entries have **no sphere at the nodes**: the node
is the junction of the tubes itself, a saddle paid in heptagons (and, on
a planar crossing, four pentagons at its poles). They exist only for
periodic nets -- a finite cage's corner is convex and cannot be built that
way -- and are the fewest-atom route to the same net.

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
    ("stube", "Supertubo (4,4), un periodo", "geodesic",
     {"graph": "supertube-(4,4)", "m": 1, "freq": 4, "rows": 2}, False),
    ("pchiral", "Schwarz P quiral GC(3,1)", "goldberg",
     {"kind": "primitive", "m": 3, "n": 1}, True),
    ("fau", "Schwarz D GC(2,0), conmensurable con FAU", "goldberg",
     {"kind": "diamond", "m": 2, "n": 0}, True),
    ("dchiral", "Schwarz D quiral GC(2,1)", "goldberg",
     {"kind": "diamond", "m": 2, "n": 1}, True),
    ("pgc11", "Schwarz P GC(1,1), sin esferas", "goldberg",
     {"kind": "primitive", "m": 1, "n": 1}, True),
    ("dgc10", "Schwarz D GC(1,0), solo heptágonos", "goldberg",
     {"kind": "diamond", "m": 1, "n": 0}, True),
    ("kneep", "Schwarz P (codos, sin esferas)", "knee_schwarzite",
     {"kind": "primitive", "circumference": 20, "arm_rows": 9}, True),
    ("kneed", "Schwarz D (codos, sin esferas)", "knee_schwarzite",
     {"kind": "diamond", "circumference": 18, "arm_rows": 5}, True),
    ("kneeg", "Giroide (codos, sin esferas)", "knee_schwarzite",
     {"kind": "gyroid", "circumference": 20, "arm_rows": 5}, True),
    ("kneesg", "Súper-grafeno (codos, sin esferas)", "knee_supernetwork",
     {"net": "super-graphene", "circumference": 22, "arm_rows": 5}, True),
    ("kneess", "Súper-cuadrada (codos, sin esferas)", "knee_supernetwork",
     {"net": "super-square", "circumference": 12, "arm_rows": 5}, True),
    ("kneelon", "D hexagonal, lonsdaleíta (codos, sin esferas)",
     "knee_schwarzite",
     {"kind": "lonsdaleite", "circumference": 18, "arm_rows": 5}, True),
    ("kneebcc", "I-WP, red bcc (codos, sin esferas)", "knee_schwarzite",
     {"kind": "bcc", "circumference": 36, "arm_rows": 13}, True),
    ("dbldia", "Doble diamante interpenetrado (codos)", "knee_schwarzite",
     {"kind": "diamond", "circumference": 18, "arm_rows": 7,
      "double": True}, True),
    ("dblgyr", "Doble giroide interpenetrado (geodésico)", "geodesic",
     {"graph": "super-srs", "m": 1, "freq": 4, "rows": 1,
      "window_radius": 3, "double": True}, False),
    ("dblprim", "Doble primitiva interpenetrada (geodésica)", "geodesic",
     {"graph": "super-cubic", "m": 2, "freq": 2, "rows": 2,
      "double": True}, True),
    ("t31212", "Lámina 3.12.12 (Wythoff)", "geodesic",
     {"graph": "wythoff-truncated-hexagonal", "m": 1, "freq": 4,
      "rows": 2}, False),
    ("st60", "Supertubo (6,0), un periodo", "geodesic",
     {"graph": "supertube-(6,0)", "m": 1, "freq": 2, "rows": 1}, True),
    ("st80", "Supertubo (8,0), un periodo", "geodesic",
     {"graph": "supertube-(8,0)", "m": 1, "freq": 4, "rows": 2}, False),
    ("kneey", "Unión Y (codos, sin esfera)", "knee_junction",
     {"kind": "y", "circumference": 14, "arm_rows": 9}, True),
    ("kneet", "Unión tetraédrica (codos, sin esfera)", "knee_junction",
     {"kind": "tetrahedral", "circumference": 18, "arm_rows": 9}, True),
]

#: English labels, by entry key. A key missing here falls back to the
#: Spanish label, and a test keeps the two tables in step.
LABELS_EN: dict[str, str] = {
    "cube": "Cube cage", "tetra": "Tetrahedral cage",
    "octa": "Octahedral cage", "ico": "Icosahedral cage",
    "dodeca": "Dodecahedral cage", "cubocta": "Cuboctahedron",
    "icosido": "Icosidodecahedron", "ttetra": "Truncated tetrahedron",
    "tocta": "Truncated octahedron", "rhombi": "Rhombicuboctahedron",
    "hyper": "Hypercube (tesseract)",
    "wtico": "Truncated icosidodecahedron (Wythoff)",
    "wtcubo": "Truncated cuboctahedron (Wythoff)",
    "wrico": "Rhombicosidodecahedron (Wythoff)",
    "wtdod": "Truncated dodecahedron (Wythoff)",
    "c60": "C60 of tubes", "scubic": "Super-cubic", "ssquare": "Super-square",
    "sbcc": "Super-bcc", "sgraph": "Super-graphene",
    "skagome": "Super-kagome", "ssrs": "Super-srs (gyroid)",
    "sdia": "Super-diamond", "sfcc": "Super-fcc",
    "slon": "Super-lonsdaleite", "stube": "Supertube (4,4), one period",
    "pchiral": "Chiral Schwarz P GC(3,1)",
    "fau": "Schwarz D GC(2,0), FAU-commensurate",
    "dchiral": "Chiral Schwarz D GC(2,1)",
    "pgc11": "Schwarz P GC(1,1), no spheres",
    "dgc10": "Schwarz D GC(1,0), heptagons only",
    "kneep": "Schwarz P (knees, no spheres)",
    "kneed": "Schwarz D (knees, no spheres)",
    "kneeg": "Gyroid (knees, no spheres)",
    "kneesg": "Super-graphene (knees, no spheres)",
    "kneess": "Super-square (knees, no spheres)",
    "kneelon": "Hexagonal D, lonsdaleite (knees, no spheres)",
    "kneebcc": "I-WP, bcc net (knees, no spheres)",
    "dbldia": "Interpenetrated double diamond (knees)",
    "dblgyr": "Interpenetrated double gyroid (geodesic)",
    "dblprim": "Interpenetrated double primitive (geodesic)",
    "t31212": "3.12.12 sheet (Wythoff)",
    "st60": "Supertube (6,0), one period",
    "st80": "Supertube (8,0), one period",
    "kneey": "Y junction (knees, no sphere)",
    "kneet": "Tetrahedral junction (knees, no sphere)",
}

#: Every string the image draws besides the labels, per language.
TEXT: dict[str, dict[str, str]] = {
    "es": {"atoms": "átomos", "cell": "celda",
           "title": "Superestructuras de carbono cristalinas — "
                    "nanocarbon_lab (pentágonos rojo, heptágonos azul, "
                    "octágonos verde)",
           "clean": "limpio", "strained": "tensionado", "broken": "roto"},
    "en": {"atoms": "atoms", "cell": "cell",
           "title": "Crystalline carbon superstructures — nanocarbon_lab "
                    "(pentagons red, heptagons blue, octagons green)",
           "clean": "clean", "strained": "strained", "broken": "broken"},
}


def label(entry: tuple, lang: str) -> str:
    """The entry's label in `lang` ("es" or "en")."""
    return LABELS_EN.get(entry[0], entry[1]) if lang == "en" else entry[1]

#: Ring fill by size; anything else is drawn black so it cannot hide.
RING_COLOURS = {5: "#d62728", 6: "#bdbdbd", 7: "#1f5fbf", 8: "#2ca02c"}
#: Hexagons of the second network of an interpenetrated pair.
SECOND_NETWORK_HEXAGON = "#e8b56a"


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
    elif builder == "goldberg":
        from nanocarbon_lab.builders.goldberg import build_goldberg

        atoms = build_goldberg(**kwargs)
    else:
        from nanocarbon_lab.builders import knee

        atoms = getattr(knee, f"build_{builder}")(**kwargs)
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
        # None when the builder does not measure it (the knee schwarzite).
        "placed": (None if atoms.info.get("disclinations_placed") is None
                   else float(atoms.info["disclinations_placed"])),
        "bonds": [float(geometry["bond_min"]), float(geometry["bond_max"])],
        "seconds": round(time.time() - started, 1),
        # An interpenetrated pair: the second network's atoms are the
        # second half, and its hexagons are drawn in their own colour.
        "networks": int(atoms.info.get("networks", 1)),
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


def _panel(ax, record: dict, label: str, lang: str = "es",
           fontsize: float = 7, linewidth: float = 0.05, zoom: float = 1.0,
           sheet_elevation: float = 70) -> None:
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
    second_from = (len(positions) // 2 if record.get("networks", 1) == 2
                   else None)
    for ring in record["rings"]:
        corners = _unwrap(positions[ring], cell, record["pbc"])
        if live.any():
            centre = corners.mean(axis=0)
            shift = np.zeros(3)
            shift[live] = np.floor(centre[live] / cell[live]) * cell[live]
            corners = corners - shift
        colour = RING_COLOURS.get(len(ring), "#000000")
        if second_from is not None and len(ring) == 6 and ring[0] >= second_from:
            colour = SECOND_NETWORK_HEXAGON
        for image in images:
            polygons.append(corners + image)
            colours.append(colour)
    collection = Poly3DCollection(
        polygons, facecolors=colours, edgecolors="#333333",
        linewidths=linewidth, alpha=1.0)
    # A 3D axes clips to its square box; a zoomed panel spills past it
    # into the rest of a wide frame, so it is not clipped.
    collection.set_clip_on(zoom == 1.0)
    ax.add_collection3d(collection)
    every = np.vstack(polygons)
    centre = every.mean(axis=0)
    half = float(np.ptp(every, axis=0).max()) / 2.0
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre,
                         strict=True):
        setter(c - half, c + half)
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    sheet = int(live.sum()) == 2
    ax.view_init(elev=sheet_elevation if sheet else 22, azim=35)
    ax.set_axis_off()
    census = " ".join(f"{k}:{v}" for k, v in sorted(
        record["census"].items(), key=lambda kv: int(kv[0])) if k != "6")
    lo, hi = record["bonds"]
    text = TEXT[lang]
    tiled = "" if not live.any() else f" ({text['cell']} ×" + "×".join(
        str(r) for r, on in zip(reps, live, strict=True) if on) + ")"
    verdict = text.get(record["verdict"], record["verdict"])
    placed = record.get("placed")
    placed = "" if placed is None else f" · {placed:.0%}"
    number = f"{len(positions):,}".replace(",", " ")
    ax.set_title(f"{label}{tiled}\n{number} {text['atoms']} · {census}\n"
                 f"{lo:.3f}–{hi:.3f} Å · {verdict}{placed}",
                 fontsize=fontsize)


def render(entries: list[tuple], cache: Path, out: Path,
           columns: int = 5, lang: str = "es", dpi: int = 170) -> Path:
    """Draw every cached entry into one grid image; returns its path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(entries) / columns))
    fig = plt.figure(figsize=(3.2 * columns, 3.4 * rows))
    # Edges are drawn in points, so they thin out as the dpi rises; keep
    # them about the same number of pixels wide.
    width = 0.05 * max(1.0, dpi / 170)
    for k, entry in enumerate(entries):
        record = json.loads((cache / f"{entry[0]}.json").read_text())
        ax = fig.add_subplot(rows, columns, k + 1, projection="3d")
        _panel(ax, record, label(entry, lang), lang, linewidth=width)
    fig.suptitle(TEXT[lang]["title"], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def render_panels(entries: list[tuple], cache: Path, folder: Path,
                  lang: str = "es") -> list[Path]:
    """One 3840 x 2160 image per entry, for zooming into a node."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for entry in entries:
        record = json.loads((cache / f"{entry[0]}.json").read_text())
        fig = plt.figure(figsize=(12.8, 7.2))
        ax = fig.add_axes((0.0, 0.0, 1.0, 0.9), projection="3d")
        # A 16:9 frame leaves a cube-shaped 3D box using less than half
        # of it, so the panel zooms in; a sheet is tilted further down so
        # its tubes read as tubes rather than as holes in a plane.
        _panel(ax, record, label(entry, lang), lang, fontsize=14,
               linewidth=0.08, zoom=1.3, sheet_elevation=40)
        path = folder / f"{entry[0]}.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--quick", action="store_true",
                        help="Only the entries that build in minutes.")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--lang", choices=sorted(TEXT), default="es")
    parser.add_argument("--dpi", type=int, default=170,
                        help="Grid resolution (400 is about 6400 px wide).")
    parser.add_argument("--panels", action="store_true",
                        help="Also one 4K image per structure.")
    parser.add_argument("--only-render", action="store_true",
                        help="Draw the cached entries; build nothing.")
    args = parser.parse_args(argv)
    out = Path(args.out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    entries = [e for e in ENTRIES if e[4] or not args.quick]
    if args.only_render:
        entries = [e for e in entries if (cache / f"{e[0]}.json").exists()]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for path in pool.map(_build, entries, [cache] * len(entries)):
                print(f"  built {path.name}", flush=True)
    name = "catalogo.png" if args.lang == "es" else "catalogue.png"
    image = render(entries, cache, out / name, lang=args.lang, dpi=args.dpi)
    print(f"catalogue: {image}")
    if args.panels:
        folder = out / f"panels_{args.lang}"
        print(f"panels: {len(render_panels(entries, cache, folder, args.lang))}"
              f" in {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
