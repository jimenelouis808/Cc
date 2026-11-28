"""Write every geodesic, Goldberg and knee superstructure preset as extxyz.

    python -m nanocarbon_lab.examples.ex13_export_structures <folder>
    python -m nanocarbon_lab.examples.ex13_export_structures <folder> \\
        --cache catalogo/cache --workers 4 --max-atoms 20000

The big superstructures take minutes to hours each, so a finished one
is worth keeping. One file per distinct build, named by its parameters
(``<graph>_m1n0_f4_r2_w1.xyz``) so two presets that ask for the same
build share a file, and a file that already exists is not rebuilt: the
script can be stopped and run again. ``--cache`` converts the catalogue
image's JSON cache (`ex12_catalogue.py`) instead of rebuilding those.

Each file carries its cell, periodicity and, in the comment line, the
parameters, the ring census, the sp2 verdict and the disclination
placement, so a file says what it is without this package. ``index.csv``
lists them with the preset names that point at each one.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

GEODESIC = "superstructure (geodesic)"
GOLDBERG = "superstructure (Goldberg)"
#: Knee modes -> the `builders.knee` function that builds them. Their
#: nodes are the junction of the tubes, with no sphere.
KNEE = {"schwarzite (knees)": "knee_schwarzite",
        "junction (knees)": "knee_junction",
        "supernetwork (knees)": "knee_supernetwork"}


def _key(spec: dict) -> str:
    """File stem naming one build by its parameters."""
    if spec["builder"] == "goldberg":
        return f"goldberg-{spec['kind']}_m{spec['m']}n{spec['n']}"
    double = "_double" if spec.get("double") else ""
    if spec["builder"].startswith("knee_"):
        kind = spec.get("kind", spec.get("net"))
        return (f"{spec['builder'].replace('_', '-')}-{kind}"
                f"_k{spec['circumference']}_r{spec['arm_rows']}{double}")
    return (f"{spec['graph']}_m{spec['m']}n{spec['n']}_f{spec['freq']}"
            f"_r{spec['rows']}_w{spec['window_radius']}{double}")


def preset_specs() -> dict[str, dict]:
    """Every geodesic and Goldberg preset, as builder arguments, by file stem."""
    from nanocarbon_lab.builders.geodesic import GEODESIC_CAGES
    from nanocarbon_lab.presets import PRESETS, knee_params

    specs: dict[str, dict] = {}
    for label, p in PRESETS.items():
        if p.get("mode_kind") == GEODESIC:
            spec = {"builder": "geodesic", "graph": p["gd_graph"],
                    "m": int(p["gd_m"]), "n": int(p.get("gd_n", 0)),
                    "freq": int(p["gd_freq"]), "rows": int(p["gd_rows"]),
                    "window_radius": int(p.get("gd_wr", 1)),
                    "tolerance_deg": float(GEODESIC_CAGES.get(
                        p["gd_graph"], {}).get("tolerance_deg", 2.5))}
            if p.get("gd_double"):
                spec["double"] = True
        elif p.get("mode_kind") == GOLDBERG:
            m = int(p["gb_m"])
            spec = {"builder": "goldberg", "kind": p["gb_kind"], "m": m,
                    "n": m if p.get("gb_class2") else int(p.get("gb_n", 0))}
        elif p.get("mode_kind") in KNEE:
            spec = {"builder": KNEE[p["mode_kind"]],
                    **knee_params(p["mode_kind"], p)}
        else:
            continue
        specs.setdefault(_key(spec), {**spec, "presets": []})
        specs[_key(spec)]["presets"].append(label)
    return specs


def estimate(spec: dict) -> int:
    """Atom count before the build (0 when unknown), for ordering."""
    if spec["builder"] != "geodesic":
        return 0
    from nanocarbon_lab.builders.geodesic import geodesic_atom_count

    try:
        return (2 if spec.get("double") else 1) * geodesic_atom_count(
            spec["graph"], freq=spec["freq"], m=spec["m"], n=spec["n"],
            rows=spec["rows"], window_radius=spec["window_radius"],
            tolerance_deg=spec["tolerance_deg"])
    except ValueError:
        return 0


def _write(path: Path, positions, cell, pbc, info: dict, tags=None) -> None:
    from ase import Atoms
    from ase.io import write

    atoms = Atoms("C" * len(positions), positions=positions, pbc=pbc)
    if tags is not None and np.any(tags):
        # Which network each atom is in, for an interpenetrated pair.
        atoms.set_tags(tags)
    cell = np.asarray(cell, dtype=float)
    atoms.set_cell(np.diag(cell) if cell.ndim == 1 else cell)
    atoms.info.update({k: v for k, v in info.items()})
    tmp = path.with_suffix(".tmp")
    write(tmp, atoms, format="extxyz")
    tmp.replace(path)


def build_one(stem: str, spec: dict, out: Path) -> dict:
    """Build one spec (unless its file exists) and return its index row."""
    path = out / f"{stem}.xyz"
    if path.exists():
        return {"file": path.name, "status": "exists"}
    from nanocarbon_lab.validation.quality import sp2_quality

    started = time.time()
    try:
        if spec["builder"] == "geodesic":
            from nanocarbon_lab.builders.geodesic import (
                build_geodesic_superstructure,
            )
            atoms = build_geodesic_superstructure(
                spec["graph"], m=spec["m"], n=spec["n"], freq=spec["freq"],
                rows=spec["rows"], window_radius=spec["window_radius"],
                tolerance_deg=spec["tolerance_deg"],
                double=bool(spec.get("double", False)))
        elif spec["builder"] == "goldberg":
            from nanocarbon_lab.builders.goldberg import build_goldberg
            atoms = build_goldberg(kind=spec["kind"], m=spec["m"], n=spec["n"])
        else:
            from nanocarbon_lab.builders import knee
            atoms = getattr(knee, f"build_{spec['builder']}")(
                **{k: v for k, v in spec.items()
                   if k not in ("builder", "presets")})
    except Exception as exc:  # recorded, so one failure does not stop the rest
        return {"file": path.name, "status": f"failed: {exc}"[:200]}
    geometry = atoms.info["geometry"]
    census = " ".join(f"{k}:{v}" for k, v in
                      sorted(atoms.info["ring_counts"].items()))
    info = {k: v for k, v in spec.items() if k not in ("presets",)}
    info.update(census=census, verdict=str(sp2_quality(geometry)[0]),
                # -1 where the builder does not measure the placement.
                placed=float(atoms.info.get("disclinations_placed", -1.0)),
                bond_min=float(geometry["bond_min"]),
                bond_max=float(geometry["bond_max"]),
                seconds=round(time.time() - started))
    if "network_gap" in atoms.info:
        info["network_gap"] = float(atoms.info["network_gap"])
    _write(path, atoms.get_positions(), atoms.cell[:], atoms.pbc, info,
           atoms.get_tags())
    return {"file": path.name, "status": "built", "atoms": len(atoms), **info}


def convert_cache(cache: Path, out: Path) -> list[dict]:
    """Write the catalogue image's cached builds as extended XYZ."""
    from nanocarbon_lab.examples.ex12_catalogue import ENTRIES

    rows = []
    for key, label, builder, kwargs, _quick in ENTRIES:
        record_path = cache / f"{key}.json"
        if not record_path.exists():
            continue
        record = json.loads(record_path.read_text())
        if builder == "geodesic":
            spec = {"builder": "geodesic", "graph": kwargs["graph"],
                    "m": kwargs.get("m", 2), "n": kwargs.get("n", 0),
                    "freq": kwargs.get("freq", 2),
                    "rows": kwargs.get("rows", 1),
                    "window_radius": kwargs.get("window_radius", 1)}
            if kwargs.get("double"):
                spec["double"] = True
        elif builder.startswith("knee_"):
            spec = {"builder": builder, **kwargs}
        else:
            spec = {"builder": "goldberg", "kind": kwargs["kind"],
                    "m": kwargs["m"], "n": kwargs.get("n", 0)}
        path = out / f"{_key(spec)}.xyz"
        if path.exists():
            continue
        census = " ".join(f"{k}:{v}" for k, v in sorted(
            record["census"].items(), key=lambda kv: int(kv[0])))
        info = {**spec, "census": census, "verdict": record["verdict"],
                "placed": (-1.0 if record["placed"] is None
                           else record["placed"]), "bond_min": record["bonds"][0],
                "bond_max": record["bonds"][1], "label": label}
        _write(path, record["positions"], record["cell"], record["pbc"], info)
        rows.append({"file": path.name, "status": "from cache",
                     "atoms": len(record["positions"]), **info})
    return rows


def write_index(out: Path, specs: dict[str, dict]) -> Path:
    """index.csv from the files themselves, so it never disagrees with them."""
    from ase.io import read

    presets = {f"{stem}.xyz": "; ".join(s["presets"])
               for stem, s in specs.items()}
    fields = ["file", "atoms", "verdict", "placed", "bond_min", "bond_max",
              "census", "presets"]
    index = out / "index.csv"
    with index.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in sorted(out.glob("*.xyz")):
            atoms = read(path, format="extxyz")
            writer.writerow({
                "file": path.name, "atoms": len(atoms),
                "verdict": atoms.info.get("verdict", ""),
                "placed": atoms.info.get("placed", ""),
                "bond_min": f"{atoms.info.get('bond_min', 0):.3f}",
                "bond_max": f"{atoms.info.get('bond_max', 0):.3f}",
                "census": atoms.info.get("census", ""),
                "presets": presets.get(path.name,
                                       atoms.info.get("label", "")),
            })
    return index


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--cache", type=Path, default=None,
                        help="the catalogue's JSON cache, converted first")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=None,
                        help="skip builds estimated above this many atoms")
    parser.add_argument("--only", default=None,
                        help="substring a file stem must contain")
    args = parser.parse_args(argv)
    out = args.folder
    out.mkdir(parents=True, exist_ok=True)

    if args.cache:
        for row in convert_cache(args.cache, out):
            print(f"{row['file']}: {row['atoms']} atoms, {row['status']}",
                  flush=True)
    specs = preset_specs()
    todo = [(estimate(s), stem, s) for stem, s in specs.items()
            if not (out / f"{stem}.xyz").exists()
            and (args.only is None or args.only in stem)]
    if args.max_atoms:
        todo = [t for t in todo if t[0] <= args.max_atoms]
    todo.sort()
    print(f"{len(specs)} distinct builds, {len(todo)} to do", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(build_one, stem, spec, out)
                   for _, stem, spec in todo]
        for future in futures:
            row = future.result()
            print(f"{row['file']}: {row.get('atoms', '-')} atoms, "
                  f"{row['status']} {row.get('verdict', '')}", flush=True)
            write_index(out, specs)
    print(f"index: {write_index(out, specs)}")


if __name__ == "__main__":
    main()
