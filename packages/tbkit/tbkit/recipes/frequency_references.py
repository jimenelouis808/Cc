"""GPAW harmonic frequencies at the relaxed geometries of a reference file.

Run (needs GPAW)::

    python -m tbkit.recipes.frequency_references gpaw_chno.json OUT.json \\
        --molecules H2O CH3OH HCOOH ... --workers 4

Each molecule's ``/eq`` structure (already relaxed with GPAW in the reference
file) goes through :func:`tbkit.references.gpaw_frequencies` (central
differences of the forces, same settings as the file). The frequencies are
validation of a parameter set and, once they enter the fit, a target: the
force RMS of a joint fit can hide a curvature 40 % off (``xu_chno`` gave the
O-H of formic acid at 4480 cm⁻¹, GPAW ~3600). Results are cached per
molecule next to ``OUT`` and never overwritten.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _one(args):
    name, references, settings = args
    from tbkit.references import gpaw_frequencies, load_references

    refs = []
    for path in references:
        refs += load_references(path)[0]
    eq = next(r for r in refs if r.label == f"{name}/eq")
    return name, [round(float(v), 2) for v in gpaw_frequencies(eq.atoms, settings)]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("references", type=Path, nargs="+")
    parser.add_argument("out", type=Path)
    parser.add_argument("--molecules", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    from tbkit.references import GPAW_DEFAULTS, gpaw_settings_record, load_references

    settings = dict(GPAW_DEFAULTS)
    _, file_settings = load_references(args.references[0])
    for key in ("xc", "mode", "basis", "h", "vacuum"):
        if key in file_settings:
            settings[key] = file_settings[key]
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [n for n in args.molecules if not (parts / f"{n}.json").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(_one, (n, [str(p) for p in args.references], settings))
                   for n in todo]
        for future in as_completed(futures):
            name, values = future.result()
            (parts / f"{name}.json").write_text(json.dumps(values))
            print(f"{name}: {len(values)} frecuencias", flush=True)
    frequencies = {n: json.loads((parts / f"{n}.json").read_text()) for n in args.molecules}
    if args.out.exists():
        raise SystemExit(f"{args.out} ya existe: no se sobrescribe.")
    args.out.write_text(json.dumps({
        "settings": gpaw_settings_record(settings), "unit": "cm^-1",
        "note": "armónicas en Γ, diferencias centrales de las fuerzas; imaginarias como "
                "negativas; incluyen los modos rígidos (≈ 0)",
        "references": [p.name for p in args.references], "frequencies": frequencies},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(frequencies)} moléculas en {args.out}")


if __name__ == "__main__":
    main()
