"""GPAW (FD) polarizabilities of the C/H/N reference molecules.

Run (needs GPAW; about an hour on four cores)::

    python -m tbkit.recipes.chn_polarizability OUT.json --workers 4

Each molecule at its GPAW-relaxed geometry from ``parameters/references/
gpaw_chn.json``; α by ±0.01 V/Å finite field on a real-space grid (FD,
h = 0.18 Å, PBE), which an LCAO dzp basis would underestimate. The training
set fixes the extra atomic polarizabilities of ``xu_chn``
(:mod:`tbkit.recipes.xu_chn_alpha`); the test set only checks them.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

TRAINING = ("CH4", "C2H6", "C2H4", "C2H2", "C6H6", "NH3", "HCN", "N2", "H3CNH2", "C5H5N",
            "CH3CN", "N2H4")
TEST = ("C3H8", "butadiene", "C4H4NH", "isobutene", "H2CCHCN")
#: Oxygen set (geometries from gpaw_chno.json), for ``--set chno``.
TRAINING_O = ("H2O", "CH3OH", "H2CO", "HCOOH", "CO2", "CO", "CH3OCH3", "C4H4O", "CH3COOH")
TEST_O = ("CH3CH2OH", "CH3COCH3", "HCOOCH3")
#: Boron, sulfur, phosphorus (geometries from gpaw_b/s/p.json), ``--set b|s|p``.
TRAINING_B = ("BH3", "BMe3", "B(OH)3", "MeB(OH)2", "borazine", "H3BNH3")
TEST_B = ("PhB(OH)2", "B(OMe)3")
TRAINING_S = ("H2S", "CH3SH", "CH3SCH3", "CH3SSCH3", "thiophene", "CS2", "DMSO", "SO2")
TEST_S = ("CH3CH2SH", "Me2SO2", "CH3SO3H")
TRAINING_P = ("PH3", "CH3PH2", "PMe3", "OPMe3", "H3PO4", "phosphinine")
TEST_P = ("MePO3H2", "PhPH2")
#: Selenium (geometries from gpaw_se.json), ``--set se``.
TRAINING_SE = ("H2Se", "CH3SeH", "CH3SeCH3", "CH3SeSeCH3", "selenophene", "CSe2", "DMSeO",
               "SeO2")
TEST_SE = ("CH3CH2SeH", "PhSeH")
SETS = {"chn": (TRAINING, TEST, "gpaw_chn.json"), "chno": (TRAINING_O, TEST_O, "gpaw_chno.json"),
        "b": (TRAINING_B, TEST_B, "gpaw_b.json"), "s": (TRAINING_S, TEST_S, "gpaw_s.json"),
        "p": (TRAINING_P, TEST_P, "gpaw_p.json"),
        "se": (TRAINING_SE, TEST_SE, "gpaw_se.json")}


def _one(args):
    name, role, settings, geometries = args
    from tbkit.references import gpaw_polarizability, load_references

    refs, _ = load_references(geometries)
    ref = next(r for r in refs if r.label == f"{name}/eq")
    alpha = gpaw_polarizability(ref.atoms, settings)
    return {"group": name, "role": role, "label": ref.label,
            "alpha": [[round(float(v), 5) for v in row] for row in alpha]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path,
                        help="archivo JSON de referencias que se escribe (los cálculos parciales "
                             "van a ARCHIVO.parts/, reanudable)")
    parser.add_argument("--workers", type=int, default=4,
                        help="procesos GPAW en paralelo (cada uno, un cálculo; vigila la memoria)")
    parser.add_argument("--set", choices=sorted(SETS), default="chn",
                        help="conjunto de moléculas: chn o chno")
    parser.add_argument("--geometries", type=Path, default=None,
                        help="archivo de referencias con las geometrías (por defecto, el del "
                             "paquete para el conjunto elegido)")
    args = parser.parse_args(argv)
    from tbkit.params import PARAMETER_DIR

    training, test, default_file = SETS[args.set]
    geometries = args.geometries or PARAMETER_DIR / "references" / default_file
    from tbkit.references import GPAW_ALPHA_DEFAULTS, gpaw_settings_record

    settings = dict(GPAW_ALPHA_DEFAULTS)
    jobs = [(n, "train", settings, geometries) for n in training] + \
        [(n, "test", settings, geometries) for n in test]
    parts = args.out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    todo = [job for job in jobs if not (parts / f"{job[0]}.json").exists()]
    from ..progress import Progress

    bar = Progress(len(jobs), "α de referencia GPAW", status=parts / "progreso.json",
                   done=len(jobs) - len(todo))
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(_one, job): job for job in todo}
        from concurrent.futures import as_completed

        for future in as_completed(futures):
            result = future.result()
            (parts / f"{result['group']}.json").write_text(json.dumps(result))
            bar.step(note=result['group'])
            print(f"{result['group']}: α medio "
                  f"{sum(result['alpha'][i][i] for i in range(3)) / 3:.3f} Å³", flush=True)
    entries = [json.loads((parts / f"{n}.json").read_text()) for n, _, _, _ in jobs]
    record = gpaw_settings_record(settings)
    for key in ("basis", "extra_bands", "fmax"):
        record.pop(key, None)
    args.out.write_text(json.dumps({"settings": record, "unit": "Å^3",
                                    "polarizabilities": entries}, indent=1,
                                   ensure_ascii=False), encoding="utf-8")
    print(f"{len(entries)} polarizabilidades en {args.out}")


if __name__ == "__main__":
    main()
