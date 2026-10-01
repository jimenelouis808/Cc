"""One frequency scale factor per parameter set, against GPAW (Scott-Radom least squares).

Run (no GPAW needed; the frequencies are already in the files)::

    python -m tbkit.recipes.frequency_scaling

As Witek and Morokuma did for SCC-DFTB (J. Comput. Chem. 25, 1858 (2004)):
λ = Σ ω_TB ω_ref / Σ ω_TB² over every internal mode of the set's validation
molecules, here against GPAW (PBE) rather than experiment. The frequencies
are recomputed from the set as it stands (the model at its own minimum against
GPAW's Hessians, ``fit.hessians``; ``xu_chn``'s five molecules from
``gpaw_chn_frequencies``), not taken from ``fit.frequencies``: a term added
after the fit (the H···H contact) changes them. Each set gets ``frequency_scale`` with the factor,
the RMS before and after, and the RMS with every molecule left out of its own
factor (how well λ transfers). One factor per set: two (above and below
2000 cm⁻¹) only helped xu_chno (99 -> 88 cm⁻¹) and is not worth a second number.

The factor never changes the model: :func:`tbkit.tasks.phonons` reports
``frequencies_scaled_cm1`` next to the raw frequencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SETS = ("xu_chn", "xu_chno", "xu_chnob", "xu_chnos", "xu_chnop", "xu_chnose")
MIN_CM1 = 50.0          # below: rigid-body leftovers, not vibrations
ROOT = Path(__file__).resolve().parents[1] / "parameters"


def _factor(tb: np.ndarray, ref: np.ndarray) -> float:
    return float(tb @ ref / (tb @ tb))


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def xu_chn_frequencies() -> dict:
    """xu_chn at its own minima against GPAW, for the molecules of gpaw_chn_frequencies."""
    from ..params import load_parameters
    from ..references import load_references
    from .xu_family import HessianTarget, frequency_validation

    data = json.loads((ROOT / "references" / "gpaw_chn_frequencies.json").read_text())
    refs, _ = load_references(ROOT / "references" / "gpaw_chn.json")
    geometry = {r.group: r.atoms for r in refs if r.label.endswith("/eq")}
    targets = [HessianTarget(name, geometry[name], None, np.array(freqs))
               for name, freqs in data["frequencies"].items()]
    return frequency_validation(load_parameters("xu_chn"), targets)


def model_frequencies(name: str, data: dict) -> dict:
    """The set as it is now (terms added after its fit included) against GPAW's Hessians."""
    from ..params import load_parameters
    from .xu_family import frequency_validation, load_hessians

    hessians = data["fit"].get("hessians")
    if not hessians:
        return xu_chn_frequencies()
    targets = load_hessians(ROOT / "references" / hessians)
    return frequency_validation(load_parameters(name), targets)


def scale_factor(frequencies: dict) -> dict:
    """λ and its errors from ``{molecule: {"tb": [...], "gpaw": [...]}}``."""
    pairs = {m: (np.asarray(v["tb"], float), np.asarray(v["gpaw"], float))
             for m, v in frequencies.items()}

    def modes(items):
        tb = np.concatenate([t for t, _ in items])
        ref = np.concatenate([g for _, g in items])
        keep = tb > MIN_CM1
        return tb[keep], ref[keep]

    tb, ref = modes(pairs.values())
    lam = _factor(tb, ref)
    left_out = []
    for name, (t, g) in pairs.items() if len(pairs) > 1 else ():
        rest = modes([p for k, p in pairs.items() if k != name])
        keep = t > MIN_CM1
        left_out.append(_factor(*rest) * t[keep] - g[keep])
    return {"value": round(lam, 4), "unit": "",
            "modes": len(tb), "molecules": len(pairs),
            "rms_before_cm1": round(_rms(tb - ref), 1),
            "rms_after_cm1": round(_rms(lam * tb - ref), 1),
            "rms_left_out_cm1": (round(_rms(np.concatenate(left_out)), 1) if left_out
                                 else None)}


def main() -> None:
    for name in SETS:
        path = ROOT / f"{name}.json"
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        frequencies = model_frequencies(name, data)
        entry = scale_factor(frequencies)
        entry["source"] = ("mínimos cuadrados (Scott-Radom; Witek y Morokuma, J. Comput. "
                           "Chem. 25, 1858 (2004)) frente a GPAW PBE, modos internos de las "
                           "moléculas de validación; receta tbkit.recipes.frequency_scaling")
        data["frequency_scale"] = entry
        path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"{name:10s} λ = {entry['value']:.4f}  RMS {entry['rms_before_cm1']:.0f} -> "
              f"{entry['rms_after_cm1']:.0f} cm⁻¹ (dejando fuera cada molécula: "
              f"{entry['rms_left_out_cm1']:.0f})")


if __name__ == "__main__":
    main()
