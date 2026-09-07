"""Atomic form factors for X-rays.

One function matters, :func:`form_factor`, and it evaluates the analytical
approximation of *International Tables for Crystallography* volume C,
table 6.1.1.4:

    f₀(s) = Z − 41.78214 · s² · Σᵢ aᵢ exp(−bᵢ s²),   s = sin θ / λ  [Å⁻¹]

Two properties of that form are worth knowing when reading the output.
It is **exact at s = 0**, where f₀ = Z by construction, and it is fitted
only up to s ≈ 2 Å⁻¹ — which is never a limitation for a laboratory
diffractometer, since Cu Kα cannot reach past s = 1/1.5406 = 0.649 Å⁻¹
even at 2θ = 180°.

**Anomalous dispersion is not included.** f′ and f″ shift the form factor
when the photon energy is near an absorption edge, which for Cu Kα means
iron, cobalt and manganese. The effect on calculated intensities is a few
per cent. The effect that actually ruins an iron measurement with a copper
tube is different and much larger: fluorescence, which raises the
*background* rather than changing any peak, and which
:func:`fluorescence_risk` warns about instead.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..database.loader import DATA_DIR

#: Characteristic wavelengths in Å of the usual laboratory anodes,
#: Kα₁ and the intensity-weighted Kα₁₂ average that a pattern measured
#: without a monochromator actually contains.
ANODES: dict[str, dict[str, float]] = {
    "Cu": {"ka1": 1.540598, "ka2": 1.544426, "kalpha": 1.541874, "kbeta": 1.392250,
           "edge_kev": 8.979},
    "Co": {"ka1": 1.788996, "ka2": 1.792835, "kalpha": 1.790260, "kbeta": 1.620790,
           "edge_kev": 7.709},
    "Fe": {"ka1": 1.936042, "ka2": 1.939980, "kalpha": 1.937355, "kbeta": 1.756610,
           "edge_kev": 7.112},
    "Mo": {"ka1": 0.709300, "ka2": 0.713590, "kalpha": 0.710730, "kbeta": 0.632250,
           "edge_kev": 20.000},
    "Cr": {"ka1": 2.289700, "ka2": 2.293606, "kalpha": 2.291000, "kbeta": 2.084870,
           "edge_kev": 5.989},
    "Ag": {"ka1": 0.559408, "ka2": 0.563798, "kalpha": 0.560870, "kbeta": 0.497069,
           "edge_kev": 25.514},
}

#: Elements whose K absorption edge sits just below the Cu Kα energy, so
#: that a copper tube excites their fluorescence.
FLUORESCES_UNDER_CU = ("Cr", "Mn", "Fe", "Co")


class ScatteringError(ValueError):
    """Raised for an element with no tabulated form factor."""


@lru_cache(maxsize=2)
def _load(directory: str) -> dict:
    path = Path(directory) / "scattering.json"
    if not path.is_file():
        raise ScatteringError(f"falta la tabla de dispersión: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _clean(symbol: str) -> str:
    """``Fe2+`` → ``Fe``, ``O-2`` → ``O``, ``Se1`` → ``Se``.

    CIF type symbols carry oxidation states and label digits. The neutral
    atom is used for all of them: the difference between f(Fe) and f(Fe³⁺)
    is three electrons out of twenty-six, and it is concentrated at low
    angle where the form factor is largest, so it perturbs intensities by a
    few per cent and never moves a peak.
    """
    letters = []
    for character in symbol.strip():
        if character.isalpha():
            letters.append(character)
        else:
            break
    cleaned = "".join(letters)
    if not cleaned:
        raise ScatteringError(f"no se reconoce el elemento en {symbol!r}")
    return cleaned[0].upper() + cleaned[1:].lower()


def atomic_number(symbol: str, directory: Optional[str | Path] = None) -> int:
    """Z for an element symbol, tolerating CIF oxidation states."""
    table = _load(str(Path(directory) if directory else DATA_DIR))["elements"]
    element = _clean(symbol)
    if element not in table:
        raise ScatteringError(f"el elemento {element!r} no está en la tabla")
    return int(table[element]["Z"])


def form_factor(
    symbol: str, s: np.ndarray | float, directory: Optional[str | Path] = None
) -> np.ndarray:
    """f₀ for one element at ``s = sinθ/λ`` in Å⁻¹.

    Parameters
    ----------
    symbol:
        Element symbol, with or without an oxidation state.
    s:
        Scalar or array of sinθ/λ, in Å⁻¹.
    directory:
        Where to load the table from.

    Returns
    -------
    numpy.ndarray
    """
    payload = _load(str(Path(directory) if directory else DATA_DIR))
    table = payload["elements"]
    element = _clean(symbol)
    if element not in table:
        raise ScatteringError(
            f"el elemento {element!r} no está en la tabla de factores de forma"
        )
    entry = table[element]
    values = np.atleast_1d(np.asarray(s, dtype=float))
    squared = values * values
    total = np.zeros_like(squared)
    for a, b in entry["ab"]:
        total += a * np.exp(-b * squared)
    return entry["Z"] - payload["prefactor"] * squared * total


def wavelength_for(anode: str, line: str = "kalpha") -> float:
    """Wavelength in Å for an anode and emission line.

    ``line`` is ``"ka1"``, ``"ka2"``, ``"kalpha"`` (the Kα₁₂ weighted mean)
    or ``"kbeta"``. The default is the Kα₁₂ mean, because that is what a
    pattern from a tube with only a Kβ filter actually contains, and using
    Kα₁ instead biases every refined lattice parameter by about 1 part in
    1000 — small, systematic, and exactly the size of the discrepancies
    people spend afternoons chasing.
    """
    if anode not in ANODES:
        raise ScatteringError(
            f"ánodo {anode!r} desconocido; disponibles: {', '.join(sorted(ANODES))}"
        )
    entry = ANODES[anode]
    if line not in entry:
        raise ScatteringError(
            f"línea {line!r} desconocida; disponibles: "
            + ", ".join(k for k in entry if k != "edge_kev")
        )
    return float(entry[line])


def fluorescence_risk(elements: Sequence[str], anode: str = "Cu") -> Optional[str]:
    """Warn when the sample will fluoresce under this tube.

    Iron under a copper tube is the classic case: the Cu Kα photon is just
    above the Fe K edge, iron fluoresces, and the fluorescence lands on the
    detector as a large flat background that buries weak reflections. It
    does not move or distort any peak — so the fix is not in the analysis,
    it is a monochromator, an energy-discriminating detector, or a cobalt
    tube.
    """
    if anode != "Cu":
        return None
    present = sorted({_clean(e) for e in elements} & set(FLUORESCES_UNDER_CU))
    if not present:
        return None
    return (
        f"la muestra contiene {', '.join(present)} y estás midiendo con ánodo "
        "de cobre: esos elementos fluorescen bajo Cu Kα y añaden un fondo "
        "grande y plano que entierra las reflexiones débiles. No deforma los "
        "picos ni los desplaza, así que no es un problema de análisis: se "
        "arregla con monocromador secundario, con un detector que discrimine "
        "en energía, o midiendo con ánodo de cobalto. Si el fondo de tu "
        "difractograma es enorme y liso, esta es la razón"
    )


__all__ = [
    "ANODES",
    "FLUORESCES_UNDER_CU",
    "ScatteringError",
    "atomic_number",
    "fluorescence_risk",
    "form_factor",
    "wavelength_for",
]
