"""Leer lo que dejaron las recetas, y devolverlo en los objetos de siempre.

Los scripts que escribe :mod:`carbonforge.dft.recipe` dejan JSON. Este
módulo lo lee y construye los MISMOS objetos que
:mod:`carbonforge.results` usa para Quantum ESPRESSO y SIESTA
(:class:`~carbonforge.results.bands.BandStructure`,
:class:`~carbonforge.results.dos.DensityOfStates`), de modo que el dibujo,
las etiquetas del camino y las medidas ya escritas sirven igual. Un lector
nuevo con su propio dibujo habría duplicado todo eso y lo habría dejado
divergir.

Lo que no tiene equivalente --el LDOS por sitio y los fonones de Γ-- se
devuelve en dataclases propias, pequeñas y con la procedencia dentro.

**Nada aquí importa GPAW.** Un resultado se lee en la máquina donde se
analiza, que casi nunca es donde se calculó.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..results.bands import BandStructure
from ..results.dos import DensityOfStates

__all__ = [
    "GammaPhonons",
    "SiteDOS",
    "available_results",
    "read_bands",
    "read_dos",
    "read_ldos",
    "read_phonons",
]


def _load(path: str | Path, what: str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No hay {what} en {path}: el paso que lo escribe no ha corrido, "
            "o corrió y falló. Mira su .out.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as problem:
        raise ValueError(
            f"{path.name} no es JSON válido ({problem}). Un cálculo "
            "interrumpido deja el archivo a medias; vuelve a correr el paso."
        ) from problem


def read_bands(path: str | Path) -> BandStructure:
    """Las bandas que escribió el paso ``bands``.

    El formato es el de ``ase.spectrum.band_structure.BandStructure``, que
    guarda energías por punto k y el camino. Las distancias acumuladas se
    recalculan aquí porque ASE guarda los puntos, no el eje.
    """
    data = _load(path, "bandas")
    energies = np.asarray(data["energies"], dtype=float)
    if energies.ndim == 3:        # (espín, k, banda) -> se toma el primero
        energies = energies[0]
    path_data = data.get("path", {})
    kpoints = np.asarray(path_data.get("kpts", []), dtype=float)

    if kpoints.size:
        cell = np.asarray(path_data.get("cell", np.eye(3)), dtype=float)
        reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
        cartesian = kpoints @ reciprocal
        steps = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
        distances = np.concatenate([[0.0], np.cumsum(steps)])
    else:
        distances = np.arange(len(energies), dtype=float)

    special: list[tuple[float, str]] = []
    labels = path_data.get("special_points", {})
    if labels and kpoints.size:
        for name, point in labels.items():
            close = np.argmin(np.linalg.norm(kpoints - np.asarray(point), axis=1))
            if np.linalg.norm(kpoints[close] - np.asarray(point)) < 1e-5:
                special.append((float(distances[close]), str(name)))
        special.sort()

    return BandStructure(
        distances=distances, energies=energies,
        kpoints=kpoints if kpoints.size else None,
        fermi_energy=(float(data["reference"]) if data.get("reference")
                      is not None else None),
        special_points=special)


def read_dos(path: str | Path) -> tuple[DensityOfStates, dict[str, np.ndarray]]:
    """El DOS total y, si lo hay, lo proyectado.

    Devuelve ``(total, proyecciones)``; el diccionario está vacío cuando el
    paso se pidió sin proyección. Las proyecciones NO suman el total: la
    proyección sobre orbitales atómicos es incompleta, y presentarlas como
    una partición del total sería mentir sobre lo que miden.
    """
    data = _load(path, "DOS")
    energies = np.asarray(data["energies"], dtype=float)
    total = DensityOfStates(
        energies=energies,
        dos=np.asarray(data["total"], dtype=float),
        fermi_energy=float(data["fermi"]) if data.get("fermi") is not None
        else None)
    projections = {
        name: np.asarray(values, dtype=float)
        for group in ("por_elemento", "por_orbital")
        for name, values in (data.get(group) or {}).items()
    }
    return total, projections


@dataclass
class SiteDOS:
    """Densidad de estados proyectada sobre un átomo concreto."""

    index: int
    symbol: str
    energies: np.ndarray
    orbitals: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def total(self) -> np.ndarray:
        """La suma de los orbitales pedidos para este sitio."""
        if not self.orbitals:
            return np.zeros_like(self.energies)
        return np.sum(list(self.orbitals.values()), axis=0)


def read_ldos(path: str | Path) -> tuple[list[SiteDOS], Optional[float]]:
    """El LDOS por sitio, y el nivel de Fermi.

    Un sitio por átomo elegido. Comparar dos sitios sólo tiene sentido si
    se calcularon con el mismo ensanchamiento, que es por qué va guardado
    con el resultado.
    """
    data = _load(path, "LDOS")
    energies = np.asarray(data["energies"], dtype=float)
    sites = [
        SiteDOS(index=int(index), symbol=str(entry["simbolo"]),
                energies=energies,
                orbitals={name: np.asarray(values, dtype=float)
                          for name, values in entry["orbitales"].items()})
        for index, entry in sorted(data["sitios"].items(), key=lambda kv: int(kv[0]))
    ]
    fermi = float(data["fermi"]) if data.get("fermi") is not None else None
    return sites, fermi


@dataclass
class GammaPhonons:
    """Frecuencias de Γ, en cm⁻¹, con las imaginarias en negativo."""

    frequencies: np.ndarray

    @property
    def imaginary(self) -> np.ndarray:
        """Las imaginarias, que se guardan como negativas.

        Más de tres (seis en una molécula) significa que la geometría no
        era un mínimo del mismo modelo, y entonces ninguna frecuencia de la
        lista describe nada.
        """
        return self.frequencies[self.frequencies < -1.0]

    @property
    def is_a_minimum(self) -> bool:
        """Si lo que queda tras los modos rígidos es todo real."""
        return len(self.imaginary) == 0


def read_phonons(path: str | Path) -> GammaPhonons:
    """Las frecuencias de Γ que escribió el paso ``phonons``."""
    data = _load(path, "fonones")
    return GammaPhonons(
        frequencies=np.asarray(data["gamma_cm1"], dtype=float))


#: Qué archivo deja cada tipo de cálculo, y con qué lector se abre.
READERS = {
    "bands": ("bands.json", read_bands),
    "dos": ("dos.json", read_dos),
    "ldos": ("ldos.json", read_ldos),
    "phonons": ("phonons.json", read_phonons),
}


def available_results(directory: str | Path) -> dict[str, Path]:
    """Qué resultados hay ya en un directorio de cálculo.

    Es lo que la pestaña de resultados necesita para ofrecer sólo lo que
    existe, en vez de un menú lleno de cosas que van a fallar al abrirse.
    """
    directory = Path(directory)
    found = {}
    for kind, (name, _reader) in READERS.items():
        candidate = directory / name
        if candidate.exists():
            found[kind] = candidate
    return found
