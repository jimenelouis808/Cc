"""Reading and plotting densities of states.

Handles both outputs of the Quantum ESPRESSO workflow:

* ``dos.x`` writes a single ``dos.dat``: energy, total DOS, integrated DOS.
* ``projwfc.x`` writes one file per atom and orbital, named
  ``pdos.pdos_atm#12(N)_wfc#2(p)``, plus a summed ``pdos.pdos_tot``. The
  filenames carry the atom index, element and orbital, so the whole
  projection can be grouped by element without any extra bookkeeping.

Grouping by element is the point of the module: for a nitrogen-doped carbon,
"how much of the density at the Fermi level is nitrogen?" is the question,
and :meth:`ProjectedDOS.by_element` answers it directly.

One caveat to state rather than bury: the projections are onto atomic
orbitals, which do not span the full plane-wave basis, so the summed PDOS
falls a few percent short of the total DOS.
:meth:`ProjectedDOS.projection_completeness` reports the shortfall so you can
see whether it is the usual few percent or something worse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

#: projwfc.x filenames look like ``pdos.pdos_atm#12(N)_wfc#2(p)``.
_PDOS_FILENAME = re.compile(
    r"pdos_atm#(?P<atom>\d+)\((?P<element>[A-Za-z]+)\)"
    r"_wfc#(?P<wfc>\d+)\((?P<orbital>[a-z0-9_]+)\)"
)


@dataclass
class DensityOfStates:
    """A total density of states.

    Attributes
    ----------
    energies
        Energy grid in eV, as written by ``dos.x`` (absolute, not referenced
        to the Fermi level).
    dos
        Density of states, states/eV.
    integrated
        Cumulative number of states below each energy, when available.
    fermi_energy
        Fermi level in eV, if it was recorded in the file header.
    """

    energies: np.ndarray
    dos: np.ndarray
    integrated: Optional[np.ndarray] = None
    fermi_energy: Optional[float] = None

    def at_energy(self, energy: float) -> float:
        """Interpolate the DOS at a given absolute energy (eV)."""
        return float(np.interp(energy, self.energies, self.dos))

    def at_fermi(self, fermi: Optional[float] = None) -> float:
        """DOS at the Fermi level, the number that says metal or not.

        A value near zero means a gap or a semimetal; a substantial value
        means states are available for conduction.
        """
        level = fermi if fermi is not None else self.fermi_energy
        if level is None:
            raise ValueError(
                "No se conoce el nivel de Fermi. Pásalo con fermi=... "
                "(lo encuentras en la salida de pw.x)."
            )
        return self.at_energy(level)

    def gap_estimate(
        self,
        fermi: Optional[float] = None,
        threshold: float = 1e-3,
    ) -> Optional[float]:
        """Estimate a gap as the empty window around the Fermi level.

        Walks outward from the Fermi level while the DOS stays below
        ``threshold`` states/eV. Returns ``None`` when the DOS at E_F is
        already above the threshold, i.e. the system conducts.

        This is a smeared, sampled estimate: the broadening used in ``dos.x``
        fills in a small gap, so treat it as indicative and read the real gap
        off the band structure.
        """
        level = fermi if fermi is not None else self.fermi_energy
        if level is None:
            raise ValueError("Se necesita el nivel de Fermi.")
        if self.at_energy(level) > threshold:
            return None

        below = self.energies[
            (self.energies < level) & (self.dos > threshold)
        ]
        above = self.energies[
            (self.energies > level) & (self.dos > threshold)
        ]
        if below.size == 0 or above.size == 0:
            return None
        return float(above.min() - below.max())


@dataclass
class ProjectedDOS:
    """A projected density of states, grouped by atom and orbital.

    Attributes
    ----------
    energies
        Shared energy grid in eV.
    contributions
        Maps ``(element, orbital)`` to its DOS curve.
    atom_contributions
        Maps ``(atom_index, element, orbital)`` to its curve, for when a
        single site matters — the one nitrogen in a supercell, say.
    total
        The total DOS from ``pdos.pdos_tot``, when present.
    fermi_energy
        Fermi level in eV, when known.
    """

    energies: np.ndarray
    contributions: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    atom_contributions: dict[tuple[int, str, str], np.ndarray] = field(
        default_factory=dict
    )
    total: Optional[np.ndarray] = None
    fermi_energy: Optional[float] = None

    @property
    def elements(self) -> list[str]:
        """Elements present in the projection, sorted."""
        return sorted({element for element, _ in self.contributions})

    def by_element(self) -> dict[str, np.ndarray]:
        """Sum the projection over orbitals, giving one curve per element."""
        summed: dict[str, np.ndarray] = {}
        for (element, _), curve in self.contributions.items():
            if element in summed:
                summed[element] = summed[element] + curve
            else:
                summed[element] = curve.copy()
        return summed

    def by_orbital(self, element: str) -> dict[str, np.ndarray]:
        """Return the per-orbital breakdown for one element."""
        return {
            orbital: curve
            for (symbol, orbital), curve in self.contributions.items()
            if symbol == element
        }

    def element_fraction_at(
        self,
        energy: float,
        element: str,
    ) -> float:
        """Fraction of the projected DOS at ``energy`` coming from ``element``.

        This is the number that answers "how much of the density here is
        nitrogen?". It is computed against the **summed projection**, not the
        total DOS, so it is a proper fraction even though the projection is
        incomplete.
        """
        summed = self.by_element()
        if element not in summed:
            raise ValueError(
                f"'{element}' no aparece en la proyección. "
                f"Elementos presentes: {', '.join(self.elements)}."
            )
        values = {
            symbol: float(np.interp(energy, self.energies, curve))
            for symbol, curve in summed.items()
        }
        denominator = sum(values.values())
        if denominator <= 0:
            return 0.0
        return values[element] / denominator

    def projection_completeness(self) -> Optional[float]:
        """Ratio of summed projection to total DOS, integrated over the grid.

        Returns ``None`` when no total was read. Values around 0.9-1.0 are
        normal; much lower suggests the projection is missing orbitals and
        any per-element fraction should be treated with suspicion.
        """
        if self.total is None:
            return None
        summed = sum(self.by_element().values())
        total_area = float(np.trapezoid(self.total, self.energies))
        if abs(total_area) < 1e-12:
            return None
        return float(np.trapezoid(summed, self.energies) / total_area)

    def summary(self, fermi: Optional[float] = None) -> str:
        """Human-readable report, centred on who contributes at E_F."""
        level = fermi if fermi is not None else self.fermi_energy
        lines = [
            f"Proyección sobre {len(self.elements)} elemento(s): "
            f"{', '.join(self.elements)}",
            f"Rejilla: {len(self.energies)} puntos, "
            f"{self.energies.min():.2f} a {self.energies.max():.2f} eV",
        ]

        completeness = self.projection_completeness()
        if completeness is not None:
            lines.append(
                f"Completitud de la proyección: {100 * completeness:.1f} % "
                "del DOS total"
            )
            if completeness < 0.8:
                lines.append(
                    "  ⚠️  Por debajo del 80 %: faltan orbitales en la "
                    "proyección. Las fracciones por elemento pueden engañar."
                )

        if level is not None:
            lines.append(f"\nEn el nivel de Fermi ({level:.3f} eV):")
            for element in self.elements:
                fraction = self.element_fraction_at(level, element)
                lines.append(f"  {element}: {100 * fraction:.1f} %")
            lines.append(
                "\nEsta es la pregunta útil en un material dopado: qué "
                "elemento aporta los estados donde ocurre la conducción."
            )
        else:
            lines.append(
                "\nSin nivel de Fermi no se puede decir quién aporta en E_F. "
                "Búscalo en la salida de pw.x y pásalo."
            )
        return "\n".join(lines)


def _read_columns(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a whitespace table, returning header comments and the data."""
    header: list[str] = []
    rows: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header.append(stripped)
            continue
        try:
            rows.append([float(token) for token in stripped.split()])
        except ValueError:
            # projwfc writes occasional non-numeric trailer lines; skip them.
            continue
    if not rows:
        raise ValueError(f"{path}: no contiene datos numéricos.")
    width = min(len(row) for row in rows)
    return header, np.array([row[:width] for row in rows])


def _fermi_from_header(header: list[str]) -> Optional[float]:
    """Extract ``EFermi = ... eV`` from a dos.x header line, if present."""
    for line in header:
        match = re.search(r"EFermi\s*=\s*(-?\d+\.?\d*)", line)
        if match:
            return float(match.group(1))
    return None


def read_dos(path: str | Path) -> DensityOfStates:
    """Parse a ``dos.x`` output file (``dos.dat``).

    The file has a comment header carrying the Fermi level, then three
    columns: energy (eV), DOS (states/eV) and integrated DOS.
    """
    header, data = _read_columns(Path(path))
    if data.shape[1] < 2:
        raise ValueError(
            f"{path}: se esperaban al menos 2 columnas (energía, DOS) y hay "
            f"{data.shape[1]}."
        )
    return DensityOfStates(
        energies=data[:, 0],
        dos=data[:, 1],
        integrated=data[:, 2] if data.shape[1] > 2 else None,
        fermi_energy=_fermi_from_header(header),
    )


def read_pdos(
    directory: str | Path,
    prefix: str = "pdos",
) -> ProjectedDOS:
    """Parse every ``projwfc.x`` projection file in a directory.

    Parameters
    ----------
    directory
        Where ``projwfc.x`` wrote its output.
    prefix
        The ``filpdos`` value used, so the right files are picked up.

    Returns
    -------
    ProjectedDOS

    Raises
    ------
    ValueError
        When no projection files are found, which usually means
        ``projwfc.x`` did not run or used a different ``filpdos``.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory}: no es un directorio.")

    energies: Optional[np.ndarray] = None
    contributions: dict[tuple[str, str], np.ndarray] = {}
    atom_contributions: dict[tuple[int, str, str], np.ndarray] = {}
    total: Optional[np.ndarray] = None
    fermi: Optional[float] = None

    for path in sorted(directory.iterdir()):
        if not path.is_file() or not path.name.startswith(prefix):
            continue

        if path.name.endswith("pdos_tot"):
            header, data = _read_columns(path)
            if energies is None:
                energies = data[:, 0]
            # pdos_tot columns: E, dos(E), pdos(E)
            total = data[:, 1] if data.shape[1] > 1 else None
            fermi = fermi or _fermi_from_header(header)
            continue

        match = _PDOS_FILENAME.search(path.name)
        if match is None:
            continue
        _, data = _read_columns(path)
        if energies is None:
            energies = data[:, 0]
        # Per-orbital files: E, ldos(E), then one pdos column per m state.
        # The ldos column is the sum over m, which is what we want.
        if data.shape[1] < 2:
            continue
        curve = data[:, 1]

        element = match.group("element")
        orbital = match.group("orbital")
        atom = int(match.group("atom"))

        key = (element, orbital)
        contributions[key] = (
            contributions[key] + curve if key in contributions else curve.copy()
        )
        atom_contributions[(atom, element, orbital)] = curve

    if energies is None or not contributions:
        raise ValueError(
            f"{directory}: no se encontraron archivos de proyección con "
            f"prefijo '{prefix}'. ¿Se ejecutó projwfc.x?"
        )

    return ProjectedDOS(
        energies=energies,
        contributions=contributions,
        atom_contributions=atom_contributions,
        total=total,
        fermi_energy=fermi,
    )


def draw_dos_on_axes(
    dos: DensityOfStates | ProjectedDOS,
    ax,
    reference: Optional[float] = None,
    energy_window: Optional[tuple[float, float]] = None,
    by_element: bool = True,
    title: str = "Densidad de estados",
) -> None:
    """Draw a DOS or PDOS onto an existing matplotlib axes.

    For a :class:`ProjectedDOS` the per-element curves are stacked over the
    total, which is the layout that makes "which element sits at E_F"
    readable at a glance.
    """
    zero = reference
    if zero is None:
        zero = dos.fermi_energy

    energies = dos.energies - zero if zero is not None else dos.energies

    if isinstance(dos, ProjectedDOS):
        if dos.total is not None:
            ax.plot(energies, dos.total, color="#333333", linewidth=1.2,
                    label="Total")
        palette = ["#1f4e9c", "#c0392b", "#27ae60", "#8e44ad", "#d68910",
                   "#16a085"]
        curves = dos.by_element() if by_element else dos.contributions
        for index, (label, curve) in enumerate(sorted(curves.items(),
                                                      key=lambda kv: str(kv[0]))):
            name = label if isinstance(label, str) else "-".join(label)
            ax.fill_between(energies, curve, alpha=0.35,
                            color=palette[index % len(palette)])
            ax.plot(energies, curve, linewidth=1.0,
                    color=palette[index % len(palette)], label=name)
        ax.legend(fontsize=8)
    else:
        ax.plot(energies, dos.dos, color="#1f4e9c", linewidth=1.2)

    if zero is not None:
        ax.axvline(0.0, color="#c0392b", linestyle="--", linewidth=0.8)
        ax.set_xlabel("E − E$_F$ (eV)")
    else:
        ax.set_xlabel("Energía (eV)")
    ax.set_ylabel("DOS (estados/eV)")
    ax.set_ylim(bottom=0)
    if energy_window is not None:
        ax.set_xlim(*energy_window)
    ax.set_title(title)


def plot_dos(
    dos: DensityOfStates | ProjectedDOS,
    reference: Optional[float] = None,
    energy_window: Optional[tuple[float, float]] = None,
    by_element: bool = True,
    title: str = "Densidad de estados",
):
    """Return a standalone ``Figure`` with the density of states."""
    import matplotlib.pyplot as plt  # noqa: WPS433

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    draw_dos_on_axes(
        dos, ax, reference=reference, energy_window=energy_window,
        by_element=by_element, title=title,
    )
    fig.tight_layout()
    return fig
