"""Parsing and plotting electronic band structures.

Supports the two formats that matter in practice:

* **Quantum ESPRESSO** ``bands.x`` writes ``filband`` (here ``bands.dat``),
  which carries explicit k-vectors, plus a ``bands.dat.gnu`` companion whose
  x-axis is already the cumulative path distance. We read both: the raw file
  when you need the k-vectors, the ``.gnu`` one when you just want to plot.
* **SIESTA** writes ``SystemLabel.bands``, which additionally contains the
  Fermi energy and the special-point positions along the path.

Energies are reported in eV throughout. QE's ``bands.dat`` is already in eV;
SIESTA's ``.bands`` file is in eV as well but states its Fermi level
separately, which we subtract only when explicitly asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


@dataclass
class BandStructure:
    """Eigenvalues along a k-path.

    Attributes
    ----------
    distances
        Cumulative distance along the path, shape ``(nk,)``. Used as the
        x-axis; the units are whatever the source file used (QE and SIESTA
        both use inverse-Bohr-ish path units, and only relative values
        matter for a plot).
    energies
        Eigenvalues in eV, shape ``(nk, nbnd)``.
    kpoints
        Optional k-vectors, shape ``(nk, 3)``. Present when parsed from a
        format that records them.
    fermi_energy
        Fermi level in eV, when the source file reports one.
    special_points
        ``(distance, label)`` pairs marking high-symmetry points, when known.
    """

    distances: np.ndarray
    energies: np.ndarray
    kpoints: Optional[np.ndarray] = None
    fermi_energy: Optional[float] = None
    special_points: list[tuple[float, str]] = field(default_factory=list)

    @property
    def n_bands(self) -> int:
        return int(self.energies.shape[1])

    @property
    def n_kpoints(self) -> int:
        return int(self.energies.shape[0])

    def shifted(self, reference: Optional[float] = None) -> "BandStructure":
        """Return a copy with energies measured from ``reference``.

        Defaults to :attr:`fermi_energy` when available. Raises if neither is
        known, rather than silently plotting an unshifted axis labelled as if
        it were referenced to the Fermi level.
        """
        zero = reference if reference is not None else self.fermi_energy
        if zero is None:
            raise ValueError(
                "No hay energía de referencia: el archivo no trae nivel de "
                "Fermi. Pásalo explícitamente con reference=..."
            )
        return BandStructure(
            distances=self.distances,
            energies=self.energies - zero,
            kpoints=self.kpoints,
            fermi_energy=0.0,
            special_points=list(self.special_points),
        )

    def band_gap(self, fermi: Optional[float] = None) -> Optional[float]:
        """Estimate the band gap in eV from the sampled eigenvalues.

        Returns ``None`` when the bands cross the reference energy, i.e. the
        system is metallic on this sampling.

        This is a *sampled* gap: it only sees the k-points on the path, so a
        band extremum lying off the path is invisible. It is a sanity check,
        not a substitute for a proper dense-mesh calculation.
        """
        zero = fermi if fermi is not None else self.fermi_energy
        if zero is None:
            raise ValueError("Se necesita el nivel de Fermi para el gap.")
        below = self.energies[self.energies <= zero]
        above = self.energies[self.energies > zero]
        if below.size == 0 or above.size == 0:
            return None
        valence_max = float(below.max())
        conduction_min = float(above.min())
        # A band crossing the reference means no gap.
        crosses = np.any(
            (self.energies.min(axis=0) < zero) & (self.energies.max(axis=0) > zero)
        )
        if crosses:
            return None
        return conduction_min - valence_max


def read_qe_bands(path: str | Path) -> BandStructure:
    """Parse Quantum ESPRESSO's ``filband`` output (e.g. ``bands.dat``).

    The format is a ``&plot nbnd=.., nks=.. /`` header, then for each
    k-point a line with its three crystal coordinates followed by the
    eigenvalues wrapped across lines.

    Parameters
    ----------
    path
        Path to the ``filband`` file.

    Returns
    -------
    BandStructure
        With :attr:`kpoints` populated and :attr:`distances` computed as the
        cumulative Euclidean length along the k-path.
    """
    text = Path(path).read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: archivo vacío.")

    header = lines[0]
    if "nbnd" not in header:
        raise ValueError(
            f"{path}: no parece un archivo filband de bands.x "
            "(falta la cabecera '&plot nbnd=...')."
        )
    # Header looks like: " &plot nbnd=   8, nks=  91 /"
    try:
        nbnd = int(header.split("nbnd=")[1].split(",")[0])
        nks = int(header.split("nks=")[1].split("/")[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"{path}: cabecera ilegible: {header!r}") from exc

    values: list[float] = []
    for line in lines[1:]:
        values.extend(float(tok) for tok in line.split())

    per_kpoint = 3 + nbnd
    expected = nks * per_kpoint
    if len(values) != expected:
        raise ValueError(
            f"{path}: se esperaban {expected} números "
            f"({nks} puntos k x (3 + {nbnd} bandas)) y hay {len(values)}."
        )

    block = np.array(values).reshape(nks, per_kpoint)
    kpoints = block[:, :3]
    energies = block[:, 3:]

    steps = np.linalg.norm(np.diff(kpoints, axis=0), axis=1)
    distances = np.concatenate([[0.0], np.cumsum(steps)])

    return BandStructure(
        distances=distances, energies=energies, kpoints=kpoints
    )


def read_qe_bands_gnu(path: str | Path) -> BandStructure:
    """Parse the ``bands.dat.gnu`` companion file.

    That file is a sequence of blank-line-separated blocks, one per band,
    each holding ``distance energy`` pairs. It has no k-vectors and no Fermi
    level, but its x-axis is already the path distance.
    """
    text = Path(path).read_text()
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = []
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        current.append((float(parts[0]), float(parts[1])))
    if current:
        blocks.append(current)

    if not blocks:
        raise ValueError(f"{path}: no se encontró ningún bloque de banda.")

    lengths = {len(b) for b in blocks}
    if len(lengths) != 1:
        raise ValueError(
            f"{path}: los bloques de banda tienen longitudes distintas "
            f"({sorted(lengths)}); el archivo parece truncado."
        )

    distances = np.array([point[0] for point in blocks[0]])
    energies = np.column_stack([[point[1] for point in b] for b in blocks])
    return BandStructure(distances=distances, energies=energies)


def read_siesta_bands(path: str | Path) -> BandStructure:
    """Parse a SIESTA ``SystemLabel.bands`` file.

    Layout::

        E_fermi
        k_min k_max
        E_min E_max
        nbands nspin nkpoints
        k1  e1 e2 e3 ...
        k2  ...
        n_special
        pos1 label1
        ...

    Returns
    -------
    BandStructure
        With :attr:`fermi_energy` and :attr:`special_points` filled in.
    """
    tokens = Path(path).read_text().split()
    if len(tokens) < 8:
        raise ValueError(f"{path}: archivo demasiado corto para ser un .bands.")

    cursor = 0

    def take(n: int) -> list[str]:
        nonlocal cursor
        chunk = tokens[cursor:cursor + n]
        cursor += n
        return chunk

    fermi = float(take(1)[0])
    take(2)  # k range, not needed
    take(2)  # energy range, not needed
    nbands, nspin, nkpoints = (int(v) for v in take(3))

    if nspin != 1:
        raise ValueError(
            f"{path}: nspin={nspin}. Solo se admite el caso no polarizado "
            "por ahora."
        )

    distances = np.zeros(nkpoints)
    energies = np.zeros((nkpoints, nbands))
    for index in range(nkpoints):
        row = take(1 + nbands)
        distances[index] = float(row[0])
        energies[index] = [float(v) for v in row[1:]]

    special: list[tuple[float, str]] = []
    if cursor < len(tokens):
        n_special = int(take(1)[0])
        for _ in range(n_special):
            pair = take(2)
            if len(pair) == 2:
                special.append((float(pair[0]), pair[1].strip("'\"")))

    return BandStructure(
        distances=distances,
        energies=energies,
        fermi_energy=fermi,
        special_points=special,
    )


def attach_path_labels(
    bands: BandStructure,
    labels: Sequence[str],
) -> BandStructure:
    """Distribute high-symmetry ``labels`` evenly over the path.

    Useful when plotting a QE band structure, whose files carry no labels.
    Assumes the segments were sampled with equal point counts, which is what
    :func:`carbonforge.calculations.kpaths.format_qe_kpath` produces. If you
    hand-edited the k-path with uneven segments, the tick positions will be
    wrong — pass ``special_points`` yourself instead.
    """
    if len(labels) < 2:
        raise ValueError("Se necesitan al menos dos etiquetas.")
    positions = np.linspace(
        bands.distances[0], bands.distances[-1], len(labels)
    )
    bands.special_points = list(zip(positions.tolist(), labels))
    return bands


def draw_bands_on_axes(
    bands: BandStructure,
    ax,
    reference: Optional[float] = None,
    energy_window: Optional[tuple[float, float]] = None,
    title: str = "Estructura de bandas",
) -> None:
    """Draw a band diagram onto an existing matplotlib axes.

    Shared by :func:`plot_bands` and by the GUI, which embeds its own canvas.
    Keeping the drawing separate from figure creation matters here: pyplot
    inside a running Tk application creates figures it then manages itself,
    which leak and can spawn stray windows.

    Parameters
    ----------
    bands
        Parsed band structure.
    ax
        Target axes.
    reference
        Energy to place at zero. Defaults to the Fermi level when the file
        provided one; otherwise the raw eigenvalues are plotted and the axis
        is labelled accordingly, rather than mislabelling them as E - E_F.
    energy_window
        ``(low, high)`` limits in eV relative to the reference.
    title
        Axes title.
    """
    zero = reference if reference is not None else bands.fermi_energy
    energies = bands.energies - zero if zero is not None else bands.energies

    for index in range(bands.n_bands):
        ax.plot(bands.distances, energies[:, index], color="#1f4e9c", linewidth=1.0)

    if zero is not None:
        ax.axhline(0.0, color="#c0392b", linestyle="--", linewidth=0.8)
        ax.set_ylabel("E − E$_F$ (eV)")
    else:
        ax.set_ylabel("Energía (eV)")

    if bands.special_points:
        positions = [p for p, _ in bands.special_points]
        names = [n for _, n in bands.special_points]
        # Γ is conventionally written with the Greek letter in figures even
        # though the input files must spell it "G".
        names = ["Γ" if n.upper() in ("G", "GAMMA") else n for n in names]
        ax.set_xticks(positions)
        ax.set_xticklabels(names)
        for position in positions:
            ax.axvline(position, color="#999999", linewidth=0.5)
    else:
        ax.set_xlabel("Camino k")

    ax.set_xlim(bands.distances[0], bands.distances[-1])
    if energy_window is not None:
        ax.set_ylim(*energy_window)
    ax.set_title(title)


def plot_bands(
    bands: BandStructure,
    reference: Optional[float] = None,
    energy_window: Optional[tuple[float, float]] = None,
    title: str = "Estructura de bandas",
):
    """Return a standalone matplotlib ``Figure`` with the band diagram.

    Convenience wrapper around :func:`draw_bands_on_axes` for scripting. In a
    GUI, call that function directly against an embedded axes instead.
    """
    import matplotlib.pyplot as plt  # noqa: WPS433

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    draw_bands_on_axes(bands, ax, reference=reference,
                       energy_window=energy_window, title=title)
    fig.tight_layout()
    return fig
