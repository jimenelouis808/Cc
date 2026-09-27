"""DFTB Slater-Koster files (``.skf``) as tbkit models.

A ``.skf`` file tabulates, on a radial grid, the two-centre Hamiltonian and
overlap integrals of one element pair (``A-B.skf``), plus -- for a
homonuclear file -- the on-site energies, Hubbard U and occupations of the
free atom. The "simple" format (Aradi et al., DFTB+ documentation,
"Slater-Koster files"):

* line 1: ``gridDist nGridPoints``
* line 2 (homonuclear only): ``Ed Ep Es SPE Ud Up Us fd fp fs``
* line 3: mass and repulsive polynomial (ignored here)
* ``nGridPoints`` lines of 20 columns: ``Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1
  Hsd0 Hsp0 Hss0`` then the same ten for S.
* a ``Spline`` block with the repulsive pair potential, read into
  :class:`tbkit.repulsive.SkfSpline` (total energies and forces).

Row i (1-based) is at ``r = i · gridDist``; units Hartree and Bohr,
converted here to eV and Å. ``n*value`` (Fortran repetition) is accepted.
The extended format (``@`` on line 1, f orbitals) is refused.

Convention for the heteronuclear ``sp`` integral: in ``A-B.skf``, ``Hsp0``
is taken as <s on A | H | p on B> (tbkit's ``(A, B, "sps")``). The d
columns are read and ignored (s/p basis only).

Licences: the published sets (mio, 3ob, pbc...) belong to their authors
and are distributed by dftb.org under their own terms. tbkit reads them
from your disk and never ships them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ase.units import Bohr, Hartree

from .params import Table, TBModel

_COLUMNS = ("Hdd0", "Hdd1", "Hdd2", "Hpd0", "Hpd1", "Hpp0", "Hpp1", "Hsd0", "Hsp0", "Hss0",
            "Sdd0", "Sdd1", "Sdd2", "Spd0", "Spd1", "Spp0", "Spp1", "Ssd0", "Ssp0", "Sss0")


@dataclass
class SKFile:
    """The contents of one ``.skf`` that tbkit uses."""

    elements: tuple[str, str]
    r: np.ndarray                           # Å
    table: dict[str, np.ndarray]            # column -> eV (H) or dimensionless (S)
    onsite: Optional[dict[str, float]]      # "s", "p", "d" -> eV (homonuclear)
    hubbard: Optional[dict[str, float]]     # eV
    occupations: Optional[dict[str, float]]
    repulsive: Optional[object] = None         # SkfSpline (eV, Å) or None


def _numbers(line: str) -> list[float]:
    out: list[float] = []
    for token in re.split(r"[,\s]+", line.strip()):
        if not token:
            continue
        if "*" in token:
            count, value = token.split("*")
            out += [float(value)] * int(count)
        else:
            out.append(float(token.replace("D", "E").replace("d", "e")))
    return out


def read_skf(path: str | Path) -> SKFile:
    """Read one simple-format ``.skf`` file named ``A-B.skf``."""
    path = Path(path)
    match = re.match(r"^([A-Z][a-z]?)-([A-Z][a-z]?)", path.stem)
    if match is None:
        raise ValueError(f"{path.name}: el nombre debe ser A-B.skf (p. ej. C-H.skf).")
    a, b = match.group(1), match.group(2)
    lines = [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()
             if ln.strip()]
    if lines[0].lstrip().startswith("@"):
        raise ValueError(f"{path.name}: formato extendido (orbitales f), no soportado.")
    grid_dist, n_points = _numbers(lines[0])[:2]
    n_points = int(n_points)
    row = 1
    onsite = hubbard = occupations = None
    if a == b:
        ed, ep, es, _spe, ud, up, us, fd, fp, fs = _numbers(lines[1])[:10]
        onsite = {"s": es * Hartree, "p": ep * Hartree, "d": ed * Hartree}
        hubbard = {"s": us * Hartree, "p": up * Hartree, "d": ud * Hartree}
        occupations = {"s": fs, "p": fp, "d": fd}
        row = 2
    row += 1                                   # mass / polynomial line
    data = []
    for line in lines[row:]:
        if line.strip().lower().startswith("spline"):
            break
        values = _numbers(line)
        if len(values) >= 20:
            data.append(values[:20])
        if len(data) == n_points:
            break
    data = np.array(data)
    if len(data) < 2:
        raise ValueError(f"{path.name}: tabla de integrales vacía o ilegible.")
    r = np.arange(1, len(data) + 1) * grid_dist * Bohr
    table = {}
    for index, name in enumerate(_COLUMNS):
        scale = Hartree if name.startswith("H") else 1.0
        table[name] = data[:, index] * scale
    from .repulsive import read_skf_spline

    return SKFile((a, b), r, table, onsite, hubbard, occupations, read_skf_spline(lines))


def _law(r: np.ndarray, values: np.ndarray) -> Optional[Table]:
    if not np.any(values):
        return None
    last = int(np.flatnonzero(values)[-1])
    stop = min(len(r), last + 2)
    return Table(tuple(r[:stop]), tuple(values[:stop]), float(r[stop - 1]))


def load_skf_set(directory: str | Path, orbitals: dict[str, tuple[str, ...]],
                 name: Optional[str] = None) -> TBModel:
    """A non-orthogonal model from the ``A-B.skf`` files of a directory.

    ``orbitals`` chooses the basis per element, e.g.
    ``{"C": ("s", "px", "py", "pz"), "H": ("s",)}``; every pair needs its file.
    On-site energies, Hubbard U and valence come from the homonuclear files.
    """
    directory = Path(directory)
    elements = list(orbitals)
    onsite, hubbard, valence = {}, {}, {}
    hopping, overlap = {}, {}
    splines: dict = {}
    for a in elements:
        for b in elements:
            path = directory / f"{a}-{b}.skf"
            if not path.exists():
                raise FileNotFoundError(f"Falta {path.name} en {directory}.")
            sk = read_skf(path)
            if sk.repulsive is not None and (b, a) not in splines:
                splines[(a, b)] = sk.repulsive
            if a == b:
                onsite[a] = {"s": sk.onsite["s"], "p": sk.onsite["p"]}
                has_p = any(o != "s" for o in orbitals[a])
                hubbard[a] = sk.hubbard["p" if has_p else "s"]
                valence[a] = sk.occupations["s"] + (sk.occupations["p"] if has_p else 0.0)
            for bond, h_col, s_col in (("sss", "Hss0", "Sss0"), ("sps", "Hsp0", "Ssp0"),
                                       ("pps", "Hpp0", "Spp0"), ("ppp", "Hpp1", "Spp1")):
                if bond != "sps" and (b, a, bond) in hopping:
                    continue
                for store, column in ((hopping, h_col), (overlap, s_col)):
                    law = _law(sk.r, sk.table[column])
                    if law is not None:
                        store[(a, b, bond)] = law
    from .repulsive import PairRepulsive

    repulsive = PairRepulsive(splines) if len(splines) == len(elements) * (len(elements) + 1) \
        // 2 else None
    return TBModel(name=name or f"integrales .skf ({directory.name})", orbitals=dict(orbitals),
                   onsite=onsite, hopping=hopping, overlap=overlap, valence=valence,
                   hubbard_u=hubbard, repulsive=repulsive,
                   metadata={"reference": f"archivos .skf de {directory}",
                             "notes": "Integrales de dos centros y repulsión de DFTB; con "
                                      "tbkit.scc es un modelo tipo DFTB2 (sin tercer orden)."})
