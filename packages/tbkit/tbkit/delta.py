"""Δ-learning: a linear SOAP correction on top of a tight-binding set.

    E = E_TB + Σ_i (w_{Z_i} · x_i + b_{Z_i}),      F = F_TB − Σ_i w_{Z_i} · ∂x_i/∂R

with x_i the SOAP vector of atom i (dscribe; analytical derivatives for
finite systems, numerical for periodic ones) and one
weight vector and bias per element. The correction is fitted by ridge least
squares to what the TB set misses against DFT (energies per structure and
forces), in the spirit of Stöhr, Medrano Sandonas and Tkatchenko, J. Phys.
Chem. Lett. 11, 6835 (2020), who learn the DFTB repulsion; here the model is
linear, so every weight is auditable and the fit is one linear solve.

What it is for: geometries and phonons closer to DFT where the TB repulsion
errs, while α, μ and bands stay those of the TB set (the correction changes
positions and modes, not the electronic structure). What it is not: a fix
outside the training data. ``novelty`` measures how far each atom's environment
is from the closest training environment of its element (1 − cosine
similarity of SOAP vectors); far atoms are reported, never silently trusted.

Optional dependency: ``pip install dscribe``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


def _soap(species, r_cut, n_max, l_max, periodic):
    try:
        from dscribe.descriptors import SOAP
    except ImportError as error:                          # pragma: no cover - optional
        raise ImportError("Δ-learning necesita dscribe: pip install dscribe") from error
    return SOAP(species=list(species), r_cut=r_cut, n_max=n_max, l_max=l_max,
                periodic=periodic, sparse=False)


@dataclass
class DeltaModel:
    """Linear SOAP correction; ``weights[Z]`` (n_features,), ``bias[Z]`` (eV)."""

    species: tuple
    r_cut: float = 4.0
    n_max: int = 4
    l_max: int = 3
    weights: dict = field(default_factory=dict)
    bias: dict = field(default_factory=dict)
    scale: Optional[np.ndarray] = None             # feature scaling used in the fit
    reference: dict = field(default_factory=dict)  # element -> (n_ref, n_features), unit norm
    info: dict = field(default_factory=dict)

    def descriptor(self, atoms: Atoms, derivatives: bool = False):
        soap = _soap(self.species, self.r_cut, self.n_max, self.l_max, bool(atoms.pbc.any()))
        if derivatives:
            d, x = soap.derivatives(atoms, method="auto", attach=True)   # numerical if periodic
            return x, d
        return soap.create(atoms), None

    def n_features(self) -> int:
        return _soap(self.species, self.r_cut, self.n_max, self.l_max, False) \
            .get_number_of_features()

    def energy_forces(self, atoms: Atoms, cached=None) -> tuple[float, np.ndarray]:
        x, d = cached if cached is not None else self.descriptor(atoms, derivatives=True)
        symbols = atoms.get_chemical_symbols()
        energy = 0.0
        forces = np.zeros((len(atoms), 3))
        for i, z in enumerate(symbols):
            w = self.weights[z] / self.scale
            energy += float(x[i] @ w) + self.bias[z]
            forces -= d[i] @ w
        return energy, forces

    def novelty(self, atoms: Atoms) -> np.ndarray:
        """Per atom: 1 − max cosine similarity to its element's training environments."""
        x, _ = self.descriptor(atoms)
        out = np.zeros(len(atoms))
        for i, z in enumerate(atoms.get_chemical_symbols()):
            v = x[i] / (np.linalg.norm(x[i]) or 1.0)
            out[i] = 1.0 - float(np.max(self.reference[z] @ v))
        return out

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {"species": list(self.species), "r_cut": self.r_cut, "n_max": self.n_max,
                "l_max": self.l_max, "units": {"energy": "eV", "length": "Å"},
                "weights": {z: w.tolist() for z, w in self.weights.items()},
                "bias": self.bias, "scale": self.scale.tolist(),
                "reference": {z: np.round(r, 6).tolist() for z, r in self.reference.items()},
                "info": self.info}

    @classmethod
    def from_dict(cls, data: dict) -> "DeltaModel":
        return cls(tuple(data["species"]), data["r_cut"], data["n_max"], data["l_max"],
                   {z: np.array(w) for z, w in data["weights"].items()}, dict(data["bias"]),
                   np.array(data["scale"]),
                   {z: np.array(r) for z, r in data["reference"].items()}, data.get("info", {}))

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "DeltaModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def fit_delta(rows: list[dict], species, ridge: float = 1e-3, energy_weight: float = 10.0,
              r_cut: float = 4.0, n_max: int = 4, l_max: int = 3,
              max_reference: int = 400) -> DeltaModel:
    """Fit the correction to ``rows``: dicts with ``atoms``, ``dE`` (DFT − TB energy, eV)
    and ``dF`` (DFT − TB forces, eV/Å). Energies enter per atom × ``energy_weight``.

    Features are scaled to unit RMS over the training atoms, then ridge ``ridge``
    (relative to the mean diagonal of the normal matrix) on the weights only."""
    model = DeltaModel(tuple(species), r_cut, n_max, l_max)
    nf = model.n_features()
    elements = list(species)
    columns = {z: slice(k * nf, (k + 1) * nf) for k, z in enumerate(elements)}
    n_w = nf * len(elements)
    blocks_a, blocks_b, xs = [], [], {z: [] for z in elements}
    for row in rows:
        atoms = row["atoms"]
        if "soap" not in row:                       # (x, ∂x/∂R), reused across fits
            row["soap"] = model.descriptor(atoms, derivatives=True)
        x, d = row["soap"]
        symbols = atoms.get_chemical_symbols()
        n = len(atoms)
        e_row = np.zeros(n_w + len(elements))
        f_rows = np.zeros((3 * n, n_w + len(elements)))
        for i, z in enumerate(symbols):
            e_row[columns[z]] += x[i]
            e_row[n_w + elements.index(z)] += 1.0
            f_rows[:, columns[z]] -= d[i].reshape(3 * n, nf)
            xs[z].append(x[i])
        blocks_a.append(e_row[None] * energy_weight / n)
        blocks_b.append(np.array([row["dE"] * energy_weight / n]))
        blocks_a.append(f_rows)
        blocks_b.append(np.asarray(row["dF"]).ravel())
    a = np.vstack(blocks_a)
    b = np.concatenate(blocks_b)
    rms = np.sqrt(np.mean(a[:, :n_w] ** 2, axis=0))
    rms[rms == 0] = 1.0
    scale = rms
    a[:, :n_w] /= scale
    normal = a.T @ a
    reg = ridge * np.mean(np.diag(normal)[:n_w])
    normal[np.arange(n_w), np.arange(n_w)] += reg
    solution = np.linalg.solve(normal, a.T @ b)
    model.scale = np.ones(nf)                    # per-element scales folded into the weights
    for z in elements:
        model.weights[z] = solution[columns[z]] / scale[columns[z]]
        model.bias[z] = float(solution[n_w + elements.index(z)])
    for z in elements:
        ref = np.array(xs[z])
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        if len(ref) > max_reference:
            ref = ref[np.linspace(0, len(ref) - 1, max_reference).round().astype(int)]
        model.reference[z] = ref
    model.info = {"ridge": ridge, "energy_weight": energy_weight, "structures": len(rows),
                  "force_components": int(sum(np.asarray(r["dF"]).size for r in rows))}
    return model


class DeltaCalculator(Calculator):
    """TB calculator plus the Δ correction (energy, forces)."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, base: Calculator, delta: DeltaModel, **kwargs):
        super().__init__(**kwargs)
        self.base = base
        self.delta = delta

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        probe = self.atoms.copy()
        probe.calc = self.base
        e_tb = probe.get_potential_energy()
        f_tb = probe.get_forces()
        e_d, f_d = self.delta.energy_forces(self.atoms)
        self.results = {"energy": e_tb + e_d, "free_energy": e_tb + e_d, "forces": f_tb + f_d}
