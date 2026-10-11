"""Corrections to the model's charges for IR: Born charges closer to DFT.

The IR of :mod:`tbkit.infrared` comes from μ = Σ Q_A R_A − Σ p_A with Mulliken
charges of a minimal basis. Its main error is the charge flux, how much charge
moves between atoms when a bond stretches (``validation/ir_tb_vs_gpaw.json``:
methylamine's C-H stretches 1-10 against GPAW's 25-82 km/mol). Here a correction
Δq_A(R), neutral by construction, is added to those charges; its dipole
Δμ = Σ Δq_A R_A gives Born-charge corrections

    ΔZ*_{a,ij} = ∂Δμ_j/∂x_{a,i} = Δq_a δ_ij + Σ_k (∂Δq_k/∂x_{a,i}) R_{k,j}

that are added to the model's own. Both models below are linear in their
parameters, so ΔZ* = Σ_p θ_p A_p(R) and the fit to DFT Born charges is one ridge
solve (:func:`fit`), cross-validated by leaving whole molecules out.

* :class:`BondFlux` — a class-IV charge model in the spirit of Cramer and
  Truhlar's CM-n charges: charge moves along each bond between unlike elements,
  g_ab(r) = [D_ab + F_ab (r − r0_ab)] w(r), +g on the alphabetically first element,
  −g on the other. D shifts static dipoles, F is the charge flux. Few parameters
  (two per element pair), one meaning each. It cannot act on bonds between like
  atoms (antisymmetry), so it leaves pure carbon untouched.
* :class:`EnvironmentFlux` — machine-learned (linear SOAP, dscribe): Δq_A = w_Z · x_A
  minus the mean over the structure (neutrality), so carbon atoms in pentagons,
  heptagons or next to a dopant can differ. Many parameters: judged only by the
  cross-validated error, like every learned model in tbkit (``delta.py``).

Both work for molecules and for wires and slabs (only bond vectors and positions
along the open axes enter). A model is used only if it lowers the error on
molecules it was not fitted to (``recipes/ir_charge_fit.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii

#: Bond switch: full weight to 1.15 × (sum of covalent radii), zero from 1.35 ×.
R_ON, R_OFF = 1.15, 1.35


def _switch(r, r_on, r_off):
    """Smooth (C¹) step 1 → 0 between r_on and r_off, and its derivative."""
    x = np.clip((r - r_on) / (r_off - r_on), 0.0, 1.0)
    w = 1 - x * x * (3 - 2 * x)
    dw = np.where((r > r_on) & (r < r_off), -6 * x * (1 - x) / (r_off - r_on), 0.0)
    return w, dw


def _pairs(atoms: Atoms):
    """Bonded pairs (i, j, vector r_j − r_i with the minimum image, distance)."""
    from ase.neighborlist import neighbor_list

    numbers = atoms.get_atomic_numbers()
    cut = R_OFF * covalent_radii[numbers]
    i, j, d, vec = neighbor_list("ijdD", atoms, cut, self_interaction=False)
    keep = i < j
    return i[keep], j[keep], d[keep], vec[keep]


def _open_positions(atoms: Atoms) -> np.ndarray:
    """Positions relative to the centroid; zero along periodic axes (no dipole there)."""
    pos = atoms.get_positions() - atoms.get_positions().mean(axis=0)
    return pos * (~np.asarray(atoms.get_pbc(), bool))[None, :]


@dataclass
class BondFlux:
    """Charge flux along bonds between unlike elements (two parameters per pair)."""

    pairs: list[tuple[str, str]]
    theta: np.ndarray | None = None           # [D_ab, F_ab for each pair] in e, e/Å
    info: dict = field(default_factory=dict)
    kind: str = "bond_flux"

    @classmethod
    def for_elements(cls, elements) -> BondFlux:
        el = sorted(set(elements))
        return cls([(a, b) for k, a in enumerate(el) for b in el[k + 1:]])

    def n_params(self) -> int:
        return 2 * len(self.pairs)

    def design(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        """(A, B): ΔZ* = A @ θ with A (P, N, 3, 3) and Δq = B @ θ with B (P, N)."""
        n = len(atoms)
        symbols = atoms.get_chemical_symbols()
        numbers = atoms.get_atomic_numbers()
        index = {p: k for k, p in enumerate(self.pairs)}
        A = np.zeros((self.n_params(), n, 3, 3))
        B = np.zeros((self.n_params(), n))
        pos = _open_positions(atoms)
        open_axes = (~np.asarray(atoms.get_pbc(), bool)).astype(float)
        for a, b, r, vec in zip(*_pairs(atoms), strict=True):
            sa, sb = symbols[a], symbols[b]
            if sa == sb:
                continue
            first, second = (a, b) if sa < sb else (b, a)
            sign_vec = vec if first == a else -vec          # r_second − r_first
            k = index.get(tuple(sorted((sa, sb))))
            if k is None:
                continue
            rsum = covalent_radii[numbers[a]] + covalent_radii[numbers[b]]
            w, dw = _switch(r, R_ON * rsum, R_OFF * rsum)
            r0 = rsum
            unit = sign_vec / r                             # from first to second
            # g = D w + F (r - r0) w ; +g on first, −g on second
            for p, (g, dg) in ((2 * k, (w, dw)),
                               (2 * k + 1, ((r - r0) * w, w + (r - r0) * dw))):
                B[p, first] += g
                B[p, second] -= g
                # ∂r/∂x_second = unit, ∂r/∂x_first = −unit; Δμ_j term Σ_k ∂Δq_k R_kj
                # = dg ∂r/∂x_ai (R_first,j − R_second,j) = −dg ∂r/∂x_ai r unit_j (open axes)
                lever = -r * unit * open_axes
                A[p, second] += dg * np.outer(unit, lever)
                A[p, first] += dg * np.outer(-unit, lever)
        A += B[:, :, None, None] * np.eye(3)[None, None] * open_axes[None, None, None, :]
        del pos
        return A, B

    def to_dict(self) -> dict:
        return {"kind": self.kind, "pairs": [list(p) for p in self.pairs],
                "theta": None if self.theta is None else [float(x) for x in self.theta],
                "theta_units": "per pair: D (e), F (e/Å)", "switch": [R_ON, R_OFF],
                "info": self.info}


@dataclass
class EnvironmentFlux:
    """Linear SOAP charge correction, one weight vector per element (needs dscribe)."""

    species: tuple
    r_cut: float = 4.0
    n_max: int = 4
    l_max: int = 3
    theta: np.ndarray | None = None
    scale: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    kind: str = "environment_flux"

    def _soap(self, periodic: bool):
        from dscribe.descriptors import SOAP

        return SOAP(species=list(self.species), r_cut=self.r_cut, n_max=self.n_max,
                    l_max=self.l_max, periodic=periodic, sparse=False)

    def n_features(self) -> int:
        return self._soap(False).get_number_of_features()

    def n_params(self) -> int:
        return len(self.species) * self.n_features()

    def design(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        soap = self._soap(bool(np.any(atoms.get_pbc())))
        d, x = soap.derivatives(atoms, method="auto", attach=True)   # (N, N, 3, F), (N, F)
        n, nf = x.shape
        symbols = atoms.get_chemical_symbols()
        pos = _open_positions(atoms)
        open_axes = (~np.asarray(atoms.get_pbc(), bool)).astype(float)
        A = np.zeros((self.n_params(), n, 3, 3))
        B = np.zeros((self.n_params(), n))
        for s, element in enumerate(self.species):
            rows = [k for k in range(n) if symbols[k] == element]
            if not rows:
                continue
            block = slice(s * nf, (s + 1) * nf)
            # Δq_k = w·x_k − mean over all atoms (neutral)
            xs = np.zeros((n, nf))
            xs[rows] = x[rows]
            B[block] = (xs - xs.mean(axis=0)).T
            ds = np.zeros((n, n, 3, nf))                # [center k, moved atom a, i, f]
            ds[rows] = d[rows]
            dq = ds - ds.mean(axis=0, keepdims=True)    # neutral derivative
            # Σ_k ∂Δq_k/∂x_ai R_kj  -> (F, a, i, j)
            A[block] += np.einsum("kaif,kj->faij", dq, pos)
        A += B[:, :, None, None] * np.eye(3)[None, None] * open_axes[None, None, None, :]
        return A, B

    def to_dict(self) -> dict:
        return {"kind": self.kind, "species": list(self.species), "r_cut": self.r_cut,
                "n_max": self.n_max, "l_max": self.l_max,
                "theta": None if self.theta is None else [float(x) for x in self.theta],
                "info": self.info}


@dataclass
class Combined:
    """Several models fitted together (their designs side by side)."""

    parts: list
    theta: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    kind: str = "combined"

    def n_params(self) -> int:
        return sum(m.n_params() for m in self.parts)

    def design(self, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
        designs = [m.design(atoms) for m in self.parts]
        return (np.concatenate([d[0] for d in designs]), np.concatenate([d[1] for d in designs]))

    def to_dict(self) -> dict:
        out = {"kind": self.kind, "parts": [m.to_dict() for m in self.parts],
               "theta": None if self.theta is None else [float(x) for x in self.theta],
               "info": self.info}
        for part in out["parts"]:
            part["theta"] = None
        return out


def correction(model, atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    """(ΔZ* (N, 3, 3), Δq (N,)) of a fitted model at ``atoms``."""
    if model.theta is None:
        raise ValueError("Modelo de cargas sin ajustar.")
    A, B = model.design(atoms)
    return np.tensordot(model.theta, A, axes=1), model.theta @ B


def load(path):
    data = json.loads(Path(path).read_text())
    return _from_dict(data)


def _from_dict(data: dict):
    theta = None if data.get("theta") is None else np.array(data["theta"])
    if data["kind"] == "combined":
        return Combined([_from_dict(p) for p in data["parts"]], theta, data.get("info", {}))
    if data["kind"] == "bond_flux":
        return BondFlux([tuple(p) for p in data["pairs"]], theta, data.get("info", {}))
    return EnvironmentFlux(tuple(data["species"]), data["r_cut"], data["n_max"], data["l_max"],
                           theta, None, data.get("info", {}))


def save(model, path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(model.to_dict(), indent=1, ensure_ascii=False))
    return path


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------

def _stack(rows, model):
    """Design and residual target over every Born charge component of every row."""
    A_all, y_all, groups = [], [], []
    for row in rows:
        if "design" not in row or row.get("design_kind") != model.kind:
            row["design"] = model.design(row["atoms"])[0]
            row["design_kind"] = model.kind
        A = row["design"]
        mask = np.isfinite(row["z_ref"]) & np.isfinite(row["z_tb"])
        A_all.append(A[:, mask].T)
        y_all.append((row["z_ref"] - row["z_tb"])[mask])
        groups += [row["group"]] * int(mask.sum())
    return np.vstack(A_all), np.concatenate(y_all), np.array(groups)


def _solve(A, y, ridge):
    scale = np.sqrt(np.mean(A ** 2, axis=0)) + 1e-12
    As = A / scale
    theta = np.linalg.solve(As.T @ As + ridge * len(y) * np.eye(As.shape[1]), As.T @ y)
    return theta / scale


def rms(rows, theta=None, model=None) -> float:
    """RMS Born-charge error (e) of TB (+ model with θ) against the reference."""
    err = []
    for row in rows:
        z = row["z_tb"].copy()
        if theta is not None:
            if "design" not in row or row.get("design_kind") != model.kind:
                row["design"] = model.design(row["atoms"])[0]
                row["design_kind"] = model.kind
            z = z + np.tensordot(theta, row["design"], axes=1)
        mask = np.isfinite(row["z_ref"]) & np.isfinite(z)
        err.append((z - row["z_ref"])[mask])
    return float(np.sqrt(np.mean(np.concatenate(err) ** 2)))


def fit(rows: list[dict], model, ridges=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
        keep_cv: bool = False) -> dict:
    """Fit ``model`` to ``rows`` ({"atoms", "group", "z_tb", "z_ref"}; Born charges
    (N, 3, 3), NaN where undefined). The ridge is chosen by leave-one-group-out
    cross-validation; the reported error is the cross-validated one, never the
    training error alone. Sets ``model.theta`` and returns the record; with ``keep_cv``
    it also holds ``"cv_theta"``: for each group, θ fitted without it."""
    A, y, groups = _stack(rows, model)
    names = sorted(set(groups))
    by_group = {g: [r for r in rows if r["group"] == g] for g in names}
    cv = {}
    for ridge in ridges:
        errs = []
        for g in names:
            train = groups != g
            theta = _solve(A[train], y[train], ridge)
            errs.append(rms(by_group[g], theta, model) ** 2 * sum(
                np.isfinite(r["z_ref"]).sum() for r in by_group[g]))
        total = sum(np.isfinite(r["z_ref"]).sum() for r in rows)
        cv[ridge] = float(np.sqrt(sum(errs) / total))
    best = min(cv, key=cv.get)
    model.theta = _solve(A, y, best)
    per_group, cv_theta = {}, {}
    for g in names:
        train = groups != g
        theta = _solve(A[train], y[train], best)
        cv_theta[g] = theta
        per_group[g] = {"tb": rms(by_group[g]), "cv": rms(by_group[g], theta, model)}
    record = {"kind": model.kind, "ridge": best, "cv_by_ridge": cv,
              "rms_tb_e": rms(rows), "rms_train_e": rms(rows, model.theta, model),
              "rms_cv_e": cv[best], "per_group": per_group, "n_params": len(model.theta),
              "n_targets": len(y)}
    model.info.update({k: record[k] for k in ("ridge", "rms_tb_e", "rms_train_e", "rms_cv_e",
                                              "n_params", "n_targets")})
    if keep_cv:
        record["cv_theta"] = cv_theta
    return record
