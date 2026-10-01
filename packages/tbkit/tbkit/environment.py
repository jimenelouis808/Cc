"""Environment-dependent tight binding (Tang, Wang, Chan and Ho, Phys. Rev. B 53, 979 (1996)).

Beyond the two-centre form: every hopping V_ssσ, V_spσ, V_ppσ, V_ppπ, the
on-site shift Δe and the repulsive pair potential φ take the form of their
Eq. (1),

    h(r_ij) = α1 R_ij^-α2 exp(-α3 R_ij^α4) (1 - S_ij),

with a **screening** S_ij = tanh ξ_ij, ξ_ij = β1 Σ_l exp[-β2 ((r_il + r_jl)/r_ij)^β3]
(Eqs. 2-3: an atom l near the line i-j weakens the i-j interaction), and a
**scaled distance** R_ij = r_ij [1 + δ/2 ((g_i - g0)/g0 + (g_j - g0)/g0)]
(Eq. 4) through the effective coordinations g_i = Σ_j (1 - S_ij) (Eq. 5, a
screening of its own). On-site energies are e_λ,i = e_λ,0 + Σ_j Δe(r_ij)
(Eq. 6, the same Δe for s and p), and E_rep = Σ_i f(Σ_j φ(r_ij)) with f a
quartic (Eq. 7, Table II). Parameters: Tables I and II and the text of the
paper; they live in ``parameters/tang_carbon.json``, never here.

Not in the paper, and therefore ours (stated in the parameter file): smooth
cutoffs. Pair functions are multiplied by a quintic switch from ``pair_cut``
(4.2 to 5.0 Å), and every third-atom term of a screening sum by the same
switch on r_il and r_jl from ``screen_cut`` (5.0 to 6.0 Å), so energies and
forces are continuous. With them the effective coordinations of the paper
(chain 2.08639 ... fcc 11.89829) are reproduced to 1e-3 (tested).

Forces are Hellmann-Feynman (orthogonal basis): ∂E/∂X = Σ_bonds ∂Tr[ρH]/∂V ·
∂V/∂X + Σ_i n_i ∂e_i/∂X + ∂E_rep/∂X, where every V depends on the bond, on the
atoms that screen it and, through R_ij, on the coordinations of both ends: the
chain rule is carried by hand in :meth:`Environment.backward` and tested
against finite differences of the energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

HOPPINGS = ("sss", "sps", "pps", "ppp")
FUNCTIONS = HOPPINGS + ("rep", "onsite")


def _switch(r, r1, rm):
    """Quintic switch: 1 below r1, 0 above rm, value/slope/curvature continuous."""
    x = np.clip((np.asarray(r, dtype=float) - r1) / (rm - r1), 0.0, 1.0)
    s = 1 - 10 * x ** 3 + 15 * x ** 4 - 6 * x ** 5
    ds = (-30 * x ** 2 + 60 * x ** 3 - 30 * x ** 4) / (rm - r1)
    return s, ds


@dataclass(frozen=True)
class Function:
    """One row of Table I: α1-α4 (Eq. 1), β1-β3 (Eq. 3), δ (Eq. 4)."""

    a1: float
    a2: float
    a3: float
    a4: float
    b1: float
    b2: float
    b3: float
    delta: float


@dataclass
class TangSpec:
    functions: dict                    # name -> Function (HOPPINGS, "rep", "onsite")
    g_screen: tuple                    # β1, β2, β3 of the coordination screening (Eq. 5)
    g0: float                          # reference coordination (diamond)
    e0: dict                           # {"s": e_s0, "p": e_p0}, eV
    poly: tuple                        # c0..c4 of f(x), eV (Table II)
    pair_cut: tuple = (4.2, 5.0)       # Å, ours
    screen_cut: tuple = (5.0, 6.0)     # Å, ours
    element: str = "C"

    @classmethod
    def from_dict(cls, data: dict) -> TangSpec:
        def value(v):
            return float(v["value"]) if isinstance(v, dict) else float(v)

        functions = {name: Function(*(value(entry[k]) for k in
                                      ("a1", "a2", "a3", "a4", "b1", "b2", "b3", "delta")))
                     for name, entry in data["functions"].items()}
        return cls(functions, tuple(value(v) for v in data["g_screen"]), value(data["g0"]),
                   {k: value(v) for k, v in data["e0"].items()},
                   tuple(value(v) for v in data["poly"]),
                   tuple(data.get("pair_cut", (4.2, 5.0))),
                   tuple(data.get("screen_cut", (5.0, 6.0))), data.get("element", "C"))

    def cutoff(self) -> float:
        return self.pair_cut[1]

    def evaluate(self, atoms: Atoms) -> Environment:
        return Environment(self, atoms)


@dataclass
class Environment:
    """Every pair quantity of one structure, and the reverse pass for forces."""

    spec: TangSpec
    atoms: Atoms
    pairs: dict = field(init=False)

    def __post_init__(self):
        spec, atoms = self.spec, self.atoms
        n = len(atoms)
        ii, jj, dd, vv, ss = neighbor_list("ijdDS", atoms, spec.screen_cut[1])
        self._lists = [np.flatnonzero(ii == a) for a in range(n)]
        self._nl = (ii, jj, dd, vv, ss)
        pair = np.flatnonzero(dd < spec.pair_cut[1])
        self.i, self.j = ii[pair], jj[pair]
        self.vec, self.r, self.shift = vv[pair], dd[pair], ss[pair]
        self._pair_index = pair
        # third atoms of each pair: neighbours l of i, with r_jl from the vectors
        self._third = []
        for p, (i, v) in enumerate(zip(self.i, self.vec, strict=True)):
            lst = self._lists[i]
            v_il = vv[lst]
            r_il = dd[lst]
            v_jl = v_il - v
            r_jl = np.linalg.norm(v_jl, axis=1)
            keep = r_jl > 1e-8                        # l is not j itself
            self._third.append((jj[lst][keep], v_il[keep], r_il[keep], v_jl[keep], r_jl[keep]))
        # coordinations (Eq. 5) over all pairs within the screening range
        self.g = np.zeros(n)
        self._g_terms = []
        for p, (i, j, v, r) in enumerate(zip(self.i, self.j, self.vec, self.r, strict=True)):
            xi, _ = self._xi(p, *spec.g_screen)
            t, _ = _switch(r, *spec.pair_cut)
            self.g[i] += t * (1 - np.tanh(xi))
        # every function on every pair
        self.values = {}
        self._parts = {}
        for name, f in spec.functions.items():
            self.values[name], self._parts[name] = self._function(f)
        self.onsite_shift = np.zeros(n)
        np.add.at(self.onsite_shift, self.i, self.values["onsite"])

    # ------------------------------------------------------------------ forward
    def _xi(self, p, b1, b2, b3):
        """ξ of pair p and dξ/du_l per third atom (u = (r_il + r_jl)/r_ij)."""
        _, _, r_il, _, r_jl = self._third[p]
        if len(r_il) == 0:
            return 0.0, (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))
        u = (r_il + r_jl) / self.r[p]
        e = np.exp(-b2 * u ** b3)
        ti, dti = _switch(r_il, *self.spec.screen_cut)
        tj, dtj = _switch(r_jl, *self.spec.screen_cut)
        xi = b1 * float(np.sum(e * ti * tj))
        de_du = -b2 * b3 * u ** (b3 - 1) * e
        return xi, (u, b1 * de_du * ti * tj, b1 * e * dti * tj, b1 * e * ti * dtj)

    def _function(self, f: Function):
        spec = self.spec
        scale = 1 + f.delta / (2 * spec.g0) * (self.g[self.i] + self.g[self.j] - 2 * spec.g0)
        big_r = self.r * scale
        base = f.a1 * big_r ** (-f.a2) * np.exp(-f.a3 * big_r ** f.a4)
        xi = np.array([self._xi(p, f.b1, f.b2, f.b3)[0] for p in range(len(self.r))])
        screen = 1 - np.tanh(xi)
        t, dt = _switch(self.r, *spec.pair_cut)
        value = base * screen * t
        dvalue_dR = value * (-f.a2 / big_r - f.a3 * f.a4 * big_r ** (f.a4 - 1))
        dvalue_dxi = -base * (1 - np.tanh(xi) ** 2) * t
        dvalue_dr_taper = base * screen * dt
        return value, (scale, dvalue_dR, dvalue_dxi, dvalue_dr_taper, f)

    # ------------------------------------------------------------------ blocks
    def block(self, p: int) -> np.ndarray:
        """4x4 s,px,py,pz block of pair p (from i to j)."""
        c = self.vec[p] / self.r[p]
        v = [self.values[k][p] for k in HOPPINGS]
        return sk_block(v, c)

    # ------------------------------------------------------------------ reverse
    def backward(self, weights: dict, angular: np.ndarray | None = None) -> np.ndarray:
        """dE/dX (N, 3) for ``E = Σ_name Σ_p weights[name][p] · value[name][p]``
        plus, with ``angular`` (P, 3), the direct dependence of the blocks on the
        bond direction at fixed values (∂E/∂vec_p)."""
        n = len(self.atoms)
        grad = np.zeros((n, 3))
        g_bar = np.zeros(n)
        unit = self.vec / self.r[:, None]

        def add_pair(p, coefficient):              # coefficient · ∂r_ij/∂X
            grad[self.j[p]] += coefficient * unit[p]
            grad[self.i[p]] -= coefficient * unit[p]

        def add_xi(p, coefficient, b):             # coefficient · ∂ξ_p/∂X
            if coefficient == 0.0:
                return
            ls, v_il, r_il, v_jl, r_jl = self._third[p]
            if len(ls) == 0:
                return
            u, dxi_du, dxi_dril, dxi_drjl = self._xi(p, *b)[1]
            r = self.r[p]
            # u depends on r_il, r_jl (1/r each) and on r_ij (-u/r)
            c_il = coefficient * (dxi_du / r + dxi_dril)
            c_jl = coefficient * (dxi_du / r + dxi_drjl)
            add_pair(p, coefficient * float(np.sum(-dxi_du * u / r)))
            e_il = v_il / r_il[:, None]
            e_jl = v_jl / r_jl[:, None]
            np.add.at(grad, ls, c_il[:, None] * e_il + c_jl[:, None] * e_jl)
            grad[self.i[p]] -= np.sum(c_il[:, None] * e_il, axis=0)
            grad[self.j[p]] -= np.sum(c_jl[:, None] * e_jl, axis=0)

        for name, w in weights.items():
            scale, dvalue_dR, dvalue_dxi, dvalue_dr_taper, f = self._parts[name]
            for p in np.flatnonzero(w):
                wp = float(w[p])
                # R = r · scale(g_i, g_j)
                add_pair(p, wp * (dvalue_dR[p] * scale[p] + dvalue_dr_taper[p]))
                dR_dg = self.r[p] * f.delta / (2 * self.spec.g0)
                g_bar[self.i[p]] += wp * dvalue_dR[p] * dR_dg
                g_bar[self.j[p]] += wp * dvalue_dR[p] * dR_dg
                add_xi(p, wp * dvalue_dxi[p], (f.b1, f.b2, f.b3))
        # g_i = Σ_p(i) t(r) (1 - tanh ξ^g)
        for p in range(len(self.r)):
            gb = g_bar[self.i[p]]
            if gb == 0.0:
                continue
            xi, _ = self._xi(p, *self.spec.g_screen)
            t, dt = _switch(self.r[p], *self.spec.pair_cut)
            add_pair(p, gb * dt * (1 - np.tanh(xi)))
            add_xi(p, -gb * t * (1 - np.tanh(xi) ** 2), self.spec.g_screen)
        if angular is not None:
            for p in range(len(self.r)):
                grad[self.j[p]] += angular[p]
                grad[self.i[p]] -= angular[p]
        return grad


def sk_block(v, c) -> np.ndarray:
    """s, px, py, pz Slater-Koster block for V = (ssσ, spσ, ppσ, ppπ), direction c."""
    sss, sps, pps, ppp = v
    out = np.empty((4, 4))
    out[0, 0] = sss
    out[0, 1:] = c * sps
    out[1:, 0] = -c * sps
    out[1:, 1:] = np.outer(c, c) * (pps - ppp) + np.eye(3) * ppp
    return out


def sk_block_gradients(v, c, r):
    """∂block/∂V_k (4 blocks) and ∂block/∂vec (3 blocks) at fixed V."""
    _, sps, pps, ppp = v
    d_v = []
    for k in range(4):
        e = np.zeros(4)
        e[k] = 1.0
        d_v.append(sk_block(e, c))
    d_vec = []
    for a in range(3):
        dc = (np.eye(3)[a] - c * c[a]) / r          # ∂c/∂vec_a
        out = np.zeros((4, 4))
        out[0, 1:] = dc * sps
        out[1:, 0] = -dc * sps
        out[1:, 1:] = (np.outer(dc, c) + np.outer(c, dc)) * (pps - ppp)
        d_vec.append(out)
    return d_v, d_vec


@dataclass
class TangRepulsive:
    """E_rep = Σ_i f(Σ_j φ(r_ij)) (Eq. 7) with the environment-dependent φ."""

    spec: TangSpec

    def cutoff(self) -> float:
        return self.spec.screen_cut[1]

    def energy_and_forces(self, atoms: Atoms, env: Environment | None = None):
        env = env or self.spec.evaluate(atoms)
        x = np.zeros(len(atoms))
        np.add.at(x, env.i, env.values["rep"])
        c = self.spec.poly
        energy = float(sum(sum(c[k] * x ** k for k in range(len(c)))))
        df = sum(k * c[k] * x ** (k - 1) for k in range(1, len(c)))
        weights = {"rep": df[env.i]}
        return energy, -env.backward(weights)

    def to_dict(self) -> dict:
        return {"type": "tang"}
