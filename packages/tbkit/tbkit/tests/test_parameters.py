"""Parameters from files and from fits.

The .skf fixtures are generated here in the documented simple format (the
published sets are not redistributable), from laws whose answer is known:
a file tabulating the Xu carbon model must give back the Xu model.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.units import Bohr, Hartree

from tbkit import System, pi_model, solve, xu_carbon
from tbkit.fit import Reference, fit, read_gpaw_eigenvalues
from tbkit.kpoints import mesh
from tbkit.params import Constant, TBModel
from tbkit.skf import load_skf_set, read_skf


def _write_skf(path, grid, columns, onsite=None, repeat_zeros=True):
    """columns: dict name -> values (Hartree / dimensionless) on r = i*grid (Bohr)."""
    names = ("Hdd0", "Hdd1", "Hdd2", "Hpd0", "Hpd1", "Hpp0", "Hpp1", "Hsd0", "Hsp0", "Hss0",
             "Sdd0", "Sdd1", "Sdd2", "Spd0", "Spd1", "Spp0", "Spp1", "Ssd0", "Ssp0", "Sss0")
    n = len(next(iter(columns.values())))
    lines = [f"{grid} {n}"]
    if onsite is not None:
        lines.append(" ".join(str(v) for v in onsite))
    lines.append("12.01 19*0.0")
    for i in range(n):
        row = [columns.get(name, np.zeros(n))[i] for name in names]
        # Fortran-style repetition for the five leading d columns.
        lines.append("5*0.0 " + " ".join(f"{v:.10e}" for v in row[5:])
                     if repeat_zeros else " ".join(f"{v:.10e}" for v in row))
    lines += ["Spline", "1 3.0", "0 0 0", "0 3.0 0 0 0 0"]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def xu_skf(tmp_path):
    xu = xu_carbon()
    grid = 0.02                                       # Bohr
    r_bohr = np.arange(1, 400) * grid
    r = r_bohr * Bohr
    columns = {"Hss0": xu.hopping[("C", "C", "sss")](r) / Hartree,
               "Hsp0": xu.hopping[("C", "C", "sps")](r) / Hartree,
               "Hpp0": xu.hopping[("C", "C", "pps")](r) / Hartree,
               "Hpp1": xu.hopping[("C", "C", "ppp")](r) / Hartree}
    # Ed Ep Es SPE Ud Up Us fd fp fs
    onsite = (0.0, 3.71 / Hartree, -2.99 / Hartree, 0.0, 0.0, 0.36, 0.36, 0.0, 2.0, 2.0)
    _write_skf(tmp_path / "C-C.skf", grid, columns, onsite)
    return tmp_path


class TestSKF:
    def test_reads_units_and_onsite(self, xu_skf):
        sk = read_skf(xu_skf / "C-C.skf")
        assert sk.onsite["s"] == pytest.approx(-2.99) and sk.onsite["p"] == pytest.approx(3.71)
        assert sk.occupations["s"] + sk.occupations["p"] == 4.0
        assert sk.r[0] == pytest.approx(0.02 * Bohr)
        assert sk.table["Hpp1"][100] == pytest.approx(
            xu_carbon().hopping[("C", "C", "ppp")](sk.r[100]))

    def test_tabulated_xu_reproduces_xu(self, xu_skf):
        model = load_skf_set(xu_skf, {"C": ("s", "px", "py", "pz")})
        assert model.scc                  # DFTB sets are made for SCC
        assert model.orthogonal                  # all overlap columns were zero
        assert model.valence["C"] == 4.0 and model.hubbard_u["C"] == pytest.approx(
            0.36 * Hartree)
        diamond = bulk("C", "diamond", a=3.567)
        k, w = mesh(diamond, 4)
        reference = solve(System.build(diamond, xu_carbon()), k, w).energies
        tabulated = solve(System.build(diamond, model), k, w).energies
        assert tabulated == pytest.approx(reference, abs=2e-3)

    def test_heteronuclear_orientation(self, tmp_path):
        """<s_C|H|s_H> and the C-H spσ sign: H on +z of C couples to C's pz with
        -V_spσ(s on H, p on C), read from H-C.skf's Hsp0."""
        grid = 0.1
        n = 60
        const = np.full(n, 1.0)
        for a, b, cols in (("C", "C", {"Hpp1": -0.1 * const}),
                           ("H", "H", {"Hss0": -0.2 * const}),
                           ("C", "H", {"Hss0": -0.3 * const, "Hsp0": 0.0 * const}),
                           ("H", "C", {"Hss0": -0.3 * const, "Hsp0": 0.25 * const})):
            onsite = None
            if a == b:
                onsite = ((0, -0.2, -0.5, 0, 0, 0.4, 0.4, 0, 2, 2) if a == "C"
                          else (0, 0, -0.24, 0, 0, 0, 0.42, 0, 0, 1))
            _write_skf(tmp_path / f"{a}-{b}.skf", grid, cols, onsite)
        model = load_skf_set(tmp_path, {"C": ("s", "px", "py", "pz"), "H": ("s",)})
        ch = Atoms("CH", positions=[[0, 0, 0], [0, 0, 1.1]])
        system = System.build(ch, model)
        h, _ = system.hamiltonian()
        pz_c, s_h = system.basis.first[0] + 3, system.basis.first[1]
        assert h[pz_c, s_h] == pytest.approx(-0.25 * Hartree, rel=1e-3)
        assert h[system.basis.first[0], s_h] == pytest.approx(-0.3 * Hartree, rel=1e-3)
        assert h[pz_c, s_h] == pytest.approx(h[s_h, pz_c])

    def test_refusals(self, tmp_path):
        bad = tmp_path / "carbono.skf"
        bad.write_text("0.02 10\n")
        with pytest.raises(ValueError, match="A-B.skf"):
            read_skf(bad)
        extended = tmp_path / "C-C.skf"
        extended.write_text("@ 0.02 10\n")
        with pytest.raises(ValueError, match="extendido"):
            read_skf(extended)
        incomplete = tmp_path / "incompleto"
        incomplete.mkdir()
        _write_skf(incomplete / "C-C.skf", 0.1, {"Hpp1": np.full(5, -0.1)},
                   (0, -0.2, -0.5, 0, 0, 0.4, 0.4, 0, 2, 2))
        with pytest.raises(FileNotFoundError, match="C-H.skf"):
            load_skf_set(incomplete, {"C": ("s",), "H": ("s",)})


class TestFit:
    def test_recovers_a_known_hopping(self):
        from tbkit.tests.test_core import _acene

        naphthalene = _acene(2)
        truth = solve(System.build(naphthalene, pi_model(t=-2.9)), kT=1e-4)
        reference = Reference(naphthalene, truth.energies[0, 0], n_occupied=5, n_below=4,
                              n_above=4, label="naftaleno")
        result = fit(lambda x: pi_model(t=float(x[0]), heteroatoms=False), [-2.4],
                     [reference], names=["t"], bounds=([-4.0], [-1.0]))
        assert result.success and result.x[0] == pytest.approx(-2.9, abs=1e-4)
        assert result.rms["naftaleno"] < 1e-6

    def test_window_needs_enough_levels(self):
        reference = Reference(molecule("C6H6"), np.array([-1.0, 1.0]), n_occupied=1,
                              n_below=3, n_above=3, label="corto")
        with pytest.raises(ValueError, match="corto"):
            reference.window()

    def test_fit_an_onsite_energy(self):
        """Two-site model with a heteroatom: fit its on-site energy."""
        def build(x):
            return TBModel("AB", orbitals={"C": ("pi",), "N": ("pi",)},
                           onsite={"C": {"p": 0.0}, "N": {"p": float(x[0])}},
                           hopping={("C", "N", "ppp"): Constant(-2.7, 1.8),
                                    ("C", "C", "ppp"): Constant(-2.7, 1.8)},
                           valence={"C": 1.0, "N": 1.0})
        pyrrole_like = molecule("C5H5N")
        truth = solve(System.build(pyrrole_like, build([-1.3])), kT=1e-4).energies[0, 0]
        # Same energy zero as the model: absolute alignment. At mid-gap the
        # on-site energy is ambiguous (a mirror solution fits as well).
        reference = Reference(pyrrole_like, truth, n_occupied=3, n_below=3, n_above=3,
                              label="piridina", align="none")
        result = fit(build, [0.0], [reference], names=["eps_N"])
        assert result.x[0] == pytest.approx(-1.3, abs=1e-4)


GPAW_TXT = """
  ___ ___ ___ _ _ _
 Fermi level: -3.45678
 Band  Eigenvalues  Occupancy
    0    -20.00000    2.00000
    1    -12.50000    2.00000
    2     -6.00000    2.00000
    3     -1.00000    0.00000

Fermi level: -3.50000
 Band  Eigenvalues  Occupancy
    0    -21.00000    2.00000
    1    -13.00000    2.00000
    2     -6.50000    2.00000
    3     -0.50000    0.00000
    4      1.50000    0.00000
"""

GPAW_SPIN = """
Fermi levels: -3.10000, -3.20000
                    Up                     Down
 Band  Eigenvalues  Occupancy  Eigenvalues  Occupancy
    0    -10.00000    1.00000    -9.80000    1.00000
    1     -4.00000    1.00000     -2.00000    0.00000
"""


def test_read_gpaw_eigenvalues(tmp_path):
    path = tmp_path / "gpaw.txt"
    path.write_text(GPAW_TXT)
    levels = read_gpaw_eigenvalues(path)
    assert levels.energies.shape == (1, 5)             # the last table
    assert levels.energies[0, 2] == pytest.approx(-6.5) and levels.n_occupied() == 3
    assert levels.fermi == pytest.approx(-3.5)
    path.write_text(GPAW_SPIN)
    spin = read_gpaw_eigenvalues(path)
    assert spin.energies.shape == (2, 2) and spin.n_occupied(0) == 2 and spin.n_occupied(1) == 1


def test_frequency_scale_is_reported_next_to_the_raw_frequencies():
    from ase.build import molecule

    from tbkit.params import load_parameters, xu_carbon
    from tbkit.recipes.frequency_scaling import SETS, scale_factor
    from tbkit.tasks import frequency_scale, phonons

    for name in SETS:
        entry = load_parameters(name).metadata["parameters"]["frequency_scale"]
        assert entry["source"] and entry["unit"] == ""
        assert entry["rms_after_cm1"] <= entry["rms_before_cm1"]
    # Scott-Radom on a model that is 5 % low everywhere gives back 1/0.95.
    got = scale_factor({"m": {"tb": [950.0, 1900.0, 2850.0], "gpaw": [1000.0, 2000.0, 3000.0]}})
    assert got["value"] == pytest.approx(1 / 0.95, abs=1e-4) and got["rms_after_cm1"] < 0.5
    assert frequency_scale(xu_carbon()) is None
    model = load_parameters("xu_chno")
    result, _ = phonons(molecule("H2O"), model)
    assert result["frequencies_scaled_cm1"] == pytest.approx(
        frequency_scale(model) * result["frequencies_cm1"])
