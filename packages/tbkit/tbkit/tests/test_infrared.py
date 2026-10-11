"""IR intensities from the TB dipole (tbkit.infrared): exact relations and symmetry."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule
from ase.optimize import BFGS

from tbkit.calculator import TBCalculator
from tbkit.hamiltonian import System
from tbkit.infrared import DEBYE_PER_EA, dipole_moment, infrared
from tbkit.params import load_parameters


@pytest.fixture(scope="module")
def chn():
    return load_parameters("xu_chn")


def _relaxed(name, model):
    atoms = molecule(name)
    atoms.calc = TBCalculator(model)
    BFGS(atoms, logfile=None).run(fmax=0.002, steps=400)
    return atoms.copy()


@pytest.fixture(scope="module")
def ammonia(chn):
    atoms = _relaxed("NH3", chn)
    return atoms, infrared(atoms, chn)


def test_born_charges_sum_to_zero(ammonia):
    _, result = ammonia
    assert np.abs(result.born.sum(axis=0)).max() < 1e-4
    assert not result.warnings or all("Born" not in w for w in result.warnings)


def test_rotation_leaves_intensities_unchanged(chn, ammonia):
    atoms, result = ammonia
    turned = atoms.copy()
    turned.rotate(37, (1, 2, 3), center="COM")
    again = infrared(turned, chn)
    assert again.frequencies == pytest.approx(result.frequencies, abs=0.5)
    assert again.intensities == pytest.approx(result.intensities, rel=1e-3, abs=1e-3)


def test_benzene_selection_rules(chn):
    """D6h: only A2u (one) and E1u (three pairs) absorb."""
    result = infrared(_relaxed("C6H6", chn), chn)
    assert np.linalg.norm(result.dipole) < 1e-6
    active = result.groups(tolerance=0.5, threshold=1e-3)
    degeneracies = sorted(g["degeneracy"] for g in active)
    assert degeneracies == [1, 2, 2, 2]


def test_ammonia_dipole_from_charges(chn):
    """Mulliken charges alone give NH3's dipole: 1.10 D at the TB geometry
    (1.31 D at GPAW's), 1.47 D measured."""
    mu = dipole_moment(System.build(_relaxed("NH3", chn), chn), onsite_dipoles=False)
    assert np.linalg.norm(mu) * DEBYE_PER_EA == pytest.approx(1.47, rel=0.3)


def test_imported_modes_give_the_same_result(chn, ammonia):
    from tbkit.raman import _model_phonons

    atoms, result = ammonia
    frequencies, modes = _model_phonons(atoms.copy(), chn, 12, 0.01, 0.005, [])
    again = infrared(atoms, chn, phonons=(frequencies, modes))
    assert again.intensities == pytest.approx(result.intensities, rel=1e-6, abs=1e-6)


def test_refuses_crystals(chn):
    from ase.build import bulk

    with pytest.raises(ValueError, match="finitos"):
        infrared(bulk("C", "diamond", a=3.56), chn)


def test_benzene_against_gpaw_on_gpaw_modes(chn):
    """TB Born charges on GPAW's own modes (so only the dipole model is tested):
    the IR-active benzene modes within a factor of 2 of GPAW."""
    import json

    from tbkit.infrared import KM_PER_MOL, born_charges
    from tbkit.params import PARAMETER_DIR
    from tbkit.references import load_references

    data = json.loads((PARAMETER_DIR / "references" / "gpaw_chn_ir.json")
                      .read_text(encoding="utf-8"))["infrared"]
    entry = next(e for e in data if e["group"] == "C6H6")
    refs, _ = load_references(PARAMETER_DIR / "references" / "gpaw_chn.json")
    atoms = next(r for r in refs if r.label == "C6H6/eq").atoms
    modes = np.array(entry["eigenvectors"]) / np.sqrt(atoms.get_masses())[None, :, None]
    gpaw = np.array(entry["intensities_km_mol"])
    z = born_charges(atoms, chn)
    tb = np.sum(np.einsum("aij,mai->mj", z, modes) ** 2, axis=1) * DEBYE_PER_EA ** 2 * KM_PER_MOL
    strong = gpaw > 0.1 * gpaw.max()
    assert np.all(np.abs(np.log10(tb[strong] / gpaw[strong])) < np.log10(2.0))


def test_wire_born_charges_match_the_molecule(tmp_path):
    """A molecule in a cell periodic along z: the transverse Born charges of
    ``born_rows`` are those of the finite molecule (to the weak coupling with its
    images 24 Å away), the axial ones are left out, and they obey the sum rule."""
    from tbkit.infrared import born_charges, born_rows, mode_intensities

    model = load_parameters("xu_chn")
    wire = molecule("CH3CN")
    wire.center(vacuum=12.0)
    wire.pbc = (False, False, True)
    finite = wire.copy()
    finite.pbc = False
    z_wire = born_rows(wire, lambda: TBCalculator(model, kpts=(1, 1, 2), kT=0.05), tmp_path)
    z_finite = born_charges(finite, model, delta=0.005, kT=0.05)
    assert np.isnan(z_wire[:, :, 2]).all()
    assert np.abs(z_wire[:, :, :2] - z_finite[:, :, :2]).max() < 1e-3
    assert np.abs(z_wire[:, :, :2].sum(axis=0)).max() < 1e-4
    rng = np.random.default_rng(0)
    modes = rng.normal(size=(3, len(wire), 3))
    intensities, info = mode_intensities(z_wire, modes)
    assert info["axes"] == [0, 1] and intensities.shape == (3,)
    assert born_rows(wire, None, tmp_path) is not None        # all cached: no calculator
