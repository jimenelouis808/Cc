"""Vibrational-mode analysis (tbkit.modes): exact relations and known characters."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule
from ase.optimize import BFGS

from tbkit.calculator import TBCalculator
from tbkit.modes import (
    Vibrations,
    cylindrical_character,
    describe,
    localisation,
    mass_versus_chemistry,
    participation,
    vibrational_dos,
    vibrations,
)
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
def benzene(chn):
    return vibrations(_relaxed("C6H6", chn), chn)


@pytest.fixture(scope="module")
def pyridine(chn):
    return vibrations(_relaxed("C5H5N", chn), chn)


def test_same_frequencies_as_the_raman_path(chn, pyridine):
    from tbkit.raman import _model_phonons

    frequencies, _ = _model_phonons(pyridine.atoms.copy(), chn, 12, 0.02, 0.005, [])
    assert np.sort(pyridine.frequencies) == pytest.approx(np.sort(frequencies), abs=0.05)


def test_eigenvectors_orthonormal_and_modes_mass_scaled(pyridine):
    flat = pyridine.eigenvectors.reshape(len(pyridine.frequencies), -1)
    assert flat @ flat.T == pytest.approx(np.eye(len(flat)), abs=1e-10)
    assert pyridine.modes == pytest.approx(
        pyridine.eigenvectors / np.sqrt(pyridine.masses)[None, :, None])


def test_scaling_every_mass_scales_every_frequency(pyridine):
    heavier = pyridine.with_masses(4 * pyridine.masses)
    internal = np.abs(pyridine.frequencies) > 100
    assert heavier.frequencies[internal] == pytest.approx(
        pyridine.frequencies[internal] / 2, rel=1e-10)


def test_participation_is_a_partition(pyridine):
    shares = participation(pyridine)
    assert set(shares) == {"C", "N", "H"}
    assert sum(shares.values()) == pytest.approx(np.ones(len(pyridine.frequencies)))
    ch = int(np.argmax(pyridine.frequencies))                  # a C-H stretch
    assert shares["H"][ch] > 0.8
    nitrogen = [i for i, s in enumerate(pyridine.atoms.get_chemical_symbols()) if s == "N"]
    assert localisation(pyridine, nitrogen).max() <= len(pyridine.atoms) + 1e-9


def test_benzene_breathing_is_radial(benzene):
    normal = np.cross(*(benzene.atoms.positions[1:3] - benzene.atoms.positions[0]))
    character = cylindrical_character(benzene, axis=normal)
    carbons = [i for i, s in enumerate(benzene.atoms.get_chemical_symbols()) if s == "C"]
    ring = participation(benzene, {"C": carbons})["C"]
    candidates = np.flatnonzero((benzene.frequencies > 900) & (benzene.frequencies < 1100))
    breathing = candidates[np.argmax(character["radial"][candidates] * ring[candidates])]
    assert character["radial"][breathing] > 0.95
    total = character["radial"] + character["tangential"] + character["axial"]
    assert total == pytest.approx(np.ones_like(total), abs=1e-8)


def test_dos_counts_the_internal_modes(benzene):
    dos = vibrational_dos(benzene, grid=np.arange(-200.0, 4000.0, 1.0), sigma=8.0)
    area = np.trapezoid(dos["total"], dos["grid"])
    assert area == pytest.approx(3 * len(benzene.atoms) - 6, rel=1e-3)
    assert dos["C"] + dos["H"] == pytest.approx(dos["total"])


def test_mass_versus_chemistry(pyridine):
    nitrogen = [i for i, s in enumerate(pyridine.atoms.get_chemical_symbols()) if s == "N"]
    result = mass_versus_chemistry(pyridine, nitrogen)
    shift = result["reference_mass"] - result["actual"]
    internal = np.abs(result["actual"]) > 100
    assert np.all(shift[internal] >= -1e-6)          # a lighter N can only raise frequencies
    assert shift[internal].max() > 1.0


def test_save_never_overwrites(pyridine, tmp_path):
    path = pyridine.save(tmp_path / "run")
    data = np.load(path)
    assert data["frequencies_cm1"] == pytest.approx(pyridine.frequencies)
    with pytest.raises(FileExistsError):
        pyridine.save(tmp_path / "run")


def test_describe(pyridine):
    line = describe(pyridine, int(np.argmax(pyridine.frequencies)))
    assert "cm⁻¹" in line and "H" in line


def test_from_hessian_round_trip(pyridine):
    again = Vibrations.from_hessian(pyridine.atoms, pyridine.hessian)
    assert again.frequencies == pytest.approx(pyridine.frequencies)


def test_gapless_crystals_warn_and_the_k_mesh_converges_the_g_mode():
    """Graphene is gapless: a coarse mesh softens G (Kohn anomaly sampled badly), so
    the modes carry a warning and kmesh_convergence finds the mesh that holds still.
    Diamond has a gap and no warning."""
    from ase.build import bulk, graphene

    from tbkit.modes import GAPLESS_WARNING, vibrations
    from tbkit.params import xu_carbon
    from tbkit.tasks import kmesh_convergence

    sheet = graphene(a=2.46, vacuum=6.0)
    sheet.pbc = [True, True, False]
    assert GAPLESS_WARNING in vibrations(sheet, xu_carbon(), kmesh=12, kT=0.05).warnings
    diamond = bulk("C", "diamond", a=3.56)
    assert GAPLESS_WARNING not in vibrations(diamond, xu_carbon(), kmesh=4, kT=0.05).warnings
    out = kmesh_convergence(sheet, xu_carbon(), meshes=(24, 36, 48), kT=0.05, top=1,
                            tolerance=5.0)
    g = [row["highest_cm1"][-1] for row in out["rows"]]
    assert g == sorted(g) and g[-1] - g[0] > 10          # softened on the coarse mesh
    assert out["converged_kmesh"] == 36


def test_cached_hessian_rows_and_embedding(tmp_path):
    """hessian_rows gives the Hessian of modes.vibrations (same differences, cached and
    shared by parts); embedding the rows of a region into a reference equal to the
    same structure's Hessian returns that Hessian."""
    from ase.build import molecule

    from tbkit.calculator import TBCalculator
    from tbkit.modes import embedded_hessian, hessian_rows, vibrations
    from tbkit.params import load_parameters

    model = load_parameters("xu_chno")
    water = molecule("CH3OH")
    vib = vibrations(water, model, delta=0.01)
    make = lambda: TBCalculator(model)                    # noqa: E731
    assert hessian_rows(water, make, tmp_path, part=(0, 2)) is None       # half done
    full = hessian_rows(water, make, tmp_path, part=(1, 2))
    assert np.allclose(0.5 * (full + full.T), vib.hessian, atol=1e-6)
    sym = 0.5 * (full + full.T)
    region = [0, 1, 4]
    rows = hessian_rows(water, make, tmp_path, indices=region)
    embedded = embedded_hessian(sym, len(water), len(water), region, rows)
    # equal up to half the finite-difference asymmetry of the rows
    assert np.abs(embedded - sym).max() <= 0.5 * np.abs(full - full.T).max() + 1e-12
    with pytest.raises(ValueError):
        embedded_hessian(sym[:-3, :-3], len(water) - 1, len(water), region, rows)
