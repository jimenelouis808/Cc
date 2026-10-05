"""Sites (rings, heteroatoms), projected frequencies, site screening, SCC on/off, phonopy."""

import numpy as np
import pytest
from ase.build import graphene, molecule

from tbkit import sites
from tbkit.gui import actions
from tbkit.modes import vibrations
from tbkit.params import xu_carbon


def _graphene_sheet():
    sheet = graphene(a=2.46, vacuum=6.0)
    sheet.pbc = [True, True, False]
    return sheet


def test_rings_of_the_5_7_coil_and_of_graphene():
    from tbkit.recipes.nanocoil import coil

    assert sites.ring_census(coil()) == {5: 12, 6: 78, 7: 12}
    sheet = _graphene_sheet().repeat((3, 3, 1))
    assert sites.ring_census(sheet) == {6: 9}          # 18 atoms, 2 per hexagon of 6 shared ×3
    groups = sites.site_groups(sheet)
    assert list(groups) == ["anillos de 6"] and len(groups["anillos de 6"]) == 18


def test_heteroatom_groups_and_their_carbon_neighbours():
    sheet = _graphene_sheet().repeat((4, 4, 1))
    sheet[0].symbol = "N"
    groups = sites.site_groups(sheet)
    assert groups["N"] == [0]
    assert len(groups["C vecinos de heteroátomo"]) == 3


def test_projected_frequency_of_an_eigenmode_is_its_frequency():
    """The Rayleigh quotient of the model's own eigenvector is its eigenvalue; with
    unequal masses (H2O) this also checks the mass weighting."""
    from tbkit.calculator import TBCalculator

    model = actions.load_model("chno")
    water = molecule("H2O")
    relaxed = actions.relax_structure(water, model, fmax=1e-3)["atoms"]
    vib = vibrations(relaxed, model)
    for index in (6, 7, 8):
        out = sites.projected_frequency(relaxed, vib.modes[index],
                                        lambda: TBCalculator(model), max_disp=0.005)
        assert out["frequency_cm1"] == pytest.approx(vib.frequencies[index], rel=5e-3)


def test_modes_sorted_by_site_put_the_dopant_mode_first():
    model = actions.load_model("chn")
    pyridine = molecule("C5H5N")
    result = actions.vibration_modes(pyridine, model)
    rows = actions.modes_by_site(result, "N")
    shares = [r[2] for r in rows]
    assert shares == sorted(shares, reverse=True)
    assert rows[0][3] > 1.0                            # enriched on N
    assert len(actions.modes_by_site(result, None)) == len(rows)


def test_site_screening_finds_one_class_in_graphene(tmp_path):
    from tbkit.recipes.site_screening import environment_classes, screen

    sheet = _graphene_sheet().repeat((3, 3, 1))
    assert len(environment_classes(sheet)) == 1
    report = screen(sheet, "N", actions.load_model("chn"), tmp_path, kmesh=2, top=1)
    assert report["classes"] == 1 and report["sites"][0]["relative_eV"] == 0.0
    assert len(list((tmp_path / "top").glob("*.extxyz"))) == 1
    again = screen(sheet, "N", actions.load_model("chn"), tmp_path, kmesh=2, top=1)
    assert again["sites"] == report["sites"]           # resumed from the files


def test_scc_reduces_the_polar_charges_of_pyridine():
    out = actions.compare_scc(molecule("C5H5N"), actions.load_model("chn"))
    rows = {r[0]: r for r in out["rows"]}
    assert rows["|q| máx. (e)"][2] < rows["|q| máx. (e)"][1]
    assert out["converged"] and len(out["charges"]) == 11


phonopy = pytest.importorskip("phonopy")


@pytest.fixture(scope="module")
def graphene_phonons():
    from tbkit import phonopy_bridge as pb

    sheet = _graphene_sheet()
    return sheet, pb.phonopy_phonons(sheet, xu_carbon(), supercell=(4, 4, 1), kmesh=12, kT=0.05)


def test_phonopy_names_graphene_g_mode_e2g(graphene_phonons):
    from tbkit import phonopy_bridge as pb

    _, result = graphene_phonons
    rows = pb.gamma_irreps(result)
    assert rows[0]["irrep"] == "acústicos" and abs(rows[0]["frequency_cm1"]) < 1.0
    g = [r for r in rows if r["irrep"] == "E2g"]
    assert len(g) == 1 and g[0]["degeneracy"] == 2 and g[0]["frequency_cm1"] > 1400
    assert pb.point_group(result) == "6/mmm"


def test_phonopy_gamma_equals_direct_gamma_phonons_in_diamond():
    """Same force constants at Γ by two routes (supercell + phonopy, and tbkit's own
    finite differences on the primitive cell with the equivalent k mesh). Diamond,
    an insulator: graphene's G mode sits on a Kohn anomaly (Γ: 1572 cm⁻¹ with 12 k,
    1656 with 24) and finite displacements in different cells are not linear there."""
    from ase.build import bulk

    from tbkit import phonopy_bridge as pb

    diamond = bulk("C", "diamond", a=3.56)
    result = pb.phonopy_phonons(diamond, xu_carbon(), supercell=(2, 2, 2), kmesh=8, kT=0.05)
    direct = vibrations(diamond, xu_carbon(), kmesh=8, kT=0.05)
    assert result.gamma_frequencies()[-3:] == pytest.approx(direct.frequencies[-3:], abs=1.0)
    assert [r["irrep"] for r in pb.gamma_irreps(result)][-1] == "T2g"   # the Raman mode


def test_phonopy_dispersion_dos_and_cache(graphene_phonons, tmp_path):
    from tbkit import phonopy_bridge as pb

    sheet, result = graphene_phonons
    band = pb.dispersion(result, sheet, npoints=40)
    assert band["labels"][0] == "Γ" and band["frequencies"].shape[1] == 6
    assert band["frequencies"].min() > -1.0                      # no imaginary branches
    dos = pb.phonon_dos(result, sheet, mesh=12)
    assert np.trapezoid(dos["total"], dos["grid"]) == pytest.approx(6, rel=0.05)
    assert np.allclose(dos["total"], dos["C"])
    pb.phonopy_phonons(sheet, xu_carbon(), supercell=(3, 3, 1), kmesh=6, cache_dir=tmp_path)
    assert list(tmp_path.glob("forces_*.npy"))
    with pytest.raises(ValueError):
        pb.phonopy_phonons(sheet, xu_carbon(), supercell=(4, 4, 1), kmesh=6, cache_dir=tmp_path)


def test_external_vibrations_with_the_tb_calculator_equal_the_tb_modes():
    """tbkit.hybrid takes modes from any calculator: with tbkit's own, they are tbkit's."""
    from tbkit.calculator import TBCalculator
    from tbkit.hybrid import external_vibrations

    model = actions.load_model("chno")
    water = actions.relax_structure(molecule("H2O"), model, fmax=1e-3)["atoms"]
    direct = vibrations(water, model)
    external = external_vibrations(water, lambda: TBCalculator(model))
    assert external.frequencies == pytest.approx(direct.frequencies, abs=0.5)


def test_structure_screening_ranks_isomers_and_refuses_mixed_compositions(tmp_path):
    """Ranks by energy and refuses two compositions. (xu_chno puts dimethyl ether 1.44 eV
    below ethanol, GPAW 0.38 above: isomers with different bonds are outside what the
    sets were fitted for, stated in their validity. Only the mechanics are tested here.)"""
    from ase.io import write

    from tbkit.recipes.structure_screening import main

    for name in ("CH3CH2OH", "CH3OCH3"):
        write(tmp_path / f"{name}.extxyz", molecule(name))
    main([str(tmp_path / "w"), str(tmp_path / "CH3CH2OH.extxyz"), str(tmp_path / "CH3OCH3.extxyz"),
          "--model", "chno", "--fmax", "0.05"])
    import json

    report = json.loads((tmp_path / "w" / "report.json").read_text())
    energies = [row["energy_eV"] for row in report["structures"]]
    assert energies == sorted(energies) and report["structures"][0]["relative_eV"] == 0.0
    write(tmp_path / "H2O.extxyz", molecule("H2O"))
    with pytest.raises(ValueError, match="Composiciones"):
        main([str(tmp_path / "w2"), str(tmp_path / "H2O.extxyz"), str(tmp_path / "CH3OCH3.extxyz")])
