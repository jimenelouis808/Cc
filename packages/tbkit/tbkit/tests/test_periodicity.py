"""Periodicity: automatic by default (bonds that cross the cell; vacuum is not a lattice
direction), editable by hand, and the checks a calculation runs before it starts."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, graphene_nanoribbon, molecule

from tbkit import structure_io as sio
from tbkit.gui import actions


def _ribbon(pbc=None):
    ribbon = graphene_nanoribbon(2, 1, type="armchair", saturated=True, vacuum=6.0)
    if pbc is not None:
        ribbon.pbc = pbc
    return ribbon


def _graphite():
    a, c = 2.46, 6.70
    cell = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
    frac = [[0, 0, 0.25], [1 / 3, 2 / 3, 0.25], [0, 0, 0.75], [2 / 3, 1 / 3, 0.75]]
    return Atoms("C4", scaled_positions=frac, cell=cell, pbc=True)


# --- the automatic rule -------------------------------------------------------------------

def test_a_tube_in_a_box_declared_3d_is_1d():
    axes, reasons = sio.auto_periodicity(_ribbon(True))
    assert axes == [False, False, True]
    assert "vacío" in reasons[0] and "declaraba periódico" in reasons[0]
    assert reasons[2] == "enlaces que cruzan la celda"


def test_a_crystal_stays_3d_and_graphite_keeps_its_vdw_axis():
    """No covalent bond crosses graphite's c axis, but 3.35 Å is a lattice, not vacuum."""
    assert sio.auto_periodicity(bulk("C", "diamond", a=3.567))[0] == [True] * 3
    axes, reasons = sio.auto_periodicity(_graphite())
    assert axes == [True, True, True] and "declarado periódico" in reasons[2]


def test_a_molecule_in_a_box_is_finite_whatever_the_file_says():
    box = molecule("C6H6")
    box.center(vacuum=6.0)
    for declared in (False, True):
        box.pbc = declared
        assert sio.auto_periodicity(box)[0] == [False] * 3


def test_no_cell_no_periodic_axis():
    axes, reasons = sio.auto_periodicity(molecule("CH4"))
    assert axes == [False] * 3 and reasons == ["sin celda"] * 3


def test_reading_applies_the_rule_and_says_what_the_file_declared(tmp_path):
    path = tmp_path / "tube.xyz"
    from ase.io import write

    write(path, _ribbon(True), format="extxyz")
    atoms = sio.read_structure(path)
    assert list(atoms.pbc) == [False, False, True]
    assert atoms.info["pbc_file"] == "TTT" and atoms.info["periodicity_mode"] == "auto"
    assert "1D (c)" in atoms.info["periodicity"] and "3D" in atoms.info["periodicity"]


def test_vacuum_gap_is_measured_across_the_boundary():
    ribbon = _ribbon(True)
    ribbon.translate([ribbon.cell[0, 0] / 2, 0, 0])           # the gap now straddles x = 0
    ribbon.wrap()
    gaps = sio.vacuum_gaps(ribbon)
    assert gaps[0] > 10.0 and gaps[2] < 2.0


def test_dimensionality_names():
    assert sio.dimensionality([False] * 3) == "0D (finita)"
    assert sio.dimensionality([False, False, True]) == "1D (c)"
    assert sio.dimensionality([True, True, False]) == "2D (a, b)"
    assert sio.dimensionality([True] * 3) == "3D"


# --- editing it ---------------------------------------------------------------------------

def test_manual_axes_are_kept_and_auto_restores_the_rule():
    ribbon = _ribbon(True)
    assert actions.set_periodicity(ribbon, "auto") == [False, False, True]
    actions.set_periodicity(ribbon, "manual", [True, True, True])
    info = actions.periodicity_info(ribbon)
    assert info["mode"] == "manual" and info["pbc"] == [True] * 3
    assert info["auto"] == [False, False, True] and info["dimensionality"] == "3D"
    assert actions.set_periodicity(ribbon, "auto") == [False, False, True]


def test_a_periodic_axis_needs_a_cell():
    with pytest.raises(ValueError, match="Celda"):
        actions.set_periodicity(molecule("CH4"), "manual", [True, False, False])


# --- checks before a calculation ----------------------------------------------------------

def _levels(atoms, model="xu_chn", kmesh=None, scc=None):
    return {level for level, _ in actions.calculation_checks(
        atoms, actions.load_model(model), kmesh, scc)}, " ".join(
        text for _, text in actions.calculation_checks(atoms, actions.load_model(model),
                                                       kmesh, scc))


def test_atoms_on_top_of_their_images_are_an_error():
    small = bulk("C", "diamond", a=3.567)
    small.set_cell(small.cell * 0.3, scale_atoms=False)
    levels, text = _levels(small, "sp3")
    assert "error" in levels and "encimados" in text


def test_bonds_cut_by_a_non_periodic_axis_are_flagged():
    levels, text = _levels(_ribbon([False, False, False]))
    assert "aviso" in levels and "enlaces cortados" in text and "por c" in text


def test_too_few_k_points_along_a_short_period():
    ribbon = _ribbon()
    levels, text = _levels(ribbon, kmesh=2, scc=False)
    assert "Pocos puntos k en c" in text
    _, fine = _levels(ribbon, kmesh=24, scc=False)
    assert "Pocos puntos k" not in fine


def test_a_mesh_that_does_not_fit_stops_before_running(monkeypatch):
    monkeypatch.setattr(actions, "_available_ram_gb", lambda: 1e-4)
    with pytest.raises(ValueError, match="GB libres"):
        actions.ground_state(_ribbon(), actions.load_model("xu_chn"), kmesh=24, scc=False)


def test_scc_with_little_vacuum_warns():
    ribbon = graphene_nanoribbon(2, 1, type="armchair", saturated=True, vacuum=2.0)
    _, text = _levels(ribbon, kmesh=12, scc=True)
    assert "SCC periódico con solo" in text


def test_a_clean_structure_has_no_errors_or_warnings():
    levels, _ = _levels(_ribbon(), kmesh=12, scc=False)
    assert not {"error", "aviso"} & levels


def test_bands_of_a_boxed_tube_follow_its_one_bonded_axis():
    boxed = _ribbon(True)
    actions.set_periodicity(boxed, "manual", [True, True, True])
    curves = actions.band_curves(boxed, actions.load_model("pi"), npoints=20, kmesh=12)
    reference = actions.band_curves(_ribbon(), actions.load_model("pi"), npoints=20, kmesh=12)
    assert curves["labels"] == reference["labels"]
    assert np.allclose(curves["energies"], reference["energies"], atol=1e-8)


def test_bands_refuse_a_box_no_bond_crosses():
    box = molecule("C6H6")
    box.center(vacuum=6.0)
    box.pbc = True
    with pytest.raises(ValueError, match="planas"):
        actions.band_curves(box, actions.load_model("pi"))


def test_a_model_that_lacks_an_element_is_an_error_not_a_crash():
    levels, text = _levels(_ribbon(), model="pi", kmesh=12)
    assert "error" in levels and "H" in text
