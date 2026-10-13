"""tbkit.structure_io: .xtl files, and periodicity found from bonds across the cell."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import graphene, molecule

from tbkit import structure_io as sio

GRAPHENE_XTL = """TITLE grafeno
DIMENSION 3
CELL
  2.46 2.46 15.0 90.0 90.0 120.0
SYMMETRY NUMBER 1
SYMMETRY LABEL P1
ATOMS
NAME X Y Z
C1 0.0 0.0 0.5
C2 0.333333 0.666667 0.5
EOF
"""


def test_xtl_p1_and_space_group(tmp_path):
    path = tmp_path / "g.xtl"
    path.write_text(GRAPHENE_XTL)
    atoms = sio.read_structure(path)
    # An .xtl is periodic in all three axes by format; c is 15 Å of vacuum, so the
    # automatic periodicity makes the sheet 2D and keeps what the file declared.
    assert atoms.get_chemical_symbols() == ["C", "C"] and list(atoms.pbc) == [True, True, False]
    assert atoms.info["pbc_file"] == "TTT" and "2D (a, b)" in atoms.info["periodicity"]
    d = atoms.get_distance(0, 1, mic=True)
    assert d == pytest.approx(2.46 / np.sqrt(3), abs=1e-3)       # the C-C bond
    # Diamond from its space group (227, origin choice 1): 8 atoms in the cubic cell.
    path.write_text("CELL\n 3.567 3.567 3.567 90 90 90\nSYMMETRY NUMBER 227\nATOMS\n"
                    "NAME X Y Z\nC1 0 0 0\nEOF\n")
    diamond = sio.read_structure(path)
    assert len(diamond) == 8
    assert min(diamond.get_distance(0, j, mic=True) for j in range(1, 8)) == \
        pytest.approx(3.567 * np.sqrt(3) / 4, abs=1e-3)


def test_labels_to_elements():
    assert [sio._element(x) for x in ("C1", "Ca2", "N_a", "Cl", "O12", "H")] == \
        ["C", "Ca", "N", "Cl", "O", "H"]
    with pytest.raises(ValueError):
        sio._element("123")


def test_periodicity_from_bonds_across_the_cell(tmp_path):
    sheet = graphene(vacuum=6.0)
    sheet.pbc = False                       # what an xyz with pbc="F F F" gives
    path = tmp_path / "g.xyz"
    sheet.write(path, format="extxyz")
    atoms = sio.read_structure(path)
    assert list(atoms.pbc) == [True, True, False]
    assert "enlaces que cruzan" in atoms.info["periodicity"]
    box = molecule("C6H6", vacuum=5.0)      # a molecule in a box stays a molecule
    box.pbc = False
    box.write(tmp_path / "b.xyz", format="extxyz")
    assert not sio.read_structure(tmp_path / "b.xyz").pbc.any()
    assert not sio.read(path).pbc.any()     # the command line reads the file as it is


def test_with_cell():
    atoms = molecule("H2")
    out = sio.with_cell(atoms, [2.46, 2.46, 15, 90, 90, 120], [True, True, False])
    assert out.cell.lengths() == pytest.approx([2.46, 2.46, 15])
    assert np.allclose(out.positions, atoms.positions)
    with pytest.raises(ValueError):
        sio.with_cell(atoms, [[1, 0, 0], [0, 1, 0], [0, 0, 0]], [True, True, True])


def test_written_files_read_back_with_cell_and_axes(tmp_path):
    sheet = graphene(vacuum=6.0)
    sheet.pbc = [True, True, False]
    for name in ("g.xtl", "g.xyz", "POSCAR"):
        back = sio.read_structure(sio.write_structure(sheet, tmp_path / name),
                                  guess_periodic=False)
        assert back.cell.cellpar() == pytest.approx(sheet.cell.cellpar(), abs=1e-5)
        assert np.allclose(back.get_scaled_positions() % 1, sheet.get_scaled_positions() % 1,
                           atol=1e-6)
        if name == "g.xyz":
            assert list(back.pbc) == [True, True, False]
