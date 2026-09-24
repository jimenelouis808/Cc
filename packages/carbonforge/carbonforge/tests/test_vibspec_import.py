"""Tests for loading your own geometry into vibspec (files from other programs)."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from carbonforge.builders import build_finite_nanoribbon, build_nanoribbon
from carbonforge.cli.main import main as cli_main
from carbonforge.vibspec.core import (
    ImportRefused,
    apply_preset,
    check_structure,
    list_library,
    load_structure,
    vacuum_per_side,
)
from carbonforge.vibspec.gui import logic


def _plain_xyz(atoms: Atoms, path):
    """Write coordinates only, no cell and no metadata, as Avogadro would."""
    bare = Atoms(atoms.get_chemical_symbols(), positions=atoms.get_positions())
    write(path, bare, format="xyz")
    return path


@pytest.fixture(scope="module")
def amine_flake():
    return apply_preset(build_finite_nanoribbon(5, 3, edge="armchair"), "amine")


class TestLoadStructure:
    def test_plain_xyz_gets_a_box(self, tmp_path, amine_flake):
        atoms, report = load_structure(_plain_xyz(amine_flake, tmp_path / "mia.xyz"))
        assert not any(atoms.get_pbc())
        assert atoms.get_chemical_formula() == amine_flake.get_chemical_formula()
        for gap in vacuum_per_side(atoms).values():
            assert gap == pytest.approx(7.0)
        # A 5x3 armchair flake is nearly square: no edge type dominates once
        # the corners are left out, and the report says to choose.
        assert "edge" not in atoms.info and "Ningún tipo domina" in report
        assert atoms.info["source_file"].endswith("mia.xyz")
        assert "Modelo:" in report
        assert "no traía celda" in report and "impiden exportar" not in report
        assert check_structure(atoms).ok

    @pytest.mark.parametrize("edge,width,length", [("armchair", 5, 6), ("zigzag", 4, 8)])
    def test_edge_inferred_on_elongated_ribbons(self, tmp_path, edge, width, length):
        flake = build_finite_nanoribbon(width, length, edge=edge)
        atoms, report = load_structure(_plain_xyz(flake, tmp_path / "r.xyz"))
        assert atoms.info["edge"] == edge
        assert f"se toma {edge}" in report

    def test_nitrogen_is_reported(self, tmp_path, amine_flake):
        _, report = load_structure(_plain_xyz(amine_flake, tmp_path / "n.xyz"))
        assert "Nitrógeno" in report

    def test_vacuum(self, tmp_path, amine_flake):
        path = _plain_xyz(amine_flake, tmp_path / "v.xyz")
        atoms, _ = load_structure(path, vacuum_per_side=8.0)
        assert min(vacuum_per_side(atoms).values()) == pytest.approx(8.0)
        with pytest.raises(ImportRefused, match="mínimo"):
            load_structure(path, vacuum_per_side=4.0)

    def test_truly_periodic_is_refused(self, tmp_path):
        path = tmp_path / "periodica.xyz"
        build_nanoribbon(4, 2, passivate=True).write(path)
        with pytest.raises(ImportRefused, match="periódica de verdad"):
            load_structure(path)

    def test_molecule_in_a_periodic_box_is_accepted(self, tmp_path, amine_flake):
        boxed = amine_flake.copy()
        boxed.pbc = True
        path = tmp_path / "caja.xyz"
        boxed.write(path)
        atoms, report = load_structure(path)
        assert not any(atoms.get_pbc())
        assert "molécula en una caja" in report

    def test_overlapping_atoms_are_refused_not_moved(self, tmp_path, amine_flake):
        broken = amine_flake.copy()
        broken += Atoms("C", positions=[broken.positions[0] + 0.1])
        with pytest.raises(ImportRefused, match="superpuestos"):
            load_structure(_plain_xyz(broken, tmp_path / "mal.xyz"))

    def test_unreadable(self, tmp_path):
        path = tmp_path / "nada.xyz"
        path.write_text("esto no es\nun xyz\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_structure(path)

    def test_last_frame_of_a_trajectory(self, tmp_path, amine_flake):
        moved = amine_flake.copy()
        moved.positions += 0.0
        moved.positions[0] += [0.05, 0.0, 0.0]
        path = tmp_path / "relax.traj"
        write(path, [amine_flake, moved])
        atoms, _ = load_structure(path)
        first, _ = load_structure(path, index=0)
        assert not np.allclose(atoms.positions, first.positions)

    def test_vibspec_files_keep_their_provenance(self, tmp_path, amine_flake):
        path = tmp_path / "propio.xyz"
        amine_flake.write(path)
        atoms, _ = load_structure(path)
        assert atoms.info["vibspec_preset"]["key"] == "amine"
        assert atoms.info["edge"] == "armchair"

    def test_preset_on_top_of_a_loaded_structure(self, tmp_path):
        pristine = build_finite_nanoribbon(5, 3, edge="armchair")
        atoms, _ = load_structure(_plain_xyz(pristine, tmp_path / "base.xyz"))
        loaded = apply_preset(atoms, "amine")
        built = apply_preset(pristine, "amine")
        assert loaded.get_chemical_formula() == built.get_chemical_formula()
        assert check_structure(loaded).ok

    def test_library(self, tmp_path, amine_flake):
        (tmp_path / "notas.txt").write_text("no es una estructura", encoding="utf-8")
        amine_flake.write(tmp_path / "b.xyz")
        amine_flake.write(tmp_path / "a.cif")
        assert [p.name for p in list_library(tmp_path)] == ["a.cif", "b.xyz"]
        assert list_library(tmp_path / "no_existe") == []


class TestCli:
    def test_import(self, tmp_path, amine_flake, capsys):
        source = _plain_xyz(amine_flake, tmp_path / "mia.xyz")
        out = tmp_path / "modelo.xyz"
        assert cli_main(["vibspec", "import", str(source), "-o", str(out)]) == 0
        assert out.exists() and "Modelo:" in capsys.readouterr().out

    def test_import_with_a_preset_on_top(self, tmp_path):
        source = _plain_xyz(build_finite_nanoribbon(5, 3), tmp_path / "base.xyz")
        out = tmp_path / "con_nitrilo.xyz"
        assert cli_main(["vibspec", "import", str(source), "--preset", "nitrile",
                         "-o", str(out)]) == 0
        assert "N" in out.read_text(encoding="utf-8")

    def test_import_refuses_periodic(self, tmp_path):
        path = tmp_path / "p.xyz"
        build_nanoribbon(4, 2, passivate=True).write(path)
        assert cli_main(["vibspec", "import", str(path), "-o", str(tmp_path / "o.xyz")]) == 1

    def test_prepare_accepts_a_plain_xyz(self, tmp_path, amine_flake):
        source = _plain_xyz(amine_flake, tmp_path / "mia.xyz")
        assert cli_main(["vibspec", "prepare", str(source), "-d", str(tmp_path / "calc")]) == 0
        assert (tmp_path / "calc" / "record.json").exists()


class TestGuiLogic:
    def test_model_from_file_keeps_its_atoms(self, tmp_path, amine_flake):
        source = _plain_xyz(amine_flake, tmp_path / "mi_cinta.xyz")
        raw = logic.defaults(logic.BUILDER_PARAMS)
        raw.update(width="9", length="9")                 # ignored with a file
        result = logic.build_model(raw, source=source)
        assert result.atoms.get_chemical_formula() == amine_flake.get_chemical_formula()
        assert "Archivo cargado" in result.summary()
        assert logic.job_name(result.atoms) == "mi_cinta"

    def test_preset_on_a_file_names_the_job(self, tmp_path):
        source = _plain_xyz(build_finite_nanoribbon(5, 3), tmp_path / "base.xyz")
        raw = logic.defaults(logic.BUILDER_PARAMS)
        raw["preset"] = "hydroxyl"
        result = logic.build_model(raw, source=source)
        assert "O" in result.atoms.get_chemical_symbols()
        assert logic.job_name(result.atoms) == "base_hydroxyl"

    def test_save_to_library_and_reload(self, tmp_path, amine_flake):
        path = logic.save_to_library(amine_flake, tmp_path / "biblio", "mi amina")
        assert path.name == "mi_amina.xyz"
        assert logic.library(tmp_path / "biblio") == [path]
        with pytest.raises(FileExistsError):
            logic.save_to_library(amine_flake, tmp_path / "biblio", "mi amina")
        result = logic.build_model(logic.defaults(logic.BUILDER_PARAMS), source=path)
        assert result.atoms.info["vibspec_preset"]["key"] == "amine"
