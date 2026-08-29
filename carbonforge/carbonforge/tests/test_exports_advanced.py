"""Tests for SIESTA export and the extended Quantum ESPRESSO workflows."""

from __future__ import annotations

import pytest

from carbonforge.builders import build_cnt, build_graphene, build_graphene_supercell
from carbonforge.calculations import phonon_setup, raman_setup, soc_setup
from carbonforge.exports.qe import (
    QESettings,
    write_qe_bands,
    write_qe_input,
    write_qe_spectroscopy,
)
from carbonforge.exports.siesta import SiestaSettings, write_siesta


class TestQEBands:
    def test_writes_all_workflow_steps(self, tmp_path):
        written = write_qe_bands(build_cnt(10, 0, length=6), tmp_path)
        assert set(written) == {"scf", "bands", "bandsx", "script"}
        for path in written.values():
            assert path.exists() and path.stat().st_size > 0

    def test_bands_step_uses_crystal_b(self, tmp_path):
        written = write_qe_bands(build_cnt(10, 0, length=6), tmp_path)
        assert "K_POINTS crystal_b" in written["bands"].read_text()

    def test_scf_step_uses_uniform_mesh(self, tmp_path):
        """The scf step must sample a mesh, not the path."""
        written = write_qe_bands(build_cnt(10, 0, length=6), tmp_path)
        assert "K_POINTS automatic" in written["scf"].read_text()

    def test_bands_step_requests_empty_states(self, tmp_path):
        written = write_qe_bands(build_cnt(10, 0, length=6), tmp_path)
        assert "nbnd" in written["bands"].read_text()

    def test_script_is_executable(self, tmp_path):
        written = write_qe_bands(build_graphene(), tmp_path)
        assert written["script"].stat().st_mode & 0o111


class TestQESpectroscopy:
    def test_writes_all_workflow_steps(self, tmp_path):
        written = write_qe_spectroscopy(
            build_cnt(10, 0, length=6), tmp_path, raman_setup()
        )
        assert set(written) == {"scf", "ph", "dynmat", "script"}
        for path in written.values():
            assert path.exists()

    def test_raman_switches_present(self, tmp_path):
        written = write_qe_spectroscopy(
            build_cnt(10, 0, length=6), tmp_path, raman_setup()
        )
        text = written["ph"].read_text()
        assert "lraman = .true." in text and "epsil = .true." in text

    def test_phonon_only_omits_intensity_switches(self, tmp_path):
        written = write_qe_spectroscopy(
            build_cnt(6, 6, length=6), tmp_path, phonon_setup()
        )
        text = written["ph"].read_text()
        assert "lraman" not in text and "epsil" not in text


class TestQESpinOrbit:
    def test_soc_fields_written(self, tmp_path):
        path = write_qe_input(
            build_graphene_supercell(2, 2), tmp_path,
            settings=QESettings(spinorbit=soc_setup()),
        )
        text = path.read_text()
        assert "noncolin = .true." in text
        assert "lspinorb = .true." in text

    def test_soc_swaps_in_relativistic_pseudos(self, tmp_path):
        path = write_qe_input(
            build_graphene_supercell(2, 2), tmp_path,
            settings=QESettings(spinorbit=soc_setup()),
        )
        assert "rel-" in path.read_text()

    def test_no_soc_keeps_scalar_pseudos(self, tmp_path):
        path = write_qe_input(build_graphene_supercell(2, 2), tmp_path)
        assert "rel-" not in path.read_text()


class TestQECellDofree:
    def test_cell_dofree_written_for_vc_relax(self, tmp_path):
        path = write_qe_input(
            build_graphene_supercell(2, 2), tmp_path,
            settings=QESettings(calculation="vc-relax", cell_dofree="2Dxy"),
        )
        text = path.read_text()
        assert "&CELL" in text and "cell_dofree = '2Dxy'" in text


class TestSiesta:
    def test_writes_required_blocks(self, tmp_path):
        path = write_siesta(build_graphene(), tmp_path)
        text = path.read_text()
        for block in (
            "ChemicalSpeciesLabel",
            "LatticeVectors",
            "AtomicCoordinatesAndAtomicSpecies",
            "kgrid_Monkhorst_Pack",
        ):
            assert block in text

    def test_atom_and_species_counts(self, tmp_path):
        atoms = build_graphene_supercell(2, 2)
        text = write_siesta(atoms, tmp_path).read_text()
        assert f"NumberOfAtoms   {len(atoms)}" in text
        assert "NumberOfSpecies 1" in text

    def test_kgrid_is_one_along_vacuum(self, tmp_path):
        """A 2D sheet must not be sampled along the vacuum direction."""
        text = write_siesta(build_graphene(), tmp_path).read_text()
        kblock = text.split("%block kgrid_Monkhorst_Pack")[1]
        third_row = kblock.splitlines()[3]
        assert third_row.split()[2] == "1"

    def test_bands_block(self, tmp_path):
        text = write_siesta(
            build_graphene(), tmp_path, settings=SiestaSettings(run_type="bands")
        ).read_text()
        assert "%block BandLines" in text
        assert "BandLinesScale" in text

    def test_spin_orbit_block(self, tmp_path):
        text = write_siesta(
            build_graphene(), tmp_path,
            settings=SiestaSettings(spinorbit=soc_setup()),
        ).read_text()
        assert "Spin               SO" in text

    def test_phonon_uses_force_constants(self, tmp_path):
        text = write_siesta(
            build_graphene(), tmp_path, settings=SiestaSettings(run_type="phonon")
        ).read_text()
        assert "MD.TypeOfRun       FC" in text

    def test_raman_request_is_flagged_as_unsupported(self, tmp_path):
        """SIESTA cannot do Raman; the file must say so rather than pretend."""
        text = write_siesta(
            build_graphene(), tmp_path,
            settings=SiestaSettings(run_type="phonon"),
            spectroscopy=raman_setup(),
        ).read_text()
        assert "no Raman implementation" in text

    def test_vc_relax_warns_about_vacuum(self, tmp_path):
        text = write_siesta(
            build_graphene(), tmp_path, settings=SiestaSettings(run_type="vc-relax")
        ).read_text()
        assert "compress the vacuum" in text

    def test_validation_blocks_export(self, tmp_path):
        import numpy as np

        atoms = build_cnt(6, 6, length=6)
        pos = atoms.get_positions()
        pos[1] = pos[0] + np.array([0.2, 0.0, 0.0])
        atoms.set_positions(pos)
        with pytest.raises(ValueError):
            write_siesta(atoms, tmp_path)
        assert write_siesta(atoms, tmp_path, force=True).exists()
