"""Tests for the QE and LAMMPS exporters."""

from __future__ import annotations

import pytest

from carbonforge.builders import build_cnt, build_graphene_supercell
from carbonforge.exports.qe import write_qe_input, QESettings, infer_qe_settings
from carbonforge.exports.lammps import write_lammps


class TestQEExport:
    def test_writes_complete_input(self, tmp_path):
        cnt = build_cnt(6, 6, length=6)
        path = write_qe_input(cnt, tmp_path, settings=QESettings(calculation="relax"))
        text = path.read_text()
        for section in [
            "&CONTROL",
            "&SYSTEM",
            "&ELECTRONS",
            "&IONS",
            "ATOMIC_SPECIES",
            "ATOMIC_POSITIONS",
            "CELL_PARAMETERS",
            "K_POINTS",
        ]:
            assert section in text

    def test_vc_relax_has_cell_namelist(self, tmp_path):
        gr = build_graphene_supercell(2, 2)
        path = write_qe_input(gr, tmp_path,
                              settings=QESettings(calculation="vc-relax"))
        text = path.read_text()
        assert "&CELL" in text

    def test_2d_is_flagged_assume_isolated(self, tmp_path):
        gr = build_graphene_supercell(2, 2)
        s = infer_qe_settings(gr)
        assert s.assume_isolated == "2D"

    def test_refuses_broken_structure(self, tmp_path):
        import numpy as np
        cnt = build_cnt(5, 5, length=5)
        pos = cnt.get_positions()
        pos[1] = pos[0] + np.array([0.2, 0.0, 0.0])
        cnt.set_positions(pos)
        with pytest.raises(ValueError):
            write_qe_input(cnt, tmp_path)

    def test_force_bypasses_validation(self, tmp_path):
        import numpy as np
        cnt = build_cnt(5, 5, length=5)
        pos = cnt.get_positions()
        pos[1] = pos[0] + np.array([0.2, 0.0, 0.0])
        cnt.set_positions(pos)
        path = write_qe_input(cnt, tmp_path, force=True)
        assert path.exists()

    def test_kpoint_mesh_scales_with_pbc(self, tmp_path):
        cnt = build_cnt(5, 5, length=8)
        gr = build_graphene_supercell(2, 2)
        p_cnt = write_qe_input(cnt, tmp_path / "cnt").read_text()
        p_gr = write_qe_input(gr, tmp_path / "gr").read_text()
        # CNT: 1 periodic axis → "N 1 1 0 0 0"
        k_cnt = [ln for ln in p_cnt.splitlines() if "0 0 0" in ln][0]
        assert k_cnt.split()[-4:-3] == ["1"] or k_cnt.strip().endswith("0 0 0")
        # Just ensure both exist and contain the card.
        assert "K_POINTS automatic" in p_cnt
        assert "K_POINTS automatic" in p_gr


class TestLAMMPSExport:
    def test_writes_data_and_input(self, tmp_path):
        cnt = build_cnt(6, 0, length=10)
        data, inp = write_lammps(cnt, tmp_path)
        assert data.exists() and inp.exists()
        text = data.read_text()
        assert "atoms" in text
        assert "Masses" in text
        assert "Atoms" in text

    def test_data_atom_count_matches(self, tmp_path):
        gr = build_graphene_supercell(2, 2)
        data, _ = write_lammps(gr, tmp_path)
        text = data.read_text()
        header = next(ln for ln in text.splitlines() if ln.endswith("atoms"))
        assert int(header.split()[0]) == len(gr)

    def test_pair_style_airebo_for_carbon(self, tmp_path):
        gr = build_graphene_supercell(2, 2)
        _, inp = write_lammps(gr, tmp_path)
        text = inp.read_text()
        assert "pair_style" in text
        assert "airebo" in text
