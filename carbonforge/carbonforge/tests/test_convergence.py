"""Tests for the convergence sweeps and their report.

As with the other output parsers, the pw.x output fixtures are synthetic and
match the documented format; they are not captured from a real run.
"""

from __future__ import annotations

import pytest

from carbonforge.builders import build_cnt, build_graphene_supercell
from carbonforge.exports.qe import QESettings
from carbonforge.workflows.convergence import (
    ConvergencePoint,
    RY_TO_EV,
    convergence_table,
    cutoff_sweep,
    kpoint_sweep,
    read_total_energies,
    read_total_energy,
)


def _pw_output(energy_ry: float, n_atoms: int = 24) -> str:
    return f"""
     Program PWSCF v.7.2 starts

     number of atoms/cell      =           {n_atoms}
     number of Kohn-Sham states=           60

     iteration #  1
     total energy              =    -100.00000000 Ry

!    total energy              =    {energy_ry:.8f} Ry
     estimated scf accuracy    <       1.0E-09 Ry

     JOB DONE.
"""


class TestCutoffSweep:
    def test_writes_one_input_per_cutoff(self, tmp_path):
        written = cutoff_sweep(
            build_cnt(6, 6, length=6), tmp_path, cutoffs=(40, 60, 80)
        )
        assert set(written) == {"ecut_40", "ecut_60", "ecut_80", "script"}
        for path in written.values():
            assert path.exists()

    def test_cutoff_and_dual_applied(self, tmp_path):
        written = cutoff_sweep(
            build_cnt(6, 6, length=6), tmp_path, cutoffs=(70,), dual=10.0
        )
        text = written["ecut_70"].read_text()
        assert "ecutwfc = 70" in text
        assert "ecutrho = 700" in text

    def test_script_is_executable(self, tmp_path):
        written = cutoff_sweep(build_cnt(6, 6, length=6), tmp_path, cutoffs=(40,))
        assert written["script"].stat().st_mode & 0o111

    def test_base_settings_preserved(self, tmp_path):
        written = cutoff_sweep(
            build_cnt(6, 6, length=6), tmp_path, cutoffs=(50,),
            settings=QESettings(prefix="mytube", conv_thr=1e-10),
        )
        text = written["ecut_50"].read_text()
        assert "prefix = 'mytube'" in text

    def test_empty_cutoffs_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            cutoff_sweep(build_cnt(6, 6, length=6), tmp_path, cutoffs=())

    def test_unphysical_dual_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="dual"):
            cutoff_sweep(
                build_cnt(6, 6, length=6), tmp_path, cutoffs=(60,), dual=2.0
            )


class TestKpointSweep:
    def test_writes_one_input_per_density(self, tmp_path):
        written = kpoint_sweep(
            build_graphene_supercell(2, 2), tmp_path, densities=(0.3, 0.2)
        )
        assert "script" in written
        assert len([k for k in written if k != "script"]) == 2

    def test_denser_mesh_has_more_points(self, tmp_path):
        written = kpoint_sweep(
            build_graphene_supercell(2, 2), tmp_path, densities=(0.4, 0.1)
        )
        def mesh_of(stem):
            line = [
                ln for ln in written[stem].read_text().splitlines()
                if ln.strip().endswith("0 0 0")
            ][0]
            return int(line.split()[0])

        coarse = mesh_of("kdens_0p4")
        fine = mesh_of("kdens_0p1")
        assert fine > coarse

    def test_empty_densities_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            kpoint_sweep(build_graphene_supercell(2, 2), tmp_path, densities=())


class TestReadEnergies:
    def test_reads_final_energy_and_atoms(self, tmp_path):
        path = tmp_path / "ecut_60.out"
        path.write_text(_pw_output(-123.456789, n_atoms=24))
        energy, n_atoms = read_total_energy(path)
        assert energy == pytest.approx(-123.456789 * RY_TO_EV)
        assert n_atoms == 24

    def test_takes_the_last_converged_value(self, tmp_path):
        """Intermediate 'total energy' lines must not win over the final one."""
        path = tmp_path / "ecut_60.out"
        path.write_text(_pw_output(-200.0))
        energy, _ = read_total_energy(path)
        assert energy == pytest.approx(-200.0 * RY_TO_EV)

    def test_unfinished_run_raises(self, tmp_path):
        path = tmp_path / "ecut_60.out"
        path.write_text("     Program PWSCF starts\n     iteration # 1\n")
        with pytest.raises(ValueError, match="no terminó"):
            read_total_energy(path)

    def test_collects_and_sorts_sweep(self, tmp_path):
        for cutoff, energy in [(40, -122.0), (80, -123.5), (60, -123.4)]:
            (tmp_path / f"ecut_{cutoff}.out").write_text(_pw_output(energy))
        points = read_total_energies(tmp_path)
        assert [p.value for p in points] == [40.0, 60.0, 80.0]
        assert all(p.energy_per_atom_ev is not None for p in points)

    def test_decodes_kdens_filenames(self, tmp_path):
        (tmp_path / "kdens_0p20.out").write_text(_pw_output(-123.0))
        points = read_total_energies(tmp_path)
        assert points[0].value == pytest.approx(0.20)

    def test_crashed_run_is_skipped_not_fatal(self, tmp_path):
        """One failed point must not hide the rest of the sweep."""
        (tmp_path / "ecut_40.out").write_text(_pw_output(-122.0))
        (tmp_path / "ecut_60.out").write_text("crashed, no energy here\n")
        points = read_total_energies(tmp_path)
        assert [p.value for p in points] == [40.0]


class TestConvergenceTable:
    def _points(self, energies_per_atom):
        return [
            ConvergencePoint(label=f"p{i}", value=float(40 + 20 * i),
                             energy_ev=e * 24, energy_per_atom_ev=e)
            for i, e in enumerate(energies_per_atom)
        ]

    def test_identifies_converged_value(self):
        # Steps: 10 meV, 0.5 meV -> converged at the second point (60).
        points = self._points([-5.000, -5.010, -5.0105])
        table = convergence_table(points, tolerance_mev_per_atom=1.0,
                                  parameter_name="ecutwfc")
        assert "Convergido en ecutwfc = 60" in table

    def test_reports_failure_to_converge(self):
        points = self._points([-5.00, -5.10, -5.20])
        table = convergence_table(points, tolerance_mev_per_atom=1.0)
        assert "NO converge" in table

    def test_needs_two_points(self):
        assert "al menos 2" in convergence_table(self._points([-5.0]))

    def test_empty_input(self):
        assert "No se encontraron" in convergence_table([])

    def test_table_lists_every_point(self):
        table = convergence_table(self._points([-5.000, -5.010, -5.0105]))
        for value in ("40", "60", "80"):
            assert value in table
