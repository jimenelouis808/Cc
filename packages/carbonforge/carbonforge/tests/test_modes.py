"""Normal modes from dynmat.axsf, shared helpers, and the Resultados page.

The .axsf fixtures are synthetic, written in the documented XCrySDen format
that dynmat.x uses for ``filout``.
"""

from __future__ import annotations

import numpy as np
import pytest

from carbonforge.results.modes import (
    mode_character,
    mode_frames,
    nearest_mode,
    normalised,
    read_axsf_modes,
)

# CO2-like: symmetric stretch, antisymmetric stretch.
AXSF = """ANIMSTEPS 2
CRYSTAL
PRIMVEC
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
PRIMCOORD 1
3 1
  8  3.84 5.0 5.0  -0.5 0.0 0.0
  6  5.00 5.0 5.0   0.0 0.0 0.0
  8  6.16 5.0 5.0   0.5 0.0 0.0
PRIMCOORD 2
3 1
  O  3.84 5.0 5.0   0.2 0.0 0.0
  C  5.00 5.0 5.0  -0.6 0.0 0.0
  O  6.16 5.0 5.0   0.2 0.0 0.0
"""

MOLECULE = """ANIMSTEPS 1
ATOMS 1
H 0.0 0.0 0.0 0.0 0.0 1.0
H 0.0 0.0 0.74 0.0 0.0 -1.0
"""


def test_reads_crystal_axsf(tmp_path):
    path = tmp_path / "dynmat.axsf"
    path.write_text(AXSF)
    atoms, vectors = read_axsf_modes(path)
    assert atoms.get_chemical_symbols() == ["O", "C", "O"]
    assert atoms.pbc.all() and atoms.cell[0, 0] == pytest.approx(10.0)
    assert len(vectors) == 2 and vectors[1][1, 0] == pytest.approx(-0.6)
    # Symmetric stretch: the carbon does not move.
    assert mode_character(atoms, vectors[0]).startswith("O 100%")


def test_reads_molecule_axsf(tmp_path):
    path = tmp_path / "m.axsf"
    path.write_text(MOLECULE)
    atoms, vectors = read_axsf_modes(path)
    assert len(atoms) == 2 and not atoms.pbc.any() and len(vectors) == 1


def test_rejects_empty(tmp_path):
    path = tmp_path / "x.axsf"
    path.write_text("ANIMSTEPS 0\n")
    with pytest.raises(ValueError, match="PRIMCOORD"):
        read_axsf_modes(path)


def test_helpers():
    vector = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    assert np.linalg.norm(normalised(vector), axis=1).max() == pytest.approx(1.0)
    frequencies = [100.0, 1000.0, 1010.0, 1600.0]
    assert nearest_mode(frequencies, [1, 0.1, 5.0, 1], 1003.0) == 2     # most active nearby
    assert nearest_mode(frequencies, None, 1003.0) == 1                  # nearest
    assert nearest_mode(frequencies, [1, 1, 1, 1], 400.0) == 0           # none within window
    assert nearest_mode([], None, 10.0) is None


def test_frames_oscillate():
    from ase import Atoms

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    frames = mode_frames(atoms, np.array([[0, 0, 1.0], [0, 0, -1.0]]), n_frames=4)
    assert frames[1][0, 2] == pytest.approx(0.35) and frames[3][0, 2] == pytest.approx(-0.35)


def test_vibspec_uses_the_same_helpers():
    from carbonforge.results import modes
    from carbonforge.vibspec.gui import logic

    assert logic.mode_character is modes.mode_character
    assert logic.view_angles is modes.view_angles


class TestRecipeManifest:
    def test_relax_then_update_then_property(self, tmp_path):
        from carbonforge.builders import build_graphene_supercell
        from carbonforge.jobs import JobManifest
        from carbonforge.workflows.pipeline import write_preset_project

        _, plan, written = write_preset_project(build_graphene_supercell(3, 3), tmp_path,
                                                "bands", n_cores=8)
        manifest = JobManifest.load(tmp_path)
        names = [step.name for step in manifest.steps]
        assert "job" in written
        if (tmp_path / "pw.relax.in").exists():
            assert names[:2] == ["relax", "geometría"]
        assert names[-3:] == ["scf", "bands", "bands.x"]
        pw_steps = [s for s in manifest.steps if s.program == "pw.x"]
        assert all(s.pools == plan.pools for s in pw_steps)

    def test_commands_for_pools_and_carbonforge_steps(self):
        import sys

        from carbonforge.jobs.manifest import Step
        from carbonforge.jobs.run import step_command

        update = Step("geometría", "carbonforge", "", "u.out", parallel=False,
                      args=["update-geometry", "pw.relax.out", "--apply-to", "."])
        assert step_command(update, 8)[:3] == [sys.executable, "-m", "carbonforge.cli.main"]

    def test_pools_only_when_they_divide(self, monkeypatch):
        from carbonforge.jobs import run as runner
        from carbonforge.jobs.manifest import Step

        monkeypatch.setattr(runner, "find_program", lambda name: "/bin/" + name)
        monkeypatch.setattr(runner, "mpi_prefix", lambda n: ["mpiexec", "-n", str(n)] if n > 1
                            else [])
        step = Step("scf", "pw.x", "pw.scf.in", "pw.scf.out", pools=4)
        assert "-nk" in runner.step_command(step, 8)
        assert "-nk" not in runner.step_command(step, 6)
        assert runner.step_command(step, 1) == ["/bin/pw.x", "-in", "pw.scf.in"]


class TestResultsPage:
    """DOS, PDOS and QE modes on the Resultados page (fake Tk, real matplotlib)."""

    @pytest.fixture()
    def app(self, monkeypatch):
        from carbonforge.tests.test_gui_widgets import _install_fake_tk, _Root

        _install_fake_tk(monkeypatch)
        from carbonforge.gui.app import CarbonForgeApp

        return CarbonForgeApp(_Root())

    def test_dos_and_pdos(self, app, tmp_path):
        from carbonforge.tests.test_dos import _gaussian

        energies = np.linspace(-10, 10, 201)
        dos = _gaussian(energies, -3, 0.5, 2) + _gaussian(energies, 3, 0.5, 2)
        path = tmp_path / "dos.dat"
        path.write_text("#  E (eV)   dos(E)   Int dos(E)  EFermi =    0.000 eV\n"
                        + "\n".join(f"{e:.4f} {d:.6f} 0.0" for e, d in zip(energies, dos)))
        app._on_open_dos(path)
        assert "Gap estimado" in app.analysis_text.text_content
        pdos = tmp_path / "pdos"
        pdos.mkdir()
        for name, centre in (("pdos.pdos_atm#1(C)_wfc#2(p)", -2.0),
                             ("pdos.pdos_atm#2(N)_wfc#2(p)", 0.0)):
            (pdos / name).write_text("\n".join(
                f"{e:.4f} {c:.6f} {c:.6f}" for e, c in
                zip(energies, _gaussian(energies, centre, 1.0, 1.0))))
        app._on_open_pdos(str(pdos))
        assert "elemento" in app.analysis_text.text_content

    def test_spectrum_modes_are_clickable(self, app, tmp_path):
        from types import SimpleNamespace

        from carbonforge.tests.test_results import DYNMAT_FULL

        out = tmp_path / "dynmat.out"
        out.write_text(DYNMAT_FULL)
        from carbonforge.results.spectra import read_dynmat

        n_modes = len(read_dynmat(out).modes)
        block = "\n".join(f"PRIMCOORD {k + 1}\n2 1\nC 0 0 0 0 0 {k + 1}.0\nC 0 0 1.4 0 0 -1.0"
                          for k in range(n_modes))
        (tmp_path / "dynmat.axsf").write_text(f"ANIMSTEPS {n_modes}\n{block}\n")
        app._on_open_spectrum(out)
        assert app._qe_modes is not None
        shown = []
        app._show_qe_mode = shown.append
        event = SimpleNamespace(xdata=float(read_dynmat(out).modes[-1].frequency_cm1),
                                inaxes=app.analysis_axes)
        app._on_analysis_click(event)
        assert shown and 0 <= shown[0] < n_modes

    def test_mismatched_modes_are_not_animated(self, app, tmp_path):
        from carbonforge.tests.test_results import DYNMAT_FULL

        out = tmp_path / "dynmat.out"
        out.write_text(DYNMAT_FULL)
        (tmp_path / "dynmat.axsf").write_text(MOLECULE)
        app._on_open_spectrum(out)
        assert app._qe_modes is None and "mismo cálculo" in app.analysis_status_var.get()
