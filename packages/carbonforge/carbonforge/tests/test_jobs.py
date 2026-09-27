"""The shared job queue and the step runner, with stand-in programs.

Real pw.x / siesta / lmp are not needed: tiny scripts named like them record
their arguments and write an output, which is all the runner can see anyway.
POSIX only (the stand-ins are shell scripts); Windows users prepare there and
run on Linux.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from carbonforge.builders import build_graphene_supercell
from carbonforge.gui.params import CALCULATION_PARAMS, export_structure
from carbonforge.jobs import (
    CANCELLED,
    DONE,
    ERROR,
    JobManifest,
    JobQueue,
    adapter_for,
    engine_of,
    manifest_for_directory,
    write_manifest,
)
from carbonforge.jobs.run import find_program, run, step_command

posix_only = pytest.mark.skipif(os.name != "posix", reason="programas de prueba en sh")


def _form(**overrides):
    values = {spec.key: spec.default for spec in CALCULATION_PARAMS}
    values["preset"] = "ninguna"
    values.update(overrides)
    return values


@pytest.fixture
def fake_bin(tmp_path, monkeypatch):
    """A PATH with stand-ins for pw.x, ph.x, dynmat.x, siesta and lmp."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()

    def make(name: str, body: str = 'echo "$0 $@"; echo JOB DONE') -> Path:
        path = bin_dir / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
        return path

    for name in ("pw.x", "ph.x", "dynmat.x", "bands.x", "siesta", "lmp"):
        make(name)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    make.dir = bin_dir
    return make


class TestManifest:
    def test_inferred_from_the_files(self, tmp_path):
        for name in ("pw.in", "ph.in", "dynmat.in"):
            (tmp_path / name).write_text("x")
        manifest = manifest_for_directory(tmp_path)
        assert manifest.engine == "qe"
        assert [s.program for s in manifest.steps] == ["pw.x", "ph.x", "dynmat.x"]
        assert [s.parallel for s in manifest.steps] == [True, True, False]
        assert manifest.steps[-1].output == "dynmat.out"

    def test_bands_siesta_lammps_and_nothing(self, tmp_path):
        bands = tmp_path / "b"
        bands.mkdir()
        for name in ("pw.scf.in", "pw.bands.in", "bands.in"):
            (bands / name).write_text("x")
        assert [s.name for s in manifest_for_directory(bands).steps] == \
            ["scf", "bands", "bands.x"]
        siesta = tmp_path / "s"
        siesta.mkdir()
        (siesta / "input.fdf").write_text("x")
        step = manifest_for_directory(siesta).steps[0]
        assert step.stdin and step.output == "input.out"
        empty = tmp_path / "e"
        empty.mkdir()
        with pytest.raises(ValueError, match="no hay entradas"):
            manifest_for_directory(empty)

    def test_round_trip_and_progress(self, tmp_path):
        (tmp_path / "pw.in").write_text("x")
        write_manifest(tmp_path)
        manifest = JobManifest.load(tmp_path)
        assert manifest.progress().startswith("preparado")
        manifest.steps[0].done = True
        manifest.set_status("done")
        manifest.save(tmp_path)
        assert JobManifest.load(tmp_path).progress() == "terminado"

    def test_export_writes_one_per_engine(self, tmp_path):
        export_structure(build_graphene_supercell(3, 3), tmp_path, ["qe", "siesta", "lammps"],
                         calculation_values=_form())
        engines = {engine_of(tmp_path / d) for d in ("qe", "siesta", "lammps")}
        assert engines == {"qe", "siesta", "lammps"}


@posix_only
class TestRunner:
    def _qe_dir(self, tmp_path):
        directory = tmp_path / "qe"
        directory.mkdir()
        for name in ("pw.in", "ph.in", "dynmat.in"):
            (directory / name).write_text("x")
        write_manifest(directory)
        return directory

    def test_runs_every_step_in_order(self, tmp_path, fake_bin):
        directory = self._qe_dir(tmp_path)
        assert run(directory) == 0
        assert "-in pw.in" in (directory / "pw.out").read_text()
        assert (directory / "dynmat.out").exists()
        manifest = JobManifest.load(directory)
        assert manifest.status == "done" and all(s.done for s in manifest.steps)

    def test_failure_stops_and_resumes_after_the_last_good_step(self, tmp_path, fake_bin):
        directory = self._qe_dir(tmp_path)
        fake_bin("ph.x", "echo roto; exit 4")
        assert run(directory) == 4
        manifest = JobManifest.load(directory)
        assert manifest.status == "error" and "ph" in manifest.history[-1]["message"]
        assert [s.done for s in manifest.steps] == [True, False, False]
        (directory / "pw.out").unlink()
        fake_bin("ph.x")                                   # fixed
        assert run(directory) == 0
        assert not (directory / "pw.out").exists()          # scf was not redone

    def test_missing_program_is_named(self, tmp_path, fake_bin):
        directory = self._qe_dir(tmp_path)
        (fake_bin.dir / "dynmat.x").unlink()
        assert run(directory) == 2
        assert "dynmat.x" in JobManifest.load(directory).history[-1]["message"]

    def test_siesta_reads_stdin_and_mpi(self, tmp_path, fake_bin, monkeypatch):
        directory = tmp_path / "siesta"
        directory.mkdir()
        (directory / "input.fdf").write_text("SystemLabel x\n")
        manifest = manifest_for_directory(directory)
        assert step_command(manifest.steps[0], 1)[-1].endswith("siesta")
        monkeypatch.setenv("CARBONFORGE_MPI", "mympi -np {n}")
        assert step_command(manifest.steps[0], 4)[:3] == ["mympi", "-np", "4"]

    def test_program_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CARBONFORGE_PW_X", "/opt/qe/pw.x")
        assert find_program("pw.x") == "/opt/qe/pw.x"


@posix_only
class TestSharedQueue:
    def _wait(self, queue, timeout=30.0):
        start = time.time()
        while queue.active():
            queue.poll()
            if time.time() - start > timeout:
                raise TimeoutError
            time.sleep(0.05)

    def test_export_then_queue(self, tmp_path, fake_bin):
        export_structure(build_graphene_supercell(3, 3), tmp_path, ["qe"],
                         calculation_values=_form())
        queue = JobQueue()
        job = queue.submit(tmp_path / "qe")
        assert adapter_for(job.directory).available(job.directory) == (True, "")
        self._wait(queue)
        assert job.state == DONE, (job.log_path.read_text())
        assert job.progress() == "terminado" and job.engine == "qe"

    def test_a_failed_step_is_an_error(self, tmp_path, fake_bin):
        (tmp_path / "pw.in").write_text("x")
        write_manifest(tmp_path)
        fake_bin("pw.x", "exit 1")
        queue = JobQueue()
        job = queue.submit(tmp_path)
        self._wait(queue)
        assert job.state == ERROR

    def test_cancel_stops_the_program_too(self, tmp_path, fake_bin):
        (tmp_path / "pw.in").write_text("x")
        write_manifest(tmp_path)
        marker = tmp_path / "still_running"
        fake_bin("pw.x", f"sleep 30; touch {marker}")
        queue = JobQueue()
        job = queue.submit(tmp_path)
        queue.poll()
        time.sleep(0.5)                                     # let pw.x start
        queue.cancel(job)
        assert job.state == CANCELLED
        time.sleep(0.3)
        assert JobManifest.load(tmp_path).status in ("cancelled", "running")
        assert not marker.exists()

    def test_unavailable_program_is_explained(self, tmp_path, monkeypatch):
        (tmp_path / "pw.in").write_text("x")
        write_manifest(tmp_path)
        monkeypatch.setenv("PATH", str(tmp_path / "vacío"))
        ok, why = adapter_for(tmp_path).available(tmp_path)
        assert not ok and "pw.x" in why and "carbonforge.jobs.run" in why


def test_not_a_calculation_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no es un directorio de cálculo"):
        JobQueue().submit(tmp_path)


def test_runner_module_is_runnable():
    import subprocess

    out = subprocess.run([sys.executable, "-m", "carbonforge.jobs.run", "--help"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "job.json" in out.stdout


def _display_available() -> bool:
    import importlib.util

    if os.name != "posix" or importlib.util.find_spec("tkinter") is None \
            or not os.environ.get("DISPLAY"):
        return False
    import tkinter

    try:
        tkinter.Tk().destroy()
    except tkinter.TclError:
        return False
    return True


@pytest.mark.skipif(not _display_available(), reason="sin Tk o sin pantalla")
def test_window_exports_queues_runs_and_opens_the_result(tmp_path, fake_bin, monkeypatch):
    """Preparar → encolar → Trabajos → Abrir resultados, with stand-in QE."""
    import tkinter
    from tkinter import filedialog, messagebox

    from carbonforge.gui.app import CarbonForgeApp
    from carbonforge.gui.params import STRUCTURES, build_structure
    from carbonforge.tests.test_results import DYNMAT_FULL

    fixture = tmp_path / "dynmat_fixture.txt"
    fixture.write_text(DYNMAT_FULL)
    fake_bin("dynmat.x", f"cat {fixture}")
    monkeypatch.setattr(filedialog, "askdirectory", lambda *a, **k: str(tmp_path / "exp"))
    monkeypatch.setattr(messagebox, "showinfo", lambda *a, **k: None)
    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: errors.append(a))

    root = tkinter.Tk()
    try:
        app = CarbonForgeApp(root, vibspec_workdir=tmp_path / "calculos")
        values = {s.key: s.default for s in STRUCTURES["cnt"].params}
        app._on_built(build_structure("cnt", {**values, "n": 7, "m": 0}))
        app._calculation_vars["task"].set("raman")
        for key, var in app._format_vars.items():
            var.set(key == "qe")
        app.force_var.set(True)
        app.enqueue_var.set(True)
        app._on_export()
        assert app._current_page() == "Trabajos" and len(app.jobs.jobs) == 1
        job = app.jobs.jobs[0]
        start = time.time()
        while app.jobs.active() and time.time() - start < 30:
            app.jobs.poll()
            root.update()
            time.sleep(0.05)
        assert job.state == DONE, job_log_text(job)
        app._refresh_jobs()
        app.jobs_tree.selection_set(str(job.directory))
        app._on_job_results()
        assert app._current_page() == "Bandas y espectros"
        assert "cargado" in app.analysis_status_var.get()
        assert not errors
    finally:
        root.destroy()


def job_log_text(job) -> str:
    return job.log_path.read_text() if job.log_path.exists() else ""
