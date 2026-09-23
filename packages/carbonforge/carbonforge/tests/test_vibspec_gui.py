"""Tests for vibspec phase 4: the GUI logic layer (no Tk, no display).

The widgets themselves are a thin layer over :mod:`carbonforge.vibspec.gui.logic`;
everything they call is exercised here, including the subprocess job queue,
with small Python scripts standing in for ``run.py``.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from carbonforge.tests.test_vibspec_workflow import _WATER_CHARGES, _water, springs_factory
from carbonforge.vibspec.core import CalcRecord, CalcSpec, prepare, run
from carbonforge.vibspec.gui import logic


def _wait(queue: logic.JobQueue, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while queue.active():
        queue.poll()
        if time.time() > deadline:
            queue.shutdown()
            raise AssertionError("la cola no terminó a tiempo")
        time.sleep(0.05)


def _script(body: str):
    """A command builder that runs ``body`` with the job directory as argv[1]."""
    def command(directory: Path, nprocs: int) -> list[str]:
        return [sys.executable, "-c", body, str(directory)]
    return command


_MARK_DONE = (
    "import sys; from carbonforge.vibspec.core import CalcRecord; "
    "r = CalcRecord.load(sys.argv[1]); r.set_status('done'); r.save(sys.argv[1]); "
    "print('hecho')"
)


@pytest.fixture
def prepared(tmp_path):
    def make(name: str = "agua") -> Path:
        _, water = _water(perturb=False)
        directory = tmp_path / name
        prepare(water, CalcSpec(), directory)
        return directory
    return make


@pytest.fixture(scope="module")
def water_done(tmp_path_factory):
    directory = tmp_path_factory.mktemp("gui") / "agua"
    reference, start = _water()
    prepare(start, CalcSpec(), directory)
    run(directory, springs_factory(reference, _WATER_CHARGES))
    return directory


class TestForms:
    def test_default_model(self):
        result = logic.build_model(logic.defaults(logic.BUILDER_PARAMS))
        assert result.atoms.get_chemical_formula() == "C58H20"
        assert not any(result.atoms.get_pbc())
        assert "AVISO" in result.summary() and "zigzag" in result.summary()

    def test_preset_and_explicit_site(self):
        raw = logic.defaults(logic.BUILDER_PARAMS)
        raw.update(preset="pyridinic_edge", site_edge="zigzag")
        result = logic.build_model(raw)
        assert result.atoms.info["nitrogen_configurations"][-1]["edge"] == "zigzag"
        index = result.atoms.info["nitrogen_configurations"][-1]["indices"][0]
        raw.update(preset="graphitic", site_index=str(index - 1), site_edge=logic.AUTO)
        with pytest.raises(ValueError):          # that atom is not an interior carbon
            logic.build_model(raw)

    def test_form_values_are_strings_as_tk_gives_them(self):
        raw = {k: str(v) for k, v in logic.defaults(logic.BUILDER_PARAMS).items()}
        raw["vacuum_per_side"] = "7,5"            # decimal comma, as typed in Spanish
        atoms = logic.build_model(raw).atoms
        assert atoms.info["vacuum_per_side"] == pytest.approx(7.5)

    def test_out_of_bounds_vacuum_refused(self):
        raw = logic.defaults(logic.BUILDER_PARAMS)
        raw["vacuum_per_side"] = "3"
        with pytest.raises(ValueError):
            logic.build_model(raw)

    def test_spec_from_form(self):
        raw = {k: str(v) for k, v in logic.defaults(logic.CALC_PARAMS).items()}
        assert logic.spec_from_form(raw) == CalcSpec()
        raw.update(spinpol="sí", nfree="4", fmax="0,005", mode="fd")
        spec = logic.spec_from_form(raw)
        assert spec.spinpol is True and spec.nfree == 4 and spec.fmax == 0.005
        raw["spinpol"] = "no"
        assert logic.spec_from_form(raw).spinpol is False

    def test_job_name(self):
        atoms = logic.build_model(logic.defaults(logic.BUILDER_PARAMS)).atoms
        assert logic.job_name(atoms) == "pristine_armchair_5x3"

    def test_prepare_job(self, tmp_path):
        atoms = logic.build_model(logic.defaults(logic.BUILDER_PARAMS)).atoms
        directory = logic.prepare_job(atoms, CalcSpec(), tmp_path)
        assert (directory / "record.json").exists() and directory.name == "pristine_armchair_5x3"
        with pytest.raises(FileExistsError):
            logic.prepare_job(atoms, CalcSpec(), tmp_path)


class TestJobQueue:
    def test_success(self, prepared):
        queue = logic.JobQueue(command=_script(_MARK_DONE))
        job = queue.submit(prepared())
        assert job.state == logic.QUEUED
        _wait(queue)
        assert job.state == logic.DONE and job.returncode == 0
        assert "hecho" in logic.job_log(job)

    def test_nonzero_exit_is_an_error(self, prepared):
        queue = logic.JobQueue(command=_script("import sys; print('falló'); sys.exit(3)"))
        job = queue.submit(prepared())
        _wait(queue)
        assert job.state == logic.ERROR and job.returncode == 3

    def test_clean_exit_without_done_record_is_an_error(self, prepared):
        # The record, not the exit code alone, says whether it finished.
        queue = logic.JobQueue(command=_script("pass"))
        job = queue.submit(prepared())
        _wait(queue)
        assert job.state == logic.ERROR

    def test_one_at_a_time(self, prepared):
        queue = logic.JobQueue(max_parallel=1,
                               command=_script("import time; time.sleep(0.5)"))
        first, second = queue.submit(prepared("a")), queue.submit(prepared("b"))
        queue.poll()
        assert (first.state, second.state) == (logic.RUNNING, logic.QUEUED)
        _wait(queue)

    def test_cancel(self, prepared):
        queue = logic.JobQueue(command=_script("import time; time.sleep(60)"))
        job = queue.submit(prepared())
        queue.poll()
        assert job.state == logic.RUNNING
        queue.cancel(job)
        assert job.state == logic.CANCELLED
        assert job.process.poll() is not None
        # The directory is untouched and can be queued again.
        assert CalcRecord.load(job.directory).status == "prepared"
        queue.submit(job.directory)

    def test_cancel_queued(self, prepared):
        queue = logic.JobQueue(command=_script(_MARK_DONE))
        job = queue.submit(prepared())
        queue.cancel(job)
        assert job.state == logic.CANCELLED and job.process is None

    def test_duplicate_and_non_calculation(self, prepared, tmp_path):
        queue = logic.JobQueue(command=_script(_MARK_DONE))
        directory = prepared()
        queue.submit(directory)
        with pytest.raises(ValueError, match="ya está"):
            queue.submit(directory)
        with pytest.raises(FileNotFoundError):
            queue.submit(tmp_path)

    def test_launch_failure(self, prepared):
        def broken(directory, nprocs):
            raise RuntimeError("No se encontró mpiexec")
        queue = logic.JobQueue(command=broken)
        job = queue.submit(prepared())
        queue.poll()
        assert job.state == logic.ERROR
        assert "mpiexec" in logic.tail(job.log_path)

    def test_shutdown(self, prepared):
        queue = logic.JobQueue(command=_script("import time; time.sleep(60)"))
        job = queue.submit(prepared())
        queue.poll()
        queue.shutdown()
        assert job.state == logic.CANCELLED and not queue.active()

    def test_real_workflow_in_a_subprocess(self, prepared):
        """The actual relax-then-IR workflow, run by the queue in a child process."""
        body = (
            "import sys; from pathlib import Path; "
            "from carbonforge.tests.test_vibspec_workflow import "
            "_water, springs_factory, _WATER_CHARGES; "
            "from carbonforge.vibspec.core import run; "
            "run(Path(sys.argv[1]), springs_factory(_water()[0], _WATER_CHARGES))"
        )
        queue = logic.JobQueue(command=_script(body))
        job = queue.submit(prepared())
        _wait(queue, timeout=60)
        assert job.state == logic.DONE, logic.job_log(job)
        assert logic.progress(job.directory) == "terminado"


class TestCommandAndProgress:
    def test_default_command(self, tmp_path, monkeypatch):
        assert logic.default_command(tmp_path, 1) == [sys.executable, str(tmp_path / "run.py")]
        monkeypatch.setattr(logic.shutil, "which", lambda name: "/usr/bin/" + name)
        command = logic.default_command(tmp_path, 4)
        assert command[:3] == ["/usr/bin/mpiexec", "-n", "4"]
        monkeypatch.setattr(logic.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError, match="mpiexec"):
            logic.default_command(tmp_path, 4)

    def test_can_run_locally(self, monkeypatch):
        monkeypatch.setattr(logic, "gpaw_available", lambda: False)
        ok, why = logic.can_run_locally()
        assert not ok and "Ubuntu" in why
        monkeypatch.setattr(logic, "gpaw_available", lambda: True)
        assert logic.can_run_locally() == (True, "")

    def test_progress(self, prepared, tmp_path):
        directory = prepared()
        assert logic.progress(directory) == "preparado"
        record = CalcRecord.load(directory)
        record.set_status("relaxing")
        record.save(directory)
        (directory / "relax.log").write_text("head\nstep0\nstep1\n", encoding="utf-8")
        assert logic.progress(directory) == "relajando: paso 2"
        record.set_status("vibrations")
        record.save(directory)
        (directory / "ir").mkdir()
        for name in ("eq", "0x+", "0x-"):
            (directory / "ir" / f"cache.{name}.json").write_text("{}", encoding="utf-8")
        assert logic.progress(directory) == "vibraciones: 3/19 desplazamientos"
        record.set_status("error", "se fue la luz")
        record.save(directory)
        assert "se fue la luz" in logic.progress(directory)
        assert logic.progress(tmp_path) == "sin record.json"


class TestModes:
    def test_load_and_pick(self, water_done):
        atoms, modes, record = logic.load_modes(water_done)
        assert len(atoms) == 3 and modes["modes"].shape == (9, 3, 3)
        frequencies = record.results["frequencies_cm1"]
        # Clicking right on a band picks its mode, and the scale factor is honoured.
        target = record.results["mode_indices"][1]
        assert logic.mode_at(record, frequencies[1] * 0.97, scale_factor=0.97) == target
        # Far from everything: the nearest mode.
        assert logic.mode_at(record, 10_000.0) == record.results["mode_indices"][-1]

    def test_unfinished_refused(self, prepared):
        with pytest.raises(ValueError, match="no ha terminado"):
            logic.load_modes(prepared())

    def test_vector_and_frames(self, water_done):
        atoms, modes, record = logic.load_modes(water_done)
        vector = logic.mode_vector(modes, record.results["mode_indices"][0])
        assert np.linalg.norm(vector, axis=1).max() == pytest.approx(1.0)
        frames = logic.mode_frames(atoms, vector, n_frames=8, amplitude=0.3)
        assert len(frames) == 8
        excursion = max(np.linalg.norm(f - atoms.positions, axis=1).max() for f in frames)
        assert excursion == pytest.approx(0.3, rel=1e-6)
        assert np.allclose(frames[0], atoms.positions)

    def test_character_of_an_oh_stretch(self, water_done):
        atoms, modes, record = logic.load_modes(water_done)
        stretch = record.results["mode_indices"][-1]
        text = logic.mode_character(atoms, logic.mode_vector(modes, stretch))
        assert text.startswith("H ")                  # an O-H stretch is mostly H

    def test_view_angles_look_down_the_plane_normal(self):
        flake = logic.build_model(logic.defaults(logic.BUILDER_PARAMS)).atoms  # x-z plane
        elevation, azimuth = logic.view_angles(flake)
        assert abs(elevation) < 1.0 and abs(abs(azimuth) - 90.0) < 1.0


def test_tail(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("\n".join(str(i) for i in range(100)), encoding="utf-8")
    assert logic.tail(path, 3) == "97\n98\n99"
    assert logic.tail(tmp_path / "nada.txt") == ""


@pytest.mark.skipif(importlib.util.find_spec("tkinter") is not None,
                    reason="con Tk instalado main() abriría una ventana")
def test_window_without_tk_says_how_to_get_it(capsys):
    from carbonforge.vibspec.gui.app import main

    assert main() == 1
    assert "python3-tk" in capsys.readouterr().out


def _display_available() -> bool:
    import os

    if importlib.util.find_spec("tkinter") is None or not os.environ.get("DISPLAY"):
        return False
    import tkinter

    try:
        tkinter.Tk().destroy()
    except tkinter.TclError:
        return False
    return True


@pytest.mark.skipif(not _display_available(), reason="sin Tk o sin pantalla")
def test_window_smoke(tmp_path, water_done):
    """Open the real window, build, prepare, load a result, animate a mode."""
    import tkinter

    from carbonforge.vibspec.gui.app import VibspecApp

    root = tkinter.Tk()
    try:
        app = VibspecApp(root, tmp_path / "calculos")
        app.builder_vars["preset"].set("amine")
        app._on_build()
        assert app.model.atoms.get_chemical_formula() == "C58H21N"
        app._on_prepare(False)
        assert (tmp_path / "calculos" / "amine_armchair_5x3" / "record.json").exists()
        app.results_dir_var.set(str(water_done))
        app._on_load_results()
        app._animate_mode(app.results_record.results["mode_indices"][0], 1.0)
        root.update()
        assert app.animation["k"] >= 1
    finally:
        root.destroy()
