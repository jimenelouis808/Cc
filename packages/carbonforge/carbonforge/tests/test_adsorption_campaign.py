"""An isotherm as a folder: the request, resuming, and the queue's view.

What is worth pinning here is not the sampling -- ``test_adsorption.py``
covers that against the published tables -- but the three properties the
folder format exists for, because each one only matters when something
goes wrong and so none of them is exercised by a happy run:

* a point already computed is never computed again, which at 31-46
  minutes a point is the difference between a cancel costing one point
  and costing a day;
* a refusal happens when the folder is WRITTEN, not an hour later inside
  a subprocess whose output nobody is watching;
* the job queue recognises the folder and reads its progress from disk,
  without the runner telling it anything.

Every run here uses a deliberately tiny move count. These tests are
about the bookkeeping; a 600-move GCMC is meaningless as physics and
says so in its own warnings.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from ase import Atoms

from carbonforge.adsorption.campaign import (PROGRESS, REQUEST, RESULTS,
                                             read_isotherm_request,
                                             read_results, run_isotherm,
                                             status, write_isotherm_request)
from carbonforge.jobs.queue import GcmcAdapter, adapter_for, engine_of


@pytest.fixture
def framework():
    """A cell big enough for the minimum image at a 9 Å cutoff.

    Two atoms rather than none, so the framework energy term is actually
    evaluated; and 30 Å so that 2 x 9 fits inside every edge.
    """
    return Atoms("C2", positions=[(0.0, 0.0, 0.0), (8.0, 8.0, 8.0)],
                 cell=[30.0] * 3, pbc=True)


def _request(framework, folder, **kwargs):
    options = dict(temperature=293.0, pressures=[1.0, 5.0], cutoff=9.0,
                   equilibration=300, production=600, porosity_samples=2000)
    options.update(kwargs)
    return write_isotherm_request(framework, folder, **options)


class TestTheRequestIsCheckedWhenItIsWritten:
    """Every refusal here would otherwise surface an hour in."""

    def test_it_writes_the_structure_and_the_request(self, framework, tmp_path):
        folder = _request(framework, tmp_path / "run", name="prueba")
        request = read_isotherm_request(folder)
        assert request["nombre"] == "prueba"
        assert request["presiones_mpa"] == [1.0, 5.0]
        assert request["atomos"] == 2
        assert (folder / request["estructura"]).exists()

    def test_the_pressures_are_sorted_and_deduplicated(self, framework,
                                                       tmp_path):
        """Results are keyed by pressure, so two of the same would make
        resuming ambiguous."""
        folder = _request(framework, tmp_path / "run",
                          pressures=[5.0, 1.0, 5.0])
        assert read_isotherm_request(folder)["presiones_mpa"] == [1.0, 5.0]

    def test_a_triclinic_cell_is_refused_now_not_later(self, tmp_path):
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)], pbc=True,
                      cell=[[30, 0, 0], [0, 30, 0], [4, 0, 30]])
        with pytest.raises(ValueError, match="orthorhombic"):
            _request(atoms, tmp_path / "run")

    def test_a_cell_shorter_than_twice_the_cutoff_is_refused_now(self,
                                                                 tmp_path):
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)], cell=[15.0] * 3,
                      pbc=True)
        with pytest.raises(ValueError, match="shortest cell edge"):
            _request(atoms, tmp_path / "run", cutoff=12.0)

    def test_a_non_periodic_cell_is_refused(self, tmp_path):
        atoms = Atoms("C", positions=[(0.0, 0.0, 0.0)], cell=[30.0] * 3,
                      pbc=False)
        with pytest.raises(ValueError, match="periodic"):
            _request(atoms, tmp_path / "run")

    def test_a_temperature_with_no_reservoir_is_refused(self, framework,
                                                        tmp_path):
        """The virial is fitted to published densities, which exist at
        77, 150 and 293 K. Anything else has no reservoir, and finding
        that out per point would waste the first one."""
        with pytest.raises(ValueError):
            _request(framework, tmp_path / "run", temperature=200.0)

    def test_no_pressures_is_refused(self, framework, tmp_path):
        with pytest.raises(ValueError, match="at least one"):
            _request(framework, tmp_path / "run", pressures=[])

    def test_zero_pressure_is_refused(self, framework, tmp_path):
        """The reservoir enters as a fugacity, undefined at zero."""
        with pytest.raises(ValueError, match="not positive"):
            _request(framework, tmp_path / "run", pressures=[0.0, 1.0])

    def test_it_will_not_overwrite_a_folder_part_way_through(self, framework,
                                                             tmp_path):
        folder = _request(framework, tmp_path / "run")
        with pytest.raises(ValueError, match="already exists"):
            _request(framework, folder)

    def test_the_structure_is_written_without_its_info(self, framework,
                                                       tmp_path):
        """A bond list in an extxyz comment line is what breaks ASE's own
        reader on a framework of this size."""
        from ase.io import read

        framework.info["bonds"] = [(0, 1)] * 5000
        folder = _request(framework, tmp_path / "run")
        assert "bonds" not in read(str(folder / "estructura.extxyz")).info


class TestResuming:
    """The property the whole format exists for."""

    def test_a_second_run_recomputes_nothing(self, framework, tmp_path):
        folder = _request(framework, tmp_path / "run")
        first = run_isotherm(folder)
        assert len(first) == 2
        stamp = (folder / RESULTS).read_text()
        seconds = [row["segundos"] for row in first]

        again = run_isotherm(folder)
        assert len(again) == 2
        # Identical file, so not one point was rerun -- `segundos` would
        # differ by the clock even if the physics came out the same.
        assert (folder / RESULTS).read_text() == stamp
        assert [row["segundos"] for row in again] == seconds

    def test_a_lost_point_is_the_only_one_recomputed(self, framework,
                                                     tmp_path):
        folder = _request(framework, tmp_path / "run")
        run_isotherm(folder)
        stored = json.loads((folder / RESULTS).read_text())
        kept = [row for row in stored["puntos"] if row["presion_mpa"] == 1.0]
        survivor = dict(kept[0])
        stored["puntos"] = kept
        (folder / RESULTS).write_text(json.dumps(stored))

        points = run_isotherm(folder)
        assert len(points) == 2
        # The surviving point is byte-for-byte what it was.
        again = next(r for r in points if r["presion_mpa"] == 1.0)
        assert again == survivor

    def test_a_truncated_results_file_does_not_lose_the_run(self, framework,
                                                            tmp_path):
        """A kill during the write leaves invalid JSON. Recomputing the
        points is bad; refusing to report the folder at all is worse."""
        folder = _request(framework, tmp_path / "run")
        run_isotherm(folder)
        (folder / RESULTS).write_text('{"puntos": [{"presion')
        assert read_results(folder) == []
        assert status(folder)["state"] in ("sin empezar", "corriendo")
        assert len(run_isotherm(folder)) == 2

    def test_only_the_named_pressures_run(self, framework, tmp_path):
        folder = _request(framework, tmp_path / "run")
        points = run_isotherm(folder, pressures=[5.0])
        assert [row["presion_mpa"] for row in points] == [5.0]

    def test_a_pressure_the_request_does_not_name_is_refused(self, framework,
                                                             tmp_path):
        """Growing a folder would make its results not the isotherm its
        own request describes."""
        folder = _request(framework, tmp_path / "run")
        with pytest.raises(ValueError, match="not pressures of this request"):
            run_isotherm(folder, pressures=[99.0])


class TestWhatTheRunRecords:
    def test_every_point_carries_both_weight_percent_conventions(
            self, framework, tmp_path):
        """The two differ by 5 % at 5 wt%, and the papers do not say
        which they mean, so a stored point names both."""
        folder = _request(framework, tmp_path / "run", pressures=[5.0])
        point = run_isotherm(folder)[0]
        assert point["wt_pct_total"] <= point["wt_pct_por_carbono"]
        assert set(point) >= {"densidad_absoluta_g_cm3",
                              "densidad_exceso_g_cm3", "densidad_gas_g_cm3",
                              "n_medio", "n_error", "aceptacion", "avisos"}

    def test_the_porosity_is_computed_once_and_stored(self, framework,
                                                      tmp_path):
        """Every point's excess uses the same helium void fraction. A
        per-point one would make the excess column wobble for reasons
        that have nothing to do with the adsorption."""
        folder = _request(framework, tmp_path / "run")
        run_isotherm(folder)
        stored = json.loads((folder / RESULTS).read_text())
        assert "helium_void_fraction" in stored["porosidad"]
        # Not carried in the file: the arrays are the distribution, and
        # they are not JSON.
        assert "pore_diameters" not in stored["porosidad"]

    def test_a_short_run_says_so_rather_than_looking_converged(
            self, framework, tmp_path):
        folder = _request(framework, tmp_path / "run", pressures=[5.0],
                          equilibration=50, production=100)
        assert run_isotherm(folder)[0]["avisos"]


class TestTheQueueSeesIt:
    """The queue reports a job by reading its directory, nothing else."""

    def test_the_adapter_claims_the_folder(self, framework, tmp_path):
        folder = _request(framework, tmp_path / "run")
        assert isinstance(adapter_for(folder), GcmcAdapter)
        assert engine_of(folder) == "gcmc"

    def test_the_adapter_comes_before_the_manifest_one(self, framework,
                                                       tmp_path):
        """StepsAdapter claims any folder holding a manifest, and an
        isotherm folder could grow one."""
        from carbonforge.jobs.manifest import MANIFEST

        folder = _request(framework, tmp_path / "run")
        (folder / MANIFEST).write_text("{}")
        assert isinstance(adapter_for(folder), GcmcAdapter)

    def test_progress_counts_points_before_during_and_after(self, framework,
                                                            tmp_path):
        folder = _request(framework, tmp_path / "run")
        adapter = adapter_for(folder)
        assert adapter.progress(folder) == "0/2 puntos"
        assert not adapter.succeeded(folder)

        run_isotherm(folder, pressures=[1.0])
        assert adapter.progress(folder).startswith("1/2 puntos")
        assert not adapter.succeeded(folder)

        run_isotherm(folder)
        assert adapter.progress(folder).startswith("2/2 puntos")
        assert adapter.succeeded(folder)

    def test_the_command_ignores_nprocs(self, framework, tmp_path):
        """One Markov chain does not split across ranks. Running it on
        four processes would compute one answer four times."""
        folder = _request(framework, tmp_path / "run")
        adapter = adapter_for(folder)
        assert adapter.command(folder, 1) == adapter.command(folder, 8)
        assert "carbonforge.adsorption.run" in adapter.command(folder, 1)

    def test_a_folder_with_no_request_is_not_claimed(self, tmp_path):
        empty = tmp_path / "nada"
        empty.mkdir()
        assert not GcmcAdapter().matches(empty)

    def test_a_folder_from_another_version_is_not_half_read(self, framework,
                                                            tmp_path):
        folder = _request(framework, tmp_path / "run")
        request = json.loads((folder / REQUEST).read_text())
        request["esquema"] = "carbonforge-isoterma/99"
        (folder / REQUEST).write_text(json.dumps(request))
        with pytest.raises(ValueError, match="not"):
            read_isotherm_request(folder)
        assert status(folder)["state"] == "sin salida"


class TestTheRunnerEntryPoint:
    """The queue spawns it, so it has to work as a subprocess."""

    def test_it_runs_a_folder_and_prints_the_table(self, framework, tmp_path):
        folder = _request(framework, tmp_path / "run", pressures=[5.0])
        result = subprocess.run(
            [sys.executable, "-m", "carbonforge.adsorption.run", str(folder)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        assert "wt% /C" in result.stdout
        assert "terminado" in result.stdout
        assert (folder / PROGRESS).exists()

    def test_a_folder_with_no_request_is_a_sentence_not_a_traceback(
            self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "carbonforge.adsorption.run",
             str(tmp_path)],
            capture_output=True, text=True, check=False)
        assert result.returncode == 1
        assert "Traceback" not in result.stderr
        assert "no se puede leer" in result.stderr.lower()


class TestTheWindowIsWiredToIt:
    """The page, checked without a display.

    ``gui/app.py`` imports tkinter inside ``__init__`` rather than at
    module scope, so the class itself — its sections, its mixins and its
    methods — can be inspected on a machine with no Tk, which is where
    this runs. What cannot be inspected is the drawing, so what is
    pinned here is the wiring: that the page exists, that it is built,
    and that the long half goes to the queue instead of running inline.
    """

    @staticmethod
    def _app():
        from carbonforge.gui.app import CarbonForgeApp

        return CarbonForgeApp

    def test_the_page_is_listed_under_calcular(self):
        assert "Adsorción (H₂)" in self._app().SECTIONS["Calcular"]

    def test_the_tab_is_one_of_the_window_s_mixins(self):
        from carbonforge.gui.tabs import AdsorptionTab

        assert AdsorptionTab in self._app().__mro__

    def test_the_page_is_actually_built(self):
        """A page in SECTIONS with no builder call is an empty frame, and
        the window would show it without complaining."""
        from pathlib import Path

        import carbonforge.gui.app as app

        source = Path(app.__file__).read_text(encoding="utf-8")
        assert '_build_adsorption_tab(self.pages["Adsorción (H₂)"])' in source

    def test_it_is_built_after_the_jobs_page(self):
        """It queues through ``submit_job``, which the jobs page owns."""
        from pathlib import Path

        import carbonforge.gui.app as app

        source = Path(app.__file__).read_text(encoding="utf-8")
        assert (source.index("_build_jobs_tab(")
                < source.index("_build_adsorption_tab("))

    def test_the_isotherm_goes_to_the_queue_and_is_not_run_inline(self):
        """45 minutes a point on the Tk thread is not a slow window, it
        is a hung one — and the Cancel button would be dead too."""
        from pathlib import Path

        import carbonforge.gui.tabs.adsorption as tab

        source = Path(tab.__file__).read_text(encoding="utf-8")
        assert "self.submit_job(" in source
        assert "run_gcmc" not in source
        assert "run_isotherm" not in source

    def test_the_porosity_does_run_inline(self):
        """Seconds, and the user is waiting on a sentence. A folder on
        disk and a row in a job list would be worse than the wait."""
        from pathlib import Path

        import carbonforge.gui.tabs.adsorption as tab

        source = Path(tab.__file__).read_text(encoding="utf-8")
        assert "analyse_porosity(" in source

    def test_only_the_temperatures_with_a_reservoir_are_offered(self):
        """A free field would let somebody type 200 K and be refused
        after the folder dialog had closed."""
        from carbonforge.adsorption.bulk import LEVESQUE_BULK
        from carbonforge.gui.tabs.adsorption import TEMPERATURES

        assert {float(t) for t in TEMPERATURES} == set(LEVESQUE_BULK)

    def test_the_default_pressures_are_the_published_ladder(self):
        """The page's default has to be a ladder the reference tables
        use, so a first run is comparable to something."""
        from carbonforge.adsorption.bundle import LEVESQUE_BUNDLE_ISOTHERMS
        from carbonforge.gui.tabs.adsorption import DEFAULT_PRESSURES

        offered = [float(p) for p in DEFAULT_PRESSURES.split(",")]
        published = [row[0] for row in LEVESQUE_BUNDLE_ISOTHERMS[77.0]]
        assert offered == published
