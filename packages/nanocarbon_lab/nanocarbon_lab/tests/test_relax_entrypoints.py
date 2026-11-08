"""The valence relaxer's two doors: the command line and the window.

The engine itself is covered by ``test_relax_valence.py``. What is
covered here is reaching it, which until now nothing but a test did:
``relax_structure`` and ``write_relaxation_request`` existed and no
command and no button called either, so a structure somebody imported
could be *described* and not fixed.

Three things are worth pinning rather than the plumbing in general:

* the worker's request dispatch, because a tag it does not recognise
  must raise and never quietly fall back to a build -- the wrong
  structure returned looks exactly like the right one;
* that a relaxation really is an improvement by the numbers the readout
  shows, so a green verdict in the window means something;
* that a non-carbon structure is warned about, because carbon's angle
  targets applied to a dichalcogenide deform it while leaving the bond
  lengths looking right.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

import nanocarbon_lab.gui as gui
from nanocarbon_lab.builders.capped_cnt import build_capped_cnt
from nanocarbon_lab.gui.worker import BUILD, RELAX, run_request
from nanocarbon_lab.jobs import Job
from nanocarbon_lab.relax import relax_structure


@pytest.fixture(scope="module")
def bent_tube():
    """A capped tube pushed off its own geometry, as a file would be.

    The parabolic shove is what an imported structure looks like after
    somebody else's unconverged run: bonds well off 1.42, angles out of
    the sp2 window, topology intact.
    """
    atoms = build_capped_cnt(n_body_rings=6)
    positions = atoms.get_positions()
    positions[:, 0] += 0.25 * (positions[:, 2] / 10.0) ** 2
    atoms.set_positions(positions)
    atoms.info.clear()
    return atoms


class TestTheWorkerDispatch:
    """One queue, two kinds of work, and no silent third."""

    def test_a_relax_request_relaxes(self, bent_tube):
        atoms = run_request(RELAX, (bent_tube.copy(), {}))
        report = atoms.info["valence_relax"]
        assert report["after"]["bond_max"] < report["before"]["bond_max"]

    def test_a_build_request_still_builds(self):
        """The tag did not break the path every build already took."""
        atoms = run_request(BUILD, Job("fullerene",
                                       {"freq": 1, "family": "C60"}))
        assert len(atoms) == 60

    def test_an_unknown_kind_raises_rather_than_building(self):
        """A typo'd tag must not return a structure at all.

        Falling back to a build would hand back something plausible from
        the wrong input, which is the one failure mode a dispatch like
        this can have.
        """
        with pytest.raises(ValueError, match="unknown request kind"):
            run_request("relaxx", (Atoms("C"), {}))

    def test_relax_options_reach_the_engine(self, bent_tube):
        """``submit_relax``'s keywords are the engine's, not a subset."""
        atoms = run_request(RELAX, (bent_tube.copy(), {"equilibrium": 1.50}))
        assert atoms.info["valence_relax"]["equilibrium"] == 1.50
        after = atoms.info["valence_relax"]["after"]
        assert 1.45 < after["bond_mean"] < 1.55


class TestTheRelaxationIsAnImprovement:
    """What the window colours green has to be true."""

    def test_bonds_and_angles_both_come_back_into_the_sp2_window(
            self, bent_tube):
        """Both, because either alone can look right while the other is
        not: a sheet can reach graphitic bond lengths by folding, which
        is the failure the angle term exists to stop."""
        atoms = relax_structure(bent_tube.copy())
        report = atoms.info["valence_relax"]
        before, after = report["before"], report["after"]
        assert before["bond_min"] < 1.35 or before["bond_max"] > 1.50
        assert 1.38 < after["bond_min"] and after["bond_max"] < 1.46
        assert after["angle_min"] > before["angle_min"]
        assert after["n_close_contacts"] == 0

    def test_the_topology_is_frozen(self, bent_tube):
        """The bond count cannot move: that is the whole contract."""
        atoms = bent_tube.copy()
        relaxed = relax_structure(atoms.copy())
        before = relax_structure(atoms.copy(), equilibrium=1.42)
        assert (relaxed.info["valence_relax"]["bonds"]
                == before.info["valence_relax"]["bonds"])
        assert len(relaxed) == len(bent_tube)


class TestNonCarbonIsWarnedAbout:
    """Carbon's angle targets on something that is not carbon."""

    def test_a_dichalcogenide_gets_a_warning_naming_its_elements(self):
        """A chalcogen at three neighbours sits near 82 deg, not 120, and
        a six-coordinate metal has no single angle at all. The relaxation
        does not fail -- it deforms the sandwich and leaves the bond
        lengths looking fine -- so the warning is the only signal."""
        atoms = Atoms("MoS2", positions=[(0.0, 0.0, 0.0),
                                         (0.0, 0.0, 2.41),
                                         (2.41, 0.0, 0.0)])
        relaxed = relax_structure(atoms, equilibrium=2.41)
        warnings = relaxed.info["valence_relax"]["warnings"]
        assert any("Mo, S" in w and "carbon's" in w for w in warnings)

    def test_an_override_silences_it(self):
        """Somebody who passed their own angle targets knows."""
        atoms = Atoms("MoS2", positions=[(0.0, 0.0, 0.0),
                                         (0.0, 0.0, 2.41),
                                         (2.41, 0.0, 0.0)])
        relaxed = relax_structure(atoms, equilibrium=2.41,
                                  angle_override={2: 82.0, 3: 82.0})
        warnings = relaxed.info["valence_relax"]["warnings"]
        assert not any("carbon's" in w for w in warnings)

    def test_plain_carbon_is_not_warned_about(self, bent_tube):
        relaxed = relax_structure(bent_tube.copy())
        warnings = relaxed.info["valence_relax"]["warnings"]
        assert not any("carbon's" in w for w in warnings)


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "nanocarbon_lab.cli.main", *args],
        capture_output=True, text=True, check=False)


class TestTheCommandLine:
    """`nanocarbon relax` and `nanocarbon relax-request`."""

    def test_relax_writes_a_file_and_prints_before_and_after(
            self, bent_tube, tmp_path):
        source = tmp_path / "bent.xyz"
        write(str(source), bent_tube)
        out = tmp_path / "clean.xyz"
        result = _cli("relax", str(source), "--out", str(out))
        assert result.returncode == 0, result.stderr
        assert "before" in result.stdout and "after" in result.stdout
        assert out.exists()
        # The relaxed file must reopen. It is written without `info`
        # because a bond list of this length in an extxyz comment line is
        # what breaks ASE's own reader, and a file that cannot be read
        # back is not an output.
        reopened = read(str(out))
        assert len(reopened) == len(bent_tube)
        moved = np.linalg.norm(
            reopened.get_positions() - bent_tube.get_positions(), axis=1)
        assert moved.max() > 0.1

    def test_relax_defaults_its_output_beside_the_input(self, bent_tube,
                                                        tmp_path):
        source = tmp_path / "bent.xyz"
        write(str(source), bent_tube)
        assert _cli("relax", str(source)).returncode == 0
        assert (tmp_path / "bent_relaxed.xyz").exists()

    def test_a_missing_file_is_a_sentence_not_a_traceback(self, tmp_path):
        result = _cli("relax", str(tmp_path / "nope.xyz"))
        assert result.returncode == 1
        assert "Could not read" in result.stderr
        assert "Traceback" not in result.stderr

    def test_relax_request_writes_the_exchange_folder(self, bent_tube,
                                                      tmp_path):
        source = tmp_path / "bent.xyz"
        write(str(source), bent_tube)
        folder = tmp_path / "request"
        result = _cli("relax-request", str(source), str(folder),
                      "--settings", json.dumps({"skf_dir": "/opt/skf"}),
                      "--reason", "which isomer is lower")
        assert result.returncode == 0, result.stderr
        request = json.loads((folder / "solicitud.json").read_text())
        assert request["esquema"] == "nanocarbon-dft/1"
        assert request["ajustes"]["skf_dir"] == "/opt/skf"
        assert request["origen"]["motivo"] == "which isomer is lower"
        assert (folder / "estructura.extxyz").exists()

    def test_relax_request_refuses_to_overwrite_a_job_in_flight(
            self, bent_tube, tmp_path):
        source = tmp_path / "bent.xyz"
        write(str(source), bent_tube)
        folder = tmp_path / "request"
        assert _cli("relax-request", str(source), str(folder)).returncode == 0
        again = _cli("relax-request", str(source), str(folder))
        assert again.returncode == 1
        assert "already exists" in again.stderr

    def test_bad_settings_json_is_reported_as_such(self, bent_tube, tmp_path):
        source = tmp_path / "bent.xyz"
        write(str(source), bent_tube)
        result = _cli("relax-request", str(source), str(tmp_path / "r"),
                      "--settings", "{not json")
        assert result.returncode == 1
        assert "not valid JSON" in result.stderr


class TestTheWindowIsWiredToIt:
    """The window cannot be opened here, so read it.

    No tkinter in CI, and the relax panel's whole job is to reach the
    engine -- so what can still be checked is that it reaches it with
    names that exist. The failure this guards is the one this package's
    notes record twice already: a GUI keyword the callee does not take,
    which is a TypeError the first time somebody clicks, with nothing
    drawn and no test having run.
    """

    @staticmethod
    def _app_source() -> str:
        return (Path(gui.__file__).parent / "app.py").read_text()

    def test_the_window_offers_both_buttons(self):
        source = self._app_source()
        assert "command=self.on_relax)" in source
        assert "command=self.on_relax_request)" in source

    def test_the_window_submits_through_the_worker_not_inline(self):
        """Inline would freeze Tk for the length of the minimisation and
        leave Cancel dead, which is the reason worker.py exists."""
        source = self._app_source()
        assert "self.worker.submit_relax(" in source
        assert "relax_structure(" not in source

    def test_the_keywords_the_window_passes_are_the_engine_s(self):
        """Every keyword in the submit call has to be a real parameter of
        ``relax_structure``, or clicking raises."""
        source = self._app_source()
        call = re.search(r"submit_relax\(([^)]*)\)", source)
        assert call is not None
        passed = set(re.findall(r"(\w+)\s*=", call.group(1)))
        accepted = set(inspect.signature(relax_structure).parameters)
        assert passed and passed <= accepted, passed - accepted

    def test_the_relax_button_is_re_enabled_when_work_finishes(self):
        """It is disabled while anything is in flight, so a build that
        finishes has to hand it back -- otherwise one build leaves the
        relax button dead for the rest of the session."""
        source = self._app_source()
        finish = source[source.index("def _finish_build("):]
        finish = finish[:finish.index("\n    def ", 10)]
        assert 'self.btn_relax.config(state="normal")' in finish

    def test_a_relaxation_is_not_reported_as_a_build(self):
        """Both come back as an Atoms under kind "done", so the poll
        handler has to branch on what it asked for."""
        source = self._app_source()
        assert 'self._task == "relax"' in source
        assert "Relaxation complete." in source
