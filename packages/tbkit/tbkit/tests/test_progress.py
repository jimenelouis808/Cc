"""tbkit.progress: the estimate is the remaining count times the recent time per step."""

from __future__ import annotations

import io
import json
import time

import pytest

from tbkit import progress as pg


def test_estimate_uses_recent_steps(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(pg.time, "time", lambda: clock[0])
    bar = pg.Progress(10, "prueba", stream=io.StringIO(), window=3)
    for dt in (10.0, 10.0, 2.0, 2.0, 2.0):            # steps get faster
        clock[0] += dt
        bar.step()
    assert bar.seconds_per_step == pytest.approx(2.0)   # only the last three count
    assert bar.remaining_seconds == pytest.approx(10.0)
    assert "quedan ~10 s" in bar.line()


def test_resumed_steps_count_for_fraction_not_rate(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(pg.time, "time", lambda: clock[0])
    bar = pg.Progress(10, stream=io.StringIO(), done=8)
    assert bar.remaining_seconds is None and "8/10 (80 %)" in bar.line()
    clock[0] += 60
    bar.step()
    assert bar.remaining_seconds == pytest.approx(60.0)


def test_status_file_and_stale_detection(tmp_path, monkeypatch):
    clock = [time.time()]
    monkeypatch.setattr(pg.time, "time", lambda: clock[0])
    path = tmp_path / "a" / "progreso.json"
    bar = pg.Progress(4, "fd", status=path, stream=io.StringIO())
    clock[0] += 120
    bar.step()
    data = json.loads(path.read_text())
    assert data["done"] == 1 and data["seconds_per_step"] == pytest.approx(120)
    assert not pg.read_status(path)["stale"]
    clock[0] += 3600                                 # an hour without news
    assert pg.read_status(path)["stale"]
    assert [d["label"] for d in pg.status_files(tmp_path)] == ["fd"]


def test_durations():
    assert pg.duration(45) == "45 s"
    assert pg.duration(12 * 60) == "12 min"
    assert pg.duration(3 * 3600 + 5 * 60) == "3 h 05 min"
    assert pg.duration(50 * 3600) == "2 d 2 h"


# --- what a long k-point calculation reports, and how it fails ---------------------------

def _chain(model_name: str = "pi"):
    from ase.build import graphene_nanoribbon

    from tbkit.gui import actions

    ribbon = graphene_nanoribbon(2, 1, type="armchair", saturated=False, vacuum=5.0)
    return ribbon, actions.load_model(model_name)


def test_a_large_mesh_reports_its_diagonalisation(monkeypatch):
    """From PROGRESS_FROM_K points the k loop sends a counter (what Procesos shows)."""
    from tbkit.hamiltonian import System
    from tbkit.kpoints import mesh
    from tbkit.solver import PROGRESS_FROM_K, diagonalise

    seen = []
    monkeypatch.setattr(pg, "SINK", seen.append)
    ribbon, model = _chain()
    kpts, _ = mesh(ribbon, PROGRESS_FROM_K)
    diagonalise(System.build(ribbon, model), kpts)
    lines = [m["line"] for m in seen if "diagonalización" in m["label"]]
    assert lines and f"{PROGRESS_FROM_K}/{PROGRESS_FROM_K}" in lines[-1]


def test_a_mesh_that_does_not_fit_fails_at_once_with_its_size(monkeypatch):
    """The arrays are allocated after the FIRST point: an hour of diagonalising before a
    64 GB allocation failed is what this replaced."""
    import numpy as np

    from tbkit.hamiltonian import System
    from tbkit.kpoints import mesh
    from tbkit.solver import diagonalise

    ribbon, model = _chain()
    system = System.build(ribbon, model)
    kpts, _ = mesh(ribbon, 40)
    calls = []
    real_eigh = __import__("tbkit.solver", fromlist=["eigh"]).eigh

    def counting_eigh(*args, **kwargs):
        calls.append(1)
        return real_eigh(*args, **kwargs)

    def no_room(shape, dtype=float):
        if np.prod(shape) > 1000:
            raise MemoryError
        return np.zeros(shape, dtype)

    monkeypatch.setattr("tbkit.solver.eigh", counting_eigh)
    monkeypatch.setattr(np, "empty", no_room)
    with pytest.raises(MemoryError, match="40 puntos k"):
        diagonalise(system, kpts)
    assert len(calls) == 1


def test_mixed_gamma_and_complex_points_keep_their_imaginary_part():
    """Γ comes first and is real; the stored vectors must still be complex."""
    import numpy as np

    from tbkit.hamiltonian import System
    from tbkit.kpoints import mesh
    from tbkit.solver import diagonalise

    ribbon, model = _chain()
    system = System.build(ribbon, model)
    kpts, _ = mesh(ribbon, 6)
    _, vectors, _ = diagonalise(system, kpts)
    assert np.iscomplexobj(vectors) and np.abs(vectors[1:].imag).max() > 1e-6
    h, _ = system.hamiltonian(kpts[1])
    e1 = np.linalg.eigvalsh(h)
    e, _, _ = diagonalise(system, kpts[1:2])
    assert np.allclose(e[0], e1)


def test_kmesh_note_counts_points_per_periodic_axis():
    from ase.build import bulk

    from tbkit.gui import actions

    diamond = bulk("C", "diamond", a=3.567)
    note = actions.kmesh_note(diamond, actions.load_model("sp3"), 24)
    assert "24×24×24 = 13824 puntos" in note and "GB" in note
    ribbon, model = _chain()
    assert "1×1×24 = 24 puntos" in actions.kmesh_note(ribbon, model, 24)


def test_scc_ground_state_uses_the_window_mesh():
    """The SCC branch ignored the control and ran on 12 points per axis."""
    from ase.build import graphene_nanoribbon

    from tbkit.gui import actions

    ribbon = graphene_nanoribbon(2, 1, type="armchair", saturated=True, vacuum=5.0)
    state = actions.ground_state(ribbon, actions.load_model("xu_chn"), kmesh=5, scc=True)
    assert len(state.solution.kpts) == 5


def test_relaxation_reports_every_step(monkeypatch):
    from ase.build import molecule

    from tbkit.gui import actions

    seen = []
    monkeypatch.setattr(pg, "SINK", seen.append)
    out = actions.relax_structure(molecule("CH4"), actions.load_model("xu_chn"), steps=5,
                                  fmax=1e-6)
    steps = [m for m in seen if m["label"] == "relajación"]
    assert steps and steps[-1]["done"] >= out["steps"] and "Fmáx" in steps[-1]["note"]


def test_vacuum_axes_declared_periodic_get_one_k_point_exactly():
    """A 1D structure in a box written with pbc on x, y and z (an .xtl, a unit cell from
    another package): k only along the axis bonds cross, with identical results. The
    user's 132-atom coil at 24 per axis was 13 824 points and 62 GB instead of 24."""
    import numpy as np
    from ase.build import graphene_nanoribbon

    from tbkit.gui import actions
    from tbkit.hamiltonian import System
    from tbkit.kpoints import mesh

    ribbon = graphene_nanoribbon(2, 1, type="armchair", saturated=True, vacuum=5.0)
    boxed = ribbon.copy()
    boxed.pbc = True
    model = actions.load_model("xu_chn")
    assert System.build(boxed, model).k_axes == (False, False, True)
    assert len(mesh(System.build(boxed, model), 6)[0]) == 6
    assert len(mesh(boxed, 6)[0]) == 216                  # the Atoms form is unchanged
    for scc in (False, True):
        a = actions.ground_state(ribbon, model, kmesh=6, scc=scc)
        b = actions.ground_state(boxed, model, kmesh=6, scc=scc)
        assert len(b.solution.kpts) == 6
        assert np.allclose(np.sort(a.solution.energies.ravel()),
                           np.sort(b.solution.energies.ravel()), atol=1e-6)
    note = actions.kmesh_note(boxed, model, 24)
    assert "1×1×24 = 24 puntos" in note and "x, y periódicos" in note
