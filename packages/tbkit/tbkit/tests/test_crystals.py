"""Crystal validation against GPAW: builders, periodic references, and resuming after a kill."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tbkit.calculator import TBCalculator
from tbkit.params import xu_carbon
from tbkit.recipes import crystal_validation as cv
from tbkit.references import ReferenceStructure, load_references, save_references


def test_every_crystal_has_the_elements_of_its_set():
    from tbkit.params import load_parameters

    for name, (build, set_name, kpts, index) in cv.CRYSTALS.items():
        atoms = build()
        assert atoms.pbc.any(), name
        assert set(atoms.get_chemical_symbols()) <= load_parameters(set_name).elements(), name
        assert len(kpts) == 3 and all(k == 1 for k, p in zip(kpts, atoms.pbc) if not p), name
        if isinstance(index, int):
            assert atoms[index].symbol != "C", name
        elif isinstance(index, str):
            assert index in atoms.get_chemical_symbols(), name


def test_pyridinic_vacancy_has_three_nitrogens_around_the_hole():
    atoms = cv._pyridinic()
    nitrogens = [a.index for a in atoms if a.symbol == "N"]
    assert len(atoms) == 31 and len(nitrogens) == 3
    # Each N lost one neighbour.
    assert all(len(cv._neighbours(atoms, n)) == 2 for n in nitrogens)


def test_strain_scales_the_periodic_axes_and_keeps_the_vacuum():
    atoms = cv._sheet()
    out = cv.strained(atoms, 1.02)
    assert np.allclose(out.cell.lengths()[:2], 1.02 * atoms.cell.lengths()[:2])
    assert out.cell.lengths()[2] == pytest.approx(atoms.cell.lengths()[2])
    assert np.allclose(out.positions[:, 2], atoms.positions[:, 2])
    assert np.allclose(out.get_scaled_positions(wrap=False)[:, :2],
                       atoms.get_scaled_positions(wrap=False)[:, :2])


def test_periodic_reference_keeps_its_cell(tmp_path):
    atoms = cv._graphane()
    ref = ReferenceStructure("graphane/relaxed", "graphane", atoms, -1.0,
                             np.zeros((len(atoms), 3)), np.zeros(0), 0, "test")
    path = save_references(tmp_path / "r.json", [ref], {"code": "test"})
    (back,), _ = load_references(path)
    assert np.allclose(back.atoms.cell.array, atoms.cell.array)
    assert list(back.atoms.pbc) == [True, True, False]
    # A molecule stays as it was written before (no cell keys).
    molecule = ReferenceStructure("m", "m", atoms[:2].copy(), 0.0, np.zeros((2, 3)),
                                  np.zeros(1), 1)
    molecule.atoms.pbc = False
    assert "cell" not in molecule.to_dict()


class _Killer:
    """A cheap calculator factory that dies after ``limit`` calculations."""

    def __init__(self, limit=None):
        self.limit, self.made = limit, 0

    def __call__(self, grid_of=None):
        self.made += 1
        if self.limit is not None and self.made > self.limit:
            raise KeyboardInterrupt("killed")
        return TBCalculator(xu_carbon(), kpts=(2, 2, 1), kT=0.1)


def test_a_killed_run_resumes_without_repeating_finished_points(tmp_path):
    killer = _Killer(limit=3)          # the relaxation and two single points
    with pytest.raises(KeyboardInterrupt):
        cv.run_crystal("graphene", tmp_path, killer, log=lambda text: None)
    folder = tmp_path / "graphene"
    finished = sorted(p.name for p in folder.glob("*.json") if p.name != "relax_bfgs.json")
    assert finished == ["relaxed.json", "rnd0.json", "rnd1.json"]
    stamps = {p: (folder / p).stat().st_mtime_ns for p in finished}
    second = _Killer()
    cv.run_crystal("graphene", tmp_path, second, log=lambda text: None)
    assert (folder / "done").exists()
    labels = [label for label, _ in cv.distorted(cv._sheet())]
    assert second.made == len(labels) - 2          # only what was missing
    assert all((folder / p).stat().st_mtime_ns == t for p, t in stamps.items())
    # Once done, nothing runs at all.
    third = _Killer()
    cv.run_crystal("graphene", tmp_path, third, log=lambda text: None)
    assert third.made == 0
    out = cv.collect(tmp_path, tmp_path / "refs.json")
    refs, settings = load_references(out)
    assert [r.label for r in refs] == ["graphene/relaxed"] + [f"graphene/{x}" for x in labels]
    assert settings["smearing_ev"] == cv.SETTINGS["smearing_ev"]


class _DiesAfter(TBCalculator):
    """Dies after ``limit`` calculations of its own, like a job killed mid-relaxation."""

    def __init__(self, limit, **kwargs):
        super().__init__(xu_carbon(), kpts=(2, 2, 1), kT=0.1, **kwargs)
        self.limit, self.calls = limit, 0

    def calculate(self, *args, **kwargs):
        self.calls += 1
        if self.calls > self.limit:
            raise KeyboardInterrupt("killed")
        super().calculate(*args, **kwargs)


def test_a_relaxation_cut_mid_way_resumes_from_its_trajectory(tmp_path, monkeypatch):
    from ase.io import read

    def bent():             # one atom out of the plane: several steps to relax
        atoms = cv._sheet()
        atoms.positions[0, 2] += 0.3
        return atoms

    monkeypatch.setitem(cv.CRYSTALS, "graphene", (bent,) + cv.CRYSTALS["graphene"][1:])
    with pytest.raises(KeyboardInterrupt):
        cv.run_crystal("graphene", tmp_path, lambda **kw: _DiesAfter(4), log=lambda text: None)
    folder = tmp_path / "graphene"
    assert not (folder / "relaxed.json").exists()
    first = read(folder / "relax.traj", index=":")
    assert len(first) >= 3 and abs(first[-1].positions[0, 2] - first[0].positions[0, 2]) > 0.01
    cv.run_crystal("graphene", tmp_path, lambda **kw: _DiesAfter(10 ** 6), log=lambda text: None)
    relaxed = json.loads((folder / "relaxed.json").read_text())
    assert relaxed["converged"]
    assert relaxed["steps"] >= len(first) - 1
    z = np.array(relaxed["positions"])[:, 2]
    assert abs(z[0] - np.median(z)) < 0.1          # the bump (0.3 Å) is gone


def test_a_strained_cell_keeps_the_grid_of_the_relaxed_one():
    pytest.importorskip("gpaw")
    atoms = cv._sheet()
    squeezed = cv.strained(atoms, 0.98)
    make = cv.gpaw_factory((1, 1, 1))
    free = make()
    free.initialize(squeezed)
    fixed = make(grid_of=atoms)
    fixed.initialize(squeezed)
    reference = make()
    reference.initialize(atoms)
    assert list(fixed.wfs.gd.N_c) == list(reference.wfs.gd.N_c)
    # Without it the grid jumps (44 -> 40 along this cell), which is the bug.
    assert list(free.wfs.gd.N_c) != list(reference.wfs.gd.N_c)
