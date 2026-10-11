"""tbkit.gui.project: a session snapshot saved and read back (no Qt needed)."""

from __future__ import annotations

import zipfile

import numpy as np
import pytest
from ase.build import graphene, molecule

from tbkit.gui import project
from tbkit.params import model_from_dict, model_to_dict, xu_carbon


def test_project_round_trip(tmp_path):
    sheet = graphene(vacuum=6.0)
    sheet.pbc = [True, True, False]
    benzene = molecule("C6H6")
    hessian = np.eye(36)
    state = project.Project(
        geometries=[("lámina", sheet), ("benceno relajado", benzene)], active=1,
        window={"model_name": "sp3", "kT": 0.05}, pages={"Geometría y modos": {"fmax": 0.03}},
        model=model_to_dict(xu_carbon()),
        modes={"hessian": hessian, "masses": benzene.get_masses(), "atoms": benzene,
               "source": "prueba"},
        record={"task": "relax"}, console="línea 1\n")
    path = project.save(tmp_path / "sesion", state)
    assert path.suffix == ".tbkproj"
    back = project.load(path)
    assert [n for n, _ in back.geometries] == ["lámina", "benceno relajado"]
    assert list(back.geometries[0][1].pbc) == [True, True, False]       # the cell survives
    assert np.allclose(back.geometries[0][1].cell, sheet.cell)
    assert back.active == 1 and back.window["kT"] == 0.05
    assert back.pages["Geometría y modos"]["fmax"] == 0.03
    assert model_from_dict(back.model).name == xu_carbon().name
    assert np.allclose(back.modes["hessian"], hessian) and back.modes["source"] == "prueba"
    assert back.record == {"task": "relax"} and back.console == "línea 1\n"


def test_project_refuses_other_files(tmp_path):
    bad = tmp_path / "x.tbkproj"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("otra.json", "{}")
    with pytest.raises(ValueError, match="no es un proyecto"):
        project.load(bad)
