"""tbkit.settings: recipe numbers overridden per run, typed, refused when unknown, recorded."""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

import pytest

from tbkit import settings as st


def _module():
    m = types.ModuleType("receta_prueba")
    m.KZ = 4
    m.ETA = 0.1
    m.LASERS = (1.96, 2.33)
    m.SOURCE = Path("out/x.extxyz")
    m.GPAW_SETTINGS = {"basis": "dzp", "h": 0.2}
    m.SCC = True
    m.PARAMS = [st.Param(n, n.lower()) for n in ("KZ", "ETA", "LASERS", "SOURCE",
                                                    "GPAW_SETTINGS", "SCC")]
    return m


def _args(*items, file=None):
    parser = argparse.ArgumentParser()
    st.add_arguments(parser)
    argv = [x for item in items for x in ("--ajuste", item)]
    if file:
        argv += ["--ajustes", str(file)]
    return parser.parse_args(argv)


def test_values_take_the_type_of_the_default(tmp_path):
    m = _module()
    changed = st.apply(m, _args("KZ=8", "ETA=0.05", "LASERS=[1.58, 2.54]", "SOURCE=a/b.xyz",
                                'GPAW_SETTINGS={"basis": "szp"}', "SCC=false"))
    assert m.KZ == 8 and isinstance(m.KZ, int)
    assert m.ETA == 0.05 and m.LASERS == (1.58, 2.54) and m.SOURCE == Path("a/b.xyz")
    assert m.GPAW_SETTINGS == {"basis": "szp", "h": 0.2}          # merged, not replaced
    assert m.SCC is False and set(changed) == {"KZ", "ETA", "LASERS", "SOURCE",
                                               "GPAW_SETTINGS", "SCC"}


def test_unknown_names_and_bad_types_are_refused():
    with pytest.raises(SystemExit, match="no es un ajuste"):
        st.apply(_module(), _args("NKK=3"))
    with pytest.raises(SystemExit, match="KZ"):
        st.apply(_module(), _args("KZ=2.5"))


def test_file_and_record(tmp_path):
    m = _module()
    path = tmp_path / "ajustes.json"
    path.write_text(json.dumps({"KZ": 6}))
    st.apply(m, _args("ETA=0.2", file=path), record=tmp_path / "work")
    history = json.loads((tmp_path / "work" / "ajustes_usados.json").read_text())
    assert history[-1]["changed"] == {"KZ": 6, "ETA": 0.2}
    assert history[-1]["values"]["LASERS"] == [1.96, 2.33]
    assert "KZ = 6" in st.describe(m)


def test_every_listed_recipe_declares_existing_names():
    import importlib

    for name in st.RECIPES:
        module = importlib.import_module(f"tbkit.recipes.{name}")
        for p in getattr(module, "PARAMS", []):
            assert hasattr(module, p.name), (name, p.name)
