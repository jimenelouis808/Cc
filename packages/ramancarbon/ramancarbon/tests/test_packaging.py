"""The package ships the data it needs.

A development checkout finds every JSON and every CIF because they are
sitting in the source tree. An installed wheel finds only what
``package-data`` lists, so a missing line there produces a program that
works for the developer and has an empty reference library for everyone
else. That difference is invisible until someone installs it.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parents[1] / "database" / "data"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_every_data_file_pattern_is_declared(pyproject):
    patterns = pyproject["tool"]["setuptools"]["package-data"]["ramancarbon"]
    on_disk = {
        f"database/data/{p.relative_to(DATA).parent}/*{p.suffix}".replace("/./", "/")
        for p in DATA.rglob("*")
        if p.is_file() and p.suffix in (".json", ".cif")
    }
    missing = {p for p in on_disk if p not in patterns}
    assert not missing, (
        f"estos archivos de datos no los recoge package-data: {sorted(missing)}"
    )


def test_the_declared_patterns_actually_match_something(pyproject):
    package = Path(__file__).resolve().parents[1]
    for pattern in pyproject["tool"]["setuptools"]["package-data"]["ramancarbon"]:
        assert list(package.glob(pattern)), f"{pattern} no encaja con nada"


def test_the_reference_library_is_not_empty():
    from ramancarbon.xrd.reference import load_library

    assert len(load_library()) >= 12


def test_every_json_database_loads():
    import json

    for path in sorted(DATA.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload, path.name


def test_the_version_is_consistent(pyproject):
    from ramancarbon import __version__

    assert pyproject["project"]["version"] == __version__


def test_the_console_scripts_point_at_something_importable(pyproject):
    import importlib

    for entry in pyproject["project"]["scripts"].values():
        module, _, attribute = entry.partition(":")
        imported = importlib.import_module(module)
        assert hasattr(imported, attribute), entry
