"""The shared current structure: how one page hands a structure to another."""

from __future__ import annotations

import pytest

from carbonforge.builders import build_cnt
from carbonforge.builders.nanoribbon import build_finite_nanoribbon
from carbonforge.gui.session import Session
from carbonforge.vibspec.core import ImportRefused, load_atoms
from carbonforge.vibspec.gui import logic


@pytest.fixture
def flake():
    return build_finite_nanoribbon(5, 3, edge="armchair")


class TestSession:
    def test_publish_copies_and_notifies(self, flake):
        session, seen = Session(), []
        session.subscribe(seen.append)
        assert seen == [None]                        # told the state at once
        current = session.publish(flake, "Construir")
        assert seen[-1] is current and current.origin == "Construir"
        flake.positions[0, 0] += 5.0                 # the publisher keeps editing
        assert current.atoms.positions[0, 0] != flake.positions[0, 0]

    def test_take_is_a_private_copy(self, flake):
        session = Session()
        assert session.take() is None
        session.publish(flake, "Importar")
        taken = session.take()
        taken.positions += 1.0
        assert (session.current.atoms.positions != taken.positions).all()

    def test_clear_and_describe(self, flake):
        session, seen = Session(), []
        session.publish(flake, "Construir")
        assert "átomos" in session.current.describe() and "Construir" in \
            session.current.describe()
        session.subscribe(seen.append)
        session.clear()
        assert seen[-1] is None and session.current is None


class TestVibspecFromMemory:
    def test_finite_structure_is_accepted_unchanged(self, flake):
        before = flake.positions.copy()
        model, report = load_atoms(flake, vacuum_per_side=8.0, label="Construir")
        assert (flake.positions == before).all()     # caller's atoms untouched
        assert model.info["source"] == "Construir" and "source_file" not in model.info
        assert not model.get_pbc().any() and model.info["vacuum_per_side"] == 8.0
        assert "Modelo:" in report

    def test_periodic_structure_is_refused_with_its_origin(self):
        with pytest.raises(ImportRefused, match="«Construir».*periódica"):
            load_atoms(build_cnt(6, 0, length=8), label="Construir")

    def test_build_model_prefers_atoms_and_applies_the_preset(self, flake):
        raw = {**logic.defaults(logic.BUILDER_PARAMS), "preset": "amine"}
        result = logic.build_model(raw, atoms=flake, label="Importar")
        assert "N" in result.atoms.get_chemical_symbols()
        assert "N" not in flake.get_chemical_symbols()
        assert "Archivo cargado" in result.summary()
