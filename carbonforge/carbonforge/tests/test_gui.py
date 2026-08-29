"""Tests for the GUI logic layer.

These exercise everything except the Tk widgets themselves, so they run on a
headless machine with no display and no Tkinter installed.
"""

from __future__ import annotations

import pytest

from carbonforge.gui.params import (
    MODIFIER_PARAMS,
    STRUCTURES,
    apply_modifiers,
    build_structure,
    coerce_value,
    collect_values,
    describe_structure,
    export_structure,
)


def _defaults(specs) -> dict:
    return {s.key: s.default for s in specs}


class TestCoerceValue:
    def test_int_parsing(self):
        spec = STRUCTURES["cnt"].params[0]  # n
        assert coerce_value(spec, "8") == 8
        assert coerce_value(spec, 8) == 8

    def test_float_accepts_comma_decimal(self):
        spec = next(s for s in STRUCTURES["cnt"].params if s.key == "length")
        # A Spanish-locale user may well type "12,5".
        assert coerce_value(spec, "12,5") == pytest.approx(12.5)

    def test_rejects_non_numeric(self):
        spec = STRUCTURES["cnt"].params[0]
        with pytest.raises(ValueError, match="entero"):
            coerce_value(spec, "abc")

    def test_rejects_empty(self):
        spec = STRUCTURES["cnt"].params[0]
        with pytest.raises(ValueError, match="vacío"):
            coerce_value(spec, "   ")

    def test_enforces_minimum(self):
        spec = STRUCTURES["cnt"].params[0]  # n, minimum=1
        with pytest.raises(ValueError, match=">="):
            coerce_value(spec, "0")

    def test_enforces_maximum(self):
        spec = next(
            s for s in STRUCTURES["nanocoil"].params
            if s.key == "stone_wales_density"
        )
        with pytest.raises(ValueError, match="<="):
            coerce_value(spec, "0.9")

    def test_bool_values(self):
        spec = next(s for s in STRUCTURES["nanoribbon"].params if s.key == "passivate")
        assert coerce_value(spec, True) is True
        assert coerce_value(spec, "sí") is True
        assert coerce_value(spec, "no") is False

    def test_choice_validation(self):
        spec = next(s for s in STRUCTURES["nanoribbon"].params if s.key == "edge")
        assert coerce_value(spec, "armchair") == "armchair"
        with pytest.raises(ValueError, match="no es válido"):
            coerce_value(spec, "hexagonal")

    def test_collect_values_fills_defaults(self):
        specs = STRUCTURES["cnt"].params
        values = collect_values(specs, {})
        assert values["n"] == 6 and values["m"] == 6


class TestBuildStructure:
    @pytest.mark.parametrize("kind", list(STRUCTURES))
    def test_every_structure_builds_with_defaults(self, kind):
        """Every default shown in the GUI must produce a valid structure."""
        atoms = build_structure(kind, _defaults(STRUCTURES[kind].params))
        assert len(atoms) > 0

    def test_unknown_kind_rejected(self):
        with pytest.raises(ValueError, match="desconocido"):
            build_structure("fullerene", {})

    def test_builder_errors_propagate(self):
        # coil_radius far below 2x the tube radius -> builder raises.
        values = _defaults(STRUCTURES["nanocoil"].params)
        values["coil_radius"] = 5.0
        values["n"], values["m"] = 10, 10
        with pytest.raises(ValueError):
            build_structure("nanocoil", values)

    def test_gui_bounds_precede_builder(self):
        """A negative length is caught by the spec, not deep in ASE."""
        values = _defaults(STRUCTURES["cnt"].params)
        values["length"] = -5
        with pytest.raises(ValueError, match="Longitud"):
            build_structure("cnt", values)


class TestModifiers:
    def test_no_dopant_is_noop(self):
        atoms = build_structure("graphene", _defaults(STRUCTURES["graphene"].params))
        out = apply_modifiers(atoms, _defaults(MODIFIER_PARAMS))
        assert out.get_chemical_symbols() == atoms.get_chemical_symbols()

    def test_doping_applied(self):
        atoms = build_structure("graphene", _defaults(STRUCTURES["graphene"].params))
        raw = _defaults(MODIFIER_PARAMS)
        raw["dopant"] = "N"
        raw["dopant_concentration"] = 0.1
        out = apply_modifiers(atoms, raw)
        assert "N" in out.get_chemical_symbols()

    def test_vacancies_applied(self):
        atoms = build_structure("graphene", _defaults(STRUCTURES["graphene"].params))
        raw = _defaults(MODIFIER_PARAMS)
        raw["vacancies"] = 3
        out = apply_modifiers(atoms, raw)
        assert len(out) == len(atoms) - 3

    def test_reproducible_given_seed(self):
        atoms = build_structure("graphene", _defaults(STRUCTURES["graphene"].params))
        raw = _defaults(MODIFIER_PARAMS)
        raw.update(dopant="N", dopant_concentration=0.1, seed=99)
        a = apply_modifiers(atoms, raw)
        b = apply_modifiers(atoms, raw)
        assert a.get_chemical_symbols() == b.get_chemical_symbols()


class TestExport:
    def test_exports_every_format(self, tmp_path):
        atoms = build_structure("cnt", _defaults(STRUCTURES["cnt"].params))
        written = export_structure(
            atoms, tmp_path, ["qe", "lammps", "xyz", "cif"]
        )
        assert len(written) == 5  # qe(1) + lammps(2) + xyz(1) + cif(1)
        for path in written:
            assert path.exists() and path.stat().st_size > 0

    def test_unknown_format_rejected(self, tmp_path):
        atoms = build_structure("cnt", _defaults(STRUCTURES["cnt"].params))
        with pytest.raises(ValueError, match="Formato"):
            export_structure(atoms, tmp_path, ["vasp"])

    def test_validation_blocks_export(self, tmp_path):
        import numpy as np

        atoms = build_structure("cnt", _defaults(STRUCTURES["cnt"].params))
        pos = atoms.get_positions()
        pos[1] = pos[0] + np.array([0.2, 0.0, 0.0])  # atomic overlap
        atoms.set_positions(pos)
        with pytest.raises(ValueError):
            export_structure(atoms, tmp_path, ["qe"])
        # ...unless explicitly forced.
        assert export_structure(atoms, tmp_path, ["qe"], force=True)


class TestDescribe:
    def test_summary_mentions_key_facts(self):
        atoms = build_structure("cnt", _defaults(STRUCTURES["cnt"].params))
        text = describe_structure(atoms)
        assert "Fórmula" in text
        assert "Dimensionalidad" in text
        assert "Validación superada" in text

    def test_summary_reports_failure(self):
        import numpy as np

        atoms = build_structure("cnt", _defaults(STRUCTURES["cnt"].params))
        pos = atoms.get_positions()
        pos[1] = pos[0] + np.array([0.2, 0.0, 0.0])
        atoms.set_positions(pos)
        assert "Validación fallida" in describe_structure(atoms)


class TestSpecsAreCoherent:
    """Guard against typos in the declarative specs themselves."""

    @pytest.mark.parametrize("kind", list(STRUCTURES))
    def test_defaults_pass_own_bounds(self, kind):
        for spec in STRUCTURES[kind].params:
            coerce_value(spec, spec.default)

    def test_modifier_defaults_pass_own_bounds(self):
        for spec in MODIFIER_PARAMS:
            coerce_value(spec, spec.default)

    @pytest.mark.parametrize("kind", list(STRUCTURES))
    def test_param_keys_match_builder_signature(self, kind):
        """Every declared param must be a real keyword of its builder."""
        import inspect

        spec = STRUCTURES[kind]
        sig = inspect.signature(spec.builder)
        for param in spec.params:
            assert param.key in sig.parameters, (
                f"{kind}: '{param.key}' no existe en {spec.builder.__name__}"
            )
