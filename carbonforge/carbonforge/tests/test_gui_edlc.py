"""Tests for the GUI's EDLC tab.

Two halves, split by what can actually be checked here:

* :mod:`carbonforge.gui.edlc_params` holds every non-cosmetic code path and
  contains no Tk, so it is tested directly.
* The tab itself is built against a stub Tk. That is not a substitute for
  opening the window — it cannot tell whether anything is legible — but it
  does catch the mistakes that are otherwise only found by launching:
  a misspelled attribute, a handler wired to a name that does not exist, a
  widget used before it is created.
"""

from __future__ import annotations

import pytest

from carbonforge.builders import build_cnt, build_graphene_supercell
from carbonforge.gui.edlc_params import (
    EDLC_PARAMS,
    MIN_SEPARATION,
    STABILITY_WINDOW,
    build_edlc,
    check_edlc_constraints,
    describe_edlc,
    estimate_size,
    export_edlc,
)


def _defaults(**overrides):
    raw = {spec.key: spec.default for spec in EDLC_PARAMS}
    raw.update(overrides)
    return raw


@pytest.fixture(scope="module")
def electrode():
    return build_graphene_supercell(6, 6)


class TestConstraints:
    def test_sensible_defaults_are_clean(self, electrode):
        assert "✅" in check_edlc_constraints(electrode, _defaults())

    def test_non_planar_electrode_is_an_error(self):
        """A nanotube cannot tile the cross-section, so it cannot be an
        electrode here. This has to be an error, not a warning: the cell
        would build and be nonsense."""
        report = check_edlc_constraints(build_cnt(6, 6, length=10.0), _defaults())
        assert "❌" in report
        assert "x e y" in report

    def test_narrow_separation_warns_per_electrolyte(self, electrode):
        """The floor differs by electrolyte: ionic-liquid layers reach much
        further from the surface than aqueous ones, so a separation that is
        fine for water is not fine for BMIM-PF6."""
        gap = 35.0
        assert MIN_SEPARATION["aqueous"] < gap < MIN_SEPARATION["ionic_liquid"]
        aqueous = check_edlc_constraints(
            electrode, _defaults(separation=gap)
        )
        assert "solapan" not in aqueous
        ionic = check_edlc_constraints(
            electrode, _defaults(separation=gap, electrolyte="ionic_liquid")
        )
        assert "solapan" in ionic

    def test_potential_is_judged_against_the_electrolyte_window(self, electrode):
        """2 V destroys water and is unremarkable in an ionic liquid."""
        assert STABILITY_WINDOW["aqueous"] < 2.0 < STABILITY_WINDOW["ionic_liquid"]
        assert "ventana de estabilidad" in check_edlc_constraints(
            electrode, _defaults(potential=2.0)
        )
        assert "ventana de estabilidad" not in check_edlc_constraints(
            electrode,
            _defaults(potential=2.0, electrolyte="ionic_liquid",
                      separation=50.0),
        )

    def test_wall_gaps_eating_the_whole_region_is_an_error(self, electrode):
        report = check_edlc_constraints(
            electrode, _defaults(separation=10.0, wall_gap=5.0)
        )
        assert "❌" in report

    def test_production_shorter_than_equilibration_warns(self, electrode):
        report = check_edlc_constraints(
            electrode, _defaults(equilibration=500000, production=1000)
        )
        assert "promedia la producción" in report

    def test_impossible_molarity_warns(self, electrode):
        assert "solubilidad" in check_edlc_constraints(
            electrode, _defaults(molarity=5.5)
        )

    def test_narrow_electrode_warns_about_averaging(self):
        assert "Sección transversal" in check_edlc_constraints(
            build_graphene_supercell(2, 2), _defaults()
        )

    def test_vacuum_skips_the_electrolyte_rules(self, electrode):
        """With no electrolyte there is nothing to overlap or electrolyse."""
        report = check_edlc_constraints(
            electrode, _defaults(electrolyte="vacuum", separation=12.0,
                                 potential=8.0)
        )
        assert "solapan" not in report
        assert "ventana de estabilidad" not in report


class TestEstimate:
    def test_estimate_matches_what_gets_built(self, electrode):
        """The estimate exists to avoid paying for the packing to find out,
        so it is worth nothing if it disagrees with the result."""
        raw = _defaults()
        cell = build_edlc(electrode, raw)
        assert f"{len(cell.atoms)} átomos" in estimate_size(electrode, raw)

    def test_estimate_matches_for_ionic_liquid(self, electrode):
        raw = _defaults(electrolyte="ionic_liquid", separation=50.0)
        cell = build_edlc(electrode, raw)
        assert f"{len(cell.atoms)} átomos" in estimate_size(electrode, raw)

    def test_vacuum_estimate_is_just_the_electrodes(self, electrode):
        text = estimate_size(electrode, _defaults(electrolyte="vacuum"))
        assert f"{2 * len(electrode)} átomos" in text


class TestBuildAndExport:
    def test_values_reach_the_cell(self, electrode):
        cell = build_edlc(
            electrode, _defaults(separation=45.0, potential=1.5)
        )
        assert cell.separation == 45.0
        assert cell.potential_v == 1.5

    def test_salt_choice_is_honoured(self, electrode):
        cell = build_edlc(electrode, _defaults(salt="KCl"))
        assert "K" in cell.electrolyte_composition

    def test_seed_makes_it_reproducible(self, electrode):
        a = build_edlc(electrode, _defaults(seed=7))
        b = build_edlc(electrode, _defaults(seed=7))
        assert a.atoms.get_positions() == pytest.approx(b.atoms.get_positions())

    def test_export_writes_the_three_files(self, electrode, tmp_path):
        cell = build_edlc(electrode, _defaults())
        written = export_edlc(cell, tmp_path, _defaults(temperature=350.0))
        assert len(written) == 3
        assert all(p.exists() for p in written)
        script = next(p for p in written if p.name == "in.edlc").read_text()
        assert "350" in script

    def test_description_reports_the_checks(self, electrode):
        cell = build_edlc(electrode, _defaults(potential=9.0))
        text = describe_edlc(cell)
        assert "Celda EDLC" in text
        assert "electroliza" in text


# ----------------------------------------------------------------------
# Tab construction against a stub Tk
# ----------------------------------------------------------------------
class _Var:
    """Stands in for tk.StringVar / tk.BooleanVar."""

    def __init__(self, value=None, **_kwargs):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _Widget:
    """Accepts anything a Tk widget is asked to do, and records nothing.

    Unknown attributes return a callable, so a method this stub has not
    anticipated does not fail the test for the wrong reason.
    """

    def __init__(self, *_args, **_kwargs):
        pass

    def __call__(self, *_args, **_kwargs):
        return _Widget()

    def __getattr__(self, _name):
        return _Widget()


class _Tk:
    Canvas = _Widget
    Text = _Widget
    StringVar = _Var
    BooleanVar = _Var


class _Ttk:
    Frame = _Widget
    LabelFrame = _Widget
    Label = _Widget
    Button = _Widget
    Scrollbar = _Widget
    Combobox = _Widget
    Checkbutton = _Widget
    Entry = _Widget
    Notebook = _Widget


def _stub_app():
    """A CarbonForgeApp with only the EDLC tab built.

    ``__init__`` is bypassed: it builds the 3D preview, which needs
    matplotlib's Tk backend and therefore a real Tk.
    """
    from carbonforge.gui.app import CarbonForgeApp

    app = CarbonForgeApp.__new__(CarbonForgeApp)
    app.tk = _Tk()
    app.ttk = _Ttk()
    app.atoms = None
    app._imported_atoms = None
    app._edlc_vars = {}
    app.edlc_cell = None
    app._edlc_source = None
    app._busy = False
    app._build_edlc_tab(_Widget())
    return app


class TestTabWiring:
    def test_tab_builds_and_registers_every_parameter(self):
        app = _stub_app()
        assert set(app._edlc_vars) == {spec.key for spec in EDLC_PARAMS}

    def test_defaults_round_trip_through_the_widgets(self, electrode):
        """What the widgets hand back has to be usable as-is: the specs
        coerce strings, so a default that does not survive the round trip
        would only fail once someone pressed the button."""
        app = _stub_app()
        raw = app._read_raw(app._edlc_vars)
        assert "✅" in check_edlc_constraints(electrode, raw)

    def test_check_without_a_structure_says_so_instead_of_raising(self):
        app = _stub_app()
        app._on_edlc_check()
        assert "No hay estructura" in app.edlc_status_var.get()

    def test_handlers_exist_for_every_button(self):
        app = _stub_app()
        for name in ("_on_edlc_check", "_on_edlc_build", "_on_edlc_export",
                     "_on_edlc_built", "_edlc_electrode", "_set_edlc_report"):
            assert callable(getattr(app, name))

    def test_export_without_a_cell_is_a_no_op(self):
        app = _stub_app()
        app._on_edlc_export()  # must not raise, and must not open a dialog

    def test_discarding_the_structure_drops_a_cell_built_from_it(self, electrode):
        """Otherwise «Exportar» would write a cell whose electrode is no
        longer the structure on screen."""
        app = _stub_app()
        app.atoms = electrode
        app.edlc_cell = build_edlc(electrode, _defaults())
        app._edlc_source = electrode

        # Stand in for the builder-tab widgets _discard_current_structure
        # also touches.
        app.export_button = _Widget()
        app.png_button = _Widget()
        app.axes = _Widget()
        app.canvas = _Widget()
        app.info_text = _Widget()
        app.status_var = _Var("")
        app._discard_current_structure()

        assert app.edlc_cell is None
        assert "descartó" in app.edlc_status_var.get()
