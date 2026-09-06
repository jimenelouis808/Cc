"""The Tk-free GUI logic, the plotting layer, and the command line.

The GUI's widget layer cannot be exercised without a display, so all its
logic lives in :mod:`ramancarbon.gui.state` and its drawing in
:mod:`ramancarbon.gui.plots`, both of which are tested here. What is left
untested is the Tk wiring itself.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from ramancarbon.cli.main import main  # noqa: E402
from ramancarbon.examples.demo_data import make_demo  # noqa: E402
from ramancarbon.gui.plots import (  # noqa: E402
    figure_for_report,
    plot_fit,
    plot_overlay,
    plot_rbm,
    plot_spectrum,
    plot_strain_doping,
)
from ramancarbon.gui.state import Session  # noqa: E402
from ramancarbon.gui.theme import (  # noqa: E402
    DARK,
    FONT_STACK,
    LIGHT,
    matplotlib_style,
    pick_font,
)


# -- theme -------------------------------------------------------------
@pytest.mark.parametrize("palette", [LIGHT, DARK])
def test_palettes_define_every_plot_role(palette):
    for role in ("data", "fitted", "residual", "baseline"):
        assert getattr(palette, role).startswith("#")
    assert len(palette.components) >= 4
    assert palette.component_colour(99) in palette.components


def test_font_picker_falls_back_safely():
    assert pick_font(["DejaVu Sans"], FONT_STACK) == "DejaVu Sans"
    assert pick_font(["Nothing Installed"], FONT_STACK) == "TkDefaultFont"


@pytest.mark.parametrize("palette", [LIGHT, DARK])
def test_matplotlib_style_is_complete(palette):
    style = matplotlib_style(palette)
    assert style["figure.facecolor"] == palette.surface
    with matplotlib.rc_context(style):
        Figure()


# -- session -----------------------------------------------------------
def test_session_analyses_a_batch_and_tabulates_it():
    session = Session()
    for kind in ("SWCNT", "MWCNT", "grafeno_1L"):
        session.add(make_demo(kind, seed=1))
    ok, bad = session.analyse_all()
    assert (ok, bad) == (3, 0)
    columns, rows = session.results_table()
    assert len(rows) == 3
    for required in ("nombre", "laser_nm", "material", "ID_IG"):
        assert required in columns


def test_results_table_columns_are_the_union_over_rows():
    """A batch mixing tubes and graphene shows the diameter column for the
    rows that have one, and blanks elsewhere."""
    session = Session()
    session.add(make_demo("SWCNT", seed=2))
    session.add(make_demo("grafeno_1L", seed=3))
    session.analyse_all()
    columns, rows = session.results_table()
    assert "d_RBM_nm" in columns
    assert all(len(row) == len(columns) for row in rows)


def test_export_refuses_when_nothing_was_analysed(tmp_path):
    session = Session()
    session.add(make_demo("SWCNT", seed=4))
    with pytest.raises(ValueError, match="analiza"):
        session.export_table(tmp_path / "x.csv")


def test_export_writes_a_header_and_a_row_per_spectrum(tmp_path):
    session = Session()
    session.add(make_demo("SWCNT", seed=5))
    session.analyse_all()
    path = session.export_table(tmp_path / "out.csv")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_changing_the_laser_invalidates_derived_results():
    """Band windows, dispersion and the crystallite formula all depend on
    it, so stale results must be dropped rather than shown."""
    session = Session()
    session.add(make_demo("MWCNT", seed=6))
    session.analyse_all()
    assert session.active.result is not None
    session.set_laser(633.0)
    assert session.active.result is None
    assert session.active.raw.laser_nm == 633.0


def test_load_collects_failures_instead_of_aborting(tmp_path):
    good = tmp_path / "ok.txt"
    good.write_text("Laser: 532 nm\n100 1\n101 2\n102 3\n", encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("no soy un espectro\n", encoding="utf-8")
    session = Session()
    assert session.load([good, bad]) == 1
    assert any(level == "error" for level, _ in session.messages)


def test_missing_laser_is_warned_about(tmp_path):
    path = tmp_path / "sin_laser.txt"
    path.write_text("100 1\n101 2\n102 3\n", encoding="utf-8")
    session = Session()
    session.load([path])
    assert any("láser" in text for _, text in session.messages)


def test_preset_specs_are_editable_starting_points():
    session = Session()
    session.add(make_demo("MWCNT", seed=7))
    specs = session.preset_specs("three_band")
    assert [s.name for s in specs] == ["D", "G", "D'"]
    assert all(s.centre_bounds for s in specs)


def test_manual_fit_runs_from_edited_specs():
    session = Session()
    session.add(make_demo("MWCNT", seed=8))
    session.preprocess_active()
    specs = session.preset_specs("three_band")
    result = session.fit_manual(specs, (1100.0, 1750.0), "linear")
    assert result is not None and result.r_squared > 0.95


def test_preprocess_settings_translate_to_kwargs():
    session = Session()
    session.preprocess_settings.baseline_method = "polynomial"
    session.preprocess_settings.baseline_order = 4
    kwargs = session.preprocess_settings.to_kwargs()
    assert kwargs["baseline_method"] == "polynomial"
    assert kwargs["baseline_kwargs"]["order"] == 4
    session.preprocess_settings.baseline_method = "none"
    assert session.preprocess_settings.to_kwargs()["baseline_method"] is None


# -- plots -------------------------------------------------------------
@pytest.mark.parametrize("kind", ["SWCNT", "MWCNT", "grafeno_1L", "GO"])
@pytest.mark.parametrize("palette", [LIGHT, DARK])
def test_summary_figure_renders(kind, palette, tmp_path):
    from ramancarbon.analysis.report import analyse

    result = analyse(make_demo(kind, seed=9))
    with matplotlib.rc_context(matplotlib_style(palette)):
        figure = figure_for_report(result, palette)
        figure.savefig(tmp_path / f"{kind}_{palette.name}.png")
    assert (tmp_path / f"{kind}_{palette.name}.png").stat().st_size > 5000


def test_individual_plots_handle_missing_data():
    """Every panel must render an explanatory empty state rather than
    raising when the spectrum lacks what it needs."""
    from ramancarbon.analysis.report import analyse

    result = analyse(make_demo("grafeno_1L", seed=10, low=800.0))
    figure = Figure()
    plot_rbm(figure.add_subplot(221), result, LIGHT)
    plot_strain_doping(figure.add_subplot(222), result, LIGHT)
    plot_spectrum(figure.add_subplot(223), result.processed, LIGHT)
    plot_overlay(figure.add_subplot(224), [result.processed], LIGHT)


def test_fit_plot_draws_a_residual_panel():
    from ramancarbon.analysis.report import analyse

    result = analyse(make_demo("MWCNT", seed=11))
    figure = Figure()
    main = figure.add_subplot(211)
    residual = figure.add_subplot(212)
    plot_fit(main, result.fit, LIGHT, residual_axes=residual)
    assert residual.lines


# -- cli ---------------------------------------------------------------
def test_cli_demo_then_analyse(tmp_path, capsys):
    assert main(["demo", str(tmp_path)]) == 0
    files = sorted(tmp_path.glob("*.txt"))
    assert len(files) >= 5
    capsys.readouterr()

    assert main(["analizar", str(files[0]), "--laser", "532"]) == 0
    out = capsys.readouterr().out
    assert "IDENTIFICACIÓN" in out


def test_cli_batch_writes_a_csv(tmp_path, capsys):
    data = tmp_path / "datos"
    assert main(["demo", str(data), "--material", "MWCNT"]) == 0
    csv = tmp_path / "res.csv"
    assert main(["lote", str(data), "--laser", "532", "--csv", str(csv)]) == 0
    lines = csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert "material" in lines[0]


def test_cli_deconvolution_comparison(tmp_path, capsys):
    assert main(["demo", str(tmp_path), "--material", "MWCNT"]) == 0
    capsys.readouterr()
    path = next(tmp_path.glob("*.txt"))
    assert main(["deconvolucionar", str(path), "--comparar"]) == 0
    assert "Comparación de modelos" in capsys.readouterr().out


def test_cli_database_queries(capsys):
    assert main(["bd"]) == 0
    assert "bandas" in capsys.readouterr().out
    assert main(["bd", "--banda", "2D", "--laser", "785"]) == 0
    out = capsys.readouterr().out
    assert "2614" in out or "2615" in out  # dispersion-corrected position
    assert main(["bd", "--rbm"]) == 0
    assert "234" in capsys.readouterr().out
    assert main(["bd", "--materiales"]) == 0
    assert "SWCNT" in capsys.readouterr().out


def test_cli_saves_a_report_and_a_figure(tmp_path, capsys):
    assert main(["demo", str(tmp_path), "--material", "SWCNT"]) == 0
    capsys.readouterr()
    path = next(tmp_path.glob("*.txt"))
    report = tmp_path / "informe.txt"
    figure = tmp_path / "figura.png"
    assert main(["analizar", str(path), "--laser", "532",
                 "--salida", str(report), "--figura", str(figure)]) == 0
    assert report.read_text(encoding="utf-8")
    assert figure.stat().st_size > 5000


def test_cli_missing_file_is_an_error(capsys):
    assert main(["analizar", "/no/existe.txt"]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_control_spectrum(tmp_path, capsys):
    from ramancarbon.core.io import write_spectrum
    from ramancarbon.examples.demo_data import add_doping

    pristine = make_demo("grafeno_1L", seed=12)
    doped = add_doping(pristine, delta_g=8.0, delta_2d=-12.0)
    a = write_spectrum(pristine, tmp_path / "pristino.txt")
    b = write_spectrum(doped, tmp_path / "dopado.txt")
    assert main(["analizar", str(b), "--laser", "532", "--control", str(a)]) == 0
    assert "control del usuario" in capsys.readouterr().out


def test_gui_package_imports_without_tkinter():
    """Importing must not require Tk; only launching does."""
    import ramancarbon.gui as gui

    assert callable(gui.main)
