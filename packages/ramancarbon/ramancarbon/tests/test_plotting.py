"""The plot engine: style, transforms, series modifiers, rendering, export."""

from __future__ import annotations

import json
import math

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ramancarbon.plotting.engine import (  # noqa: E402
    Inset,
    Plot,
    _sqrt_ticks,
    _tidy,
)
from ramancarbon.plotting.layout import PanelFigure, grid, panel_label  # noqa: E402
from ramancarbon.plotting.series import (  # noqa: E402
    Series,
    from_pattern,
    from_spectrum,
    stack_offsets,
)
from ramancarbon.plotting.style import (  # noqa: E402
    COLUMN_WIDTHS,
    MARKERS,
    OKABE_ITO,
    PRESETS,
    AxisStyle,
    LegendStyle,
    PlotStyle,
    SecondaryAxis,
    journal,
    preset,
)
from ramancarbon.plotting.transforms import (  # noqa: E402
    TRANSFORMS,
    secondary_ticks,
)


@pytest.fixture
def curves() -> list[Series]:
    x = np.linspace(100.0, 2000.0, 400)
    out = []
    for index, (centre, height) in enumerate([(1350, 100.0), (1580, 250.0), (2700, 40.0)]):
        y = height * np.exp(-0.5 * ((x - centre) / 45.0) ** 2) + 2.0
        out.append(Series(x=x, y=y, label=f"m{index}"))
    return out


# -- style -------------------------------------------------------------

def test_every_preset_builds_and_names_itself():
    for name in PRESETS:
        style = preset(name)
        assert style.width_in > 0 and style.height_in > 0
        assert style.dpi >= 100


def test_unknown_preset_says_which_ones_exist():
    with pytest.raises(ValueError, match="preajuste desconocido"):
        preset("science")


def test_journal_presets_use_the_published_column_width():
    for name, (single, double) in COLUMN_WIDTHS.items():
        assert journal(name).width_in == pytest.approx(single)
        assert journal(name, double=True).width_in == pytest.approx(double)


def test_journal_preset_is_reachable_through_preset_by_name():
    assert preset("acs").width_in == pytest.approx(COLUMN_WIDTHS["acs"][0])
    assert preset("acs-doble").width_in == pytest.approx(COLUMN_WIDTHS["acs"][1])


def test_style_survives_a_round_trip_through_json():
    style = PlotStyle(
        width_in=3.25, palette="grises", line_width=2.0, cycle_markers=True,
        x=AxisStyle(label="2θ (°)", limits=(10.0, 80.0), minor_ticks=5),
        y=AxisStyle(label="Intensidad", scale="sqrt", invert=False),
        secondary_x=SecondaryAxis(kind="d", parameter=1.5406),
        legend=LegendStyle(outside=True, columns=2),
        normalise="max", offset=0.3,
    )
    again = PlotStyle.from_dict(json.loads(json.dumps(style.to_dict())))
    assert again == style


def test_from_dict_ignores_fields_a_newer_version_wrote():
    payload = PlotStyle().to_dict()
    payload["holografia"] = True
    payload["x"]["profundidad"] = 3
    assert PlotStyle.from_dict(payload) == PlotStyle()


def test_from_dict_fills_in_fields_an_older_version_lacked():
    payload = PlotStyle(line_width=3.0).to_dict()
    del payload["marker_size"]
    del payload["legend"]
    rebuilt = PlotStyle.from_dict(payload)
    assert rebuilt.line_width == 3.0
    assert rebuilt.marker_size == PlotStyle().marker_size
    assert rebuilt.legend == LegendStyle()


def test_font_sizes_fall_back_to_the_base_size():
    style = PlotStyle(font_size=9.0, tick_size=7.0)
    assert style.resolved("tick_size") == 7.0
    assert style.resolved("label_size") == 9.0


def test_colours_and_markers_cycle_without_running_out():
    style = PlotStyle(cycle_markers=True)
    assert style.colour(0) == OKABE_ITO[0]
    assert style.colour(len(OKABE_ITO)) == OKABE_ITO[0]
    assert style.marker_for(len(MARKERS) + 1) == MARKERS[1]


def test_markers_only_cycle_when_asked():
    assert PlotStyle(marker="s").marker_for(3) == "s"


def test_greyscale_preset_carries_no_colour():
    style = preset("grises")
    assert all(c.startswith("#") and c[1:3] == c[3:5] == c[5:7] for c in style.colours)
    assert style.cycle_dashes


# -- transforms --------------------------------------------------------

@pytest.mark.parametrize("key", sorted(TRANSFORMS))
def test_each_transform_inverts_itself(key):
    transform = TRANSFORMS[key]
    parameter = {"d": 1.5406, "q": 1.5406, "lambda": 532.0, "referencia": 0.197}.get(key, 0.0)
    values = np.linspace(5.0, 80.0, 25)
    there = transform.forward(values, parameter)
    back = transform.inverse(there, parameter)
    assert np.allclose(back, values, rtol=1e-9, atol=1e-9)


def test_bragg_transform_matches_the_law_it_implements():
    d = TRANSFORMS["d"].forward(np.array([28.44]), 1.5406)[0]
    assert d == pytest.approx(1.5406 / (2 * math.sin(math.radians(28.44 / 2))), rel=1e-12)


def test_q_and_d_agree_with_each_other():
    two_theta = np.array([20.0, 45.0, 70.0])
    d = TRANSFORMS["d"].forward(two_theta, 1.5406)
    q = TRANSFORMS["q"].forward(two_theta, 1.5406)
    assert np.allclose(q, 2 * math.pi / d)


def test_raman_shift_to_wavelength_puts_the_laser_line_at_zero():
    assert TRANSFORMS["lambda"].forward(np.array([0.0]), 532.0)[0] == pytest.approx(532.0)
    stokes = TRANSFORMS["lambda"].forward(np.array([1580.0]), 532.0)[0]
    assert stokes > 532.0
    anti = TRANSFORMS["lambda"].forward(np.array([-1580.0]), 532.0)[0]
    assert anti < 532.0


def test_secondary_ticks_are_round_in_the_secondary_units():
    positions, labels = secondary_ticks(TRANSFORMS["d"], (10.0, 80.0), 1.5406)
    assert positions and len(positions) == len(labels)
    for value in labels:
        assert value == pytest.approx(round(value, 1))
    for position, value in zip(positions, labels):
        assert 10.0 <= position <= 80.0
        assert TRANSFORMS["d"].forward(np.array([position]), 1.5406)[0] == pytest.approx(value)


def test_secondary_ticks_never_leave_the_visible_range():
    positions, _ = secondary_ticks(TRANSFORMS["q"], (30.0, 50.0), 1.5406)
    assert positions
    assert all(30.0 <= p <= 50.0 for p in positions)


def test_secondary_ticks_thin_out_labels_that_would_overprint():
    positions, _ = secondary_ticks(TRANSFORMS["d"], (10.0, 80.0), 1.5406)
    span = 70.0
    assert all(b - a >= span / (6 * 1.6) for a, b in zip(positions, positions[1:]))


def test_an_explicit_tick_list_is_drawn_as_asked():
    # no thinning, even where the values crowd

    positions, labels = secondary_ticks(
        TRANSFORMS["d"], (10.0, 80.0), 1.5406, explicit=[3.0, 2.0, 1.5]
    )
    assert labels == [3.0, 2.0, 1.5]


def test_secondary_ticks_on_a_transform_without_a_preferred_list():
    positions, labels = secondary_ticks(TRANSFORMS["mev"], (200.0, 1800.0), 0.0)
    assert len(labels) >= 3
    step = labels[1] - labels[0]
    assert all(b - a == pytest.approx(step) for a, b in zip(labels, labels[1:]))


# -- series ------------------------------------------------------------

def test_mismatched_x_and_y_are_refused_with_the_series_name():
    with pytest.raises(ValueError, match="rota"):
        Series(x=[1, 2, 3], y=[1, 2], label="rota")


def test_an_error_bar_must_match_the_data():
    with pytest.raises(ValueError, match="error"):
        Series(x=[1, 2, 3], y=[1, 2, 3], error=[0.1, 0.1])


@pytest.mark.parametrize(
    "mode,expected_max",
    [("max", 1.0), ("minmax", 1.0), ("0-100", 100.0)],
)
def test_normalisations_reach_their_stated_top(curves, mode, expected_max):
    y = curves[1].normalised(mode)
    assert float(np.max(y)) == pytest.approx(expected_max)


def test_minmax_normalisation_also_puts_the_floor_at_zero(curves):
    assert float(np.min(curves[0].normalised("minmax"))) == pytest.approx(0.0)


def test_area_normalisation_integrates_to_one(curves):
    from ramancarbon.core.compat import trapezoid

    item = curves[1]
    y = item.normalised("area")
    assert float(trapezoid(y, item.x)) == pytest.approx(1.0)


def test_a_flat_series_normalises_without_dividing_by_zero():
    flat = Series(x=[1.0, 2.0, 3.0], y=[5.0, 5.0, 5.0])
    assert np.allclose(flat.normalised("minmax"), 0.0)
    assert np.allclose(flat.normalised("max"), 1.0)


def test_an_unknown_normalisation_is_an_error(curves):
    with pytest.raises(ValueError, match="normalización desconocida"):
        curves[0].normalised("logaritmica")


def test_drawn_applies_normalisation_then_scale_then_shifts(curves):
    item = Series(x=curves[0].x, y=curves[0].y, scale=2.0, shift=0.5, x_shift=-10.0)
    x, y = item.drawn("max", 3.0)
    assert np.allclose(x, item.x - 10.0)
    assert float(np.max(y)) == pytest.approx(2.0 + 0.5 + 3.0)


def test_the_legend_admits_a_magnified_curve():
    assert Series(x=[0], y=[0], label="2D", scale=5.0).display_label() == "2D (×5)"
    assert Series(x=[0], y=[0], label="2D", scale=0.2).display_label() == "2D (×1/5)"
    assert Series(x=[0], y=[0], label="2D").display_label() == "2D"


def test_stack_offsets_step_by_a_fraction_of_the_tallest_curve(curves):
    offsets = stack_offsets(curves, 0.25)
    span = float(np.max(curves[1].y) - np.min(curves[1].y))
    assert offsets[0] == 0.0
    assert offsets[1] == pytest.approx(0.25 * span)
    assert offsets[2] == pytest.approx(0.50 * span)


def test_stack_offsets_measure_the_span_after_normalisation(curves):
    offsets = stack_offsets(curves, 0.5, normalisation="max")
    assert offsets[1] == pytest.approx(0.5, rel=0.02)


def test_an_absolute_step_overrides_the_fraction(curves):
    assert stack_offsets(curves, 0.25, absolute=10.0) == [0.0, 10.0, 20.0]


def test_no_offset_requested_means_no_offset(curves):
    assert stack_offsets(curves, 0.0) == [0.0, 0.0, 0.0]


def test_a_series_round_trips_with_its_data():
    item = Series(x=[1.0, 2.0], y=[3.0, 4.0], label="a", kind="scatter", marker="s")
    again = Series.from_dict(json.loads(json.dumps(item.to_dict(include_data=True))))
    assert again.label == "a" and again.kind == "scatter" and again.marker == "s"
    assert np.allclose(again.x, item.x) and np.allclose(again.y, item.y)


def test_a_series_saved_without_data_keeps_its_style():
    payload = Series(x=[1.0], y=[2.0], label="a", colour="#ff0000").to_dict()
    assert "x" not in payload and "y" not in payload
    assert Series.from_dict(payload).colour == "#ff0000"


def test_series_from_a_spectrum_carries_the_laser():
    from ramancarbon.examples.demo_data import make_demo

    spectrum = make_demo("MWCNT")
    item = from_spectrum(spectrum)
    assert item.metadata["laser_nm"] == spectrum.laser_nm
    assert np.allclose(item.x, spectrum.shift)


def test_series_from_a_pattern_carries_the_wavelength():
    from ramancarbon.examples.demo_data import make_xrd_demo

    pattern = make_xrd_demo("CNT_FeSe")
    item = from_pattern(pattern)
    assert item.metadata["wavelength"] == pytest.approx(pattern.wavelength)
    assert np.allclose(item.x, pattern.two_theta)


# -- rendering ---------------------------------------------------------

def _draw(plot: Plot):
    figure, ax = plt.subplots()
    try:
        plot.draw(ax)
        figure.canvas.draw()
        return ax
    finally:
        plt.close(figure)


def test_a_plot_draws_one_line_per_visible_series(curves):
    plot = Plot(series=list(curves))
    plot.series[2].visible = False
    figure, ax = plt.subplots()
    plot.draw(ax)
    assert len(ax.get_lines()) == 2
    plt.close(figure)


@pytest.mark.parametrize(
    "kind", ["line", "scatter", "step", "bar", "stick", "errorbar"]
)
def test_every_series_kind_renders(curves, kind):
    item = Series(
        x=curves[0].x[:40], y=curves[0].y[:40], label=kind, kind=kind,
        error=np.full(40, 1.0) if kind == "errorbar" else None,
    )
    figure, ax = plt.subplots()
    Plot(series=[item]).draw(ax)
    figure.canvas.draw()
    plt.close(figure)


def test_the_style_reaches_the_line(curves):
    style = PlotStyle(line_width=3.3, alpha=0.5, palette="brillante")
    figure, ax = plt.subplots()
    Plot(series=[curves[0]], style=style).draw(ax)
    line = ax.get_lines()[0]
    assert line.get_linewidth() == pytest.approx(3.3)
    assert line.get_alpha() == pytest.approx(0.5)
    plt.close(figure)


def test_a_series_overrides_the_style(curves):
    item = Series(x=curves[0].x, y=curves[0].y, colour="#123456", line_width=0.4)
    figure, ax = plt.subplots()
    Plot(series=[item], style=PlotStyle(line_width=3.0)).draw(ax)
    line = ax.get_lines()[0]
    assert line.get_linewidth() == pytest.approx(0.4)
    assert matplotlib.colors.to_hex(line.get_color()) == "#123456"
    plt.close(figure)


def test_axis_labels_limits_and_inversion_are_applied(curves):
    style = PlotStyle(
        x=AxisStyle(label="Desplazamiento (cm⁻¹)", limits=(1200.0, 1700.0), invert=True),
        y=AxisStyle(label="Intensidad (u.a.)"),
        title="Grafito",
    )
    figure, ax = plt.subplots()
    Plot(series=list(curves), style=style).draw(ax)
    assert ax.get_xlabel().startswith("Desplazamiento")
    assert ax.get_ylabel().startswith("Intensidad")
    assert ax.get_title() == "Grafito"
    assert ax.get_xlim() == (1700.0, 1200.0)
    plt.close(figure)


def test_only_one_end_of_a_limit_can_be_given(curves):
    style = PlotStyle(x=AxisStyle(limits=(1000.0, None)))
    figure, ax = plt.subplots()
    Plot(series=list(curves), style=style).draw(ax)
    assert ax.get_xlim()[0] == pytest.approx(1000.0)
    assert ax.get_xlim()[1] > 1900.0
    plt.close(figure)


def test_a_log_axis_is_a_log_axis(curves):
    style = PlotStyle(y=AxisStyle(scale="log"))
    figure, ax = plt.subplots()
    Plot(series=[curves[0]], style=style).draw(ax)
    assert ax.get_yscale() == "log"
    plt.close(figure)


def test_the_square_root_scale_puts_its_ticks_at_round_roots():
    ticks = _sqrt_ticks(0.0, 10000.0)
    assert ticks
    roots = [math.sqrt(t) for t in ticks]
    step = roots[1] - roots[0]
    assert all(b - a == pytest.approx(step) for a, b in zip(roots, roots[1:]))
    assert all(r == pytest.approx(round(r, 6)) for r in roots)


def test_the_square_root_scale_is_used_when_asked(curves):
    style = PlotStyle(y=AxisStyle(scale="sqrt"))
    figure, ax = plt.subplots()
    Plot(series=[curves[1]], style=style).draw(ax)
    figure.canvas.draw()
    assert ax.get_yscale() == "function"
    ticks = [t for t in ax.get_yticks() if 0 <= t <= float(np.max(curves[1].y))]
    assert len(ticks) >= 3
    plt.close(figure)


def test_markers_bands_and_annotations_reach_the_axes(curves):
    plot = Plot(series=[curves[0]])
    plot.mark(1350.0, 1580.0, label="D")
    plot.shade(1300.0, 1400.0, label="banda D")
    plot.annotate(1580.0, 200.0, "G")
    figure, ax = plt.subplots()
    plot.draw(ax)
    assert len(ax.lines) >= 3            # the curve plus two rules
    assert len(ax.patches) >= 1
    assert any(t.get_text() == "G" for t in ax.texts)
    plt.close(figure)


def test_a_horizontal_marker_goes_on_the_other_axis(curves):
    plot = Plot(series=[curves[0]])
    plot.mark(50.0, axis="y")
    figure, ax = plt.subplots()
    plot.draw(ax)
    plt.close(figure)


def test_peak_labels_appear_when_asked(curves):
    style = PlotStyle(annotate_peaks=True, peak_label_format="{:.0f}")
    figure, ax = plt.subplots()
    Plot(series=[curves[1]], style=style).draw(ax)
    labels = [t.get_text() for t in ax.texts]
    assert any(abs(float(text) - 1580.0) < 20 for text in labels if text)
    plt.close(figure)


def test_a_secondary_axis_is_drawn_with_readable_ticks(curves):
    style = PlotStyle(
        x=AxisStyle(label="2θ (°)", limits=(10.0, 80.0)),
        secondary_x=SecondaryAxis(kind="d", parameter=1.5406),
    )
    x = np.linspace(10.0, 80.0, 500)
    plot = Plot(series=[Series(x=x, y=np.ones_like(x))], style=style)
    figure, ax = plt.subplots()
    plot.draw(ax)
    figure.canvas.draw()
    twins = [child for child in figure.axes if child is not ax]
    assert twins, "no se creó el eje secundario"
    assert twins[0].get_xlabel().startswith("d")
    plt.close(figure)


def test_the_legend_shows_the_labels_that_were_given(curves):
    figure, ax = plt.subplots()
    Plot(series=list(curves)).draw(ax)
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["m0", "m1", "m2"]
    plt.close(figure)


def test_an_unlabelled_figure_gets_no_legend():
    figure, ax = plt.subplots()
    Plot(series=[Series(x=[1.0, 2.0], y=[1.0, 2.0])]).draw(ax)
    assert ax.get_legend() is None
    plt.close(figure)


def test_the_legend_can_be_switched_off(curves):
    style = PlotStyle(legend=LegendStyle(show=False))
    figure, ax = plt.subplots()
    Plot(series=list(curves), style=style).draw(ax)
    assert ax.get_legend() is None
    plt.close(figure)


def test_an_inset_does_not_move_the_main_axes(curves):
    style = PlotStyle(x=AxisStyle(limits=(100.0, 2000.0)))
    plot = Plot(
        series=[curves[1]], style=style,
        inset=Inset(x_limits=(1500.0, 1660.0)),
    )
    figure, ax = plt.subplots()
    plot.draw(ax)
    figure.canvas.draw()
    assert ax.get_xlim() == pytest.approx((100.0, 2000.0))
    plt.close(figure)


def test_an_empty_plot_still_draws():
    figure, ax = plt.subplots()
    Plot().draw(ax)
    plt.close(figure)


# -- export ------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".png", ".pdf", ".svg", ".eps"])
def test_a_figure_saves_in_every_offered_format(curves, tmp_path, suffix):
    path = Plot(series=list(curves)).save(tmp_path / f"figura{suffix}")
    assert path.exists() and path.stat().st_size > 0


def test_an_unknown_format_is_refused_by_name(curves, tmp_path):
    with pytest.raises(ValueError):
        Plot(series=list(curves)).save(tmp_path / "figura.docx")


def test_saving_creates_the_folder(curves, tmp_path):
    path = Plot(series=[curves[0]]).save(tmp_path / "sub" / "carpeta" / "f.png")
    assert path.exists()


def test_the_exported_table_is_what_was_drawn(curves):
    style = PlotStyle(normalise="max", offset=0.5)
    plot = Plot(series=list(curves), style=style)
    columns, rows = plot.data_table()
    assert columns == ["m0_x", "m0_y", "m1_x", "m1_y", "m2_x", "m2_y"]
    offsets = plot.offsets()
    for index, item in enumerate(plot.series):
        _, y = item.drawn("max", offsets[index])
        assert rows[0][2 * index + 1] == pytest.approx(float(y[0]))


def test_an_invisible_series_is_not_exported(curves):
    plot = Plot(series=list(curves))
    plot.series[1].visible = False
    columns, _ = plot.data_table()
    assert columns == ["m0_x", "m0_y", "m2_x", "m2_y"]


def test_series_of_different_lengths_pad_rather_than_truncate():
    plot = Plot(series=[
        Series(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0], label="larga"),
        Series(x=[1.0], y=[9.0], label="corta"),
    ])
    columns, rows = plot.data_table()
    assert len(rows) == 3
    assert math.isnan(rows[2][3])


def test_the_data_file_is_readable_text(curves, tmp_path):
    plot = Plot(series=list(curves), name="grafito")
    path = plot.save_data(tmp_path / "datos.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# grafito"
    assert lines[1].split(",")[0] == "m0_x"
    assert len(lines) == 2 + len(curves[0].x)


def test_an_empty_plot_exports_an_empty_table():
    assert Plot().data_table() == ([], [])


def test_a_whole_figure_round_trips_through_json(curves):
    plot = Plot(series=list(curves), name="grafito")
    plot.style = preset("acs").replace(normalise="max", offset=0.4)
    plot.mark(1580.0, label="G")
    plot.shade(1300.0, 1400.0)
    plot.annotate(1580.0, 1.0, "G")
    plot.inset = Inset(x_limits=(1500.0, 1660.0))
    payload = json.loads(json.dumps(plot.to_dict(include_data=True)))
    again = Plot.from_dict(payload)
    assert again.name == "grafito"
    assert again.style == plot.style
    assert [s.label for s in again.series] == ["m0", "m1", "m2"]
    assert again.markers[0].label == "G"
    assert again.bands[0].low == 1300.0
    assert again.annotations[0].text == "G"
    assert again.inset.x_limits == (1500.0, 1660.0)
    assert again.data_table() == plot.data_table()


def test_tick_labels_lose_their_trailing_zeros():
    assert _tidy(2.0) == "2"
    assert _tidy(2.5) == "2.5"
    assert _tidy(0.0) == "0"
    assert _tidy(15000.0) == "1.5e+04"


# -- panels ------------------------------------------------------------

def test_panel_labels_follow_the_chosen_style():
    assert [panel_label(i) for i in range(3)] == ["(a)", "(b)", "(c)"]
    assert panel_label(0, "a)") == "a)"
    assert panel_label(1, "A") == "B"


def test_a_grid_fills_rows_first():
    plots = [Plot() for _ in range(5)]
    figure = grid(plots, columns=2)
    assert (figure.rows, figure.columns) == (3, 2)


def test_a_panel_figure_draws_one_axes_per_plot(curves, tmp_path):
    panels = grid([Plot(series=[c], name=c.label) for c in curves], columns=2)
    figure = panels.figure()
    try:
        assert len([a for a in figure.axes if a.get_subplotspec() is not None]) >= 3
        texts = {t.get_text() for a in figure.axes for t in a.texts}
        assert {"(a)", "(b)", "(c)"} <= texts
    finally:
        plt.close(figure)


def test_a_shared_style_does_not_overwrite_each_panels_axes(curves):
    raman = Plot(series=[curves[0]], style=PlotStyle(x=AxisStyle(label="cm⁻¹")))
    xrd = Plot(series=[curves[1]], style=PlotStyle(x=AxisStyle(label="2θ (°)")))
    panels = grid([raman, xrd], columns=2, style=preset("acs").replace(line_width=2.5))
    figure = panels.figure()
    try:
        axes = [a for a in figure.axes if a.get_subplotspec() is not None]
        assert axes[0].get_xlabel() == "cm⁻¹"
        assert axes[1].get_xlabel() == "2θ (°)"
        assert axes[0].get_lines()[0].get_linewidth() == pytest.approx(2.5)
    finally:
        plt.close(figure)


def test_a_panel_figure_saves_and_exports_one_file_per_panel(curves, tmp_path):
    panels = grid([Plot(series=[c], name=c.label) for c in curves], columns=2)
    panels.name = "resumen"
    image = panels.save(tmp_path / "panel.png")
    files = panels.save_data(tmp_path / "datos")
    assert image.exists()
    assert [f.name for f in files] == ["resumen_a.csv", "resumen_b.csv", "resumen_c.csv"]
    assert all(f.exists() for f in files)


def test_a_panel_figure_round_trips_through_json(curves):
    panels = grid(
        [Plot(series=[c], name=c.label) for c in curves],
        columns=2, style=preset("nature"),
    )
    panels.share_x = True
    again = PanelFigure.from_dict(json.loads(json.dumps(panels.to_dict(include_data=True))))
    assert (again.rows, again.columns) == (panels.rows, panels.columns)
    assert again.share_x is True
    assert again.style == panels.style
    assert [p.name for p in again.plots] == ["m0", "m1", "m2"]
