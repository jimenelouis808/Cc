"""Tests for vibspec phase 3: broadening, FTIR import, band matching and the figure.

Synthetic spectra with known answers throughout: bands at chosen positions,
a computed spectrum that is the experiment divided by a known scale factor.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from carbonforge.cli.main import main as cli_main
from carbonforge.results.spectra import VibrationalMode, VibrationalSpectrum, broaden
from carbonforge.tests.test_vibspec_workflow import _WATER_CHARGES, _water, springs_factory
from carbonforge.vibspec.core import (
    CalcSpec,
    computed_curve,
    export_csv,
    find_bands,
    fit_scale_factor,
    match_bands,
    match_table,
    normalise,
    plot_ir_comparison,
    prepare,
    prepare_experiment,
    read_ftir,
    rubberband_baseline,
    run,
    search_scale_factor,
    to_absorbance,
)

BANDS = np.array([1000.0, 1600.0, 3000.0])
SCALE = 0.97


def _spectrum(frequencies, intensities) -> VibrationalSpectrum:
    modes = [VibrationalMode(index=i, frequency_cm1=float(f), ir_activity=float(a))
             for i, (f, a) in enumerate(zip(frequencies, intensities, strict=True))]
    return VibrationalSpectrum(modes=modes, has_ir=True, expected_acoustic=0)


def _computed() -> VibrationalSpectrum:
    """Raw frequencies that land on BANDS once multiplied by SCALE, plus a weak mode."""
    return _spectrum(list(BANDS / SCALE) + [2200.0], [1.0, 0.6, 0.3, 0.01])


def _fwhm(x, y):
    above = x[y >= y.max() / 2]
    return above.max() - above.min()


def _write_ftir(path, header, rows, delimiter=",", decimal="."):
    lines = list(header)
    for x, y in rows:
        text = f"{x:.2f}{delimiter}{y:.5f}"
        lines.append(text.replace(".", decimal) if decimal != "." else text)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _synthetic_absorbance(x):
    y = np.zeros_like(x)
    for centre, height in zip(BANDS, (1.0, 0.6, 0.3), strict=True):
        y += height * np.exp(-0.5 * ((x - centre) / 10.0) ** 2)
    return y


class TestBroadening:
    @pytest.mark.parametrize("profile", ["lorentzian", "gaussian"])
    def test_area_and_width(self, profile):
        grid = np.linspace(1000, 3000, 40001)
        _, y = broaden([2000.0], [2.5], width_cm1=6.0, grid=grid, profile=profile)
        assert np.trapezoid(y, grid) == pytest.approx(2.5, rel=0.02 if profile == "lorentzian"
                                                      else 1e-4)
        assert _fwhm(grid, y) == pytest.approx(12.0, abs=0.1)

    def test_unknown_profile(self):
        with pytest.raises(ValueError, match="profile"):
            broaden([1000.0], [1.0], profile="voigt")

    @pytest.mark.parametrize("profile", ["lorentzian", "gaussian"])
    def test_computed_curve_width_is_fwhm_and_scale_moves_peak(self, profile):
        grid = np.linspace(500, 2500, 20001)
        spectrum = _spectrum([1000.0], [1.0])
        y = computed_curve(spectrum, grid, fwhm_cm1=20.0, profile=profile, scale_factor=1.5)
        assert grid[np.argmax(y)] == pytest.approx(1500.0, abs=0.2)
        assert _fwhm(grid, y) == pytest.approx(20.0, abs=0.2)

    def test_stored_frequencies_are_not_scaled(self):
        spectrum = _spectrum([1000.0], [1.0])
        computed_curve(spectrum, np.linspace(0, 2000, 100), scale_factor=0.9)
        assert spectrum.frequencies[0] == 1000.0


class TestReadFtir:
    def test_comma_csv_with_absorbance_header(self, tmp_path):
        rows = [(4000 - i, 0.1 * i) for i in range(5)]
        exp = read_ftir(_write_ftir(tmp_path / "a.csv", ["wavenumber,Absorbance"], rows))
        assert exp.quantity == "absorbance" and exp.quantity_source == "header"
        assert np.all(np.diff(exp.wavenumber) > 0)            # sorted ascending

    def test_semicolon_and_decimal_comma_percent_t(self, tmp_path):
        rows = [(1000 + i, 90.5 - i) for i in range(5)]
        exp = read_ftir(_write_ftir(tmp_path / "t.txt", ["Muestra X", "cm-1;%T"], rows,
                                    delimiter=";", decimal=","))
        assert exp.quantity == "transmittance" and exp.percent
        assert exp.wavenumber[0] == pytest.approx(1000.0)
        assert exp.values[0] == pytest.approx(90.5)
        assert exp.header == ["Muestra X", "cm-1;%T"]

    def test_whitespace_decimal_comma(self, tmp_path):
        path = tmp_path / "w.txt"
        path.write_text("1000,5 0,12\n1001,5 0,13\n1002,5 0,14\n", encoding="utf-8")
        exp = read_ftir(path)
        assert exp.wavenumber[0] == pytest.approx(1000.5)
        assert exp.values[-1] == pytest.approx(0.14)

    def test_guess_from_values_is_labelled_as_a_guess(self, tmp_path):
        rows = [(1000 + i, 80.0 + i) for i in range(5)]
        exp = read_ftir(_write_ftir(tmp_path / "g.txt", [], rows, delimiter="\t"))
        assert exp.quantity == "transmittance" and exp.quantity_source == "values"
        low = read_ftir(_write_ftir(tmp_path / "h.txt", [], [(1000 + i, 0.2) for i in range(5)]))
        assert low.quantity == "absorbance" and low.quantity_source == "values"

    def test_explicit_quantity_wins(self, tmp_path):
        rows = [(1000 + i, 80.0) for i in range(5)]
        exp = read_ftir(_write_ftir(tmp_path / "e.csv", ["x,absorbance"], rows),
                        quantity="transmittance")
        assert exp.quantity == "transmittance" and exp.quantity_source == "given"

    def test_extra_columns_ignored(self, tmp_path):
        path = tmp_path / "x.csv"
        path.write_text("1000,0.1,9\n1001,0.2,9\n1002,0.3,9\n", encoding="utf-8")
        assert read_ftir(path).values.tolist() == [0.1, 0.2, 0.3]

    def test_not_a_spectrum(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("hola\nesto no es un espectro\n", encoding="utf-8")
        with pytest.raises(ValueError, match="dos columnas"):
            read_ftir(path)

    def test_duplicate_wavenumbers(self, tmp_path):
        path = tmp_path / "d.csv"
        path.write_text("1000,0.1\n1000,0.2\n1001,0.3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="repetidos"):
            read_ftir(path)


class TestPreprocessing:
    def test_to_absorbance(self, tmp_path):
        rows = [(1000.0, 10.0), (1001.0, 100.0), (1002.0, 0.0)]
        exp = to_absorbance(read_ftir(_write_ftir(tmp_path / "t.csv", ["x,%T"], rows)))
        assert exp.quantity == "absorbance"
        assert exp.values.tolist() == pytest.approx([1.0, 0.0, 4.0])   # 0 %T clipped

    def test_fractional_transmittance(self, tmp_path):
        rows = [(1000.0, 0.1), (1001.0, 1.0), (1002.0, 0.5)]
        exp = to_absorbance(read_ftir(_write_ftir(tmp_path / "t.csv", ["x,transmittance"], rows)))
        assert exp.values[0] == pytest.approx(1.0)

    def test_rubberband_recovers_a_sloped_baseline(self):
        x = np.linspace(400, 4000, 3601)
        baseline = 0.2 + 1e-4 * x
        y = baseline + _synthetic_absorbance(x)
        found = rubberband_baseline(x, y)
        corrected = y - found
        assert corrected.min() >= -1e-9
        far = np.abs(x - 2300) < 100            # nowhere near a band
        assert np.allclose(found[far], baseline[far], atol=1e-3)

    def test_normalise(self):
        x = np.array([1.0, 2.0, 3.0])
        assert normalise(x, np.array([1.0, 4.0, 2.0])).max() == 1.0
        assert normalise(x, np.array([1.0, 4.0, 2.0]), window=(2.5, 3.5))[2] == 1.0
        with pytest.raises(ValueError):
            normalise(x, np.array([1.0, 4.0, 2.0]), window=(10, 20))

    def test_prepare_experiment(self, tmp_path):
        x = np.arange(4000, 399, -2.0)
        transmittance = 100 * 10 ** -(_synthetic_absorbance(x) + 0.05)
        path = _write_ftir(tmp_path / "s.txt", ["cm-1;%T"], zip(x, transmittance, strict=True),
                           delimiter=";", decimal=",")
        wx, wy = prepare_experiment(read_ftir(path))
        assert wy.max() == pytest.approx(1.0)
        assert wx[np.argmax(wy)] == pytest.approx(1000.0, abs=2.0)
        assert abs(wy[np.abs(wx - 2300) < 50]).max() < 1e-3        # baseline gone


class TestBands:
    def test_find_bands(self):
        x = np.linspace(400, 4000, 3601)
        found = find_bands(x, _synthetic_absorbance(x))
        assert found == pytest.approx(BANDS, abs=1.5)

    def test_match_with_the_right_scale(self):
        matches = match_bands(_computed(), BANDS, scale_factor=SCALE)
        assert [m.experimental_cm1 for m in matches] == list(BANDS)
        assert all(abs(m.delta_cm1) < 1e-6 for m in matches)
        assert all(m.mode_index != 3 for m in matches)               # the weak mode is ignored

    def test_unscaled_frequencies_miss_the_bands(self):
        # 3000/0.97 is 93 cm-1 away from 3000: the reason the search exists.
        matches = match_bands(_computed(), BANDS, scale_factor=1.0, tolerance_cm1=30.0)
        assert matches[-1].experimental_cm1 is None

    def test_search_recovers_the_scale_factor(self):
        factor, matches = search_scale_factor(_computed(), BANDS)
        assert factor == pytest.approx(SCALE, abs=1e-4)
        assert sum(m.experimental_cm1 is not None for m in matches) == 3

    def test_search_with_nothing_nearby(self):
        with pytest.raises(ValueError, match="Ningún factor"):
            search_scale_factor(_spectrum([500.0], [1.0]), [3500.0])

    def test_fit_needs_three_pairs(self):
        matches = match_bands(_computed(), BANDS[:2], scale_factor=SCALE)
        with pytest.raises(ValueError, match="al menos 3"):
            fit_scale_factor(matches)

    def test_fit_is_least_squares(self):
        matches = match_bands(_computed(), BANDS, scale_factor=SCALE)
        assert fit_scale_factor(matches, unscaled_by=SCALE) == pytest.approx(SCALE)

    def test_table(self):
        table = match_table(match_bands(_computed(), BANDS[:2], scale_factor=SCALE))
        assert "calc" in table and "—" in table and "propuesta" in table


class TestFigureAndExport:
    def test_figure(self, tmp_path):
        x = np.linspace(400, 4000, 3601)
        experiment = (x, _synthetic_absorbance(x))
        matches = match_bands(_computed(), BANDS, scale_factor=SCALE)
        figure, drawn = plot_ir_comparison(_computed(), title="prueba", experiment=experiment,
                                           scale_factor=SCALE, matches=matches)
        ax = figure.axes[0]
        assert ax.get_xlim()[0] > ax.get_xlim()[1]                   # FTIR convention
        assert drawn["computed"].max() == pytest.approx(1.0)
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any("0.970" in label for label in labels) and "FTIR" in labels
        figure.savefig(tmp_path / "f.png")
        assert (tmp_path / "f.png").stat().st_size > 0

    def test_figure_without_experiment(self):
        figure, drawn = plot_ir_comparison(_computed(), window=(900, 1100))
        # Unscaled, only the mode at 1000/0.97 = 1031 cm-1 is in the window.
        assert drawn["stick_positions"] == pytest.approx([1000.0 / SCALE])
        assert figure.axes[0].get_xlim() == (1100.0, 900.0)

    def test_export_csv(self, tmp_path):
        grid = np.array([1.0, 2.0])
        paths = export_csv(tmp_path / "out" / "r", grid, grid, grid, grid, (grid, grid))
        assert [p.name for p in paths] == ["r_calculado.csv", "r_barras.csv",
                                           "r_experimental.csv"]
        with paths[0].open(encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert rows[0][0].startswith("numero_de_onda") and len(rows) == 3


@pytest.fixture
def water_done(tmp_path):
    reference, start = _water()
    directory = tmp_path / "agua"
    prepare(start, CalcSpec(), directory)
    record = run(directory, springs_factory(reference, _WATER_CHARGES))
    return directory, record


class TestPlotCli:
    def test_plot_with_ftir_and_fitted_scale(self, water_done, tmp_path, capsys):
        directory, record = water_done
        computed = np.array(record.results["frequencies_cm1"])
        # The spring model's "stretch" sits near 4700 cm-1, hence the wide window.
        x = np.arange(5000, 399, -2.0)
        y = np.zeros_like(x)
        for centre in computed * 0.98:
            y += np.exp(-0.5 * ((x - centre) / 8.0) ** 2)
        ftir = _write_ftir(tmp_path / "ftir.csv", ["cm-1,absorbance"], zip(x, y, strict=True))
        out = tmp_path / "ir.png"
        code = cli_main(["vibspec", "plot", str(directory), "--ftir", str(ftir), "--fit-scale",
                         "--min-intensity", "0", "--xmax", "5000", "-o", str(out),
                         "--csv", str(tmp_path / "c")])
        text = capsys.readouterr().out
        assert code == 0 and out.exists()
        assert "Factor de escala ajustado: 0.98" in text
        assert (tmp_path / "c_experimental.csv").exists()

    def test_plot_without_ftir(self, water_done, tmp_path):
        directory, _ = water_done
        assert cli_main(["vibspec", "plot", str(directory), "-o", str(tmp_path / "a.png")]) == 0

    def test_missing_ftir_file(self, water_done, tmp_path, capsys):
        directory, _ = water_done
        assert cli_main(["vibspec", "plot", str(directory), "--ftir",
                         str(tmp_path / "nope.csv"), "-o", str(tmp_path / "a.png")]) == 1
        assert "No se pudo leer" in capsys.readouterr().out

    def test_unfinished_calculation(self, tmp_path):
        _, water = _water(perturb=False)
        prepare(water, CalcSpec(), tmp_path / "p")
        assert cli_main(["vibspec", "plot", str(tmp_path / "p"), "-o",
                         str(tmp_path / "a.png")]) == 1
