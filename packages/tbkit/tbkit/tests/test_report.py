"""tbkit.report: the page, the CSV and the figures come from the files, nothing else."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from tbkit import report as rp


def _small_report() -> rp.Report:
    rep = rp.Report("Prueba </script> título", meta={"origen": "test"})
    x = np.linspace(0, 10, 11)
    sec = rep.section("Curvas")
    sec.text("Un párrafo.")
    sec.items(["uno", "dos"])
    sec.add(rp.Figure("dos_curvas", "Dos curvas", "x (cm⁻¹)", "y", [
        rp.Series("a", x, x ** 2), rp.Series("b", x, 2 * x)]))
    sec.add(rp.Table("t", "Tabla", ["nombre", "valor"], [["uno", 1.5], ["dos", None]]))
    return rep


def test_html_is_self_contained_and_carries_the_data(tmp_path):
    path = rp.write_html(_small_report(), tmp_path / "sub" / "r.html")
    text = path.read_text(encoding="utf-8")
    assert "<title>Prueba </script> título</title>" not in text      # escaped in the title
    assert "http://" not in text.replace("http://www.w3.org/2000/svg", "")
    assert "https://" not in text                                   # opens without internet
    start = text.index('<script id="data" type="application/json">') + 42
    data = json.loads(text[start:text.index("</script>", start)].replace("<\\/", "</"))
    fig = data["sections"][0]["blocks"][2]
    assert fig["type"] == "figure" and fig["series"][0]["y"][-1] == 100.0
    assert data["sections"][0]["blocks"][3]["rows"][1] == ["dos", None]


def test_csv_export_round_trips(tmp_path):
    folder = rp.export_data(_small_report(), tmp_path / "datos")
    with (folder / "figura_dos_curvas.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["x (cm⁻¹)", "a", "b"]                        # shared x: one column
    assert np.allclose(np.array(rows[1:], float), np.column_stack(
        [np.linspace(0, 10, 11), np.linspace(0, 10, 11) ** 2, 2 * np.linspace(0, 10, 11)]))
    with (folder / "tabla_t.csv").open(encoding="utf-8-sig") as handle:
        assert list(csv.reader(handle)) == [["nombre", "valor"], ["uno", "1.5"], ["dos", ""]]
    index = json.loads((folder / "indice.json").read_text(encoding="utf-8"))
    assert {f["archivo"] for f in index["archivos"]} == {"figura_dos_curvas.csv", "tabla_t.csv"}


def test_journal_figures(tmp_path):
    pytest.importorskip("matplotlib")
    written = rp.export_figures(_small_report(), tmp_path / "fig", width="doble")
    assert sorted(p.suffix for p in written) == [".pdf", ".png", ".svg"]
    assert all(p.stat().st_size > 0 for p in written)
    assert rp._mpl("x (cm⁻¹)") == "x (cm$^{-1}$)"


def test_spectrum_file_and_cli(tmp_path):
    grid = np.linspace(100, 200, 51)
    np.savez(tmp_path / "s.npz", grid=grid, a=np.exp(-((grid - 150) / 5) ** 2),
             ir_grid=np.arange(3.0), b_ir=np.ones(3))
    rep = rp.recognise(tmp_path / "s.npz")
    ids = [f.id for f in rep.figures()]
    assert ids == ["grid", "ir_grid"]
    from tbkit.cli import main

    assert main(["report", str(tmp_path / "s.npz"), "--figuras", "no"]) == 0
    assert (tmp_path / "reporte_s.html").exists()
    assert (tmp_path / "reporte_s_datos" / "figura_grid.csv").exists()


def test_doped_raman_folder(tmp_path):
    """The adapter computes the G centroid and D estimate from the stored activities."""
    w = np.array([1300.0, 1600.0])
    entry = {"raman_frequencies_cm1": w.tolist(), "breathing_B": [0.5, 0.0],
             "activities_2.33": [1.0, 2.0],
             "ir": {"intensities_km_mol": [3.0, 4.0], "sum_rule_residual_e": 1e-4,
                    "strongest": [{"frequency_cm1": 1600.0, "intensity_km_mol": 4.0}]}}
    (tmp_path / "report.json").write_text(json.dumps({"lasers_eV": [2.33], "pristine": entry}))
    grid = np.linspace(900, 3700, 2801)
    np.savez(tmp_path / "spectra.npz", grid=grid, pristine_2_33=np.zeros_like(grid),
             **{"pristine_2.33": np.ones_like(grid), "pristine_2.33_breathing": np.ones_like(grid),
                "pristine_ir": np.ones(11)}, ir_grid=np.linspace(0, 3500, 11))
    rep = rp.recognise(tmp_path)
    table = next(t for t in rep.tables() if t.id == "raman_resumen")
    _, name, g, d, d_over_g, rel = table.rows[0]
    assert name == "sin dopar" and rel == 1.0
    assert (g, d) == pytest.approx((1600.0, 1300.0))
    nu = 2.33 * 8065.544
    sigma = [a * (nu - x) ** 4 / x / (1 - np.exp(-1.4387769 * x / 300))
             for a, x in zip((1.0, 2.0), w, strict=True)]
    assert d_over_g == pytest.approx(0.5 * sigma[0] / sigma[1])
    assert {f.id for f in rep.figures()} == {"raman_2.33", "banda_d", "ir"}


def test_gui_action_reports_the_folder_of_report_json(tmp_path):
    from tbkit.gui.actions import HELP, export_report

    grid = np.linspace(100, 200, 11)
    np.savez(tmp_path / "s.npz", grid=grid, a=grid)
    written = export_report(tmp_path / "s.npz")
    assert written["html"].exists() and written["datos"].is_dir()
    assert "Exportar reporte…" in HELP
