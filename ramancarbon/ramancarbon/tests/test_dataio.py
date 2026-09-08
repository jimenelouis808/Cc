"""Import and export: detection, JCAMP, tables, project files."""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from ramancarbon.core.io import write_spectrum
from ramancarbon.dataio.detect import KNOWN_SUFFIXES, detect, scan
from ramancarbon.dataio.export import (
    Table,
    export,
    series_table,
    significant,
    summary_table,
    with_uncertainty,
)
from ramancarbon.dataio.jcamp import (
    JCAMPError,
    parse_jcamp,
    parse_records,
    read_jcamp,
    write_jcamp,
)
from ramancarbon.dataio.load import LoadError, load, load_folder
from ramancarbon.dataio.project import FORMAT_VERSION, Project, ProjectError
from ramancarbon.echem.io import write_cv, write_eis, write_gcd
from ramancarbon.examples.demo_data import (
    make_cv_demo,
    make_demo,
    make_eis_demo,
    make_gcd_demo,
    make_xrd_demo,
)
from ramancarbon.xrd.io import write_pattern


@pytest.fixture
def folder(tmp_path):
    """One file of every kind the suite reads, plus one it does not."""
    write_spectrum(make_demo("MWCNT"), tmp_path / "raman.txt")
    write_pattern(make_xrd_demo("CNT_FeSe"), tmp_path / "drx.xy")
    write_cv(make_cv_demo("pseudocondensador"), tmp_path / "cv.txt")
    write_gcd(make_gcd_demo(), tmp_path / "gcd.txt")
    write_eis(make_eis_demo(), tmp_path / "eis.txt")
    (tmp_path / "leeme.md").write_text("notas de la muestra\n", encoding="utf-8")
    return tmp_path


# -- detection ---------------------------------------------------------

@pytest.mark.parametrize(
    "name,kind",
    [("raman.txt", "raman"), ("drx.xy", "xrd"), ("cv.txt", "cv"),
     ("gcd.txt", "gcd"), ("eis.txt", "eis")],
)
def test_each_instrument_is_recognised_from_its_numbers(folder, name, kind):
    found = detect(folder / name)
    assert found.kind == kind
    assert found.confidence in ("alta", "media")
    assert found.reasons, "una detección sin motivo no se puede discutir"


def test_the_extension_does_not_decide(folder, tmp_path):
    """The same spectrum under four extensions is still a spectrum."""
    text = (folder / "raman.txt").read_text(encoding="utf-8")
    for suffix in (".dat", ".csv", ".asc", ".espectro"):
        target = tmp_path / f"copia{suffix}"
        target.write_text(text, encoding="utf-8")
        assert detect(target).kind == "raman"


def test_a_voltammogram_is_known_by_coming_back_on_itself(folder):
    found = detect(folder / "cv.txt")
    assert found.kind == "cv"
    assert "sentido" in found.reasons[0]


def test_a_charge_discharge_curve_is_not_read_as_a_diffractogram(folder):
    assert detect(folder / "gcd.txt").kind == "gcd"


def test_a_file_that_is_not_a_measurement_says_so(folder):
    found = detect(folder / "leeme.md")
    assert found.kind == "desconocido"
    assert found.advice


def test_a_missing_file_is_a_result_not_an_exception(tmp_path):
    assert detect(tmp_path / "no_existe.txt").kind == "desconocido"


def test_a_binary_file_says_what_to_export_instead(tmp_path):
    binary = tmp_path / "medida.wdf"
    binary.write_bytes(b"WDF1\x00\x00\x00\x00" + bytes(range(256)))
    found = detect(binary)
    assert found.kind == "binario"
    assert "exp" in found.advice.lower()


def test_binary_content_is_caught_whatever_the_extension(tmp_path):
    disguised = tmp_path / "datos.txt"
    disguised.write_bytes(b"\x00\x01\x02" * 100)
    assert detect(disguised).kind == "binario"


def test_a_cif_is_recognised_by_its_contents(tmp_path):
    cif = tmp_path / "fase.txt"
    cif.write_text("data_x\n_cell_length_a 3.7734\n", encoding="utf-8")
    assert detect(cif).kind == "cif"


def test_scanning_a_folder_sorts_it(folder):
    sorted_files = scan(folder)
    assert {"raman", "xrd", "cv", "gcd", "eis"} <= set(sorted_files)
    assert sorted_files["desconocido"][0].name == "leeme.md"


def test_every_named_suffix_has_a_kind():
    assert all(value for value in KNOWN_SUFFIXES.values())


# -- JCAMP -------------------------------------------------------------

def test_a_plain_jcamp_block_reads():
    text = (
        "##TITLE=prueba\n##JCAMP-DX=4.24\n##DATA TYPE=RAMAN SPECTRUM\n"
        "##XUNITS=1/CM\n##FIRSTX=100.0\n##LASTX=109.0\n##NPOINTS=10\n"
        "##XFACTOR=1.0\n##YFACTOR=2.0\n##XYDATA=(X++(Y..Y))\n"
        "100.0 1 2 3 4 5\n105.0 6 7 8 9 10\n##END=\n"
    )
    parsed = parse_jcamp(text)
    assert np.allclose(parsed["x"], np.arange(100.0, 110.0))
    assert np.allclose(parsed["y"], 2 * np.arange(1.0, 11.0))
    assert parsed["x_units"] == "cm-1"
    assert parsed["data_type"] == "RAMAN SPECTRUM"


def test_the_y_factor_is_applied_and_not_ignored():
    base = ("##TITLE=t\n##XUNITS=1/CM\n##FIRSTX=0\n##LASTX=4\n##NPOINTS=5\n"
            "##YFACTOR={}\n##XYDATA=(X++(Y..Y))\n0 1 2 3 4 5\n##END=\n")
    one = parse_jcamp(base.format("1.0"))["y"]
    ten = parse_jcamp(base.format("10.0"))["y"]
    assert np.allclose(ten, 10 * one)


def test_squeezed_difference_and_duplicate_forms_all_decode():
    text = ("##TITLE=t\n##XUNITS=1/CM\n##FIRSTX=100\n##LASTX=106\n"
            "##NPOINTS=7\n##XYDATA=(X++(Y..Y))\n100 A0JJJ\n104 A3JJJ\n##END=\n")
    parsed = parse_jcamp(text)
    assert np.allclose(parsed["y"], [10, 11, 12, 13, 14, 15, 16])
    assert np.allclose(parsed["x"], np.arange(100.0, 107.0))


def test_a_duplicate_after_a_difference_repeats_the_difference():
    """DUP after DIF continues the ramp; reading it as a repeated value
    turns a straight line into a plateau."""
    text = ("##TITLE=t\n##XUNITS=1/CM\n##FIRSTX=0\n##LASTX=2\n##NPOINTS=3\n"
            "##XYDATA=(X++(Y..Y))\n0 A0JT\n##END=\n")
    assert np.allclose(parse_jcamp(text)["y"], [10, 11, 12])


def test_a_line_in_the_wrong_place_is_caught():
    text = ("##TITLE=t\n##XUNITS=1/CM\n##FIRSTX=100\n##LASTX=109\n"
            "##NPOINTS=10\n##XYDATA=(X++(Y..Y))\n"
            "100 1 2 3 4 5\n120 6 7 8 9 10\n##END=\n")
    with pytest.raises(JCAMPError, match="consistente"):
        parse_jcamp(text)


def test_a_truncated_file_is_caught_by_its_own_point_count():
    text = ("##TITLE=t\n##XUNITS=1/CM\n##FIRSTX=100\n##LASTX=109\n"
            "##NPOINTS=10\n##XYDATA=(X++(Y..Y))\n100 1 2 3 4 5\n##END=\n")
    with pytest.raises(JCAMPError, match="truncado"):
        parse_jcamp(text)


def test_the_difference_check_catches_a_lost_line():
    text = ("##TITLE=t\n##XUNITS=1/CM\n##FIRSTX=100\n##LASTX=106\n"
            "##NPOINTS=7\n##XYDATA=(X++(Y..Y))\n100 A0JJJ\n104 A9JJJ\n##END=\n")
    with pytest.raises(JCAMPError, match="comprobación Y"):
        parse_jcamp(text)


def test_xy_point_pairs_are_read_and_sorted():
    text = ("##TITLE=t\n##XUNITS=1/CM\n##XYPOINTS=(XY..XY)\n"
            "3.0, 30\n1.0, 10\n2.0, 20\n##END=\n")
    parsed = parse_jcamp(text)
    assert np.allclose(parsed["x"], [1.0, 2.0, 3.0])
    assert np.allclose(parsed["y"], [10.0, 20.0, 30.0])


def test_a_file_with_no_data_block_says_so():
    with pytest.raises(JCAMPError, match="bloque de datos"):
        parse_jcamp("##TITLE=vacío\n##XUNITS=1/CM\n##END=\n")


def test_something_that_is_not_jcamp_at_all_is_refused():
    with pytest.raises(JCAMPError, match="etiqueta"):
        parse_jcamp("1 2\n3 4\n")


def test_labels_are_matched_however_they_are_spaced():
    records = dict(parse_records("##FIRST X=1\n##data-type=RAMAN\n##END=\n"))
    assert records["FIRSTX"] == "1"
    assert records["DATATYPE"] == "RAMAN"


def test_the_laser_is_found_wherever_the_vendor_put_it():
    text = ("##TITLE=muestra a 532 nm\n##XUNITS=1/CM\n##XYPOINTS=(XY..XY)\n"
            "1,1\n2,2\n##END=\n")
    assert parse_jcamp(text)["laser_nm"] == 532.0


def test_a_number_that_is_not_a_laser_wavelength_is_not_taken_for_one():
    text = ("##TITLE=muestra de 20 nm\n##XUNITS=1/CM\n##XYPOINTS=(XY..XY)\n"
            "1,1\n2,2\n##END=\n")
    assert parse_jcamp(text)["laser_nm"] is None


def test_a_written_jcamp_reads_back_as_the_same_numbers(tmp_path):
    spectrum = make_demo("MWCNT")
    path = write_jcamp(spectrum.shift, spectrum.intensity, tmp_path / "s.jdx",
                       title="M1", laser_nm=532.0)
    parsed = read_jcamp(path)
    assert np.allclose(parsed["x"], spectrum.shift, atol=1e-4)
    assert np.allclose(parsed["y"], spectrum.intensity, atol=1e-4)
    assert parsed["laser_nm"] == 532.0
    assert parsed["title"] == "M1"


def test_writing_refuses_mismatched_columns(tmp_path):
    with pytest.raises(ValueError):
        write_jcamp([1, 2, 3], [1, 2], tmp_path / "malo.jdx")


# -- loading -----------------------------------------------------------

def test_load_opens_each_kind_as_the_right_object(folder):
    kinds = {}
    loaded, failures = load_folder(folder)
    for item in loaded:
        kinds[item.kind] = type(item.data).__name__
    assert kinds == {
        "raman": "Spectrum", "xrd": "Pattern", "cv": "Voltammogram",
        "gcd": "ChargeDischarge", "eis": "Impedance",
    }
    assert [p.name for p, _ in failures] == ["leeme.md"]


def test_a_folder_import_does_not_stop_at_the_first_bad_file(folder):
    loaded, failures = load_folder(folder)
    assert len(loaded) == 5 and len(failures) == 1


def test_a_jcamp_spectrum_loads_as_a_spectrum(tmp_path):
    spectrum = make_demo("SWCNT")
    write_jcamp(spectrum.shift, spectrum.intensity, tmp_path / "s.jdx",
                laser_nm=633.0)
    result = load(tmp_path / "s.jdx")
    assert result.kind == "raman"
    assert result.data.laser_nm == 633.0
    assert np.allclose(result.data.shift, spectrum.shift, atol=1e-4)


def test_a_jcamp_in_nanometres_without_a_laser_is_refused(tmp_path):
    write_jcamp([500.0, 501.0, 502.0], [1.0, 2.0, 3.0], tmp_path / "n.jdx",
                x_units="NANOMETERS")
    with pytest.raises(LoadError, match="laser_nm"):
        load(tmp_path / "n.jdx")


def test_forcing_a_kind_records_the_disagreement(folder):
    result = load(folder / "drx.xy", kind="xrd")
    assert result.kind == "xrd"
    assert not result.warnings
    forced = load(folder / "raman.txt", kind="raman")
    assert not forced.warnings


def test_loading_a_binary_file_explains_what_to_do(tmp_path):
    binary = tmp_path / "m.wdf"
    binary.write_bytes(b"\x00" * 600)
    with pytest.raises(LoadError, match="exp"):
        load(binary)


# -- tables ------------------------------------------------------------

def test_significant_figures_are_significant_not_decimal():
    assert significant(50.1234567) == "50.12"
    assert significant(0.000123456) == "0.0001235"
    assert significant(1234567.0) == "1.235e+06"
    assert significant(0) == "0"


def test_a_value_is_written_to_its_uncertainty():
    assert with_uncertainty(50.1234567, 0.42) == "50.1 ± 0.4"
    assert with_uncertainty(2.4635912, 0.0004) == "2.4636 ± 0.0004"
    assert with_uncertainty(3.1, None) == "3.1"


@pytest.mark.parametrize(
    "suffix", [".csv", ".tsv", ".json", ".md", ".tex", ".html"]
)
def test_a_table_writes_itself_in_every_format(tmp_path, suffix):
    table = summary_table(
        [{"muestra": "M1", "I_D/I_G": 1.23}, {"muestra": "M2", "I_D/I_G": 0.9}],
        title="Cocientes", notes=["I_D/I_G no es monótona con el desorden"],
    )
    path = table.save(tmp_path / f"tabla{suffix}")
    text = path.read_text(encoding="utf-8")
    assert "M1" in text and "M2" in text
    assert "monótona" in text


def test_an_unknown_suffix_is_refused_by_name(tmp_path):
    with pytest.raises(ValueError, match="formato"):
        Table(columns=["a"], rows=[[1]]).save(tmp_path / "t.docx")


def test_the_units_row_is_separate_from_the_column_name():
    table = Table(columns=["C"], units=["F/g"], rows=[[50.0]])
    lines = table.to_csv().splitlines()
    assert lines[0] == "C"
    assert lines[1] == "F/g"


def test_units_must_match_the_columns():
    with pytest.raises(ValueError, match="unidades"):
        Table(columns=["a", "b"], units=["m"], rows=[])


def test_json_export_keeps_the_computed_value_not_the_rounded_text():
    table = Table(columns=["x"], rows=[[50.1234567]])
    assert json.loads(table.to_json())["datos"]["x"] == [50.1234567]


def test_latex_escapes_what_would_break_the_document():
    text = Table(columns=["I_D/I_G"], rows=[["100 %"]]).to_latex()
    assert r"I\_D/I\_G" in text and r"100 \%" in text


def test_a_missing_value_stays_empty_rather_than_becoming_zero():
    table = summary_table([{"a": 1.0}, {"a": 2.0, "b": 3.0}])
    assert table.text_rows[0][1] == ""


def test_a_charge_discharge_export_keeps_its_current_column():
    table = series_table(make_gcd_demo())
    assert table.columns[:3] == ["tiempo", "potencial", "corriente"]
    assert table.units[:3] == ["s", "V", "A"]


def test_an_impedance_export_says_the_sign_convention():
    table = series_table(make_eis_demo())
    assert table.columns == ["frecuencia", "Z_real", "Z_imag"]
    assert any("signo" in note for note in table.notes)


def test_a_spectrum_export_carries_the_laser():
    assert "532" in " ".join(series_table(make_demo("MWCNT")).notes)


def test_something_untabulatable_says_so():
    with pytest.raises(TypeError):
        series_table(object())


def test_export_dispatches_on_the_suffix(tmp_path):
    spectrum = make_demo("MWCNT")
    assert export(spectrum, tmp_path / "s.jdx").exists()
    assert export(make_xrd_demo("CNT_FeSe"), tmp_path / "p.xy").exists()
    assert export(spectrum, tmp_path / "s.csv").exists()


def test_an_exported_jcamp_can_be_loaded_back(tmp_path):
    spectrum = make_demo("MWCNT")
    export(spectrum, tmp_path / "s.jdx")
    back = load(tmp_path / "s.jdx").data
    assert np.allclose(back.intensity, spectrum.intensity, atol=1e-4)


# -- projects ----------------------------------------------------------

@pytest.fixture
def measurements():
    spectrum = make_demo("MWCNT_FeSe")
    spectrum.name = "M1"
    return [spectrum, make_xrd_demo("CNT_FeSe"), make_cv_demo("pseudocondensador"),
            make_gcd_demo(), make_eis_demo()]


def test_every_measurement_survives_a_project_round_trip(tmp_path, measurements):
    project = Project(name="sesión", notes="serie de septiembre")
    for item in measurements:
        project.add(item)
    reopened = Project.load(project.save(tmp_path / "s"))

    assert reopened.name == "sesión" and reopened.notes == "serie de septiembre"
    assert [d.kind for d in reopened.datasets] == ["raman", "xrd", "cv", "gcd", "eis"]
    for dataset, original in zip(reopened.datasets, measurements):
        rebuilt = dataset.to_object()
        for attribute in ("shift", "intensity", "two_theta", "potential",
                          "current", "time", "frequency", "z"):
            if hasattr(original, attribute):
                assert np.allclose(getattr(rebuilt, attribute),
                                   getattr(original, attribute)), attribute


def test_the_numbers_needed_to_analyse_are_stored_with_the_numbers(tmp_path, measurements):
    project = Project()
    for item in measurements:
        project.add(item)
    reopened = Project.load(project.save(tmp_path / "s"))
    by_kind = {d.kind: d for d in reopened.datasets}
    assert by_kind["raman"].options["laser_nm"] == 532.0
    assert by_kind["xrd"].options["wavelength"] == pytest.approx(1.540598, rel=1e-6)
    assert by_kind["cv"].options["scan_rate"] > 0
    assert by_kind["cv"].options["electrodo"]["masa_mg"]


def test_the_electrode_comes_back_whole(tmp_path):
    original = make_cv_demo("pseudocondensador")
    project = Project()
    project.add(original)
    rebuilt = Project.load(project.save(tmp_path / "s")).datasets[0].to_object()
    assert rebuilt.electrode.mass_mg == original.electrode.mass_mg
    assert rebuilt.electrode.reference == original.electrode.reference
    assert rebuilt.electrode.ph == original.electrode.ph


def test_a_project_is_a_zip_of_readable_parts(tmp_path, measurements):
    project = Project(name="s")
    project.add(measurements[0])
    path = project.save(tmp_path / "s")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert "proyecto.json" in names
        assert any(n.startswith("datos/") and n.endswith(".csv") for n in names)
        manifest = json.loads(archive.read("proyecto.json").decode("utf-8"))
    assert manifest["formato"] == FORMAT_VERSION
    assert manifest["conjuntos"][0]["tipo"] == "raman"


def test_the_suffix_is_forced(tmp_path, measurements):
    project = Project()
    project.add(measurements[0])
    assert project.save(tmp_path / "sesion.txt").suffix == ".rcproj"


def test_a_figure_travels_with_the_project(tmp_path, measurements):
    from ramancarbon.plotting import Plot, from_spectrum
    from ramancarbon.plotting.style import preset

    plot = Plot(series=[from_spectrum(measurements[0])], name="fig1")
    plot.style = preset("acs")
    project = Project()
    project.add(measurements[0])
    project.add_figure(plot)
    reopened = Project.load(project.save(tmp_path / "s"))
    assert len(reopened.figures) == 1
    assert Plot.from_dict(reopened.figures[0]).style == plot.style


def test_a_results_table_travels_with_the_project(tmp_path, measurements):
    project = Project()
    project.add(measurements[0])
    project.add_table(summary_table([{"muestra": "M1", "I_D/I_G": 1.23}],
                                    title="Cocientes"))
    reopened = Project.load(project.save(tmp_path / "s"))
    assert reopened.results[0]["titulo"] == "Cocientes"
    assert reopened.results[0]["filas"][0][1] == pytest.approx(1.23)


def test_two_measurements_with_the_same_name_get_different_identifiers(tmp_path):
    project = Project()
    first, second = make_demo("MWCNT"), make_demo("MWCNT")
    project.add(first)
    project.add(second)
    assert project.datasets[0].identifier != project.datasets[1].identifier


def test_asking_for_a_dataset_that_is_not_there_lists_the_ones_that_are(tmp_path):
    project = Project()
    project.add(make_demo("MWCNT"))
    with pytest.raises(KeyError, match="hay"):
        project.dataset("no_existe")


def test_a_project_from_a_newer_version_still_opens(tmp_path, measurements):
    project = Project(name="futuro")
    project.add(measurements[0])
    path = project.save(tmp_path / "s")

    with zipfile.ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(parts["proyecto.json"].decode("utf-8"))
    manifest["formato"] = FORMAT_VERSION + 5
    manifest["algo_del_futuro"] = {"lo que sea": True}
    parts["proyecto.json"] = json.dumps(manifest).encode("utf-8")
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)

    reopened = Project.load(path)
    assert "[Aviso]" in reopened.notes
    assert len(reopened.datasets) == 1
    assert reopened.datasets[0].to_object().shift.size > 100


def test_something_that_is_not_a_project_says_so(tmp_path):
    plain = tmp_path / "no.rcproj"
    plain.write_text("no soy un zip", encoding="utf-8")
    with pytest.raises(ProjectError, match="ZIP"):
        Project.load(plain)


def test_a_zip_without_a_manifest_says_so(tmp_path):
    path = tmp_path / "vacio.rcproj"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("otra_cosa.txt", "hola")
    with pytest.raises(ProjectError, match="proyecto.json"):
        Project.load(path)


def test_a_manifest_that_names_a_missing_file_says_which(tmp_path, measurements):
    project = Project()
    project.add(measurements[0])
    path = project.save(tmp_path / "s")
    with zipfile.ZipFile(path) as archive:
        parts = {n: archive.read(n) for n in archive.namelist()
                 if not n.startswith("datos/")}
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in parts.items():
            archive.writestr(name, payload)
    with pytest.raises(ProjectError, match="no está en el proyecto"):
        Project.load(path)


def test_a_missing_project_file_is_refused(tmp_path):
    with pytest.raises(ProjectError, match="no existe"):
        Project.load(tmp_path / "no_hay_nada.rcproj")


def test_objects_can_be_filtered_by_kind(tmp_path, measurements):
    project = Project()
    for item in measurements:
        project.add(item)
    assert len(project.objects("raman")) == 1
    assert len(project.objects()) == 5
