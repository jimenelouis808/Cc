"""The command line for the two new instruments."""

from __future__ import annotations

import pytest

from ramancarbon.cli.main import build_parser, main


def run(args, capsys) -> tuple[int, str]:
    code = main(args)
    return code, capsys.readouterr().out


@pytest.fixture(scope="module")
def demo_xrd(tmp_path_factory):
    folder = tmp_path_factory.mktemp("drx")
    assert main(["demo-datos", "drx", str(folder)]) == 0
    return folder


@pytest.fixture(scope="module")
def demo_echem(tmp_path_factory):
    folder = tmp_path_factory.mktemp("echem")
    assert main(["demo-datos", "echem", str(folder)]) == 0
    return folder


# -- the parser -------------------------------------------------------


def test_every_subcommand_is_reachable():
    parser = build_parser()
    action = next(
        a for a in parser._actions if getattr(a, "choices", None)
        and "analizar" in a.choices
    )
    assert {"drx", "drx-lote", "echem", "demo-datos",
            "mapa", "figura", "exportar", "proyecto", "micro",
            "tiempos"} <= set(action.choices)


def test_help_mentions_the_new_instruments(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert "difracción" in text
    assert "electroquímica" in text.lower()


# -- diffraction ------------------------------------------------------


def test_demo_data_writes_diffractograms(demo_xrd):
    files = sorted(demo_xrd.glob("*.xye"))
    assert len(files) >= 4
    assert all(f.stat().st_size > 1000 for f in files)


def test_the_library_can_be_listed(capsys):
    code, text = run(["drx", "--biblioteca"], capsys)
    assert code == 0
    assert "FeSe_tetragonal" in text
    assert "crystallography.net" in text


def test_identifying_phases_without_refining(demo_xrd, capsys):
    path = demo_xrd / "demo_drx_CNT_FeSe.xye"
    code, text = run(["drx", str(path), "--sin-refinar", "--breve"], capsys)
    assert code == 0
    assert "FeSe_tetragonal" in text
    assert "grafito_2H" in text
    assert "AJUSTE DE RIETVELD" not in text


def test_a_full_run_writes_report_figure_and_calculated_pattern(demo_xrd, tmp_path,
                                                               capsys):
    path = demo_xrd / "demo_drx_CNT_FeSe.xye"
    report = tmp_path / "informe.txt"
    figure = tmp_path / "fig.png"
    calculated = tmp_path / "calc.txt"
    code, _ = run(
        ["drx", str(path), "--breve", "--salida", str(report),
         "--figura", str(figure), "--calculado", str(calculated)],
        capsys,
    )
    assert code == 0
    assert "Rwp" in report.read_text(encoding="utf-8")
    assert figure.stat().st_size > 10000
    header, *rows = calculated.read_text(encoding="utf-8").splitlines()
    assert "observado" in header and "diferencia" in header
    assert len(rows) > 1000


def test_restricting_to_named_phases(demo_xrd, capsys):
    path = demo_xrd / "demo_drx_CNT_FeSe.xye"
    code, text = run(
        ["drx", str(path), "--sin-refinar", "--breve", "--fases", "grafito_2H"],
        capsys,
    )
    assert code == 0
    assert "grafito_2H" in text
    assert "SIN explicar" in text  # the FeSe lines have nowhere to go


def test_an_unknown_phase_name_is_an_error(demo_xrd, capsys):
    code, _ = run(
        ["drx", str(demo_xrd / "demo_drx_CNT_FeSe.xye"), "--fases", "noexiste"],
        capsys,
    )
    assert code == 1


def test_a_malformed_texture_axis_is_an_error(demo_xrd, capsys):
    code, _ = run(
        ["drx", str(demo_xrd / "demo_drx_CNT_FeSe.xye"), "--textura", "zzz"], capsys
    )
    assert code == 1


def test_texture_is_refined_when_the_axis_is_given(demo_xrd, capsys):
    path = demo_xrd / "demo_drx_MoS2_texturado.xye"
    code, text = run(["drx", str(path), "--textura", "001", "--breve"], capsys)
    assert code == 0
    assert "orientación pref" in text


def test_the_batch_writes_one_row_per_pattern(demo_xrd, tmp_path, capsys):
    csv = tmp_path / "fases.csv"
    code, text = run(
        ["drx-lote", str(demo_xrd), "--csv", str(csv), "--sin-refinar"], capsys
    )
    assert code == 0
    lines = csv.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # header plus four demos
    assert "fases" in lines[0]


# -- electrochemistry -------------------------------------------------


def test_demo_data_writes_electrochemistry_files(demo_echem):
    files = list(demo_echem.glob("*.txt"))
    assert len(files) >= 20
    assert any("eis" in f.name for f in files)
    assert any("gcd" in f.name for f in files)


def test_written_curves_record_their_reference_electrode(demo_echem):
    text = (demo_echem / "demo_lsv_OER.txt").read_text(encoding="utf-8")
    assert "# reference: RHE" in text


def test_no_measurement_is_an_error(capsys):
    code, _ = run(["echem", "--masa", "2"], capsys)
    assert code == 1


def test_a_voltammogram_gives_both_capacitance_conventions(demo_echem, capsys):
    path = demo_echem / "demo_cv_condensador_20mVs.txt"
    code, text = run(
        ["echem", "--cv", str(path), "--masa", "2", "--area", "1", "--breve"],
        capsys,
    )
    assert code == 0
    assert "lazo cerrado" in text and "rama anódica" in text
    assert "F/g" in text


def test_a_battery_electrode_is_told_not_to_use_farads(demo_echem, capsys):
    code, text = run(
        [
            "echem",
            "--cv", str(demo_echem / "demo_cv_bateria_20mVs.txt"),
            "--gcd", str(demo_echem / "demo_gcd_bateria.txt"),
            "--masa", "2", "--area", "1",
        ],
        capsys,
    )
    assert code == 0
    assert "NUNCA F/g" in text
    assert "mAh/g" in text


def test_impedance_runs_kramers_kronig_before_the_circuit(demo_echem, capsys):
    code, text = run(
        ["echem", "--eis", str(demo_echem / "demo_eis.txt"),
         "--circuito", "R0-(R1|Q1)-Q2", "--breve"],
        capsys,
    )
    assert code == 0
    assert text.index("Kramers-Kronig") < text.index("Circuito:")
    assert "consistente" in text


def test_a_polarisation_curve_gives_the_slope_it_was_built_with(demo_echem, capsys):
    code, text = run(
        [
            "echem", "--polarizacion", str(demo_echem / "demo_lsv_OER.txt"),
            "--referencia", "RHE", "--area", "1", "--resistencia", "3",
            "--reaccion", "OER", "--breve",
        ],
        capsys,
    )
    assert code == 0
    assert "60." in text and "mV/dec" in text
    assert "η a 10 mA/cm²" in text


def test_a_wrong_reference_electrode_is_caught(demo_echem, capsys):
    """The file says RHE; asking for Ag/AgCl adds 1.04 V to every
    overpotential, and nothing else in the output would reveal it."""
    code, text = run(
        [
            "echem", "--polarizacion", str(demo_echem / "demo_lsv_OER.txt"),
            "--referencia", "Ag/AgCl_3M", "--ph", "14", "--area", "1",
        ],
        capsys,
    )
    assert code == 0
    assert "el archivo declara la referencia" in text
    assert "corriente de intercambio" in text  # the implausible j0 guard


def test_a_rate_series_gives_b_and_the_ecsa(demo_echem, capsys):
    paths = sorted(str(p) for p in demo_echem.glob("demo_cv_condensador_*.txt"))
    code, text = run(
        ["echem", "--velocidades", *paths, "--masa", "2", "--area", "1", "--breve"],
        capsys,
    )
    assert code == 0
    assert "Análisis de b" in text
    assert "ECSA" in text


def test_a_csv_row_can_be_written(demo_echem, tmp_path, capsys):
    csv = tmp_path / "fila.csv"
    code, _ = run(
        ["echem", "--gcd", str(demo_echem / "demo_gcd_condensador.txt"),
         "--masa", "2", "--csv", str(csv), "--breve"],
        capsys,
    )
    assert code == 0
    header, row = csv.read_text(encoding="utf-8").splitlines()
    assert "C_gcd_F" in header
    assert len(row.split(",")) == len(header.split(","))


# -- maps, figures, export, projects, microstructure ---------------------

@pytest.fixture
def sample_folder(tmp_path):
    """One measurement of each kind, on disk."""
    from ramancarbon.core.io import write_spectrum
    from ramancarbon.examples.demo_data import (
        make_demo,
        make_map_demo,
        make_xrd_demo,
    )
    from ramancarbon.mapping.io import write_map
    from ramancarbon.xrd.io import write_pattern

    write_spectrum(make_demo("MWCNT"), tmp_path / "a.txt")
    write_spectrum(make_demo("SWCNT"), tmp_path / "b.txt")
    write_pattern(make_xrd_demo("CNT_FeSe"), tmp_path / "p.xy")
    write_map(make_map_demo("dos_fases", rows=8, columns=10, seed=1),
              tmp_path / "mapa.txt")
    return tmp_path


def test_the_map_command_reports_what_is_in_the_map(sample_folder, capsys):
    assert main(["mapa", str(sample_folder / "mapa.txt"), "--punto", "1",
                 "--cociente", "1280", "1420", "1500", "1660"]) == 0
    out = capsys.readouterr().out
    assert "Rayos cósmicos" in out
    assert "Cobertura" in out
    assert "PCA" in out and "grupos" in out


def test_the_map_command_can_draw_the_cluster_spectra(sample_folder, tmp_path):
    figure = tmp_path / "grupos.png"
    assert main(["mapa", str(sample_folder / "mapa.txt"), "--figura",
                 str(figure), "--preajuste", "acs"]) == 0
    assert figure.exists() and figure.stat().st_size > 0


def test_the_figure_command_draws_several_spectra(sample_folder, tmp_path):
    figure = tmp_path / "f.png"
    data = tmp_path / "f.csv"
    assert main(["figura", str(sample_folder / "a.txt"),
                 str(sample_folder / "b.txt"), "--salida", str(figure),
                 "--preajuste", "acs", "--normalizar", "max",
                 "--desplazar", "0.2", "--datos", str(data)]) == 0
    assert figure.exists()
    assert "a_x" in data.read_text(encoding="utf-8")


def test_mixing_instruments_in_one_figure_is_refused(sample_folder, tmp_path,
                                                     capsys):
    """A spectrum and a diffractogram do not share an x axis."""
    code = main(["figura", str(sample_folder / "a.txt"),
                 str(sample_folder / "p.xy"), "--salida",
                 str(tmp_path / "f.png")])
    assert code == 1
    assert "los ejes no son los mismos" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", [".csv", ".json", ".tex", ".jdx"])
def test_the_export_command_writes_every_offered_format(sample_folder,
                                                        tmp_path, suffix):
    destination = tmp_path / f"salida{suffix}"
    assert main(["exportar", str(sample_folder / "a.txt"),
                 str(destination)]) == 0
    assert destination.exists() and destination.stat().st_size > 0


def test_a_project_is_built_from_a_folder_and_read_back(sample_folder,
                                                        tmp_path, capsys):
    project = tmp_path / "sesion.rcproj"
    assert main(["proyecto", "crear", str(sample_folder), str(project),
                 "--notas", "prueba"]) == 0
    assert project.exists()
    capsys.readouterr()

    assert main(["proyecto", "ver", str(project)]) == 0
    out = capsys.readouterr().out
    assert "prueba" in out
    assert "raman" in out and "xrd" in out


def test_a_folder_of_mixed_measurements_does_not_break_on_one_reader(
        sample_folder, tmp_path):
    """Passing --laser to every reader made every non-Raman file fail with
    a TypeError about a keyword argument."""
    assert main(["proyecto", "crear", str(sample_folder),
                 str(tmp_path / "s.rcproj"), "--laser", "532"]) == 0
    from ramancarbon.dataio import Project

    kinds = {d.kind for d in Project.load(tmp_path / "s.rcproj").datasets}
    assert {"raman", "xrd"} <= kinds


def test_the_microstructure_command_reports_all_three_methods(sample_folder,
                                                              capsys):
    assert main(["micro", str(sample_folder / "p.xy"),
                 "--instrumento", "0.05"]) == 0
    out = capsys.readouterr().out
    assert "Williamson-Hall" in out
    assert "Tamaño-deformación" in out
    assert "Halder-Wagner" in out


def test_the_benchmark_command_runs(capsys):
    assert main(["tiempos", "--rapido", "--repeticiones", "1"]) == 0
    out = capsys.readouterr().out
    assert "Raman" in out and "Suma de los mejores tiempos" in out


def test_the_number_of_components_can_be_imposed_from_analizar(sample_folder,
                                                               capsys):
    """The README documented `analizar --picos-d 3 --picos-g 2` and only
    `deconvolucionar` accepted it."""
    assert main(["analizar", str(sample_folder / "a.txt"), "--laser", "532",
                 "--picos-d", "3", "--picos-g", "2"]) == 0
    out = capsys.readouterr().out
    assert "modelo impuesto" in out
    assert "5 componentes" in out


def test_an_imposed_model_says_it_was_not_compared(sample_folder, capsys):
    main(["analizar", str(sample_folder / "a.txt"), "--laser", "532",
          "--picos-d", "2", "--picos-g", "1"])
    out = capsys.readouterr().out
    assert "impuesto" in out
    assert "no dicen si este es el mejor" in out
