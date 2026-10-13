"""Las tres páginas juntas, con un Tk de verdad."""

from __future__ import annotations

import json

import pytest
from ase.build import molecule

tk = pytest.importorskip("tkinter", reason="sin Tk")

from carbonforge.dft.app import ORIGIN, DftApp  # noqa: E402
from carbonforge.dft.kinds import KINDS  # noqa: E402
from carbonforge.gui.session import Session  # noqa: E402


class _Queue:
    """Una cola que sólo anota: lanzar de verdad es cosa de JobQueue."""

    def __init__(self):
        self.submitted = []

    def submit(self, directory, nprocs=1):
        self.submitted.append(directory)
        return directory


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError as exc:                        # pragma: no cover
        pytest.skip(f"sin pantalla: {exc}")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def app(root, tmp_path):
    return DftApp(root, tmp_path / "calculos",
                  session=Session(), queue=_Queue())


@pytest.fixture
def benzene():
    atoms = molecule("C6H6")
    atoms.center(vacuum=10.0)
    return atoms


def _choose(app, name: str) -> None:
    app.recipe_panel.combo.set(KINDS[name].label)
    app.recipe_panel._on_kind()


class TestLaEstructuraLlegaALasTresPaginas:
    def test_traerla_de_la_ventana(self, app, benzene):
        app.session.publish(benzene, origin="Construir")
        app.take_current()
        assert app.atoms is not None
        assert app.structure_panel.view is not None
        assert app.recipe_panel.atoms is not None

    def test_sin_estructura_lo_dice(self, app):
        app.take_current()
        assert "No hay estructura actual" in app.status_var.get()

    def test_publicarla_de_vuelta(self, app, benzene):
        app.set_structure(benzene)
        app.publish()
        assert app.session.current.origin == ORIGIN

    def test_la_validacion_de_la_receta_ya_ve_la_estructura(self, app, benzene):
        app.set_structure(benzene)
        _choose(app, "scf")
        app.recipe_panel._vars["kpts"].set("4,4,4")
        assert any("no es periódico" in p for p in app.recipe_panel.problems())


class TestLosSitiosElegidosViajanSolos:
    """Elegir átomos en el visor es la ENTRADA del LDOS y del XPS: copiar
    índices a mano entre pestañas es donde se cuelan los errores."""

    def test_la_seleccion_llena_el_parametro(self, app, benzene):
        app.set_structure(benzene)
        _choose(app, "ldos")
        app.structure_panel.set_selection([2, 5])
        assert app.recipe_panel._vars["sites"].get() == "2, 5"
        assert app.recipe_panel.values()["sites"] == [2, 5]

    def test_un_calculo_sin_sitios_no_se_entera(self, app, benzene):
        app.set_structure(benzene)
        _choose(app, "scf")
        app.structure_panel.set_selection([1])      # no debe reventar
        assert "sites" not in app.recipe_panel._vars

    def test_un_representante_por_entorno(self, app, benzene):
        app.set_structure(benzene)
        _choose(app, "xps")
        app.structure_panel.select_representatives(only=("C",))
        assert len(app.recipe_panel.values()["sites"]) == 1


class TestPreparar:
    def test_sin_estructura(self, app):
        assert app.prepare(False) is None
        assert "trae o abre una estructura" in app.status_var.get()

    def test_con_la_receta_vacia(self, app, benzene):
        app.set_structure(benzene)
        assert app.prepare(False) is None
        assert "vacía" in app.status_var.get()

    def test_escribe_el_directorio(self, app, benzene):
        app.set_structure(benzene)
        _choose(app, "relax")
        app.recipe_panel.add_step()
        directory = app.prepare(False)
        assert directory is not None and directory.exists()
        assert (directory / "01_relax.py").exists()
        assert (directory / "job.json").exists()

    def test_una_receta_que_no_cierra_se_niega_con_su_razon(self, app, benzene):
        app.set_structure(benzene)
        _choose(app, "dos")
        app.recipe_panel._vars["kpts"].set("1,1,1")
        app.recipe_panel.add_step()
        assert app.prepare(False) is None
        assert "Estado base" in app.status_var.get()

    def test_la_cadena_se_valida_antes_que_el_generador(self, app, benzene):
        """Un XPS sin estado base delante se rechaza por eso, no por no
        tener generador todavía: se dice la primera razón, no la última."""
        app.set_structure(benzene)
        _choose(app, "xps")
        app.structure_panel.select_representatives(only=("C",))
        app.recipe_panel._vars["spinpol"].set(True)
        app.recipe_panel.add_step()
        assert app.prepare(False) is None
        assert "Estado base" in app.status_var.get()

    def test_un_tipo_sin_generador_lo_dice_por_su_nombre(self, app, benzene):
        app.set_structure(benzene)
        _choose(app, "scf")
        app.recipe_panel.add_step()
        _choose(app, "xps")
        app.structure_panel.select_representatives(only=("C",))
        app.recipe_panel._vars["spinpol"].set(True)
        app.recipe_panel.add_step()
        assert app.prepare(False) is None
        assert "xps" in app.status_var.get()
        assert "hueco de core" in app.status_var.get()

    def test_correr_pasa_por_la_cola_y_no_por_la_ventana(self, app, benzene):
        """La ventana nunca corre un cálculo en su proceso: un hilo no se
        puede cancelar."""
        app.set_structure(benzene)
        _choose(app, "relax")
        app.recipe_panel.add_step()
        directory = app.prepare(True)
        assert app.queue.submitted == [directory]

    def test_sin_cola_el_boton_de_correr_esta_apagado(self, root, tmp_path):
        alone = DftApp(root, tmp_path / "c", session=Session())
        assert str(alone.run_button["state"]) == "disabled"


class TestResultados:
    def _with_dos(self, app, tmp_path):
        directory = tmp_path / "hecho"
        directory.mkdir()
        (directory / "dos.json").write_text(json.dumps({
            "fermi": -1.0, "energies": [-1.0, 0.0, 1.0],
            "total": [0.1, 0.4, 0.2],
            "por_orbital": {"p": [0.05, 0.3, 0.1]}}), encoding="utf-8")
        app.var_results.set(str(directory))
        return directory

    def test_solo_se_ofrece_lo_que_existe(self, app, tmp_path):
        self._with_dos(app, tmp_path)
        found = app.scan_results()
        assert set(found) == {"dos"}
        assert app.results_combo.cget("values") == (KINDS["dos"].label,)

    def test_un_directorio_vacio_lo_dice(self, app, tmp_path):
        app.var_results.set(str(tmp_path))
        app.scan_results()
        assert "Sin resultados" in app.status_var.get()

    def test_dibuja_y_cuenta_lo_que_dibujo(self, app, tmp_path):
        self._with_dos(app, tmp_path)
        app.scan_results()
        app.draw_result()
        assert app.axes.get_lines()
        assert "Fermi" in app.txt_results.get("1.0", "end")

    def test_avisa_de_que_las_proyecciones_no_suman_el_total(self, app,
                                                             tmp_path):
        self._with_dos(app, tmp_path)
        app.scan_results()
        app.draw_result()
        assert "NO suman" in app.txt_results.get("1.0", "end")

    def test_los_fonones_imaginarios_se_dicen(self, app, tmp_path):
        directory = tmp_path / "fon"
        directory.mkdir()
        (directory / "phonons.json").write_text(
            json.dumps({"gamma_cm1": [-150.0, -0.2, 0.1, 1580.0]}),
            encoding="utf-8")
        app.var_results.set(str(directory))
        app.scan_results()
        app.draw_result()
        assert "IMAGINARIO" in app.txt_results.get("1.0", "end")

    def test_dibujar_sin_buscar_antes(self, app):
        app.draw_result()
        assert "Buscar" in app.status_var.get()
