"""El editor de recetas, con un Tk de verdad detrás."""

from __future__ import annotations

import pytest
from ase.build import molecule

tk = pytest.importorskip("tkinter", reason="sin Tk")

from carbonforge.dft.kinds import KINDS  # noqa: E402
from carbonforge.dft.recipe_panel import RecipePanel  # noqa: E402


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError as exc:                       # pragma: no cover
        pytest.skip(f"sin pantalla: {exc}")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def panel(root):
    widget = RecipePanel(root)
    widget.pack(fill="both", expand=True)
    return widget


@pytest.fixture
def benzene():
    atoms = molecule("C6H6")
    atoms.center(vacuum=10.0)
    return atoms


def _choose(panel, name: str) -> None:
    panel.combo.set(KINDS[name].label)
    panel._on_kind()


class TestLosPanelesSalenDelCatalogo:
    def test_cada_tipo_pinta_sus_parametros(self, panel):
        for name, calculation in KINDS.items():
            _choose(panel, name)
            visible = set(panel._vars)
            expected = {p.name for p in calculation.parameters
                        if not p.advanced}
            assert expected <= visible, name

    def test_lo_avanzado_se_esconde_pero_no_se_pierde(self, panel):
        _choose(panel, "scf")
        assert "convergence_density" not in panel._vars
        panel.var_advanced.set(True)
        panel._fill()
        assert "convergence_density" in panel._vars
        # y sigue guardándose con la receta aunque esté oculto
        panel.var_advanced.set(False)
        panel._fill()
        assert "convergence_density" in panel.values()

    def test_los_valores_por_defecto_son_los_del_catalogo(self, panel):
        _choose(panel, "relax")
        assert panel.values()["fmax"] == KINDS["relax"].defaults()["fmax"]

    def test_la_ayuda_aparece_al_elegir_el_calculo(self, panel):
        _choose(panel, "xps")
        assert "hueco" in panel.txt_help.get("1.0", "end")

    def test_el_resumen_dice_que_hace_falta_antes(self, panel):
        _choose(panel, "bands")
        assert "Estado base" in panel.lbl_summary.cget("text")


class TestLoQueSeTeclea:
    def test_la_coma_decimal_se_acepta(self, panel):
        """Un usuario en español teclea 0,05 tan a menudo como 0.05."""
        _choose(panel, "relax")
        panel._vars["fmax"].set("0,05")
        assert panel.values()["fmax"] == pytest.approx(0.05)

    def test_los_indices_se_leen_con_comas_o_espacios(self, panel):
        _choose(panel, "ldos")
        panel._vars["sites"].set("3, 7 11")
        assert panel.values()["sites"] == [3, 7, 11]

    def test_lo_que_no_convierte_se_deja_crudo_para_que_lo_nombre(self, panel):
        _choose(panel, "relax")
        panel._vars["fmax"].set("mucho")
        assert panel.values()["fmax"] == "mucho"
        assert any("Fuerza máxima" in p for p in panel.problems())


class TestLaValidacionCorreMientrasSeEscribe:
    def test_puntos_k_sobre_un_eje_que_no_se_repite(self, panel, benzene):
        panel.set_structure(benzene)
        _choose(panel, "scf")
        panel._vars["kpts"].set("4,4,4")
        assert any("no es periódico" in p for p in panel.problems())

    def test_el_panel_lo_muestra(self, panel, benzene):
        panel.set_structure(benzene)
        _choose(panel, "scf")
        panel._vars["kpts"].set("4,4,4")
        assert "periódico" in panel.txt_problems.get("1.0", "end")

    def test_sin_problemas_lo_dice(self, panel, benzene):
        panel.set_structure(benzene)
        _choose(panel, "scf")
        assert "Sin problemas" in panel.txt_problems.get("1.0", "end")

    def test_avisa_a_quien_escucha(self, root, benzene):
        heard = []
        widget = RecipePanel(root, on_change=heard.append)
        widget.pack()
        widget.set_structure(benzene)
        _choose(widget, "scf")
        widget._vars["kpts"].set("4,4,4")
        assert heard and any("periódico" in p for p in heard[-1])


class TestComponerLaReceta:
    def test_anadir_un_paso(self, panel, benzene):
        panel.set_structure(benzene)
        _choose(panel, "relax")
        assert panel.add_step()
        assert [s.kind for s in panel.steps] == ["relax"]
        assert panel.tree.get_children()

    def test_un_paso_con_problemas_no_entra(self, panel, benzene):
        panel.set_structure(benzene)
        _choose(panel, "ldos")          # sin sitios elegidos
        assert not panel.add_step()
        assert panel.steps == []

    def test_una_nota_de_coste_no_impide_anadir(self, panel, benzene):
        """Los fonones dicen cuánto van a costar; eso informa, no bloquea."""
        panel.set_structure(benzene)
        _choose(panel, "phonons")
        assert panel.add_step()

    def test_la_cadena_se_valida_como_cadena(self, panel, benzene):
        panel.set_structure(benzene)
        _choose(panel, "dos")
        panel._vars["kpts"].set("1,1,1")
        panel.add_step()
        assert any("Estado base" in p
                   for p in panel.recipe().check(benzene))

    def test_quitar_y_vaciar(self, panel, benzene):
        panel.set_structure(benzene)
        _choose(panel, "relax")
        panel.add_step()
        _choose(panel, "scf")
        panel.add_step()
        panel.remove_last()
        assert [s.kind for s in panel.steps] == ["relax"]
        panel.clear()
        assert panel.steps == []
        assert panel.tree.get_children() == ()

    def test_la_receta_lleva_el_nombre_tecleado(self, panel):
        panel.var_name.set("coil204")
        assert panel.recipe().name == "coil204"

    def test_los_pasos_guardan_lo_que_estaba_puesto(self, panel, benzene):
        panel.set_structure(benzene)
        _choose(panel, "relax")
        panel._vars["fmax"].set("0.005")
        panel.add_step()
        assert panel.steps[0].values["fmax"] == pytest.approx(0.005)
