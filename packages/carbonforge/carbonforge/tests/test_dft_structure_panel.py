"""El panel de estructura, con un Tk de verdad detrás.

Estas pruebas abren widgets. Donde no haya Tk o pantalla se saltan, como
el resto de las de GUI del paquete.
"""

from __future__ import annotations

import pytest
from ase.build import molecule

tk = pytest.importorskip("tkinter", reason="sin Tk")

from carbonforge.dft.structure_panel import StructurePanel


@pytest.fixture
def root():
    try:
        window = tk.Tk()
    except tk.TclError as exc:                      # pragma: no cover
        pytest.skip(f"sin pantalla: {exc}")
    window.withdraw()
    yield window
    window.destroy()


@pytest.fixture
def panel(root):
    widget = StructurePanel(root)
    widget.pack(fill="both", expand=True)
    return widget


@pytest.fixture
def benzene():
    atoms = molecule("C6H6")
    atoms.center(vacuum=6.0)
    return atoms


class TestPintaLaEstructura:
    def test_antes_de_cargar_no_revienta(self, panel):
        assert panel.view is None
        assert "Carga una estructura" in panel.txt_cell.get("1.0", "end")

    def test_la_formula_y_la_caja(self, panel, benzene):
        panel.show(benzene)
        assert panel.lbl_formula.cget("text") == "C6H6"
        text = panel.txt_cell.get("1.0", "end")
        assert "Celda:" in text and "vacío" in text

    def test_una_fila_por_atomo(self, panel, benzene):
        panel.show(benzene)
        assert len(panel.tree_atoms.get_children()) == len(benzene)

    def test_los_entornos_se_listan_agrupados(self, panel, benzene):
        panel.show(benzene)
        assert len(panel.tree_env.get_children()) == 2

    def test_cambiar_de_modo_cambia_el_veredicto(self, panel, benzene):
        """El mismo vacío basta para LCAO y no para ondas planas: eso tiene
        que verse en el panel sin relanzar nada."""
        panel.show(benzene)
        panel.var_mode.set("lcao")
        assert "suficiente" in panel.txt_cell.get("1.0", "end")
        panel.var_mode.set("pw")
        assert "dipolos" in panel.txt_cell.get("1.0", "end")


class TestLaSeleccion:
    def test_un_atomo_por_entorno(self, panel, benzene):
        panel.show(benzene)
        panel.select_representatives()
        assert len(panel.selection) == 2

    def test_solo_de_una_especie(self, panel, benzene):
        panel.show(benzene)
        panel.select_representatives(only=("C",))
        assert len(panel.selection) == 1
        assert benzene[panel.selection[0]].symbol == "C"

    def test_elegir_un_entorno_toma_el_representante_y_no_los_seis(
            self, panel, benzene):
        panel.show(benzene)
        first = panel.tree_env.get_children()[0]
        panel.tree_env.selection_set(first)
        panel.update()
        assert len(panel.selection) == 1

    def test_la_seleccion_llega_a_quien_escucha(self, root, benzene):
        heard = []
        widget = StructurePanel(root, on_selection=heard.append)
        widget.pack()
        widget.show(benzene)
        widget.select_representatives()
        assert heard and len(heard[-1]) == 2

    def test_limpiar(self, panel, benzene):
        panel.show(benzene)
        panel.select_representatives()
        panel.clear_selection()
        assert panel.selection == ()

    def test_seleccionar_por_criterio(self, panel, benzene):
        panel.show(benzene)
        panel.select_where(element="C")
        assert len(panel.selection) == 6

    def test_cargar_otra_estructura_limpia_lo_elegido(self, panel, benzene):
        panel.show(benzene)
        panel.select_representatives()
        panel.show(molecule("H2O"))
        assert panel.selection == ()


class TestElFiltro:
    def test_filtra_por_simbolo(self, panel, benzene):
        panel.show(benzene)
        panel.var_filter.set("H")
        panel.update()
        assert len(panel.tree_atoms.get_children()) == 6

    def test_filtra_por_entorno(self, panel, benzene):
        panel.show(benzene)
        panel.var_filter.set("sp2")
        panel.update()
        assert len(panel.tree_atoms.get_children()) == 6

    def test_un_filtro_que_no_casa_deja_la_tabla_vacia(self, panel, benzene):
        panel.show(benzene)
        panel.var_filter.set("wolframio")
        panel.update()
        assert panel.tree_atoms.get_children() == ()

    def test_la_seleccion_sobrevive_al_filtro(self, panel, benzene):
        panel.show(benzene)
        panel.select_where(element="C")
        panel.var_filter.set("C")
        panel.update()
        assert len(panel.selection) == 6
