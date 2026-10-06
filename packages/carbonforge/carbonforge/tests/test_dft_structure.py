"""Lo que el panel de estructura tiene que acertar antes de pintar nada."""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.build import graphene_nanoribbon, molecule

from carbonforge.dft.structure import (
    VACUUM_PER_SIDE,
    describe,
    representatives,
    select,
)


@pytest.fixture
def benzene() -> Atoms:
    atoms = molecule("C6H6")
    atoms.center(vacuum=6.0)
    return atoms


@pytest.fixture
def ribbon() -> Atoms:
    """Periódica en un eje, con vacío en los otros dos."""
    atoms = graphene_nanoribbon(2, 2, type="armchair", saturated=True, vacuum=6.0)
    return atoms


class TestLaCaja:
    def test_el_vacio_es_por_lado_y_no_total(self, benzene):
        view = describe(benzene)
        axis = 2
        total = view.cell.lengths[axis] - view.cell.span[axis]
        assert view.cell.vacuum_per_side[axis] == pytest.approx(total / 2.0)
        assert view.cell.vacuum_per_side[axis] == pytest.approx(6.0, abs=0.1)

    def test_dice_que_ejes_se_repiten(self, ribbon):
        view = describe(ribbon)
        assert view.cell.dimensionality == sum(ribbon.pbc)
        assert view.cell.periodic_axes == tuple(
            i for i, p in enumerate(ribbon.pbc) if p)

    def test_ondas_planas_piden_mas_vacio_que_lcao(self, benzene):
        """El mismo vacío basta para LCAO y se queda corto para PW: ése es
        el error que cuesta una tanda entera."""
        view = describe(benzene)
        assert VACUUM_PER_SIDE["pw"] > VACUUM_PER_SIDE["lcao"]
        assert not view.cell.vacuum_verdict("lcao")
        problems = view.cell.vacuum_verdict("pw")
        assert problems and "dipolos" in problems[0]

    def test_un_eje_periodico_no_se_juzga_por_su_vacio(self, ribbon):
        view = describe(ribbon)
        for message in view.cell.vacuum_verdict("pw"):
            for axis in view.cell.periodic_axes:
                assert f"Eje {'xyz'[axis]}" not in message

    def test_la_separacion_entre_imagenes(self, benzene):
        view = describe(benzene)
        assert view.cell.image_separation(0) == pytest.approx(
            view.cell.lengths[0] - view.cell.span[0])


class TestLaTablaDeAtomos:
    def test_hay_una_fila_por_atomo_y_en_orden(self, benzene):
        view = describe(benzene)
        assert len(view.rows) == len(benzene)
        assert [r.index for r in view.rows] == list(range(len(benzene)))

    def test_el_benceno_sale_como_es(self, benzene):
        view = describe(benzene)
        carbons = [r for r in view.rows if r.symbol == "C"]
        hydrogens = [r for r in view.rows if r.symbol == "H"]
        assert len(carbons) == 6 and len(hydrogens) == 6
        for row in carbons:
            assert row.coordination == 3
            assert 6 in row.rings
        for row in hydrogens:
            assert row.coordination == 1
            assert row.neighbour_symbols == ("C",)

    def test_los_vecinos_son_reciprocos(self, benzene):
        view = describe(benzene)
        for row in view.rows:
            for other in row.neighbours:
                assert row.index in view.row(other).neighbours


class TestLosEntornos:
    def test_el_benceno_tiene_dos_entornos_y_no_doce(self, benzene):
        """Los seis carbonos son equivalentes: calcular el core de los seis
        es repetir el mismo numero seis veces."""
        view = describe(benzene)
        assert len(view.environments) == 2
        assert {e.count for e in view.environments} == {6}
        assert len(representatives(view)) == 2

    def test_solo_los_de_una_especie(self, benzene):
        view = describe(benzene)
        assert len(representatives(view, only=("C",))) == 1

    def test_el_nitrogeno_piridinico_se_llama_por_su_nombre(self):
        """Dos carbonos vecinos y un hexagono: piridinico."""
        atoms = molecule("C5H5N")
        atoms.center(vacuum=6.0)
        view = describe(atoms)
        nitrogen = select(view, element="N")
        assert len(nitrogen) == 1
        assert view.row(nitrogen[0]).environment == "N piridínico"

    def test_el_oxigeno_carbonilo_se_distingue_del_hidroxilo(self):
        carbonyl = describe(_with_vacuum(molecule("H2CO")))
        oxygen = select(carbonyl, element="O")[0]
        assert carbonyl.row(oxygen).environment == "O carbonilo (C=O)"

        alcohol = describe(_with_vacuum(molecule("CH3OH")))
        oxygen = select(alcohol, element="O")[0]
        assert alcohol.row(oxygen).environment == "O hidroxilo (-OH)"

    def test_la_amina_se_reconoce(self):
        """Metilamina a mano: ASE no la trae en su coleccion g2."""
        amine = Atoms(
            "CNH5",
            positions=[
                (0.000, 0.000, 0.000),   # C
                (1.471, 0.000, 0.000),   # N
                (-0.38, 1.020, 0.000),   # H sobre C
                (-0.38, -0.51, 0.883),
                (-0.38, -0.51, -0.883),
                (1.820, -0.46, 0.826),   # H sobre N
                (1.820, -0.46, -0.826),
            ])
        view = describe(_with_vacuum(amine))
        nitrogen = select(view, element="N")[0]
        assert view.row(nitrogen).coordination == 3
        assert view.row(nitrogen).environment == "N de amina (-NH2)"

    def test_sp3_frente_a_sp2(self):
        sp3 = describe(_with_vacuum(molecule("CH4")))
        assert sp3.row(select(sp3, element="C")[0]).environment == "C sp3"
        sp2 = describe(_with_vacuum(molecule("C2H4")))
        assert "sp2" in sp2.row(select(sp2, element="C")[0]).environment

    def test_cada_atomo_esta_en_exactamente_un_entorno(self, benzene):
        view = describe(benzene)
        seen = [i for e in view.environments for i in e.members]
        assert sorted(seen) == list(range(len(benzene)))

    def test_el_representante_pertenece_a_su_grupo(self, benzene):
        view = describe(benzene)
        for environment in view.environments:
            assert environment.representative in environment.members


class TestLaSeleccion:
    def test_por_elemento(self, benzene):
        view = describe(benzene)
        assert len(select(view, element="C")) == 6

    def test_los_criterios_se_combinan_con_y(self, benzene):
        view = describe(benzene)
        assert len(select(view, element="C", ring=6)) == 6
        assert select(view, element="C", ring=5) == ()

    def test_por_coordinacion(self, benzene):
        view = describe(benzene)
        assert len(select(view, coordination=1)) == 6

    def test_por_entorno_es_por_subcadena(self, benzene):
        view = describe(benzene)
        assert len(select(view, environment="sp2")) == 6


class TestLoQueAvisa:
    def test_un_atomo_suelto_se_dice(self, benzene):
        lonely = benzene.copy()
        lonely.append(Atoms("He", positions=[[0.0, 0.0, 0.0]])[0])
        lonely.positions[-1] = lonely.get_cell().diagonal() * 0.02
        view = describe(lonely)
        assert any("sin ningún enlace" in n for n in view.notes)

    def test_la_formula_viene_puesta(self, benzene):
        assert describe(benzene).formula == "C6H6"


def _with_vacuum(atoms: Atoms) -> Atoms:
    atoms = atoms.copy()
    atoms.center(vacuum=6.0)
    return atoms
