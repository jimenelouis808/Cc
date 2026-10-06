"""El catálogo de cálculos, y sobre todo lo que se niega a lanzar.

Un formulario que sólo comprueba que un entero sea un entero deja pasar
las barbaridades que cuestan horas de cola: puntos k sobre un eje que no
se repite, un hueco de core sin decir en qué átomo, bandas sin estado
base. Eso es lo que se prueba aquí.
"""

from __future__ import annotations

import pytest
from ase import Atoms
from ase.build import molecule

from carbonforge.dft.kinds import (
    BASE_PARAMETERS,
    KINDS,
    kind,
    kind_names,
    validate_chain,
)


@pytest.fixture
def finite() -> Atoms:
    """Una molécula con vacío de sobra y ningún eje periódico."""
    atoms = molecule("C6H6")
    atoms.center(vacuum=10.0)
    return atoms


@pytest.fixture
def wire() -> Atoms:
    """Periódica sólo en z, como el coil: vacío en x e y."""
    atoms = molecule("C6H6")
    atoms.center(vacuum=10.0)
    atoms.set_pbc((False, False, True))
    return atoms


class TestElCatalogoEstaBienFormado:
    def test_todo_parametro_tiene_ayuda(self):
        """La ayuda es el motivo de que esto sea datos y no un formulario."""
        for name, calculation in KINDS.items():
            for parameter in calculation.parameters:
                assert parameter.help.strip(), f"{name}.{parameter.name} sin ayuda"
                assert len(parameter.help) > 40, (
                    f"{name}.{parameter.name}: la ayuda no dice nada útil")

    def test_todo_requisito_nombra_un_tipo_real(self):
        for name, calculation in KINDS.items():
            for needed in calculation.requires:
                assert needed in KINDS, f"{name} pide {needed!r}, que no existe"

    def test_todo_tipo_deja_algo_escrito(self):
        for name, calculation in KINDS.items():
            assert calculation.produces, f"{name} no declara qué produce"

    def test_los_valores_por_defecto_se_aceptan_a_si_mismos(self, finite):
        """Un defecto que no pasa su propia validación es una trampa."""
        for name, calculation in KINDS.items():
            values = calculation.defaults()
            if calculation.needs_sites:
                values["sites"] = [0]
            if name == "xps":
                values["spinpol"] = True
            if name in ("bands", "dos"):
                continue  # piden periodicidad o malla k; van en su propia prueba
            problems = [p for p in calculation.check(values, finite)
                        if not p.startswith("Nota de coste")]
            assert not problems, (name, problems)

    def test_pedir_un_tipo_que_no_existe_lo_dice_con_la_lista(self):
        with pytest.raises(KeyError, match="relax"):
            kind("relajacion")


class TestLaValidacionEsFisica:
    def test_puntos_k_sobre_un_eje_que_no_se_repite(self, wire):
        values = kind("scf").defaults() | {"kpts": "4,4,8"}
        problems = kind("scf").check(values, wire)
        assert any("no es periódico" in p for p in problems)
        # y sobre el eje que sí lo es, ninguna queja
        ok = kind("scf").check(kind("scf").defaults() | {"kpts": "1,1,8"}, wire)
        assert not any("periódico" in p for p in ok)

    def test_ondas_planas_con_poco_vacio(self):
        tight = molecule("H2O")
        tight.center(vacuum=3.0)
        values = kind("scf").defaults() | {"mode": "pw"}
        problems = kind("scf").check(values, tight)
        assert any("Hartree" in p for p in problems)

    def test_relajar_la_celda_con_vacio_la_colapsa(self, wire):
        values = kind("relax").defaults() | {"relax_cell": True}
        assert any("colapsaría" in p for p in kind("relax").check(values, wire))

    def test_para_vibraciones_la_fuerza_floja_se_rechaza(self, finite):
        values = kind("relax").defaults() | {"fmax": 0.1, "for_vibrations": True}
        assert any("imaginarias" in p for p in kind("relax").check(values, finite))

    def test_bandas_de_una_molecula(self, finite):
        values = kind("bands").defaults()
        assert any("niveles discretos" in p
                   for p in kind("bands").check(values, finite))

    def test_dos_con_malla_k_gruesa(self, wire):
        values = kind("dos").defaults() | {"kpts": "1,1,2"}
        assert any("artefactos del muestreo" in p
                   for p in kind("dos").check(values, wire))

    def test_ldos_sin_atomos_elegidos(self, finite):
        assert any("sobre qué átomos" in p
                   for p in kind("ldos").check(kind("ldos").defaults(), finite))

    def test_un_atomo_que_no_existe(self, finite):
        values = kind("ldos").defaults() | {"sites": [9999]}
        assert any("no existe" in p for p in kind("ldos").check(values, finite))

    def test_xps_sobre_hidrogeno(self, finite):
        hydrogen = next(i for i, a in enumerate(finite) if a.symbol == "H")
        values = kind("xps").defaults() | {"sites": [hydrogen], "spinpol": True}
        assert any("hidrógeno" in p for p in kind("xps").check(values, finite))

    def test_xps_sin_espin(self, finite):
        values = kind("xps").defaults() | {"sites": [0], "spinpol": False}
        assert any("desapareado" in p for p in kind("xps").check(values, finite))

    def test_raman_resonante_sobre_un_solido(self, wire):
        values = kind("raman_resonant").defaults() | {"sites": [0]}
        assert any("finitos" in p
                   for p in kind("raman_resonant").check(values, wire))

    def test_los_fonones_dicen_cuanto_van_a_costar(self, finite):
        values = kind("phonons").defaults() | {"supercell": "3,1,1"}
        notes = kind("phonons").check(values, finite)
        coste = [n for n in notes if n.startswith("Nota de coste")]
        assert coste and str(6 * len(finite) * 3) in coste[0]


class TestUnaRecetaTieneQueCerrar:
    def test_dos_sin_estado_base(self):
        assert validate_chain(["dos"])

    def test_el_orden_importa(self):
        assert validate_chain(["dos", "scf"])
        assert not validate_chain(["scf", "dos"])

    def test_una_receta_larga_valida(self):
        assert not validate_chain(["relax", "scf", "bands", "dos", "ldos", "xps"])

    def test_un_nombre_inventado(self):
        assert any("no es un tipo" in p for p in validate_chain(["xrd"]))


class TestLosParametrosSeValidanSolos:
    def test_una_eleccion_fuera_de_la_lista(self):
        parameter = kind("scf").parameter("xc")
        assert parameter.check("B3LYP")
        assert parameter.check("PBE") is None

    def test_un_numero_fuera_de_rango(self):
        parameter = kind("scf").parameter("h")
        assert "por debajo" in parameter.check(0.01)
        assert "pasa de" in parameter.check(5.0)

    def test_un_entero_que_no_lo_es(self):
        assert "entero" in kind("bands").parameter("npoints").check(1.5)

    def test_indices_negativos(self):
        assert "no son negativos" in kind("ldos").parameter("sites").check([-1])

    def test_los_compartidos_estan_en_todos(self):
        shared = {p.name for p in BASE_PARAMETERS}
        for name in kind_names():
            assert shared <= {p.name for p in kind(name).parameters}, name
