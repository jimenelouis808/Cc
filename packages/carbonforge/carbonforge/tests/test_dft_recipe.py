"""Recetas: que cierren antes de lanzarse, y que el directorio salga entero."""

from __future__ import annotations

import ast
import json

import pytest
from ase.build import molecule

from carbonforge.dft.recipe import (
    GPAW_PROGRAM,
    RECIPE,
    STRUCTURE,
    Recipe,
    RecipeStep,
    script_for,
    write,
)
from carbonforge.jobs.manifest import JobManifest


@pytest.fixture
def benzene():
    atoms = molecule("C6H6")
    atoms.center(vacuum=6.0)
    return atoms


@pytest.fixture
def basic() -> Recipe:
    return Recipe("prueba", [
        RecipeStep("relax", {"steps": 5}),
        RecipeStep("scf"),
        RecipeStep("dos", {"kpts": "1,1,1"}),
    ])


class TestLaRecetaSeValidaAntes:
    def test_una_cadena_que_no_cierra(self, benzene):
        recipe = Recipe("mala", [RecipeStep("dos")])
        assert any("Estado base" in p for p in recipe.check(benzene))

    def test_un_paso_con_un_valor_imposible(self, benzene):
        recipe = Recipe("mala", [RecipeStep("relax", {"fmax": -1.0})])
        problems = recipe.check(benzene)
        assert any("Paso 1 (relax)" in p for p in problems)

    def test_una_buena_no_tiene_quejas(self, basic, benzene):
        assert basic.check(benzene) == []

    def test_escribir_una_mala_se_niega(self, benzene, tmp_path):
        recipe = Recipe("mala", [RecipeStep("bands")])
        with pytest.raises(ValueError, match="no se puede correr"):
            write(recipe, benzene, tmp_path / "x")

    def test_los_valores_que_faltan_se_rellenan(self):
        step = RecipeStep("relax", {"fmax": 0.01})
        filled = step.filled()
        assert filled["fmax"] == 0.01
        assert filled["xc"] == "PBE"      # del catálogo

    def test_ida_y_vuelta_por_json(self, basic):
        again = Recipe.from_json(basic.to_json())
        assert again.name == basic.name
        assert [s.kind for s in again.steps] == [s.kind for s in basic.steps]


class TestElDirectorioQueDeja:
    def test_estan_todos_los_ficheros(self, basic, benzene, tmp_path):
        directory = write(basic, benzene, tmp_path / "corrida")
        names = {p.name for p in directory.iterdir()}
        assert {STRUCTURE, RECIPE, "job.json"} <= names
        assert {"01_relax.py", "02_scf.py", "03_dos.py"} <= names

    def test_un_paso_del_manifiesto_por_cada_uno_de_la_receta(
            self, basic, benzene, tmp_path):
        directory = write(basic, benzene, tmp_path / "corrida")
        manifest = JobManifest.load(directory)
        assert manifest.engine == "gpaw"
        assert [s.name for s in manifest.steps] == ["relax", "scf", "dos"]
        assert {s.program for s in manifest.steps} == {GPAW_PROGRAM}

    def test_lo_que_sigue_a_una_relajacion_lee_la_geometria_relajada(
            self, basic, benzene, tmp_path):
        """El error que arruina una cadena entera: calcular el estado base
        sobre la geometría de entrada y no sobre la que acaba de salir."""
        directory = write(basic, benzene, tmp_path / "corrida")
        assert STRUCTURE in (directory / "01_relax.py").read_text()
        assert "relaxed.xyz" in (directory / "02_scf.py").read_text()

    def test_sin_relajacion_todos_leen_la_de_entrada(self, benzene, tmp_path):
        recipe = Recipe("sin relax", [RecipeStep("scf")])
        directory = write(recipe, benzene, tmp_path / "corrida")
        assert STRUCTURE in (directory / "01_scf.py").read_text()

    def test_la_receta_se_guarda_tal_cual(self, basic, benzene, tmp_path):
        directory = write(basic, benzene, tmp_path / "corrida")
        saved = json.loads((directory / RECIPE).read_text())
        assert saved["name"] == "prueba"
        assert saved["steps"][0]["values"]["steps"] == 5

    def test_la_estructura_se_guarda_sin_el_info(self, benzene, tmp_path):
        """El info de los constructores lleva anillos y enlaces, y hace que
        el extxyz no se pueda volver a leer."""
        dirty = benzene.copy()
        dirty.info = {"rings": [[0, 1, 2]] * 50, "bonds": [(0, 1)] * 50}
        recipe = Recipe("x", [RecipeStep("scf")])
        directory = write(recipe, dirty, tmp_path / "corrida")
        from ase.io import read
        assert len(read(str(directory / STRUCTURE))) == len(benzene)


class TestLosScripts:
    @pytest.mark.parametrize(
        "name", ["relax", "scf", "bands", "dos", "ldos", "phonons"])
    def test_cada_uno_es_python_valido(self, name):
        values = {"sites": [0]} if name == "ldos" else {}
        ast.parse(script_for(RecipeStep(name, values)))

    def test_ondas_planas_traen_su_corte_y_su_import(self):
        source = script_for(RecipeStep("scf", {"mode": "pw", "ecut": 600.0}))
        assert "from gpaw import PW" in source
        assert "PW(600)" in source

    def test_lcao_lleva_base_y_malla(self):
        source = script_for(RecipeStep("scf", {"mode": "lcao"}))
        assert "basis='dzp'" in source and "h=0.18" in source
        assert "PW(" not in source

    def test_la_relajacion_se_niega_a_mentir_si_no_convergio(self):
        """Un paso que acaba sin converger y dice que sí arruina lo que
        venga detrás."""
        source = script_for(RecipeStep("relax"))
        assert "SystemExit" in source and "NO es un minimo" in source

    def test_los_fonones_avisan_de_los_modos_imaginarios(self):
        assert "imaginarios" in script_for(RecipeStep("phonons"))

    def test_el_ldos_lleva_los_sitios_elegidos(self):
        source = script_for(RecipeStep("ldos", {"sites": [3, 7]}))
        assert "sites = [3, 7]" in source

    @pytest.mark.parametrize("name", ["xps", "raman_resonant"])
    def test_los_que_aun_no_se_generan_lo_dicen_por_su_nombre(self, name):
        with pytest.raises(NotImplementedError, match=name):
            script_for(RecipeStep(name, {"sites": [0]}))
