"""Tests for the code catalogues: advanced parameters, checked and written.

The manuals belong to their projects and are not shipped, so the importers
are tested on small synthetic excerpts written in each manual's own format.
Set ``CARBONFORGE_MANUALS`` to a folder holding ``INPUT_PW.def`` and
``siesta.tex`` to also run them against the real files.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from carbonforge.builders import build_graphene_supercell, build_nanoribbon
from carbonforge.codes import (
    Catalog,
    Parameter,
    import_manual,
    import_qe_helpdoc,
    import_siesta_tex,
    load_catalog,
    save_imported,
)
from carbonforge.exports.qe import QESettings, write_qe_input
from carbonforge.exports.siesta import SiestaSettings, write_siesta
from carbonforge.gui.advanced import check_overrides, describe_parameter, import_documentation
from carbonforge.gui.params import (
    ADVANCED_KEY,
    CALCULATION_PARAMS,
    advanced_overrides,
    check_parameter_constraints,
    export_structure,
    validate_calculation_report,
)

QE_DEF = r"""
input_description -distribution {Quantum ESPRESSO} -package PWscf -program pw.x {
    namelist &CONTROL {
        var calculation -type CHARACTER {
            default { 'scf' }
            options {
                info { A string describing the task to be performed. Options are: }
                opt -val 'scf' {}
                opt -val 'relax' {}
                opt -val 'vc-relax' {}
            }
        }
        var tprnfor -type LOGICAL {
            info { calculate forces }
        }
    }
    namelist &SYSTEM {
        var ecutrho -type REAL {
            default { 4 * @ref ecutwfc }
            info { Kinetic energy cutoff (Ry) for charge density }
        }
        dimension starting_magnetization -start 1 -end ntyp -type REAL {
            default { 0 }
            info { Starting spin polarization on atomic type i }
        }
        var nosym -type LOGICAL {
            default { .FALSE. }
            info { if (.TRUE.) symmetry is not used }
        }
        vargroup -type INTEGER {
            var nr1b
            var nr2b
            info { dimensions of the "box" grid }
        }
    }
    card ATOMIC_POSITIONS {
        flag units -use optional { enum { alat | bohr | angstrom } }
    }
}
"""

SIESTA_TEX = r"""
\begin{fdfentry}{Mesh!Cutoff}[energy]<$300\,\mathrm{Ry}$>
  \index{grid}%

  Defines the plane wave cutoff for the grid.

  % an internal comment
\end{fdfentry}

\begin{fdfentry}{PAO!BasisSize}[string]<DZP>

  It defines usual basis sizes.

  \begin{fdfoptions}
    \option[SZ|minimal] single-zeta
    \option[DZP|standard] double-zeta polarized
  \end{fdfoptions}
\end{fdfentry}

\begin{fdfentry}{SCF.DM!Tolerance}[real]<$10^{-4}$>%
  \fdfindex*{DM.Tolerance}

  Tolerance of Density Matrix.
\end{fdfentry}

\begin{fdflogicalF}{SaveRho}

  Save the density.
\end{fdflogicalF}
"""


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """A private CARBONFORGE_HOME: imports must not touch (or read) the user's."""
    monkeypatch.setenv("CARBONFORGE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def qe_def(tmp_path):
    path = tmp_path / "INPUT_PW.def"
    path.write_text(QE_DEF, encoding="utf-8")
    return path


@pytest.fixture
def siesta_tex(tmp_path):
    path = tmp_path / "siesta.tex"
    path.write_text(SIESTA_TEX, encoding="utf-8")
    return path


def _form(**overrides):
    values = {spec.key: spec.default for spec in CALCULATION_PARAMS}
    values["preset"] = "ninguna"
    values.update(overrides)
    return values


class TestParameter:
    def test_coerce_types(self):
        assert Parameter("qe", "n", kind="integer").coerce(" 4 ") == 4
        assert Parameter("qe", "x", kind="real").coerce("1.5d-3") == pytest.approx(1.5e-3)
        assert Parameter("qe", "b", kind="logical").coerce(".TRUE.") is True
        assert Parameter("siesta", "m", kind="quantity").coerce("400,5 Ry") == "400.5 Ry"
        choice = Parameter("qe", "o", kind="string", choices=("'smearing'", "'fixed'"))
        assert choice.coerce("SMEARING") == "smearing"

    @pytest.mark.parametrize("kind,value", [
        ("integer", "4.5"), ("real", "abc"), ("logical", "maybe"), ("quantity", "Ry 300"),
    ])
    def test_coerce_rejects(self, kind, value):
        with pytest.raises(ValueError, match="'k'"):
            Parameter("qe", "k", kind=kind).coerce(value)

    def test_gpaw_python_values(self):
        occupations = Parameter("gpaw", "occupations")
        assert occupations.coerce("{'name': 'fermi-dirac', 'width': 0.05}") == {
            "name": "fermi-dirac", "width": 0.05}
        assert occupations.coerce("fermi") == "fermi"

    def test_siesta_names_ignore_dots_and_case(self):
        catalog = load_catalog("siesta")
        assert catalog.get("Mesh.Cutoff") is catalog.get("meshcutoff")


class TestCatalogCheck:
    def test_curated_unknown_is_a_warning(self, home):
        typed, report = load_catalog("qe").check({"system.no_such_thing": "1"})
        assert report.ok and report.warnings
        assert typed == {"system.no_such_thing": "1"}

    def test_documented_unknown_is_an_error(self, home, qe_def):
        save_imported(import_qe_helpdoc(qe_def))
        _, report = load_catalog("qe").check({"system.no_such_thing": "1"})
        assert not report.ok

    def test_managed_override_is_reported(self, home):
        typed, report = load_catalog("qe").check({"ecutwfc": "70"})
        assert typed == {"system.ecutwfc": 70.0}
        assert any("carbonforge" in w for w in report.warnings)

    def test_bad_choice(self, home):
        _, report = load_catalog("qe").check({"system.occupations": "sometimes"})
        assert not report.ok and "Opciones" in report.errors[0]


class TestQEHelpdoc:
    def test_reads_namelists(self, qe_def):
        catalog = import_qe_helpdoc(qe_def)
        calc = catalog.get("calculation", "control")
        assert calc.kind == "string" and calc.choices == ("scf", "relax", "vc-relax")
        assert calc.default == "'scf'"
        assert catalog.get("ecutrho", "system").default == "4 * ecutwfc"
        assert catalog.get("starting_magnetization(i)", "system").kind == "real"
        assert catalog.get("nosym").kind == "logical"
        assert catalog.get("nr2b").kind == "integer"
        assert catalog.get("units") is None                      # cards are not read

    def test_merge_keeps_managed(self, home, qe_def):
        save_imported(import_qe_helpdoc(qe_def))
        merged = load_catalog("qe")
        assert merged.documented
        assert merged.get("ecutrho").managed                      # curated flag kept
        assert merged.get("ecutrho").default == "4 * ecutwfc"     # manual text wins

    def test_rejects_other_files(self, tmp_path):
        bad = tmp_path / "x.def"
        bad.write_text("nothing here", encoding="utf-8")
        with pytest.raises(ValueError, match="namelist"):
            import_qe_helpdoc(bad)


class TestSiestaTex:
    def test_reads_entries(self, siesta_tex):
        catalog = import_siesta_tex(siesta_tex)
        mesh = catalog.get("Mesh.Cutoff")
        assert mesh.name == "Mesh.Cutoff" and mesh.kind == "quantity"
        assert mesh.default == "300 Ry"
        assert mesh.description == "Defines the plane wave cutoff for the grid."
        basis = catalog.get("PAO.BasisSize")
        assert basis.choices == ("SZ", "minimal", "DZP", "standard")
        assert basis.description == "It defines usual basis sizes."
        assert catalog.get("SCF.DM.Tolerance").default == "1e-4"
        assert catalog.get("DM.Tolerance").description.startswith("Alias de SCF.DM.Tolerance")
        assert catalog.get("SaveRho").default == "false"

    def test_import_manual_dispatch(self, qe_def, siesta_tex, tmp_path):
        assert import_manual(qe_def).code == "qe"
        assert import_manual(siesta_tex).code == "siesta"
        with pytest.raises(ValueError, match="formato"):
            import_manual(tmp_path / "manual.pdf")

    def test_import_documentation_checks_the_code(self, home, siesta_tex):
        with pytest.raises(ValueError, match="SIESTA"):
            import_documentation("qe", siesta_tex)
        catalog, stored = import_documentation("siesta", siesta_tex)
        assert stored.parent == home / "catalogos" and catalog.documented


class TestWriters:
    def test_qe_extra_overrides_and_adds(self, tmp_path):
        sheet = build_graphene_supercell(3, 3)
        settings = QESettings(extra={"system.nosym": True, "system.ecutwfc": 75.0,
                                     "electrons.electron_maxstep": "300"})
        text = write_qe_input(sheet, tmp_path, settings=settings).read_text()
        assert "nosym = .true." in text
        assert text.count("ecutwfc") == 1 and "ecutwfc = 75" in text
        assert "electron_maxstep = 300" in text                  # a number, not '300'

    def test_qe_extra_for_unwritten_namelist(self, tmp_path):
        sheet = build_graphene_supercell(3, 3)
        with pytest.raises(ValueError, match="IONS"):
            write_qe_input(sheet, tmp_path, settings=QESettings(extra={"ions.upscale": 10}))

    def test_siesta_extra_replaces_line(self, tmp_path):
        sheet = build_graphene_supercell(3, 3)
        path = write_siesta(sheet, tmp_path, settings=SiestaSettings(
            extra={"Mesh.Cutoff": "450 Ry", "SaveRho": True}))
        text = path.read_text()
        assert "MeshCutoff" not in text.replace("Mesh.Cutoff", "")
        assert "Mesh.Cutoff        450 Ry" in text and "SaveRho            .true." in text


class TestGuiPlumbing:
    def test_overrides_reach_the_input(self, tmp_path, home):
        sheet = build_graphene_supercell(3, 3)
        values = _form(**{ADVANCED_KEY: {"qe": {"system.nosym": "true"},
                                         "siesta": {"Mesh.Cutoff": "400 Ry"}}})
        export_structure(sheet, tmp_path, ["qe", "siesta"], calculation_values=values)
        qe = "\n".join(p.read_text() for p in (tmp_path / "qe").rglob("*.in"))
        assert "nosym = .true." in qe
        assert "400 Ry" in (tmp_path / "siesta" / "input.fdf").read_text()

    def test_invalid_override_blocks_export(self, tmp_path, home):
        values = _form(**{ADVANCED_KEY: {"qe": {"system.nbnd": "many"}}})
        with pytest.raises(ValueError, match="nbnd"):
            export_structure(build_graphene_supercell(3, 3), tmp_path, ["qe"],
                             calculation_values=values)

    def test_namelist_must_be_written(self, home):
        _, report = advanced_overrides(
            _form(task="scf", **{ADVANCED_KEY: {"qe": {"ions.ion_dynamics": "bfgs"}}}), "qe")
        assert any("&IONS" in e for e in report.errors)

    def test_override_is_what_gets_validated(self, home):
        """occupations='fixed' typed as an override on a metal is still caught."""
        sheet = build_graphene_supercell(3, 3)
        values = _form(**{ADVANCED_KEY: {"qe": {"system.occupations": "fixed"}}})
        report = validate_calculation_report(sheet, values)
        assert not report.ok and any("fixed" in e for e in report.errors)

    def test_constraint_text_lists_overrides(self, home):
        text = check_parameter_constraints(None, _form(**{ADVANCED_KEY: {
            "qe": {"system.nosym": "yes"}}}))
        assert "Parámetros avanzados (qe)" in text and "nosym = True" in text

    def test_zigzag_report_unchanged_without_overrides(self):
        ribbon = build_nanoribbon(6, 4, edge="zigzag", passivate=True)
        assert validate_calculation_report(ribbon, _form()).ok


class TestDialogLogic:
    def test_describe(self):
        text = describe_parameter(load_catalog("qe").get("ecutrho"))
        assert "&SYSTEM" in text and "carbonforge lo decide" in text

    def test_check_overrides_messages(self, home):
        typed, messages = check_overrides("siesta", {"Mesh.Cutoff": "300 Ry",
                                                     "PAO.BasisSize": "huge"})
        assert typed["MeshCutoff"] == "300 Ry" or typed.get("Mesh.Cutoff") == "300 Ry"
        assert messages[0].startswith("ERROR")


class TestVibspecExtras:
    def test_extra_reaches_gpaw_kwargs(self):
        from carbonforge.vibspec.core.calcspec import CalcSpec
        from carbonforge.vibspec.core.engines import gpaw_parameters

        spec = CalcSpec(extra={"maxiter": 500, "symmetry": {"point_group": True}})
        params = gpaw_parameters(spec, spinpol=False)
        assert params["maxiter"] == 500 and params["symmetry"] == "off"
        assert any("simetría" in e for e in spec.validate().errors)

    def test_form_checks_advanced(self):
        from carbonforge.vibspec.gui import logic

        raw = {spec.key: spec.default for spec in logic.CALC_PARAMS}
        spec = logic.spec_from_form(raw, {"maxiter": "400"})
        assert spec.extra == {"maxiter": 400}
        with pytest.raises(ValueError, match="maxiter"):
            logic.spec_from_form(raw, {"maxiter": "lots"})
        assert CalcSpecRoundTrip.ok(spec)


class CalcSpecRoundTrip:
    @staticmethod
    def ok(spec) -> bool:
        from carbonforge.vibspec.core.calcspec import CalcSpec

        return CalcSpec.from_dict(spec.to_dict()) == spec


_MANUALS = Path(os.environ.get("CARBONFORGE_MANUALS", "/nonexistent"))


@pytest.mark.skipif(not (_MANUALS / "INPUT_PW.def").exists(), reason="sin manual de QE")
def test_real_qe_manual():
    catalog = import_qe_helpdoc(_MANUALS / "INPUT_PW.def")
    assert len(catalog.parameters) > 200
    assert "smearing" in catalog.get("occupations").choices
    assert "2Dxy" in catalog.get("cell_dofree").choices


@pytest.mark.skipif(not (_MANUALS / "siesta.tex").exists(), reason="sin manual de SIESTA")
def test_real_siesta_manual():
    catalog = import_siesta_tex(_MANUALS / "siesta.tex")
    assert len(catalog.parameters) > 400
    assert catalog.get("Mesh.Cutoff").default == "300 Ry"
    assert "polarized" in catalog.get("Spin").choices


def test_catalog_json_round_trip():
    catalog = load_catalog("siesta")
    again = Catalog.from_json(catalog.to_json())
    assert again.parameters == catalog.parameters
