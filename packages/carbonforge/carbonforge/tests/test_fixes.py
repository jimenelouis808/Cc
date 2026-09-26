"""Tests for fixable warnings: structured remedies attached to reported problems.

A warning the user cannot act on from where they are is noise. Each check
that knows its cure attaches it as a :class:`Fix` (a setting and a value);
the GUI offers a button, and applying it must make the problem go away --
and must change what is actually written to disk.
"""

from __future__ import annotations

import pytest
from ase import Atoms

from carbonforge.builders import build_cnt, build_graphene_supercell, build_nanoribbon
from carbonforge.calculations.electronic import setup_antiferromagnetic_edges
from carbonforge.calculations.spectroscopy import raman_setup
from carbonforge.gui.constraints import check_constraints
from carbonforge.gui.params import (
    CALCULATION_PARAMS,
    apply_fix,
    collect_fixes,
    export_structure,
    validate_calculation_report,
)
from carbonforge.validation.calculations import (
    check_calculation_type,
    check_electronic_setup,
    check_spectroscopy,
)
from carbonforge.validation.checks import Fix, ValidationReport


def _form(**overrides):
    values = {spec.key: spec.default for spec in CALCULATION_PARAMS}
    values["preset"] = "ninguna"
    values.update(overrides)
    return values


@pytest.fixture(scope="module")
def zigzag():
    return build_nanoribbon(6, 4, edge="zigzag", passivate=True)


class TestReport:
    def test_merge_deduplicates(self):
        a, b = ValidationReport(), ValidationReport()
        fix = Fix("spin", "afm_edges", "x")
        a.fixes.append(fix)
        b.fixes.append(fix)
        a.merge(b)
        assert a.fixes == [fix]
        assert "CORRECCIONES POSIBLES" in a.summary()


class TestChecksCarryTheirCure:
    def test_zigzag_without_spin(self, zigzag):
        report = check_electronic_setup(zigzag, None)
        assert Fix("spin", "afm_edges", "Activar espín antiferromagnético en los bordes") \
            in report.fixes
        assert "--spin afm" not in " ".join(report.errors)      # a flag that never existed

    @pytest.mark.parametrize("atoms,expected", [
        (build_graphene_supercell(3, 3), "2Dxy"),
        (build_cnt(6, 0, length=8), "z"),
    ])
    def test_vc_relax_cell_dofree(self, atoms, expected):
        report = check_calculation_type(atoms, "vc-relax")
        assert [f.value for f in report.fixes if f.setting == "cell_dofree"] == [expected]

    def test_vc_relax_molecule(self):
        molecule = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]], cell=[10, 10, 10])
        report = check_calculation_type(molecule, "vc-relax")
        assert any(f.setting == "task" and f.value == "relax" for f in report.fixes)

    def test_raman_with_paw(self):
        sheet = build_graphene_supercell(3, 3)
        report = check_spectroscopy(sheet, raman_setup(), {"C": "C.pbe-n-kjpaw_psl.1.0.0.UPF"})
        assert any(f.setting == "pseudo_family" and f.value == "NC" for f in report.fixes)


class TestAntiferromagneticEdges:
    def test_passivated_zigzag_edges_are_found(self, zigzag):
        """Regression: H-terminated edges used to be invisible here."""
        tagged, spec = setup_antiferromagnetic_edges(zigzag)
        assert spec.is_spin_polarized
        tags = set(tagged.get_tags())
        assert {1, 2} <= tags


class TestGuiFixes:
    def test_auto_spin_leaves_nothing_to_fix(self, zigzag):
        report = validate_calculation_report(zigzag, _form())
        assert report.ok, report.summary()

    def test_fix_round_trip(self, zigzag):
        values = _form(spin="none", task="raman")
        fixes = collect_fixes(zigzag, values)
        assert {(f.setting, f.value) for _, _, f in fixes} >= {("spin", "afm_edges")}
        for _, _, fix in fixes:
            values = apply_fix(values, fix)
        assert values["spin"] == "afm_edges" and values["task"] == "fonones"
        assert validate_calculation_report(zigzag, values).ok
        assert collect_fixes(zigzag, values) == []

    def test_fixes_are_unique(self, zigzag):
        fixes = collect_fixes(zigzag, _form(spin="none", task="raman"))
        keys = [(f.setting, f.value) for _, _, f in fixes]
        assert len(keys) == len(set(keys))

    def test_apply_fix_maps_labels_and_rejects_unknown(self):
        assert apply_fix(_form(), Fix("task", "phonon", ""))["task"] == "fonones"
        with pytest.raises(KeyError):
            apply_fix(_form(), Fix("hubbard", "none", ""))

    def test_cutoff_constraint(self):
        assert not check_constraints(_form(ecutwfc=60, ecutrho=0), None)
        error = [v for v in check_constraints(_form(ecutwfc=60, ecutrho=200), None) if v.blocking]
        assert error and error[0].fix.value == 0.0
        assert not check_constraints(_form(ecutwfc=60, ecutrho=300, pseudo_family="NC"), None)
        warning = check_constraints(_form(ecutwfc=60, ecutrho=300), None)
        assert warning and warning[0].fix.value == 480.0


class TestWhatIsWritten:
    """A fix is only real if the exported input changes."""

    def _read_qe(self, directory):
        return "\n".join(p.read_text(encoding="utf-8") for p in directory.rglob("*.in"))

    def test_afm_edges_reach_the_input(self, zigzag, tmp_path):
        export_structure(zigzag, tmp_path, ["qe"], calculation_values=_form(spin="afm_edges"))
        text = self._read_qe(tmp_path)
        assert "nspin" in text and "starting_magnetization" in text
        assert "C1" in text and "C2" in text

    def test_raman_writes_norm_conserving(self, tmp_path):
        tube = build_cnt(7, 0, length=8)           # semiconducting: Raman is allowed
        export_structure(tube, tmp_path, ["qe"], calculation_values=_form(task="raman"),
                         force=True)
        text = self._read_qe(tmp_path)
        assert "ONCV" in text and "kjpaw" not in text
        assert "ecutrho" in text and "240" in text             # 4 x 60 Ry for NC

    def test_vdw_and_occupations(self, tmp_path):
        sheet = build_graphene_supercell(3, 3)
        export_structure(sheet, tmp_path, ["qe"],
                         calculation_values=_form(vdw="grimme-d3", degauss=0.02))
        text = self._read_qe(tmp_path)
        assert "grimme-d3" in text.lower() or "dft-d3" in text.lower()
        assert "0.02" in text
