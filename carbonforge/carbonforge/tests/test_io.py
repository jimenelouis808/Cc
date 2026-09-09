"""Tests for structure import, repair, and pseudopotential cataloguing."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

from carbonforge.builders import build_graphene_supercell, build_nanoribbon
from carbonforge.dopants import dope_random
from carbonforge.io import (
    add_missing_cell,
    autofix,
    diagnose,
    download_instructions,
    import_structure,
    match_requirements,
    read_upf_header,
    remove_duplicate_atoms,
    scan_directory,
    suggest_periodicity,
)
from carbonforge.validation import run_basic_checks

# --- UPF fixtures, both header layouts -------------------------------------

UPF_V2_PAW = """<UPF version="2.0.1">
  <PP_HEADER element="C" pseudo_type="PAW" relativistic="scalar"
     is_ultrasoft="F" is_paw="T" functional="PBE" z_valence="4.000"
     wfc_cutoff="4.5E+1" rho_cutoff="3.2E+2"/>
</UPF>"""

UPF_V2_NC = """<UPF version="2.0.1">
  <PP_HEADER element="C" pseudo_type="NC" relativistic="scalar"
     functional="PBE" z_valence="4.000" wfc_cutoff="8.0E+1"/>
</UPF>"""

UPF_V2_REL = """<UPF version="2.0.1">
  <PP_HEADER element="N" pseudo_type="NC" relativistic="full"
     functional="PBE" z_valence="5.000"/>
</UPF>"""

UPF_V1 = """<PP_INFO></PP_INFO>
<PP_HEADER>
   0                   Version Number
   N                   Element
   NC                  Norm-Conserving pseudopotential
   T                   Nonlinear Core Correction
 SLA  PW   PBX  PBC    PBE  Exchange-Correlation functional
   5.00000000000      Z valence
  30.00000000000      Suggested cutoff for wavefunctions
 120.00000000000      Suggested cutoff for charge density
</PP_HEADER>"""


@pytest.fixture()
def pseudo_dir(tmp_path):
    (tmp_path / "C.pbe-n-kjpaw_psl.1.0.0.UPF").write_text(UPF_V2_PAW)
    (tmp_path / "C_ONCV_PBE-1.2.upf").write_text(UPF_V2_NC)
    (tmp_path / "N.rel-pbe-nc.UPF").write_text(UPF_V2_REL)
    (tmp_path / "N.pbe-vbc.UPF").write_text(UPF_V1)
    (tmp_path / "roto.UPF").write_text("no soy un UPF")
    return tmp_path


class TestUPFHeaders:
    def test_reads_v2_attributes(self, pseudo_dir):
        info = read_upf_header(pseudo_dir / "C.pbe-n-kjpaw_psl.1.0.0.UPF")
        assert info.element == "C"
        assert info.family == "PAW"
        assert info.functional == "PBE"
        assert info.z_valence == pytest.approx(4.0)
        assert info.upf_version == 2

    def test_reads_scientific_notation_cutoffs(self, pseudo_dir):
        info = read_upf_header(pseudo_dir / "C.pbe-n-kjpaw_psl.1.0.0.UPF")
        assert info.suggested_wfc_cutoff == pytest.approx(45.0)
        assert info.suggested_rho_cutoff == pytest.approx(320.0)

    def test_reads_v1_labelled_block(self, pseudo_dir):
        info = read_upf_header(pseudo_dir / "N.pbe-vbc.UPF")
        assert info.element == "N"
        assert info.family == "NC"
        assert info.upf_version == 1
        assert info.suggested_wfc_cutoff == pytest.approx(30.0)

    def test_detects_full_relativity(self, pseudo_dir):
        assert read_upf_header(pseudo_dir / "N.rel-pbe-nc.UPF").is_fully_relativistic
        assert not read_upf_header(pseudo_dir / "C_ONCV_PBE-1.2.upf").is_fully_relativistic

    def test_family_drives_raman_support(self, pseudo_dir):
        assert read_upf_header(pseudo_dir / "C_ONCV_PBE-1.2.upf").supports_raman
        assert not read_upf_header(
            pseudo_dir / "C.pbe-n-kjpaw_psl.1.0.0.UPF"
        ).supports_raman

    def test_header_beats_a_misleading_filename(self, tmp_path):
        """The point of reading the file: names can lie, headers cannot."""
        path = tmp_path / "C.pbe-n-kjpaw_psl.1.0.0.UPF"  # looks like PAW
        path.write_text(UPF_V2_NC)                        # actually NC
        assert read_upf_header(path).family == "NC"

    def test_rejects_non_upf(self, tmp_path):
        path = tmp_path / "x.UPF"
        path.write_text("cualquier cosa")
        with pytest.raises(ValueError, match="PP_HEADER"):
            read_upf_header(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="no existe"):
            read_upf_header(tmp_path / "nope.UPF")


class TestCatalog:
    def test_scan_finds_everything_readable(self, pseudo_dir):
        catalog = scan_directory(pseudo_dir)
        assert len(catalog) == 4
        assert catalog.elements == ["C", "N"]

    def test_unreadable_files_do_not_abort_the_scan(self, pseudo_dir):
        catalog = scan_directory(pseudo_dir)
        assert len(catalog.unreadable) == 1

    def test_filter_by_family_and_relativity(self, pseudo_dir):
        catalog = scan_directory(pseudo_dir)
        assert len(catalog.for_element("C", family="NC")) == 1
        assert len(catalog.for_element("N", relativistic=True)) == 1

    def test_match_picks_a_suitable_file(self, pseudo_dir):
        catalog = scan_directory(pseudo_dir)
        atoms = dope_random(build_graphene_supercell(3, 3), "N", 0.1, seed=0)
        match = match_requirements(catalog, atoms)
        assert match.ok
        assert set(match.resolved) == {"C", "N"}

    def test_raman_selects_norm_conserving_automatically(self, pseudo_dir):
        catalog = scan_directory(pseudo_dir)
        atoms = build_graphene_supercell(3, 3)
        match = match_requirements(catalog, atoms, needs_raman=True)
        assert match.resolved["C"].family == "NC"

    def test_unsuitable_is_distinguished_from_missing(self, pseudo_dir):
        """'You have none' and 'yours is the wrong kind' need different fixes."""
        catalog = scan_directory(pseudo_dir)
        atoms = build_graphene_supercell(3, 3)
        match = match_requirements(catalog, atoms, needs_soc=True)
        assert not match.ok
        assert "C" in match.unsuitable
        assert any("relativista" in reason for _, reason in match.unsuitable["C"])

    def test_cutoff_hint_takes_the_strictest_file(self, pseudo_dir):
        catalog = scan_directory(pseudo_dir)
        atoms = build_graphene_supercell(3, 3)
        match = match_requirements(catalog, atoms, needs_raman=True)
        # The NC carbon asks for 80 Ry, the highest of the chosen files.
        assert match.cutoff_hint == pytest.approx(80.0)

    def test_pseudopotential_map_is_usable_by_qe_settings(self, pseudo_dir):
        catalog = scan_directory(pseudo_dir)
        mapping = match_requirements(
            catalog, build_graphene_supercell(2, 2)
        ).pseudopotential_map()
        assert mapping["C"].endswith((".UPF", ".upf"))

    def test_download_instructions_name_the_right_table(self):
        text = download_instructions(["C"], needs_raman=True, needs_soc=True)
        assert "nc-fr" in text

    def test_scan_rejects_a_file_path(self, pseudo_dir):
        with pytest.raises(ValueError, match="no es un directorio"):
            scan_directory(pseudo_dir / "roto.UPF")


class TestImport:
    def test_reads_extended_xyz_with_its_cell(self, tmp_path):
        from ase.io import write as ase_write

        original = build_graphene_supercell(3, 3)
        path = tmp_path / "sheet.xyz"
        ase_write(path, original)
        result = import_structure(path)
        assert len(result.atoms) == len(original)

    def test_plain_xyz_has_no_cell_and_says_so(self, tmp_path):
        path = tmp_path / "molecule.xyz"
        path.write_text(
            "2\nsin celda\nC 0.0 0.0 0.0\nC 0.0 0.0 1.42\n"
        )
        result = import_structure(path)
        assert not result.ok
        assert any(i.code == "no_cell" for i in result.issues)

    def test_duplicates_are_detected(self, tmp_path):
        from ase.io import write as ase_write

        atoms = build_graphene_supercell(3, 3)
        atoms += atoms[[0, 1]]
        path = tmp_path / "dup.xyz"
        ase_write(path, atoms)
        result = import_structure(path)
        assert any(i.code == "duplicates" for i in result.issues)

    def test_thin_vacuum_warns(self, tmp_path):
        from ase.io import write as ase_write

        atoms = build_graphene_supercell(3, 3)
        cell = np.array(atoms.cell)
        cell[2, 2] = 4.0  # far too little
        atoms.set_cell(cell)
        path = tmp_path / "thin.xyz"
        ase_write(path, atoms)
        result = import_structure(path)
        assert any(i.code.startswith("thin_vacuum") for i in result.issues)

    def test_unknown_elements_warn(self, tmp_path):
        from ase.io import write as ase_write

        atoms = Atoms("Fe2", positions=[[0, 0, 0], [0, 0, 2.5]],
                      cell=np.eye(3) * 20, pbc=False)
        path = tmp_path / "iron.xyz"
        ase_write(path, atoms)
        result = import_structure(path)
        assert any(i.code == "unknown_elements" for i in result.issues)

    def test_missing_file_names_the_formats(self, tmp_path):
        with pytest.raises(ValueError, match="no existe"):
            import_structure(tmp_path / "nope.cif")

    def test_unreadable_file_explains_itself(self, tmp_path):
        path = tmp_path / "bad.cif"
        path.write_text("esto no es un CIF")
        with pytest.raises(ValueError, match="Extensiones reconocidas"):
            import_structure(path)


class TestAutofix:
    def test_adds_a_missing_cell(self):
        atoms = Atoms("CH4", positions=[
            [0, 0, 0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6],
            [0.6, -0.6, -0.6], [-0.6, 0.6, -0.6],
        ])
        fixed = autofix(atoms)
        assert abs(np.linalg.det(np.array(fixed.atoms.cell))) > 1.0
        assert any(r.code == "no_cell" for r in fixed.applied)

    def test_repaired_molecule_becomes_exportable(self):
        atoms = Atoms("C2", positions=[[0, 0, 0], [0, 0, 1.42]])
        fixed = autofix(atoms)
        assert run_basic_checks(fixed.atoms).ok

    def test_removes_duplicates(self):
        atoms = build_graphene_supercell(3, 3)
        n_original = len(atoms)
        atoms += atoms[[0, 1, 2]]
        fixed = autofix(atoms)
        assert len(fixed.atoms) == n_original

    def test_duplicate_removal_keeps_the_first(self):
        atoms = Atoms("C2", positions=[[0, 0, 0], [0.01, 0, 0]],
                      cell=np.eye(3) * 20)
        cleaned, removed = remove_duplicate_atoms(atoms)
        assert removed == 1
        assert len(cleaned) == 1

    def test_overlap_is_reported_but_never_moved(self):
        """Nudging atoms apart would invent a structure the user never had."""
        atoms = Atoms("C2", positions=[[0, 0, 0], [0.5, 0, 0]],
                      cell=np.eye(3) * 20, pbc=False)
        fixed = autofix(atoms)
        assert any(code == "overlap" for code, _ in fixed.skipped)
        # Positions untouched.
        np.testing.assert_allclose(
            fixed.atoms.get_positions(), atoms.get_positions()
        )

    def test_added_cell_is_left_non_periodic(self):
        """Assuming periodicity would be the riskier guess."""
        atoms = Atoms("C2", positions=[[0, 0, 0], [0, 0, 1.42]])
        fixed = add_missing_cell(atoms)
        assert not any(fixed.get_pbc())

    def test_periodicity_guess_follows_how_atoms_fill_the_cell(self):
        sheet = build_graphene_supercell(3, 3)
        guess = suggest_periodicity(sheet)
        assert guess[0] and guess[1]      # atoms tile x and y
        assert not guess[2]               # z is vacuum

    def test_wrapping_applied_when_periodic(self):
        atoms = build_graphene_supercell(3, 3)
        positions = atoms.get_positions()
        positions[0] += np.array([atoms.cell[0][0] * 2, 0, 0])
        atoms.set_positions(positions)
        issues = diagnose(atoms)
        assert any(i.code == "outside_cell" for i in issues)
        fixed = autofix(atoms, issues)
        assert any(r.code == "outside_cell" for r in fixed.applied)

    def test_thin_vacuum_is_grown(self):
        """The guarantee is on the vacuum gap, not the cell length.

        A flat sheet spans nothing along z, so a 15 Å vacuum means a 15 Å
        cell exactly — asserting the cell is *larger* than 15 would be
        testing something the function never promised.
        """
        atoms = build_graphene_supercell(3, 3)
        cell = np.array(atoms.cell)
        cell[2, 2] = 4.0
        atoms.set_cell(cell)

        fixed = autofix(atoms, vacuum=15.0)
        positions = fixed.atoms.get_positions()
        span = float(np.ptp(positions[:, 2]))
        gap = float(np.array(fixed.atoms.cell)[2, 2]) - span
        assert gap >= 15.0 - 1e-6

    def test_clean_structure_needs_nothing(self):
        fixed = autofix(build_graphene_supercell(3, 3))
        assert not fixed.applied
        assert "No hizo falta" in fixed.summary()

    def test_summary_reports_what_was_declined(self):
        atoms = Atoms("C2", positions=[[0, 0, 0], [0.5, 0, 0]],
                      cell=np.eye(3) * 20, pbc=False)
        assert "inventaría" in autofix(atoms).summary()


class TestConstraints:
    def test_cutoff_ratio_below_four_is_an_error(self):
        from carbonforge.gui.constraints import check_constraints

        violations = check_constraints({"ecutwfc": 60, "ecutrho": 150})
        assert any(v.blocking for v in violations)

    def test_cutoff_ratio_between_four_and_eight_warns(self):
        from carbonforge.gui.constraints import check_constraints

        violations = check_constraints({"ecutwfc": 60, "ecutrho": 300})
        assert violations and not any(v.blocking for v in violations)

    def test_standard_dual_is_clean(self):
        from carbonforge.gui.constraints import check_constraints

        assert not check_constraints({"ecutwfc": 60, "ecutrho": 480})

    def test_zigzag_without_preset_is_blocked(self):
        from carbonforge.gui.constraints import check_constraints

        violations = check_constraints(
            {"preset": "ninguna"}, build_nanoribbon(6, 3, edge="zigzag")
        )
        assert any(v.blocking for v in violations)

    def test_zigzag_with_preset_is_fine(self):
        from carbonforge.gui.constraints import check_constraints

        violations = check_constraints(
            {"preset": "bands"}, build_nanoribbon(6, 3, edge="zigzag")
        )
        assert not any("antiferromagn" in v.message for v in violations)

    def test_raman_on_metal_is_blocked(self):
        from carbonforge.gui.constraints import check_constraints

        violations = check_constraints(
            {"task": "raman"}, build_graphene_supercell(3, 3)
        )
        assert any(v.blocking for v in violations)

    def test_too_many_groups_blocked(self):
        from carbonforge.gui.constraints import check_constraints

        violations = check_constraints(
            {"group": "NH2", "group_count": 99, "group_site": "edge"},
            build_nanoribbon(4, 3, edge="armchair"),
        )
        assert any("solo hay" in v.message for v in violations)

    def test_no_edge_sites_on_periodic_sheet(self):
        from carbonforge.gui.constraints import check_constraints

        violations = check_constraints(
            {"group": "OH", "group_count": 1, "group_site": "edge"},
            build_graphene_supercell(3, 3),
        )
        assert any("no tiene bordes" in v.message for v in violations)

    def test_thin_vacuum_warns(self):
        from carbonforge.gui.constraints import check_constraints

        violations = check_constraints({"vacuum": 5.0})
        assert any("poco para DFT" in v.message for v in violations)

    def test_a_broken_rule_never_blocks_the_user(self):
        """Rules run defensively: garbage in must not raise."""
        from carbonforge.gui.constraints import check_constraints

        check_constraints({"ecutwfc": "no soy un número", "group_count": "x"})

    def test_formatting_separates_errors_from_warnings(self):
        from carbonforge.gui.constraints import check_constraints, format_violations

        text = format_violations(
            check_constraints({"ecutwfc": 60, "ecutrho": 100, "vacuum": 5.0})
        )
        assert "impiden calcular" in text and "Avisos" in text
