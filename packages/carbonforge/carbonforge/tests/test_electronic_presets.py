"""Tests for electronic-structure settings, presets and the run pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from carbonforge.builders import (
    build_carbon_foam,
    build_graphene_supercell,
    build_nanoribbon,
)
from carbonforge.calculations.electronic import (
    ElectronicSpec,
    exx_grid,
    qe_system_fields,
    setup_antiferromagnetic_edges,
    tagged_species,
)
from carbonforge.exports.qe import QESettings, write_qe_input
from carbonforge.validation.calculations import check_electronic_setup
from carbonforge.workflows.pipeline import (
    count_kpoints,
    plan_run,
    suggest_pools,
    write_preset_project,
)
from carbonforge.workflows.presets import PRESETS, apply_preset, describe_presets


class TestElectronicSpec:
    def test_defaults_are_plain_pbe(self):
        spec = ElectronicSpec()
        assert not spec.is_spin_polarized
        assert not spec.is_hybrid
        assert spec.cost_multiplier() == pytest.approx(1.0)

    def test_rejects_unknown_options(self):
        with pytest.raises(ValueError, match="vdW"):
            ElectronicSpec(vdw_correction="magic")
        with pytest.raises(ValueError, match="Funcional"):
            ElectronicSpec(functional="madeup")
        with pytest.raises(ValueError):
            ElectronicSpec(exx_grid_factor=0)
        with pytest.raises(ValueError):
            ElectronicSpec(exx_fraction=2.0)

    def test_magnetization_without_spin_is_rejected(self):
        """A silent no-op would be worse than an error here."""
        with pytest.raises(ValueError, match="spin='none'"):
            ElectronicSpec(starting_magnetization={"C": 0.5})

    def test_cost_multiplier_reflects_the_physics(self):
        assert ElectronicSpec(spin="collinear").cost_multiplier() == 2.0
        hybrid = ElectronicSpec(functional="hse")
        assert hybrid.cost_multiplier() >= 20

    def test_describe_warns_about_expensive_setups(self):
        assert "⚠️" in ElectronicSpec(functional="hse").describe()

    def test_qe_fields_index_species_by_position(self):
        """QE indexes starting_magnetization by ATOMIC_SPECIES order."""
        spec = ElectronicSpec(
            spin="collinear", starting_magnetization={"C1": 0.5, "C2": -0.5}
        )
        fields = qe_system_fields(spec, ["C", "C1", "C2"])
        assert fields["nspin"] == 2
        assert fields["starting_magnetization(2)"] == 0.5
        assert fields["starting_magnetization(3)"] == -0.5

    def test_vdw_and_functional_fields(self):
        spec = ElectronicSpec(vdw_correction="grimme-d3", functional="hse")
        fields = qe_system_fields(spec, ["C"])
        assert fields["vdw_corr"] == "grimme-d3"
        assert fields["input_dft"] == "hse"
        assert "screening_parameter" in fields

    def test_exx_grid_is_coarser_than_the_kmesh(self):
        assert exx_grid((8, 8, 1), 2) == (4, 4, 1)
        assert exx_grid((1, 1, 1), 2) == (1, 1, 1)


class TestAntiferromagneticEdges:
    def test_tags_two_sublattices(self):
        tagged, spec = setup_antiferromagnetic_edges(
            build_nanoribbon(6, 3, edge="zigzag")
        )
        info = tagged.info["afm_edges"]
        assert info["n_up"] > 0 and info["n_down"] > 0
        assert spec.spin == "collinear"

    def test_moments_have_opposite_signs(self):
        """Same sign would give a ferromagnetic guess, not the ground state."""
        _, spec = setup_antiferromagnetic_edges(
            build_nanoribbon(6, 3, edge="zigzag")
        )
        assert spec.starting_magnetization["C1"] > 0
        assert spec.starting_magnetization["C2"] < 0

    def test_uses_ase_tags_not_invented_symbols(self):
        """ASE rejects symbols like 'C_up', so the split rides on tags."""
        tagged, _ = setup_antiferromagnetic_edges(
            build_nanoribbon(6, 3, edge="zigzag")
        )
        assert set(tagged.get_chemical_symbols()) == {"C"}
        assert set(tagged.get_tags()) == {0, 1, 2}

    def test_tagged_species_expands_to_qe_labels(self):
        tagged, _ = setup_antiferromagnetic_edges(
            build_nanoribbon(6, 3, edge="zigzag")
        )
        labels, pseudo_map = tagged_species(tagged)
        assert set(labels) == {"C", "C1", "C2"}
        # All three are carbon and must share one pseudopotential.
        assert set(pseudo_map.values()) == {"C"}

    def test_periodic_sheet_has_no_edges_to_split(self):
        with pytest.raises(ValueError):
            setup_antiferromagnetic_edges(build_graphene_supercell(3, 3))

    def test_written_input_carries_three_species(self, tmp_path):
        tagged, spec = setup_antiferromagnetic_edges(
            build_nanoribbon(6, 3, edge="zigzag")
        )
        path = write_qe_input(
            tagged, tmp_path, settings=QESettings(electronic=spec)
        )
        text = path.read_text()
        assert "ntyp = 3" in text
        assert "nspin = 2" in text
        assert text.count("C.pbe-n-kjpaw_psl.1.0.0.UPF") == 3


class TestElectronicValidation:
    def test_zigzag_ribbon_without_spin_is_an_error(self):
        """The headline case: a silently wrong ground state."""
        report = check_electronic_setup(build_nanoribbon(6, 3, edge="zigzag"))
        assert not report.ok
        assert any("antiferromagn" in e for e in report.errors)

    def test_zigzag_ribbon_with_afm_passes(self):
        tagged, spec = setup_antiferromagnetic_edges(
            build_nanoribbon(6, 3, edge="zigzag")
        )
        assert check_electronic_setup(tagged, spec).ok

    def test_armchair_ribbon_needs_no_spin(self):
        report = check_electronic_setup(
            build_nanoribbon(7, 3, edge="armchair"), ElectronicSpec()
        )
        assert report.ok

    def test_collinear_without_moments_warns(self):
        report = check_electronic_setup(
            build_graphene_supercell(3, 3), ElectronicSpec(spin="collinear")
        )
        assert any("magnetizaciones iniciales" in w for w in report.warnings)

    def test_foam_without_dispersion_warns(self):
        foam = build_carbon_foam(box_size=20, n_flakes=4, flake_radius=3, seed=0)
        report = check_electronic_setup(foam, ElectronicSpec())
        assert any("van der Waals" in w for w in report.warnings)

    def test_hybrid_warns_about_cost(self):
        report = check_electronic_setup(
            build_graphene_supercell(4, 4), ElectronicSpec(functional="hse")
        )
        assert any("híbrido" in w for w in report.warnings)

    def test_hubbard_on_light_elements_warns(self):
        report = check_electronic_setup(
            build_graphene_supercell(3, 3),
            ElectronicSpec(hubbard_u={"C": 4.0}),
        )
        assert any("metales de transición" in w for w in report.warnings)


class TestPools:
    @pytest.mark.parametrize(
        "cores,kpoints,expected",
        [
            (32, 16, 16),   # divides both
            (32, 7, 4),     # largest divisor of 32 not exceeding 7
            (8, 64, 4),     # capped by cores/min_cores_per_pool
            (4, 1, 1),      # Gamma only
        ],
    )
    def test_pool_choice(self, cores, kpoints, expected):
        assert suggest_pools(cores, kpoints) == expected

    def test_pools_always_divide_the_core_count(self):
        for cores in range(2, 65):
            for kpoints in (1, 3, 8, 27):
                assert cores % suggest_pools(cores, kpoints) == 0

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError):
            suggest_pools(0, 4)

    def test_plan_explains_itself(self):
        plan = plan_run(build_graphene_supercell(4, 4), QESettings(), n_cores=16)
        assert "pools" in plan.explain()
        assert "-nk" in plan.mpi_command

    def test_kpoint_count_matches_the_mesh(self):
        sheet = build_graphene_supercell(4, 4)
        assert count_kpoints(sheet, 0.2) >= 1


class TestPresets:
    def test_catalogue_lists_everything(self):
        text = describe_presets()
        for key in PRESETS:
            assert key in text

    def test_unknown_preset_lists_options(self):
        with pytest.raises(ValueError, match="Opciones"):
            apply_preset(build_graphene_supercell(3, 3), "nope")

    def test_zigzag_ribbon_gets_spin_automatically(self):
        """The point of presets: the user does not have to know to ask."""
        result = apply_preset(build_nanoribbon(6, 3, edge="zigzag"), "bands")
        assert result.electronic.spin == "collinear"
        assert any("antiferromagn" in d for d in result.decisions)

    def test_armchair_ribbon_stays_unpolarised(self):
        result = apply_preset(build_nanoribbon(7, 3, edge="armchair"), "bands")
        assert result.electronic.spin == "none"

    def test_adsorption_enables_dispersion(self):
        result = apply_preset(build_graphene_supercell(3, 3), "adsorption")
        assert result.electronic.vdw_correction == "grimme-d3"

    def test_foam_gets_dispersion_from_its_structure_type(self):
        foam = build_carbon_foam(box_size=20, n_flakes=4, flake_radius=3, seed=0)
        result = apply_preset(foam, "geometry")
        assert result.electronic.vdw_correction == "grimme-d3"

    def test_hse_preset_sets_the_functional_and_warns(self):
        result = apply_preset(build_graphene_supercell(3, 3), "bands-hse")
        assert result.electronic.functional == "hse"
        assert result.warnings

    def test_quick_is_cheaper_than_standard(self):
        quick = apply_preset(build_graphene_supercell(3, 3), "quick")
        standard = apply_preset(build_graphene_supercell(3, 3), "bands")
        assert quick.settings.ecutwfc < standard.settings.ecutwfc
        assert quick.settings.kpoint_density > standard.settings.kpoint_density

    def test_accurate_tightens_everything(self):
        normal = apply_preset(build_graphene_supercell(3, 3), "bands")
        fine = apply_preset(build_graphene_supercell(3, 3), "bands", accurate=True)
        assert fine.settings.ecutwfc > normal.settings.ecutwfc
        assert fine.settings.kpoint_density < normal.settings.kpoint_density

    def test_phonons_tighten_scf_convergence(self):
        """Force constants are second derivatives and amplify density noise."""
        scf = apply_preset(build_graphene_supercell(3, 3), "quick")
        phonon = apply_preset(build_graphene_supercell(3, 3), "phonon")
        assert phonon.settings.conv_thr < scf.settings.conv_thr

    def test_raman_switches_to_norm_conserving(self):
        result = apply_preset(build_graphene_supercell(3, 3), "raman")
        pseudos = result.settings.pseudopotentials or {}
        assert any("ONCV" in name for name in pseudos.values())

    def test_low_dimensional_relaxation_constrains_the_cell(self):
        result = apply_preset(build_graphene_supercell(3, 3), "geometry")
        assert result.settings.cell_dofree == "2Dxy"

    def test_explanation_mentions_convergence(self):
        result = apply_preset(build_graphene_supercell(3, 3), "bands")
        assert "converge" in result.explain()


class TestPresetProject:
    def test_bands_project_chains_relaxation_first(self, tmp_path):
        result, plan, written = write_preset_project(
            build_graphene_supercell(3, 3), tmp_path, "bands", n_cores=8
        )
        assert result.relax_first
        assert "relax" in written
        script = written["script"].read_text()
        # The relaxation must come before the property, with the geometry
        # carried across between them.
        assert script.index("pw.relax.in") < script.index("update-geometry")
        assert script.index("update-geometry") < script.index("run_bands.sh")

    def test_script_carries_the_pool_count(self, tmp_path):
        _, plan, written = write_preset_project(
            build_graphene_supercell(4, 4), tmp_path, "bands", n_cores=16
        )
        assert f"POOLS:-{plan.pools}" in written["script"].read_text()

    def test_decisions_are_written_to_disk(self, tmp_path):
        """The reasoning has to outlive the terminal session."""
        _, _, written = write_preset_project(
            build_nanoribbon(6, 3, edge="zigzag"), tmp_path, "bands"
        )
        notes = written["notes"].read_text()
        assert "antiferromagn" in notes
        assert "Paralelización" in notes

    def test_dos_project_writes_the_full_chain(self, tmp_path):
        _, _, written = write_preset_project(
            build_graphene_supercell(3, 3), tmp_path, "dos"
        )
        assert {"scf", "nscf", "dos", "projwfc"} <= set(written)

    def test_quick_preset_needs_no_relaxation(self, tmp_path):
        result, _, written = write_preset_project(
            build_graphene_supercell(3, 3), tmp_path, "quick"
        )
        assert not result.relax_first
        assert "update-geometry" not in written["script"].read_text()

    def test_script_is_executable(self, tmp_path):
        _, _, written = write_preset_project(
            build_graphene_supercell(3, 3), tmp_path, "quick"
        )
        assert written["script"].stat().st_mode & 0o111


class TestGeometryUpdate:
    def test_replaces_positions_in_downstream_inputs(self, tmp_path):
        """Mimics what the chained script does between relax and property."""
        from carbonforge.workflows.pipeline import _replace_card

        original = write_qe_input(
            build_graphene_supercell(2, 2), tmp_path, filename="pw.bands.in"
        )
        text = original.read_text()
        updated = _replace_card(
            text, "ATOMIC_POSITIONS", "  C  1.0 2.0 3.0"
        )
        assert "C  1.0 2.0 3.0" in updated
        # Everything else must survive untouched.
        assert "&CONTROL" in updated and "K_POINTS" in updated

    def test_missing_card_is_left_alone(self, tmp_path):
        from carbonforge.workflows.pipeline import _replace_card

        assert _replace_card("sin tarjetas", "ATOMIC_POSITIONS", "x") == (
            "sin tarjetas"
        )

    def test_missing_output_file_is_reported(self, tmp_path):
        from carbonforge.workflows.pipeline import read_relaxed_geometry

        with pytest.raises(ValueError, match="no existe"):
            read_relaxed_geometry(tmp_path / "nope.out")
