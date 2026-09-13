"""Tests for classical force fields, electrolytes and EDLC cells."""

from __future__ import annotations

import numpy as np
import pytest

from carbonforge.builders import build_graphene_supercell, build_nanoribbon
from carbonforge.exports.lammps_edlc import EDLCSettings, write_edlc
from carbonforge.forcefields import (
    assign_types,
    build_aqueous,
    build_edlc_cell,
    build_ionic_liquid,
    check_edlc_setup,
    describe_provenance,
    get_params,
)
from carbonforge.forcefields.charges import (
    apply_derived_charges,
    charge_spread,
    charges_by_type,
    read_lowdin_charges,
)
from carbonforge.functionalization import (
    functionalize_random,
    make_graphitic_n,
    make_pyridinic_n,
)


def _electrode(n=5):
    return build_graphene_supercell(n, n)


class TestAtomTyping:
    def test_pristine_graphene_is_all_sp2(self):
        result = assign_types(_electrode())
        assert result.unique_types == ["C_sp2"]
        assert not result.unresolved

    def test_graphitic_nitrogen_and_its_neighbours(self):
        """The neighbours must type differently: that is the doping effect."""
        doped = make_graphitic_n(_electrode(6), n_sites=2, seed=0)
        counts = assign_types(doped).counts()
        assert counts["N_graph"] == 2
        assert counts.get("C_N", 0) > 0

    def test_pyridinic_nitrogen_is_distinguished_from_graphitic(self):
        """They differ only in coordination, and carry different charges."""
        pyridinic = make_pyridinic_n(_electrode(6), n_defects=1, seed=0)
        assert "N_pyri" in assign_types(pyridinic).unique_types

    def test_hydroxyl_group_types(self):
        ribbon = functionalize_random(
            build_nanoribbon(6, 3, edge="armchair"), "OH", n_groups=2, seed=0
        )
        types = assign_types(ribbon).unique_types
        assert "O_hydroxyl" in types
        assert "H_O" in types

    def test_amine_group_types(self):
        ribbon = functionalize_random(
            build_nanoribbon(6, 3, edge="armchair"), "NH2", n_groups=2, seed=0
        )
        types = assign_types(ribbon).unique_types
        assert "N_amine" in types and "H_N" in types

    def test_every_type_has_parameters(self):
        """A typed atom with no parameters would silently get a wrong charge."""
        from carbonforge.forcefields.typing import ATOM_TYPES

        for name in ATOM_TYPES:
            params = get_params(name)
            assert params.sigma >= 0.0


class TestParameters:
    def test_water_charges_balance(self):
        """SPC/E is neutral by construction."""
        total = get_params("OW").charge + 2 * get_params("HW").charge
        assert total == pytest.approx(0.0, abs=1e-9)

    def test_ions_carry_formal_charges(self):
        assert get_params("Na").charge == pytest.approx(1.0)
        assert get_params("Cl").charge == pytest.approx(-1.0)

    def test_pristine_carbon_is_neutral(self):
        assert get_params("C_sp2").charge == pytest.approx(0.0)

    def test_nitrogen_is_negative_and_its_carbon_positive(self):
        """The whole point of doping: charge moves from C to N."""
        assert get_params("N_graph").charge < 0
        assert get_params("C_N").charge > 0

    def test_pyridinic_is_more_negative_than_graphitic(self):
        """It keeps a lone pair, so it holds more density."""
        assert get_params("N_pyri").charge < get_params("N_graph").charge

    def test_provenance_flags_uncertain_charges(self):
        """Doped-carbon charges must not be presented as settled."""
        text = describe_provenance(["C_sp2", "N_graph"])
        assert "REPRESENTATIVAS" in text
        assert "Löwdin" in text or "Bader" in text

    def test_provenance_cites_established_models(self):
        text = describe_provenance(["OW", "Na"])
        assert "SPC/E" in text
        assert "Joung" in text

    def test_unknown_type_rejected(self):
        with pytest.raises(ValueError, match="Tipo sin parámetros"):
            get_params("Xx_made_up")


class TestElectrolyte:
    def test_aqueous_is_charge_neutral(self):
        """Ewald needs neutrality; equal ion counts guarantee it."""
        box = build_aqueous((20.0, 20.0, 60.0), (10.0, 50.0),
                            molarity=1.0, seed=0)
        total = sum(get_params(t).charge for t in box.types)
        assert total == pytest.approx(0.0, abs=1e-6)

    def test_water_molecules_have_three_atoms_each(self):
        box = build_aqueous((20.0, 20.0, 60.0), (10.0, 50.0),
                            molarity=0.0, seed=0)
        assert box.types.count("OW") * 2 == box.types.count("HW")

    def test_molecule_ids_group_each_water(self):
        box = build_aqueous((20.0, 20.0, 50.0), (10.0, 40.0),
                            molarity=0.0, seed=0)
        first_water = [
            i for i, m in enumerate(box.molecule_ids) if m == 1
        ]
        assert len(first_water) == 3

    def test_no_intermolecular_overlap(self):
        """The floor should be the internal O-H bond, nothing shorter."""
        box = build_aqueous((25.0, 25.0, 60.0), (10.0, 50.0),
                            molarity=1.0, seed=0)
        distances = box.atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() >= 0.99

    def test_reproducible_with_seed(self):
        a = build_aqueous((20.0, 20.0, 50.0), (10.0, 40.0), seed=7)
        b = build_aqueous((20.0, 20.0, 50.0), (10.0, 40.0), seed=7)
        np.testing.assert_allclose(
            a.atoms.get_positions(), b.atoms.get_positions()
        )

    def test_higher_molarity_gives_more_ions(self):
        low = build_aqueous((25.0, 25.0, 60.0), (10.0, 50.0),
                            molarity=0.5, seed=0)
        high = build_aqueous((25.0, 25.0, 60.0), (10.0, 50.0),
                             molarity=2.0, seed=0)
        assert high.composition["Na"] > low.composition["Na"]

    def test_ionic_liquid_is_neutral_and_paired(self):
        box = build_ionic_liquid((30.0, 30.0, 60.0), (10.0, 50.0), seed=0)
        assert box.composition["BMIM"] == box.composition["PF6"]

    def test_unknown_salt_rejected(self):
        with pytest.raises(ValueError, match="Sal desconocida"):
            build_aqueous((20.0, 20.0, 50.0), (10.0, 40.0), salt="NaBr")

    def test_impossible_packing_is_reported(self):
        """The density is self-consistent, so the guard fires on spacing.

        Asking for liquid density AND a separation larger than the molecules
        themselves have cannot be satisfied, and must fail loudly rather than
        silently placing fewer molecules than requested.
        """
        with pytest.raises(ValueError, match="sin solaparse"):
            build_aqueous(
                (20.0, 20.0, 50.0), (10.0, 40.0),
                molarity=0.0, seed=0, min_distance=6.0,
            )


class TestEDLCCell:
    def test_assembles_three_regions(self):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        assert cell.groups["bottom"] and cell.groups["top"]
        assert cell.groups["electrolyte"]
        assert len(cell.atoms) == sum(len(g) for g in cell.groups.values())

    def test_electrolyte_stays_neutral(self):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"molarity": 1.0, "seed": 0})
        assert cell.net_charge() == pytest.approx(0.0, abs=1e-6)

    def test_electrodes_do_not_touch_the_electrolyte(self):
        """The filler knows nothing about the electrodes; the gap is explicit."""
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        distances = cell.atoms.get_all_distances(mic=True)
        np.fill_diagonal(distances, np.inf)
        # Nothing closer than the internal O-H bond of water.
        assert distances.min() >= 0.99

    def test_potential_is_split_symmetrically(self):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               potential_v=2.0,
                               electrolyte_kwargs={"seed": 0})
        assert cell.potential_v == pytest.approx(2.0)

    def test_slab_geometry_is_periodic_only_in_plane(self):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        assert list(cell.atoms.get_pbc()) == [True, True, False]

    def test_finite_electrode_rejected(self):
        """A ribbon cannot tile the cross-section."""
        with pytest.raises(ValueError, match="periódico en x e y"):
            build_edlc_cell(build_nanoribbon(6, 3), separation=35.0)

    def test_separation_smaller_than_the_wall_gaps_rejected(self):
        with pytest.raises(ValueError):
            build_edlc_cell(_electrode(5), separation=4.0)

    def test_doped_electrode_carries_its_charges(self):
        doped = make_graphitic_n(_electrode(6), n_sites=3, seed=0)
        cell = build_edlc_cell(doped, separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        assert "N_graph" in cell.unique_types
        assert "C_N" in cell.unique_types

    def test_vacuum_electrolyte_leaves_the_gap_empty(self):
        cell = build_edlc_cell(_electrode(4), separation=30.0,
                               electrolyte="vacuum")
        assert cell.groups["electrolyte"] == []


class TestEDLCChecks:
    def test_narrow_separation_warns_about_overlapping_layers(self):
        cell = build_edlc_cell(_electrode(5), separation=20.0,
                               electrolyte_kwargs={"seed": 0})
        assert any("solapar" in w for w in check_edlc_setup(cell))

    def test_excessive_potential_warns(self):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               potential_v=8.0,
                               electrolyte_kwargs={"seed": 0})
        assert any("electroliza" in w for w in check_edlc_setup(cell))

    def test_sampling_warning_counts_particles_not_atoms(self):
        """A coarse-grained ion is one bead; water is three atoms.

        Counting atoms would judge the ionic liquid on the wrong scale and
        report a number that means something different for each electrolyte.
        """
        il = build_edlc_cell(_electrode(6), separation=50.0,
                             electrolyte="ionic_liquid",
                             electrolyte_kwargs={"seed": 0})
        n_atoms = len(il.groups["electrolyte"])
        n_particles = sum(il.electrolyte_composition.values())
        assert n_atoms == n_particles  # one bead per ion
        message = next(w for w in check_edlc_setup(il) if "iones" in w)
        assert f"{n_particles} moléculas/iones" in message

        water = build_edlc_cell(_electrode(6), separation=40.0,
                                electrolyte_kwargs={"seed": 0})
        # Three atoms per molecule: the atom count is far above the particle
        # count, so the two criteria are genuinely different.
        assert len(water.groups["electrolyte"]) > 2 * sum(
            water.electrolyte_composition.values()
        )

    def test_reasonable_setup_is_clean(self):
        cell = build_edlc_cell(_electrode(6), separation=40.0,
                               potential_v=1.0,
                               electrolyte_kwargs={"molarity": 1.0, "seed": 0})
        assert check_edlc_setup(cell) == []


class TestEDLCExport:
    def test_writes_data_input_and_notes(self, tmp_path):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        written = write_edlc(cell, tmp_path)
        assert set(written) == {"data", "input", "notes"}
        for path in written.values():
            assert path.exists() and path.stat().st_size > 0

    def test_data_uses_atom_style_full_with_charges(self, tmp_path):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        text = write_edlc(cell, tmp_path)["data"].read_text()
        assert "Atoms  # full" in text
        # id mol type q x y z is seven columns.
        body = text.split("Atoms  # full")[1].strip().splitlines()
        assert len(body[1].split()) == 7

    def test_input_sets_the_slab_correction(self, tmp_path):
        """Non-optional for a 2D-periodic cell with a dipole."""
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        text = write_edlc(cell, tmp_path)["input"].read_text()
        assert "kspace_modify   slab" in text

    def test_input_uses_constant_potential(self, tmp_path):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               potential_v=1.0,
                               electrolyte_kwargs={"seed": 0})
        text = write_edlc(cell, tmp_path)["input"].read_text()
        assert "electrode/conp" in text
        # Symmetric split keeps the cell neutral.
        assert "-0.5000" in text and "0.5000" in text

    def test_input_minimises_before_dynamics(self, tmp_path):
        """The built configuration is close but not relaxed."""
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        text = write_edlc(cell, tmp_path)["input"].read_text()
        # "electrode/conp" also appears in the header comment about the
        # required package, so match the actual fix line.
        fix_line = text.index("fix             cpm")
        assert text.index("minimize") < fix_line

    def test_production_follows_equilibration(self, tmp_path):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        text = write_edlc(cell, tmp_path)["input"].read_text()
        assert text.index("equilibration") < text.index("production")

    def test_header_names_the_required_package(self, tmp_path):
        cell = build_edlc_cell(_electrode(5), separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        text = write_edlc(cell, tmp_path)["input"].read_text()
        assert "ELECTRODE package" in text

    def test_notes_carry_the_provenance_warning(self, tmp_path):
        doped = make_graphitic_n(_electrode(6), n_sites=2, seed=0)
        cell = build_edlc_cell(doped, separation=35.0,
                               electrolyte_kwargs={"seed": 0})
        text = write_edlc(cell, tmp_path)["notes"].read_text()
        assert "REPRESENTATIVAS" in text

    def test_rejects_a_timestep_water_cannot_survive(self):
        with pytest.raises(ValueError, match="SHAKE"):
            EDLCSettings(rigid_water=False, timestep_fs=1.0)

    def test_rejects_an_absurd_timestep_even_with_shake(self):
        with pytest.raises(ValueError, match="demasiado"):
            EDLCSettings(rigid_water=True, timestep_fs=5.0)


class TestChargesFromDFT:
    def _projwfc_output(self, charges: dict[int, float]) -> str:
        lines = ["     Lowdin Charges:", ""]
        for index, value in sorted(charges.items()):
            lines.append(
                f"     Atom #{index:4d}: total charge = {value:8.4f}, "
                "s, p, d"
            )
        return "\n".join(lines) + "\n"

    def test_reads_lowdin_populations(self, tmp_path):
        path = tmp_path / "projwfc.out"
        path.write_text(self._projwfc_output({1: 4.10, 2: 3.90}))
        derived = read_lowdin_charges(path)
        assert len(derived.charges) == 2

    def test_converts_populations_to_charges(self, tmp_path):
        """Charge = valence electrons - Lowdin population."""
        from ase import Atoms as ASEAtoms

        atoms = ASEAtoms("CN", positions=[[0, 0, 0], [1.4, 0, 0]])
        path = tmp_path / "projwfc.out"
        path.write_text(self._projwfc_output({1: 3.80, 2: 5.40}))
        derived = read_lowdin_charges(path, atoms)
        # C: 4 - 3.8 = +0.2 ; N: 5 - 5.4 = -0.4
        assert derived.charges[0] == pytest.approx(0.2)
        assert derived.charges[1] == pytest.approx(-0.4)

    def test_missing_lowdin_section_reported(self, tmp_path):
        path = tmp_path / "projwfc.out"
        path.write_text("nada útil aquí\n")
        with pytest.raises(ValueError, match="Löwdin"):
            read_lowdin_charges(path)

    def test_atom_count_mismatch_reported(self, tmp_path):
        path = tmp_path / "projwfc.out"
        path.write_text(self._projwfc_output({1: 4.0}))
        with pytest.raises(ValueError, match="mismo cálculo"):
            read_lowdin_charges(path, _electrode(3))

    def test_charges_average_by_type(self):
        doped = make_graphitic_n(_electrode(6), n_sites=2, seed=0)
        charges = np.full(len(doped), 0.1)
        means = charges_by_type(doped, charges)
        assert all(v == pytest.approx(0.1) for v in means.values())

    def test_spread_flags_inconsistent_types(self):
        types = ["C_sp2", "C_sp2", "C_sp2"]
        charges = np.array([0.0, 0.5, -0.5])
        assert charge_spread(types, charges)["C_sp2"] > 0.1

    def test_report_warns_on_wide_spread(self):
        """A type whose atoms disagree cannot be one number."""
        from carbonforge.forcefields.charges import DerivedCharges

        sheet = _electrode(4)
        charges = np.linspace(-0.5, 0.5, len(sheet))
        _, report = apply_derived_charges(
            sheet, DerivedCharges(charges=charges)
        )
        assert "dispersión" in report

    def test_report_confirms_consistent_types(self):
        from carbonforge.forcefields.charges import DerivedCharges

        sheet = _electrode(4)
        charges = np.full(len(sheet), 0.02)
        _, report = apply_derived_charges(
            sheet, DerivedCharges(charges=charges)
        )
        assert "✅" in report
