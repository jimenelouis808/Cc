"""Tests for the shared job description behind the GUI and the CLI.

Three things have to hold for this module to be worth having. A job must
survive a trip through pickle, because the GUI hands it to a subprocess.
The atom-count estimate must be close enough to be worth showing before a
build. And the generated command line must actually be accepted by the
CLI -- a command that looks right and does not run is worse than none.
"""

from __future__ import annotations

import pickle
import shlex

import pytest

from nanocarbon_lab.jobs import (
    IMPLICIT_MODES,
    MODES,
    Job,
    estimate_atoms,
    estimate_cost,
    to_cli,
)

# One representative job per mode, kept small enough to build in a test.
SAMPLES: dict[str, Job] = {
    "capped tube": Job("capped tube", {"n_body_rings": 8, "freq": 3}),
    "fullerene": Job("fullerene", {"freq": 1, "family": "C60"}),
    "nano-onion": Job("nano-onion",
                      {"n_shells": 3, "inner_freq": 1, "family": "C60"}),
    "multi-wall": Job("multi-wall", {"n_shells": 2, "inner_freq": 2,
                                     "freq_step": 2, "n_body_rings": 4}),
    "bundle": Job("bundle", {"n_rings_across": 1, "freq": 2, "n_body_rings": 4}),
    "junction": Job("junction", {"kind": "Y", "tube_radius": 6.0,
                                 "arm_length": 22.0, "blend": 4.0}),
    "schwarzite": Job("schwarzite", {"kind": "gyroid", "cell": 36.0}),
    "coil (relaxed)": Job("coil (relaxed)",
                          {"coil_radius": 34.0, "pitch": 20.0, "turns": 1.25,
                           "tube_radius": 4.5}),
    "TMD layers": Job("TMD layers",
                      {"material": "MoS2", "n_layers": 2, "nx": 2, "ny": 2}),
    "TMD bulk": Job("TMD bulk", {"material": "MoS2", "stacking": "2H"}),
    "TMD ribbon": Job("TMD ribbon",
                      {"material": "MoS2", "width": 6, "length": 2}),
    "TMD nanotube": Job("TMD nanotube", {"material": "MoS2", "n": 40, "m": 0}),
    # The smallest cell the builder accepts, to keep this buildable in a
    # test: it still meshes and relaxes, so it is the slow one here.
    "TMD schwarzite": Job("TMD schwarzite",
                          {"material": "MoS2", "kind": "primitive",
                           "cell": 30.0, "parity": "none"}),
    "twisted bilayer": Job("twisted bilayer",
                           {"layer": "graphene", "target_angle": 21.8}),
    "vdW stack": Job("vdW stack",
                     {"layers": ["graphene", "hBN"], "nx": 2, "ny": 2}),
    "TMD coil": Job("TMD coil",
                    {"material": "MoS2", "n": 20, "m": 0,
                     "coil_radius": 140.0, "pitch": 60.0, "turns": 0.15}),
    # An L is the cheapest junction: two arms rather than three or four,
    # and a radius just above the 2*h floor below which the sandwich's
    # inner wall would meet on the axis.
    "TMD junction": Job("TMD junction",
                        {"material": "MoS2", "kind": "L", "tube_radius": 10.0,
                         "arm_length": 20.0, "parity": "split"}),
}


def test_every_mode_has_a_sample():
    assert set(SAMPLES) == set(MODES)


class TestJob:
    def test_unknown_mode_is_rejected_at_construction(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            Job("nanotube-ish")

    def test_a_job_survives_pickling(self):
        """The GUI sends jobs to a subprocess, so this is not academic."""
        for job in SAMPLES.values():
            assert pickle.loads(pickle.dumps(job)) == job

    def test_with_params_does_not_mutate_the_original(self):
        job = SAMPLES["fullerene"]
        bigger = job.with_params(freq=3)
        assert bigger.params["freq"] == 3
        assert job.params["freq"] == 1


class TestEstimates:
    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            # Exact for the seed-polyhedron modes: the count is a closed
            # form, and these are the numbers the builders really return.
            ("capped tube", 720),
            ("fullerene", 60),
            ("nano-onion", 840),
            ("multi-wall", 800),
            ("bundle", 1120),
            # And for every dichalcogenide, which is placed on ideal
            # lattice sites: three atoms per formula unit, times the cells.
            ("TMD layers", 24),
            ("TMD bulk", 6),
            ("TMD ribbon", 36),
            ("TMD nanotube", 240),
            # A coil is periods x one tube cell, and the period count is
            # set by the helix arc length, not by anything the caller
            # types as a size -- so it is worth pinning exactly.
            ("TMD coil", 2880),
            # A stack is exact too: cells per layer times the sites in
            # each, and the cell count is fixed by the commensurate pair.
            ("twisted bilayer", 28),
            ("vdW stack", 16),
        ],
    )
    def test_seed_modes_are_predicted_exactly(self, mode, expected):
        assert estimate_atoms(SAMPLES[mode]) == expected

    @pytest.mark.parametrize("mode", ["TMD layers", "TMD bulk", "TMD ribbon",
                                      "TMD nanotube", "TMD coil"])
    def test_dichalcogenide_estimates_match_the_real_build(self, mode):
        """These are exact, not approximate, so assert equality against a
        real build rather than a tolerance."""
        from nanocarbon_lab.jobs import build

        job = SAMPLES[mode]
        assert estimate_atoms(job) == len(build(job))

    @pytest.mark.parametrize(
        ("mode", "measured"),
        [("schwarzite", 1502), ("coil (relaxed)", 2782), ("junction", 1100),
         # The MX2 junction is estimated the same way but has to correct
         # for arms burying each other: summing them counts the junction
         # body once per arm, and the shortfall grows with the arm count.
         # Measured 1530 for this L; a Y at r=12/arm 26 is 3153 against
         # 3074 predicted.
         ("TMD junction", 1530)],
    )
    def test_implicit_modes_are_predicted_within_a_tenth(self, mode, measured):
        """Surface area over ring area. Not exact, but the point is to
        tell a 60-atom cage from a 3000-atom coil before committing."""
        assert estimate_atoms(SAMPLES[mode]) == pytest.approx(measured, rel=0.2)

    @pytest.mark.parametrize("mode", ["twisted bilayer", "vdW stack"])
    def test_stack_estimates_match_the_real_build(self, mode):
        from nanocarbon_lab.jobs import build

        job = SAMPLES[mode]
        assert estimate_atoms(job) == len(build(job))

    def test_a_small_twist_angle_is_flagged_as_expensive(self):
        """The surprise worth warning about: the atom count is set by the
        angle, not by anything that reads as a size. 21.8 deg is 28 atoms
        and 1.1 deg is 11 164."""
        wide = SAMPLES["twisted bilayer"]
        tight = wide.with_params(target_angle=1.1, max_index=40)
        assert estimate_cost(wide)[0] == "fast"
        assert estimate_cost(tight)[0] in ("slow", "very slow")

    def test_the_schwarzite_is_estimated_from_the_surface_area(self):
        """Not enumerated like the others: it is meshed, so the count is
        area over the area one triangle of side `a` covers. Measured 1066
        atoms at 36 A against 1054 predicted."""
        job = SAMPLES["TMD schwarzite"].with_params(cell=36.0)
        assert estimate_atoms(job) == pytest.approx(1066, rel=0.05)

    def test_dichalcogenides_are_never_called_slow(self):
        """No meshing and no relaxation -- exact lattice placement, so the
        only cost is drawing them."""
        for mode in ("TMD layers", "TMD bulk", "TMD ribbon", "TMD nanotube"):
            assert estimate_cost(SAMPLES[mode])[0] == "fast"

    def test_a_coil_is_costed_by_its_atom_count_not_by_its_mode(self):
        """Still no meshing, but a coil's length is the helix arc, so it
        is the one TMD mode that can reach six figures by accident."""
        assert estimate_cost(SAMPLES["TMD coil"])[0] == "fast"
        huge = SAMPLES["TMD coil"].with_params(coil_radius=4000.0, turns=3.0)
        assert estimate_cost(huge)[0] == "very slow"

    def test_a_cage_is_called_fast_and_a_coil_is_not(self):
        """Cost is not atom count alone: the implicit modes mesh and
        remesh before relaxing, so a 2000-atom coil takes minutes where a
        2000-atom cage takes under a second."""
        assert estimate_cost(SAMPLES["fullerene"])[0] == "fast"
        assert estimate_cost(SAMPLES["coil (relaxed)"])[0] == "very slow"
        assert estimate_cost(Job("nano-onion", {"n_shells": 3})) [0] == "fast"

    def test_every_implicit_mode_is_flagged_as_at_least_slow(self):
        for mode in IMPLICIT_MODES:
            severity, _ = estimate_cost(SAMPLES[mode])
            assert severity in ("slow", "very slow"), mode

    def test_estimates_are_positive_everywhere(self):
        for mode, job in SAMPLES.items():
            assert estimate_atoms(job) > 0, mode


class TestCommandLine:
    @pytest.mark.parametrize("mode", sorted(SAMPLES))
    def test_the_generated_command_parses(self, mode):
        """A command that looks right and will not run is worse than none."""
        from nanocarbon_lab.cli.main import build_parser

        command = to_cli(SAMPLES[mode], out="out/x")
        argv = shlex.split(command)[1:]  # drop the program name
        build_parser().parse_args(argv)  # raises SystemExit if malformed

    def test_handedness_is_translated_not_printed_as_a_number(self):
        """Builders take +1/-1; the CLI takes right/left. Printing the
        number would produce a command the parser rejects."""
        command = to_cli(SAMPLES["coil (relaxed)"].with_params(handedness=-1))
        assert "--handedness left" in command
        assert "-1" not in command.split("--handedness")[1][:12]

    def test_doping_and_seed_are_carried_through(self):
        job = Job("fullerene", {"freq": 1}, dopant="N", dopant_conc=0.02, seed=7)
        command = to_cli(job)
        assert "--dopant N" in command
        assert "--dopant-conc 0.02" in command
        assert "--seed 7" in command

    def test_a_flag_only_option_appears_without_a_value(self):
        command = to_cli(SAMPLES["coil (relaxed)"].with_params(pin_ends=True))
        assert "--pin-ends" in command
        assert "--pin-ends True" not in command

    def test_no_dopant_means_no_dopant_flag(self):
        assert "--dopant" not in to_cli(SAMPLES["fullerene"])


class TestDoping:
    """The one policy the GUI and the CLI both go through."""

    def test_carbon_is_the_default(self):
        """A job with no dopant builds pure carbon, and the command line
        it generates says nothing about doping."""
        job = SAMPLES["fullerene"]
        assert job.dopant is None
        assert "--dopant" not in to_cli(job, out="out/x")

    def test_the_placement_reaches_the_command_line(self):
        job = Job("capped tube", SAMPLES["capped tube"].params,
                  dopant="N", dopant_conc=0.1, dopant_site="pentagon")
        command = to_cli(job, out="out/x")
        assert "--dopant N" in command
        assert "--dopant-site pentagon" in command

    def test_random_placement_is_left_implicit(self):
        """It is the default on both sides, so emitting it would be
        noise in every generated command."""
        job = Job("capped tube", SAMPLES["capped tube"].params,
                  dopant="N", dopant_conc=0.1)
        assert "--dopant-site" not in to_cli(job, out="out/x")

    def test_an_unknown_placement_is_rejected(self):
        from nanocarbon_lab.builders import build_fullerene
        from nanocarbon_lab.jobs import apply_doping

        job = Job("fullerene", dopant="N", dopant_conc=0.1, dopant_site="face")
        with pytest.raises(ValueError, match="Unknown dopant_site"):
            apply_doping(build_fullerene(family="C60", freq=1), job)

    def test_pentagon_placement_goes_through_build(self):
        """End to end: the placement the Job names is the one applied."""
        from nanocarbon_lab.jobs import build

        job = Job("fullerene", SAMPLES["fullerene"].params, dopant="N",
                  dopant_conc=0.1, dopant_site="pentagon", seed=0)
        atoms = build(job)
        assert atoms.info["doping_mode"] == "ring5"
        assert atoms.get_chemical_symbols().count("N") == 6


class TestDichalcogenideChemistry:
    """The MX2 counterpart of doping, and the same single-policy rule."""

    @pytest.mark.parametrize(
        ("edit", "element", "amount"),
        [("janus", "Se", 1.0), ("alloy", "W", 0.4),
         ("vacancies", "Se", 3), ("antisites", "Se", 2)],
    )
    def test_each_edit_changes_the_structure(self, edit, element, amount):
        from nanocarbon_lab.jobs import build

        params = {"material": "MoS2", "n_layers": 1, "nx": 3, "ny": 3}
        pristine = build(Job("TMD layers", params))
        edited = build(Job("TMD layers", params, tmd_edit=edit,
                           tmd_edit_element=element, tmd_edit_amount=amount,
                           seed=0))
        assert (edited.get_chemical_formula()
                != pristine.get_chemical_formula())

    def test_no_edit_leaves_the_structure_pristine(self):
        from nanocarbon_lab.jobs import build

        params = {"material": "MoS2", "n_layers": 1, "nx": 2, "ny": 2}
        assert (build(Job("TMD layers", params)).get_chemical_formula()
                == build(Job("TMD layers", params,
                             tmd_edit=None)).get_chemical_formula())

    def test_an_unknown_edit_is_rejected(self):
        from nanocarbon_lab.jobs import apply_tmd_chemistry, build

        atoms = build(Job("TMD layers", {"material": "MoS2"}))
        with pytest.raises(ValueError, match="Unknown tmd_edit"):
            apply_tmd_chemistry(atoms, Job("TMD layers", tmd_edit="doping"))

    def test_the_alloy_sublattice_follows_the_element(self):
        """A chalcogen alloys the chalcogens and a metal the metals.
        Asking the user to state it as well would only let the two
        disagree."""
        from nanocarbon_lab.jobs import build

        params = {"material": "MoS2", "n_layers": 1, "nx": 3, "ny": 3}
        metal = build(Job("TMD layers", params, tmd_edit="alloy",
                          tmd_edit_element="W", tmd_edit_amount=0.5, seed=0))
        chalcogen = build(Job("TMD layers", params, tmd_edit="alloy",
                              tmd_edit_element="Se", tmd_edit_amount=0.5,
                              seed=0))
        assert "W" in metal.get_chemical_symbols()
        assert "Se" in chalcogen.get_chemical_symbols()
        assert metal.get_chemical_symbols().count("Mo") < 9
        assert chalcogen.get_chemical_symbols().count("Mo") == 9

    @pytest.mark.parametrize("edit", ["janus", "alloy", "vacancies", "antisites"])
    def test_the_generated_command_parses(self, edit):
        """The GUI's copy-as-command-line button has to produce something
        that runs, for every edit and not only the one that was tried."""
        import shlex

        from nanocarbon_lab.cli.main import build_parser

        job = Job("TMD layers", {"material": "MoS2"}, tmd_edit=edit,
                  tmd_edit_element="W" if edit == "alloy" else "Se",
                  tmd_edit_amount=2)
        command = to_cli(job, out="out/x")
        build_parser().parse_args(shlex.split(command)[1:])

    def test_a_pristine_job_emits_no_chemistry_flags(self):
        assert "--janus" not in to_cli(SAMPLES["TMD layers"], out="out/x")
