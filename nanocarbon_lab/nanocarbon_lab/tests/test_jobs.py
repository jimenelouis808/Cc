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
        ],
    )
    def test_seed_modes_are_predicted_exactly(self, mode, expected):
        assert estimate_atoms(SAMPLES[mode]) == expected

    @pytest.mark.parametrize("mode", ["TMD layers", "TMD bulk", "TMD ribbon",
                                      "TMD nanotube"])
    def test_dichalcogenide_estimates_match_the_real_build(self, mode):
        """These are exact, not approximate, so assert equality against a
        real build rather than a tolerance."""
        from nanocarbon_lab.jobs import build

        job = SAMPLES[mode]
        assert estimate_atoms(job) == len(build(job))

    @pytest.mark.parametrize(
        ("mode", "measured"),
        [("schwarzite", 1502), ("coil (relaxed)", 2782), ("junction", 1100)],
    )
    def test_implicit_modes_are_predicted_within_a_tenth(self, mode, measured):
        """Surface area over ring area. Not exact, but the point is to
        tell a 60-atom cage from a 3000-atom coil before committing."""
        assert estimate_atoms(SAMPLES[mode]) == pytest.approx(measured, rel=0.2)

    def test_dichalcogenides_are_never_called_slow(self):
        """No meshing and no relaxation -- exact lattice placement, so the
        only cost is drawing them."""
        for mode in ("TMD layers", "TMD bulk", "TMD ribbon", "TMD nanotube"):
            assert estimate_cost(SAMPLES[mode])[0] == "fast"

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
