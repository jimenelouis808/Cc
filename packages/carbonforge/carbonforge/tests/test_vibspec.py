"""Tests for vibspec phase 1: finite ribbons, presets, sites and physical checks.

Everything here is geometry and bookkeeping on flakes of a few dozen atoms.
No DFT code is needed or called.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ase.build import molecule

from carbonforge.builders import build_finite_nanoribbon, build_nanoribbon
from carbonforge.cli.main import main as cli_main
from carbonforge.defects import introduce_vacancies
from carbonforge.functionalization import make_graphitic_n, nitrogen_report
from carbonforge.topology import build_bond_graph
from carbonforge.vibspec.core import (
    PRESETS,
    apply_preset,
    check_ready_for_vibrations,
    check_structure,
    check_vibration_settings,
    edge_sites,
    electron_count,
    interior_carbons,
    pick_edge_site,
    pick_interior_carbon,
    strip_hydrogen,
    suggest_spin,
    undercoordinated_atoms,
    unformed_pyrrolic_nitrogens,
    vacuum_per_side,
    zigzag_runs,
)


@pytest.fixture(scope="module")
def armchair():
    return build_finite_nanoribbon(5, 3, edge="armchair")


@pytest.fixture(scope="module")
def zigzag():
    return build_finite_nanoribbon(4, 4, edge="zigzag")


class TestFiniteNanoribbon:
    @pytest.mark.parametrize("edge,width,length", [
        ("armchair", 5, 3), ("armchair", 7, 2), ("zigzag", 3, 3), ("zigzag", 4, 4),
    ])
    def test_finite_and_fully_terminated(self, edge, width, length):
        atoms = build_finite_nanoribbon(width, length, edge=edge)
        assert not any(atoms.get_pbc())
        graph = build_bond_graph(atoms)
        symbols = atoms.get_chemical_symbols()
        for i, s in enumerate(symbols):
            expected = 3 if s == "C" else 1
            assert graph.degree[i] == expected, (i, s, graph.degree[i])
        assert symbols.count("H") == atoms.info["n_hydrogen"]

    def test_equal_vacuum_on_every_side(self, armchair):
        for gap in vacuum_per_side(armchair).values():
            assert gap == pytest.approx(7.0)

    def test_rejects_too_little_vacuum(self):
        with pytest.raises(ValueError, match="vacío|vacuum"):
            build_finite_nanoribbon(5, 3, vacuum_per_side=4.0)

    def test_ends_have_the_other_edge_type(self, armchair, zigzag):
        assert armchair.info["end_edge"] == "zigzag"
        assert zigzag.info["end_edge"] == "armchair"

    def test_hydrogen_is_not_a_functional_group(self, armchair):
        assert "functionalization" not in armchair.info

    def test_periodic_builder_unchanged(self):
        atoms = build_nanoribbon(4, 2, edge="zigzag")
        assert tuple(atoms.get_pbc()) == (False, False, True)


class TestDeterministicSites:
    def test_vacancy_at_given_site(self, armchair):
        site = pick_interior_carbon(armchair)
        out = introduce_vacancies(armchair, sites=[site])
        assert out.info["defects"][-1]["removed_indices"] == [site]
        assert len(out) == len(armchair) - 1

    def test_divacancy_takes_a_neighbour(self, armchair):
        site = pick_interior_carbon(armchair)
        removed = introduce_vacancies(armchair, kind="di", sites=[site]).info["defects"][-1][
            "removed_indices"
        ]
        assert site in removed and len(removed) == 2
        assert build_bond_graph(armchair).has_edge(*removed)

    def test_vacancy_site_out_of_range(self, armchair):
        with pytest.raises(IndexError):
            introduce_vacancies(armchair, sites=[10_000])

    def test_graphitic_at_given_index(self, armchair):
        site = pick_interior_carbon(armchair)
        out = make_graphitic_n(armchair, indices=[site])
        assert out[site].symbol == "N"
        assert out.info["nitrogen_configurations"][-1]["indices"] == [site]

    def test_graphitic_rejects_non_basal(self, armchair):
        hydrogen = armchair.get_chemical_symbols().index("H")
        with pytest.raises(ValueError, match="coordinación 3"):
            make_graphitic_n(armchair, indices=[hydrogen])

    def test_middle_site_has_the_ribbon_edge_type(self, armchair, zigzag):
        assert pick_edge_site(armchair).edge == "armchair"
        assert pick_edge_site(zigzag).edge == "zigzag"

    def test_zigzag_runs(self, armchair):
        # Two zigzag ends of four sites, and a two-site step at two corners.
        assert [len(run) for run in zigzag_runs(armchair)] == [4, 4, 2, 2]

    def test_armchair_edges_come_in_pairs(self, armchair):
        # A corner carbon may be labelled zigzag, so the partner is only
        # guaranteed to be *some* terminated carbon.
        graph = build_bond_graph(armchair)
        terminated = {s.carbon for s in edge_sites(armchair)}
        for site in edge_sites(armchair):
            if site.edge == "armchair":
                assert any(n in terminated for n in graph.neighbors(site.carbon))

    def test_center_is_interior_and_central(self, armchair):
        site = pick_interior_carbon(armchair)
        assert site in interior_carbons(armchair)
        centre = armchair.get_positions()[[s == "C" for s in armchair.get_chemical_symbols()]]
        distances = np.linalg.norm(armchair.get_positions()[interior_carbons(armchair)]
                                   - centre.mean(axis=0), axis=1)
        assert np.linalg.norm(armchair[site].position - centre.mean(axis=0)) == pytest.approx(
            distances.min()
        )

    def test_near_edge_is_bonded_to_the_middle_edge_site(self, armchair):
        site = pick_interior_carbon(armchair, "near_edge")
        assert build_bond_graph(armchair).has_edge(site, pick_edge_site(armchair).carbon)

    def test_explicit_index_validated(self, armchair):
        with pytest.raises(ValueError):
            pick_edge_site(armchair, pick_interior_carbon(armchair))
        with pytest.raises(ValueError):
            pick_interior_carbon(armchair, pick_edge_site(armchair).carbon)

    def test_strip_hydrogen_tracks_the_carbon(self, armchair):
        site = pick_edge_site(armchair)
        out, carbon = strip_hydrogen(armchair, site)
        assert len(out) == len(armchair) - 1
        assert np.allclose(out[carbon].position, armchair[site.carbon].position)


#: Composition change each preset makes on a terminated flake.
_DELTA = {
    "pristine": {},
    "graphitic": {"C": -1, "N": 1},
    "pyridinic_edge": {"C": -1, "H": -1, "N": 1},
    "pyridinic_vacancy": {"C": -4, "N": 3},
    "pyrrolic_precursor": {"C": -3, "N": 1, "H": 1},
    "amine": {"N": 1, "H": 1},
    "nitrile": {"C": 1, "N": 1, "H": -1},
    "pyridinic_n_oxide": {"C": -1, "H": -1, "N": 1, "O": 1},
    "hydroxyl": {"O": 1},
    "carboxyl": {"C": 1, "O": 2},
    "carbonyl": {"H": -1, "O": 1},
    "epoxide": {"O": 1},
}

#: Presets that leave an odd electron count on an even flake.
_OPEN_SHELL = {"graphitic", "pyridinic_vacancy", "carbonyl"}


class TestPresets:
    def test_every_preset_has_an_expectation(self):
        assert set(PRESETS) == set(_DELTA)

    @pytest.mark.parametrize("flake", ["armchair", "zigzag"])
    @pytest.mark.parametrize("key", sorted(_DELTA))
    def test_composition_and_geometry(self, key, flake, request):
        host = request.getfixturevalue(flake)
        out = apply_preset(host, key)
        change = Counter(out.get_chemical_symbols())
        change.subtract(Counter(host.get_chemical_symbols()))
        assert {k: v for k, v in change.items() if v} == _DELTA[key]

        report = check_structure(out)
        assert report.ok, report.summary()
        for gap in vacuum_per_side(out).values():
            assert gap == pytest.approx(7.0)
        assert out.info["vibspec_preset"]["key"] == key

    @pytest.mark.parametrize("key", sorted(_DELTA))
    def test_electron_parity(self, key, armchair):
        assert electron_count(armchair) % 2 == 0
        odd = electron_count(apply_preset(armchair, key)) % 2 == 1
        assert odd == (key in _OPEN_SHELL)

    @pytest.mark.parametrize("key", sorted(_DELTA))
    def test_deterministic(self, key, armchair):
        first, second = apply_preset(armchair, key), apply_preset(armchair, key)
        assert first.get_chemical_symbols() == second.get_chemical_symbols()
        assert np.allclose(first.get_positions(), second.get_positions())

    def test_input_not_mutated(self, armchair):
        before = armchair.get_positions().copy()
        apply_preset(armchair, "carboxyl")
        assert np.array_equal(before, armchair.get_positions())

    def test_edge_group_replaces_the_hydrogen(self, armchair):
        site = pick_edge_site(armchair)
        out = apply_preset(armchair, "amine")
        _, carbon = strip_hydrogen(armchair, site)
        graph = build_bond_graph(out)
        neighbours = sorted(out[n].symbol for n in graph.neighbors(carbon))
        assert neighbours == ["C", "C", "N"]

    def test_edge_selection_by_type(self, armchair):
        out = apply_preset(armchair, "pyridinic_edge", edge="zigzag")
        assert out.info["nitrogen_configurations"][-1]["edge"] == "zigzag"

    def test_nitrogen_report_understands_presets(self, armchair):
        text = nitrogen_report(apply_preset(armchair, "pyridinic_n_oxide"))
        assert "pyridinic_n_oxide" in text

    def test_rejects_periodic_structures(self):
        with pytest.raises(ValueError, match="finitas"):
            apply_preset(build_nanoribbon(4, 2, passivate=True), "amine")

    def test_unknown_preset(self, armchair):
        with pytest.raises(ValueError, match="Disponibles"):
            apply_preset(armchair, "nitro-something")


class TestStructureChecks:
    def test_periodic_is_an_error(self):
        report = check_structure(build_nanoribbon(4, 2, passivate=True))
        assert not report.ok
        assert any("periódica" in e for e in report.errors)

    def test_small_vacuum_is_an_error(self, armchair):
        tight = armchair.copy()
        tight.set_cell(np.array(tight.cell) - np.diag([6.0, 6.0, 6.0]), scale_atoms=False)
        tight.translate(-3.0)
        report = check_structure(tight)
        assert any("vacío por lado" in e for e in report.errors)

    def test_nitrile_carbon_is_not_dangling(self, armchair):
        assert undercoordinated_atoms(apply_preset(armchair, "nitrile")) == []

    def test_pyrrolic_precursor_is_flagged(self, armchair):
        out = apply_preset(armchair, "pyrrolic_precursor")
        assert unformed_pyrrolic_nitrogens(out)
        assert any("pentágono" in w for w in check_structure(out).warnings)

    def test_closed_pyrrole_ring_passes(self):
        pyrrole = molecule("C4H4NH")
        nitrogen = pyrrole.get_chemical_symbols().index("N")
        pyrrole.info["nitrogen_configurations"] = [
            {"type": "pyrrolic_precursor", "indices": [nitrogen]}
        ]
        assert unformed_pyrrolic_nitrogens(pyrrole) == []


class TestSpinAdvice:
    def test_closed_shell_flake_without_long_zigzag(self, zigzag):
        assert max(len(r) for r in zigzag_runs(zigzag)) < 4
        advice = suggest_spin(zigzag)
        assert not advice.spinpol
        assert not advice.magmoms.any()

    def test_odd_electron_count(self, zigzag):
        advice = suggest_spin(apply_preset(zigzag, "graphitic"))
        assert advice.spinpol
        assert advice.magmoms.sum() == pytest.approx(1.0)
        assert any("impar" in r for r in advice.reasons)

    def test_zigzag_ends_start_antiparallel(self, armchair):
        advice = suggest_spin(armchair)
        assert advice.spinpol
        ends = [run for run in zigzag_runs(armchair) if len(run) >= 4]
        signs = [set(np.sign(advice.magmoms[run])) for run in ends]
        assert len(signs[0]) == len(signs[1]) == 1
        assert signs[0] == {-s for s in signs[1]}
        assert advice.magmoms.sum() == pytest.approx(0.0)

    def test_dangling_bonds(self, armchair):
        out = apply_preset(armchair, "pyrrolic_precursor")
        advice = suggest_spin(out)
        assert any("colgantes" in r for r in advice.reasons)

    def test_benzene_is_closed_shell(self):
        assert not suggest_spin(molecule("C6H6")).spinpol


class TestVibrationGate:
    def test_unrelaxed_is_an_error(self, zigzag):
        report = check_ready_for_vibrations(zigzag, relaxed_fmax=None)
        assert any("no está relajada" in e for e in report.errors)

    @pytest.mark.parametrize("fmax,ok,warned", [(0.005, True, False), (0.03, True, True),
                                                (0.1, False, False)])
    def test_residual_force(self, zigzag, fmax, ok, warned):
        # This flake needs no spin (no long zigzag edge, even electrons).
        report = check_ready_for_vibrations(zigzag, relaxed_fmax=fmax, spinpol=False)
        assert report.ok == ok
        assert any("Fuerza residual" in w for w in report.warnings) == warned

    def test_unclosed_pyrrolic_after_relax_is_an_error(self, armchair):
        out = apply_preset(armchair, "pyrrolic_precursor")
        report = check_ready_for_vibrations(out, relaxed_fmax=0.005, spinpol=True)
        assert any("no está en un pentágono" in e for e in report.errors)

    def test_odd_electrons_without_spin_is_an_error(self, zigzag):
        out = apply_preset(zigzag, "graphitic")
        report = check_ready_for_vibrations(out, relaxed_fmax=0.005, spinpol=False)
        assert any("sin espín" in e for e in report.errors)
        assert check_ready_for_vibrations(out, relaxed_fmax=0.005, spinpol=True).ok

    def test_long_zigzag_without_spin_is_an_error(self, armchair):
        report = check_ready_for_vibrations(armchair, relaxed_fmax=0.005, spinpol=False)
        assert any("zigzag" in e for e in report.errors)
        assert check_ready_for_vibrations(armchair, relaxed_fmax=0.005, spinpol=True).ok

    def test_settings(self):
        assert check_vibration_settings(fmax=0.01).ok
        assert not check_vibration_settings(fmax=0.1).ok
        assert check_vibration_settings(fmax=0.03).warnings
        assert not check_vibration_settings(fmax=0.01, nfree=3).ok
        assert check_vibration_settings(fmax=0.01, delta=0.001).warnings
        assert check_vibration_settings(fmax=0.01, delta=0.05).warnings
        assert check_vibration_settings(fmax=0.01, mode="lcao", h=0.2).warnings
        assert not check_vibration_settings(fmax=0.01, mode="pw", h=0.2).warnings
        assert check_vibration_settings(fmax=0.01, scale_factor=1.04).ok
        assert check_vibration_settings(fmax=0.01, scale_factor=1.04).warnings == []
        assert check_vibration_settings(fmax=0.01, scale_factor=1.3).warnings


class TestCli:
    def test_build_and_check(self, tmp_path, capsys):
        out = tmp_path / "amine.xyz"
        assert cli_main(["vibspec", "build", "--preset", "amine", "-o", str(out)]) == 0
        assert out.exists()
        assert cli_main(["vibspec", "check", str(out)]) == 0
        assert "n_electrons" in capsys.readouterr().out

    def test_check_fails_on_periodic(self, tmp_path):
        path = tmp_path / "periodic.xyz"
        build_nanoribbon(4, 2, passivate=True).write(path)
        assert cli_main(["vibspec", "check", str(path)]) == 1

    def test_presets(self, capsys):
        assert cli_main(["vibspec", "presets"]) == 0
        assert "pyridinic_n_oxide" in capsys.readouterr().out


def test_core_imports_no_gui():
    """The core must run headless on a cluster: importing it loads no GUI code."""
    import subprocess
    import sys

    code = (
        "import sys, carbonforge.vibspec.core; "
        "bad = [m for m in sys.modules if m.startswith(('carbonforge.gui', 'tkinter'))]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
