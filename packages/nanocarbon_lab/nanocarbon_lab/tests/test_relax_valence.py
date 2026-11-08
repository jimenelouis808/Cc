"""The valence force field, reached from a structure rather than a builder."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from ase import Atoms

from nanocarbon_lab.builders.cnt import build_cnt
from nanocarbon_lab.builders.fullerene_mesh import _valence_terms
from nanocarbon_lab.exports import write_relaxation_request
from nanocarbon_lab.relax import (
    ANGLE_FOR_COORDINATION,
    angle_targets,
    bonds_from_atoms,
    relax_structure,
)
from nanocarbon_lab.relax.valence import _angle_triplets


def _roughed_tube(sigma: float = 0.12, seed: int = 3):
    """A (6,6) tube with every atom displaced, so there is work to do."""
    atoms = build_cnt(6, 6, length=12.0)
    rng = np.random.default_rng(seed)
    atoms.set_positions(
        atoms.get_positions() + rng.normal(0, sigma, (len(atoms), 3)))
    return atoms


class TestAngleTargetOrdering:
    """The invariant the whole per-angle scheme rests on."""

    def test_the_triplets_match_the_engine_index_for_index(self):
        """Targets are matched to the engine's angles POSITIONALLY.

        `relax_structure` computes one target per triplet and hands the
        array to the engine, which builds its own triplet list from the
        same bonds. If the two orders ever diverge, every target lands
        on a different angle and the structure relaxes toward nonsense
        that still looks like a relaxed structure. Nothing else would
        catch it, so this is pinned here.
        """
        atoms = build_cnt(6, 6, length=9.0)
        bonds, _, _ = bonds_from_atoms(atoms)
        mine = _angle_triplets(bonds)
        _, theirs, _ = _valence_terms(list(bonds), len(atoms))
        assert mine.shape == theirs.shape
        assert np.array_equal(mine, theirs)


class TestAngleTargets:
    def test_a_three_coordinate_centre_asks_for_120_degrees(self):
        triplets = np.array([[0, 1, 2]])
        coordination = np.array([1, 3, 1])
        assert angle_targets(triplets, coordination)[0] == pytest.approx(120.0)

    def test_a_four_coordinate_centre_asks_for_the_tetrahedral_angle(self):
        triplets = np.array([[0, 1, 2]])
        coordination = np.array([1, 4, 1])
        assert angle_targets(triplets, coordination)[0] == pytest.approx(
            math.degrees(math.acos(-1 / 3)))

    def test_a_centre_with_no_ideal_angle_gets_no_target(self):
        """Six-coordinate has several angles at once, so it gets none.

        An averaged target would pull a perfectly good octahedron toward
        an angle no octahedron has; the non-bonded repulsion holds those
        ligands apart instead.
        """
        triplets = np.array([[0, 1, 2]])
        coordination = np.array([1, 6, 1])
        assert not np.isfinite(angle_targets(triplets, coordination)[0])

    def test_an_override_replaces_the_default(self):
        triplets = np.array([[0, 1, 2]])
        coordination = np.array([1, 3, 1])
        out = angle_targets(triplets, coordination, override={3: 108.0})
        assert out[0] == pytest.approx(108.0)
        assert ANGLE_FOR_COORDINATION[3] == 120.0, "el catálogo no se toca"


class TestBondSource:
    def test_a_recorded_bond_list_is_preferred_over_distances(self):
        """Because a radius cutoff invents bonds on a curved structure.

        The package says this in several places already: on a saddle
        plenty of non-bonded atoms sit closer than a bond length. A
        builder that recorded its bonds knows better than any cutoff.
        """
        atoms = build_cnt(6, 6, length=9.0)
        atoms.info["bonds"] = [(0, 1)]
        bonds, source, warnings = bonds_from_atoms(atoms)
        assert source == "info"
        assert bonds == [(0, 1)]
        assert not warnings

    def test_guessing_warns_when_a_carbon_comes_out_overcoordinated(self):
        """Five bonds on a carbon is an invented bond, and it is said."""
        # A carbon with five neighbours packed inside the cutoff: not a
        # real structure, which is the point -- the guesser must not
        # return it quietly.
        positions = [(0, 0, 0)] + [
            (1.4 * math.cos(t), 1.4 * math.sin(t), 0.0)
            for t in np.linspace(0, 2 * math.pi, 5, endpoint=False)
        ]
        atoms = Atoms("C6", positions=positions)
        _, source, warnings = bonds_from_atoms(atoms)
        assert source == "distances"
        assert any("more than" in w for w in warnings)

    def test_a_structure_with_no_bonds_at_all_is_refused(self):
        atoms = Atoms("C2", positions=[(0, 0, 0), (12, 0, 0)])
        with pytest.raises(ValueError, match="no bonds"):
            relax_structure(atoms)


class TestRelaxStructure:
    def test_it_cleans_a_roughed_up_tube(self):
        atoms = _roughed_tube()
        relax_structure(atoms)
        info = atoms.info["valence_relax"]
        before, after = info["before"], info["after"]
        assert after["bond_std"] < before["bond_std"]
        assert after["angle_min"] > before["angle_min"]
        assert after["angle_max"] < before["angle_max"]

    def test_the_topology_cannot_change(self):
        """The bond list is held for the whole minimisation, by design.

        This is what separates a geometry cleaner from a simulation: the
        ring census of the input is the ring census of the output, so
        relaxing can never be blamed for a census that was already
        wrong.
        """
        atoms = _roughed_tube()
        before = sorted(bonds_from_atoms(atoms)[0])
        relax_structure(atoms, bonds=before)
        assert atoms.info["valence_relax"]["bonds"] == len(before)

    def test_a_triclinic_periodic_cell_is_refused_with_the_reason(self):
        """Not relaxed against wrong minimum images.

        The engine's neighbour list is a cKDTree with boxsize, which is
        orthorhombic-only. Relaxing a triclinic cell anyway would take
        the wrong image across every seam, and the output would look
        perfectly fine.
        """
        atoms = build_cnt(6, 6, length=9.0)
        cell = np.array(atoms.cell)
        cell[2, 0] = 3.0
        atoms.set_cell(cell)
        atoms.set_pbc(True)
        with pytest.raises(ValueError, match="triclinic"):
            relax_structure(atoms)

    def test_a_free_structure_gets_no_box(self):
        atoms = _roughed_tube()
        atoms.set_pbc(False)
        relax_structure(atoms)
        assert atoms.info["valence_relax"]["bonds"] > 0


class TestRelaxationRequest:
    def test_it_writes_a_folder_the_format_describes(self, tmp_path):
        atoms = build_cnt(6, 6, length=9.0)
        atoms.info["bonds"] = [(0, 1)]
        folder = write_relaxation_request(atoms, tmp_path / "pedido",
                                          reason="prueba")
        request = json.loads((folder / "solicitud.json").read_text())
        assert request["esquema"] == "nanocarbon-dft/1"
        assert request["codigo"] == "dftb+"
        assert request["tareas"][0]["tipo"] == "relajar"
        assert request["origen"]["programa"] == "nanocarbon_lab"
        assert (folder / "estructura.extxyz").exists()

    def test_the_structure_goes_out_without_info(self, tmp_path):
        """A bond list in the extxyz comment line breaks ASE's reader.

        Measured on an 18 130-atom superfullerene: written with its
        9 005 bonds in `info`, the file ASE produces is one ASE then
        refuses to read back ("inhomogeneous shape"). The external code
        derives its own bonding anyway, which is the point of asking it.
        """
        from ase.io import read

        atoms = build_cnt(6, 6, length=9.0)
        atoms.info["bonds"] = [(i, i + 1) for i in range(len(atoms) - 1)]
        folder = write_relaxation_request(atoms, tmp_path / "pedido")
        back = read(folder / "estructura.extxyz")
        assert "bonds" not in back.info
        assert len(back) == len(atoms)

    def test_an_unknown_code_is_refused(self, tmp_path):
        atoms = build_cnt(6, 6, length=9.0)
        with pytest.raises(ValueError, match="unknown code"):
            write_relaxation_request(atoms, tmp_path / "p", code="vasp")

    def test_it_will_not_overwrite_a_job_in_flight(self, tmp_path):
        """A request folder holds its own partial results."""
        atoms = build_cnt(6, 6, length=9.0)
        folder = write_relaxation_request(atoms, tmp_path / "pedido")
        with pytest.raises(ValueError, match="already exists"):
            write_relaxation_request(atoms, folder)
