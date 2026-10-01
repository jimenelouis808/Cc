"""Intra-atomic s-p dipoles in the optical response (tbkit.dipoles).

What must hold whatever the numbers: the screened linear response equals the
finite-field solution of the same functional; unscreened, it equals the sum
over states with the full position operator; flipping the phase convention
of the p orbitals changes nothing physical; and the zeros that point charges
force (α⊥ of a chain, out-of-plane α of a flat molecule) are gone.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule

from tbkit.dipoles import (
    dipole_matrices,
    multipole_kernel,
    polarizability_finite_field,
    signed_dipoles,
)
from tbkit.hamiltonian import System
from tbkit.optics import dielectric_constant, polarizability_finite, polarizability_linear_response
from tbkit.params import load_parameters, xu_carbon
from tbkit.scc import self_consistent


@dataclasses.dataclass(frozen=True)
class Negated:
    """``-law(d)``: flips the sign convention of a Slater-Koster integral."""

    law: object

    @property
    def cutoff(self):
        return self.law.cutoff

    def __call__(self, d):
        return -self.law(d)


def flipped_p(model):
    """The same model with every p orbital's phase reversed (all spσ negated)."""
    hopping = {key: (Negated(law) if key[2] == "sps" else law)
               for key, law in model.hopping.items()}
    return dataclasses.replace(model, hopping=hopping, metadata={})


@pytest.fixture(scope="module")
def chn():
    """xu_chn without its extra atomic polarizability: the TB response alone."""
    return dataclasses.replace(load_parameters("xu_chn"), extra_polarizability={})


class TestSignAndMatrices:
    def test_sign_follows_the_model_convention(self, chn):
        assert all(d > 0 for d in signed_dipoles(chn).values())         # Xu: ssσ < 0 < spσ
        assert all(d < 0 for d in signed_dipoles(flipped_p(chn)).values())

    def test_dipole_matrix_blocks(self, chn):
        system = System.build(molecule("HCN"), chn)
        d = dipole_matrices(system)
        assert np.allclose(d, d.transpose(0, 2, 1))
        c = system.basis.of_atom(0)                     # molecule("HCN"): C, N, H
        block = d[2][c.start:c.stop, c.start:c.stop]
        assert block[0, 3] == pytest.approx(0.4952)     # <s|z|pz>
        h = system.basis.of_atom(2)
        assert not d[:, h.start:h.stop, :].any()        # H has no p: no dipole

    def test_kernel_symmetric_with_positive_self_terms(self, chn):
        system = System.build(molecule("C5H5N"), chn)
        kernel = multipole_kernel(system)
        assert np.allclose(kernel, kernel.T)
        n = len(system.basis.atoms)
        assert np.all(np.diag(kernel)[n:] > 0)


class TestExactRelations:
    @pytest.mark.parametrize("name", ["HCN", "C5H5N"])
    def test_linear_response_equals_finite_field(self, chn, name):
        system = System.build(molecule(name), chn)
        lr = polarizability_linear_response(system)
        ff = polarizability_finite_field(system, field=0.002)
        assert lr == pytest.approx(ff, rel=1e-5, abs=1e-5)

    def test_finite_field_with_a_bare_ground_state(self):
        ring = Atoms("C4", positions=[[0, 0, 0], [1.4, 0, 0], [2.1, 1.2, 0.3], [0.7, 1.3, -0.2]])
        model = dataclasses.replace(xu_carbon())
        system = System.build(ring, model)
        lr = polarizability_linear_response(system)
        ff = polarizability_finite_field(system, field=0.002)
        assert lr == pytest.approx(ff, rel=1e-5, abs=1e-5)

    def test_unscreened_equals_sum_over_states(self, chn):
        system = System.build(molecule("C5H5N"), chn)
        unscreened = polarizability_linear_response(system, screened=False)
        sos = polarizability_finite(self_consistent(system, tol=1e-10).solution)
        assert unscreened == pytest.approx(sos, rel=1e-8, abs=1e-8)

    def test_flipping_the_p_convention_changes_nothing(self, chn):
        atoms = molecule("C5H5N")
        a = polarizability_linear_response(System.build(atoms, chn))
        b = polarizability_linear_response(System.build(atoms, flipped_p(chn)))
        assert a == pytest.approx(b, rel=1e-8, abs=1e-8)

    def test_no_dipoles_means_the_old_result(self, chn):
        atoms = molecule("C5H5N")
        without = dataclasses.replace(chn, onsite_dipole={})
        a = polarizability_linear_response(System.build(atoms, without))
        b = polarizability_linear_response(System.build(atoms, chn), onsite_dipoles=False)
        assert a == pytest.approx(b, rel=1e-10)


class TestThePhysics:
    def test_linear_molecule_has_perpendicular_alpha(self, chn):
        alpha = polarizability_linear_response(System.build(molecule("HCN"), chn))
        values = np.sort(np.linalg.eigvalsh(alpha))
        assert values[0] > 0.1 and values[0] == pytest.approx(values[1], rel=1e-6)
        point = polarizability_linear_response(System.build(molecule("HCN"), chn),
                                               onsite_dipoles=False)
        assert np.sort(np.linalg.eigvalsh(point))[0] == pytest.approx(0.0, abs=1e-8)

    def test_flat_molecule_has_out_of_plane_alpha(self, chn):
        alpha = polarizability_linear_response(System.build(molecule("C6H6"), chn))
        assert np.sort(np.linalg.eigvalsh(alpha))[0] > 0.5

    def test_bond_alpha_shrinks_with_hybrid_centroids(self, chn):
        """σ→σ* transition dipoles join hybrid centroids, closer than the nuclei:
        along the bonds the point-dipole α is an overestimate."""
        system = System.build(molecule("C6H6"), chn)
        with_d = np.sort(np.linalg.eigvalsh(polarizability_linear_response(system)))
        point = np.sort(np.linalg.eigvalsh(polarizability_linear_response(
            system, onsite_dipoles=False)))
        assert with_d[-1] < point[-1]

    def test_diamond_stays_cubic(self):
        eps = dielectric_constant(System.build(bulk("C", "diamond", a=3.567), xu_carbon()),
                                  kmesh=6)
        assert np.allclose(eps, eps[0, 0] * np.eye(3), atol=1e-6)
        assert eps[0, 0] > 1.0


class TestExtraPolarizability:
    @pytest.fixture(scope="class")
    @staticmethod
    def extra(chn):
        return dataclasses.replace(chn, extra_polarizability={"H": 0.3, "C": 0.8, "N": 0.6})

    @pytest.mark.parametrize("dipoles", [True, False])
    def test_linear_response_equals_finite_field(self, extra, dipoles):
        system = System.build(molecule("C5H5N"), extra)
        lr = polarizability_linear_response(system, onsite_dipoles=dipoles)
        ff = polarizability_finite_field(system, field=0.002, onsite_dipoles=dipoles)
        assert lr == pytest.approx(ff, rel=1e-5, abs=1e-5)

    def test_unscreened_adds_the_atomic_sum(self, extra):
        system = System.build(molecule("HCN"), extra)
        unscreened = polarizability_linear_response(system, screened=False)
        sos = polarizability_finite(self_consistent(system, tol=1e-10).solution)
        assert unscreened == pytest.approx(sos, rel=1e-8, abs=1e-8)
        bare = polarizability_finite(self_consistent(system, tol=1e-10).solution,
                                     extra_polarizability=False)
        assert np.diag(sos - bare) == pytest.approx([0.3 + 0.8 + 0.6] * 3)

    def test_switching_off_recovers_the_model(self, chn, extra):
        atoms = molecule("C5H5N")
        a = polarizability_linear_response(System.build(atoms, chn))
        b = polarizability_linear_response(System.build(atoms, extra),
                                           extra_polarizability=False)
        assert a == pytest.approx(b, rel=1e-10)

    def test_extra_dipoles_have_no_self_interaction(self, extra):
        from tbkit.dipoles import response_kernel

        system = System.build(molecule("HCN"), extra)
        kernel = response_kernel(system, dipoles=True, extra=True)
        n = len(system.basis.atoms)
        start = 4 * n
        for atom in range(n):
            block = kernel[start + 3 * atom:start + 3 * atom + 3, :]
            assert not block[:, start + 3 * atom:start + 3 * atom + 3].any()
            assert not block[:, n + 3 * atom:n + 3 * atom + 3].any()
        assert np.allclose(kernel, kernel.T)

    def test_raises_every_component(self, chn, extra):
        atoms = molecule("C6H6")
        a = np.sort(np.linalg.eigvalsh(polarizability_linear_response(System.build(atoms, chn))))
        b = np.sort(np.linalg.eigvalsh(polarizability_linear_response(
            System.build(atoms, extra))))
        assert np.all(b > a)


def test_periodic_screened_alpha_is_the_molecule_with_its_lorentz_field():
    """A molecule in a cubic box: the unscreened α is the molecule's exactly, and
    the screened one is α / (1 - 4π α / 3V), the Lorentz field of its images
    (Ewald drops G = 0, so the applied field is the macroscopic one)."""
    from tbkit.optics import polarizability_linear_response, polarizability_periodic_screened

    model = load_parameters("xu_chno")
    mol = molecule("C6H6")
    box = mol.copy()
    box.set_cell([16.0] * 3)
    box.center()
    box.pbc = True
    kw = {"onsite_dipoles": False, "extra_polarizability": False}
    gamma_only = {"kpts": np.zeros((1, 3)), "weights": np.ones(1)}
    for screened in (False, True):
        finite = polarizability_linear_response(System.build(mol, model), screened=screened, **kw)
        crystal = polarizability_periodic_screened(System.build(box, model), screened=screened,
                                                   **gamma_only, **kw)
        expected = finite[0, 0] / (1 - 4 * np.pi * finite[0, 0] / (3 * box.get_volume())) \
            if screened else finite[0, 0]
        assert crystal[0, 0] == pytest.approx(expected, rel=2e-3)
        assert crystal[2, 2] == pytest.approx(finite[2, 2], abs=1e-6)


def test_charge_local_fields_vanish_by_symmetry_and_appear_when_it_is_broken():
    """In h-BN every atom sits on a C3 axis: a uniform in-plane field cannot put a
    net charge on it, so the charge local fields are exactly zero. Displace one
    atom and they appear, lowering α."""
    from ase.build import graphene

    from tbkit.optics import polarizability_periodic_screened

    model = load_parameters("xu_chnob")
    kw = {"kmesh": (8, 8, 1), "onsite_dipoles": False, "extra_polarizability": False}

    def both(atoms):
        system = System.build(atoms, model)
        return (polarizability_periodic_screened(system, screened=False, **kw),
                polarizability_periodic_screened(system, screened=True, **kw))

    hbn = graphene("BN", a=2.50, vacuum=7.5)
    bare, screened = both(hbn)
    assert screened == pytest.approx(bare, abs=1e-8)
    assert screened[0, 0] == pytest.approx(screened[1, 1], rel=1e-8)    # hexagonal
    distorted = hbn.repeat((2, 2, 1))
    distorted.positions[0] += [0.08, 0.03, 0.0]
    bare, screened = both(distorted)
    assert 0 < screened[0, 0] < bare[0, 0] - 1e-3
