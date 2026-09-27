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
    return load_parameters("xu_chn")


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
