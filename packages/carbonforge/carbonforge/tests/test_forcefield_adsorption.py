"""The adsorption force field: groups retyped, charged neutral, and the
electrode table (which the EDLC cells use) left exactly as it was."""

from collections import Counter

import numpy as np
import pytest

from carbonforge.adsorption.co2 import wall_parameters
from carbonforge.builders import build_graphene_supercell
from carbonforge.forcefields.adsorption import (
    ADSORPTION_PARAMS, assign_adsorption_types)
from carbonforge.forcefields.parameters import ELECTRODE_PARAMS
from carbonforge.functionalization.attach import functionalize_random


def _sheet(group, n):
    sheet = build_graphene_supercell(nx=6, ny=4, vacuum=10.0)
    return functionalize_random(sheet, group, n_groups=n,
                                site_kind="basal", seed=0)


def test_the_electrode_table_is_untouched():
    assert ELECTRODE_PARAMS["C_O"].charge == pytest.approx(0.15)
    assert ELECTRODE_PARAMS["N_amine"].charge == pytest.approx(-0.90)
    assert "C_carboxyl" not in ELECTRODE_PARAMS


@pytest.mark.parametrize("group,n,per_group", [
    ("COOH", 2, 0.0), ("OH", 2, 0.0), ("NH2", 2, -0.109)])
def test_groups_carry_their_published_net_charge(group, n, per_group):
    atoms = _sheet(group, n)
    typed = assign_adsorption_types(atoms)
    assert not typed.unresolved
    net = sum(ADSORPTION_PARAMS[t].charge for t in typed.types)
    assert net == pytest.approx(n * per_group, abs=1e-9)


def test_a_carboxyl_is_typed_as_one():
    counts = Counter(assign_adsorption_types(_sheet("COOH", 2)).types)
    for t in ("C_carboxyl", "C_base_carboxyl", "O_carboxyl_double",
              "O_carboxyl_hydroxyl", "H_carboxyl"):
        assert counts[t] == 2, t
    assert counts["C_O"] == 0


def test_the_wall_uses_it_by_default_and_the_electrode_on_request():
    atoms = _sheet("COOH", 2)
    assert wall_parameters(atoms, neutralise=False)["net_charge"] == \
        pytest.approx(0.0, abs=1e-9)
    electrode = wall_parameters(atoms, neutralise=False,
                                forcefield="electrode")
    assert electrode["net_charge"] == pytest.approx(-1.06, abs=1e-6)
    with pytest.raises(ValueError):
        wall_parameters(atoms, forcefield="nope")


def test_lattice_nitrogen_keeps_the_electrode_charges():
    from carbonforge.dopants.substitutional import dope_random
    atoms = dope_random(build_graphene_supercell(nx=6, ny=4, vacuum=10.0),
                        "N", 0.05, seed=1)
    a = wall_parameters(atoms, neutralise=False)
    e = wall_parameters(atoms, neutralise=False, forcefield="electrode")
    assert np.allclose(a["charge"], e["charge"])
