"""Functional groups and nitrogen configurations on carbon surfaces.

Two different chemistries live here, and keeping them apart matters:

* **Functional groups** (:mod:`~carbonforge.functionalization.groups`,
  :mod:`~carbonforge.functionalization.attach`) are *attached* to a carbon,
  at an edge or on the basal plane. Amine, nitro, hydroxyl, carboxyl,
  epoxide and the rest.
* **Nitrogen configurations** (:mod:`~carbonforge.functionalization.nitrogen`)
  are *lattice* modifications: graphitic N substitutes a carbon, pyridinic N
  needs a vacancy first. These are what N 1s XPS separates, and they are not
  interchangeable with an attached amine.

Both produce idealised geometries. Relax before drawing conclusions.
"""

from .attach import (
    attach_bridging_group,
    attach_group,
    coverage,
    functionalize,
    functionalize_bridges,
    functionalize_random,
    passivate_edges,
    repad_vacuum,
)
from .groups import (
    BRIDGING_GROUPS,
    EDGE_ONLY_GROUPS,
    GROUPS,
    NITROGEN_GROUPS,
    FunctionalGroup,
    describe_groups,
    get_group,
)
from .nitrogen import (
    XPS_BINDING_ENERGY_EV,
    make_graphitic_n,
    make_pyridinic_n,
    make_pyridinic_n_oxide,
    make_pyrrolic_like,
    nitrogen_report,
)
from .sites import AttachmentSite, find_bridge_sites, find_sites

__all__ = [
    # groups
    "GROUPS",
    "NITROGEN_GROUPS",
    "EDGE_ONLY_GROUPS",
    "BRIDGING_GROUPS",
    "FunctionalGroup",
    "get_group",
    "describe_groups",
    # sites
    "AttachmentSite",
    "find_sites",
    "find_bridge_sites",
    # attaching
    "attach_group",
    "attach_bridging_group",
    "functionalize",
    "functionalize_random",
    "functionalize_bridges",
    "passivate_edges",
    "repad_vacuum",
    "coverage",
    # nitrogen
    "make_graphitic_n",
    "make_pyridinic_n",
    "make_pyrrolic_like",
    "make_pyridinic_n_oxide",
    "nitrogen_report",
    "XPS_BINDING_ENERGY_EV",
]
