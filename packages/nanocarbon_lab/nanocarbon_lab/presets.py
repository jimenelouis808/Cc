"""The catalogue of one-click structures, and how a knee mode reads it.

This is deliberately **not** in :mod:`nanocarbon_lab.gui.app`, where it
lived until a bug that no test could reach: the application read the
Blender representation box (``"ballstick"``) where it meant the
structure mode, so every knee kind kept whatever shape the boxes
happened to hold. A Y node wants circumference 14 and a tetrahedral one
18, so the diamond junction refused outright and the gyroid quietly
built a cell three times the size of the one its own entry documents.

The lesson is the reason for this module. A preset is **data**, and the
map from a preset to a builder's arguments is a **pure function**; only
the widgets need a display. Keeping them here lets the tests build every
preset and check its census against what its entry claims, which is the
check that was missing.

Keys are the parameter names the application registers with ``_var``, so
applying a preset stays a plain loop and the same format serves the
save/load file.
"""

from __future__ import annotations

from typing import Any

from .builders.knee import SOUND_PERIODIC_COIL, default_knee_shape

#: The knee modes whose arguments come from a node table.
KNEE_NODE_MODES = ("schwarzite (knees)", "junction (knees)",
                   "supernetwork (knees)")
#: The knee modes that size themselves from their own arguments.
KNEE_SHAPE_MODES = ("toroid (knees)", "coil (knees)",
                    "coil (knees, periodic)")
KNEE_MODES = KNEE_NODE_MODES + KNEE_SHAPE_MODES

_SIDES, _COIL_K, _PITCH, _RADIUS = SOUND_PERIODIC_COIL

#: Starting values for the boxes the knee modes own. The application
#: initialises its variables from this, so a default stated here is the
#: default the user sees.
KNEE_DEFAULTS: dict[str, float] = {
    "knee_k": 20.0,
    "knee_rows": 9.0,
    # Six knees is the curvature-exact count for a ring: a knee turns the
    # axis 60 degrees and a torus asks for 360.
    "kt_knees": 6.0,
    # Zero means "choose": the builder asks `clean_shapes` for the
    # smallest pair whose census is exact, which is not a number anyone
    # could have typed.
    "kt_k": 0.0,
    "kt_rows": 0.0,
    "kc_radius": float(_RADIUS),
    "kc_pitch": float(_PITCH),
    "kc_sides": float(_SIDES),
    "kc_k": float(_COIL_K),
    "kc_turns": 2.0,
}


def knee_params(mode: str, values: dict[str, Any]) -> dict[str, Any]:
    """Builder arguments for one knee mode, from the parameter values.

    `values` is the flat ``{name: number}`` mapping the application keeps
    and a preset writes into -- not widgets, so this runs without a
    display and the tests call exactly what the application calls.

    A node mode takes its shape from the kind's own table unless the
    values carry one, because the kinds do not share a shape: a Y node
    closes at circumference 14 and a tetrahedral one does not close there
    at all.
    """
    if mode not in KNEE_MODES:
        raise KeyError(f"{mode!r} is not a knee mode.")

    def number(name: str, cast=int):
        return cast(values.get(name, KNEE_DEFAULTS[name]))

    if mode in KNEE_NODE_MODES:
        if mode == "schwarzite (knees)":
            field, kind = "kind", values.get("knee_cell", "primitive")
        elif mode == "junction (knees)":
            field, kind = "kind", values.get("knee_node", "y")
        else:
            field, kind = "net", values.get("knee_net", "super-graphene")
        shape = default_knee_shape(mode, kind)
        circumference = int(values.get("knee_k", (shape or (20, 9))[0]))
        rows = int(values.get("knee_rows", (shape or (20, 9))[1]))
        return {field: kind, "circumference": circumference,
                "arm_rows": rows}

    if mode == "toroid (knees)":
        return {
            "knees": number("kt_knees"),
            "circumference": number("kt_k") or None,
            "arm_rows": number("kt_rows") or None,
        }

    params = {
        "coil_radius": number("kc_radius", float),
        "pitch": number("kc_pitch", float),
        "sides_per_turn": number("kc_sides"),
        "circumference": number("kc_k"),
    }
    if mode == "coil (knees)":
        # The periodic cell IS one turn, closed on the z-torus, so it
        # takes no turn count.
        params["turns"] = number("kc_turns")
    return params


# Structures worth having one click away. Keys are parameter names as
# registered with `_var`, so applying a preset is a plain loop and the
# same format serves the save/load file.
PRESETS: dict[str, dict[str, object]] = {
    # --- carbon cages
    "C60 buckyball": {
        "mode_kind": "fullerene", "cage_family": "C60", "cage_freq": 1},
    "C540 giant cage": {
        "mode_kind": "fullerene", "cage_family": "C60", "cage_freq": 3},
    "Nano-onion C60@C240@C540": {
        "mode_kind": "nano-onion", "cage_family": "C60", "cage_freq": 1,
        "onion_shells": 3},
    # --- carbon tubes
    "Capped nanotube": {
        "mode_kind": "capped tube", "rings": 10, "freq": 3, "shape": "straight",
        "roughness": 0.0, "n_sw": 0, "n_dv": 0},
    "N-doped nanotube": {
        "mode_kind": "capped tube", "rings": 10, "freq": 3,
        "dopant": "N", "dopant_conc": 0.03},
    "Double-wall nanotube": {
        "mode_kind": "multi-wall", "mw_shells": 2, "mw_inner": 3, "rings": 10},
    "Seven-tube rope": {
        "mode_kind": "bundle", "bundle_shells": 1, "freq": 3, "rings": 10},
    # The crystalline ring: no disclinations at all. 110 periods of a
    # (5,5) is the smallest that fits the 8% strain budget.
    "Carbon toroid (all hexagons)": {
        "mode_kind": "toroid (polyhex)", "tp_n": 5, "tp_m": 5,
        "tp_periods": 110},
    # The third ring, and the exact one: six knees of two pentagon-heptagon
    # pairs each. A pair turns the axis 30 deg and a torus asks for exactly
    # twelve, so six knees is the whole budget. 1032 atoms, {5: 12,
    # 6: 492, 7: 12}, every disclination alone in hexagons. The knee
    # count is editable and the two shape boxes take 0 for "choose",
    # which asks the builder for the smallest pair whose census is
    # exact -- a question no typed number can ask.
    "Carbon toroid (12 knees, exact)": {
        "mode_kind": "toroid (knees)", "kt_knees": 6, "kt_k": 0, "kt_rows": 0,
        "anneal": 0},
    # The same knee wound on a helix. `nanocoil` winds a finished lattice
    # and refuses a 25 A coil outright; this one is 12 A, D/d 3.73, inside
    # the band the single-wall coil papers report.
    "Nanocoil (knees, D/d 3.7)": {
        "mode_kind": "coil (knees)", "kc_radius": 12.0, "kc_pitch": 12.0,
        "kc_sides": 8, "kc_k": 8, "kc_turns": 2, "anneal": 0},
    # 968 atoms, {6: 456, 7: 24}, sum(6-n) = -24 and not one pentagon --
    # which is what a minimal surface must look like. The meshed route
    # returns 33 pentagons at a comparable cell.
    # One turn of the same coil welded through the cell: periodic along
    # the axis, no rims, D/d 3.52 inside the published single-wall band.
    # 672 atoms, {5: 12, 6: 312, 7: 12} -- a periodic coil cell is a
    # TORUS, so sum(6-n) = 0 and the two come out equal.
    "Nanocoil (knees, periodic, DFT-ready)": {
        "mode_kind": "coil (knees, periodic)", "kc_radius": 14.0,
        "kc_pitch": 15.0, "kc_sides": 6, "kc_k": 10, "anneal": 0},
    "Schwarz P (knees, no pentagons)": {
        "mode_kind": "schwarzite (knees)", "knee_cell": "primitive",
        "knee_k": 20, "knee_rows": 9, "anneal": 0},
    # Eight of those tetrahedral nodes on the diamond lattice: 1440
    # atoms, {6: 608, 7: 96}, sum(6-n) = -96 = 6*chi at genus 9, and not
    # one pentagon. The D surface is what the schwarzite figures call
    # D216; this is its conventional cubic cell.
    "Schwarz D (knees, no pentagons)": {
        "mode_kind": "schwarzite (knees)", "knee_cell": "diamond",
        "knee_k": 18, "knee_rows": 5, "anneal": 0},
    # The third minimal surface, and its node is the SAME planar Y as the
    # junction preset below: srs is 3-coordinate, and three unit vectors
    # at 120 deg have no choice but to be coplanar. 1744 atoms,
    # {6: 816, 7: 48}, sum(6-n) = -48 at genus 5. It wants a fatter tube
    # than P or D -- its arms leave at 120 deg rather than 109.47, so the
    # saddle is tighter and every narrower cell relaxes to a broken wall.
    "Gyroid (knees, no pentagons)": {
        "mode_kind": "schwarzite (knees)", "knee_cell": "gyroid",
        "knee_k": 20, "knee_rows": 5, "anneal": 0},
    # The same Y node repeated on a honeycomb instead of standing
    # alone: 920 atoms, {6: 432, 7: 24}, sum(6-n) = -24 = 12(V-E), and
    # bonds 1.408-1.436 A -- the tightest of anything this route builds.
    "Super-graphene (knees, tubes on a honeycomb)": {
        "mode_kind": "supernetwork (knees)", "knee_net": "super-graphene",
        "knee_k": 22, "knee_rows": 5, "anneal": 0},
    # A sheet of planar crossings. 180 atoms, {5: 4, 6: 68, 7: 16} --
    # the pentagons are the poles of each crossing and belong there.
    "Super-square (knees, tubes on a square net)": {
        "mode_kind": "supernetwork (knees)", "knee_net": "super-square",
        "knee_k": 12, "knee_rows": 5, "anneal": 0},
    # 536 atoms, {6: 240, 7: 6} and not one pentagon -- exactly the six
    # heptagons Gauss-Bonnet asks of a three-arm node, where the meshed
    # route returns fifty pentagons and thirty-eight heptagons. Bonds
    # 1.410-1.431 A, the tightest of any junction here.
    "Y junction (knees, six heptagons)": {
        "mode_kind": "junction (knees)", "knee_node": "y", "knee_k": 14,
        "knee_rows": 9, "anneal": 0},
    # The one node here whose PENTAGONS are right: a four-way planar
    # crossing has a pillow above and below the crossing point that is
    # genuinely positively curved. 780 atoms, {5: 4, 6: 328, 7: 16},
    # sum(6-n) = -12, and all twenty disclinations on the correct side.
    "X junction (knees, four poles + sixteen crotches)": {
        "mode_kind": "junction (knees)", "knee_node": "x", "knee_k": 20,
        "knee_rows": 9, "anneal": 0},
    # The Schwarz D node standing alone rather than tiled: four arms at
    # 109.47 deg, twelve heptagons, no pentagon.
    "Diamond junction (knees, twelve heptagons)": {
        "mode_kind": "junction (knees)", "knee_node": "tetrahedral",
        "knee_k": 18, "knee_rows": 9, "anneal": 0},
    "Carbon toroid (R/r = 4)": {
        "mode_kind": "toroid", "tor_major": 20.0, "tor_minor": 5.0,
        "anneal": 0},
    # The only disclination the sector cut places cleanly: one apex
    # pentagon, 210 hexagons, bonds 1.391–1.420 Å.
    "Nanocone (112.9°, one pentagon)": {
        "mode_kind": "nanocone", "nc_pent": 1, "nc_radius": 22.0,
        "nc_strict": True},
    # --- 2D carbon allotropes
    # Pentagons and heptagons only, no hexagons: the lattice published as
    # pentaheptite. The catalogue entry is an exact cover, so its census
    # is exact rather than whatever the rules let through.
    "Haeckelite R5,7 sheet": {
        "mode_kind": "haeckelite", "hk_pattern": "r57", "hk_nx": 4,
        "hk_ny": 4},
    "Haeckelite R5,7 tube": {
        "mode_kind": "haeckelite tube", "ht_pattern": "r57", "ht_nx": 12,
        "ht_ny": 4, "ht_roll": "a"},
    # A trivalent net of nothing but heptagons: the one entry here whose
    # answer is a proof rather than a structure. `hp_strict` is False on
    # purpose. Applying a preset BUILDS immediately, and under strict the
    # builder refuses by design -- so selecting this preset fired its
    # refusal as an error dialog before the window had drawn anything,
    # which is the worst possible way to deliver a result. Unticked, it
    # returns the strained lattice to look at, and the panel's own text
    # plus `sp2 verdict` carry the finding. Tick the box to see the
    # refusal in full.
    "Heptanene (Klein quartic, genus 3)": {
        "mode_kind": "heptanene", "hp_strict": False},
    # --- coils. Two routes, and they are not interchangeable, so the
    # names say which. One preset each; the old menu carried three
    # meshed coils differing only in radius and turn count, which is a
    # parameter sweep rather than three structures.
    #
    # The **rolled lattice** winds a real (n, m) nanotube: every ring is
    # a hexagon and the wall is graphitic. Bending a finished lattice can
    # only stretch it, so the coil has to be wide -- which is why real
    # carbon nanocoils are tens to hundreds of Å across.
    "Nanocoil (graphitic, rolled lattice)": {
        "mode_kind": "nanocoil", "cnt_shape": "helix", "cnt_n": 5, "cnt_m": 5,
        "coil_radius": 45.0, "coil_pitch": 12.0, "coil_turns": 2.0,
        "n_sw": 0, "n_dv": 0, "roughness": 0.0},
    # The **meshed wall** route fits a surface to the helix and tiles it,
    # so it reaches any radius -- but what it tiles with is an amorphous
    # CVD-like network, not a rolled lattice. `anneal` is 0 and must
    # stay 0: on a curved surface the 5-7 pairs ARE how the net covers
    # its curvature, and annealing them away leaves the survivors to
    # carry all of it. Measured on this Y junction, 80 sweeps widen the
    # bond spread from 0.0136 to 0.0175 Å.
    "Nanocoil (meshed wall)": {
        "mode_kind": "coil (relaxed)", "coil_radius": 18.0, "coil_pitch": 13.0,
        "coil_turns": 2.0, "coil_tube_radius": 4.5, "anneal": 0,
        "roughness": 0.0, "n_sw": 0, "n_dv": 0},
    # The one that goes into a plane-wave code: one turn closed on the
    # z-torus, so it is a cell and not a fragment with two dangling ends.
    # Six sides, because seen down its axis a real single-wall coil is a
    # POLYGON -- Liu et al. show the (6,6) as a hexagonal torus -- and at
    # D/d = 3.92 the hexagon puts 85% of its disclinations on the correct
    # side against the smooth helix's 68%.
    "Nanocoil (periodic cell, DFT)": {
        "mode_kind": "coil (periodic, DFT)", "coil_radius": 8.75,
        "coil_pitch": 9.6, "coil_tube_radius": 3.0, "coil_sides": 6,
        "roughness": 0.0, "n_sw": 0, "n_dv": 0},
    # --- junctions and periodic 3D carbon
    "Y junction": {
        "mode_kind": "junction", "j_kind": "Y", "j_radius": 6.0,
        "j_arm": 22.0, "j_blend": 4.0, "anneal": 0},
    "Gyroid schwarzite": {
        "mode_kind": "schwarzite", "s_kind": "gyroid", "s_cell": 36.0,
        "anneal": 0},
    "Schwarz P schwarzite": {
        "mode_kind": "schwarzite", "s_kind": "primitive", "s_cell": 36.0,
        "anneal": 0},
    "Nanotube network (cubic)": {
        "mode_kind": "network", "net_kind": "cubic", "net_cell": 40.0,
        "net_radius": 6.0, "net_blend": 5.0, "anneal": 0},
    # --- superlattices of nanotubes
    "Super-graphene (tubes at 120°)": {
        "mode_kind": "supernetwork", "sn_graph": "super-graphene",
        "sn_scale": 34.0, "sn_radius": 5.0, "sn_blend": 4.0, "anneal": 0},
    "Super-diamond (tubes at 109.47°)": {
        "mode_kind": "supernetwork", "sn_graph": "super-diamond",
        "sn_scale": 60.0, "sn_radius": 5.0, "sn_blend": 4.0, "anneal": 0},
    # Each of the 4-cube's 32 edges a nanotube. The struts are
    # deliberately unequal -- that is what a 4D object looks like in 3D.
    "Hypercube of tubes (4-cube)": {
        "mode_kind": "supernetwork", "sn_graph": "super-hypercube",
        "sn_scale": 20.0, "sn_radius": 3.5, "sn_blend": 2.5, "anneal": 0},
    # Super-graphene rolled: a nanotube whose every bond is a nanotube.
    "Supertube (6,6) of super-graphene": {
        "mode_kind": "supernetwork", "sn_graph": "supertube-(6,6)",
        "sn_scale": 14.0, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    # The five Platonic cages. `sn_scale` is the STRUT LENGTH here, not a
    # cell edge, and each vertex eats about tube_radius + blend of either
    # end of every edge it touches -- so 24 Å leaves about 10 Å of real
    # tube at these radii. Measured, all five meet 12(V-E) exactly with no
    # close contacts: -24, -48, -72, -120, -216.
    #
    # Their walls are meshed and therefore amorphous, and that is not a
    # choice: a cage of exact KNEE nodes cannot exist, because a node's
    # census comes out clean only when its arms sum to zero and a convex
    # polyhedron's vertex lies on its own hull.
    "Tetrahedral cage of tubes": {
        "mode_kind": "supernetwork", "sn_graph": "super-tetrahedron",
        "sn_scale": 24.0, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    "Cubic cage of tubes": {
        "mode_kind": "supernetwork", "sn_graph": "super-cube",
        "sn_scale": 24.0, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    "Octahedral cage of tubes": {
        "mode_kind": "supernetwork", "sn_graph": "super-octahedron",
        "sn_scale": 24.0, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    "Dodecahedral cage of tubes": {
        "mode_kind": "supernetwork", "sn_graph": "super-dodecahedron",
        "sn_scale": 24.0, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    "Icosahedral cage of tubes": {
        "mode_kind": "supernetwork", "sn_graph": "super-icosahedron",
        "sn_scale": 24.0, "sn_radius": 4.0, "sn_blend": 3.0, "anneal": 0},
    "Superfullerene (C60 of tubes)": {
        "mode_kind": "supernetwork", "sn_graph": "superfullerene-C60",
        "sn_scale": 14.2, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    # The three periodic nets that had no preset. Without one the Net
    # box is the only way to reach them, and picking a net there leaves
    # the cell, radius and blend at whatever the LAST preset set -- which
    # is how super-fcc got built at super-diamond's scale=60, radius=5
    # and ran for ninety minutes. Every number below is measured: fcc is
    # sound at 334.0 deg in 413 s at 40/3.0/2.0, and at 60/5/4 it asks
    # for 32_000 A^2 of wall instead of 12_800.
    "Super-square (tubes at 90°)": {
        "mode_kind": "supernetwork", "sn_graph": "super-square",
        "sn_scale": 34.0, "sn_radius": 5.0, "sn_blend": 4.0, "anneal": 0},
    "Super-cubic (tubes along the axes)": {
        "mode_kind": "supernetwork", "sn_graph": "super-cubic",
        "sn_scale": 34.0, "sn_radius": 5.0, "sn_blend": 4.0, "anneal": 0},
    "Super-fcc (twelve tubes per node)": {
        "mode_kind": "supernetwork", "sn_graph": "super-fcc",
        "sn_scale": 40.0, "sn_radius": 3.0, "sn_blend": 2.0, "anneal": 0},
    # --- dichalcogenides
    "MoS2 monolayer (2H)": {
        "mode_kind": "TMD layers", "tmd_material": "MoS2", "tmd_phase": "2H",
        "tmd_layers": 1, "tmd_nx": 1, "tmd_ny": 1},
    "MoS2 bilayer (2H)": {
        "mode_kind": "TMD layers", "tmd_material": "MoS2", "tmd_phase": "2H",
        "tmd_layers": 2, "tmd_stacking": "2H"},
    "MoS2 monolayer (1T)": {
        "mode_kind": "TMD layers", "tmd_material": "MoS2", "tmd_phase": "1T",
        "tmd_layers": 1},
    "MoS2 bulk crystal": {
        "mode_kind": "TMD bulk", "tmd_material": "MoS2", "tmd_stacking": "2H"},
    "MoS2 zigzag ribbon": {
        "mode_kind": "TMD ribbon", "tmd_material": "MoS2", "tmd_width": 8,
        "tmd_length": 2, "tmd_edge": "zigzag", "tmd_termination": "mixed"},
    "MoS2 nanotube (40,0)": {
        "mode_kind": "TMD nanotube", "tmd_material": "MoS2", "tmd_n": 40,
        "tmd_m": 0},
    "WSe2 monolayer": {
        "mode_kind": "TMD layers", "tmd_material": "WSe2", "tmd_phase": "2H",
        "tmd_layers": 1},
    "MoS2 Y junction": {
        "mode_kind": "TMD junction", "tmd_material": "MoS2",
        "tmd_j_kind": "Y", "tmd_j_radius": 12.0, "tmd_j_arm": 26.0,
        "tmd_j_parity": "split"},
    "MoS2 schwarzite (Schwarz P)": {
        "mode_kind": "TMD schwarzite", "tmd_material": "MoS2",
        "tmd_sw_kind": "primitive", "tmd_sw_cell": 36.0,
        "tmd_sw_parity": "flip"},
    "MoS2 coil (quarter turn)": {
        "mode_kind": "TMD coil", "tmd_material": "MoS2", "tmd_n": 30,
        "tmd_m": 0, "tmd_coil_radius": 220.0, "tmd_coil_pitch": 90.0,
        "tmd_coil_turns": 0.25, "tmd_coil_hand": "right"},
    # --- heterostructures
    # A quarter turn keeps the MX2 coil near 11k atoms. A full turn at a
    # radius loose enough to be unstrained runs to six figures, which is
    # the physics rather than a timid default.
    "Twisted bilayer graphene 21.8°": {
        "mode_kind": "twisted bilayer", "het_bottom": "graphene",
        "het_top": "same", "het_angle": 21.79, "het_max_index": 40},
    "Magic-angle bilayer 1.08°": {
        "mode_kind": "twisted bilayer", "het_bottom": "graphene",
        "het_top": "same", "het_angle": 1.08, "het_max_index": 40},
    "Graphene on hBN": {
        "mode_kind": "twisted bilayer", "het_bottom": "graphene",
        "het_top": "hBN", "het_angle": 7.34, "het_max_index": 40},
    "MoS2/WS2 stack": {
        "mode_kind": "vdW stack", "het_bottom": "MoS2", "het_top": "WS2",
        "het_third": "none", "het_nx": 2, "het_ny": 2},
}
