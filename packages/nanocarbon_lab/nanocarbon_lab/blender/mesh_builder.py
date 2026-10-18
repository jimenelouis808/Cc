"""Turn a render bundle (XYZ + JSON sidecar) into Blender mesh objects.

Imported **from inside Blender's Python** (``bpy``/``bmesh`` exist only
there) -- run it via ``render_cnt.py`` with
``blender -b -P nanocarbon_lab/blender/render_cnt.py -- ...`` rather than
importing it in an ordinary interpreter. :mod:`styles` and
:mod:`bundle` are the pure halves, testable without Blender -- the
palettes and the per-atom colour decision live there, and everything
here needs Blender to turn that decision into geometry.

Two complementary representations:

* :func:`build_ball_and_stick` -- one merged sphere mesh and one merged
  cylinder mesh, each face carrying a ``material_index`` so a single
  Blender material list colours the whole structure. Good for close-ups
  where the lattice is the subject.
* :func:`build_smooth_surface` -- one vertex per atom with a Skin +
  Subdivision modifier stack, giving a continuous glossy tube with no
  visible atoms. Good for wide shots where the silhouette carries the
  image.

**What the colour means is a parameter, not a property of the pipeline.**
It used to be ring size and nothing else, which is the right thing to say
about a defected cage and the wrong thing about almost everything else:
on a closed fullerene every atom sits on a pentagon, so the picture said
"all pentagons" about a structure that is twelve pentagons and twenty
hexagons; and on a doped shell, a functionalised surface or a composition
of three structures it said nothing at all. :data:`styles.COLOUR_MODES`
lists what it can mean now, and two consequences fall out of the choice:

* **Bonds are split in half when the colour belongs to the atom.** A C-O
  bond drawn in carbon's grey says the oxygen is carbon. Colouring by
  element or by structure therefore draws two half-cylinders per bond,
  each taking its own end's colour, which is what every molecular viewer
  does; colouring by ring keeps one cylinder, a ring being a property of
  neither end individually. :data:`styles.SPLIT_BONDS` holds the rule.
* **Atom radii follow the element** when the colour does. Equal spheres
  are right for an all-carbon lattice and wrong the moment there is a
  hydrogen and a gold atom in the same picture.
"""

from __future__ import annotations

import bmesh
import bpy
import styles as style_defs
from bundle import atom_radii, colour_keys, load_bundle
from mathutils import Matrix, Vector

#: Re-exported so ``render_cnt`` keeps one import for the whole pipeline.
__all__ = ["atom_radii", "build_ball_and_stick", "build_smooth_surface",
           "colour_keys", "load_bundle"]


def build_ball_and_stick(
    bundle: dict,
    style,
    colour_by: str = "ring",
    atom_scale: float = 1.0,
    bond_radius: float = 0.12,
    sphere_subdiv: int = 2,
    bond_segments: int = 8,
    finish: str = "style",
    bond_finish: str | None = None,
    bond_colour: str = "auto",
    model: str = "ballstick",
) -> list[bpy.types.Object]:
    """Merged sphere + cylinder meshes, coloured by ``colour_by``.

    Returns ``[atoms_object, bonds_object]``.
    """
    from render_cnt import material_from_spec  # local: avoid a load cycle

    positions = bundle["positions"]
    bonds = bundle["bonds"]
    keys, slot_per_atom = colour_keys(bundle, colour_by)
    specs = style_defs.materials_for(style, colour_by, keys, finish)
    materials = [material_from_spec(f"mat_{colour_by}_{key}", spec)
                 for key, spec in zip(keys, specs, strict=True)]
    radii = atom_radii(bundle, colour_by, atom_scale, model)
    symbols = bundle["symbols"]

    # --- atoms -------------------------------------------------------
    bm = bmesh.new()
    for index, position in enumerate(positions):
        if radii[index] <= 0.0:
            # licorice and wireframe draw no spheres at all.
            continue
        before = len(bm.faces)
        bmesh.ops.create_icosphere(
            bm, subdivisions=sphere_subdiv, radius=radii[index],
            matrix=Matrix.Translation(position), calc_uvs=False,
        )
        bm.faces.ensure_lookup_table()
        slot = slot_per_atom[index]
        for face in list(bm.faces)[before:]:
            face.material_index = slot
    atoms_mesh = bpy.data.meshes.new("AtomsMesh")
    bm.to_mesh(atoms_mesh)
    bm.free()
    for material in materials:
        atoms_mesh.materials.append(material)
    atoms_obj = bpy.data.objects.new("Atoms", atoms_mesh)
    bpy.context.collection.objects.link(atoms_obj)

    # --- bonds -------------------------------------------------------
    # Decided separately from the atoms: see styles.bond_materials.
    bond_specs, split = style_defs.bond_materials(
        style, colour_by, keys, bond_colour,
        finish if bond_finish is None else bond_finish)
    bm2 = bmesh.new()
    draw_any = style_defs.draws_bonds(model)
    for i, j in bonds:
        if not draw_any:
            break
        if i >= len(positions) or j >= len(positions):
            # A sidecar whose bonds outrun its atoms. Skip rather than
            # raise: an index error here aborts a render that is minutes
            # in, and the structure is still worth seeing.
            continue
        if not style_defs.bond_is_drawn(model, symbols[i], symbols[j]):
            # Under `auto`, a bond with a space-filling atom at either
            # end would start inside a sphere of its own radius and come
            # out the far side -- and between two metals it is an
            # invention anyway.
            continue
        a, b = Vector(positions[i]), Vector(positions[j])
        if (b - a).length < 1e-6:
            continue
        if split:
            mid = (a + b) / 2.0
            for start, end, slot in ((a, mid, slot_per_atom[i]),
                                     (mid, b, slot_per_atom[j])):
                _cylinder(bm2, start, end, bond_radius, bond_segments, slot)
        else:
            _cylinder(bm2, a, b, bond_radius, bond_segments, 0)

    # --- seam stubs --------------------------------------------------
    # Half-bonds where a periodic item's cell face cut the structure.
    # The neighbour is genuinely not in the scene, so there is no second
    # atom and no second colour: the stub takes its own atom's slot,
    # which is also what makes a split-bond render look continuous
    # across the seam rather than changing colour at it.
    for stub in bundle.get("seam_stubs", []):
        index = int(stub[0])
        if index >= len(positions) or not draw_any:
            continue
        if not style_defs.bond_is_drawn(model, symbols[index],
                                        symbols[index]):
            continue
        start = Vector(positions[index])
        end = start + Vector(tuple(stub[1:4]))
        if (end - start).length < 1e-6:
            continue
        _cylinder(bm2, start, end, bond_radius, bond_segments,
                  slot_per_atom[index] if split else 0)

    bonds_mesh = bpy.data.meshes.new("BondsMesh")
    bm2.to_mesh(bonds_mesh)
    bm2.free()
    if split:
        # One material per atom slot, so each half takes the colour of
        # the atom it belongs to. Built fresh rather than reusing the
        # atoms' materials: the bonds may carry a different finish, and
        # sharing the datablock would silently apply it to the atoms too.
        for key, spec in zip(keys, bond_specs, strict=True):
            bonds_mesh.materials.append(
                material_from_spec(f"mat_bond_{key}", spec))
    else:
        bonds_mesh.materials.append(
            material_from_spec("mat_bond", bond_specs[0]))
    bonds_obj = bpy.data.objects.new("Bonds", bonds_mesh)
    bpy.context.collection.objects.link(bonds_obj)

    for obj in (atoms_obj, bonds_obj):
        obj.data.polygons.foreach_set(
            "use_smooth", [True] * len(obj.data.polygons))
    return [atoms_obj, bonds_obj]


def _cylinder(bm, start: Vector, end: Vector, radius: float, segments: int,
              slot: int) -> None:
    """One capped cylinder from ``start`` to ``end``, in material ``slot``."""
    before = len(bm.faces)
    direction = end - start
    length = direction.length
    rot = direction.to_track_quat("Z", "Y").to_matrix().to_4x4()
    matrix = Matrix.Translation((start + end) / 2.0) @ rot
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=segments,
        radius1=radius, radius2=radius, depth=length, matrix=matrix,
        calc_uvs=False,
    )
    bm.faces.ensure_lookup_table()
    for face in list(bm.faces)[before:]:
        face.material_index = slot


def build_smooth_surface(
    bundle: dict,
    style,
    skin_radius: float = 1.6,
    subsurf_levels: int = 2,
    finish: str = "style",
) -> bpy.types.Object:
    """One continuous glossy tube (Skin + Subdivision modifiers).

    Uses the bond graph itself as the modifier's control cage: one vertex
    per atom, one edge per bond, a uniform skin radius. Individual atoms
    and rings disappear entirely, which is what a wide shot wants -- and
    is also why this representation takes **no colour mode**: there are
    no atoms left to colour by element and no rings left to colour by
    size, so it uses the style's bulk material and says so.
    """
    from render_cnt import material_from_spec

    mesh = bpy.data.meshes.new("SurfaceMesh")
    mesh.from_pydata(bundle["positions"], bundle["bonds"], [])
    mesh.update()
    obj = bpy.data.objects.new("Surface", mesh)
    bpy.context.collection.objects.link(obj)

    skin = obj.modifiers.new("Skin", type="SKIN")
    skin.use_smooth_shade = True
    for vertex in mesh.skin_vertices[0].data:
        vertex.radius = (skin_radius, skin_radius)
    subsurf = obj.modifiers.new("Subsurf", type="SUBSURF")
    subsurf.levels = subsurf_levels
    subsurf.render_levels = subsurf_levels

    obj.data.materials.append(material_from_spec(
        "mat_surface", style_defs.apply_finish(style.hexagon, finish)))
    return obj
