"""Turn a nanocarbon_lab render bundle (XYZ + rings/bonds JSON) into Blender
mesh objects, coloured by ring type.

This module is imported **from inside Blender's Python** (``bpy``/``bmesh``
are only available there) -- run it via ``render_cnt.py`` with
``blender -b -P nanocarbon_lab/blender/render_cnt.py -- ...`` rather than
importing it in a normal ``python`` interpreter.

Two complementary representations are built:

* :func:`build_ball_and_stick` -- one merged sphere mesh + one merged
  cylinder mesh, each face carrying a ``material_index`` set from the
  dominant ring type at that atom/bond (0=hexagon, 1=pentagon,
  2=heptagon, 3=octagon), so a 5-material Blender material list lets you
  colour curvature/defects directly. Good for close-up "you can see the
  lattice" shots.
* :func:`build_smooth_surface` -- a single-vertex-per-atom edge mesh with
  a Skin + Subdivision Surface modifier stack, giving a continuous glossy
  tube (no visible individual atoms). Good for wide macro cover shots
  where the tube itself is the hero object.

Both read the same bundle format written by
:func:`nanocarbon_lab.exports.xyz.write_render_bundle`.
"""

from __future__ import annotations

import json
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector

RING_MATERIAL_INDEX = {5: 1, 7: 2, 8: 3}  # everything else (6, or no ring info) -> 0


def load_bundle(xyz_path: str, json_path: str | None = None):
    """Read atom positions (and, if present, bonds/ring metadata).

    Parameters
    ----------
    xyz_path
        Path to a plain ``.xyz`` file (as written by
        :func:`nanocarbon_lab.exports.xyz.write_xyz`).
    json_path
        Path to the companion ``.json`` sidecar (as written by
        :func:`nanocarbon_lab.exports.xyz.write_render_bundle`). If
        ``None`` or the file does not exist, bonds are inferred from a
        simple distance cutoff and every atom is treated as a hexagon
        (no ring colouring).

    Returns
    -------
    (positions, bonds, ring_sizes_per_atom)
        ``positions`` -- list of ``(x, y, z)`` tuples (Å).
        ``bonds`` -- list of ``(i, j)`` index pairs.
        ``ring_sizes_per_atom`` -- list (len == len(positions)) of lists
        of ring sizes each atom belongs to (empty list if unknown).
    """
    lines = Path(xyz_path).read_text().splitlines()
    n = int(lines[0])
    positions = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        positions.append(tuple(float(x) for x in parts[1:4]))

    bonds: list[tuple[int, int]] = []
    ring_sizes_per_atom: list[list[int]] = [[] for _ in range(n)]
    if json_path and Path(json_path).exists():
        bundle = json.loads(Path(json_path).read_text())
        bonds = [tuple(b) for b in bundle.get("bonds", [])]
        loaded = bundle.get("ring_sizes_per_atom")
        if loaded and len(loaded) == n:
            ring_sizes_per_atom = loaded
    if not bonds:
        bonds = _bonds_from_distance(positions)
    return positions, bonds, ring_sizes_per_atom


def _bonds_from_distance(
    positions: list[tuple[float, float, float]], cutoff: float = 1.8
) -> list[tuple[int, int]]:
    """Fallback bond guesser for a bare ``.xyz`` with no sidecar JSON."""
    from scipy.spatial import cKDTree

    pts = [Vector(p) for p in positions]
    tree = cKDTree([tuple(p) for p in pts])
    pairs = tree.query_pairs(r=cutoff)
    return sorted(pairs)


def _dominant_material_index(ring_sizes: list[int]) -> int:
    """Pick which of the 4 preset materials an atom/bond should use.

    An atom sits on 3 rings; if any of them is non-hexagonal, colour by
    the *smallest* non-hexagonal ring present (pentagons are the most
    visually informative -- they mark caps/convex points).
    """
    for size in (5, 7, 8):
        if size in ring_sizes:
            return RING_MATERIAL_INDEX[size]
    return 0


def _make_materials(style) -> list[bpy.types.Material]:
    """Build the shared 4-slot [hexagon, pentagon, heptagon, octagon] list."""
    from render_cnt import material_from_spec  # local import: avoid cycle at module load

    return [
        material_from_spec("mat_hexagon", style.hexagon),
        material_from_spec("mat_pentagon", style.pentagon),
        material_from_spec("mat_heptagon", style.heptagon),
        material_from_spec("mat_octagon", style.octagon),
    ]


def build_ball_and_stick(
    positions,
    bonds,
    ring_sizes_per_atom,
    style,
    sphere_radius: float = 0.35,
    bond_radius: float = 0.12,
    sphere_subdiv: int = 2,
    bond_segments: int = 8,
) -> list[bpy.types.Object]:
    """Build merged sphere + cylinder meshes, coloured by ring type.

    Returns the created ``[atoms_object, bonds_object]``.
    """
    materials = _make_materials(style)
    from render_cnt import material_from_spec

    bond_material = material_from_spec("mat_bond", style.bond)

    # --- atoms: one bmesh, one icosphere per atom, per-face material index.
    bm = bmesh.new()
    for i, pos in enumerate(positions):
        before = len(bm.faces)
        bmesh.ops.create_icosphere(
            bm,
            subdivisions=sphere_subdiv,
            radius=sphere_radius,
            matrix=Matrix.Translation(pos),
            calc_uvs=False,
        )
        bm.faces.ensure_lookup_table()
        sizes = (ring_sizes_per_atom[i]
                 if i < len(ring_sizes_per_atom) else [])
        idx = _dominant_material_index(sizes)
        for f in list(bm.faces)[before:]:
            f.material_index = idx
    atoms_mesh = bpy.data.meshes.new("AtomsMesh")
    bm.to_mesh(atoms_mesh)
    bm.free()
    for mat in materials:
        atoms_mesh.materials.append(mat)
    atoms_obj = bpy.data.objects.new("Atoms", atoms_mesh)
    bpy.context.collection.objects.link(atoms_obj)

    # --- bonds: one bmesh, one cylinder per bond.
    bm2 = bmesh.new()
    for i, j in bonds:
        a, b = Vector(positions[i]), Vector(positions[j])
        mid = (a + b) / 2.0
        direction = b - a
        length = direction.length
        if length < 1e-6:
            continue
        rot = direction.to_track_quat("Z", "Y").to_matrix().to_4x4()
        matrix = Matrix.Translation(mid) @ rot
        bmesh.ops.create_cone(
            bm2,
            cap_ends=True,
            cap_tris=False,
            segments=bond_segments,
            radius1=bond_radius,
            radius2=bond_radius,
            depth=length,
            matrix=matrix,
            calc_uvs=False,
        )
    bonds_mesh = bpy.data.meshes.new("BondsMesh")
    bm2.to_mesh(bonds_mesh)
    bm2.free()
    bonds_mesh.materials.append(bond_material)
    bonds_obj = bpy.data.objects.new("Bonds", bonds_mesh)
    bpy.context.collection.objects.link(bonds_obj)

    for obj in (atoms_obj, bonds_obj):
        obj.data.polygons.foreach_set(
            "use_smooth", [True] * len(obj.data.polygons)
        )
    return [atoms_obj, bonds_obj]


def build_smooth_surface(
    positions,
    bonds,
    style,
    skin_radius: float = 1.6,
    subsurf_levels: int = 2,
) -> bpy.types.Object:
    """Build a single continuous glossy tube (Skin + Subdivision modifiers).

    Uses the honeycomb graph itself as the modifier's control cage: one
    vertex per atom, one edge per bond, uniform "skin" radius. This hides
    individual atoms/rings entirely -- best for wide macro shots where the
    tube's silhouette and material carry the image (the ``gold_nanotech``
    and ``nature_dark`` presets both read well this way).
    """
    from render_cnt import material_from_spec

    mesh = bpy.data.meshes.new("SurfaceMesh")
    mesh.from_pydata(positions, bonds, [])
    mesh.update()
    obj = bpy.data.objects.new("Surface", mesh)
    bpy.context.collection.objects.link(obj)

    skin = obj.modifiers.new("Skin", type="SKIN")
    skin.use_smooth_shade = True
    for v in mesh.skin_vertices[0].data:
        v.radius = (skin_radius, skin_radius)
    subsurf = obj.modifiers.new("Subsurf", type="SUBSURF")
    subsurf.levels = subsurf_levels
    subsurf.render_levels = subsurf_levels

    mat = material_from_spec("mat_surface", style.hexagon)
    obj.data.materials.append(mat)
    return obj
