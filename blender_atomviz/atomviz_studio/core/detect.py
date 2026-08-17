"""Discovery of Atomic Blender structures in the scene.

Atomic Blender (``io_mesh_atomic``, the bundled PDB/XYZ importer) builds a
structure like this::

    Empty  "molecule.xyz"                 <- structure root
      +- Mesh "Carbon_mesh"               <- one vertex per atom, instance_type='VERTS'
      |    +- Mesh "Carbon_ball"          <- the sphere that gets instanced (holds the material)
      +- Mesh "Nitrogen_mesh"
      |    +- Mesh "Nitrogen_ball"
      +- Mesh "Sticks"                    <- bond geometry

Names differ between Blender releases and between the PDB and XYZ importers,
and users happily rename things, so detection is heuristic and tolerant:
objects are classified by element name/symbol found anywhere in their object,
mesh or material name, and instancer parents are followed to the child object
that actually carries the material.

Structures built by hand (plain spheres and cylinders) are still picked up:
whatever cannot be resolved to an element is grouped under
:data:`~atomviz_studio.core.elements.UNKNOWN_SYMBOL` so the styling operators
keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import bpy

from .elements import UNKNOWN_SYMBOL, guess_element, is_stick_name, sort_symbols
from .mathutil import Vec3, bbox, bbox_center, bounding_radius


@dataclass
class AtomGroup:
    """All objects representing one element inside a structure."""

    symbol: str
    #: Objects that carry the visible material (ball / sphere objects).
    shaded: list[bpy.types.Object] = field(default_factory=list)
    #: Vertex-cloud instancers, one vertex per atom (may be empty).
    instancers: list[bpy.types.Object] = field(default_factory=list)

    @property
    def atom_count(self) -> int:
        """Number of atoms, counted from the instancer vertices when present."""
        total = 0
        for obj in self.instancers:
            mesh = getattr(obj, "data", None)
            if mesh is not None and hasattr(mesh, "vertices"):
                total += len(mesh.vertices)
        return total or len(self.shaded)


@dataclass
class Structure:
    """One detected atomic structure."""

    name: str
    root: bpy.types.Object | None
    groups: dict[str, AtomGroup] = field(default_factory=dict)
    sticks: list[bpy.types.Object] = field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        """Element symbols present, sorted by atomic number."""
        return sort_symbols(self.groups.keys())

    @property
    def objects(self) -> list[bpy.types.Object]:
        """Every object belonging to the structure, root included."""
        found: list[bpy.types.Object] = []
        for group in self.groups.values():
            found.extend(group.shaded)
            found.extend(group.instancers)
        found.extend(self.sticks)
        if self.root is not None:
            found.append(self.root)
        # Preserve order while removing duplicates.
        seen: set[str] = set()
        unique = []
        for obj in found:
            if obj.name not in seen:
                seen.add(obj.name)
                unique.append(obj)
        return unique

    @property
    def atom_count(self) -> int:
        """Total number of atoms across all element groups."""
        return sum(group.atom_count for group in self.groups.values())

    def atom_positions(self, limit: int = 20000) -> list[Vec3]:
        """World-space atom positions, sampled from the instancer meshes.

        Falls back to object origins when a structure has no vertex clouds
        (hand-built structures, or after "Make Instances Real").

        Args:
            limit: Hard cap on returned positions; large systems are strided
                so effect placement stays interactive.
        """
        points: list[Vec3] = []
        for group in self.groups.values():
            for obj in group.instancers:
                mesh = getattr(obj, "data", None)
                if mesh is None or not hasattr(mesh, "vertices"):
                    continue
                matrix = obj.matrix_world
                for vertex in mesh.vertices:
                    world = matrix @ vertex.co
                    points.append((world.x, world.y, world.z))
            if not group.instancers:
                for obj in group.shaded:
                    loc = obj.matrix_world.translation
                    points.append((loc.x, loc.y, loc.z))
        if len(points) > limit:
            stride = len(points) // limit + 1
            points = points[::stride]
        return points

    def bounds(self) -> tuple[Vec3, float]:
        """Return ``(center, radius)`` of the structure in world space.

        Raises:
            ValueError: when the structure has no usable geometry.
        """
        points = self.atom_positions()
        if not points:
            points = [tuple(obj.matrix_world.translation) for obj in self.objects]  # type: ignore[misc]
        if not points:
            raise ValueError(f"structure {self.name!r} has no geometry to measure")
        center = bbox_center(*bbox(points))
        return center, max(1e-3, bounding_radius(points, center))


def _root_of(obj: bpy.types.Object) -> bpy.types.Object:
    """Walk up the parent chain and return the topmost ancestor."""
    current = obj
    guard = 0
    while current.parent is not None and guard < 32:
        current = current.parent
        guard += 1
    return current


def _names_of(obj: bpy.types.Object) -> list[str]:
    """Object, data and material names, in resolution priority order."""
    names = [obj.name]
    data = getattr(obj, "data", None)
    if data is not None and hasattr(data, "name"):
        names.append(data.name)
    for slot in getattr(obj, "material_slots", []):
        if slot.material is not None:
            names.append(slot.material.name)
    return names


def classify(obj: bpy.types.Object) -> tuple[str, str | None]:
    """Classify one object.

    Returns:
        ``("atom", symbol)``, ``("stick", None)`` or ``("other", None)``.
    """
    if obj.type not in {"MESH", "CURVE", "SURFACE", "META"}:
        return "other", None
    names = _names_of(obj)
    if any(is_stick_name(name) for name in names):
        return "stick", None
    for name in names:
        symbol = guess_element(name)
        if symbol is not None:
            return "atom", symbol
    return "other", None


def _instanced_children(obj: bpy.types.Object) -> list[bpy.types.Object]:
    """Children that *obj* instances on its vertices/faces."""
    if getattr(obj, "instance_type", "NONE") not in {"VERTS", "FACES"}:
        return []
    return [child for child in obj.children if child.type in {"MESH", "CURVE", "SURFACE", "META"}]


def _unique(objects: list[bpy.types.Object]) -> list[bpy.types.Object]:
    """Remove duplicate objects while preserving order."""
    seen: set[str] = set()
    result = []
    for obj in objects:
        if obj.name not in seen:
            seen.add(obj.name)
            result.append(obj)
    return result


def _collect(objects: object) -> list[Structure]:
    """Group *objects* into structures by their topmost ancestor."""
    buckets: dict[str, Structure] = {}
    handled: set[str] = set()

    for obj in objects:  # type: ignore[arg-type]
        if obj.name in handled:
            continue
        kind, symbol = classify(obj)
        if kind == "other":
            continue
        root = _root_of(obj)
        # Parentless atoms (hand-built scenes) share one virtual structure
        # instead of becoming one structure per sphere.
        parented = root is not obj
        key = root.name if parented else "__loose__"
        structure = buckets.get(key)
        if structure is None:
            structure = Structure(
                name=root.name if parented else "Loose atoms",
                root=root if parented else None,
            )
            buckets[key] = structure

        if kind == "stick":
            structure.sticks.append(obj)
            handled.add(obj.name)
            continue

        assert symbol is not None
        group = structure.groups.setdefault(symbol, AtomGroup(symbol=symbol))
        children = _instanced_children(obj)
        if children:
            group.instancers.append(obj)
            for child in children:
                group.shaded.append(child)
                handled.add(child.name)
        else:
            group.shaded.append(obj)
        handled.add(obj.name)

    structures = [s for s in buckets.values() if s.groups or s.sticks]
    for structure in structures:
        for group in structure.groups.values():
            group.shaded = _unique(group.shaded)
            group.instancers = _unique(group.instancers)
        structure.sticks = _unique(structure.sticks)
    return structures


def detect_structures(
    context: bpy.types.Context,
    scope: str = "SCENE",
) -> list[Structure]:
    """Find atomic structures in the scene.

    Args:
        context: Blender context.
        scope: ``"SCENE"`` (everything), ``"SELECTED"`` (selection and its
            descendants) or ``"COLLECTION"`` (the active collection).

    Returns:
        Structures ordered by atom count, largest first.
    """
    if scope == "SELECTED" and context.selected_objects:
        pool: list[bpy.types.Object] = []
        for obj in context.selected_objects:
            pool.append(obj)
            pool.extend(obj.children_recursive)
    elif scope == "COLLECTION" and context.collection is not None:
        pool = list(context.collection.all_objects)
    else:
        pool = list(context.scene.objects)

    structures = _collect(pool)

    if not structures:
        # Nothing looked like an element: treat plain geometry as one generic
        # structure so the styling operators still do something useful.
        meshes = [obj for obj in pool if obj.type == "MESH"]
        if meshes:
            fallback = Structure(name="Unnamed structure", root=None)
            group = AtomGroup(symbol=UNKNOWN_SYMBOL)
            for obj in meshes:
                if _instanced_children(obj):
                    group.instancers.append(obj)
                    group.shaded.extend(_instanced_children(obj))
                else:
                    group.shaded.append(obj)
            fallback.groups[UNKNOWN_SYMBOL] = group
            structures = [fallback]

    structures.sort(key=lambda s: s.atom_count, reverse=True)
    return structures


def active_structure(context: bpy.types.Context, scope: str = "SCENE") -> Structure | None:
    """Return the largest detected structure, or ``None`` when there is none."""
    structures = detect_structures(context, scope)
    return structures[0] if structures else None
