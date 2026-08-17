"""Small helpers to build shader / world / compositor node trees.

Blender's node API is verbose and its node identifiers drifted between
versions (``ShaderNodeMixRGB`` -> ``ShaderNodeMix``, Musgrave removal in 4.1).
These helpers keep the effect modules readable and version tolerant; only
nodes that exist in **every** supported release are used directly.
"""

from __future__ import annotations

import bpy

from .colors import hex_to_linear


def clear(tree: bpy.types.NodeTree) -> None:
    """Remove every node from *tree*."""
    tree.nodes.clear()


def new(
    tree: bpy.types.NodeTree,
    bl_idname: str,
    location: tuple[float, float] = (0.0, 0.0),
    label: str = "",
    **props: object,
) -> bpy.types.Node:
    """Create a node and set attributes/inputs in one call.

    Keys in *props* are applied as node attributes when they exist
    (``noise_dimensions``, ``operation``, ...) and otherwise as input socket
    defaults by name (``Scale``, ``Strength``, ...).

    Raises:
        RuntimeError: if *bl_idname* does not exist in this Blender build.
    """
    try:
        node = tree.nodes.new(bl_idname)
    except RuntimeError as exc:  # pragma: no cover - build dependent
        raise RuntimeError(f"node type {bl_idname!r} is not available in this Blender") from exc
    node.location = location
    if label:
        node.label = label
    for key, value in props.items():
        if hasattr(node, key):
            setattr(node, key, value)
            continue
        socket = node.inputs.get(key) or node.inputs.get(key.replace("_", " ").title())
        if socket is not None:
            socket.default_value = value  # type: ignore[assignment]
    return node


def link(
    tree: bpy.types.NodeTree,
    source: bpy.types.Node,
    target: bpy.types.Node,
    from_socket: str | int = 0,
    to_socket: str | int = 0,
) -> bpy.types.NodeLink:
    """Connect two nodes by socket name or index."""
    out = source.outputs[from_socket]
    inp = target.inputs[to_socket]
    return tree.links.new(out, inp)


def ramp(node: bpy.types.Node, stops: object) -> bpy.types.Node:
    """Fill a Color Ramp node with ``(position, hex_or_rgba)`` *stops*.

    The two default elements are reused so at least one always survives
    (Blender refuses to delete the last element).
    """
    elements = node.color_ramp.elements
    stop_list = list(stops)  # type: ignore[arg-type]
    if not stop_list:
        return node
    while len(elements) > len(stop_list) and len(elements) > 1:
        elements.remove(elements[-1])
    while len(elements) < len(stop_list):
        elements.new(1.0)
    for element, (position, color) in zip(elements, stop_list):
        element.position = float(position)
        element.color = hex_to_linear(color) if isinstance(color, str) else tuple(color)
    return node


def output_node(tree: bpy.types.NodeTree, bl_idname: str) -> bpy.types.Node:
    """Return the tree's output node, creating it when missing."""
    for node in tree.nodes:
        if node.bl_idname == bl_idname:
            return node
    return new(tree, bl_idname, location=(600.0, 0.0))


def material_output(tree: bpy.types.NodeTree) -> bpy.types.Node:
    """Return (or create) the Material Output node."""
    return output_node(tree, "ShaderNodeOutputMaterial")


def world_output(tree: bpy.types.NodeTree) -> bpy.types.Node:
    """Return (or create) the World Output node."""
    return output_node(tree, "ShaderNodeOutputWorld")


def new_material(name: str, reuse: bool = True) -> tuple[bpy.types.Material, bpy.types.NodeTree]:
    """Create (or reset) a node-based material called *name*.

    Args:
        name: Material name. Existing materials are reused and cleared so
            re-running an operator updates instead of piling up ``.001`` copies.
        reuse: When ``False`` a fresh material is always created.

    Returns:
        ``(material, node_tree)`` with an empty node tree.
    """
    material = bpy.data.materials.get(name) if reuse else None
    if material is None:
        material = bpy.data.materials.new(name)
    material.use_nodes = True
    clear(material.node_tree)
    return material, material.node_tree


def mix_rgb(
    tree: bpy.types.NodeTree,
    location: tuple[float, float] = (0.0, 0.0),
    blend_type: str = "MIX",
    factor: float = 0.5,
) -> tuple[bpy.types.Node, bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket, bpy.types.NodeSocket]:
    """Create a colour mix node, using ``ShaderNodeMix`` when available.

    The 3.4+ ``ShaderNodeMix`` node exposes several same-named sockets (one per
    data type), so sockets are resolved by index rather than by name.

    Returns:
        ``(node, factor_socket, a_socket, b_socket, result_socket)``.
    """
    if hasattr(bpy.types, "ShaderNodeMix"):
        node = new(tree, "ShaderNodeMix", location=location)
        node.data_type = "RGBA"
        node.blend_type = blend_type
        node.inputs[0].default_value = factor
        return node, node.inputs[0], node.inputs[6], node.inputs[7], node.outputs[2]
    node = new(tree, "ShaderNodeMixRGB", location=location)  # pragma: no cover - <3.4
    node.blend_type = blend_type
    node.inputs[0].default_value = factor
    return node, node.inputs[0], node.inputs[1], node.inputs[2], node.outputs[0]


def emission_shader(
    tree: bpy.types.NodeTree,
    color: str | tuple[float, float, float, float],
    strength: float,
    location: tuple[float, float] = (0.0, 0.0),
) -> bpy.types.Node:
    """Create an Emission shader with *color* (hex or RGBA) and *strength*."""
    node = new(tree, "ShaderNodeEmission", location=location)
    node.inputs["Color"].default_value = hex_to_linear(color) if isinstance(color, str) else color
    node.inputs["Strength"].default_value = strength
    return node


def gradient_world(
    tree: bpy.types.NodeTree,
    top: str,
    bottom: str,
    strength: float = 1.0,
    location: tuple[float, float] = (-600.0, 0.0),
) -> bpy.types.Node:
    """Build a vertical two-colour gradient and return its Background node.

    The gradient is driven by the *Z* component of the view vector, so it
    behaves like a studio backdrop no matter where the camera points.
    """
    coord = new(tree, "ShaderNodeTexCoord", location=location)
    separate = new(tree, "ShaderNodeSeparateXYZ", location=(location[0] + 200, location[1]))
    map_range = new(
        tree,
        "ShaderNodeMapRange",
        location=(location[0] + 380, location[1]),
        From_Min=-0.6,
        From_Max=0.8,
    )
    color_ramp = new(tree, "ShaderNodeValToRGB", location=(location[0] + 560, location[1]))
    ramp(color_ramp, [(0.0, bottom), (1.0, top)])
    background = new(tree, "ShaderNodeBackground", location=(location[0] + 860, location[1]))
    background.inputs["Strength"].default_value = strength

    link(tree, coord, separate, from_socket="Generated", to_socket=0)
    link(tree, separate, map_range, from_socket="Z", to_socket=0)
    link(tree, map_range, color_ramp, from_socket=0, to_socket="Fac")
    link(tree, color_ramp, background, from_socket="Color", to_socket="Color")
    return background
