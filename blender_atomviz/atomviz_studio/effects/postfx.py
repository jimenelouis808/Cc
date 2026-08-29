"""Compositor post-processing: glow, streaks, vignette and colour grading.

Bloom is done in the compositor rather than with EEVEE's legacy bloom pass so
the exact same result comes out of EEVEE, EEVEE Next and Cycles — Blender 4.2
removed the render-time bloom option entirely.

Two compositor APIs are supported:

* **3.3 - 4.5** — ``scene.use_nodes`` + ``scene.node_tree``, Glare settings as
  node *properties*, ``CompositorNodeComposite`` as the output.
* **5.0+** — a ``CompositorNodeTree`` node group assigned to
  ``scene.compositing_node_group``, Glare settings as *input sockets*, and a
  ``NodeGroupOutput`` fed through the group interface. ``MixRGB`` is gone, so
  the vignette multiply uses ``ShaderNodeMix``.

Both build the same chain::

    Render Layers -> Glare -> [Streaks] -> Vignette -> Tint -> Contrast -> Saturation -> Output
"""

from __future__ import annotations

import bpy

from ..core.colors import hex_to_linear

GLARE_TYPES = ("NONE", "FOG_GLOW", "STREAKS", "GHOSTS", "SIMPLE_STAR")

#: Modern Glare nodes take the type as a menu socket with a label, not an enum.
_GLARE_SOCKET_LABELS = {
    "FOG_GLOW": "Fog Glow",
    "STREAKS": "Streaks",
    "GHOSTS": "Ghosts",
    "SIMPLE_STAR": "Simple Star",
    "BLOOM": "Bloom",
}

#: ``True`` on Blender 5.0+, where the scene compositor is a node group.
USES_NODE_GROUP = not hasattr(bpy.types.Scene, "node_tree")

COMPOSITOR_GROUP_NAME = "AV_Compositor"


def _set(node: bpy.types.Node, attr: str, value: object) -> bool:
    """Set a node property when this Blender build still has it."""
    if not hasattr(node, attr):
        return False
    try:
        setattr(node, attr, value)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _set_socket(node: bpy.types.Node, name: str, value: object) -> bool:
    """Set an input socket by name, ignoring sockets this build does not have."""
    socket = node.inputs.get(name)
    if socket is None:
        return False
    try:
        socket.default_value = value  # type: ignore[assignment]
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _configure_glare(
    node: bpy.types.Node,
    glare_type: str,
    threshold: float,
    size: int,
    streak_count: int = 4,
    strength: float = 0.35,
) -> None:
    """Apply Glare settings through whichever API this Blender exposes."""
    # Legacy (properties).
    _set(node, "glare_type", glare_type)
    _set(node, "quality", "HIGH")
    _set(node, "threshold", threshold)
    _set(node, "size", max(1, min(9, size)))
    _set(node, "streaks", streak_count)
    _set(node, "angle_offset", 0.35)
    _set(node, "mix", -(1.0 - strength))  # legacy: -1 = original only, 0 = 50/50
    # Modern (sockets).
    _set_socket(node, "Type", _GLARE_SOCKET_LABELS.get(glare_type, "Fog Glow"))
    _set_socket(node, "Quality", "High")
    _set_socket(node, "Threshold", threshold)
    _set_socket(node, "Size", max(0.05, min(1.0, size / 9.0)))
    _set_socket(node, "Streaks", streak_count)
    _set_socket(node, "Strength", strength)


# --------------------------------------------------------------------------
# Tree creation / teardown
# --------------------------------------------------------------------------
def _legacy_tree(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    """Enable compositing on *scene* and return its emptied node tree."""
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    return tree


def _group_tree(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    """Create (or reset) the scene's compositing node group (Blender 5.0+)."""
    tree = bpy.data.node_groups.get(COMPOSITOR_GROUP_NAME)
    if tree is None or tree.bl_idname != "CompositorNodeTree":
        tree = bpy.data.node_groups.new(COMPOSITOR_GROUP_NAME, "CompositorNodeTree")
    tree.nodes.clear()
    for item in list(tree.interface.items_tree):
        tree.interface.remove(item)
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = tree
    if hasattr(scene, "use_nodes"):
        try:
            scene.use_nodes = True
        except (AttributeError, TypeError):  # pragma: no cover - deprecated in 5.0
            pass
    return tree


def _tree(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    """Return an empty compositor tree for *scene*, whichever API applies."""
    return _group_tree(scene) if USES_NODE_GROUP else _legacy_tree(scene)


def _new(tree: bpy.types.NodeTree, bl_idname: str, location: tuple[float, float]) -> bpy.types.Node:
    """Create a compositor node, or raise a clear error when unavailable."""
    try:
        node = tree.nodes.new(bl_idname)
    except RuntimeError as exc:  # pragma: no cover - build dependent
        raise RuntimeError(f"compositor node {bl_idname!r} unavailable") from exc
    node.location = location
    return node


def _multiply(tree: bpy.types.NodeTree, location: tuple[float, float], factor: float):
    """Colour multiply node, tolerant of the compositor Mix rewrite.

    Returns:
        ``(node, image_socket, factor_socket_target, result_socket)`` where the
        second socket takes the image and the third the mask.
    """
    for bl_idname in ("CompositorNodeMixRGB", "ShaderNodeMix"):
        try:
            node = tree.nodes.new(bl_idname)
        except RuntimeError:
            continue
        node.location = location
        if bl_idname == "CompositorNodeMixRGB":
            node.blend_type = "MULTIPLY"
            node.inputs[0].default_value = factor
            return node, node.inputs[1], node.inputs[2], node.outputs[0]
        node.data_type = "RGBA"
        node.blend_type = "MULTIPLY"
        node.inputs[0].default_value = factor
        return node, node.inputs[6], node.inputs[7], node.outputs[2]
    raise RuntimeError("no compositor mix node available")  # pragma: no cover


def _overlay(tree: bpy.types.NodeTree, location: tuple[float, float], factor: float, color):
    """Overlay a flat colour over the image; same socket dance as ``_multiply``."""
    node, image_socket, other_socket, result = _multiply(tree, location, factor)
    _set(node, "blend_type", "OVERLAY")
    other_socket.default_value = color
    return node, image_socket, result


def _output(tree: bpy.types.NodeTree, location: tuple[float, float]) -> bpy.types.Node:
    """Create the tree's final node (Composite, or the group output on 5.0+)."""
    if not USES_NODE_GROUP:
        return _new(tree, "CompositorNodeComposite", location)
    return _new(tree, "NodeGroupOutput", location)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def setup(
    scene: bpy.types.Scene,
    glare: str = "FOG_GLOW",
    threshold: float = 0.7,
    glare_size: int = 8,
    streaks: bool = False,
    vignette: float = 0.3,
    contrast: float = 0.0,
    saturation: float = 1.0,
    tint: str | None = None,
) -> bpy.types.NodeTree:
    """Build the post-processing chain.

    Args:
        scene: Target scene.
        glare: One of :data:`GLARE_TYPES`; ``"NONE"`` skips the glow stage.
        threshold: Brightness above which pixels bloom. Lower = more glow.
        glare_size: Fog-glow radius on Blender's 1..9 scale.
        streaks: Add a second Glare node in streak mode (anamorphic look).
        vignette: Corner darkening strength in ``[0, 1]``; ``0`` disables it.
        contrast: Bright/contrast offset applied at the end.
        saturation: Colour saturation multiplier.
        tint: Optional hex colour overlaid at 12 % (mood grade).

    Returns:
        The compositor node tree that was built.
    """
    tree = _tree(scene)
    render_layers = _new(tree, "CompositorNodeRLayers", (-900, 0))
    current = render_layers.outputs["Image"]

    if glare and glare.upper() != "NONE":
        node = _new(tree, "CompositorNodeGlare", (-660, 120))
        _configure_glare(node, glare.upper(), threshold, glare_size)
        tree.links.new(current, node.inputs["Image"])
        current = node.outputs["Image"]

    if streaks:
        node = _new(tree, "CompositorNodeGlare", (-460, 260))
        _configure_glare(node, "STREAKS", max(threshold, 0.9), glare_size, strength=0.22)
        tree.links.new(current, node.inputs["Image"])
        current = node.outputs["Image"]

    if vignette > 0.0:
        mask = _new(tree, "CompositorNodeEllipseMask", (-660, -320))
        # 3.3-4.5 use width/height properties; 5.0 uses a Size vector socket.
        _set(mask, "width", 0.92)
        _set(mask, "height", 0.92)
        _set_socket(mask, "Size", (0.92, 0.92, 0.0))
        blur = _new(tree, "CompositorNodeBlur", (-460, -320))
        _set(blur, "filter_type", "GAUSS")
        _set(blur, "use_relative", True)
        _set(blur, "factor_x", 28.0)
        _set(blur, "factor_y", 28.0)
        _set_socket(blur, "Size", (0.25, 0.25, 0.0))
        node, image_socket, mask_socket, result = _multiply(tree, (-240, -120), max(0.0, min(1.0, vignette)))
        tree.links.new(mask.outputs[0], blur.inputs["Image"])
        tree.links.new(current, image_socket)
        tree.links.new(blur.outputs["Image"], mask_socket)
        current = result

    if tint:
        node, image_socket, result = _overlay(tree, (-40, -220), 0.12, hex_to_linear(tint))
        tree.links.new(current, image_socket)
        current = result

    if abs(contrast) > 1e-6:
        node = _new(tree, "CompositorNodeBrightContrast", (140, 60))
        _set_socket(node, "Contrast", contrast)
        tree.links.new(current, node.inputs["Image"])
        current = node.outputs["Image"]

    if abs(saturation - 1.0) > 1e-6:
        node = _new(tree, "CompositorNodeHueSat", (340, 60))
        _set_socket(node, "Saturation", saturation)
        tree.links.new(current, node.inputs["Image"])
        current = node.outputs["Image"]

    output = _output(tree, (620, 0))
    tree.links.new(current, output.inputs[0])
    try:  # The viewer node is a convenience, not a requirement.
        viewer = _new(tree, "CompositorNodeViewer", (620, -220))
        tree.links.new(current, viewer.inputs["Image"])
    except RuntimeError:  # pragma: no cover - build dependent
        pass
    return tree


def disable(scene: bpy.types.Scene) -> None:
    """Remove the post-processing chain and turn compositing off."""
    if USES_NODE_GROUP:
        scene.compositing_node_group = None
        tree = bpy.data.node_groups.get(COMPOSITOR_GROUP_NAME)
        if tree is not None:
            bpy.data.node_groups.remove(tree)
        return
    if scene.node_tree is not None:
        scene.node_tree.nodes.clear()
    scene.use_nodes = False
