"""Compositor post-processing: glow, streaks, vignette and colour grading.

Bloom is done in the compositor rather than with EEVEE's legacy bloom pass so
the exact same result comes out of EEVEE, EEVEE Next and Cycles — Blender 4.2
removed the render-time bloom option entirely.

The node chain is::

    Render Layers -> Glare -> Vignette (multiply) -> Grade -> Composite
"""

from __future__ import annotations

import bpy

from ..core.colors import hex_to_linear

GLARE_TYPES = ("NONE", "FOG_GLOW", "STREAKS", "GHOSTS", "SIMPLE_STAR")


def _tree(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    """Enable compositing on *scene* and return an empty node tree."""
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    return tree


def _new(tree: bpy.types.NodeTree, bl_idname: str, location: tuple[float, float]) -> bpy.types.Node:
    """Create a compositor node, or raise a clear error when unavailable."""
    try:
        node = tree.nodes.new(bl_idname)
    except RuntimeError as exc:  # pragma: no cover - build dependent
        raise RuntimeError(f"compositor node {bl_idname!r} unavailable") from exc
    node.location = location
    return node


def _set(node: bpy.types.Node, attr: str, value: object) -> bool:
    """Set a node property when this Blender build still has it."""
    if not hasattr(node, attr):
        return False
    try:
        setattr(node, attr, value)
    except (AttributeError, TypeError, ValueError):
        return False
    return True


def _mix_node(tree: bpy.types.NodeTree, location: tuple[float, float], blend_type: str):
    """Compositor colour-mix node, tolerant of the 4.x node rename."""
    for bl_idname in ("CompositorNodeMixRGB", "CompositorNodeMix"):
        try:
            node = tree.nodes.new(bl_idname)
        except RuntimeError:
            continue
        node.location = location
        node.blend_type = blend_type
        return node
    raise RuntimeError("no compositor mix node available")  # pragma: no cover


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
        glare_size: Fog-glow radius exponent (Blender's own 6..9 scale).
        streaks: Add a second Glare node in streak mode (anamorphic look).
        vignette: Corner darkening strength in ``[0, 1]``; ``0`` disables it.
        contrast: Bright/contrast offset applied at the end.
        saturation: Colour saturation multiplier.
        tint: Optional hex colour mixed over the image at 12 % (mood grade).

    Returns:
        The scene's compositor node tree.
    """
    tree = _tree(scene)
    render_layers = _new(tree, "CompositorNodeRLayers", (-900, 0))
    current = render_layers.outputs["Image"]

    if glare and glare.upper() != "NONE":
        node = _new(tree, "CompositorNodeGlare", (-660, 120))
        _set(node, "glare_type", glare.upper())
        _set(node, "quality", "HIGH")
        _set(node, "threshold", threshold)
        _set(node, "size", max(1, min(9, glare_size)))
        _set(node, "mix", 0.0)
        tree.links.new(current, node.inputs["Image"])
        current = node.outputs["Image"]

    if streaks:
        node = _new(tree, "CompositorNodeGlare", (-460, 260))
        _set(node, "glare_type", "STREAKS")
        _set(node, "quality", "HIGH")
        _set(node, "threshold", max(threshold, 0.9))
        _set(node, "streaks", 4)
        _set(node, "angle_offset", 0.35)
        _set(node, "mix", -0.55)
        tree.links.new(current, node.inputs["Image"])
        current = node.outputs["Image"]

    if vignette > 0.0:
        mask = _new(tree, "CompositorNodeEllipseMask", (-660, -320))
        mask.width = 0.92
        mask.height = 0.92
        blur = _new(tree, "CompositorNodeBlur", (-460, -320))
        blur.filter_type = "GAUSS"
        blur.use_relative = True
        blur.factor_x = 28.0
        blur.factor_y = 28.0
        mix = _mix_node(tree, (-240, -120), "MULTIPLY")
        mix.inputs[0].default_value = max(0.0, min(1.0, vignette))
        tree.links.new(mask.outputs["Mask"], blur.inputs["Image"])
        tree.links.new(current, mix.inputs[1])
        tree.links.new(blur.outputs["Image"], mix.inputs[2])
        current = mix.outputs[0]

    if tint:
        mix = _mix_node(tree, (-40, -220), "OVERLAY")
        mix.inputs[0].default_value = 0.12
        tree.links.new(current, mix.inputs[1])
        mix.inputs[2].default_value = hex_to_linear(tint)
        current = mix.outputs[0]

    if abs(contrast) > 1e-6:
        node = _new(tree, "CompositorNodeBrightContrast", (140, 60))
        node.inputs["Contrast"].default_value = contrast
        tree.links.new(current, node.inputs["Image"])
        current = node.outputs["Image"]

    if abs(saturation - 1.0) > 1e-6:
        node = _new(tree, "CompositorNodeHueSat", (340, 60))
        node.inputs["Saturation"].default_value = saturation
        tree.links.new(current, node.inputs["Image"])
        current = node.outputs["Image"]

    composite = _new(tree, "CompositorNodeComposite", (620, 0))
    tree.links.new(current, composite.inputs["Image"])
    viewer = _new(tree, "CompositorNodeViewer", (620, -220))
    tree.links.new(current, viewer.inputs["Image"])
    return tree


def disable(scene: bpy.types.Scene) -> None:
    """Remove the post-processing chain and turn compositing off."""
    if scene.node_tree is not None:
        scene.node_tree.nodes.clear()
    scene.use_nodes = False
