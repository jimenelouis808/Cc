"""A cover project in Blender from one of the scenes' .extxyz, fast.

    <python con bpy> modelos_SEM/portada_blend.py -- \
        --extxyz modelos_SEM/xyz/escena_placas.extxyz \
        --blend modelos_SEM/blend/escena_placas.blend [--png vista.png]

    blender -b -P modelos_SEM/portada_blend.py -- --extxyz ... --blend ...

The studio is the one already chosen for these samples: the package's
`core_shell_cover` style, applied by `render_cnt.py`'s own functions --
its materials, world, floor, light rig, measured camera and render
settings. Nothing about the look is redefined here, so a change to the
style reaches this script by itself; `--style` takes any other.

What IS different is the geometry, and it is the whole reason for the
script. **Every atom is an INSTANCE.** `render_cnt.py` builds one
icosphere and one cylinder per atom and bond in Python: 88 minutes
measured for 10 000 atoms, and a file in the hundreds of megabytes.
Here each element is one mesh of bare vertices with a single sphere
instanced on them, and the C-C bonds are one mesh of tiny triangles --
each perpendicular to its bond at the midpoint -- with a single
cylinder instanced on the faces. A 78 000-atom scene builds in seconds
into a file of a few megabytes, and Cycles instances natively.

On top of the style, the camera gets depth of field focused on the
scene's centre (`--no-dof` to leave it out). The style's glare is a
pass over the finished PNG, so it is applied to `--png` and is not
something the .blend carries.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nanocarbon_lab" / "blender"))

import effects as fx  # noqa: E402
import render_cnt  # noqa: E402
import styles as style_defs  # noqa: E402

#: Sphere radius of the metals, in A: large enough that neighbours
#: fuse and a core reads as a solid. Carbon takes the style's own
#: ball-and-stick radius.
SOLID_RADIUS = {"Fe": 1.22, "Se": 1.30}

BOND_MAX = 1.75      # A, C-C

#: Van der Waals radii for the ``revista`` look, in A: Fe from Batsanov
#: (2001), Se and C from Bondi (1964). At these the metal is a closed
#: space-filling solid, which is how a cover draws a particle.
VDW_RADIUS = {"Fe": 2.00, "Se": 1.90}

#: The ``revista`` palette: a deep indigo field, carbon as pearl with a
#: cold rim, iron as polished copper and selenium as teal -- warm
#: against cold, so the metal and the shell never merge into one hue.
REVISTA_MATERIALS = {
    "C": style_defs.MaterialSpec(
        base_color=(0.62, 0.70, 0.80, 1.0), metallic=0.85, roughness=0.18,
        clearcoat=0.6, coat_roughness=0.05, specular_tint=(0.80, 0.90, 1.0),
        rim=2.2, rim_color=(0.35, 0.80, 1.00), rim_power=3.0),
    "Fe": style_defs.MaterialSpec(
        base_color=(0.96, 0.43, 0.15, 1.0), metallic=1.0, roughness=0.16,
        clearcoat=0.4, coat_roughness=0.04, specular_tint=(1.0, 0.70, 0.45),
        rim=0.8, rim_color=(1.00, 0.55, 0.25), rim_power=4.0),
    "Se": style_defs.MaterialSpec(
        base_color=(0.04, 0.58, 0.62, 1.0), metallic=0.9, roughness=0.20,
        clearcoat=0.4, coat_roughness=0.05, specular_tint=(0.55, 1.0, 0.95),
        rim=0.8, rim_color=(0.30, 1.00, 0.90), rim_power=4.0),
}
REVISTA_MATERIALS.update({
    "N": style_defs.MaterialSpec(
        base_color=(0.16, 0.36, 1.00, 1.0), metallic=0.3, roughness=0.22,
        clearcoat=0.7, coat_roughness=0.04,
        rim=1.4, rim_color=(0.45, 0.65, 1.00), rim_power=3.5),
    "O": style_defs.MaterialSpec(
        base_color=(0.95, 0.10, 0.20, 1.0), metallic=0.3, roughness=0.22,
        clearcoat=0.7, coat_roughness=0.04,
        rim=1.2, rim_color=(1.00, 0.45, 0.50), rim_power=3.5),
    "H": style_defs.MaterialSpec(
        base_color=(0.92, 0.94, 0.98, 1.0), metallic=0.0, roughness=0.25,
        clearcoat=0.6, coat_roughness=0.04,
        rim=0.8, rim_color=(0.80, 0.90, 1.00), rim_power=3.5),
})
REVISTA_BOND = dataclasses.replace(REVISTA_MATERIALS["C"], rim=1.4)

def _spec(rgb, metallic=0.9, roughness=0.18, **extra):
    """A MaterialSpec from a colour and the few knobs a palette turns."""
    return style_defs.MaterialSpec(base_color=(*rgb, 1.0), metallic=metallic,
                                   roughness=roughness, **extra)


#: Alternative palettes for `--style revista`, each a full set of
#: element materials plus the bond's. All of them push the carbon
#: NETWORK forward: the bond carries the colour or the light, so the
#: hexagons read, rather than the atoms, which a close-up shows as a
#: field of beads.
PALETTES: dict[str, dict] = {
    "revista": {**REVISTA_MATERIALS, "bond": REVISTA_BOND},
    # Black glass atoms and glowing cyan bonds: the honeycomb as a
    # lit wireframe. Gold iron, magenta selenium.
    "neon": {
        "C": _spec((0.015, 0.02, 0.03), 0.6, 0.25, clearcoat=0.8,
                   coat_roughness=0.03),
        # Emission 1.2: at 5 the whole tube burnt out to white and the
        # glare pass spread it over the frame.
        "bond": _spec((0.01, 0.10, 0.16), 0.0, 0.3,
                      emission_color=(0.05, 0.75, 1.00),
                      emission_strength=1.2),
        "Fe": _spec((1.00, 0.70, 0.22), 1.0, 0.12,
                    specular_tint=(1.0, 0.85, 0.5)),
        "Se": _spec((0.95, 0.12, 0.70), 0.85, 0.18, clearcoat=0.5),
        "N": _spec((0.25, 0.45, 1.00), 0.2, 0.2,
                   emission_color=(0.3, 0.5, 1.0), emission_strength=1.5),
        "O": _spec((1.00, 0.15, 0.25), 0.2, 0.2,
                   emission_color=(1.0, 0.2, 0.3), emission_strength=1.0),
        "H": _spec((0.90, 0.92, 1.00), 0.0, 0.25),
    },
    # Black lacquer and gold: the network as jewellery. Rose-gold iron,
    # emerald selenium.
    "oro": {
        "C": _spec((0.012, 0.012, 0.015), 0.3, 0.08, clearcoat=1.0,
                   coat_roughness=0.02),
        "bond": _spec((1.00, 0.74, 0.30), 1.0, 0.14,
                      specular_tint=(1.0, 0.85, 0.55)),
        "Fe": _spec((0.98, 0.52, 0.40), 1.0, 0.14,
                    specular_tint=(1.0, 0.75, 0.65)),
        "Se": _spec((0.04, 0.62, 0.34), 0.9, 0.16, clearcoat=0.6),
        "N": _spec((0.20, 0.35, 0.95), 0.5, 0.18, clearcoat=0.8),
        "O": _spec((0.90, 0.10, 0.18), 0.5, 0.18, clearcoat=0.8),
        "H": _spec((0.95, 0.95, 0.95), 0.2, 0.2),
    },
    # Silver under a thin film: the colour changes with the angle, so a
    # curved wall shifts from violet to gold across the tube. Deep
    # orange iron, violet selenium.
    "iridiscente": {
        "C": _spec((0.85, 0.87, 0.92), 1.0, 0.10, thin_film=420.0,
                   thin_film_ior=1.45),
        "bond": _spec((0.85, 0.87, 0.92), 1.0, 0.10, thin_film=520.0,
                      thin_film_ior=1.45),
        "Fe": _spec((1.00, 0.38, 0.06), 1.0, 0.14),
        "Se": _spec((0.45, 0.20, 0.95), 0.9, 0.16, clearcoat=0.5),
        "N": _spec((0.15, 0.40, 1.00), 0.4, 0.2),
        "O": _spec((1.00, 0.12, 0.20), 0.4, 0.2),
        "H": _spec((0.95, 0.95, 1.00), 0.0, 0.25),
    },
}


#: Elements drawn, in the order their carriers are built.
ELEMENT_ORDER = ("C", "N", "O", "H", "Fe", "Se")

#: Ball radii for atoms drawn ball-and-stick (bonded, or in a `--bola`
#: group), in A. Carbon takes the style's own.
BALL_RADIUS = {"N": 0.44, "O": 0.42, "H": 0.26, "Se": 0.66, "Fe": 0.72}

#: Bonds between non-metals are drawn in three length classes (X-H,
#: X-X, X-Se), each with a stick of its own median length.
def _cutoff(a: str, b: str) -> tuple[float, str]:
    """Bond cutoff and length class for an element pair."""
    pair = {a, b}
    if "H" in pair:
        return (1.25, "XH") if pair != {"H"} else (0.0, "")
    if "Se" in pair:
        return (2.10, "XSe") if pair != {"Se"} else (2.45, "XSe")
    return 1.75, "XX"


def all_bonds(symbols: np.ndarray, positions: np.ndarray):
    """Bonds among the non-metals, and each bond's length class.

    Iron is left out: in a carbide or a selenide its contacts are part
    of a solid drawn space-filling, and in ferrocene the ring-to-metal
    contact is a hapto bond, which a stick would misdraw.
    """
    light = np.flatnonzero(symbols != "Fe")
    pairs = carbon_bonds(positions[light], 2.45)
    if not len(pairs):
        return np.empty((0, 2), dtype=int), np.empty(0, dtype=object)
    pairs = light[pairs]
    keep, classes = [], []
    lengths = np.linalg.norm(positions[pairs[:, 0]] - positions[pairs[:, 1]],
                             axis=1)
    for (i, j), length in zip(pairs, lengths, strict=True):
        cutoff, label = _cutoff(symbols[i], symbols[j])
        keep.append(length < cutoff)
        classes.append(label)
    keep = np.asarray(keep)
    return pairs[keep], np.asarray(classes, dtype=object)[keep]


def revista_style(base: style_defs.Style) -> style_defs.Style:
    """`base` re-lit for a high-impact cover.

    The field is a deep indigo graded to near-black, so the subject's
    cold rim reads against it; the key is warm and high, a strong cold
    back light draws every silhouette, and a magenta kicker low on the
    far side gives the shadows a colour instead of black.
    """
    return dataclasses.replace(
        base,
        name="revista",
        # A black mirror under the subject: dark enough to stay a
        # floor, glossy enough to give back a soft copy of the scene.
        background=style_defs.Background(
            kind="gradient", top=(0.020, 0.024, 0.075),
            bottom=(0.002, 0.003, 0.010), strength=1.0,
            floor=True, floor_color=(0.006, 0.007, 0.016),
            floor_roughness=0.07, floor_metallic=0.85),
        lights=[
            style_defs.LightSpec("AREA", 5800.0, (1.00, 0.88, 0.70),
                                 (7.0, -7.0, 11.0), size=7.0),
            style_defs.LightSpec("AREA", 7000.0, (0.40, 0.75, 1.00),
                                 (-4.0, 9.0, 10.0), size=5.0),
            style_defs.LightSpec("AREA", 2200.0, (0.85, 0.30, 0.95),
                                 (-9.0, -3.0, -4.0), size=10.0),
            style_defs.LightSpec("AREA", 1200.0, (0.55, 0.65, 1.00),
                                 (0.0, -10.0, 1.0), size=16.0),
        ],
        effects=(fx.EffectSpec("glare", {"strength": 0.42, "size": 0.55,
                                         "threshold": 0.70,
                                         "maximum": 3.0}),),
    )


def read_extxyz(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Symbols, positions and the `grupo` column (zeros if absent)."""
    lines = path.read_text().splitlines()
    count = int(lines[0])
    symbols, rows = [], []
    for line in lines[2:2 + count]:
        fields = line.split()
        symbols.append(fields[0])
        rows.append([float(v) for v in fields[1:5]] if len(fields) > 4
                    else [float(v) for v in fields[1:4]] + [0.0])
    table = np.asarray(rows)
    return symbols, table[:, :3], table[:, 3].astype(int)


def carbon_bonds(positions: np.ndarray, cutoff: float) -> np.ndarray:
    """Pairs closer than `cutoff`, by a sweep along x. numpy only."""
    order = np.argsort(positions[:, 0])
    sorted_positions = positions[order]
    last = np.searchsorted(sorted_positions[:, 0],
                           sorted_positions[:, 0] + cutoff)
    pairs = []
    for index in range(len(order)):
        window = sorted_positions[index + 1:last[index]]
        if not len(window):
            continue
        near = np.flatnonzero(
            ((window - sorted_positions[index]) ** 2).sum(axis=1)
            < cutoff * cutoff)
        pairs.extend((order[index], order[index + 1 + k]) for k in near)
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def link(obj: bpy.types.Object) -> bpy.types.Object:
    bpy.context.scene.collection.objects.link(obj)
    return obj


def instanced(name: str, vertices: np.ndarray, faces: list,
              child: bpy.types.Object, kind: str) -> bpy.types.Object:
    """A carrier mesh with `child` instanced on its vertices or faces."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(), [], faces)
    mesh.update()
    carrier = link(bpy.data.objects.new(name, mesh))
    carrier.instance_type = kind
    carrier.show_instancer_for_render = False
    carrier.show_instancer_for_viewport = False
    child.parent = carrier
    return carrier


def primitive(name: str, make, mat: bpy.types.Material) -> bpy.types.Object:
    """One smooth-shaded primitive at the origin, carrying `mat`."""
    make()
    obj = bpy.context.active_object
    obj.name = name
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.materials.append(mat)
    return obj


def calor_backdrop(invert: bool = False) -> None:
    """The field graded across the frame: amber left, near-black middle,
    cold blue right. With `--calor`; window coordinates, so it stays put
    under the camera. (Multiplied into the style's vertical ramp it was
    too dark to read; it replaces it.)
    """
    world = bpy.context.scene.world
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = next(n for n in nodes if n.type == "BACKGROUND")
    coords = nodes.new("ShaderNodeTexCoord")
    split = nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeValToRGB")
    warm, cold = (0.07, 0.022, 0.004, 1.0), (0.006, 0.028, 0.085, 1.0)
    if invert:
        warm, cold = cold, warm
    ramp.color_ramp.elements[0].color = warm
    ramp.color_ramp.elements[1].color = cold
    middle = ramp.color_ramp.elements.new(0.5)
    middle.color = (0.012, 0.012, 0.022, 1.0)
    links.new(coords.outputs["Window"], split.inputs[0])
    links.new(split.outputs["X"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], background.inputs["Color"])


def add_labels(path: Path, camera: bpy.types.Object) -> int:
    """3D captions from a scene's `.rotulos.json`, facing the camera.

    Each entry is ``{"texto": str, "pos": [x, y, z], "tam": float,
    "color": [r, g, b]}`` in the scene's own coordinates and A. The
    text glows rather than being lit, so it reads the same wherever
    the lamps are, and it is a real object: move or retype it in
    Blender like any other.
    """
    if not path.exists():
        return 0
    import json
    entries = json.loads(path.read_text())
    for k, entry in enumerate(entries):
        bpy.ops.object.text_add(location=entry["pos"])
        text = bpy.context.active_object
        text.name = f"rotulo_{k}"
        text.data.body = entry["texto"]
        text.data.size = float(entry.get("tam", 12.0))
        text.data.align_x, text.data.align_y = "CENTER", "CENTER"
        text.data.extrude = 0.04 * text.data.size
        mat = bpy.data.materials.new(f"mat_rotulo_{k}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (
            *entry.get("color", (1.0, 0.85, 0.6)), 1.0)
        emission.inputs["Strength"].default_value = float(
            entry.get("brillo", 3.0))
        output = nodes.new("ShaderNodeOutputMaterial")
        mat.node_tree.links.new(emission.outputs[0],
                                output.inputs["Surface"])
        text.data.materials.append(mat)
        track = text.constraints.new("TRACK_TO")
        track.target = camera
        track.track_axis, track.up_axis = "TRACK_Z", "UP_Y"
    return len(entries)


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extxyz", type=Path, required=True)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--png", type=Path, default=None,
                        help="also render a still here")
    parser.add_argument("--style", default="core_shell_cover",
                        choices=[*style_defs.list_styles(), "revista"],
                        help="a package style, or `revista`: this "
                             "script's high-impact look, metals at vdW")
    parser.add_argument("--efecto", action="append", default=[],
                        help="an extra effect, as render_cnt takes it, "
                             "e.g. bokeh:color=cyan,count=40 (repeatable)")
    parser.add_argument("--view", default="auto")
    parser.add_argument("--elev", type=float, default=None,
                        help="camera elevation in degrees; with --azim, "
                             "overrides --view. On a floor this sets "
                             "where the horizon falls: higher puts it "
                             "above the subject")
    parser.add_argument("--lente", type=float, default=None,
                        help="lens in mm, overriding the style's. Short "
                             "(24-35) brings the camera close, so what is "
                             "in front grows against what is behind")
    parser.add_argument("--bola", type=int, nargs="+", default=None,
                        help="`grupo` values drawn ball-and-stick whatever "
                             "their bonds, e.g. a cloud of precursor "
                             "molecules whose iron has no stick")
    parser.add_argument("--ampliar", type=float, default=1.0,
                        help="factor by which the `--bola` groups were "
                             "magnified about each molecule's own centre "
                             "in the scene: their bonds are found at true "
                             "scale and their balls and sticks drawn this "
                             "much thicker, so a magnified molecule reads "
                             "as a molecule and not as a stretched one")
    parser.add_argument("--paleta", default="revista", choices=list(PALETTES),
                        help="materials for `--style revista`: revista, "
                             "neon (glowing bonds), oro (black and gold), "
                             "iridiscente (thin-film silver)")
    parser.add_argument("--calor", action="store_true",
                        help="two coloured side lights, warm amber from the "
                             "left and white-hot blue from the right: "
                             "a lower and a higher temperature, shown and "
                             "not written")
    parser.add_argument("--invertir-calor", dest="invertir_calor",
                        action="store_true",
                        help="with --calor: cold on the left and warm on "
                             "the right")
    parser.add_argument("--se-grande", dest="se_grande", type=float,
                        default=1.0,
                        help="extra ball factor for bonded Se (on top of "
                             "--dopantes), to make the Se sites stand out")
    parser.add_argument("--dopantes", type=float, default=1.0,
                        help="ball radius factor for bonded N, O, H and Se "
                             "outside the `--bola` groups: the dopants and "
                             "grafted groups of a wall, which at cover "
                             "scale are otherwise too small to see")
    parser.add_argument("--encuadrar", type=int, nargs="+", default=None,
                        help="frame only the atoms whose `grupo` is one of "
                             "these, e.g. the hero of a field; the rest of "
                             "the scene stays where it is, round and behind")
    parser.add_argument("--azim", type=float, default=0.0,
                        help="camera azimuth in degrees from the front")
    parser.add_argument("--fill", type=float, default=0.95,
                        help="fraction of the frame the scene takes")
    parser.add_argument("--resolution", type=int, nargs=2,
                        default=[1600, 2000])
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--atom-scale", type=float, default=0.55)
    parser.add_argument("--bond-scale", type=float, default=0.5)
    parser.add_argument("--no-dof", dest="dof", action="store_false")
    parser.add_argument("--desenfoque", type=float, default=1.0,
                        help="depth-of-field strength: 1 blurs a point one "
                             "scene radius behind the focus by 1/1500 of "
                             "the frame; 4 makes a background soft")
    parser.add_argument("--floor", action="store_true",
                        help="keep the style's glossy floor. Off by "
                             "default for the package styles, on for "
                             "`revista`")
    parser.add_argument("--sin-piso", dest="no_floor", action="store_true",
                        help="no floor, even for `revista`")
    args = parser.parse_args(argv)

    symbols, positions, groups = read_extxyz(args.extxyz)
    symbols = np.asarray(symbols)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    revista = args.style == "revista"
    style = (revista_style(style_defs.get_style("core_shell_cover"))
             if revista else style_defs.get_style(args.style))

    # --- bonds first, because they decide how an atom is drawn: one
    # that is bonded to something belongs to a molecule or a lattice
    # and is drawn as a ball; one with no bond is part of a crystal core
    # and fills its van der Waals sphere. That is what lets a selenium
    # dopant in a tube wall be a small ball while the selenium of the
    # FeSe core beside it is a solid surface.
    palette = PALETTES[args.paleta]
    zoom = float(args.ampliar)
    magnified = (np.isin(groups, args.bola) if args.bola and zoom != 1.0
                 else np.zeros(len(positions), dtype=bool))
    rest = np.flatnonzero(~magnified)
    pairs, classes = all_bonds(symbols[rest], positions[rest])
    pairs = rest[pairs] if len(pairs) else pairs
    if magnified.any():
        # Found at true scale: the scene keeps magnified molecules at
        # least `zoom` times a contact apart, so dividing by `zoom`
        # brings each one back to its own bond lengths without bringing
        # two of them into bonding distance.
        big = np.flatnonzero(magnified)
        more, more_classes = all_bonds(symbols[big], positions[big] / zoom)
        if len(more):
            pairs = np.vstack((pairs.reshape(-1, 2), big[more]))
            classes = np.concatenate((classes, more_classes + "_amp"))
    bonded = np.zeros(len(positions), dtype=bool)
    bonded[pairs.ravel()] = True
    if args.bola:
        bonded |= np.isin(groups, args.bola)

    # --- atoms: one carrier of vertices per (element, radius).
    present = [key for key in ELEMENT_ORDER if (symbols == key).any()]
    radii = VDW_RADIUS if revista else SOLID_RADIUS
    carriers = []
    for key in present:
        spec = (palette.get(key, palette["C"])
                if revista else style_defs.materials_for(
                    style, "element", [key], "style")[0])
        mat = render_cnt.material_from_spec(f"mat_element_{key}", spec)
        ball_size = (BALL_RADIUS.get(key, 0.45) if key in BALL_RADIUS
                     else style_defs.element_radius(key) * args.atom_scale
                     * style.atom_scale)
        if key == "C":
            ball_size = (style_defs.element_radius(key) * args.atom_scale
                         * style.atom_scale)
        solid_size = radii.get(key, ball_size)
        for tag, size, mask in (("", solid_size,
                                 (symbols == key) & ~bonded),
                                ("_bola", ball_size * (
                                    args.dopantes if key != "C" else 1.0)
                                 * (args.se_grande if key == "Se" else 1.0),
                                 (symbols == key) & bonded & ~magnified),
                                ("_amp", ball_size * zoom,
                                 (symbols == key) & magnified)):
            if not mask.any():
                continue
            ball = primitive(
                f"esfera_{key}{tag}",
                lambda size=size: bpy.ops.mesh.primitive_ico_sphere_add(
                    subdivisions=2 if size < 1.0 else 3, radius=size),
                mat)
            carriers.append(instanced(f"atomos_{key}{tag}", positions[mask],
                                      [], ball, "VERTS"))

    # --- bonds: one tiny triangle per pair, normal along the bond, and
    # one carrier per length class so every stick is its bond's length.
    bond_specs, _split = style_defs.bond_materials(
        style, "element", ["C"], "auto", "style")
    bond_mat = render_cnt.material_from_spec(
        "mat_bond", palette["bond"] if revista else bond_specs[0])
    for label in np.unique(classes):
        chosen = pairs[classes == label]
        start, end = positions[chosen[:, 0]], positions[chosen[:, 1]]
        axis = end - start
        length = np.linalg.norm(axis, axis=1)
        axis /= length[:, None]
        helper = np.where(np.abs(axis[:, :1]) < 0.9, [[1.0, 0.0, 0.0]],
                          [[0.0, 1.0, 0.0]])
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u, axis=1)[:, None]
        v = np.cross(axis, u)
        middle = 0.5 * (start + end)
        corners = np.empty((len(chosen) * 3, 3))
        for k, angle in enumerate((0.0, 2.0 * np.pi / 3.0,
                                   4.0 * np.pi / 3.0)):
            corners[k::3] = middle + 0.05 * (np.cos(angle) * u
                                             + np.sin(angle) * v)
        faces = [(3 * k, 3 * k + 1, 3 * k + 2) for k in range(len(chosen))]
        depth = float(np.median(length))
        radius = 0.12 * args.bond_scale * style.bond_scale * (
            zoom if str(label).endswith("_amp") else 1.0)

        def cylinder(depth=depth, radius=radius):
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=10, radius=radius, depth=depth,
                end_fill_type="NOTHING")

        stick = primitive(f"enlace_{label}", cylinder, bond_mat)
        carriers.append(instanced(f"enlaces_{label}", corners, faces,
                                  stick, "FACES"))

    # --- the studio: the style's own, placed by the pipeline's own
    # functions against this scene's size.
    bpy.context.view_layer.update()
    center, radius = render_cnt._bounding_sphere(carriers)
    render_cnt._setup_world(style.background)
    if args.calor:
        calor_backdrop(args.invertir_calor)
    if args.elev is not None:
        elev, azim = np.radians(args.elev), np.radians(args.azim)
        render_cnt.camera.NAMED_VIEWS["portada"] = (
            float(np.cos(elev) * np.sin(azim)),
            float(-np.cos(elev) * np.cos(azim)), float(np.sin(elev)))
        args.view = "portada"
    use_floor = (args.floor or revista) and not args.no_floor
    if use_floor and args.view == "auto":
        # A floor wants the camera low. Seen from above, the auto view's
        # usual pick, it fills the frame and mirrors the light rig.
        args.view = "hero"
    if use_floor and style.background.has_floor:
        render_cnt._add_floor(style.background, center, radius,
                              bottom=float(positions[:, 2].min()) - 1.5)
    render_cnt._setup_lights(style, center, radius)
    if args.calor:
        # One spot per half, each on the products of its own side (the
        # lowest 40 % of the atoms, split at the middle): amber on the
        # left, white-hot blue on the right. Two broad lamps from the
        # sides were tried first and lit everything with both colours,
        # which summed to white.
        low = positions[:, 2] < np.quantile(positions[:, 2], 0.40)
        middle_x = float(np.median(positions[low, 0]))
        hues = ((1.0, 0.50, 0.12), (0.55, 0.80, 1.0))
        if args.invertir_calor:
            hues = hues[::-1]
        for name, side, rgb in (("calor_izq", -1.0, hues[0]),
                                ("calor_der", 1.0, hues[1])):
            half = low & ((positions[:, 0] - middle_x) * side > 0)
            target = Vector(positions[half].mean(axis=0))
            data = bpy.data.lights.new(name, "SPOT")
            data.spot_size = np.radians(32.0)
            data.spot_blend = 0.6
            data.color = rgb
            data.shadow_soft_size = 0.15 * radius
            lamp = link(bpy.data.objects.new(name, data))
            lamp.location = target + Vector((side * 0.9 * radius,
                                             -1.6 * radius, 0.9 * radius))
            reach = (target - lamp.location).length
            data.energy = 60.0 * reach ** 2
            aim = target - lamp.location
            lamp.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
    # The pipeline's floor is FLOOR_EXTENT radii across, so its far edge
    # shows as a horizon line through the middle of the frame. Stretched
    # far past the camera's reach, the edge goes to infinity and the
    # floor fades into the field. (Hiding the lamps from reflections, to
    # keep them out of the mirror, was tried and undone: a metal's whole
    # brightness IS its reflection of the lamps, and the scene went dull.)
    floor_obj = bpy.data.objects.get("Floor")
    if floor_obj is not None:
        floor_obj.scale = (60.0, 60.0, 1.0)
        # Light linking: every lamp lights the subject and NOT the
        # floor, so the mirror gives back the scene and never a lamp.
        # Without it each lamp came back as a white rectangle under the
        # subject -- the back light even with the anemone's silhouette
        # in it -- and moving or hiding lamps from reflections either
        # only moved the rectangle or left the metal dull, since a
        # metal's highlights ARE its reflection of the lamps. The
        # collection holds the floor alone, marked EXCLUDE, so
        # everything else stays lit.
        excluded = bpy.data.collections.new("sin_luz_directa")
        excluded.objects.link(floor_obj)
        excluded.collection_objects[0].light_linking.link_state = "EXCLUDE"
        for obj in bpy.context.scene.objects:
            if obj.type == "LIGHT":
                obj.light_linking.receiver_collection = excluded
    centre, framed = render_cnt._setup_camera(
        style, carriers, args.lente or style.camera_lens_mm, args.resolution,
        view=args.view, fill=args.fill,
        points=(positions[np.isin(groups, args.encuadrar)]
                if args.encuadrar else positions))
    if args.encuadrar:
        # The clip range was fitted to the framed atoms alone; widen it
        # to the whole scene, or everything behind the hero is cut off.
        cam = bpy.context.scene.camera
        reach = float(np.linalg.norm(
            positions - np.asarray(cam.location), axis=1).max())
        cam.data.clip_start = min(cam.data.clip_start, 1.0)
        cam.data.clip_end = max(cam.data.clip_end, 1.2 * reach)
    labels = add_labels(args.extxyz.with_suffix(".rotulos.json"),
                        bpy.context.scene.camera)
    specs = list(style.effects) + [fx.parse_effect(text)
                                   for text in args.efecto]
    fx.build_effects(specs, centre, framed, positions)
    render_cnt._setup_render(style, SimpleNamespace(
        engine="cycles", resolution=args.resolution,
        transparent_background=False, samples=args.samples,
        out=str(args.png or args.blend.with_suffix(".png"))))
    scene = bpy.context.scene
    scene.cycles.use_denoising = True
    try:
        scene.cycles.device = "CPU" if not render_cnt._can_rasterise() \
            else scene.cycles.device
    except Exception:
        pass

    camera = scene.camera
    if args.dof and camera is not None:
        # Blender reads 1 unit as 1 m, so a 500-unit model is a mountain
        # and a real f-number leaves it all sharp. Solve for the
        # f-number that blurs a point one scene radius behind the focus
        # by 1/1500 of the frame: the far arms soften, nothing smears.
        target = link(bpy.data.objects.new("foco", None))
        # Focus on what is framed: the bounding centre of a scene with a
        # hero in front and a backdrop behind falls between the two, and
        # left the backdrop as sharp as the hero.
        target.location = (Vector(positions[np.isin(groups, args.encuadrar)]
                                  .mean(axis=0)) if args.encuadrar
                           else center)
        distance = float((camera.location - center).length)
        focal = camera.data.lens / 1000.0
        blur = (camera.data.sensor_width / 1000.0) / 1500.0 * args.desenfoque
        camera.data.dof.use_dof = True
        camera.data.dof.focus_object = target
        camera.data.dof.aperture_fstop = max(
            focal * focal / (blur * distance) * radius / (distance + radius),
            1e-4)

    args.blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()),
                                compress=True)
    print(f"{args.blend}: {len(positions)} átomos, {len(pairs)} enlaces "
          f"entre no metales, {labels} rótulos, estilo {args.style}, "
          f"radio {radius:.0f} Å")
    if args.png is not None:
        bpy.ops.render.render(write_still=True)
        for spec in specs:
            if spec.kind == "glare":
                fx.apply_glare(str(args.png.resolve()), spec)
        print(f"{args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
