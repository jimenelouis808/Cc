"""Scene effects: beams, arcs, resonances, haloes, fog and glare.

The part of a cover image that is **not** the structure. A journal cover
is a picture of an idea, not an elevation drawing, and the idea is
usually something happening *to* the structure -- a laser on it, a field
around it, a resonance in it. None of that is chemistry and none of it is
measured: an effect is decoration, it is parametrised so it can be
tuned, and it is never confused with the structure's own geometry.

**No ``bpy`` at module level.** The specs, their defaults, the parser and
the geometry of an arc are all arithmetic, so they are importable and
testable in an ordinary interpreter; ``bpy`` is imported inside the
builders, which only run inside Blender. That is the same split
:mod:`styles` and :mod:`bundle` already follow.

Everything is expressed in **subject radii**, never in Ångström. A beam
of "reach 6" is six subject radii long whether the subject is a 3 Å cage
or a 90 Å network, so an effect means the same thing at every size --
the property the light rig already has and for the same reason.

An effect is written on the command line as ``name:key=value,key=value``:

    --effect "laser:color=#30ffa0,strength=80,azimuth=120"
    --effect arc:count=10,color=electric
    --effect glare:strength=0.5
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

RGB = tuple[float, float, float]

__all__ = ["EFFECTS", "EFFECT_DEFAULTS", "EffectSpec", "NAMED_COLOURS",
           "apply_glare", "arc_polyline", "build_effects",
           "describe_effects", "glare_image", "parse_effect",
           "fractal_offsets", "parse_colour", "surface_along",
           "taper_profile"]

#: Colours worth having a name for, so a command line is readable.
NAMED_COLOURS: dict[str, RGB] = {
    "laser-green": (0.10, 1.00, 0.35),
    "laser-red": (1.00, 0.08, 0.10),
    "laser-blue": (0.15, 0.45, 1.00),
    "electric": (0.45, 0.75, 1.00),
    "plasma": (0.75, 0.35, 1.00),
    "amber": (1.00, 0.62, 0.15),
    "cyan": (0.10, 0.90, 1.00),
    "magenta": (1.00, 0.15, 0.70),
    "white": (1.00, 1.00, 1.00),
    "gold": (1.00, 0.76, 0.30),
    "ultraviolet": (0.55, 0.30, 1.00),
    "sodium": (1.00, 0.80, 0.42),
}


def parse_colour(text: str | tuple) -> RGB:
    """A colour from a name, a ``#rrggbb`` or three slash-separated numbers.

    Slashes rather than commas because a comma already separates one
    effect parameter from the next, and a colour written ``1,0,0`` would
    be read as three parameters with no value.
    """
    if not isinstance(text, str):
        values = tuple(float(value) for value in text)
        return (values[0], values[1], values[2])
    text = text.strip()
    if text in NAMED_COLOURS:
        return NAMED_COLOURS[text]
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) != 6:
            raise ValueError(f"{text!r} is not a #rrggbb colour.")
        return tuple(int(digits[i:i + 2], 16) / 255.0
                     for i in (0, 2, 4))      # type: ignore[return-value]
    if "/" in text:
        parts = [float(value) for value in text.split("/")]
        if len(parts) != 3:
            raise ValueError(f"{text!r} needs three numbers, r/g/b.")
        return (parts[0], parts[1], parts[2])
    raise ValueError(
        f"unknown colour {text!r}. Use a name ({', '.join(sorted(NAMED_COLOURS))}), "
        "a #rrggbb, or r/g/b."
    )


#: Every effect, and what each parameter means. Distances are in
#: **subject radii** and angles in degrees.
EFFECT_DEFAULTS: dict[str, dict] = {
    # A beam arriving at the structure, and the spot where it lands.
    # `spread` > 0 fans several beams out of one source. This is the one
    # effect that lights the subject by default: a beam that illuminates
    # nothing is a line drawn over the picture.
    "laser": {
        "color": "laser-green", "strength": 55.0, "width": 0.016,
        "reach": 7.0, "azimuth": 135.0, "elevation": 22.0,
        "count": 1, "spread": 0.0, "spot": 0.10, "stop": 0.85,
        "sheath": 4.0, "sparks": 10.0, "light": 1.0, "seed": 0,
    },
    # Lightning: jagged emissive filaments arcing around the subject.
    "arc": {
        "color": "electric", "strength": 9.0, "width": 0.0030,
        "count": 6, "reach": 1.12, "jitter": 0.30, "segments": 64,
        "branch": 0.45, "span": 55.0, "sheath": 3.2, "spread": 0.55,
        "light": 0.0, "seed": 0,
    },
    # Standing-wave shells: concentric rings around the structure, each
    # fainter than the last, for a vibrational or plasmonic resonance.
    "resonance": {
        "color": "cyan", "strength": 7.0, "width": 0.0035,
        "count": 4, "reach": 1.30, "spacing": 0.22, "tilt": 20.0,
        "falloff": 0.55, "segments": 96, "light": 0.0, "seed": 0,
    },
    # Orbit rings, like a textbook atom: wide thin ellipses at angles.
    "orbit": {
        "color": "amber", "strength": 8.0, "width": 0.004,
        "count": 3, "reach": 1.5, "tilt": 62.0, "segments": 128,
        "light": 0.0, "seed": 0,
    },
    # A soft shell of light around the subject.
    # A VOLUME, not a shell: a surface has a silhouette and reads as a
    # disc however it is tuned. `density` is how much of it there is to
    # look through and `strength` how brightly it emits.
    "halo": {
        "color": "electric", "strength": 0.26, "reach": 1.6,
        "density": 0.008, "light": 0.0, "seed": 0,
    },
    # World volume scatter: haze, so beams and lights become visible as
    # shafts rather than only lighting what they hit. Easy to overdo --
    # the density is small on purpose.
    "fog": {
        "color": "white", "strength": 0.0025, "anisotropy": 0.3,
    },
    # Post-processing bloom, run over the finished image rather than in
    # the scene, which is why it costs nothing in sample count. `size`
    # is a fraction of the image's short side.
    "glare": {
        "strength": 0.35, "size": 0.5, "threshold": 0.75, "maximum": 3.0,
    },
}

EFFECTS = tuple(EFFECT_DEFAULTS)


@dataclass(frozen=True)
class EffectSpec:
    """One effect and its parameters, already validated."""

    kind: str
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EFFECT_DEFAULTS:
            raise ValueError(
                f"unknown effect {self.kind!r}; known: {', '.join(EFFECTS)}")
        unknown = set(self.params) - set(EFFECT_DEFAULTS[self.kind])
        if unknown:
            raise ValueError(
                f"{self.kind}: unknown parameter(s) {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(EFFECT_DEFAULTS[self.kind]))}."
            )

    def get(self, name: str):
        """A parameter, falling back to the effect's own default."""
        return self.params.get(name, EFFECT_DEFAULTS[self.kind][name])

    def colour(self) -> RGB:
        return parse_colour(self.get("color"))


def parse_effect(text: str) -> EffectSpec:
    """``"laser:strength=80,color=cyan"`` to an :class:`EffectSpec`.

    Values are kept as floats where they parse as numbers and as strings
    otherwise, so ``color=cyan`` and ``type=STREAKS`` survive while every
    numeric parameter arrives as a number.
    """
    name, _, rest = text.partition(":")
    name = name.strip()
    params: dict = {}
    for chunk in rest.split(",") if rest.strip() else []:
        key, sep, value = chunk.partition("=")
        if not sep:
            raise ValueError(
                f"{text!r}: expected key=value, got {chunk!r}.")
        key, value = key.strip(), value.strip()
        try:
            params[key] = float(value)
        except ValueError:
            params[key] = value
    return EffectSpec(kind=name, params=params)


def describe_effects() -> str:
    """The catalogue and every parameter, for ``--list-effects``."""
    lines = []
    for name, defaults in EFFECT_DEFAULTS.items():
        pairs = ", ".join(f"{key}={value}" for key, value in defaults.items())
        lines.append(f"{name}\n    {pairs}")
    lines.append("\nDistances are in SUBJECT RADII, not Angstrom, so an "
                 "effect means the same thing on a 3 A cage and a 90 A "
                 "network.")
    lines.append("Colours: a name (" + ", ".join(sorted(NAMED_COLOURS))
                 + "), a #rrggbb, or r/g/b with slashes.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# geometry that needs no Blender
# ----------------------------------------------------------------------
def fractal_offsets(count: int, rng, roughness: float = 0.62
                    ) -> np.ndarray:
    """Self-similar 2D displacements for ``count`` points, ends at zero.

    **Midpoint displacement**, the standard way lightning is generated:
    set the midpoint of a segment to the mean of its ends plus a random
    kick, halve the kick, and recurse. What comes out has large bends
    with smaller kinks on them and smaller ones on those -- the
    self-similar structure a discharge actually has.

    The first version used a **random walk** instead, and the difference
    is not subtle: a walk's displacement is the sum of its steps, so it
    drifts slowly and smoothly and the result is a meander. Drawn as a
    filament it reads as a wire someone bent, which is exactly what the
    arcs looked like.

    ``roughness`` below 0.5 gives a nearly straight line with fine fuzz,
    above 0.75 a tangle that crosses itself. The ends are pinned to zero
    so an arc stays attached to whatever it is striking.
    """
    count = max(2, int(count))
    # Midpoint displacement needs 2^k + 1 points; the result is sampled
    # back down to the length asked for.
    levels = max(1, math.ceil(math.log2(max(2, count - 1))))
    size = 2 ** levels + 1
    out = np.zeros((size, 2))
    step = size - 1
    scale = 1.0
    while step > 1:
        half = step // 2
        for start in range(0, size - 1, step):
            middle = start + half
            out[middle] = (0.5 * (out[start] + out[start + step])
                           + rng.normal(0.0, scale, size=2))
        step = half
        scale *= roughness
    if size != count:
        source = np.linspace(0.0, 1.0, size)
        target = np.linspace(0.0, 1.0, count)
        out = np.column_stack([np.interp(target, source, out[:, axis])
                               for axis in (0, 1)])
    # Normalised, so `jitter` means the same thing at every segment
    # count -- otherwise a finer arc is also a wilder one.
    peak = float(np.abs(out).max())
    return out / peak if peak > 1e-12 else out


def arc_polyline(start: np.ndarray, end: np.ndarray, segments: int,
                 jitter: float, scale: float, rng) -> np.ndarray:
    """A jagged path from ``start`` to ``end``, as lightning draws it.

    The displacement is **perpendicular to the path** and tapers to zero
    at both ends, so the arc wanders without its endpoints moving -- an
    arc whose ends drift is an arc that has come off the thing it is
    supposed to be striking.

    The shape of the wander is :func:`fractal_offsets`; see there for
    why it is not a random walk.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-9 or segments < 2:          # pragma: no cover - degenerate
        return np.vstack([start, end])
    axis = direction / length
    # Any two vectors perpendicular to the path and to each other.
    seed_vector = (np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9
                   else np.array([1.0, 0.0, 0.0]))
    u = np.cross(axis, seed_vector)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    t = np.linspace(0.0, 1.0, segments + 1)
    taper = np.sin(math.pi * t)[:, None]
    offset = fractal_offsets(segments + 1, rng) * taper * jitter * scale
    spine = start + np.outer(t, direction)
    return spine + offset[:, 0:1] * u + offset[:, 1:2] * v


def sphere_points(count: int, radius: float, rng) -> np.ndarray:
    """``count`` points spread quasi-uniformly on a sphere of ``radius``."""
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    out = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * index + phase
        out.append((r * math.cos(theta) * radius,
                    r * math.sin(theta) * radius, z * radius))
    return np.asarray(out, dtype=float)


def direction_from_angles(azimuth_deg: float, elevation_deg: float
                          ) -> np.ndarray:
    """A unit vector from an azimuth in the xy plane and an elevation."""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.array([
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    ])


# ----------------------------------------------------------------------
# builders -- these run inside Blender
# ----------------------------------------------------------------------
def _output_node(nodes, kind: str = "ShaderNodeOutputMaterial"):
    """The tree's output node, found by type rather than by name."""
    found = next((node for node in nodes if node.bl_idname == kind), None)
    return found if found is not None else nodes.new(kind)


def _fresh_tree(nodes, kind: str = "ShaderNodeOutputMaterial"):
    """Empty the node tree and return a brand-new output node.

    Not "keep the output and delete the rest", which is the obvious
    version and is wrong: ``nodes.remove()`` can reallocate the
    collection, so a reference taken *before* the removals can be left
    dangling. It does not raise on access -- it comes back as a node
    whose input sockets are missing, and the failure surfaces as
    ``KeyError: 'Surface'`` on a node that plainly has a Surface input.
    Clearing everything and making a new output cannot go stale.
    """
    nodes.clear()
    return nodes.new(kind)


def _emissive(name: str, colour: RGB, strength: float, alpha: float = 1.0):
    """A pure-emission material. Imports ``bpy`` only when called."""
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    out = _fresh_tree(nodes)
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*colour, 1.0)
    emission.inputs["Strength"].default_value = float(strength)
    if alpha >= 1.0:
        links.new(emission.outputs[0], out.inputs["Surface"])
        return material
    # A halo is a shell you see through: mixing emission against a
    # transparent BSDF is what makes it glow without hiding the
    # structure inside it.
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs["Fac"].default_value = float(alpha)
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs["Surface"])
    material.blend_method = "BLEND" if hasattr(material, "blend_method") else \
        material.blend_method
    return material


def _set_light_contribution(obj, lights: bool) -> None:
    """Whether an emissive object lights the scene or merely glows.

    Cycles treats emission as a light source, so a dozen bright arcs
    around a structure do not only *look* like lightning -- they light
    everything, and with any haze in the air they wash the whole frame
    out. Turning off the diffuse, glossy and scatter rays leaves the
    object visible to the camera and invisible as a lamp, which is what
    most of these effects want: an arc is drawn *over* the picture, a
    laser is *in* it.
    """
    for flag in ("visible_diffuse", "visible_glossy", "visible_volume_scatter",
                 "visible_transmission", "visible_shadow"):
        if hasattr(obj, flag):
            setattr(obj, flag, bool(lights))


def taper_profile(count: int, kind: str = "none") -> np.ndarray:
    """Per-point radius multipliers along a path.

    A filament of constant width is the single thing that makes an
    effect read as a drawn line rather than as light. Real discharges
    and beams are thickest somewhere and thin out, so every stroke here
    carries a profile:

    ``none``    flat -- for a ring, where constant width is correct.
    ``ends``    thin at both ends, fat in the middle: an arc that
                detaches from its anchors instead of butting into them.
    ``taper``   fat at the start, thin at the end: a branch, a spark.
    ``beam``    nearly flat with a slight waist, for a laser core.
    """
    t = np.linspace(0.0, 1.0, max(2, count))
    if kind == "ends":
        # sin is zero at both ends; the floor keeps the stroke from
        # vanishing entirely, which would leave visible gaps.
        return 0.18 + 0.82 * np.sin(math.pi * t)
    if kind == "taper":
        return 1.0 - 0.9 * t
    if kind == "beam":
        return 0.85 + 0.15 * np.cos(2.0 * math.pi * t)
    return np.ones_like(t)


def _polyline_object(name: str, paths, radius: float, material,
                     taper: str = "none", weights=None):
    """One mesh of thin tubes following each path in ``paths``.

    Built as a curve with a bevel rather than as thousands of cylinders:
    a 26-segment arc times ten arcs is 260 joints, and a curve carries
    them as one object with round caps for free.

    The bevel is modulated per point through the spline's **radius**
    channel, which is what ``taper`` and ``weights`` drive -- a curve
    whose width varies is the difference between a filament and a wire.
    ``weights`` scales each whole path, so a population of arcs can vary
    in brightness-by-thickness rather than being identical strokes.
    """
    import bpy

    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = float(radius)
    curve.bevel_resolution = 3
    curve.use_fill_caps = True
    for index, path in enumerate(paths):
        points = np.asarray(path, dtype=float)
        if len(points) < 2:
            continue
        spline = curve.splines.new("POLY")
        spline.points.add(len(points) - 1)
        profile = taper_profile(len(points), taper)
        scale = 1.0 if weights is None else float(weights[index])
        for position, point, width in zip(
                range(len(points)), points, profile, strict=True):
            spline.points[position].co = (*point, 1.0)
            spline.points[position].radius = float(width * scale)
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    return obj


def _glow_pair(name: str, paths, radius: float, colour: RGB,
               strength: float, taper: str = "none", weights=None,
               sheath: float = 3.2, sheath_alpha: float = 0.13) -> list:
    """A bright core inside a wide, faint sheath.

    One emissive tube is a glowing wire. What makes a discharge or a
    beam look like *light* is that it has a hot thin core and a soft
    halo around it that the eye reads as air lit up -- two objects, not
    one, because a single material cannot be both opaque at the centre
    and nearly transparent at the edge.

    The sheath is mostly transparent and its emission is scaled down in
    proportion, so widening it spreads the glow rather than adding more
    total light.
    """
    # The core is brighter than the halo but must keep the hue: lifting
    # it most of the way to white, which was the first version, turned
    # every coloured filament into a white wire and threw away the one
    # thing the colour parameter controls.
    core_colour = tuple(min(1.0, channel * 0.75 + 0.25) for channel in colour)
    core = _polyline_object(
        f"{name}_core", paths, radius,
        _emissive(f"{name}_core_mat", core_colour, strength),
        taper, weights)
    halo = _polyline_object(
        f"{name}_glow", paths, radius * sheath,
        _emissive(f"{name}_glow_mat", colour, strength * 0.5,
                  alpha=sheath_alpha),
        taper, weights)
    return [core, halo]


def _ico(name: str, centre, radius: float, material, subdivisions: int = 3):
    import bmesh
    import bpy
    from mathutils import Matrix

    bm = bmesh.new()
    bmesh.ops.create_icosphere(
        bm, subdivisions=subdivisions, radius=float(radius),
        matrix=Matrix.Translation(tuple(centre)), calc_uvs=False)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    mesh.materials.append(material)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def _ring_path(centre, radius: float, normal, segments: int) -> np.ndarray:
    """A closed circle of ``radius`` about ``centre`` in a plane."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    seed_vector = (np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9
                   else np.array([1.0, 0.0, 0.0]))
    u = np.cross(normal, seed_vector)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angles = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    return (np.asarray(centre, dtype=float)
            + radius * (np.cos(angles)[:, None] * u
                        + np.sin(angles)[:, None] * v))


def surface_along(points, centre, direction, quantile: float = 0.98
                  ) -> float | None:
    """How far the structure reaches from ``centre`` along ``direction``.

    A beam has to stop where the structure actually is, and "a fraction
    of the radius" is only right for something round. On a capped
    nanotube the framed radius is its half-*length* while the tube is a
    few Ångström across, so a beam aimed at ``0.85 * radius`` stopped in
    mid-air well to the side of it -- a glowing spot next to the
    subject, which is worse than no beam.

    Projecting the atoms onto the beam direction and taking a high
    quantile gives the real surface along that line. A quantile rather
    than the maximum because a single outlying atom -- a grafted group,
    a stray cap atom -- should not push the impact point off the body.

    Returns ``None`` when there are no points to measure, so the caller
    can fall back to the radius.
    """
    if points is None or len(points) == 0:
        return None
    points = np.asarray(points, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1e-12)
    along = (points - np.asarray(centre, dtype=float)) @ direction
    return float(np.quantile(along, quantile))


def _build_laser(spec: EffectSpec, centre, radius: float,
                 points=None) -> list:
    """A beam arriving at the structure, and the spot where it lands."""
    rng = np.random.default_rng(int(spec.get("seed")))
    colour = spec.colour()
    strength = float(spec.get("strength"))
    width = float(spec.get("width")) * radius
    reach = float(spec.get("reach")) * radius
    count = max(1, int(spec.get("count")))
    spread = float(spec.get("spread"))

    base = direction_from_angles(float(spec.get("azimuth")),
                                 float(spec.get("elevation")))
    centre = np.asarray(centre, dtype=float)
    paths = []
    targets = []
    for index in range(count):
        direction = base
        if spread > 0 and count > 1:
            # Fan the beams about the base direction rather than aiming
            # them randomly: a fan reads as one source, random aim reads
            # as a mistake.
            tilt = math.radians(spread) * (index / (count - 1) - 0.5)
            axis = np.cross(base, [0.0, 0.0, 1.0])
            axis = (axis / np.linalg.norm(axis)
                    if np.linalg.norm(axis) > 1e-9 else np.array([1.0, 0, 0]))
            direction = (base * math.cos(tilt)
                         + np.cross(axis, base) * math.sin(tilt))
        # The beam stops at the SURFACE, not at the centre. Aimed at
        # the centre it passes straight through the structure and comes
        # out of the far side, which reads as a line drawn over the
        # picture rather than as something striking it. Measured along
        # the beam where the atoms are known, because an isotropic
        # radius is only right for something round.
        reached = surface_along(points, centre, direction)
        stop = (float(spec.get("stop"))
                * (reached if reached is not None else radius))
        target = (centre + direction * stop
                  + rng.normal(0.0, 0.10 * radius, size=3) * (count > 1))
        paths.append(np.vstack([target + direction * reach, target]))
        targets.append(target)
    # A hot white core inside a coloured halo, with a slight waist: one
    # flat emissive cylinder is a drawn line, and this is the difference
    # between that and a beam.
    objects = _glow_pair("fx_laser", paths, width, colour, strength,
                         taper="beam", sheath=float(spec.get("sheath")))
    lights = bool(spec.get("light"))

    spot = float(spec.get("spot"))
    if spot > 0:
        hot = _emissive("fx_laser_spot", (1.0, 1.0, 1.0), strength * 2.0)
        bloomy = _emissive("fx_laser_flash", colour, strength * 0.6,
                           alpha=0.10)
        for index, target in enumerate(targets):
            objects.append(
                _ico(f"fx_laser_spot{index}", target, spot * radius * 0.5,
                     hot, subdivisions=2))
            # A flash around the impact, which is what sells it as
            # something landing rather than a dot pasted on.
            objects.append(
                _ico(f"fx_laser_flash{index}", target, spot * radius * 1.8,
                     bloomy, subdivisions=3))

    # Sparks thrown off the impact point: short tapered filaments, so
    # the beam has a consequence instead of simply ending.
    count_sparks = int(spec.get("sparks"))
    if count_sparks > 0 and targets:
        spark_paths = []
        for target in targets:
            for direction in sphere_points(count_sparks, 1.0, rng):
                length = float(rng.uniform(0.10, 0.34)) * radius
                tip = target + direction * length
                spark_paths.append(arc_polyline(target, tip, 5, 0.25,
                                                radius * 0.4, rng))
        objects += _glow_pair("fx_spark", spark_paths, width * 0.45,
                              colour, strength * 0.7, taper="taper",
                              sheath=2.4)
    for obj in objects:
        _set_light_contribution(obj, lights)
    return objects


def _build_arc(spec: EffectSpec, centre, radius: float) -> list:
    """Jagged filaments arcing around the structure."""
    rng = np.random.default_rng(int(spec.get("seed")))
    count = max(1, int(spec.get("count")))
    reach = float(spec.get("reach")) * radius
    segments = max(2, int(spec.get("segments")))
    jitter = float(spec.get("jitter"))
    branch = float(spec.get("branch"))
    centre = np.asarray(centre, dtype=float)

    # Each arc joins a point to a NEARBY one, not to a random partner.
    # Random pairs on a sphere are antipodal on average, so every arc
    # came out as long as the structure is wide -- on a composition that
    # is a zigzag right across the frame rather than a discharge round
    # the subject. `span` caps the angle between the two ends, which is
    # what makes an arc hug the surface.
    span = math.radians(float(spec.get("span")))
    anchors = sphere_points(count, reach, rng) + centre
    paths = []
    for index in range(count):
        start = anchors[index]
        radial = (start - centre) / max(np.linalg.norm(start - centre), 1e-9)
        seed_vector = (np.array([0.0, 0.0, 1.0]) if abs(radial[2]) < 0.9
                       else np.array([1.0, 0.0, 0.0]))
        tangent = np.cross(radial, seed_vector)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(radial, tangent)
        turn = float(rng.uniform(0.0, 2.0 * math.pi))
        swing = float(rng.uniform(0.35, 1.0)) * span
        axis = math.cos(turn) * tangent + math.sin(turn) * bitangent
        end = centre + reach * (math.cos(swing) * radial
                                + math.sin(swing) * axis)
        path = arc_polyline(start, end, segments, jitter, radius, rng)
        paths.append(path)
        # A discharge forks. One branch per arc at most, leaving from a
        # random point along it and reaching a fraction of the way out.
        if branch > 0 and rng.random() < branch:
            cut = int(rng.integers(segments // 4, 3 * segments // 4))
            tip = path[cut] + rng.normal(0.0, 0.5 * radius, size=3)
            paths.append(arc_polyline(path[cut], tip, max(4, segments // 3),
                                      jitter * 1.4, radius, rng))
    # Filaments differ in brightness and thickness; a population of
    # identical strokes reads as a pattern, which is the opposite of a
    # discharge. `spread` is how much they are allowed to vary.
    vary = float(spec.get("spread"))
    weights = 1.0 - vary * rng.random(len(paths))
    objects = _glow_pair(
        "fx_arc", paths, float(spec.get("width")) * radius, spec.colour(),
        float(spec.get("strength")), taper="ends", weights=weights,
        sheath=float(spec.get("sheath")))
    for obj in objects:
        _set_light_contribution(obj, bool(spec.get("light")))
    return objects


def _build_resonance(spec: EffectSpec, centre, radius: float) -> list:
    """Concentric shells, each fainter than the one inside it."""
    count = max(1, int(spec.get("count")))
    reach = float(spec.get("reach")) * radius
    spacing = float(spec.get("spacing")) * radius
    falloff = float(spec.get("falloff"))
    strength = float(spec.get("strength"))
    tilt = math.radians(float(spec.get("tilt")))
    segments = max(8, int(spec.get("segments")))
    normal = np.array([math.sin(tilt), 0.0, math.cos(tilt)])
    objects = []
    for index in range(count):
        # Fainter outward: a standing wave loses amplitude with distance,
        # and rings of equal brightness read as a stack of hoops.
        path = _ring_path(centre, reach + index * spacing, normal, segments)
        made = _glow_pair(f"fx_res{index}", [path],
                          float(spec.get("width")) * radius, spec.colour(),
                          strength * (falloff ** index), sheath=3.0)
        for obj in made:
            _set_light_contribution(obj, bool(spec.get("light")))
        objects.extend(made)
    return objects


def _build_orbit(spec: EffectSpec, centre, radius: float) -> list:
    """Rings at even angles about the subject, like a textbook atom."""
    count = max(1, int(spec.get("count")))
    tilt = math.radians(float(spec.get("tilt")))
    paths = []
    for index in range(count):
        spin = math.pi * index / count
        normal = np.array([math.sin(tilt) * math.cos(spin),
                           math.sin(tilt) * math.sin(spin),
                           math.cos(tilt)])
        paths.append(_ring_path(centre, float(spec.get("reach")) * radius,
                                normal, int(spec.get("segments"))))
    objects = _glow_pair("fx_orbit", paths,
                         float(spec.get("width")) * radius, spec.colour(),
                         float(spec.get("strength")), sheath=2.8)
    for obj in objects:
        _set_light_contribution(obj, bool(spec.get("light")))
    return objects


def _fresnel_shell(name: str, colour: RGB, strength: float,
                   power: float = 3.0):
    """A material that glows only where the surface turns edge-on.

    A uniformly emissive sphere around a structure does not read as a
    halo: it reads as a frosted ball with something inside it, because
    its brightness is the same in the middle of the silhouette as at the
    rim. An atmosphere is the other way round -- you look through more
    of it near the limb, so it brightens toward the edge and disappears
    face-on.

    ``LayerWeight.Facing`` is exactly that quantity, so driving both the
    emission strength and the transparency mix with it turns the shell
    into a limb glow with no disc at all.
    """
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    out = _fresh_tree(nodes)

    weight = nodes.new("ShaderNodeLayerWeight")
    weight.inputs["Blend"].default_value = 0.5
    curve = nodes.new("ShaderNodeMath")
    curve.operation = "POWER"
    curve.inputs[1].default_value = float(power)
    gain = nodes.new("ShaderNodeMath")
    gain.operation = "MULTIPLY"
    gain.inputs[1].default_value = float(strength)

    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*colour, 1.0)
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    mix = nodes.new("ShaderNodeMixShader")

    links.new(weight.outputs["Facing"], curve.inputs[0])
    links.new(curve.outputs[0], gain.inputs[0])
    links.new(gain.outputs[0], emission.inputs["Strength"])
    # The same curve mixes away the surface face-on, so the shell is
    # invisible where it is not glowing rather than merely dim.
    links.new(curve.outputs[0], mix.inputs["Fac"])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs["Surface"])
    return material


def _volume_glow(name: str, colour: RGB, density: float, strength: float):
    """A material that is light in the air, with no surface at all.

    The honest tool for a halo. A **surface** shell -- emissive, or
    Fresnel-driven at its limb -- always keeps the sphere's silhouette,
    so it reads as a disc or a ring drawn around the structure however
    it is tuned; that was tried both ways and neither stopped being a
    hard edge. A volume has no silhouette: there is nothing to see edge
    on, only more or less of it to look through, which is what a glow
    is.

    The surface output is left empty on purpose, so the shell is
    invisible to the camera except through what its volume emits.
    """
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    out = _fresh_tree(nodes)

    volume = nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = float(density)
    volume.inputs["Color"].default_value = (*colour, 1.0)
    if "Emission Color" in volume.inputs:
        volume.inputs["Emission Color"].default_value = (*colour, 1.0)
    if "Emission Strength" in volume.inputs:
        volume.inputs["Emission Strength"].default_value = float(strength)
    links.new(volume.outputs[0], out.inputs["Volume"])
    return material


def _build_halo(spec: EffectSpec, centre, radius: float) -> list:
    """A soft ball of light the structure sits inside."""
    # Volume emission accumulates along the ray, so a shell twice as
    # wide glows twice as brightly for the same setting -- which breaks
    # the rule that an effect means the same thing at every size, and
    # broke it spectacularly: the first try came back a white frame with
    # the structure a ghost inside it. Dividing by the path length makes
    # `strength` the glow across the whole shell rather than per unit.
    reach = float(spec.get("reach")) * radius
    material = _volume_glow("fx_halo", spec.colour(),
                            float(spec.get("density")),
                            float(spec.get("strength"))
                            / max(2.0 * reach, 1e-6))
    obj = _ico("fx_halo", centre, float(spec.get("reach")) * radius,
               material, subdivisions=4)
    # A glowing shell that casts a shadow on the floor is a glowing
    # ball, which is not the effect.
    _set_light_contribution(obj, bool(spec.get("light")))
    return [obj]


def _build_fog(spec: EffectSpec, centre, radius: float) -> list:
    """World volume scatter, so beams and lights become visible shafts."""
    import bpy

    del centre, radius
    world = bpy.context.scene.world
    if world is None:                           # pragma: no cover - no world
        return []
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    out = _output_node(nodes, "ShaderNodeOutputWorld")
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Color"].default_value = (*spec.colour(), 1.0)
    scatter.inputs["Density"].default_value = float(spec.get("strength"))
    if "Anisotropy" in scatter.inputs:
        scatter.inputs["Anisotropy"].default_value = float(
            spec.get("anisotropy"))
    links.new(scatter.outputs[0], out.inputs["Volume"])
    return []


def _gaussian_blur(plane: np.ndarray, radius: float,
                   passes: int = 3) -> np.ndarray:
    """Blur a 2D plane by repeated box blur, which converges on Gaussian.

    Three box passes is the standard approximation and is within a
    couple of per cent of a true Gaussian -- indistinguishable in a
    glow, and it costs two cumulative sums per axis per pass instead of
    a convolution. No scipy, which Blender's bundled Python does not
    have.
    """
    box = max(1, int(round(radius)))
    out = plane.astype(np.float64)
    for _ in range(passes):
        for axis in (0, 1):
            out = np.swapaxes(out, 0, axis)
            padded = np.pad(out, ((box, box), (0, 0)), mode="edge")
            cumulative = np.cumsum(padded, axis=0)
            cumulative = np.vstack([np.zeros((1, out.shape[1])), cumulative])
            window = 2 * box + 1
            out = ((cumulative[window:window + out.shape[0]]
                    - cumulative[:out.shape[0]]) / window)
            out = np.swapaxes(out, 0, axis)
    return out


def glare_image(rgb: np.ndarray, strength: float = 0.35,
                size: float = 0.5, threshold: float = 0.75,
                maximum: float = 3.0,
                tint: RGB | None = None) -> np.ndarray:
    """Add bloom to a rendered image, as an array operation.

    Everything above ``threshold`` in luminance is blurred and added
    back, which is what a lens does with a bright source and what makes
    an emissive cover image read as *lit* rather than merely bright.

    **Done here rather than in Blender's compositor, and that is
    measured rather than preferred.** Blender 5 moved compositing to
    ``scene.compositing_node_group``; assigning one in a headless
    ``bpy`` build returns a black frame -- 0.0003 mean against 0.2125
    for the same scene without it, reproduced on a bare emissive sphere
    with nothing else in the scene. A feature that silently turns a
    render black is worse than no feature, so the glare is an array pass
    over the finished image: it works in every Blender, it costs nothing
    in sample count, and unlike a node graph it can be tested in an
    ordinary interpreter.

    ``size`` is a 0-1 factor of the image's short side, matching how
    Blender 5 parametrises its own Glare node.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] < 3:       # pragma: no cover - guard
        raise ValueError("glare_image needs an (h, w, 3+) array.")
    colour = rgb[:, :, :3]
    luminance = colour @ np.array([0.2126, 0.7152, 0.0722])
    # Capped, not just floored. An emissive filament is many times
    # brighter than white in linear light, so without a ceiling its glow
    # swamps the frame -- the whole image came back washed out, which
    # reads as a broken render rather than as bloom. Blender's own Glare
    # node has a Maximum input for the same reason.
    excess = np.clip(luminance - float(threshold), 0.0, float(maximum))
    if not np.any(excess > 0):
        return rgb
    # Keep the hue of what is glowing: a white bloom around a green
    # laser is a lens flare from a different photograph.
    scale = np.divide(excess, np.maximum(luminance, 1e-9))
    bright = colour * scale[:, :, None]
    radius = max(1.0, float(size) * 0.06 * min(colour.shape[:2]))
    glow = np.dstack([_gaussian_blur(bright[:, :, channel], radius)
                      for channel in range(3)])
    if tint is not None:
        glow = glow * np.asarray(tint, dtype=float)[None, None, :]
    out = rgb.copy()
    out[:, :, :3] = colour + float(strength) * glow
    return out


def apply_glare(path, spec: EffectSpec) -> None:
    """Run :func:`glare_image` over a rendered file, in place.

    Pixels go through Blender's own image API rather than an imaging
    library, because ``bpy`` is the one dependency this is guaranteed to
    have.
    """
    import bpy

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = image.size
        channels = image.channels
        pixels = np.array(image.pixels[:], dtype=np.float64)
        frame = pixels.reshape(height, width, channels)
        frame = glare_image(frame, float(spec.get("strength")),
                            float(spec.get("size")),
                            float(spec.get("threshold")),
                            float(spec.get("maximum")))
        # float32, not the float64 the maths ran in: foreach_set writes
        # straight into Blender's buffer and refuses a wider dtype.
        image.pixels.foreach_set(
            np.clip(frame, 0.0, 1.0).ravel().astype(np.float32))
        image.filepath_raw = str(path)
        image.file_format = "PNG"
        image.save()
    finally:
        bpy.data.images.remove(image)


def _build_glare(spec: EffectSpec, centre, radius: float) -> list:
    """No scene objects: the glare runs over the finished image.

    Kept in the effect registry so it is asked for, listed and
    parametrised exactly like the others; :func:`apply_glare` is called
    by the render script once the file exists.
    """
    del spec, centre, radius
    return []


_BUILDERS = {
    "laser": _build_laser,
    "arc": _build_arc,
    "resonance": _build_resonance,
    "orbit": _build_orbit,
    "halo": _build_halo,
    "fog": _build_fog,
    "glare": _build_glare,
}


def build_effects(specs, centre, radius: float, points=None) -> list:
    """Build every effect, and return the objects they added.

    Returned so the caller can keep them **out of the camera framing**: a
    beam seven subject radii long would otherwise set the frame and the
    structure would be a dot in the middle of it. That is the same
    mistake the floor plane makes, and it has the same fix -- measure the
    subject, then add the scenery.

    ``points`` are the atom positions, when the caller has them. Only
    the laser uses them, and only to find where the structure's surface
    is along its own beam; see :func:`surface_along`.
    """
    objects: list = []
    for spec in specs:
        builder = _BUILDERS[spec.kind]
        if spec.kind == "laser":
            objects.extend(builder(spec, centre, radius, points))
        else:
            objects.extend(builder(spec, centre, radius))
    return objects
