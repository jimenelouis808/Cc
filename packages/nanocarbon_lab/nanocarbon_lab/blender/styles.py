"""Looks for the Blender pipeline: styles, backgrounds, shaders, colours.

Pure data and pure-Python helpers -- **no `bpy` import here** -- so this
module can be unit-tested without a Blender interpreter. :mod:`render_cnt`
(which does import ``bpy``) consumes it to build materials, the world, the
lighting rig and the camera.

**What the hue means and how the surface behaves are two questions**, and
conflating them was the original design's one real limitation. A style
used to hardcode a colour per *ring size*, so the only thing a render
could say with colour was "pentagon, hexagon, heptagon, octagon" -- which
is the right thing to say about a defected cage and says nothing at all
about a composition of three structures, or about a doped shell, or about
a particle with a metal core. So the two are separated:

* a **style** supplies the material *character* -- metallic, roughness,
  clearcoat, emission strength, the shader kind -- plus the world, the
  lights and the camera;
* a **colour mode** (:data:`COLOUR_MODES`) supplies the base colour:
  ``"ring"`` as before, ``"element"`` from :data:`ELEMENT_COLOURS`,
  ``"structure"`` from :data:`STRUCTURE_PALETTE` for a composition, or
  ``"uniform"`` for one colour throughout.

:meth:`MaterialSpec.with_colour` is what joins them, and it is pure data:
it returns the same spec with a different base colour, so a gold style
rendered by element gives *gold-finish* carbon and oxygen rather than two
flat primaries.

Each style evokes a class of materials-science figure rather than any
journal's house style -- Nature, ACS Nano and Small publish no fixed
visual template, their covers being commissioned art.
"""

from __future__ import annotations

import sys
from pathlib import Path

# These modules are package *data*, loaded by path rather than imported
# as a package, so a sibling import has to be made to work rather than
# assumed: inside Blender this directory is on the path because
# ``render_cnt`` put it there, and outside it nothing has.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataclasses import dataclass, field, replace

from effects import EffectSpec

RGB = tuple[float, float, float]
RGBA = tuple[float, float, float, float]

#: The shader each :class:`MaterialSpec` asks for. ``principled`` is
#: Blender's general-purpose BSDF and the default; the rest are node
#: setups :mod:`render_cnt` builds, and they are here because a
#: Principled BSDF cannot express them by parameter alone.
#:
#: ``glass``   -- refractive, for an ice/resin look.
#: ``toon``    -- flat banded shading, for a schematic.
#: ``emission``-- pure light, for a glowing accent or a wireframe.
#: ``clay``    -- diffuse only, no specular at all, for a matte model.
SHADERS = ("principled", "glass", "toon", "emission", "clay")


@dataclass(frozen=True)
class MaterialSpec:
    """One shading role: its colour and, separately, its character."""

    base_color: RGBA
    metallic: float = 0.0
    roughness: float = 0.4
    transmission: float = 0.0
    ior: float = 1.45
    emission_color: RGB = (0.0, 0.0, 0.0)
    emission_strength: float = 0.0
    clearcoat: float = 0.0
    shader: str = "principled"
    #: Light scattered under the surface. What makes a cage read as
    #: wax or jade rather than plastic.
    subsurface: float = 0.0
    subsurface_color: RGB = (1.0, 1.0, 1.0)
    #: A soft velvet rim. Small amounts stop a matte material reading
    #: as flat grey at its silhouette.
    sheen: float = 0.0
    #: Brushed rather than polished: a metal whose highlight is
    #: stretched along the surface instead of being a round spot. This
    #: is what separates a machined part from a chrome ball, and no
    #: amount of roughness alone produces it.
    anisotropy: float = 0.0
    anisotropy_rotation: float = 0.0
    #: Thin-film interference, in nanometres of film. 0 is off; 200-600
    #: is the soap-bubble and anodised-titanium range, where the colour
    #: changes with viewing angle because the film's two reflections
    #: interfere. Real, angle-dependent iridescence -- not a colour ramp
    #: faking it, which cannot change with the view.
    thin_film: float = 0.0
    thin_film_ior: float = 1.33
    #: Tint of the clear coat, for a lacquer over a metal.
    coat_tint: RGB = (1.0, 1.0, 1.0)
    coat_roughness: float = 0.03
    #: Colour of the specular highlight, which on a real metal is not
    #: white -- gold reflects gold.
    specular_tint: RGB = (1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.shader not in SHADERS:
            raise ValueError(
                f"unknown shader {self.shader!r}; known: {', '.join(SHADERS)}"
            )

    def with_colour(self, rgb: RGB | RGBA) -> MaterialSpec:
        """The same material with a different base colour.

        The point of the whole split: a style's *character* -- its
        metallic, its roughness, its clearcoat, its shader -- is what
        makes it that style, and the colour is what the render is being
        asked to encode. Swapping only the colour means colouring by
        element inside ``gold_nanotech`` gives a gold-finish carbon and a
        gold-finish oxygen, rather than abandoning the style for two flat
        primaries.

        An emissive material's emission colour is swapped with it, since
        a red-hot accent that glows blue is not a colour choice, it is a
        mistake.
        """
        rgba: RGBA = (*rgb[:3], rgb[3] if len(rgb) > 3 else 1.0)
        if self.emission_strength > 0:
            return replace(self, base_color=rgba, emission_color=rgba[:3])
        return replace(self, base_color=rgba)


@dataclass(frozen=True)
class LightSpec:
    kind: str  # "SUN" | "AREA" | "POINT"
    energy: float
    color: RGB
    location: tuple[float, float, float]
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: float = 1.0  # AREA light extent (m), ignored for SUN/POINT


#: How the world behind the subject is built.
#:
#: ``gradient``    -- a vertical ramp between two colours, the original.
#: ``solid``       -- one flat colour, which is what a figure for a
#:                    paper wants: a gradient reads as decoration.
#: ``transparent`` -- no background at all, alpha straight through, for
#:                    compositing into a layout.
#: ``studio``      -- a gradient *plus* a floor plane, so the subject
#:                    has a surface to cast onto and sit on.
#: ``space``       -- a starfield: a dark field with distant stars,
#:                    generated procedurally rather than from an image,
#:                    so it needs no texture file and is resolution
#:                    independent.
#: ``nebula``      -- the starfield plus drifting coloured cloud, for
#:                    the deep-field look.
BACKGROUNDS = ("gradient", "solid", "transparent", "studio", "space",
               "nebula")


@dataclass(frozen=True)
class Background:
    """The world, and optionally a floor to stand the subject on.

    Separate from :class:`Style` so a style's look can be kept while its
    backdrop changes -- the same shading on white for a paper figure and
    on black for a cover is two uses of one style, not two styles.
    """

    kind: str = "gradient"
    top: RGB = (0.0, 0.0, 0.0)
    bottom: RGB = (0.0, 0.0, 0.0)
    strength: float = 1.0
    #: Ground plane, for ``studio`` (and allowed with any kind).
    floor: bool = False
    floor_color: RGB = (0.5, 0.5, 0.52)
    floor_roughness: float = 0.4
    #: A floor that catches shadow but is itself invisible, so the
    #: subject keeps its contact shadow over a transparent background.
    floor_is_shadow_catcher: bool = False
    # -- starfield, for "space" and "nebula" ---------------------------
    #: How many stars, as a texture scale. Higher is more and smaller.
    star_density: float = 220.0
    #: Fraction of cells that light up. Small: a sky that is mostly
    #: stars reads as noise, not as space.
    star_amount: float = 0.04
    #: Brightness of the brightest stars.
    star_brightness: float = 18.0
    #: The two cloud colours of a ``nebula``, mixed by a noise field.
    nebula_colors: tuple[RGB, RGB] = ((0.08, 0.02, 0.18),
                                      (0.00, 0.10, 0.20))
    nebula_scale: float = 1.6
    nebula_strength: float = 0.9

    @property
    def has_stars(self) -> bool:
        return self.kind in ("space", "nebula")

    def __post_init__(self) -> None:
        if self.kind not in BACKGROUNDS:
            raise ValueError(
                f"unknown background {self.kind!r}; known: "
                f"{', '.join(BACKGROUNDS)}"
            )

    @property
    def is_transparent(self) -> bool:
        return self.kind == "transparent"

    @property
    def has_floor(self) -> bool:
        return self.floor or self.kind == "studio"


@dataclass(frozen=True)
class Style:
    """One complete look: materials, world, lights, camera, engine."""

    name: str
    description: str
    background: Background
    hexagon: MaterialSpec
    pentagon: MaterialSpec
    heptagon: MaterialSpec
    octagon: MaterialSpec
    bond: MaterialSpec
    lights: list[LightSpec] = field(default_factory=list)
    camera_lens_mm: float = 50.0
    camera_dof_fstop: float | None = 2.8  # None disables depth of field
    engine: str = "CYCLES"
    samples: int = 256
    #: What this style's own colouring means, used when the caller does
    #: not say. A cover style defaults to ring colouring because that is
    #: what it was tuned for; a figure style defaults to element.
    default_colour_by: str = "ring"
    #: Scene effects the style brings of its own, as
    #: :class:`effects.EffectSpec`. A cover style that is *about* a laser
    #: should arrive with the laser rather than needing it typed in;
    #: ``--no-style-effects`` drops them. Held as a tuple because a
    #: mutable default on a frozen dataclass is shared between every
    #: instance, and a style that accumulated effects across renders
    #: would be very hard to diagnose.
    effects: tuple = ()

    # -- compatibility: the three fields these replaced ----------------
    # Read-only, and delegating rather than duplicated, so there is one
    # source of truth for the backdrop.
    @property
    def world_color_top(self) -> RGB:
        return self.background.top

    @property
    def world_color_bottom(self) -> RGB:
        return self.background.bottom

    @property
    def world_strength(self) -> float:
        return self.background.strength

    @property
    def background_is_transparent(self) -> bool:
        return self.background.is_transparent

    def ring_material(self, size: int) -> MaterialSpec:
        """The material for a ring of this size.

        Anything that is not 5, 7 or 8 is the hexagon material, which is
        also the right answer for an atom with no ring metadata at all.
        """
        return {5: self.pentagon, 7: self.heptagon,
                8: self.octagon}.get(size, self.hexagon)


# ----------------------------------------------------------------------
# colour modes
# ----------------------------------------------------------------------
#: What the colour of an atom is allowed to mean.
#:
#: ``ring``      -- ring size, as the original pipeline did. Says where
#:                  the curvature and the defects are, and says nothing
#:                  about chemistry; on a closed cage every atom is on a
#:                  pentagon, so it says little there either.
#: ``element``   -- the chemical element. What a doped shell, a
#:                  functionalised surface or a metal core needs, and
#:                  what every other structure viewer does by default.
#: ``structure`` -- which item of a composition the atom belongs to.
#:                  Meaningless for a single structure, and the only
#:                  useful colouring for a figure of three of them.
#: ``uniform``   -- one colour throughout, for a silhouette.
COLOUR_MODES = ("ring", "element", "structure", "uniform")

#: Element colours, CPK/Jmol-like.
#:
#: **Like, not identical to, and not VESTA's.** CPK is a convention with
#: several incompatible tabulations; Jmol's is the most widely copied and
#: this follows it for the elements it covers. VESTA ships its own
#: palette which differs for a number of elements, so a figure rendered
#: here and the same structure opened in VESTA will not match element for
#: element. Anything absent falls back to :data:`UNKNOWN_ELEMENT_COLOUR`,
#: which is deliberately an unmistakable magenta rather than a plausible
#: grey: a missing element should look wrong, not look like carbon.
ELEMENT_COLOURS: dict[str, RGB] = {
    "H": (1.00, 1.00, 1.00),
    "B": (1.00, 0.71, 0.71),
    "C": (0.35, 0.35, 0.35),
    "N": (0.19, 0.31, 0.97),
    "O": (1.00, 0.05, 0.05),
    "F": (0.56, 0.88, 0.31),
    "Na": (0.67, 0.36, 0.95),
    "Mg": (0.54, 1.00, 0.00),
    "Al": (0.75, 0.65, 0.65),
    "Si": (0.94, 0.78, 0.63),
    "P": (1.00, 0.50, 0.00),
    "S": (1.00, 1.00, 0.19),
    "Cl": (0.12, 0.94, 0.12),
    "K": (0.56, 0.25, 0.83),
    "Ca": (0.24, 1.00, 0.00),
    "Ti": (0.75, 0.76, 0.78),
    "V": (0.65, 0.65, 0.67),
    "Cr": (0.54, 0.60, 0.78),
    "Mn": (0.61, 0.48, 0.78),
    "Fe": (0.88, 0.40, 0.20),
    "Co": (0.94, 0.56, 0.63),
    "Ni": (0.31, 0.82, 0.31),
    "Cu": (0.78, 0.50, 0.20),
    "Zn": (0.49, 0.50, 0.69),
    "Ge": (0.40, 0.56, 0.56),
    "Se": (1.00, 0.63, 0.00),
    "Zr": (0.58, 0.88, 0.88),
    "Nb": (0.45, 0.76, 0.79),
    "Mo": (0.33, 0.71, 0.71),
    "Ru": (0.14, 0.56, 0.56),
    "Rh": (0.04, 0.49, 0.55),
    "Pd": (0.00, 0.41, 0.52),
    "Ag": (0.75, 0.75, 0.75),
    "Sn": (0.40, 0.50, 0.50),
    "Te": (0.83, 0.48, 0.00),
    "Hf": (0.30, 0.76, 1.00),
    "Ta": (0.30, 0.65, 1.00),
    "W": (0.13, 0.58, 0.84),
    "Re": (0.15, 0.49, 0.67),
    "Pt": (0.82, 0.82, 0.88),
    "Au": (1.00, 0.82, 0.14),
    "Pb": (0.34, 0.35, 0.38),
    "Bi": (0.62, 0.31, 0.71),
}

#: An element with no entry. Magenta on purpose -- see
#: :data:`ELEMENT_COLOURS`.
UNKNOWN_ELEMENT_COLOUR: RGB = (1.0, 0.0, 1.0)

#: **Display** radii per element, in Å, for colouring by element.
#:
#: Display radii, not covalent or van der Waals ones, and the difference
#: matters: drawn at their real van der Waals radii a carbon and an iron
#: overlap into a solid lump with no bond visible, which is why every
#: structure viewer draws a ball-and-stick model at a fraction of them.
#: These are roughly a third of the van der Waals radius, scaled so
#: carbon lands on the 0.35 Å the pipeline already used, and they
#: preserve the *ordering* -- hydrogen smaller than carbon, a transition
#: metal larger -- which is the only thing a reader takes from the sizes.
#:
#: Anything absent takes :data:`DEFAULT_ELEMENT_RADIUS`, which is
#: carbon's: an unknown element should be visible and unremarkable in
#: size while its magenta colour says it is unknown.
ELEMENT_RADII: dict[str, float] = {
    "H": 0.23, "B": 0.38, "C": 0.35, "N": 0.33, "O": 0.31, "F": 0.29,
    "Na": 0.50, "Mg": 0.47, "Al": 0.45, "Si": 0.42, "P": 0.40, "S": 0.39,
    "Cl": 0.37, "K": 0.58, "Ca": 0.54, "Ti": 0.49, "V": 0.47, "Cr": 0.46,
    "Mn": 0.46, "Fe": 0.45, "Co": 0.44, "Ni": 0.44, "Cu": 0.45, "Zn": 0.45,
    "Ge": 0.44, "Se": 0.42, "Zr": 0.53, "Nb": 0.51, "Mo": 0.50, "Ru": 0.47,
    "Rh": 0.47, "Pd": 0.47, "Ag": 0.50, "Sn": 0.49, "Te": 0.47, "Hf": 0.52,
    "Ta": 0.51, "W": 0.50, "Re": 0.49, "Pt": 0.48, "Au": 0.49, "Pb": 0.52,
    "Bi": 0.51,
}

#: Radius for an element with no entry in :data:`ELEMENT_RADII`.
DEFAULT_ELEMENT_RADIUS = 0.35

#: Whether a colour mode wants the bonds split in half, each half taking
#: its own atom's colour.
#:
#: True where the colour means something about the *atom* -- its element,
#: or which structure it belongs to -- because a C-O bond drawn in
#: carbon's grey says the oxygen is carbon. False where it means
#: something about the *ring*, which is a property of neither end
#: individually, and for a uniform render where there is nothing to
#: split. This is what every molecular viewer does, and it costs one
#: extra cylinder per bond.
SPLIT_BONDS = {"ring": False, "element": True, "structure": True,
               "uniform": False}

#: Colours for per-structure colouring, in order of use.
#:
#: Eight hues chosen to stay distinguishable from one another: the first
#: five are Paul Tol's bright qualitative set, which is designed to
#: survive the common colour-vision deficiencies, and the last three
#: continue it. A composition of more items than this cycles, which the
#: report says rather than silently repeating a colour.
STRUCTURE_PALETTE: tuple[RGB, ...] = (
    (0.27, 0.47, 0.77),   # blue
    (0.80, 0.37, 0.00),   # orange-red
    (0.13, 0.62, 0.42),   # green
    (0.80, 0.73, 0.27),   # yellow
    (0.59, 0.31, 0.64),   # purple
    (0.27, 0.73, 0.83),   # cyan
    (0.84, 0.47, 0.64),   # pink
    (0.45, 0.45, 0.45),   # grey
)


def element_colour(symbol: str) -> RGB:
    """The colour for an element symbol, case-insensitively."""
    if not symbol:
        return UNKNOWN_ELEMENT_COLOUR
    key = symbol[0].upper() + symbol[1:].lower()
    return ELEMENT_COLOURS.get(key, UNKNOWN_ELEMENT_COLOUR)


def element_radius(symbol: str) -> float:
    """The display radius for an element symbol, in Å."""
    if not symbol:
        return DEFAULT_ELEMENT_RADIUS
    key = symbol[0].upper() + symbol[1:].lower()
    return ELEMENT_RADII.get(key, DEFAULT_ELEMENT_RADIUS)


def structure_colour(index: int) -> RGB:
    """The colour for item ``index`` of a composition, cycling."""
    return STRUCTURE_PALETTE[int(index) % len(STRUCTURE_PALETTE)]


def resolve_colour_by(requested: str | None, style: Style) -> str:
    """Which colour mode to use, given a request and a style.

    ``None`` takes the style's own default, so a cover style keeps
    colouring by ring and a figure style by element without the caller
    having to know which is which.
    """
    mode = requested or style.default_colour_by
    if mode not in COLOUR_MODES:
        raise ValueError(
            f"unknown colour mode {mode!r}; known: {', '.join(COLOUR_MODES)}"
        )
    return mode


# ----------------------------------------------------------------------
# finishes
# ----------------------------------------------------------------------
#: A **finish** replaces a material's character while keeping its colour,
#: so it is a third axis beside the style and the colour mode: "element
#: colours, in glass, on the cosmos backdrop" is one render rather than a
#: style nobody wrote.
#:
#: ``"style"`` means "leave it alone", which is the default: a style's
#: author chose its character and a finish overrides that choice, so it
#: has to be asked for.
FINISHES = ("style", "glass", "crystal", "glow", "neon", "metal", "gold",
            "brushed", "chrome", "plastic", "ceramic", "clay", "matte",
            "wax", "iridescent", "toon")

#: How much of a material's own colour a glass finish keeps. 0 is clear
#: glass with no tint at all and 1 is the colour at full strength, which
#: on anything dark comes out black -- see :func:`apply_finish`.
GLASS_TINT = 0.25


def apply_finish(spec: MaterialSpec, finish: str) -> MaterialSpec:
    """The same material with a different surface character.

    The **colour survives and everything else is replaced**, which is
    what makes a finish composable with a colour mode: colouring by
    element and then asking for glass gives a glass carbon and a glass
    oxygen, each still its own colour, rather than one grey glass
    object.

    Transmission is done on the Principled BSDF rather than with the
    Glass shader, for two reasons that matter here: Principled keeps the
    base colour tinting the transmitted light (a Glass node's colour
    multiplies differently and washes out), and EEVEE renders Principled
    transmission while its handling of a raw Glass node is poorer. The
    ``glass_cryo`` style still uses the Glass node directly, which is a
    legitimate second look rather than a duplicate.
    """
    if finish in ("style", "", None):
        return spec
    if finish not in FINISHES:
        raise ValueError(
            f"unknown finish {finish!r}; known: {', '.join(FINISHES)}")
    colour = spec.base_color
    rgb = colour[:3]
    # Transmitted light is multiplied by the base colour at every
    # bounce, so a glass version of a dark material is not clear glass:
    # carbon's element grey (0.35) through a cage's near and far walls
    # comes out nearly black, which reads as soot rather than as glass.
    # Lifting the tint toward white keeps *which* element an atom is
    # while letting the light through, which is what "in glass" means.
    tint = tuple(1.0 - GLASS_TINT * (1.0 - channel) for channel in rgb)

    if finish == "glass":
        return replace(spec, base_color=(*tint, colour[3]),
                       metallic=0.0, roughness=0.04,
                       transmission=1.0, ior=1.52, clearcoat=0.0,
                       emission_strength=0.0, shader="principled",
                       anisotropy=0.0, thin_film=0.0)
    if finish == "crystal":
        # A higher index bends light more, so facets and edges catch
        # colour the way a cut stone does. 2.4 is diamond.
        return replace(spec, base_color=(*tint, colour[3]),
                       metallic=0.0, roughness=0.015,
                       transmission=1.0, ior=2.4, clearcoat=0.0,
                       emission_strength=0.0, shader="principled",
                       anisotropy=0.0, thin_film=0.0)
    if finish == "glow":
        # A surface that also emits: it still catches light and still
        # has a silhouette, which pure emission does not.
        return replace(spec, metallic=0.0, roughness=0.3,
                       emission_color=rgb, emission_strength=3.0,
                       transmission=0.0, shader="principled",
                       clearcoat=0.2)
    if finish == "neon":
        # Pure light, no surface response. Reads as a tube sign.
        return replace(spec, emission_color=rgb, emission_strength=8.0,
                       shader="emission")
    if finish == "metal":
        return replace(spec, metallic=1.0, roughness=0.25,
                       transmission=0.0, emission_strength=0.0,
                       clearcoat=0.0, shader="principled",
                       specular_tint=rgb, anisotropy=0.0)
    if finish == "gold":
        # The colour is overridden here on purpose: "gold" is a
        # material, not a hue to apply to someone else's colour, and a
        # blue gold is a contradiction.
        return replace(spec, base_color=(0.86, 0.66, 0.24, colour[3]),
                       metallic=1.0, roughness=0.20, clearcoat=0.35,
                       coat_tint=(1.0, 0.92, 0.75),
                       specular_tint=(1.0, 0.88, 0.62),
                       transmission=0.0, emission_strength=0.0,
                       shader="principled")
    if finish == "brushed":
        return replace(spec, metallic=1.0, roughness=0.35,
                       anisotropy=0.85, anisotropy_rotation=0.25,
                       transmission=0.0, emission_strength=0.0,
                       shader="principled")
    if finish == "chrome":
        return replace(spec, metallic=1.0, roughness=0.02,
                       transmission=0.0, emission_strength=0.0,
                       clearcoat=0.0, shader="principled")
    if finish == "plastic":
        return replace(spec, metallic=0.0, roughness=0.25,
                       clearcoat=0.6, coat_roughness=0.05,
                       transmission=0.0, emission_strength=0.0,
                       shader="principled")
    if finish == "ceramic":
        return replace(spec, metallic=0.0, roughness=0.12,
                       clearcoat=0.8, coat_roughness=0.02,
                       subsurface=0.15, subsurface_color=rgb,
                       transmission=0.0, emission_strength=0.0,
                       shader="principled")
    if finish == "clay":
        return replace(spec, roughness=0.75, shader="clay",
                       metallic=0.0, transmission=0.0,
                       emission_strength=0.0)
    if finish == "matte":
        return replace(spec, metallic=0.0, roughness=0.9, sheen=0.25,
                       clearcoat=0.0, transmission=0.0,
                       emission_strength=0.0, shader="principled")
    if finish == "wax":
        # Light travelling a short way inside before coming back out is
        # what separates wax, jade and skin from painted plastic.
        return replace(spec, metallic=0.0, roughness=0.35,
                       subsurface=0.7, subsurface_color=rgb,
                       transmission=0.0, emission_strength=0.0,
                       shader="principled")
    if finish == "iridescent":
        return replace(spec, metallic=1.0, roughness=0.10,
                       thin_film=430.0, thin_film_ior=1.45,
                       transmission=0.0, emission_strength=0.0,
                       shader="principled")
    if finish == "toon":
        return replace(spec, shader="toon", metallic=0.0,
                       transmission=0.0, emission_strength=0.0)
    raise AssertionError(f"finish {finish!r} is in FINISHES with no rule.")


def describe_finishes() -> str:
    """One line per finish, for ``--list-finishes``."""
    notes = {
        "style": "leave the style's own character alone (the default)",
        "glass": "refractive, colour tinting the light through it",
        "crystal": "glass at diamond's index -- edges catch colour",
        "glow": "a surface that also emits: still lit, still a silhouette",
        "neon": "pure light, no surface response, like a tube sign",
        "metal": "polished, with the highlight taking the base colour",
        "gold": "polished gold under a tinted clear coat (sets the colour)",
        "brushed": "anisotropic metal: the highlight stretches",
        "chrome": "a mirror",
        "plastic": "coloured body under a glossy clear coat",
        "ceramic": "glazed, with a little light scattering under it",
        "clay": "purely diffuse, no specular at all",
        "matte": "rough with a velvet rim",
        "wax": "subsurface scattering -- wax, jade, skin",
        "iridescent": "thin-film interference, colour shifts with angle",
        "toon": "flat banded shading",
    }
    return "\n".join(f"{name:11s} {notes[name]}" for name in FINISHES)


def materials_for(style: Style, colour_by: str, keys,
                  finish: str = "style") -> list[MaterialSpec]:
    """The material list for a colour mode, one entry per key.

    Parameters
    ----------
    style
        Supplies the character, and for ``"ring"`` the colours too.
    colour_by
        One of :data:`COLOUR_MODES`.
    keys
        What the materials are indexed by: ring sizes for ``"ring"``,
        element symbols for ``"element"``, item indices for
        ``"structure"``, ignored for ``"uniform"``.
    finish
        One of :data:`FINISHES`. ``"style"`` leaves each material's
        character alone, which is the default; anything else replaces
        the character and keeps the colour.

    Returns
    -------
    list of MaterialSpec
        In the same order as ``keys``, so the caller can build a Blender
        material slot per key and set a face's ``material_index`` to the
        key's position.

    Notes
    -----
    Everything but ``"ring"`` takes the **hexagon** material as its
    character: it is a style's bulk material, the one its author tuned
    for the body of the structure, so an element-coloured render looks
    like that style rather than like a default.
    """
    if colour_by == "ring":
        specs = [style.ring_material(int(key)) for key in keys]
    elif colour_by == "element":
        specs = [style.hexagon.with_colour(element_colour(str(key)))
                 for key in keys]
    elif colour_by == "structure":
        specs = [style.hexagon.with_colour(structure_colour(int(key)))
                 for key in keys]
    elif colour_by == "uniform":
        specs = [style.hexagon]
    else:
        raise ValueError(f"unknown colour mode {colour_by!r}.")
    return [apply_finish(spec, finish) for spec in specs]


# ----------------------------------------------------------------------
# the catalogue
# ----------------------------------------------------------------------
def _accent(base: RGBA, glow: RGB, strength: float, **extra) -> MaterialSpec:
    """A defect-accent material: coloured, and emitting that colour."""
    return MaterialSpec(base_color=base, emission_color=glow,
                        emission_strength=strength, **extra)


STYLES: dict[str, Style] = {
    "nature_dark": Style(
        name="nature_dark",
        description=(
            "Matte black void, single dramatic key light + cool rim light, "
            "near-black graphite body with glowing defect accents. Evokes "
            "the moody single-subject cover style common in Nature/Nature "
            "Nanotechnology."
        ),
        background=Background(kind="gradient", top=(0.0, 0.0, 0.0),
                              bottom=(0.0, 0.0, 0.0), strength=0.02),
        hexagon=MaterialSpec(
            base_color=(0.03, 0.03, 0.035, 1.0), metallic=0.6, roughness=0.28,
        ),
        pentagon=_accent((0.85, 0.15, 0.05, 1.0), (1.0, 0.25, 0.05), 1.4,
                         metallic=0.1, roughness=0.25),
        heptagon=_accent((0.05, 0.35, 0.9, 1.0), (0.1, 0.45, 1.0), 1.4,
                         metallic=0.1, roughness=0.25),
        octagon=_accent((0.9, 0.8, 0.05, 1.0), (1.0, 0.85, 0.05), 1.8,
                        metallic=0.1, roughness=0.25),
        bond=MaterialSpec(base_color=(0.02, 0.02, 0.025, 1.0), metallic=0.7,
                          roughness=0.3),
        lights=[
            LightSpec("AREA", 1400.0, (1.0, 0.93, 0.82), (6.0, -8.0, 7.0), size=3.0),
            LightSpec("AREA", 500.0, (0.5, 0.7, 1.0), (-8.0, 4.0, 2.0), size=4.0),
        ],
        camera_lens_mm=85.0,
        camera_dof_fstop=1.8,
    ),
    "acs_nano_vivid": Style(
        name="acs_nano_vivid",
        description=(
            "Deep blue-to-magenta gradient backdrop, glossy clear-coated "
            "colour with vivid emissive defect highlights and punchy "
            "three-point lighting -- the saturated, energetic look common "
            "on ACS Nano covers."
        ),
        background=Background(kind="gradient", top=(0.05, 0.02, 0.25),
                              bottom=(0.35, 0.02, 0.4), strength=0.6),
        hexagon=MaterialSpec(
            base_color=(0.05, 0.5, 0.9, 1.0), metallic=0.2, roughness=0.15,
            clearcoat=1.0,
        ),
        pentagon=_accent((1.0, 0.2, 0.5, 1.0), (1.0, 0.15, 0.55), 2.2,
                         metallic=0.1, roughness=0.1, clearcoat=1.0),
        heptagon=_accent((0.2, 1.0, 0.4, 1.0), (0.25, 1.0, 0.45), 2.2,
                         metallic=0.1, roughness=0.1, clearcoat=1.0),
        octagon=_accent((1.0, 0.85, 0.1, 1.0), (1.0, 0.85, 0.15), 2.6,
                        metallic=0.1, roughness=0.1, clearcoat=1.0),
        bond=MaterialSpec(base_color=(0.08, 0.35, 0.65, 1.0), metallic=0.3,
                          roughness=0.2),
        lights=[
            LightSpec("AREA", 1800.0, (1.0, 0.4, 0.7), (7.0, -7.0, 6.0), size=3.0),
            LightSpec("AREA", 1200.0, (0.3, 0.7, 1.0), (-7.0, -3.0, 4.0), size=3.0),
            LightSpec("AREA", 400.0, (1.0, 1.0, 1.0), (0.0, 8.0, 3.0), size=5.0),
        ],
        camera_lens_mm=70.0,
        camera_dof_fstop=3.2,
    ),
    "small_minimal": Style(
        name="small_minimal",
        description=(
            "Clean white seamless background, large soft studio lights, "
            "matte pastel body -- the airy, minimal, editorial look used "
            "for schematic-adjacent covers in Small / Advanced Materials."
        ),
        background=Background(kind="gradient", top=(0.97, 0.97, 0.98),
                              bottom=(0.88, 0.89, 0.92), strength=1.4),
        hexagon=MaterialSpec(base_color=(0.75, 0.78, 0.82, 1.0), roughness=0.55),
        pentagon=MaterialSpec(base_color=(0.9, 0.35, 0.3, 1.0), roughness=0.45),
        heptagon=MaterialSpec(base_color=(0.25, 0.45, 0.85, 1.0), roughness=0.45),
        octagon=MaterialSpec(base_color=(0.95, 0.7, 0.15, 1.0), roughness=0.45),
        bond=MaterialSpec(base_color=(0.6, 0.62, 0.66, 1.0), roughness=0.5),
        lights=[
            LightSpec("AREA", 900.0, (1.0, 1.0, 1.0), (5.0, -6.0, 8.0), size=6.0),
            LightSpec("AREA", 600.0, (1.0, 1.0, 1.0), (-6.0, -4.0, 5.0), size=6.0),
            LightSpec("AREA", 300.0, (1.0, 1.0, 1.0), (0.0, 6.0, 2.0), size=6.0),
        ],
        camera_lens_mm=60.0,
        camera_dof_fstop=None,
    ),
    "blueprint_technical": Style(
        name="blueprint_technical",
        description=(
            "Deep navy void with thin glowing emissive bonds and defect "
            "rings picked out like a circuit diagram -- a technical / "
            "schematic-illustration mood for a diagram-style cover."
        ),
        background=Background(kind="gradient", top=(0.0, 0.01, 0.04),
                              bottom=(0.0, 0.0, 0.02), strength=0.03),
        hexagon=_accent((0.02, 0.15, 0.35, 1.0), (0.1, 0.55, 1.0), 0.6,
                        roughness=0.3),
        pentagon=_accent((0.35, 0.02, 0.05, 1.0), (1.0, 0.2, 0.15), 2.2,
                         roughness=0.3),
        heptagon=_accent((0.02, 0.35, 0.1, 1.0), (0.2, 1.0, 0.4), 2.2,
                         roughness=0.3),
        octagon=_accent((0.35, 0.3, 0.02, 1.0), (1.0, 0.85, 0.15), 2.4,
                        roughness=0.3),
        bond=_accent((0.05, 0.4, 0.9, 1.0), (0.15, 0.6, 1.0), 1.0,
                     roughness=0.3),
        lights=[
            LightSpec("AREA", 300.0, (0.4, 0.7, 1.0), (4.0, -6.0, 6.0), size=4.0),
        ],
        camera_lens_mm=50.0,
        camera_dof_fstop=None,
    ),
    "gold_nanotech": Style(
        name="gold_nanotech",
        description=(
            "Warm near-black background, polished gold/bronze metal body "
            "under a three-point studio rig -- the precious-metal "
            "'nanotech jewel' look seen on many materials-chemistry covers."
        ),
        background=Background(kind="gradient", top=(0.02, 0.015, 0.01),
                              bottom=(0.05, 0.03, 0.01), strength=0.15),
        hexagon=MaterialSpec(base_color=(0.83, 0.68, 0.22, 1.0), metallic=1.0,
                             roughness=0.18),
        pentagon=MaterialSpec(base_color=(0.95, 0.95, 0.95, 1.0), metallic=1.0,
                              roughness=0.12),
        heptagon=MaterialSpec(base_color=(0.75, 0.35, 0.12, 1.0), metallic=1.0,
                              roughness=0.15),
        octagon=_accent((0.85, 0.1, 0.1, 1.0), (0.8, 0.1, 0.05), 0.6,
                        metallic=0.7, roughness=0.2),
        bond=MaterialSpec(base_color=(0.6, 0.48, 0.15, 1.0), metallic=1.0,
                          roughness=0.2),
        lights=[
            LightSpec("AREA", 2000.0, (1.0, 0.85, 0.6), (6.0, -7.0, 7.0), size=2.5),
            LightSpec("AREA", 700.0, (0.7, 0.8, 1.0), (-7.0, 3.0, 3.0), size=4.0),
            LightSpec("AREA", 300.0, (1.0, 0.9, 0.7), (0.0, 7.0, -2.0), size=5.0),
        ],
        camera_lens_mm=90.0,
        camera_dof_fstop=2.0,
    ),
    # ---------------- added, and deliberately not cover art -----------
    "paper_figure": Style(
        name="paper_figure",
        description=(
            "Flat white background, element colours, soft even light and "
            "no depth of field -- a figure for a manuscript rather than a "
            "cover. The one style that defaults to colouring by ELEMENT, "
            "because that is what a figure caption refers to, and the one "
            "with a solid rather than a graded backdrop, because a "
            "gradient behind a figure reads as decoration."
        ),
        background=Background(kind="solid", top=(1.0, 1.0, 1.0),
                              bottom=(1.0, 1.0, 1.0), strength=1.6),
        hexagon=MaterialSpec(base_color=(0.35, 0.35, 0.35, 1.0),
                             roughness=0.45, sheen=0.1),
        pentagon=MaterialSpec(base_color=(0.85, 0.30, 0.25, 1.0), roughness=0.45),
        heptagon=MaterialSpec(base_color=(0.20, 0.40, 0.80, 1.0), roughness=0.45),
        octagon=MaterialSpec(base_color=(0.95, 0.70, 0.15, 1.0), roughness=0.45),
        bond=MaterialSpec(base_color=(0.55, 0.55, 0.58, 1.0), roughness=0.5),
        lights=[
            LightSpec("AREA", 700.0, (1.0, 1.0, 1.0), (5.0, -5.0, 7.0), size=8.0),
            LightSpec("AREA", 500.0, (1.0, 1.0, 1.0), (-5.0, -5.0, 4.0), size=8.0),
            LightSpec("AREA", 250.0, (1.0, 1.0, 1.0), (0.0, 7.0, 1.0), size=8.0),
        ],
        camera_lens_mm=55.0,
        camera_dof_fstop=None,
        samples=128,
        default_colour_by="element",
    ),
    "electron_micrograph": Style(
        name="electron_micrograph",
        description=(
            "Greyscale, hard raking light and a dark grey field -- the look "
            "of a secondary-electron image rather than of a drawing. No "
            "colour at all, so it defaults to UNIFORM: a micrograph that "
            "colour-codes its defects is a drawing again."
        ),
        background=Background(kind="solid", top=(0.06, 0.06, 0.06),
                              bottom=(0.06, 0.06, 0.06), strength=0.25),
        hexagon=MaterialSpec(base_color=(0.62, 0.62, 0.62, 1.0), roughness=0.62),
        pentagon=MaterialSpec(base_color=(0.70, 0.70, 0.70, 1.0), roughness=0.62),
        heptagon=MaterialSpec(base_color=(0.54, 0.54, 0.54, 1.0), roughness=0.62),
        octagon=MaterialSpec(base_color=(0.78, 0.78, 0.78, 1.0), roughness=0.62),
        bond=MaterialSpec(base_color=(0.50, 0.50, 0.50, 1.0), roughness=0.66),
        lights=[
            LightSpec("AREA", 2200.0, (1.0, 1.0, 1.0), (9.0, -2.0, 2.0), size=1.2),
            LightSpec("AREA", 180.0, (1.0, 1.0, 1.0), (-6.0, -6.0, 5.0), size=7.0),
        ],
        camera_lens_mm=80.0,
        camera_dof_fstop=5.6,
        samples=192,
        default_colour_by="uniform",
    ),
    "glass_cryo": Style(
        name="glass_cryo",
        description=(
            "Refractive glass body on a cold graded field, so the far side "
            "of a cage shows through its near side. Uses the GLASS shader, "
            "which a Principled BSDF cannot express by parameter alone -- "
            "and it is the style that most rewards a low sample count being "
            "raised, caustics being slow to converge."
        ),
        background=Background(kind="gradient", top=(0.16, 0.22, 0.32),
                              bottom=(0.04, 0.06, 0.10), strength=0.9),
        hexagon=MaterialSpec(base_color=(0.80, 0.90, 1.00, 1.0), shader="glass",
                             roughness=0.05, transmission=1.0, ior=1.45),
        pentagon=MaterialSpec(base_color=(1.00, 0.72, 0.60, 1.0), shader="glass",
                              roughness=0.05, transmission=1.0, ior=1.5),
        heptagon=MaterialSpec(base_color=(0.60, 0.85, 1.00, 1.0), shader="glass",
                              roughness=0.05, transmission=1.0, ior=1.5),
        octagon=MaterialSpec(base_color=(1.00, 0.95, 0.65, 1.0), shader="glass",
                             roughness=0.05, transmission=1.0, ior=1.5),
        bond=MaterialSpec(base_color=(0.85, 0.93, 1.00, 1.0), shader="glass",
                          roughness=0.08, transmission=1.0, ior=1.45),
        lights=[
            LightSpec("AREA", 2600.0, (1.0, 0.98, 0.95), (6.0, -7.0, 8.0), size=4.0),
            LightSpec("AREA", 1200.0, (0.6, 0.8, 1.0), (-7.0, 2.0, 3.0), size=5.0),
            LightSpec("AREA", 600.0, (1.0, 1.0, 1.0), (0.0, 6.0, -4.0), size=6.0),
        ],
        camera_lens_mm=75.0,
        camera_dof_fstop=4.0,
        samples=512,
    ),
    "clay_studio": Style(
        name="clay_studio",
        description=(
            "Matte unglazed clay on a seamless studio sweep with a FLOOR, "
            "so the subject sits on something and casts onto it. Purely "
            "diffuse -- the CLAY shader has no specular at all -- which is "
            "the standard way to show a shape without its material "
            "distracting from the form."
        ),
        background=Background(
            kind="studio", top=(0.90, 0.89, 0.87), bottom=(0.74, 0.72, 0.70),
            strength=1.1, floor=True, floor_color=(0.82, 0.80, 0.78),
            floor_roughness=0.75,
        ),
        hexagon=MaterialSpec(base_color=(0.79, 0.74, 0.69, 1.0), shader="clay",
                             roughness=0.85, sheen=0.2),
        pentagon=MaterialSpec(base_color=(0.84, 0.56, 0.48, 1.0), shader="clay",
                              roughness=0.85),
        heptagon=MaterialSpec(base_color=(0.52, 0.62, 0.74, 1.0), shader="clay",
                              roughness=0.85),
        octagon=MaterialSpec(base_color=(0.86, 0.76, 0.50, 1.0), shader="clay",
                             roughness=0.85),
        bond=MaterialSpec(base_color=(0.74, 0.70, 0.66, 1.0), shader="clay",
                          roughness=0.9),
        lights=[
            LightSpec("AREA", 1100.0, (1.0, 0.99, 0.97), (5.0, -6.0, 8.0), size=7.0),
            LightSpec("AREA", 500.0, (0.98, 0.98, 1.0), (-6.0, -3.0, 4.0), size=7.0),
        ],
        camera_lens_mm=65.0,
        camera_dof_fstop=6.0,
        samples=192,
    ),
    "neon_cyber": Style(
        name="neon_cyber",
        description=(
            "Black field, saturated emissive body and bonds that are "
            "themselves light sources -- the structure glows rather than "
            "being lit. Needs almost no lighting rig, which is why it has "
            "one small fill light: everything visible is emission."
        ),
        background=Background(kind="gradient", top=(0.02, 0.0, 0.05),
                              bottom=(0.0, 0.0, 0.0), strength=0.02),
        hexagon=MaterialSpec(base_color=(0.05, 0.95, 0.85, 1.0),
                             shader="emission", emission_color=(0.05, 0.95, 0.85),
                             emission_strength=2.6),
        pentagon=MaterialSpec(base_color=(1.0, 0.15, 0.55, 1.0),
                              shader="emission", emission_color=(1.0, 0.15, 0.55),
                              emission_strength=5.0),
        heptagon=MaterialSpec(base_color=(0.65, 0.25, 1.0, 1.0),
                              shader="emission", emission_color=(0.65, 0.25, 1.0),
                              emission_strength=5.0),
        octagon=MaterialSpec(base_color=(1.0, 0.85, 0.1, 1.0),
                             shader="emission", emission_color=(1.0, 0.85, 0.1),
                             emission_strength=6.0),
        bond=MaterialSpec(base_color=(0.1, 0.55, 0.95, 1.0), shader="emission",
                          emission_color=(0.1, 0.55, 0.95),
                          emission_strength=1.8),
        lights=[
            LightSpec("AREA", 120.0, (0.4, 0.6, 1.0), (5.0, -6.0, 5.0), size=5.0),
        ],
        camera_lens_mm=70.0,
        camera_dof_fstop=2.2,
        samples=256,
    ),
    "toon_schematic": Style(
        name="toon_schematic",
        description=(
            "Flat banded shading on a pale field -- the TOON shader, for a "
            "diagram that reads at thumbnail size or in a slide. Flat "
            "colour carries structure poorly, so this defaults to colouring "
            "by ELEMENT, where a handful of flat hues is exactly the point."
        ),
        background=Background(kind="solid", top=(0.95, 0.96, 0.98),
                              bottom=(0.95, 0.96, 0.98), strength=1.3),
        hexagon=MaterialSpec(base_color=(0.30, 0.34, 0.40, 1.0), shader="toon",
                             roughness=0.5),
        pentagon=MaterialSpec(base_color=(0.88, 0.32, 0.28, 1.0), shader="toon",
                              roughness=0.5),
        heptagon=MaterialSpec(base_color=(0.22, 0.46, 0.84, 1.0), shader="toon",
                              roughness=0.5),
        octagon=MaterialSpec(base_color=(0.96, 0.72, 0.16, 1.0), shader="toon",
                             roughness=0.5),
        bond=MaterialSpec(base_color=(0.45, 0.48, 0.54, 1.0), shader="toon",
                          roughness=0.5),
        lights=[
            LightSpec("SUN", 4.0, (1.0, 1.0, 1.0), (6.0, -6.0, 9.0)),
            LightSpec("AREA", 300.0, (1.0, 1.0, 1.0), (-6.0, -4.0, 3.0), size=8.0),
        ],
        camera_lens_mm=55.0,
        camera_dof_fstop=None,
        samples=96,
        default_colour_by="element",
    ),
    "transparent_overlay": Style(
        name="transparent_overlay",
        description=(
            "No background at all -- alpha straight through, with a "
            "shadow-catching floor so the subject keeps its contact shadow "
            "when it is composited into a slide or a layout. Element "
            "colours and a neutral finish, since whatever it lands on "
            "supplies the mood."
        ),
        background=Background(
            kind="transparent", top=(0.5, 0.5, 0.5), bottom=(0.5, 0.5, 0.5),
            strength=1.0, floor=True, floor_is_shadow_catcher=True,
        ),
        hexagon=MaterialSpec(base_color=(0.40, 0.40, 0.42, 1.0), roughness=0.4,
                             clearcoat=0.3),
        pentagon=MaterialSpec(base_color=(0.86, 0.32, 0.26, 1.0), roughness=0.35),
        heptagon=MaterialSpec(base_color=(0.22, 0.44, 0.82, 1.0), roughness=0.35),
        octagon=MaterialSpec(base_color=(0.95, 0.72, 0.16, 1.0), roughness=0.35),
        bond=MaterialSpec(base_color=(0.55, 0.55, 0.58, 1.0), roughness=0.45),
        lights=[
            LightSpec("AREA", 900.0, (1.0, 1.0, 1.0), (5.0, -6.0, 8.0), size=7.0),
            LightSpec("AREA", 450.0, (1.0, 1.0, 1.0), (-6.0, -4.0, 4.0), size=7.0),
        ],
        camera_lens_mm=60.0,
        camera_dof_fstop=None,
        samples=160,
        default_colour_by="element",
    ),

    # ------------------------------------------------------------------
    # Cover art. These bring their own effects, because a style that is
    # *about* a laser should arrive with the laser rather than needing it
    # typed in beside the style name. --no-style-effects drops them.
    # ------------------------------------------------------------------
    "cosmos": Style(
        name="cosmos",
        description=(
            "The structure alone in deep space: a procedural starfield "
            "with a nebula behind it, polished gold with a tinted clear "
            "coat, and bloom. The scale joke a nanoscience cover is "
            "usually making -- an object a nanometre across photographed "
            "as though it were an astronomical one."
        ),
        background=Background(
            kind="nebula", top=(0.01, 0.01, 0.03), bottom=(0.0, 0.0, 0.01),
            strength=1.0, star_density=260.0, star_amount=0.035,
            star_brightness=22.0,
            nebula_colors=((0.10, 0.02, 0.20), (0.00, 0.08, 0.18)),
            nebula_strength=0.8,
        ),
        hexagon=MaterialSpec(base_color=(0.86, 0.66, 0.24, 1.0),
                             metallic=1.0, roughness=0.22, clearcoat=0.4,
                             coat_tint=(1.0, 0.92, 0.75),
                             specular_tint=(1.0, 0.88, 0.62)),
        pentagon=_accent((1.0, 0.45, 0.15, 1.0), (1.0, 0.45, 0.10), 2.5,
                         metallic=0.9, roughness=0.25),
        heptagon=_accent((0.35, 0.65, 1.0, 1.0), (0.25, 0.60, 1.0), 2.5,
                         metallic=0.9, roughness=0.25),
        octagon=_accent((1.0, 0.25, 0.55, 1.0), (1.0, 0.20, 0.50), 3.0,
                        metallic=0.9, roughness=0.25),
        bond=MaterialSpec(base_color=(0.70, 0.54, 0.20, 1.0), metallic=1.0,
                          roughness=0.28, specular_tint=(1.0, 0.88, 0.62)),
        lights=[
            LightSpec("AREA", 2200.0, (1.00, 0.95, 0.88), (8.0, -9.0, 7.0),
                      size=6.0),
            LightSpec("AREA", 700.0, (0.55, 0.70, 1.00), (-9.0, 4.0, -3.0),
                      size=9.0),
        ],
        camera_lens_mm=85.0,
        camera_dof_fstop=3.2,
        samples=320,
        default_colour_by="ring",
        effects=(EffectSpec("glare", {"strength": 0.45, "threshold": 0.7}),),
    ),

    "plasmon": Style(
        name="plasmon",
        description=(
            "Resonance made visible: brushed anisotropic metal under a "
            "cold rim light, with standing-wave shells around the "
            "structure and bloom. For a paper about an optical or "
            "vibrational mode, where the point of the picture is that "
            "something is RINGING rather than merely sitting there."
        ),
        background=Background(
            kind="gradient", top=(0.02, 0.03, 0.06),
            bottom=(0.00, 0.00, 0.01), strength=1.0,
        ),
        hexagon=MaterialSpec(base_color=(0.72, 0.75, 0.80, 1.0),
                             metallic=1.0, roughness=0.30,
                             anisotropy=0.75, anisotropy_rotation=0.25),
        pentagon=_accent((0.20, 0.95, 1.00, 1.0), (0.15, 0.90, 1.00), 4.0),
        heptagon=_accent((1.00, 0.30, 0.60, 1.0), (1.00, 0.25, 0.55), 4.0),
        octagon=_accent((1.00, 0.80, 0.20, 1.0), (1.00, 0.75, 0.15), 4.5),
        bond=MaterialSpec(base_color=(0.55, 0.58, 0.64, 1.0), metallic=1.0,
                          roughness=0.35, anisotropy=0.6),
        lights=[
            LightSpec("AREA", 1500.0, (0.80, 0.90, 1.00), (7.0, -8.0, 6.0),
                      size=5.0),
            LightSpec("AREA", 900.0, (0.30, 0.60, 1.00), (-8.0, 5.0, 2.0),
                      size=8.0),
        ],
        camera_lens_mm=80.0,
        camera_dof_fstop=3.5,
        samples=288,
        default_colour_by="ring",
        effects=(
            EffectSpec("resonance", {"count": 5, "color": "cyan"}),
            EffectSpec("glare", {"strength": 0.4}),
        ),
    ),

    "laser_probe": Style(
        name="laser_probe",
        description=(
            "A beam striking the structure in a hazy dark room, with the "
            "shaft visible in the air. For spectroscopy, ablation, "
            "photocatalysis -- anything where light is being DONE to the "
            "sample rather than merely used to see it."
        ),
        background=Background(
            kind="gradient", top=(0.02, 0.02, 0.03),
            bottom=(0.00, 0.00, 0.00), strength=0.8,
        ),
        hexagon=MaterialSpec(base_color=(0.10, 0.10, 0.11, 1.0),
                             roughness=0.42, clearcoat=0.25),
        pentagon=_accent((1.00, 0.45, 0.10, 1.0), (1.00, 0.40, 0.08), 3.0),
        heptagon=_accent((0.20, 0.80, 1.00, 1.0), (0.15, 0.75, 1.00), 3.0),
        octagon=_accent((0.85, 0.30, 1.00, 1.0), (0.80, 0.25, 1.00), 3.5),
        bond=MaterialSpec(base_color=(0.16, 0.16, 0.18, 1.0), roughness=0.5),
        lights=[
            LightSpec("AREA", 600.0, (0.85, 0.90, 1.00), (6.0, -7.0, 5.0),
                      size=5.0),
        ],
        camera_lens_mm=75.0,
        camera_dof_fstop=2.8,
        samples=384,
        default_colour_by="ring",
        effects=(
            EffectSpec("laser", {"color": "laser-green", "azimuth": 150.0}),
            EffectSpec("fog", {"strength": 0.004}),
            EffectSpec("glare", {"strength": 0.5, "threshold": 0.65}),
        ),
    ),

    "electric_storm": Style(
        name="electric_storm",
        description=(
            "Discharge: filaments arcing over the structure against "
            "black, with a faint halo. For conductivity, field emission, "
            "dielectric breakdown, or any cover that wants the subject "
            "to look energetic rather than inert."
        ),
        background=Background(
            kind="solid", top=(0.005, 0.008, 0.015),
            bottom=(0.005, 0.008, 0.015), strength=1.0,
        ),
        hexagon=MaterialSpec(base_color=(0.12, 0.13, 0.16, 1.0),
                             metallic=0.6, roughness=0.35, clearcoat=0.3),
        pentagon=_accent((0.40, 0.80, 1.00, 1.0), (0.35, 0.75, 1.00), 5.0),
        heptagon=_accent((0.85, 0.45, 1.00, 1.0), (0.80, 0.40, 1.00), 5.0),
        octagon=_accent((1.00, 0.90, 0.40, 1.0), (1.00, 0.85, 0.35), 5.5),
        bond=MaterialSpec(base_color=(0.18, 0.20, 0.24, 1.0), metallic=0.6,
                          roughness=0.4),
        lights=[
            LightSpec("AREA", 500.0, (0.60, 0.80, 1.00), (6.0, -7.0, 4.0),
                      size=6.0),
        ],
        camera_lens_mm=70.0,
        camera_dof_fstop=3.0,
        samples=288,
        default_colour_by="ring",
        effects=(
            EffectSpec("arc", {"count": 9, "color": "electric"}),
            EffectSpec("halo", {"strength": 0.6, "color": "electric"}),
            EffectSpec("glare", {"strength": 0.5, "threshold": 0.6}),
        ),
    ),

    "iridescent": Style(
        name="iridescent",
        description=(
            "Thin-film interference: a soap-bubble or anodised-titanium "
            "sheen whose colour CHANGES WITH THE VIEWING ANGLE, because "
            "it is computed as interference rather than painted on with "
            "a colour ramp. Needs Blender 4.3 or newer for the thin-film "
            "sockets; on an older build it renders as plain lacquer."
        ),
        background=Background(
            kind="studio", top=(0.10, 0.11, 0.14), bottom=(0.03, 0.03, 0.05),
            strength=0.9, floor=True, floor_color=(0.07, 0.07, 0.09),
            floor_roughness=0.25,
        ),
        hexagon=MaterialSpec(base_color=(0.05, 0.05, 0.06, 1.0),
                             metallic=1.0, roughness=0.12,
                             thin_film=430.0, thin_film_ior=1.45),
        pentagon=MaterialSpec(base_color=(0.05, 0.05, 0.06, 1.0),
                              metallic=1.0, roughness=0.10,
                              thin_film=620.0, thin_film_ior=1.45),
        heptagon=MaterialSpec(base_color=(0.05, 0.05, 0.06, 1.0),
                              metallic=1.0, roughness=0.10,
                              thin_film=280.0, thin_film_ior=1.45),
        octagon=MaterialSpec(base_color=(0.05, 0.05, 0.06, 1.0),
                             metallic=1.0, roughness=0.10,
                             thin_film=520.0, thin_film_ior=1.45),
        bond=MaterialSpec(base_color=(0.05, 0.05, 0.06, 1.0), metallic=1.0,
                          roughness=0.18, thin_film=380.0),
        lights=[
            LightSpec("AREA", 1800.0, (1.0, 1.0, 1.0), (7.0, -8.0, 7.0),
                      size=9.0),
            LightSpec("AREA", 900.0, (0.9, 0.95, 1.0), (-8.0, -3.0, 4.0),
                      size=9.0),
            LightSpec("AREA", 600.0, (1.0, 0.95, 0.9), (0.0, 9.0, 3.0),
                      size=9.0),
        ],
        camera_lens_mm=85.0,
        camera_dof_fstop=4.0,
        samples=320,
        default_colour_by="ring",
        effects=(EffectSpec("glare", {"strength": 0.25}),),
    ),
}


def list_styles() -> list[str]:
    """Return the available style names, for ``--style`` help text."""
    return sorted(STYLES)


def get_style(name: str) -> Style:
    """Look up a style by name, raising a friendly error on a typo."""
    try:
        return STYLES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown style {name!r}. Available: {', '.join(list_styles())}."
        ) from exc


def with_background(style: Style, kind: str) -> Style:
    """The same style over a different backdrop.

    The point of :class:`Background` being its own object: "this style, on
    white, for the paper" and "this style, on black, for the cover" are
    two uses of one style rather than two styles. A ``studio`` request
    keeps the style's own floor colours if it has them and falls back to
    a neutral grey if not.
    """
    current = style.background
    if kind == "transparent":
        background = Background(kind="transparent", top=current.top,
                                bottom=current.bottom,
                                strength=current.strength,
                                floor=current.floor,
                                floor_color=current.floor_color,
                                floor_roughness=current.floor_roughness,
                                floor_is_shadow_catcher=True)
    elif kind == "solid":
        background = Background(kind="solid", top=current.top,
                                bottom=current.top, strength=current.strength)
    elif kind == "studio":
        background = Background(kind="studio", top=current.top,
                                bottom=current.bottom,
                                strength=max(current.strength, 0.6),
                                floor=True,
                                floor_color=current.floor_color,
                                floor_roughness=current.floor_roughness)
    elif kind in ("space", "nebula"):
        # A starfield keeps the style's own darkness: `nature_dark` in
        # space should still be nearly black behind the stars, and
        # `small_minimal` in space is a contradiction the caller is
        # entitled to ask for and see.
        background = Background(kind=kind, top=current.top,
                                bottom=current.bottom,
                                strength=current.strength)
    elif kind == "gradient":
        background = Background(kind="gradient", top=current.top,
                                bottom=current.bottom,
                                strength=current.strength)
    else:
        raise ValueError(
            f"unknown background {kind!r}; known: {', '.join(BACKGROUNDS)}"
        )
    return replace(style, background=background)


__all__ = [
    "BACKGROUNDS",
    "COLOUR_MODES",
    "DEFAULT_ELEMENT_RADIUS",
    "ELEMENT_COLOURS",
    "ELEMENT_RADII",
    "FINISHES",
    "SHADERS",
    "SPLIT_BONDS",
    "STRUCTURE_PALETTE",
    "STYLES",
    "UNKNOWN_ELEMENT_COLOUR",
    "Background",
    "LightSpec",
    "MaterialSpec",
    "Style",
    "apply_finish",
    "describe_finishes",
    "element_colour",
    "element_radius",
    "get_style",
    "list_styles",
    "materials_for",
    "resolve_colour_by",
    "structure_colour",
    "with_background",
]
