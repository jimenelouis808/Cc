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
    #: **Rim light.** Emission added only where the surface turns away
    #: from the camera, by Fresnel. This is the single most recognisable
    #: thing about a cover render and it cannot be done with a light: a
    #: rim lamp draws a bright edge on *one* side, while this draws one
    #: on every silhouette the viewer can see, including the inside of a
    #: cage seen through its own far wall. 0 is off.
    rim: float = 0.0
    rim_color: RGB = (1.0, 1.0, 1.0)
    #: How sharply the rim falls off. Higher is a thinner, harder edge;
    #: 1.0 is a soft wrap that reads as haze rather than as an edge.
    rim_power: float = 3.0

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
    #: A floor that reflects. At 1.0 with a low roughness it is a
    #: mirror, which is what a product shot stands its subject on.
    floor_metallic: float = 0.0
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
    #: How fat the atoms and the sticks are drawn, as multipliers.
    #: A model kit's balls are nowhere near scale -- that is most of
    #: what makes a kit read as a kit rather than as a diagram -- while
    #: a dense lattice needs them small or the picture becomes a wall.
    #: ``--atom-scale`` and ``--bond-scale`` still multiply these.
    atom_scale: float = 1.0
    bond_scale: float = 1.0
    #: The representation this style was tuned for. A model-kit look is
    #: ball-and-stick by definition and a metal product shot is
    #: space-filling; ``--model`` still overrides it.
    default_model: str = "ballstick"
    #: Skin radius for the ``surface`` model, in Angstrom. The default
    #: 1.6 is wider than a graphene hexagon's circumradius (1.42), so
    #: adjacent struts merge and every pore closes over -- right for a
    #: solid wide shot and wrong for a porous net, whose holes are the
    #: subject. A style whose point is the holes sets it well under 1.0.
    skin_radius: float = 1.6
    #: Overrides :data:`DEPTH_RAMP` for ``--colour-by depth``. Left
    #: empty the viridis ramp is used, which is monotone in brightness
    #: and survives a greyscale print; a hue-cycling ramp does not, so
    #: a style that sets one is a cover style and says so.
    depth_ramp: tuple = ()
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
#: ``depth``     -- position along the structure's own longest axis,
#:                  as a gradient. The one mode that says anything on an
#:                  all-carbon superstructure, where element is a single
#:                  colour and ring is nearly all hexagons -- and the
#:                  standard way a journal figure shows the inside of a
#:                  porous solid.
COLOUR_MODES = ("ring", "element", "structure", "uniform", "depth")

#: How many bands a depth gradient is quantised into. Material slots are
#: discrete, so this is a real limit rather than a setting: 24 bands is
#: past where the eye sees stepping on a smooth ramp.
DEPTH_BANDS = 24

#: The ramp a depth gradient runs along. Viridis-like: perceptually
#: even and legible in greyscale, which a rainbow is not -- a figure
#: printed in black and white must not lose its ordering.
DEPTH_RAMP: tuple[RGB, ...] = (
    (0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150), (0.993, 0.906, 0.144),
)


def depth_colour(fraction: float, ramp: tuple = ()) -> RGB:
    """A colour from a depth ramp, interpolated.

    ``ramp`` defaults to :data:`DEPTH_RAMP`, which is viridis-like and
    monotone in brightness so a greyscale print keeps its ordering. A
    style may override it (:attr:`Style.depth_ramp`) when the figure is
    a cover rather than a measurement -- an iridescent ramp cycles in
    hue and is the better picture, and the worse data.
    """
    ramp = ramp or DEPTH_RAMP
    fraction = min(1.0, max(0.0, float(fraction)))
    position = fraction * (len(ramp) - 1)
    low = int(position)
    high = min(low + 1, len(ramp) - 1)
    weight = position - low
    return tuple(ramp[low][channel] * (1.0 - weight)
                 + ramp[high][channel] * weight
                 for channel in range(3))      # type: ignore[return-value]


#: Oil-on-water: the hue cycles rather than climbing, which is what
#: makes it read as interference instead of as a scale.
IRIDESCENT_RAMP: tuple[RGB, ...] = (
    (0.286, 0.098, 0.502), (0.180, 0.247, 0.733), (0.098, 0.482, 0.800),
    (0.125, 0.698, 0.694), (0.263, 0.804, 0.459), (0.631, 0.847, 0.278),
    (0.953, 0.776, 0.251), (0.976, 0.514, 0.306), (0.898, 0.278, 0.482),
    (0.667, 0.208, 0.639), (0.400, 0.157, 0.627),
)

#: Aurora: green through teal into magenta, on a dark sky.
AURORA_RAMP: tuple[RGB, ...] = (
    (0.043, 0.220, 0.247), (0.047, 0.420, 0.392), (0.110, 0.655, 0.471),
    (0.306, 0.839, 0.502), (0.569, 0.922, 0.616), (0.745, 0.890, 0.812),
    (0.733, 0.714, 0.918), (0.667, 0.478, 0.890), (0.624, 0.282, 0.776),
    (0.510, 0.157, 0.580), (0.329, 0.090, 0.396),
)

#: Hot metal: a thermal scale, deep violet through red into white.
PLASMA_RAMP: tuple[RGB, ...] = (
    (0.051, 0.030, 0.165), (0.247, 0.012, 0.420), (0.447, 0.004, 0.471),
    (0.627, 0.090, 0.412), (0.780, 0.216, 0.318), (0.890, 0.357, 0.220),
    (0.961, 0.514, 0.129), (0.988, 0.682, 0.078), (0.969, 0.839, 0.157),
    (0.941, 0.925, 0.400), (0.976, 0.976, 0.749),
)

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

#: **Van der Waals radii, in Angstrom.** These are the real ones (Bondi,
#: and Alvarez 2013 for the metals), not display radii: a space-filling
#: model is defined by atoms touching at their vdW contact, so scaling
#: them for taste would stop it being a space-filling model.
VDW_RADII: dict[str, float] = {
    "H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80,
    "Cl": 1.75, "K": 2.75, "Ca": 2.31, "Ti": 2.15, "V": 2.05, "Cr": 2.05,
    "Mn": 2.05, "Fe": 2.04, "Co": 2.00, "Ni": 1.97, "Cu": 1.96, "Zn": 2.01,
    "Ge": 2.11, "Se": 1.90, "Zr": 2.23, "Nb": 2.18, "Mo": 2.17, "Ru": 2.13,
    "Rh": 2.10, "Pd": 2.10, "Ag": 2.11, "Sn": 2.17, "Te": 2.06, "Hf": 2.23,
    "Ta": 2.22, "W": 2.18, "Re": 2.16, "Pt": 2.13, "Au": 2.14, "Pb": 2.02,
    "Bi": 2.07,
}
DEFAULT_VDW_RADIUS = 1.80

#: Elements drawn space-filling by ``--model auto``. A metal has no
#: directional bonds to draw -- a gold cluster's atoms are packed, not
#: bonded in pairs -- so sticks between them would be an invention,
#: while a carbon lattice's bonds are the whole point of the picture.
SPACEFILL_ELEMENTS = frozenset({
    "Na", "Mg", "Al", "K", "Ca", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Sn", "Hf",
    "Ta", "W", "Re", "Pt", "Au", "Pb", "Bi",
})

#: Mesh resolution per atom and per bond. Measured, not estimated: an
#: icosphere is **80** triangles at subdivision 2, 320 at 3, 1280 at 4
#: and 5120 at 5. The pipeline was fixed at 2 -- eighty triangles --
#: which is visibly faceted as soon as an atom is more than a few per
#: cent of the frame, and a close-up of a small molecule is exactly
#: that.
#:
#: The cost is linear in the atom count, so this cannot simply be set
#: high: 2000 atoms at subdivision 5 is ten million triangles before a
#: single bond is drawn. :func:`detail_for` picks from the atom count
#: when the caller does not say.
DETAIL_LEVELS: dict[str, dict[str, int]] = {
    "low":    {"sphere_subdiv": 2, "bond_segments": 8},
    "normal": {"sphere_subdiv": 3, "bond_segments": 12},
    "high":   {"sphere_subdiv": 4, "bond_segments": 20},
    "ultra":  {"sphere_subdiv": 5, "bond_segments": 32},
}

#: Atom counts at or below which each level is chosen automatically.
#: Above the last one, "low" is the only affordable answer -- and at
#: that density an atom is a few pixels across, where subdivision 2 is
#: indistinguishable from 5.
DETAIL_THRESHOLDS = ((120, "ultra"), (600, "high"), (3000, "normal"))


def detail_for(n_atoms: int, requested: str | None = None
               ) -> dict[str, int]:
    """Mesh resolution for a structure of this size.

    Automatic by default, because the right answer genuinely depends on
    the subject: a 60-atom cage fills the frame with twenty-pixel
    spheres and needs every triangle, while in a 2000-atom supernetwork
    an atom is three pixels across and subdivision 5 would be ten
    million triangles that nobody can see.
    """
    if requested:
        try:
            return dict(DETAIL_LEVELS[requested])
        except KeyError:
            raise ValueError(
                f"unknown detail {requested!r}; known: "
                f"{', '.join(DETAIL_LEVELS)}") from None
    for limit, level in DETAIL_THRESHOLDS:
        if n_atoms <= limit:
            return dict(DETAIL_LEVELS[level])
    return dict(DETAIL_LEVELS["low"])


#: How the structure is drawn.
#:
#: ``ballstick``  -- spheres at display radii joined by sticks. The right
#:                   model for a lattice, where the connectivity *is* the
#:                   subject.
#: ``spacefill``  -- spheres at their van der Waals radii and no sticks:
#:                   the shape the molecule actually occupies. The right
#:                   model for a metal particle, which has no pair bonds
#:                   to draw.
#: ``auto``       -- space-filling for metals, ball-and-stick for
#:                   everything else, in the same picture. A carbon-
#:                   coated gold particle is then a solid metal core
#:                   inside an open carbon cage, which is what it is.
#: ``licorice``   -- sticks only, no spheres, for a dense lattice whose
#:                   atoms would otherwise merge into a wall.
#: ``wireframe``  -- thin sticks only.
#: ``surface``    -- one continuous skinned surface over the bond graph,
#:                   no atoms and no sticks. At a thousand atoms a
#:                   ball-and-stick picture is a thicket and the shape
#:                   is what is left to say; this says it.
MODELS = ("ballstick", "spacefill", "auto", "licorice", "wireframe",
          "surface")

#: Fraction of the van der Waals radius a space-filling sphere is drawn
#: at. 1.0 is the strict definition; a hair under keeps neighbouring
#: spheres from z-fighting where a lattice puts them exactly in contact.
SPACEFILL_SCALE = 0.98


def vdw_radius(symbol: str) -> float:
    """Van der Waals radius in Angstrom, with a fallback."""
    return VDW_RADII.get(str(symbol), DEFAULT_VDW_RADIUS)


def is_spacefill_element(symbol: str) -> bool:
    """Would ``--model auto`` draw this element space-filling?"""
    return str(symbol) in SPACEFILL_ELEMENTS


def model_radius(symbol: str, model: str, scale: float = 1.0) -> float:
    """The drawn radius of one atom under a representation.

    ``auto`` is the interesting one and the reason this is per atom
    rather than per render: it asks the element, so a metal core comes
    out solid and the carbon shell around it stays open, in one picture.
    """
    if model not in MODELS:
        raise ValueError(
            f"unknown model {model!r}; known: {', '.join(MODELS)}")
    if model in ("licorice", "wireframe", "surface"):
        return 0.0
    if model == "spacefill" or (model == "auto"
                                and is_spacefill_element(symbol)):
        return vdw_radius(symbol) * SPACEFILL_SCALE * scale
    return element_radius(symbol) * scale


def draws_bonds(model: str) -> bool:
    """Does this representation draw sticks at all?"""
    return model not in ("spacefill", "surface")


def bond_is_drawn(model: str, symbol_a: str, symbol_b: str) -> bool:
    """Is the bond between these two atoms drawn?

    Under ``auto`` a bond with a space-filling atom at either end is
    **not**: it would start inside a sphere of its own radius and come
    out the other side, and for a metal it is an invention anyway. That
    is what makes a gold cluster on a graphene sheet read correctly --
    solid clusters, open lattice, and nothing joining the two.
    """
    if model in ("spacefill", "surface"):
        return False
    if model != "auto":
        return True
    return not (is_spacefill_element(symbol_a)
                or is_spacefill_element(symbol_b))

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
               "depth": True,
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
#: Resolution and sampling as one choice, because they are one
#: decision: a 6000 px frame at 64 samples is a big noisy picture and a
#: 600 px one at 1024 is a slow thumbnail. The long side is given; the
#: short side keeps whatever aspect ``--resolution`` implies.
#:
#: "print" is 300 dpi across a single journal column (~8.5 cm) with room
#: to crop; "cover" is 300 dpi across a full page.
QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "draft":   {"long_side": 640,  "samples": 48},
    "preview": {"long_side": 1280, "samples": 128},
    "print":   {"long_side": 2400, "samples": 512},
    "cover":   {"long_side": 4096, "samples": 1024},
}



def render_cost(resolution, samples: int) -> dict:
    """What a render will COST, stated as work rather than as time.

    Returns the pixel count, the sample count and their product --
    "pixel-samples", the unit the sampling stage actually does -- plus
    the ratio against every quality preset at the same aspect.

    **No seconds.** This module's own history says why: three attempts
    to turn a size into a predicted build time all failed and are pinned
    by tests so they are not tried again, and the render path has the
    same shape. Its time is two terms -- building the scene, which does
    not depend on the resolution at all (measured: 291 s of a 348 s
    draft render of a 1532-atom particle), and sampling, which scales
    with pixel-samples. A single number would hide that a 136x rise in
    pixel-samples is not a 136x rise in wall time.

    What the ratio IS good for is knowing what is being committed to
    before committing to it: cover against draft is 136x the sampling
    work, so a composition settled at draft costs almost nothing and a
    composition settled at cover costs that.
    """
    width, height = (int(resolution[0]), int(resolution[1]))
    pixels = width * height
    samples = int(samples)
    work = pixels * samples
    ratios = {}
    for name, preset in QUALITY_PRESETS.items():
        factor = preset["long_side"] / max(width, height)
        other = (max(16, round(width * factor))
                 * max(16, round(height * factor))
                 * preset["samples"])
        ratios[name] = other / work if work else float("inf")
    return {"width": width, "height": height, "pixels": pixels,
            "samples": samples, "pixel_samples": work, "ratios": ratios}


#: How the bonds are coloured, independently of the atoms.
#:
#: ``atoms``   -- each half takes its own atom's colour, which is what
#:               every molecular viewer does and what makes a C-O bond
#:               stop claiming the oxygen is carbon. Only meaningful
#:               when the colour belongs to the atom; with ring
#:               colouring there are no halves to take.
#: ``style``   -- the style's own single bond material throughout. A
#:               lattice of one element reads better as a frame of one
#:               colour than as a thousand identical halves.
#: ``neutral`` -- one grey, whatever the style and whatever the atoms,
#:               for when the atoms carry all the colour and the bonds
#:               should stay out of the way.
BOND_COLOURS = ("auto", "atoms", "style", "neutral")

#: The grey ``"neutral"`` uses. Mid-tone on purpose: it reads as a frame
#: on a light background and on a dark one.
NEUTRAL_BOND = (0.45, 0.45, 0.47)


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


def bond_materials(style: Style, colour_by: str, keys,
                   bond_colour: str = "auto",
                   finish: str = "style") -> tuple[list[MaterialSpec], bool]:
    """The bond materials, and whether they are split per atom.

    Returns ``(specs, split)``. When ``split`` is true the list matches
    the atoms' own material list one for one, so each half-bond takes
    its atom's slot; when false the list holds exactly one material for
    every bond.

    **The bonds are decided separately from the atoms on purpose.** They
    were not: whenever the colour belonged to the atom the bonds simply
    reused the atoms' material list, so asking for glass atoms gave
    glass bonds and there was no way to say otherwise. A figure often
    wants the opposite -- element-coloured atoms on a neutral frame, or
    glass atoms on metal bonds -- and that is two decisions, not one.

    ``"auto"`` keeps the old behaviour, which is the right default: split
    where the colour belongs to the atom, the style's own bond material
    where it does not.
    """
    if bond_colour not in BOND_COLOURS:
        raise ValueError(
            f"unknown bond colour {bond_colour!r}; known: "
            f"{', '.join(BOND_COLOURS)}")
    if bond_colour == "auto":
        bond_colour = "atoms" if SPLIT_BONDS.get(colour_by, False) else "style"

    if bond_colour == "atoms":
        if not SPLIT_BONDS.get(colour_by, False):
            # Ring colouring has no per-atom colour to halve: an atom on
            # a pentagon and a hexagon has two, and picking one would be
            # arbitrary. Fall through to the style's bond rather than
            # inventing a rule.
            return [apply_finish(style.bond, finish)], False
        return materials_for(style, colour_by, keys, finish), True
    if bond_colour == "neutral":
        return [apply_finish(style.bond.with_colour(NEUTRAL_BOND),
                             finish)], False
    return [apply_finish(style.bond, finish)], False


#: One line per model, for the window's help panel.
MODEL_HELP = {
    "ballstick": "spheres joined by sticks -- for a lattice, where the "
                 "connectivity is the subject",
    "spacefill": "spheres at their van der Waals radii, no sticks -- the "
                 "shape the structure actually occupies",
    "auto": "space-filling for metals, ball-and-stick for everything "
            "else: a coated metal particle as a solid core in an open cage",
    "licorice": "sticks only -- for a dense net whose atoms would hide "
                "the topology",
    "wireframe": "thin sticks only",
    "surface": "one continuous skin over the bond graph -- at a thousand "
               "atoms the thicket of sticks hides the shape, and the "
               "shape is what is left to say",
}

#: One line per colour mode.
COLOUR_HELP = {
    "ring": "ring size: where the curvature and the defects are. Says "
            "nothing about chemistry, and on a closed cage every atom "
            "is on a pentagon",
    "element": "the chemical element. What a doped, functionalised or "
               "encapsulated structure needs; one colour on pure carbon",
    "structure": "which item of a composition an atom came from",
    "uniform": "one colour throughout, for a silhouette",
    "depth": "position along the structure's longest axis, as a "
             "gradient: the one mode that says anything on an "
             "all-carbon net, and how a figure shows inside a porous "
             "solid",
}

#: One line per background.
BACKGROUND_HELP = {
    "gradient": "a vertical ramp between two colours",
    "solid": "one flat colour -- what a figure wants, since a gradient "
             "behind a figure reads as decoration",
    "transparent": "no background: alpha straight through, for "
                   "compositing into a layout",
    "studio": "a gradient plus a floor to sit on and cast onto",
    "space": "a procedural starfield",
    "nebula": "the starfield plus drifting coloured cloud",
}


def style_summary(style: Style) -> str:
    """Everything a style decides, as a few lines of plain text.

    The window shows this under the style menu. A preset is a bundle of
    a dozen choices, and a name plus a sentence of mood does not say
    which ones -- so the ones that change what you see are listed.
    """
    lines = [style.description, ""]
    lines.append(f"backdrop   {style.background.kind}"
                 + (", with a floor" if style.background.has_floor else ""))
    lines.append(f"colour by  {style.default_colour_by} -- "
                 + COLOUR_HELP.get(style.default_colour_by, ""))
    lines.append(f"model      {style.default_model} -- "
                 + MODEL_HELP.get(style.default_model, ""))
    lines.append(f"camera     {style.camera_lens_mm:.0f} mm, "
                 + ("no depth of field"
                    if style.camera_dof_fstop is None
                    else f"f/{style.camera_dof_fstop:g}"))
    lines.append(f"render     {style.engine.title()} at {style.samples} "
                 "samples")
    if style.atom_scale != 1.0 or style.bond_scale != 1.0:
        lines.append(f"scale      atoms x{style.atom_scale:g}, "
                     f"bonds x{style.bond_scale:g}")
    if style.effects:
        lines.append("effects    "
                     + ", ".join(spec.kind for spec in style.effects)
                     + "  (the checkboxes can drop them)")
    else:
        lines.append("effects    none")
    return "\n".join(lines)


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
    elif colour_by == "depth":
        ramp = getattr(style, "depth_ramp", ())
        specs = [style.hexagon.with_colour(
                     depth_colour(int(key) / max(1, DEPTH_BANDS - 1), ramp))
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
                             metallic=0.6, roughness=0.35, clearcoat=0.3,
                             rim=1.6, rim_color=(0.45, 0.75, 1.00),
                             rim_power=4.0),
        pentagon=_accent((0.40, 0.80, 1.00, 1.0), (0.35, 0.75, 1.00), 1.8),
        heptagon=_accent((0.85, 0.45, 1.00, 1.0), (0.80, 0.40, 1.00), 1.8),
        octagon=_accent((1.00, 0.90, 0.40, 1.0), (1.00, 0.85, 0.35), 2.2),
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

    # ------------------------------------------------------------------
    # Journal looks. Each is a different ANSWER to the same question --
    # what does this journal's art direction reward -- rather than five
    # restatements of "dark and glossy". They differ in backdrop, in
    # lighting rig, in lens, in how much of the frame the subject takes
    # and in whether the colour means chemistry or curvature.
    # ------------------------------------------------------------------
    "science_cover": Style(
        name="science_cover",
        description=(
            "High-key, single subject, shallow depth of field on a pale "
            "graded ground: the look Science and Science Advances "
            "reward, where the cover has to read at thumbnail size and "
            "survive a masthead across the top. Long lens, tight crop, "
            "nothing in the frame but the structure and its shadow."
        ),
        background=Background(
            kind="gradient", top=(0.93, 0.94, 0.96),
            bottom=(0.62, 0.66, 0.72), strength=1.0,
            floor=True, floor_color=(0.86, 0.88, 0.91),
            floor_roughness=0.35,
        ),
        hexagon=MaterialSpec(base_color=(0.22, 0.25, 0.30, 1.0),
                             roughness=0.26, clearcoat=0.55,
                             coat_roughness=0.04, specular_tint=(1, 1, 1)),
        pentagon=MaterialSpec(base_color=(0.90, 0.32, 0.20, 1.0),
                              roughness=0.22, clearcoat=0.5),
        heptagon=MaterialSpec(base_color=(0.16, 0.45, 0.80, 1.0),
                              roughness=0.22, clearcoat=0.5),
        octagon=MaterialSpec(base_color=(0.95, 0.72, 0.12, 1.0),
                             roughness=0.22, clearcoat=0.5),
        bond=MaterialSpec(base_color=(0.38, 0.41, 0.46, 1.0),
                          roughness=0.3, clearcoat=0.4),
        lights=[
            LightSpec("AREA", 2600.0, (1.00, 0.99, 0.97), (6.0, -8.0, 9.0),
                      size=12.0),
            LightSpec("AREA", 1100.0, (0.94, 0.96, 1.00), (-9.0, -2.0, 3.0),
                      size=12.0),
            LightSpec("AREA", 700.0, (1.0, 1.0, 1.0), (0.0, 9.0, -2.0),
                      size=10.0),
        ],
        camera_lens_mm=95.0,
        camera_dof_fstop=2.4,
        samples=384,
        default_colour_by="ring",
        effects=(EffectSpec("glare", {"strength": 0.18,
                                      "threshold": 0.88}),),
    ),

    "nature_nano": Style(
        name="nature_nano",
        description=(
            "Nature Nanotechnology's house mood taken seriously: a near-"
            "black field, one hard key from high and behind for a rim, a "
            "cold fill, and a body dark enough that the rim IS the "
            "drawing. The structure is a silhouette with its edges lit, "
            "which is what survives being printed small."
        ),
        background=Background(
            kind="gradient", top=(0.015, 0.02, 0.035),
            bottom=(0.0, 0.0, 0.006), strength=1.0,
        ),
        hexagon=MaterialSpec(base_color=(0.045, 0.05, 0.065, 1.0),
                             metallic=0.35, roughness=0.30, clearcoat=0.45,
                             rim=2.2, rim_color=(0.35, 0.72, 1.00),
                             rim_power=3.2),
        pentagon=_accent((0.95, 0.52, 0.18, 1.0), (1.00, 0.45, 0.10), 2.2,
                         roughness=0.3),
        heptagon=_accent((0.25, 0.70, 1.00, 1.0), (0.20, 0.65, 1.00), 2.2,
                         roughness=0.3),
        octagon=_accent((0.80, 0.35, 1.00, 1.0), (0.75, 0.30, 1.00), 2.6,
                        roughness=0.3),
        bond=MaterialSpec(base_color=(0.07, 0.08, 0.10, 1.0),
                          metallic=0.35, roughness=0.36,
                          rim=1.1, rim_color=(0.30, 0.62, 1.00)),
        lights=[
            LightSpec("AREA", 1800.0, (0.80, 0.90, 1.00), (-7.0, 7.0, 8.0),
                      size=4.0),
            LightSpec("AREA", 320.0, (1.00, 0.80, 0.60), (8.0, -6.0, -2.0),
                      size=10.0),
        ],
        camera_lens_mm=85.0,
        camera_dof_fstop=2.8,
        samples=384,
        default_colour_by="ring",
        effects=(EffectSpec("glare", {"strength": 0.4,
                                      "threshold": 0.7}),),
    ),

    "advanced_materials": Style(
        name="advanced_materials",
        description=(
            "Saturated, glossy and crowded with colour, which is what "
            "Advanced Materials and Small actually print: a warm-to-cool "
            "gradient, a wet clear coat over strong hues, and element "
            "colouring, because these covers are usually about a "
            "composition rather than about curvature."
        ),
        background=Background(
            kind="gradient", top=(0.16, 0.07, 0.30),
            bottom=(0.03, 0.10, 0.26), strength=1.0,
        ),
        hexagon=MaterialSpec(base_color=(0.30, 0.34, 0.42, 1.0),
                             roughness=0.16, clearcoat=0.9,
                             coat_roughness=0.02,
                             rim=0.8, rim_color=(0.55, 0.80, 1.00)),
        pentagon=_accent((1.00, 0.30, 0.45, 1.0), (1.00, 0.25, 0.40), 2.0,
                         roughness=0.15, clearcoat=0.9),
        heptagon=_accent((0.20, 0.95, 0.85, 1.0), (0.15, 0.90, 0.80), 2.0,
                         roughness=0.15, clearcoat=0.9),
        octagon=_accent((1.00, 0.85, 0.20, 1.0), (1.00, 0.80, 0.15), 2.4,
                        roughness=0.15, clearcoat=0.9),
        bond=MaterialSpec(base_color=(0.45, 0.48, 0.56, 1.0),
                          roughness=0.2, clearcoat=0.8),
        lights=[
            LightSpec("AREA", 2200.0, (1.00, 0.92, 0.85), (7.0, -8.0, 6.0),
                      size=7.0),
            LightSpec("AREA", 1400.0, (0.45, 0.70, 1.00), (-8.0, 3.0, 5.0),
                      size=9.0),
            LightSpec("AREA", 600.0, (1.00, 0.40, 0.75), (0.0, 8.0, -4.0),
                      size=8.0),
        ],
        camera_lens_mm=70.0,
        camera_dof_fstop=3.0,
        samples=320,
        default_colour_by="element",
        effects=(EffectSpec("glare", {"strength": 0.35}),),
    ),

    "acs_nano_studio": Style(
        name="acs_nano_studio",
        description=(
            "ACS Nano's other register -- not the saturated gradient but "
            "the product shot: one object on a seamless mid-grey sweep, "
            "lit like a watch. Brushed metal, a floor it sits on, and no "
            "accent colour at all, so the geometry carries the picture."
        ),
        background=Background(
            kind="studio", top=(0.30, 0.31, 0.34), bottom=(0.10, 0.10, 0.12),
            strength=1.0, floor=True, floor_color=(0.17, 0.17, 0.19),
            floor_roughness=0.18,
        ),
        hexagon=MaterialSpec(base_color=(0.66, 0.68, 0.72, 1.0),
                             metallic=1.0, roughness=0.28,
                             anisotropy=0.7, anisotropy_rotation=0.15,
                             specular_tint=(0.95, 0.97, 1.0)),
        pentagon=MaterialSpec(base_color=(0.80, 0.80, 0.84, 1.0),
                              metallic=1.0, roughness=0.18, anisotropy=0.7),
        heptagon=MaterialSpec(base_color=(0.52, 0.54, 0.58, 1.0),
                              metallic=1.0, roughness=0.34, anisotropy=0.7),
        octagon=MaterialSpec(base_color=(0.88, 0.86, 0.78, 1.0),
                             metallic=1.0, roughness=0.22, anisotropy=0.7),
        bond=MaterialSpec(base_color=(0.58, 0.60, 0.64, 1.0), metallic=1.0,
                          roughness=0.32, anisotropy=0.6),
        lights=[
            LightSpec("AREA", 2400.0, (1.0, 1.0, 1.0), (6.0, -7.0, 8.0),
                      size=14.0),
            LightSpec("AREA", 1200.0, (1.0, 1.0, 1.0), (-8.0, -4.0, 5.0),
                      size=14.0),
            LightSpec("AREA", 800.0, (1.0, 1.0, 1.0), (0.0, 8.0, 2.0),
                      size=12.0),
        ],
        camera_lens_mm=90.0,
        camera_dof_fstop=5.0,
        samples=448,
        default_colour_by="uniform",
        effects=(),
    ),

    "matter_section": Style(
        name="matter_section",
        description=(
            "A journal's inside-page figure rather than its cover: flat "
            "even light, white ground, element colours, a neutral bond "
            "frame and no bloom, depth of field or accent. Everything "
            "that would be decoration on a cover is a distraction here, "
            "so it is all off -- this is the one to use when a reviewer "
            "has to count atoms."
        ),
        background=Background(
            kind="solid", top=(1.0, 1.0, 1.0), bottom=(1.0, 1.0, 1.0),
            strength=1.05,
        ),
        hexagon=MaterialSpec(base_color=(0.38, 0.40, 0.44, 1.0),
                             roughness=0.46, clearcoat=0.12),
        pentagon=MaterialSpec(base_color=(0.86, 0.33, 0.25, 1.0),
                              roughness=0.46),
        heptagon=MaterialSpec(base_color=(0.22, 0.46, 0.82, 1.0),
                              roughness=0.46),
        octagon=MaterialSpec(base_color=(0.94, 0.72, 0.16, 1.0),
                             roughness=0.46),
        bond=MaterialSpec(base_color=(0.55, 0.57, 0.60, 1.0), roughness=0.5),
        lights=[
            LightSpec("AREA", 1500.0, (1.0, 1.0, 1.0), (6.0, -7.0, 8.0),
                      size=16.0),
            LightSpec("AREA", 1200.0, (1.0, 1.0, 1.0), (-7.0, -5.0, 6.0),
                      size=16.0),
            LightSpec("AREA", 900.0, (1.0, 1.0, 1.0), (0.0, 8.0, 4.0),
                      size=16.0),
            LightSpec("AREA", 600.0, (1.0, 1.0, 1.0), (0.0, 0.0, -9.0),
                      size=16.0),
        ],
        camera_lens_mm=65.0,
        camera_dof_fstop=None,
        samples=192,
        default_colour_by="element",
        effects=(),
    ),

    # ------------------------------------------------------------------
    # Looks worked back from reference photographs and stock renders
    # rather than invented. Each names what it is copying, because a
    # preset whose provenance is "it looked nice" cannot be corrected.
    # ------------------------------------------------------------------
    "model_kit": Style(
        name="model_kit",
        description=(
            "A physical ball-and-stick model kit photographed on a bench: "
            "thick glossy injection-moulded spheres in CPK colours, matte "
            "white rods, a bright softbox overhead and a background thrown "
            "far out of focus. Ball-and-stick by definition, element "
            "colours, and fat atoms -- a kit's balls are nowhere near "
            "scale and that is what makes it read as a kit. Meant for a "
            "SMALL subject: a kit models a molecule or a fragment, and "
            "a 240-atom cage at these radii is a dense ball of spheres "
            "whatever the lighting does."
        ),
        background=Background(
            kind="gradient", top=(0.88, 0.87, 0.85), bottom=(0.55, 0.53, 0.52),
            strength=1.0, floor=True, floor_color=(0.90, 0.90, 0.91),
            floor_roughness=0.22, floor_metallic=0.0,
        ),
        hexagon=MaterialSpec(base_color=(0.05, 0.05, 0.06, 1.0),
                             roughness=0.08, clearcoat=1.0,
                             coat_roughness=0.02),
        pentagon=MaterialSpec(base_color=(0.85, 0.08, 0.08, 1.0),
                              roughness=0.08, clearcoat=1.0),
        heptagon=MaterialSpec(base_color=(0.10, 0.60, 0.90, 1.0),
                              roughness=0.08, clearcoat=1.0),
        octagon=MaterialSpec(base_color=(0.95, 0.80, 0.05, 1.0),
                             roughness=0.08, clearcoat=1.0),
        # The rods are matte white plastic, nothing like the balls.
        bond=MaterialSpec(base_color=(0.93, 0.93, 0.94, 1.0),
                          roughness=0.55, clearcoat=0.1),
        lights=[
            LightSpec("AREA", 3000.0, (1.00, 0.98, 0.95), (3.0, -6.0, 11.0),
                      size=16.0),
            LightSpec("AREA", 900.0, (0.95, 0.97, 1.00), (-9.0, -3.0, 3.0),
                      size=12.0),
            LightSpec("AREA", 500.0, (1.0, 1.0, 1.0), (7.0, 6.0, 1.0),
                      size=10.0),
        ],
        camera_lens_mm=100.0,
        # Wide open: the blurred bench behind is half of what makes it a
        # photograph of an object rather than a diagram.
        camera_dof_fstop=1.6,
        # A kit's balls are deliberately out of scale and its rods are
        # thick dowels. Drawn at chemical radii the same scene reads as
        # a lattice diagram, which is the thing this style is not --
        # but there is a ceiling: a real kit's ball is about half a bond
        # long across, and past that the spheres merge into a blob with
        # no rods visible between them, which is a space-filling model
        # by accident.
        atom_scale=1.9,
        bond_scale=2.6,
        samples=384,
        default_colour_by="element",
        default_model="ballstick",
        effects=(),
    ),

    "clusters_on_sheet": Style(
        name="clusters_on_sheet",
        description=(
            "Metal clusters sitting on a lattice, the way the stock "
            "libraries draw a supported catalyst: warm gold packed "
            "spheres against a cold blue sheet, a dark glossy floor "
            "underneath, strong rim. Defaults to the AUTO model -- the "
            "metal solid, the carbon open -- which is the whole point of "
            "the picture."
        ),
        background=Background(
            kind="gradient", top=(0.04, 0.20, 0.38), bottom=(0.01, 0.05, 0.12),
            strength=1.0, floor=True, floor_color=(0.02, 0.05, 0.09),
            floor_roughness=0.08, floor_metallic=0.8,
        ),
        hexagon=MaterialSpec(base_color=(0.10, 0.34, 0.55, 1.0),
                             metallic=0.75, roughness=0.22, clearcoat=0.4,
                             rim=1.4, rim_color=(0.40, 0.80, 1.00),
                             rim_power=3.0),
        pentagon=_accent((0.95, 0.70, 0.25, 1.0), (1.00, 0.65, 0.20), 1.6),
        heptagon=_accent((0.30, 0.75, 1.00, 1.0), (0.25, 0.70, 1.00), 1.6),
        octagon=_accent((1.00, 0.45, 0.20, 1.0), (1.00, 0.40, 0.15), 1.8),
        bond=MaterialSpec(base_color=(0.14, 0.40, 0.62, 1.0),
                          metallic=0.7, roughness=0.26,
                          rim=0.9, rim_color=(0.35, 0.75, 1.00)),
        lights=[
            LightSpec("AREA", 2400.0, (1.00, 0.88, 0.66), (7.0, -7.0, 8.0),
                      size=7.0),
            LightSpec("AREA", 1000.0, (0.35, 0.65, 1.00), (-8.0, 4.0, 4.0),
                      size=10.0),
            LightSpec("AREA", 500.0, (1.00, 0.75, 0.45), (0.0, 7.0, -3.0),
                      size=8.0),
        ],
        camera_lens_mm=55.0,
        camera_dof_fstop=2.2,
        samples=384,
        default_colour_by="element",
        default_model="auto",
        effects=(EffectSpec("glare", {"strength": 0.3}),),
    ),

    "clear_glass": Style(
        name="clear_glass",
        description=(
            "Translucent blue-grey glass on a pale warm ground, with the "
            "lattice receding into a soft blur: the clean, airy look used "
            "for abstract molecular stock art. Everything is transmissive, "
            "so the far side of the structure shows through the near side "
            "-- which is why it wants a long lens and a wide aperture."
        ),
        # Darker at the bottom than the reference's print would suggest:
        # a transmissive structure on an evenly pale ground has nothing
        # to refract and disappears. The gradient is the contrast.
        background=Background(
            kind="gradient", top=(0.95, 0.94, 0.92), bottom=(0.34, 0.45, 0.60),
            strength=1.1,
        ),
        hexagon=MaterialSpec(base_color=(0.82, 0.88, 0.95, 1.0),
                             metallic=0.0, roughness=0.03,
                             transmission=1.0, ior=1.48, clearcoat=0.3),
        pentagon=MaterialSpec(base_color=(0.95, 0.80, 0.78, 1.0),
                              roughness=0.03, transmission=1.0, ior=1.48),
        heptagon=MaterialSpec(base_color=(0.76, 0.86, 0.98, 1.0),
                              roughness=0.03, transmission=1.0, ior=1.48),
        octagon=MaterialSpec(base_color=(0.95, 0.92, 0.78, 1.0),
                             roughness=0.03, transmission=1.0, ior=1.48),
        # Solid metal rods through glass balls: the frame is what gives
        # the eye something to hold on to when everything else is
        # transparent.
        bond=MaterialSpec(base_color=(0.42, 0.48, 0.58, 1.0),
                          metallic=1.0, roughness=0.15),
        lights=[
            LightSpec("AREA", 2800.0, (1.00, 0.99, 0.97), (5.0, -8.0, 9.0),
                      size=18.0),
            LightSpec("AREA", 1600.0, (0.96, 0.98, 1.00), (-8.0, -5.0, 4.0),
                      size=18.0),
            LightSpec("AREA", 900.0, (1.0, 1.0, 1.0), (0.0, 9.0, 2.0),
                      size=14.0),
        ],
        camera_lens_mm=105.0,
        camera_dof_fstop=1.8,
        # Glass is the slowest thing here: every sphere refracts twice
        # and the rays behind it are the picture.
        samples=640,
        default_colour_by="ring",
        default_model="ballstick",
        atom_scale=1.7,
        bond_scale=0.8,
        effects=(EffectSpec("glare", {"strength": 0.2,
                                      "threshold": 0.9}),),
    ),

    "neon_violet": Style(
        name="neon_violet",
        description=(
            "Deep blue-violet, glossy dark spheres with a magenta rim, a "
            "faint grid under them and coloured bokeh behind: the "
            "nightclub register of molecular stock art. The bokeh is an "
            "effect, not a texture, so it is really out of focus -- which "
            "is why the style ships a wide aperture with it."
        ),
        background=Background(
            kind="gradient", top=(0.10, 0.04, 0.32), bottom=(0.02, 0.01, 0.10),
            strength=1.0, floor=True, floor_color=(0.05, 0.02, 0.16),
            floor_roughness=0.12, floor_metallic=0.6,
        ),
        hexagon=MaterialSpec(base_color=(0.16, 0.12, 0.42, 1.0),
                             metallic=0.6, roughness=0.10, clearcoat=0.9,
                             rim=2.4, rim_color=(1.00, 0.30, 0.85),
                             rim_power=2.6),
        pentagon=_accent((0.45, 0.35, 1.00, 1.0), (0.40, 0.30, 1.00), 2.0),
        heptagon=_accent((1.00, 0.25, 0.75, 1.0), (1.00, 0.20, 0.70), 2.0),
        octagon=_accent((0.25, 0.85, 1.00, 1.0), (0.20, 0.80, 1.00), 2.4),
        bond=MaterialSpec(base_color=(0.20, 0.16, 0.50, 1.0),
                          metallic=0.6, roughness=0.14,
                          rim=1.4, rim_color=(0.70, 0.35, 1.00)),
        lights=[
            LightSpec("AREA", 1400.0, (0.50, 0.35, 1.00), (-7.0, -6.0, 6.0),
                      size=8.0),
            LightSpec("AREA", 900.0, (1.00, 0.25, 0.70), (8.0, 3.0, 3.0),
                      size=8.0),
        ],
        camera_lens_mm=85.0,
        camera_dof_fstop=1.4,
        samples=416,
        default_colour_by="ring",
        default_model="ballstick",
        effects=(
            EffectSpec("bokeh", {"count": 34, "color": "magenta"}),
            EffectSpec("glare", {"strength": 0.45, "threshold": 0.65}),
        ),
    ),

    "teal_lab": Style(
        name="teal_lab",
        description=(
            "Saturated red and cyan enamel on a flat teal ground with a "
            "softly reflective floor: the bright, friendly register that "
            "press releases and journal TOC graphics use. Everything is "
            "glossy and nothing is metallic, so the colour stays the "
            "colour instead of turning into a reflection."
        ),
        background=Background(
            kind="gradient", top=(0.62, 0.84, 0.86), bottom=(0.38, 0.68, 0.72),
            strength=1.1, floor=True, floor_color=(0.55, 0.78, 0.80),
            floor_roughness=0.14, floor_metallic=0.25,
        ),
        hexagon=MaterialSpec(base_color=(0.78, 0.22, 0.14, 1.0),
                             roughness=0.10, clearcoat=1.0,
                             coat_roughness=0.02),
        pentagon=MaterialSpec(base_color=(0.14, 0.70, 0.78, 1.0),
                              roughness=0.10, clearcoat=1.0),
        heptagon=MaterialSpec(base_color=(0.95, 0.95, 0.96, 1.0),
                              roughness=0.12, clearcoat=1.0),
        octagon=MaterialSpec(base_color=(0.96, 0.72, 0.18, 1.0),
                             roughness=0.10, clearcoat=1.0),
        bond=MaterialSpec(base_color=(0.80, 0.84, 0.86, 1.0),
                          metallic=0.7, roughness=0.22),
        lights=[
            LightSpec("AREA", 2600.0, (1.0, 1.0, 1.0), (5.0, -7.0, 10.0),
                      size=16.0),
            LightSpec("AREA", 1300.0, (0.92, 0.98, 1.00), (-8.0, -4.0, 4.0),
                      size=14.0),
            LightSpec("AREA", 700.0, (1.0, 1.0, 1.0), (2.0, 8.0, 1.0),
                      size=12.0),
        ],
        camera_lens_mm=75.0,
        camera_dof_fstop=2.6,
        samples=320,
        # RING, not element: the red-and-cyan enamel IS this style, and
        # element colouring would replace it with CPK and leave nothing
        # of the look behind.
        default_colour_by="ring",
        default_model="ballstick",
        atom_scale=1.5,
        bond_scale=1.6,
        effects=(),
    ),

    # ------------------------------------------------------------------
    # Superstructures: 900-2000 atoms, where the subject is a LATTICE
    # rather than an object. Three things change at that size and all
    # three are in these presets rather than left to the caller:
    # the atoms have to shrink or the picture becomes a wall of spheres;
    # the depth of field has to open or two thirds of the structure is
    # a blur; and the colour has to come from position or curvature,
    # because an all-carbon net coloured by element is one grey mass.
    # ------------------------------------------------------------------
    "lattice_dark": Style(
        name="lattice_dark",
        description=(
            "A large periodic network against black, lit from behind so "
            "the struts separate by their rims rather than by their "
            "colour. Small atoms, thin bonds, deep focus: at a thousand "
            "atoms the subject is the ARCHITECTURE, and anything that "
            "fattens an atom closes the holes the eye reads depth "
            "through."
        ),
        background=Background(
            kind="gradient", top=(0.015, 0.018, 0.030),
            bottom=(0.0, 0.0, 0.005), strength=1.0,
        ),
        hexagon=MaterialSpec(base_color=(0.07, 0.08, 0.10, 1.0),
                             metallic=0.5, roughness=0.32, clearcoat=0.3,
                             rim=2.6, rim_color=(0.30, 0.70, 1.00),
                             rim_power=2.8),
        pentagon=_accent((1.00, 0.55, 0.15, 1.0), (1.00, 0.50, 0.10), 3.0),
        heptagon=_accent((0.20, 0.90, 1.00, 1.0), (0.15, 0.85, 1.00), 3.0),
        octagon=_accent((0.90, 0.30, 1.00, 1.0), (0.85, 0.25, 1.00), 3.4),
        bond=MaterialSpec(base_color=(0.09, 0.10, 0.13, 1.0),
                          metallic=0.5, roughness=0.34,
                          rim=1.8, rim_color=(0.25, 0.65, 1.00)),
        lights=[
            LightSpec("AREA", 2000.0, (0.70, 0.85, 1.00), (-6.0, 8.0, 7.0),
                      size=5.0),
            LightSpec("AREA", 400.0, (1.00, 0.75, 0.55), (8.0, -6.0, -2.0),
                      size=10.0),
        ],
        camera_lens_mm=50.0,
        # Deep: a lattice photographed at f/2 is a blur with a sharp
        # band through it, which looks like a mistake rather than a
        # choice.
        camera_dof_fstop=8.0,
        samples=448,
        default_colour_by="ring",
        default_model="ballstick",
        atom_scale=0.72,
        bond_scale=0.85,
        effects=(EffectSpec("glare", {"strength": 0.35}),),
    ),

    "lattice_depth": Style(
        name="lattice_depth",
        description=(
            "The same large net, but photographed INTO rather than at: "
            "haze in the air so the far side fades, a warm key at the "
            "front and a cold fill behind, and just enough depth of "
            "field to say which end is near. Atmospheric perspective is "
            "the only cue that still works when a structure is mostly "
            "its own repetition."
        ),
        background=Background(
            kind="gradient", top=(0.10, 0.14, 0.22),
            bottom=(0.02, 0.03, 0.06), strength=1.0,
        ),
        hexagon=MaterialSpec(base_color=(0.30, 0.36, 0.46, 1.0),
                             metallic=0.3, roughness=0.38, clearcoat=0.25,
                             rim=1.0, rim_color=(0.70, 0.85, 1.00)),
        pentagon=_accent((1.00, 0.60, 0.25, 1.0), (1.00, 0.55, 0.20), 1.6),
        heptagon=_accent((0.35, 0.85, 1.00, 1.0), (0.30, 0.80, 1.00), 1.6),
        octagon=_accent((1.00, 0.35, 0.55, 1.0), (1.00, 0.30, 0.50), 1.8),
        bond=MaterialSpec(base_color=(0.26, 0.31, 0.40, 1.0),
                          metallic=0.3, roughness=0.4),
        lights=[
            LightSpec("AREA", 2400.0, (1.00, 0.90, 0.78), (5.0, -9.0, 5.0),
                      size=8.0),
            LightSpec("AREA", 1100.0, (0.45, 0.70, 1.00), (-6.0, 8.0, 3.0),
                      size=12.0),
        ],
        camera_lens_mm=38.0,
        camera_dof_fstop=5.6,
        samples=512,
        default_colour_by="ring",
        default_model="ballstick",
        atom_scale=0.78,
        bond_scale=0.9,
        # The haze IS the style: it is what turns repetition into depth.
        effects=(
            EffectSpec("fog", {"strength": 0.0045, "color": "white"}),
            EffectSpec("glare", {"strength": 0.3}),
        ),
    ),

    "lattice_blueprint": Style(
        name="lattice_blueprint",
        description=(
            "A superstructure as a technical drawing: thin emissive "
            "struts on a dark grid, no spheres at all. LICORICE by "
            "default -- at this size the atoms are what hide the "
            "topology, and the topology is the result. For a figure "
            "about connectivity rather than about matter."
        ),
        background=Background(
            kind="gradient", top=(0.02, 0.04, 0.09),
            bottom=(0.0, 0.01, 0.03), strength=1.0,
        ),
        hexagon=_accent((0.30, 0.75, 1.00, 1.0), (0.25, 0.70, 1.00), 2.2,
                        roughness=0.3),
        pentagon=_accent((1.00, 0.65, 0.20, 1.0), (1.00, 0.60, 0.15), 3.2,
                         roughness=0.3),
        heptagon=_accent((1.00, 0.30, 0.55, 1.0), (1.00, 0.25, 0.50), 3.2,
                         roughness=0.3),
        octagon=_accent((0.70, 1.00, 0.40, 1.0), (0.65, 1.00, 0.35), 3.4,
                        roughness=0.3),
        bond=_accent((0.30, 0.75, 1.00, 1.0), (0.25, 0.70, 1.00), 2.2,
                     roughness=0.3),
        lights=[
            LightSpec("AREA", 600.0, (0.60, 0.80, 1.00), (6.0, -7.0, 6.0),
                      size=10.0),
        ],
        camera_lens_mm=42.0,
        camera_dof_fstop=None,
        samples=320,
        default_colour_by="ring",
        default_model="licorice",
        atom_scale=1.0,
        bond_scale=0.75,
        effects=(
            EffectSpec("grid", {"color": "cyan", "strength": 0.9,
                                "lines": 20}),
            EffectSpec("glare", {"strength": 0.45, "threshold": 0.6}),
        ),
    ),

    "lattice_studio": Style(
        name="lattice_studio",
        description=(
            "A large net as a white-background object photograph: even "
            "soft light from four sides, a faint contact shadow, no "
            "accent and no bloom. The one to put in a paper when the "
            "structure is big and the caption has to carry the meaning."
        ),
        background=Background(
            kind="solid", top=(0.97, 0.97, 0.98), bottom=(0.97, 0.97, 0.98),
            strength=1.05, floor=True, floor_color=(0.95, 0.95, 0.96),
            floor_roughness=0.6, floor_is_shadow_catcher=False,
        ),
        hexagon=MaterialSpec(base_color=(0.34, 0.36, 0.40, 1.0),
                             roughness=0.42, clearcoat=0.2),
        pentagon=MaterialSpec(base_color=(0.84, 0.36, 0.26, 1.0),
                              roughness=0.4),
        heptagon=MaterialSpec(base_color=(0.24, 0.48, 0.80, 1.0),
                              roughness=0.4),
        octagon=MaterialSpec(base_color=(0.92, 0.72, 0.20, 1.0),
                             roughness=0.4),
        bond=MaterialSpec(base_color=(0.52, 0.54, 0.58, 1.0), roughness=0.45),
        lights=[
            LightSpec("AREA", 1700.0, (1.0, 1.0, 1.0), (7.0, -8.0, 8.0),
                      size=18.0),
            LightSpec("AREA", 1300.0, (1.0, 1.0, 1.0), (-8.0, -6.0, 6.0),
                      size=18.0),
            LightSpec("AREA", 900.0, (1.0, 1.0, 1.0), (0.0, 9.0, 5.0),
                      size=16.0),
            LightSpec("AREA", 600.0, (1.0, 1.0, 1.0), (0.0, 0.0, -10.0),
                      size=16.0),
        ],
        camera_lens_mm=55.0,
        camera_dof_fstop=None,
        samples=256,
        default_colour_by="ring",
        default_model="ballstick",
        atom_scale=0.8,
        bond_scale=0.9,
        effects=(),
    ),

    # ------------------------------------------------------------------
    # The elaborate end: these combine the surface model, depth
    # colouring and the lighting rigs rather than only restyling a
    # ball-and-stick picture. Each is built around one idea about what
    # a large porous structure is FOR.
    # ------------------------------------------------------------------
    "porous_depth": Style(
        name="porous_depth",
        description=(
            "A porous solid graded through its own thickness: depth "
            "colouring on a viridis ramp, back-lit so the far wall glows "
            "through the near one, and haze to separate the layers. The "
            "point is that you can see INTO it -- which on an all-carbon "
            "net is the only thing colour can say, since element gives "
            "one hue and the rings are nearly all hexagons."
        ),
        background=Background(
            kind="gradient", top=(0.03, 0.04, 0.07),
            bottom=(0.0, 0.005, 0.015), strength=1.0,
        ),
        hexagon=MaterialSpec(base_color=(0.30, 0.50, 0.60, 1.0),
                             metallic=0.25, roughness=0.34, clearcoat=0.35,
                             rim=1.5, rim_color=(0.60, 0.95, 1.00),
                             rim_power=3.0),
        pentagon=_accent((1.00, 0.55, 0.20, 1.0), (1.00, 0.50, 0.15), 2.0),
        heptagon=_accent((0.30, 0.90, 1.00, 1.0), (0.25, 0.85, 1.00), 2.0),
        octagon=_accent((0.95, 0.35, 0.70, 1.0), (0.90, 0.30, 0.65), 2.2),
        bond=MaterialSpec(base_color=(0.28, 0.46, 0.56, 1.0),
                          metallic=0.25, roughness=0.36,
                          rim=1.0, rim_color=(0.55, 0.90, 1.00)),
        lights=[
            LightSpec("AREA", 1250.0, (0.85, 0.95, 1.00), (-5.0, 9.0, 6.0),
                      size=9.0),
            LightSpec("AREA", 380.0, (1.00, 0.80, 0.60), (7.0, -7.0, 2.0),
                      size=11.0),
        ],
        camera_lens_mm=45.0,
        camera_dof_fstop=6.3,
        samples=512,
        default_colour_by="depth",
        default_model="ballstick",
        atom_scale=0.7,
        bond_scale=0.8,
        # The haze is what drowned this. A fog volume scatters the key
        # light across the whole frame, so at 0.0035 with a 2600 W key
        # the backdrop lifted to grey and took the gradient's contrast
        # with it -- the colours were still there and nothing could be
        # read off them. A trace of fog separates the layers; a visible
        # amount replaces them.
        effects=(
            EffectSpec("fog", {"strength": 0.0008}),
            EffectSpec("glare", {"strength": 0.22, "threshold": 0.85}),
        ),
    ),

    # ------------------------------------------------------------------
    # The iridescent family. All three colour by DEPTH on a hue-cycling
    # ramp, against a dark field with no haze: the gradient IS the
    # picture, so nothing may be allowed to wash it out. They are cover
    # styles and the docstrings say so -- a cycling ramp does not
    # survive a greyscale print, which is the price of the look.
    # ------------------------------------------------------------------

    "oil_film": Style(
        name="oil_film",
        description=(
            "Oil on water: the depth gradient runs through a hue that "
            "CYCLES rather than climbing, over a thin-film interference "
            "coat, so the colour also shifts with the viewing angle. "
            "The most literally iridescent thing here, and the closest "
            "to what an Advanced Materials cover does with a film."
        ),
        background=Background(
            kind="gradient", top=(0.010, 0.012, 0.028),
            bottom=(0.002, 0.004, 0.012), strength=1.0,
        ),
        hexagon=MaterialSpec(base_color=(0.40, 0.55, 0.70, 1.0),
                             metallic=0.55, roughness=0.14, clearcoat=0.6,
                             thin_film=430.0, thin_film_ior=1.45,
                             rim=1.3, rim_color=(0.70, 0.90, 1.00),
                             rim_power=2.6),
        pentagon=_accent((1.00, 0.55, 0.25, 1.0), (1.00, 0.50, 0.20), 2.0),
        heptagon=_accent((0.35, 0.90, 1.00, 1.0), (0.30, 0.85, 1.00), 2.0),
        octagon=_accent((0.95, 0.35, 0.75, 1.0), (0.90, 0.30, 0.70), 2.2),
        bond=MaterialSpec(base_color=(0.34, 0.48, 0.62, 1.0),
                          metallic=0.55, roughness=0.18,
                          thin_film=380.0, thin_film_ior=1.45,
                          rim=0.9, rim_color=(0.65, 0.88, 1.00)),
        lights=[
            LightSpec("AREA", 1100.0, (0.88, 0.95, 1.00), (-5.0, 8.0, 6.0),
                      size=10.0),
            LightSpec("AREA", 420.0, (1.00, 0.78, 0.62), (7.0, -6.0, 3.0),
                      size=12.0),
            LightSpec("AREA", 520.0, (0.70, 0.85, 1.00), (0.0, -3.0, -9.0),
                      size=14.0),
        ],
        camera_lens_mm=48.0,
        camera_dof_fstop=7.1,
        samples=512,
        default_colour_by="depth",
        default_model="ballstick",
        depth_ramp=IRIDESCENT_RAMP,
        atom_scale=0.72,
        bond_scale=0.8,
        effects=(EffectSpec("glare", {"strength": 0.24,
                                      "threshold": 0.82}),),
    ),

    "aurora_net": Style(
        name="aurora_net",
        description=(
            "An aurora read through the lattice: green through teal "
            "into magenta along the structure's own depth, over a cold "
            "sky with far stars. Emissive rather than lit, so the net "
            "glows of itself and the backdrop stays black -- which is "
            "what makes it read at thumbnail size on a cover."
        ),
        background=Background(
            kind="space", top=(0.004, 0.010, 0.020),
            bottom=(0.0, 0.002, 0.008), strength=1.0,
            # star_density is the Voronoi SCALE, not a fraction: at 0.22
            # one cell covers the whole sky and the field is starless.
            star_density=300.0, star_amount=0.030,
        ),
        hexagon=MaterialSpec(base_color=(0.25, 0.70, 0.55, 1.0),
                             metallic=0.1, roughness=0.3,
                             emission_strength=0.55,
                             rim=2.2, rim_color=(0.45, 1.00, 0.80),
                             rim_power=2.4),
        pentagon=_accent((1.00, 0.50, 0.85, 1.0), (1.00, 0.45, 0.80), 2.8),
        heptagon=_accent((0.40, 1.00, 0.85, 1.0), (0.35, 1.00, 0.80), 2.8),
        octagon=_accent((0.80, 0.45, 1.00, 1.0), (0.75, 0.40, 1.00), 3.0),
        bond=MaterialSpec(base_color=(0.22, 0.60, 0.50, 1.0),
                          metallic=0.1, roughness=0.34,
                          emission_strength=0.35,
                          rim=1.4, rim_color=(0.45, 1.00, 0.80)),
        lights=[
            LightSpec("AREA", 700.0, (0.70, 1.00, 0.88), (-5.0, 8.0, 5.0),
                      size=12.0),
            LightSpec("AREA", 300.0, (0.85, 0.70, 1.00), (7.0, -6.0, 2.0),
                      size=12.0),
        ],
        camera_lens_mm=52.0,
        camera_dof_fstop=8.0,
        samples=512,
        default_colour_by="depth",
        default_model="ballstick",
        depth_ramp=AURORA_RAMP,
        atom_scale=0.68,
        bond_scale=0.78,
        effects=(EffectSpec("glare", {"strength": 0.34,
                                      "threshold": 0.70}),),
    ),

    "plasma_depth": Style(
        name="plasma_depth",
        description=(
            "The thermal reading: deep violet at the far face through "
            "red and orange to white at the near one, on a black field. "
            "The hottest-looking of the three and the one that survives "
            "a greyscale print, its ramp climbing in brightness the "
            "whole way -- so it is the one to pick when the figure has "
            "to work inside the paper as well as on the cover."
        ),
        background=Background(
            kind="gradient", top=(0.020, 0.008, 0.030),
            bottom=(0.004, 0.001, 0.010), strength=1.0,
        ),
        hexagon=MaterialSpec(base_color=(0.60, 0.30, 0.45, 1.0),
                             metallic=0.3, roughness=0.26, clearcoat=0.4,
                             emission_strength=0.3,
                             rim=1.8, rim_color=(1.00, 0.70, 0.45),
                             rim_power=2.8),
        pentagon=_accent((1.00, 0.90, 0.60, 1.0), (1.00, 0.85, 0.55), 2.4),
        heptagon=_accent((0.65, 0.20, 0.80, 1.0), (0.60, 0.15, 0.75), 2.4),
        octagon=_accent((1.00, 0.45, 0.25, 1.0), (1.00, 0.40, 0.20), 2.6),
        bond=MaterialSpec(base_color=(0.52, 0.26, 0.40, 1.0),
                          metallic=0.3, roughness=0.3,
                          emission_strength=0.2,
                          rim=1.1, rim_color=(1.00, 0.65, 0.40)),
        lights=[
            LightSpec("AREA", 1300.0, (1.00, 0.88, 0.78), (-5.0, 8.0, 6.0),
                      size=9.0),
            LightSpec("AREA", 450.0, (0.75, 0.60, 1.00), (7.0, -6.0, 3.0),
                      size=11.0),
        ],
        camera_lens_mm=45.0,
        camera_dof_fstop=6.3,
        samples=512,
        default_colour_by="depth",
        default_model="ballstick",
        depth_ramp=PLASMA_RAMP,
        atom_scale=0.72,
        bond_scale=0.8,
        effects=(EffectSpec("glare", {"strength": 0.28,
                                      "threshold": 0.78}),),
    ),

    "membrane": Style(
        name="membrane",
        description=(
            "The net as one continuous SURFACE rather than as atoms: a "
            "pale translucent skin over the bond graph, lit from behind, "
            "so what reads is the shape and the holes through it. For a "
            "paper about transport, sieving or surface area, where the "
            "lattice is a container and not a list of atoms."
        ),
        # A pale translucent skin needs something dark BEHIND it. On the
        # light-blue gradient this style first had, the membrane rendered
        # as a white shape on a white field and its holes disappeared:
        # back-lighting only reads against a darker backdrop.
        # These are LINEAR values and the frame is written as sRGB, so
        # a linear 0.09 displays near 0.33: the first dark backdrop here
        # came out mid blue-grey on screen. Set for the sRGB result.
        background=Background(
            kind="gradient", top=(0.006, 0.012, 0.020),
            bottom=(0.020, 0.045, 0.065), strength=1.0,
        ),
        # A pale base plus 0.72 transmission plus a hot key blew the skin
        # out to white and the holes with it. The tint is deeper so the
        # transmitted light carries colour instead of saturating.
        hexagon=MaterialSpec(base_color=(0.20, 0.55, 0.72, 1.0),
                             metallic=0.0, roughness=0.10,
                             transmission=0.76, ior=1.42, clearcoat=0.5,
                             rim=1.15, rim_color=(0.55, 0.92, 1.00),
                             rim_power=2.2),
        pentagon=MaterialSpec(base_color=(0.95, 0.62, 0.42, 1.0),
                              roughness=0.14, transmission=0.55,
                              rim=0.8, rim_color=(1.00, 0.72, 0.50)),
        heptagon=MaterialSpec(base_color=(0.42, 0.72, 0.98, 1.0),
                              roughness=0.14, transmission=0.55,
                              rim=0.8, rim_color=(0.55, 0.85, 1.00)),
        octagon=MaterialSpec(base_color=(0.96, 0.86, 0.48, 1.0),
                             roughness=0.14, transmission=0.55,
                             rim=0.8, rim_color=(1.00, 0.92, 0.60)),
        bond=MaterialSpec(base_color=(0.46, 0.76, 0.90, 1.0),
                          roughness=0.16, transmission=0.65,
                          rim=0.6, rim_color=(0.55, 0.92, 1.00)),
        lights=[
            LightSpec("AREA", 1400.0, (1.00, 0.99, 0.96), (-4.0, 9.0, 7.0),
                      size=16.0),
            LightSpec("AREA", 600.0, (0.92, 0.96, 1.00), (7.0, -7.0, 4.0),
                      size=16.0),
            # The back light is the one that matters here: it is what
            # pushes light THROUGH the skin and makes the holes visible.
            LightSpec("AREA", 2000.0, (0.80, 0.94, 1.00), (0.0, -2.0, -11.0),
                      size=18.0),
        ],
        camera_lens_mm=50.0,
        camera_dof_fstop=7.1,
        # Transmission through a skinned surface is as slow as glass.
        samples=640,
        default_colour_by="uniform",
        default_model="surface",
        # The holes ARE the subject here, so the skin has to stay well
        # inside a hexagon's circumradius or it welds the pores shut.
        skin_radius=0.55,
        effects=(EffectSpec("glare", {"strength": 0.18,
                                      "threshold": 0.88}),),
    ),

    "catalyst_stage": Style(
        name="catalyst_stage",
        description=(
            "A supported catalyst on a lit stage: the net in dark metal, "
            "a glowing floor grid under it, motes in the air and a cold "
            "rim. Built for the AUTO model, so a metal phase in the "
            "structure comes out solid against the open support -- the "
            "picture a catalysis paper's graphical abstract wants."
        ),
        # A near-mirror floor (roughness 0.10, metallic 0.7) reflects the
        # area lamps as two bright rectangles -- the stage reads as deep
        # space with two glowing panels in it. Roughened, the floor
        # returns a soft pool of light instead of the lamps' shape, and
        # the backdrop is lifted off pure black so there is a stage
        # behind the subject rather than a void.
        background=Background(
            kind="gradient", top=(0.05, 0.07, 0.10),
            bottom=(0.02, 0.04, 0.06), strength=1.0,
            floor=True, floor_color=(0.03, 0.05, 0.08),
            floor_roughness=0.38, floor_metallic=0.35,
        ),
        hexagon=MaterialSpec(base_color=(0.12, 0.14, 0.17, 1.0),
                             metallic=0.8, roughness=0.26, clearcoat=0.3,
                             rim=2.0, rim_color=(0.35, 0.80, 1.00),
                             rim_power=3.2),
        pentagon=_accent((1.00, 0.60, 0.20, 1.0), (1.00, 0.55, 0.15), 2.6),
        heptagon=_accent((0.25, 0.85, 1.00, 1.0), (0.20, 0.80, 1.00), 2.6),
        octagon=_accent((1.00, 0.30, 0.60, 1.0), (1.00, 0.25, 0.55), 3.0),
        bond=MaterialSpec(base_color=(0.14, 0.16, 0.20, 1.0),
                          metallic=0.8, roughness=0.3,
                          rim=1.3, rim_color=(0.30, 0.70, 1.00)),
        lights=[
            LightSpec("AREA", 2200.0, (0.75, 0.88, 1.00), (-6.0, 8.0, 7.0),
                      size=7.0),
            LightSpec("AREA", 600.0, (1.00, 0.72, 0.45), (8.0, -5.0, 1.0),
                      size=9.0),
        ],
        camera_lens_mm=48.0,
        camera_dof_fstop=5.6,
        samples=512,
        default_colour_by="ring",
        default_model="auto",
        atom_scale=0.8,
        bond_scale=0.85,
        effects=(
            EffectSpec("grid", {"color": "cyan", "strength": 0.7,
                                "lines": 22}),
            # Motes, not stars: at count 180 / strength 2.2 / the default
            # 0.010 size they scattered across the backdrop as a
            # starfield and the stage became outer space.
            EffectSpec("dust", {"count": 45, "strength": 0.7,
                                "size": 0.005, "reach": 1.05,
                                "color": "#ffe8c8"}),
            EffectSpec("glare", {"strength": 0.4}),
        ),
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
    "BOND_COLOURS",
    "DEPTH_BANDS",
    "DEPTH_RAMP",
    "AURORA_RAMP",
    "IRIDESCENT_RAMP",
    "PLASMA_RAMP",
    "DETAIL_LEVELS",
    "MODELS",
    "QUALITY_PRESETS",
    "render_cost",
    "SPACEFILL_ELEMENTS",
    "VDW_RADII",
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
    "style_summary",
    "bond_materials",
    "describe_finishes",
    "element_colour",
    "bond_is_drawn",
    "depth_colour",
    "detail_for",
    "draws_bonds",
    "element_radius",
    "is_spacefill_element",
    "model_radius",
    "get_style",
    "list_styles",
    "materials_for",
    "resolve_colour_by",
    "structure_colour",
    "vdw_radius",
    "with_background",
]
