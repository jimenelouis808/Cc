"""Journal-cover style presets for the nanocarbon_lab Blender pipeline.

Pure data + pure-Python helpers -- **no `bpy` import here** -- so this
module can be unit-tested without a Blender interpreter. :mod:`render_cnt`
(which does import ``bpy``) consumes :data:`STYLES` to build materials,
world background, lighting rig and camera settings.

Each style is tuned to evoke the look of a specific class of high-impact
materials-science cover art rather than any one journal's exact house
style (Nature, ACS Nano and Small do not publish fixed visual templates --
covers are commissioned art). Ring-type colouring is shared machinery
(pentagons/heptagons/octagons get an accent colour so curvature and
defects read visually), only the palette and lighting mood change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MaterialSpec:
    """Principled-BSDF-ish parameters for one shading role."""

    base_color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.4
    transmission: float = 0.0
    ior: float = 1.45
    emission_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emission_strength: float = 0.0
    clearcoat: float = 0.0


@dataclass(frozen=True)
class LightSpec:
    kind: str  # "SUN" | "AREA" | "POINT"
    energy: float
    color: tuple[float, float, float]
    location: tuple[float, float, float]
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: float = 1.0  # AREA light extent (m), ignored for SUN/POINT


@dataclass(frozen=True)
class Style:
    """One complete look: materials, world, lights, camera, render engine."""

    name: str
    description: str
    world_color_top: tuple[float, float, float]
    world_color_bottom: tuple[float, float, float]
    world_strength: float
    hexagon: MaterialSpec
    pentagon: MaterialSpec
    heptagon: MaterialSpec
    octagon: MaterialSpec
    bond: MaterialSpec
    lights: list[LightSpec] = field(default_factory=list)
    camera_lens_mm: float = 50.0
    camera_dof_fstop: float | None = 2.8  # None disables depth of field
    background_is_transparent: bool = False
    engine: str = "CYCLES"
    samples: int = 256


STYLES: dict[str, Style] = {
    "nature_dark": Style(
        name="nature_dark",
        description=(
            "Matte black void, single dramatic key light + cool rim light, "
            "near-black graphite body with glowing defect accents. Evokes "
            "the moody single-subject cover style common in Nature/Nature "
            "Nanotechnology."
        ),
        world_color_top=(0.0, 0.0, 0.0),
        world_color_bottom=(0.0, 0.0, 0.0),
        world_strength=0.02,
        hexagon=MaterialSpec(
            base_color=(0.03, 0.03, 0.035, 1.0), metallic=0.6, roughness=0.28,
        ),
        pentagon=MaterialSpec(
            base_color=(0.85, 0.15, 0.05, 1.0), metallic=0.1, roughness=0.25,
            emission_color=(1.0, 0.25, 0.05), emission_strength=1.4,
        ),
        heptagon=MaterialSpec(
            base_color=(0.05, 0.35, 0.9, 1.0), metallic=0.1, roughness=0.25,
            emission_color=(0.1, 0.45, 1.0), emission_strength=1.4,
        ),
        octagon=MaterialSpec(
            base_color=(0.9, 0.8, 0.05, 1.0), metallic=0.1, roughness=0.25,
            emission_color=(1.0, 0.85, 0.05), emission_strength=1.8,
        ),
        bond=MaterialSpec(base_color=(0.02, 0.02, 0.025, 1.0), metallic=0.7, roughness=0.3),
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
        world_color_top=(0.05, 0.02, 0.25, 1.0)[:3],
        world_color_bottom=(0.35, 0.02, 0.4, 1.0)[:3],
        world_strength=0.6,
        hexagon=MaterialSpec(
            base_color=(0.05, 0.5, 0.9, 1.0), metallic=0.2, roughness=0.15,
            clearcoat=1.0,
        ),
        pentagon=MaterialSpec(
            base_color=(1.0, 0.2, 0.5, 1.0), metallic=0.1, roughness=0.1,
            emission_color=(1.0, 0.15, 0.55), emission_strength=2.2, clearcoat=1.0,
        ),
        heptagon=MaterialSpec(
            base_color=(0.2, 1.0, 0.4, 1.0), metallic=0.1, roughness=0.1,
            emission_color=(0.25, 1.0, 0.45), emission_strength=2.2, clearcoat=1.0,
        ),
        octagon=MaterialSpec(
            base_color=(1.0, 0.85, 0.1, 1.0), metallic=0.1, roughness=0.1,
            emission_color=(1.0, 0.85, 0.15), emission_strength=2.6, clearcoat=1.0,
        ),
        bond=MaterialSpec(base_color=(0.08, 0.35, 0.65, 1.0), metallic=0.3, roughness=0.2),
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
        world_color_top=(0.97, 0.97, 0.98),
        world_color_bottom=(0.88, 0.89, 0.92),
        world_strength=1.4,
        hexagon=MaterialSpec(base_color=(0.75, 0.78, 0.82, 1.0), metallic=0.0, roughness=0.55),
        pentagon=MaterialSpec(base_color=(0.9, 0.35, 0.3, 1.0), metallic=0.0, roughness=0.45),
        heptagon=MaterialSpec(base_color=(0.25, 0.45, 0.85, 1.0), metallic=0.0, roughness=0.45),
        octagon=MaterialSpec(base_color=(0.95, 0.7, 0.15, 1.0), metallic=0.0, roughness=0.45),
        bond=MaterialSpec(base_color=(0.6, 0.62, 0.66, 1.0), metallic=0.0, roughness=0.5),
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
        world_color_top=(0.0, 0.01, 0.04),
        world_color_bottom=(0.0, 0.0, 0.02),
        world_strength=0.03,
        hexagon=MaterialSpec(
            base_color=(0.02, 0.15, 0.35, 1.0), roughness=0.3,
            emission_color=(0.1, 0.55, 1.0), emission_strength=0.6,
        ),
        pentagon=MaterialSpec(
            base_color=(0.35, 0.02, 0.05, 1.0), roughness=0.3,
            emission_color=(1.0, 0.2, 0.15), emission_strength=2.2,
        ),
        heptagon=MaterialSpec(
            base_color=(0.02, 0.35, 0.1, 1.0), roughness=0.3,
            emission_color=(0.2, 1.0, 0.4), emission_strength=2.2,
        ),
        octagon=MaterialSpec(
            base_color=(0.35, 0.3, 0.02, 1.0), roughness=0.3,
            emission_color=(1.0, 0.85, 0.15), emission_strength=2.4,
        ),
        bond=MaterialSpec(
            base_color=(0.05, 0.4, 0.9, 1.0), roughness=0.3,
            emission_color=(0.15, 0.6, 1.0), emission_strength=1.0,
        ),
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
        world_color_top=(0.02, 0.015, 0.01),
        world_color_bottom=(0.05, 0.03, 0.01),
        world_strength=0.15,
        hexagon=MaterialSpec(base_color=(0.83, 0.68, 0.22, 1.0), metallic=1.0, roughness=0.18),
        pentagon=MaterialSpec(
            base_color=(0.95, 0.95, 0.95, 1.0), metallic=1.0, roughness=0.12,
        ),
        heptagon=MaterialSpec(
            base_color=(0.75, 0.35, 0.12, 1.0), metallic=1.0, roughness=0.15,
        ),
        octagon=MaterialSpec(
            base_color=(0.85, 0.1, 0.1, 1.0), metallic=0.7, roughness=0.2,
            emission_color=(0.8, 0.1, 0.05), emission_strength=0.6,
        ),
        bond=MaterialSpec(base_color=(0.6, 0.48, 0.15, 1.0), metallic=1.0, roughness=0.2),
        lights=[
            LightSpec("AREA", 2000.0, (1.0, 0.85, 0.6), (6.0, -7.0, 7.0), size=2.5),
            LightSpec("AREA", 700.0, (0.7, 0.8, 1.0), (-7.0, 3.0, 3.0), size=4.0),
            LightSpec("AREA", 300.0, (1.0, 0.9, 0.7), (0.0, 7.0, -2.0), size=5.0),
        ],
        camera_lens_mm=90.0,
        camera_dof_fstop=2.0,
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
