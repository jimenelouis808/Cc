"""Declarative presets: render formats and one-click cover "looks".

Everything in this module is plain data — no ``bpy`` — so the whole preset
catalogue can be validated by the test suite. The Blender-facing appliers live
in :mod:`atomviz_studio.looks.apply` and :mod:`atomviz_studio.scene.render`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Render presets
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RenderPreset:
    """Output format + sampling settings.

    Attributes:
        key: Stable identifier.
        label: Human readable name.
        description: Tooltip text.
        width: Horizontal resolution in pixels.
        height: Vertical resolution in pixels.
        engine: ``"EEVEE"`` or ``"CYCLES"``.
        samples: Render samples (Cycles) or TAA samples (EEVEE).
        film_transparent: Render without a background (for compositing).
        view_transform: ``"AgX"`` (4.0+), ``"Filmic"`` or ``"Standard"``.
        look: Colour-management look, e.g. ``"AgX - Punchy"``.
        note: Editorial guidance shown in the UI.
    """

    key: str
    label: str
    description: str
    width: int
    height: int
    engine: str = "CYCLES"
    samples: int = 256
    film_transparent: bool = False
    view_transform: str = "AgX"
    look: str = "None"
    note: str = ""

    @property
    def aspect(self) -> float:
        """Width / height ratio."""
        return self.width / self.height


_RENDER_PRESET_LIST: tuple[RenderPreset, ...] = (
    RenderPreset(
        key="preview_fast",
        label="Preview (fast)",
        description="Small EEVEE preview for look development.",
        width=1080,
        height=1350,
        engine="EEVEE",
        samples=64,
        note="Iterate here, then switch to a Cycles preset for the final frame.",
    ),
    RenderPreset(
        key="cover_a4_300",
        label="Cover A4 @ 300 dpi",
        description="210x297 mm portrait cover, print resolution.",
        width=2480,
        height=3508,
        samples=512,
        look="AgX - Medium High Contrast",
        note="Most European journals. Keep the masthead area (top ~15%) clean.",
    ),
    RenderPreset(
        key="cover_letter_300",
        label="Cover US Letter @ 300 dpi",
        description="8.5x11 in portrait cover, print resolution.",
        width=2550,
        height=3300,
        samples=512,
        look="AgX - Medium High Contrast",
        note="ACS / Wiley style covers.",
    ),
    RenderPreset(
        key="cover_square",
        label="Cover square @ 300 dpi",
        description="Square format for inside covers and social posts.",
        width=3000,
        height=3000,
        samples=512,
    ),
    RenderPreset(
        key="toc_graphic",
        label="TOC / graphical abstract",
        description="Wide table-of-contents graphic.",
        width=2400,
        height=1260,
        samples=256,
        note="Readable at ~8 cm wide: avoid thin text and tiny atoms.",
    ),
    RenderPreset(
        key="poster_uhd",
        label="Poster / talk (UHD)",
        description="16:9 ultra-HD frame for slides and posters.",
        width=3840,
        height=2160,
        samples=384,
    ),
    RenderPreset(
        key="print_a4_600",
        label="Print A4 @ 600 dpi",
        description="Very large portrait render for full-bleed printing.",
        width=4961,
        height=7016,
        samples=768,
        look="AgX - Medium High Contrast",
        note="Slow. Render this once the look is locked.",
    ),
    RenderPreset(
        key="alpha_overlay",
        label="Transparent overlay",
        description="Square render with a transparent film for compositing.",
        width=2400,
        height=2400,
        samples=384,
        film_transparent=True,
    ),
)

#: Render preset key -> :class:`RenderPreset`.
RENDER_PRESETS: dict[str, RenderPreset] = {p.key: p for p in _RENDER_PRESET_LIST}

#: Default render preset key.
DEFAULT_RENDER_PRESET = "preview_fast"


def list_render_presets() -> list[tuple[str, str, str]]:
    """Return ``(key, label, description)`` triples for UI enum construction."""
    return [(p.key, p.label, p.description) for p in _RENDER_PRESET_LIST]


# --------------------------------------------------------------------------
# Cover looks
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Look:
    """A complete cover recipe: palette + shading + world + light + effects.

    Every field maps onto one operator of the add-on, so a look is exactly
    "what you would have clicked" — nothing happens behind your back.
    """

    key: str
    label: str
    description: str
    palette: str = "midnight_neon"
    accent: str = "#22d3ee"
    style: str = "glossy_ceramic"
    style_params: dict[str, float] = field(default_factory=dict)
    stick_style: str = "polished_metal"
    stick_color: str = "#4b5563"
    background: str = "gradient"
    background_params: dict[str, object] = field(default_factory=dict)
    lighting: str = "three_point"
    lighting_params: dict[str, object] = field(default_factory=dict)
    volumetrics: float = 0.0
    electricity: dict[str, object] | None = None
    lasers: dict[str, object] | None = None
    postfx: dict[str, object] = field(default_factory=dict)
    camera: dict[str, object] = field(default_factory=dict)
    render_preset: str = "preview_fast"


_LOOK_LIST: tuple[Look, ...] = (
    Look(
        key="clean_journal",
        label="Clean Journal",
        description="White studio, ceramic atoms, soft light. The safe cover.",
        palette="cpk",
        style="glossy_ceramic",
        style_params={"roughness": 0.28, "clearcoat": 0.6},
        stick_style="matte_clay",
        stick_color="#b8bcc4",
        background="studio_white",
        lighting="studio_soft",
        lighting_params={"energy": 900.0, "warm": "#fff6e8", "cool": "#e8f1ff"},
        postfx={"glare": "FOG_GLOW", "glare_threshold": 1.4, "vignette": 0.0},
        camera={"focal_mm": 65.0, "elevation_deg": 12.0, "dof": True, "fstop": 4.0},
        render_preset="cover_a4_300",
    ),
    Look(
        key="neon_lab",
        label="Neon Lab",
        description="Graphite framework, glowing dopants, electric arcs.",
        palette="midnight_neon",
        accent="#22d3ee",
        style="polished_metal",
        style_params={"roughness": 0.18, "emissive_dopants": 3.0},
        stick_style="polished_metal",
        stick_color="#2b303b",
        background="nebula",
        background_params={"top": "#05070f", "bottom": "#0d1b2a", "accent": "#1b3b6f"},
        lighting="neon_rim",
        lighting_params={"color_a": "#22d3ee", "color_b": "#f472b6", "energy": 600.0},
        volumetrics=0.03,
        electricity={"count": 6, "chaos": 0.18, "thickness": 0.03, "color": "#7dd3fc",
                     "strength": 9.0, "branches": 2, "flicker": True},
        postfx={"glare": "FOG_GLOW", "glare_threshold": 1.1, "streaks": True,
                "vignette": 0.35},
        camera={"focal_mm": 50.0, "elevation_deg": 15.0, "dof": True, "fstop": 2.8},
        render_preset="cover_a4_300",
    ),
    Look(
        key="plasma_storm",
        label="Plasma Storm",
        description="Hot heteroatoms inside a discharging plasma cage.",
        palette="ice_fire",
        accent="#f97316",
        style="glossy_ceramic",
        style_params={"roughness": 0.22, "emissive_dopants": 4.0},
        stick_style="polished_metal",
        stick_color="#334155",
        background="plasma",
        background_params={"top": "#0b0410", "bottom": "#2a0a2f", "accent": "#7c2d92"},
        lighting="dramatic_top",
        lighting_params={"color_a": "#a855f7", "color_b": "#38bdf8", "energy": 800.0},
        volumetrics=0.05,
        electricity={"count": 10, "chaos": 0.24, "thickness": 0.025, "color": "#c084fc",
                     "strength": 11.0, "branches": 3, "cage": True, "flicker": True},
        postfx={"glare": "FOG_GLOW", "glare_threshold": 1.2, "streaks": True,
                "vignette": 0.45},
        camera={"focal_mm": 40.0, "elevation_deg": 8.0, "dof": True, "fstop": 2.2},
        render_preset="cover_a4_300",
    ),
    Look(
        key="laser_lab",
        label="Laser Lab",
        description="Crystal structure crossed by laser beams in a hazy room.",
        palette="mono_accent",
        accent="#ef4444",
        style="glass_crystal",
        style_params={"roughness": 0.05, "ior": 1.5},
        stick_style="polished_metal",
        stick_color="#1f2937",
        background="solid",
        background_params={"top": "#05060a", "bottom": "#05060a"},
        lighting="neon_rim",
        lighting_params={"color_a": "#ef4444", "color_b": "#3b82f6", "energy": 1100.0},
        volumetrics=0.09,
        lasers={"count": 3, "radius": 0.04, "color": "#ff2d2d", "strength": 22.0,
                "impact": True, "haze": True},
        postfx={"glare": "STREAKS", "glare_threshold": 1.1, "vignette": 0.5},
        camera={"focal_mm": 85.0, "elevation_deg": 6.0, "dof": True, "fstop": 2.0},
        render_preset="cover_a4_300",
    ),
    Look(
        key="crystal_ice",
        label="Crystal Ice",
        description="Frosted glass atoms on a cold gradient. Very clean.",
        palette="blueprint",
        accent="#7dd3fc",
        style="frosted_glass",
        style_params={"roughness": 0.28, "ior": 1.42},
        stick_style="frosted_glass",
        stick_color="#bae6fd",
        background="gradient",
        background_params={"top": "#0b2545", "bottom": "#8ecae6", "accent": "#caf0f8"},
        lighting="studio_soft",
        lighting_params={"energy": 700.0, "warm": "#e0f2fe", "cool": "#bae6fd"},
        postfx={"glare": "FOG_GLOW", "glare_threshold": 0.9, "vignette": 0.2},
        camera={"focal_mm": 70.0, "elevation_deg": 14.0, "dof": True, "fstop": 3.2},
        render_preset="cover_square",
    ),
    Look(
        key="gold_catalysis",
        label="Gold Catalysis",
        description="Warm metals on a dark warm gradient. Catalysis papers.",
        palette="gold_lab",
        accent="#ffcf40",
        style="polished_metal",
        style_params={"roughness": 0.22},
        stick_style="polished_metal",
        stick_color="#5a4a2f",
        background="gradient",
        background_params={"top": "#160e04", "bottom": "#4a3010", "accent": "#c08a2e",
                           "strength": 1.1},
        lighting="three_point",
        lighting_params={"energy": 2400.0, "warm": "#ffd9a0", "cool": "#9ec5ff"},
        volumetrics=0.02,
        postfx={"glare": "FOG_GLOW", "glare_threshold": 1.2, "vignette": 0.35},
        camera={"focal_mm": 85.0, "elevation_deg": 10.0, "dof": True, "fstop": 3.5},
        render_preset="cover_letter_300",
    ),
    Look(
        key="xray_schematic",
        label="X-Ray Schematic",
        description="Fresnel-lit transparent atoms. Reads as a diagram.",
        palette="blueprint",
        accent="#7dd3fc",
        style="xray_fresnel",
        style_params={"emission_strength": 3.5, "alpha": 0.25},
        stick_style="emissive_neon",
        stick_color="#38bdf8",
        background="solid",
        background_params={"top": "#020617", "bottom": "#020617"},
        lighting="neon_rim",
        lighting_params={"color_a": "#38bdf8", "color_b": "#a5f3fc", "energy": 300.0},
        postfx={"glare": "FOG_GLOW", "glare_threshold": 1.0, "vignette": 0.4},
        camera={"focal_mm": 60.0, "elevation_deg": 5.0, "dof": False},
        render_preset="toc_graphic",
    ),
    Look(
        key="toon_outreach",
        label="Toon Outreach",
        description="Flat cel shading for press releases and teaching.",
        palette="spectral",
        style="toon_cel",
        style_params={"steps": 3.0},
        stick_style="toon_cel",
        stick_color="#3f3f46",
        background="solid",
        background_params={"top": "#f8fafc", "bottom": "#e2e8f0"},
        lighting="studio_soft",
        lighting_params={"energy": 800.0, "warm": "#ffffff", "cool": "#ffffff"},
        postfx={"glare": "NONE", "vignette": 0.0},
        camera={"focal_mm": 55.0, "elevation_deg": 15.0, "dof": False},
        render_preset="toc_graphic",
    ),
    Look(
        key="hologram_future",
        label="Hologram",
        description="Scan-lined emissive structure floating in a starfield.",
        palette="midnight_neon",
        accent="#5eead4",
        style="hologram",
        style_params={"emission_strength": 3.0, "scanlines": 90.0, "alpha": 0.45},
        stick_style="emissive_neon",
        stick_color="#5eead4",
        background="starfield",
        background_params={"top": "#01030a", "bottom": "#061024", "accent": "#22d3ee"},
        lighting="neon_rim",
        lighting_params={"color_a": "#5eead4", "color_b": "#6366f1", "energy": 250.0},
        volumetrics=0.04,
        electricity={"count": 4, "chaos": 0.12, "thickness": 0.02, "color": "#5eead4",
                     "strength": 8.0, "branches": 1, "flicker": True},
        postfx={"glare": "FOG_GLOW", "glare_threshold": 0.9, "streaks": True,
                "vignette": 0.4},
        camera={"focal_mm": 45.0, "elevation_deg": 12.0, "dof": True, "fstop": 2.4},
        render_preset="cover_square",
    ),
)

#: Look key -> :class:`Look`.
LOOKS: dict[str, Look] = {look.key: look for look in _LOOK_LIST}

#: Default look key.
DEFAULT_LOOK = "neon_lab"


def list_looks() -> list[tuple[str, str, str]]:
    """Return ``(key, label, description)`` triples for UI enum construction."""
    return [(look.key, look.label, look.description) for look in _LOOK_LIST]


def get_look(key: str) -> Look:
    """Return the look for *key*, falling back to :data:`DEFAULT_LOOK`."""
    return LOOKS.get(key, LOOKS[DEFAULT_LOOK])
