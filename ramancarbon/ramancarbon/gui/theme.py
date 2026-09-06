"""Colours, fonts and spacing for the desktop application.

Kept free of Tkinter imports on purpose: the palette is data, so it can be
unit-tested and reused by the matplotlib figures without a display. Only
:func:`apply_theme` touches ttk, and it is called from the app.

The design brief was "friendly and elegant", which in a scientific tool
means: one accent colour used sparingly, generous whitespace, a restrained
type scale, and — the part that actually matters — plots whose colours
carry meaning. The fit curve, the individual components, the residual and
the baseline each have a fixed colour throughout the application, so a user
learns the code once.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """One colour scheme."""

    name: str
    background: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_muted: str
    accent: str
    accent_active: str
    accent_text: str
    success: str
    warning: str
    danger: str

    # plot roles
    data: str
    fitted: str
    residual: str
    baseline: str
    components: tuple[str, ...]

    def component_colour(self, index: int) -> str:
        """Colour for the *n*-th fitted component, cycling if needed."""
        return self.components[index % len(self.components)]


#: Light scheme. Slate-blue accent on a warm off-white; the component
#: colours are chosen to stay distinguishable in greyscale print, since
#: these figures end up in theses.
LIGHT = Palette(
    name="claro",
    background="#f4f5f7",
    surface="#ffffff",
    surface_alt="#eceef1",
    border="#d3d7de",
    text="#1d2129",
    text_muted="#68707d",
    accent="#2f6f8f",
    accent_active="#255a74",
    accent_text="#ffffff",
    success="#2e7d5b",
    warning="#a86a15",
    danger="#a83232",
    data="#3b414b",
    fitted="#c8461e",
    residual="#8a9099",
    baseline="#2f6f8f",
    components=("#2f6f8f", "#c8461e", "#2e7d5b", "#8a5fa8", "#a86a15", "#3f7d8c"),
)

#: Dark scheme, same accent family.
DARK = Palette(
    name="oscuro",
    background="#1b1e24",
    surface="#23272f",
    surface_alt="#2b303a",
    border="#39404c",
    text="#e6e9ee",
    text_muted="#9aa3b0",
    accent="#4f9fc4",
    accent_active="#63b2d6",
    accent_text="#10141a",
    success="#5fbf95",
    warning="#d8a247",
    danger="#e0736b",
    data="#d5dae2",
    fitted="#ef7a4e",
    residual="#7b8492",
    baseline="#4f9fc4",
    components=("#4f9fc4", "#ef7a4e", "#5fbf95", "#b58ad4", "#d8a247", "#6fb6c4"),
)

PALETTES = {"claro": LIGHT, "oscuro": DARK}

#: Font families tried in order; the first one present on the system wins.
FONT_STACK = ("Inter", "Segoe UI", "SF Pro Text", "Cantarell", "DejaVu Sans", "Helvetica")
MONO_STACK = ("JetBrains Mono", "Cascadia Mono", "Menlo", "DejaVu Sans Mono", "Courier")

#: Type scale, in points.
SIZES = {"title": 16, "heading": 12, "body": 10, "small": 9, "mono": 9}

#: Padding scale, in pixels. Used everywhere instead of magic numbers.
PAD = {"xs": 3, "sm": 6, "md": 12, "lg": 18, "xl": 26}


def pick_font(available: list[str], stack: tuple[str, ...], fallback: str = "TkDefaultFont") -> str:
    """First family from ``stack`` that the system actually has.

    Parameters
    ----------
    available:
        Families reported by ``tkinter.font.families()``.
    stack:
        Preference order.
    fallback:
        Returned when none of the stack is installed. ``TkDefaultFont`` is
        always present, so the application never fails over a font.
    """
    lowered = {name.lower() for name in available}
    for family in stack:
        if family.lower() in lowered:
            return family
    return fallback


def matplotlib_style(palette: Palette) -> dict:
    """rcParams making a matplotlib figure match the application chrome.

    Returned as a plain dict so it can be tested and so the plotting code
    stays independent of whether a GUI is running.
    """
    return {
        "figure.facecolor": palette.surface,
        "axes.facecolor": palette.surface,
        "axes.edgecolor": palette.border,
        "axes.labelcolor": palette.text,
        "axes.titlecolor": palette.text,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": palette.border,
        "grid.alpha": 0.45,
        "grid.linewidth": 0.6,
        "xtick.color": palette.text_muted,
        "ytick.color": palette.text_muted,
        "xtick.labelsize": SIZES["small"],
        "ytick.labelsize": SIZES["small"],
        "axes.labelsize": SIZES["body"],
        "axes.titlesize": SIZES["heading"],
        "legend.frameon": False,
        "legend.fontsize": SIZES["small"],
        "lines.linewidth": 1.3,
        "figure.autolayout": False,
        "text.color": palette.text,
        "savefig.facecolor": palette.surface,
        "savefig.dpi": 200,
    }


def apply_theme(root, palette: Palette) -> dict:
    """Configure ttk styles for a palette and return the chosen fonts.

    Tkinter's default widgets look like 1995. ttk with the ``clam`` theme
    and explicit colours gets most of the way to something modern; the
    remaining gap is closed by using flat frames as "cards" with a single
    border colour, and by never using the default button relief.

    Parameters
    ----------
    root:
        The Tk root window.
    palette:
        Which scheme to apply.

    Returns
    -------
    dict
        ``{"body": ..., "mono": ..., "title": ...}`` font tuples, for the
        widgets that set their own font.
    """
    from tkinter import font as tkfont
    from tkinter import ttk

    available = list(tkfont.families(root))
    family = pick_font(available, FONT_STACK)
    mono = pick_font(available, MONO_STACK, fallback="TkFixedFont")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:  # pragma: no cover - only on exotic Tk builds
        pass

    root.configure(background=palette.background)

    style.configure(".", background=palette.background, foreground=palette.text,
                    font=(family, SIZES["body"]))
    style.configure("TFrame", background=palette.background)
    style.configure("Card.TFrame", background=palette.surface, relief="flat",
                    borderwidth=1)
    style.configure("Toolbar.TFrame", background=palette.surface_alt)
    style.configure("TLabel", background=palette.background, foreground=palette.text)
    style.configure("Card.TLabel", background=palette.surface, foreground=palette.text)
    style.configure("Title.TLabel", background=palette.background,
                    foreground=palette.text, font=(family, SIZES["title"], "bold"))
    style.configure("Heading.TLabel", background=palette.surface,
                    foreground=palette.text, font=(family, SIZES["heading"], "bold"))
    style.configure("Muted.TLabel", background=palette.surface,
                    foreground=palette.text_muted, font=(family, SIZES["small"]))
    style.configure("Status.TLabel", background=palette.surface_alt,
                    foreground=palette.text_muted, font=(family, SIZES["small"]))
    style.configure("Success.TLabel", background=palette.surface,
                    foreground=palette.success)
    style.configure("Warning.TLabel", background=palette.surface,
                    foreground=palette.warning)
    style.configure("Danger.TLabel", background=palette.surface,
                    foreground=palette.danger)

    style.configure("TButton", background=palette.surface_alt, foreground=palette.text,
                    borderwidth=1, relief="flat", padding=(PAD["md"], PAD["sm"]))
    style.map("TButton",
              background=[("active", palette.border), ("disabled", palette.surface_alt)],
              foreground=[("disabled", palette.text_muted)])
    style.configure("Accent.TButton", background=palette.accent,
                    foreground=palette.accent_text, borderwidth=0,
                    padding=(PAD["md"], PAD["sm"]), font=(family, SIZES["body"], "bold"))
    style.map("Accent.TButton",
              background=[("active", palette.accent_active),
                          ("disabled", palette.border)],
              foreground=[("disabled", palette.text_muted)])

    style.configure("TNotebook", background=palette.background, borderwidth=0,
                    tabmargins=(PAD["sm"], PAD["sm"], PAD["sm"], 0))
    style.configure("TNotebook.Tab", background=palette.surface_alt,
                    foreground=palette.text_muted, padding=(PAD["md"], PAD["sm"]),
                    borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", palette.surface)],
              foreground=[("selected", palette.text)])

    style.configure("TEntry", fieldbackground=palette.surface,
                    foreground=palette.text, bordercolor=palette.border,
                    insertcolor=palette.text)
    style.configure("TCombobox", fieldbackground=palette.surface,
                    background=palette.surface, foreground=palette.text,
                    arrowcolor=palette.text_muted, bordercolor=palette.border)
    style.configure("TCheckbutton", background=palette.surface,
                    foreground=palette.text)
    style.configure("TRadiobutton", background=palette.surface,
                    foreground=palette.text)
    style.configure("TSeparator", background=palette.border)
    style.configure("TLabelframe", background=palette.surface,
                    bordercolor=palette.border, relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=palette.surface,
                    foreground=palette.text_muted,
                    font=(family, SIZES["small"], "bold"))
    style.configure("Treeview", background=palette.surface,
                    fieldbackground=palette.surface, foreground=palette.text,
                    bordercolor=palette.border, rowheight=24)
    style.configure("Treeview.Heading", background=palette.surface_alt,
                    foreground=palette.text_muted,
                    font=(family, SIZES["small"], "bold"), relief="flat")
    style.map("Treeview", background=[("selected", palette.accent)],
              foreground=[("selected", palette.accent_text)])
    style.configure("Horizontal.TProgressbar", background=palette.accent,
                    troughcolor=palette.surface_alt, borderwidth=0)

    return {
        "body": (family, SIZES["body"]),
        "small": (family, SIZES["small"]),
        "heading": (family, SIZES["heading"], "bold"),
        "title": (family, SIZES["title"], "bold"),
        "mono": (mono, SIZES["mono"]),
    }


__all__ = [
    "DARK",
    "FONT_STACK",
    "LIGHT",
    "MONO_STACK",
    "PAD",
    "PALETTES",
    "SIZES",
    "Palette",
    "apply_theme",
    "matplotlib_style",
    "pick_font",
]
