"""PRISMA 2020 flow diagram, drawn from the pipeline's own manifest.

Journals want the figure, not a paragraph of counts. Generating it from
``manifest.json`` means the diagram cannot drift out of step with the corpus:
re-run the pipeline and the figure changes with it.

Output is a self-contained SVG — no plotting library, no fonts to embed, and it
scales cleanly for print. Convert to PDF or TIFF with any vector tool if the
journal insists.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path

__all__ = ["PrismaCounts", "counts_from_manifest", "render_svg", "write_svg"]

_W, _H = 900, 470
_BOX_W, _BOX_H = 340, 74
_GAP_Y = 40
_LEFT_X = 60
_RIGHT_X = 500

_STYLE = """
  .bx   { fill:#f7f7f5; stroke:#52514e; stroke-width:1.4; rx:6; }
  .bx-x { fill:#fdf1ec; stroke:#eb6834; stroke-width:1.4; rx:6; }
  .hd   { fill:#2a78d6; }
  .t    { font:600 13px system-ui,-apple-system,Segoe UI,Roboto,sans-serif; fill:#0b0b0b; }
  .s    { font:400 12px system-ui,-apple-system,Segoe UI,Roboto,sans-serif; fill:#52514e; }
  .hdt  { font:700 10px system-ui,-apple-system,Segoe UI,Roboto,sans-serif; fill:#ffffff;
          letter-spacing:.04em; }
  .ln   { stroke:#52514e; stroke-width:1.4; fill:none; }
"""


@dataclass(slots=True)
class PrismaCounts:
    """The numbers a PRISMA 2020 flow diagram needs.

    Everything except ``excluded_screening`` and ``included`` comes straight
    from the pipeline. Those two depend on decisions only a human makes — the
    manual screening of flagged records — so they default to ``None`` and the
    diagram says "pendiente" until you fill them in.
    """

    scopus: int = 0
    wos: int = 0
    duplicates_removed: int = 0
    screened: int = 0
    no_abstract: int = 0
    flagged_manual: int = 0
    excluded_screening: int | None = None
    included: int | None = None

    @property
    def identified(self) -> int:
        """Records retrieved from all databases before deduplication."""
        return self.scopus + self.wos

    def check_consistency(self) -> list[str]:
        """Return the arithmetic problems a reviewer would spot in the diagram.

        PRISMA diagrams are checked by subtraction, and a flow whose numbers do
        not add up is one of the fastest ways to lose a reviewer's trust. Empty
        list means the figure is internally consistent.
        """
        problems: list[str] = []
        if self.identified - self.duplicates_removed != self.screened:
            problems.append(
                f"identificados ({self.identified}) − duplicados "
                f"({self.duplicates_removed}) = {self.identified - self.duplicates_removed}, "
                f"pero los cribados son {self.screened}."
            )
        included = self.resolved_included()
        if included is not None:
            if included > self.screened:
                problems.append(
                    f"incluidos ({included}) supera a los cribados ({self.screened})."
                )
            if included < 0:
                problems.append(f"incluidos negativo ({included}).")
        if self.excluded_screening is not None and self.excluded_screening < 0:
            problems.append("excluidos en cribado negativo.")
        return problems

    def resolved_included(self) -> int | None:
        """Final included count, derived when it was not supplied."""
        if self.included is not None:
            return self.included
        if self.excluded_screening is not None:
            return self.screened - self.excluded_screening
        return None


def counts_from_manifest(
    manifest: dict | str | Path,
    *,
    excluded_screening: int | None = None,
    included: int | None = None,
) -> PrismaCounts:
    """Build :class:`PrismaCounts` from a ``manifest.json`` (path or dict).

    ``excluded_screening`` and ``included`` are the human decisions; pass them
    once the manual screening described in ``docs/WORKFLOW.md`` §5 is done.
    """
    if not isinstance(manifest, dict):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    prisma = manifest.get("prisma", {})
    identified = prisma.get("records_identified_by_source") or {}
    scopus = identified.get("scopus")
    wos = identified.get("wos")
    if scopus is None or wos is None:
        # Older manifests only carry the post-deduplication split. Using it here
        # would understate identification by exactly the cross-database overlap,
        # so reconstruct the raw counts from the overlap table instead.
        overlap = manifest.get("overlap", {})
        both = overlap.get("both", 0)
        scopus = overlap.get("scopus_only", 0) + both
        wos = overlap.get("wos_only", 0) + both
    return PrismaCounts(
        scopus=int(scopus),
        wos=int(wos),
        duplicates_removed=int(prisma.get("duplicates_removed", 0)),
        screened=int(prisma.get("records_screened", 0)),
        no_abstract=int(prisma.get("records_without_abstract", 0)),
        flagged_manual=int(prisma.get("flagged_host_ambiguous", 0)),
        excluded_screening=excluded_screening,
        included=included,
    )


def _box(x: int, y: int, title: str, lines: list[str], *, excluded: bool = False) -> str:
    """One flow-diagram box with a bold title and up to two detail lines."""
    css = "bx-x" if excluded else "bx"
    out = [f'<rect class="{css}" x="{x}" y="{y}" width="{_BOX_W}" height="{_BOX_H}"/>',
           f'<text class="t" x="{x + 14}" y="{y + 25}">{html.escape(title)}</text>']
    for i, line in enumerate(lines[:2]):
        out.append(f'<text class="s" x="{x + 14}" y="{y + 45 + i * 17}">{html.escape(line)}</text>')
    return "\n  ".join(out)


def _band(y: int, height: int, label: str) -> str:
    """Vertical stage band down the left edge (Identification / Screening / …).

    ``height`` spans the stage, not one box, so a stage covering two rows gets a
    band tall enough for its label instead of clipping it.
    """
    mid = y + height / 2
    return (
        f'<rect class="hd" x="14" y="{y}" width="30" height="{height}" rx="5"/>\n  '
        f'<text class="hdt" x="29" y="{mid}" text-anchor="middle" dominant-baseline="middle" '
        f'transform="rotate(-90 29 {mid})">{html.escape(label)}</text>'
    )


def _arrow_down(x: int, y0: int, y1: int) -> str:
    """Vertical connector with an arrowhead."""
    return f'<path class="ln" d="M {x} {y0} L {x} {y1}" marker-end="url(#a)"/>'


def _arrow_right(x0: int, x1: int, y: int) -> str:
    """Horizontal connector into an exclusion box."""
    return f'<path class="ln" d="M {x0} {y} L {x1} {y}" marker-end="url(#a)"/>'


def render_svg(counts: PrismaCounts, *, title: str = "") -> str:
    """Render the flow diagram as a standalone SVG string.

    The right-hand column holds the exclusions, in PRISMA's usual layout. Boxes
    whose value depends on a decision you have not made yet read "pendiente"
    rather than showing a fabricated number.
    """
    cx_left = _LEFT_X + _BOX_W // 2
    top = 70 if title else 40
    # Three stages, the PRISMA 2020 skeleton: identification, screening,
    # inclusion. The pipeline's automated screening is already reflected in
    # `screened`, so a separate "records screened" box would repeat the number.
    rows = [top + i * (_BOX_H + _GAP_Y) for i in range(3)]
    included = counts.resolved_included()

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        f'width="{_W}" height="{_H}" font-family="system-ui, sans-serif">',
        f"<style>{_STYLE}</style>",
        '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#52514e"/></marker></defs>',
        f'<rect width="{_W}" height="{_H}" fill="#fcfcfb"/>',
    ]
    if title:
        parts.append(
            f'<text class="t" x="{_LEFT_X}" y="36" font-size="15">{html.escape(title)}</text>'
        )

    parts.append(_band(rows[0], _BOX_H, "IDENTIF."))
    parts.append(_box(_LEFT_X, rows[0], f"Registros identificados (n = {counts.identified})",
                      [f"Scopus: {counts.scopus}", f"Web of Science: {counts.wos}"]))
    parts.append(_box(_RIGHT_X, rows[0], f"Duplicados eliminados (n = {counts.duplicates_removed})",
                      ["DOI exacto, luego título difuso"], excluded=True))

    parts.append(_band(rows[1], _BOX_H, "CRIBADO"))
    parts.append(_box(_LEFT_X, rows[1], f"Registros cribados (n = {counts.screened})",
                      ["Unión Scopus ∪ WoS deduplicada",
                       "Criterios de docs/PROTOCOL.md §3"]))
    excl_text = (f"n = {counts.excluded_screening}" if counts.excluded_screening is not None
                 else "n = pendiente")
    parts.append(_box(_RIGHT_X, rows[1], f"Registros excluidos ({excl_text})",
                      [f"Sin resumen: {counts.no_abstract}",
                       f"Marcados para revisión manual: {counts.flagged_manual}"],
                      excluded=True))

    parts.append(_band(rows[2], _BOX_H, "INCLUSIÓN"))
    included_text = f"n = {included}" if included is not None else "n = pendiente"
    parts.append(_box(_LEFT_X, rows[2], f"Estudios incluidos en el review ({included_text})",
                      ["Corpus analítico final"] if included is not None
                      else ["Completa el cribado manual y vuelve a generar"]))

    for a, b in ((0, 1), (1, 2)):
        parts.append(_arrow_down(cx_left, rows[a] + _BOX_H, rows[b]))
    for row in (rows[0], rows[1]):
        parts.append(_arrow_right(_LEFT_X + _BOX_W, _RIGHT_X, row + _BOX_H // 2))

    parts.append(
        f'<text class="s" x="{_LEFT_X}" y="{_H - 20}">'
        "Diagrama PRISMA 2020 generado desde manifest.json — se regenera con cada corrida."
        "</text>"
    )
    # A diagram whose numbers do not subtract correctly is worse than none:
    # say so on the figure itself rather than letting it reach a reviewer.
    for i, problem in enumerate(counts.check_consistency()):
        parts.append(
            f'<text class="s" x="{_LEFT_X}" y="{_H - 42 - i * 16}" fill="#eb6834">'
            f"⚠ {html.escape(problem)}</text>"
        )
    parts.append("</svg>")
    return "\n  ".join(parts)


def write_svg(counts: PrismaCounts, path: str | Path, *, title: str = "") -> Path:
    """Write the diagram to ``path`` and return it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_svg(counts, title=title), encoding="utf-8")
    return path
