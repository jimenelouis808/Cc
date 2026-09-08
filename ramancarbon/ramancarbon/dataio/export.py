"""Getting numbers back out, in the formats people actually paste into.

A result that can only be read on screen has to be retyped, and retyped
numbers are wrong numbers. Everything the suite computes goes through
:class:`Table`, which writes itself as CSV, TSV, JSON, Markdown, LaTeX or
HTML — the six destinations that cover a spreadsheet, a manuscript, a
thesis, a report and a colleague's e-mail.

Two decisions that are not cosmetic:

**Units live in their own row**, not glued to the column name. ``C (F/g)``
in a header cannot be parsed back; a separate units row can, and a
spreadsheet still shows it above the numbers.

**Significant figures are honoured, not decimals.** A capacitance of
50.1234567 F/g written to six decimals claims a precision nobody has. The
formatter rounds to significant figures, and where a value arrives with an
uncertainty it is written to match it — 50.1 ± 0.4, never 50.1234 ± 0.4.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

#: Suffix → format name.
FORMATS: dict[str, str] = {
    ".csv": "csv", ".tsv": "tsv", ".txt": "tsv", ".dat": "tsv",
    ".json": "json", ".md": "markdown", ".markdown": "markdown",
    ".tex": "latex", ".html": "html", ".htm": "html",
}

_LATEX_ESCAPES = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
}


def significant(value: float, digits: int = 4) -> str:
    """A number to ``digits`` significant figures, without exponent noise."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    if -4 <= magnitude < digits + 2:
        decimals = max(0, digits - 1 - magnitude)
        text = f"{value:.{decimals}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"
    return f"{value:.{digits - 1}e}"


def with_uncertainty(value: float, sigma: Optional[float]) -> str:
    """``50.1 ± 0.4`` — the value rounded to the uncertainty's place.

    Writing more digits than the uncertainty supports is the most common
    way a table overstates what was measured.
    """
    if sigma is None or not math.isfinite(sigma) or sigma <= 0:
        return significant(value)
    place = math.floor(math.log10(abs(sigma)))
    decimals = max(0, -place + (1 if round(abs(sigma) / 10.0 ** place) < 3 else 0))
    return f"{value:.{decimals}f} ± {sigma:.{decimals}f}"


def _cell(value: Any, digits: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "sí" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return significant(float(value), digits)
    return str(value)


@dataclass
class Table:
    """A rectangle of values that knows how to write itself."""

    columns: list[str]
    rows: list[list[Any]]
    title: str = ""
    units: Optional[list[str]] = None
    """One per column, or ``None``. Written as its own row."""
    notes: list[str] = field(default_factory=list)
    """Caveats that belong with the numbers. They are written as comments
    in every format that has them, because a warning left behind in the
    program is a warning that did not happen."""
    digits: int = 4

    def __post_init__(self) -> None:
        if self.units is not None and len(self.units) != len(self.columns):
            raise ValueError(
                f"hay {len(self.columns)} columnas y {len(self.units)} unidades"
            )

    @property
    def text_rows(self) -> list[list[str]]:
        return [[_cell(v, self.digits) for v in row] for row in self.rows]

    # -- writers -------------------------------------------------------
    def to_csv(self, separator: str = ",") -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, delimiter=separator, lineterminator="\n")
        if self.title:
            writer.writerow([f"# {self.title}"])
        for note in self.notes:
            writer.writerow([f"# {note}"])
        writer.writerow(self.columns)
        if self.units:
            writer.writerow(self.units)
        writer.writerows(self.text_rows)
        return buffer.getvalue()

    def to_tsv(self) -> str:
        return self.to_csv(separator="\t")

    def to_json(self) -> str:
        """Column-oriented, with the raw numbers rather than the rounded text.

        A JSON export is read by a program, and a program wants the value
        that was computed. The rounding is for humans.
        """
        payload: dict[str, Any] = {
            "titulo": self.title,
            "columnas": self.columns,
            "unidades": self.units,
            "notas": self.notes,
            "datos": {
                name: [_jsonable(row[index]) for row in self.rows]
                for index, name in enumerate(self.columns)
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        widths = [
            max(len(column), *(len(row[index]) for row in self.text_rows))
            if self.rows else len(column)
            for index, column in enumerate(self.columns)
        ]
        lines = []
        if self.title:
            lines += [f"**{self.title}**", ""]
        header = self.columns
        if self.units:
            header = [f"{c} ({u})" if u else c
                      for c, u in zip(self.columns, self.units)]
            widths = [max(w, len(h)) for w, h in zip(widths, header)]
        lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
        lines.append("| " + " | ".join("-" * w for w in widths) + " |")
        for row in self.text_rows:
            lines.append("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")
        for note in self.notes:
            lines += ["", f"> {note}"]
        return "\n".join(lines) + "\n"

    def to_latex(self, environment: str = "tabular", caption: str = "") -> str:
        """A ``booktabs`` table, ready to paste into a manuscript."""
        alignment = "l" + "r" * (len(self.columns) - 1)
        lines = []
        wrap = caption or self.title
        if wrap:
            lines += [r"\begin{table}[htbp]", r"  \centering",
                      rf"  \caption{{{_latex(wrap)}}}"]
        lines.append(rf"  \begin{{{environment}}}{{{alignment}}}")
        lines.append(r"    \toprule")
        header = [_latex(c) for c in self.columns]
        lines.append("    " + " & ".join(header) + r" \\")
        if self.units:
            lines.append("    " + " & ".join(_latex(u) for u in self.units) + r" \\")
        lines.append(r"    \midrule")
        for row in self.text_rows:
            cells = [_latex(v).replace("±", r"$\pm$") for v in row]
            lines.append("    " + " & ".join(cells) + r" \\")
        lines.append(r"    \bottomrule")
        lines.append(rf"  \end{{{environment}}}")
        for note in self.notes:
            lines.append(rf"  \\[2pt] \footnotesize {_latex(note)}")
        if wrap:
            lines.append(r"\end{table}")
        return "\n".join(lines) + "\n"

    def to_html(self) -> str:
        parts = ["<table>"]
        if self.title:
            parts.append(f"  <caption>{_html(self.title)}</caption>")
        parts.append("  <thead><tr>"
                     + "".join(f"<th>{_html(c)}</th>" for c in self.columns)
                     + "</tr>")
        if self.units:
            parts.append("    <tr>"
                         + "".join(f"<th>{_html(u)}</th>" for u in self.units)
                         + "</tr>")
        parts.append("  </thead>")
        parts.append("  <tbody>")
        for row in self.text_rows:
            parts.append("    <tr>"
                         + "".join(f"<td>{_html(v)}</td>" for v in row)
                         + "</tr>")
        parts.append("  </tbody>")
        for note in self.notes:
            parts.append(f"  <tfoot><tr><td colspan=\"{len(self.columns)}\">"
                         f"{_html(note)}</td></tr></tfoot>")
        parts.append("</table>")
        return "\n".join(parts) + "\n"

    def render(self, fmt: str) -> str:
        """Any of the formats by name."""
        writers = {
            "csv": self.to_csv, "tsv": self.to_tsv, "json": self.to_json,
            "markdown": self.to_markdown, "latex": self.to_latex,
            "html": self.to_html,
        }
        if fmt not in writers:
            raise ValueError(
                f"formato desconocido: {fmt!r}; disponibles: "
                + ", ".join(sorted(writers))
            )
        return writers[fmt]()

    def save(self, path: str | Path, fmt: Optional[str] = None) -> Path:
        """Write the table, choosing the format from the suffix."""
        destination = Path(path)
        chosen = fmt or FORMATS.get(destination.suffix.lower())
        if chosen is None:
            raise ValueError(
                f"no sé qué formato es «{destination.suffix}»; usa uno de "
                + ", ".join(sorted(set(FORMATS.values())))
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.render(chosen), encoding="utf-8")
        return destination


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _latex(text: str) -> str:
    out = []
    for character in str(text):
        out.append(_LATEX_ESCAPES.get(character, character))
    return "".join(out)


def _html(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# -- tables from the suite's own objects --------------------------------

def _stack(obj: Any, pieces: Sequence[tuple[str, str, Any]],
           notes: Sequence[str], digits: int) -> Table:
    """A table from several equal-length columns."""
    arrays = [np.asarray(values, dtype=float) for _, _, values in pieces]
    return Table(
        columns=[name for name, _, _ in pieces],
        units=[unit for _, unit, _ in pieces],
        rows=[[float(column[index]) for column in arrays]
              for index in range(len(arrays[0]))],
        title=getattr(obj, "name", "datos"),
        notes=list(notes), digits=digits,
    )


def series_table(obj: Any, digits: int = 6) -> Table:
    """The x–y numbers of any measurement the suite reads.

    Deliberately generic: whatever it is, somebody wants the two columns
    in a spreadsheet.
    """
    name = getattr(obj, "name", "datos")
    if hasattr(obj, "shift") and hasattr(obj, "intensity"):
        columns, units = ["desplazamiento", "intensidad"], ["cm-1", "u.a."]
        x, y = obj.shift, obj.intensity
        notes = ([f"láser {obj.laser_nm:g} nm"] if getattr(obj, "laser_nm", None)
                 else ["láser sin declarar"])
    elif hasattr(obj, "two_theta"):
        columns, units = ["2theta", "intensidad"], ["grados", "cuentas"]
        x, y = obj.two_theta, obj.intensity
        notes = [f"longitud de onda {obj.wavelength:.5f} Å"]
    elif hasattr(obj, "potential") and hasattr(obj, "current") \
            and hasattr(obj, "scan_rate"):
        columns, units = ["potencial", "corriente"], ["V", "A"]
        extra = ([("ciclo", "", obj.cycle)]
                 if getattr(obj, "cycle", None) is not None else [])
        return _stack(
            obj, [("potencial", "V", obj.potential),
                  ("corriente", "A", obj.current)] + extra,
            notes=[f"velocidad de barrido {obj.scan_rate * 1e3:g} mV/s"],
            digits=digits,
        )
    elif hasattr(obj, "time") and hasattr(obj, "potential"):
        # The current column goes with it. Its SIGN is what splits the
        # branches, so a charge-discharge curve exported without it cannot
        # be analysed again — it can only be looked at.
        extra = ([("ciclo", "", obj.cycle)]
                 if getattr(obj, "cycle", None) is not None else [])
        return _stack(
            obj, [("tiempo", "s", obj.time),
                  ("potencial", "V", obj.potential),
                  ("corriente", "A", obj.current)] + extra,
            notes=["la corriente va con signo: positiva al cargar"],
            digits=digits,
        )
    elif hasattr(obj, "frequency"):
        table = Table(
            columns=["frecuencia", "Z_real", "Z_imag"],
            units=["Hz", "ohm", "ohm"],
            rows=[[float(f), float(z.real), float(z.imag)]
                  for f, z in zip(obj.frequency, obj.z)],
            title=name, digits=digits,
            notes=["Z″ va con su signo físico: negativo si es capacitivo"],
        )
        return table
    else:
        raise TypeError(f"no sé cómo tabular un {type(obj).__name__}")

    return Table(
        columns=columns, units=units,
        rows=[[float(a), float(b)] for a, b in zip(x, y)],
        title=name, notes=notes, digits=digits,
    )


def summary_table(rows: Sequence[dict[str, Any]], title: str = "",
                  notes: Optional[Sequence[str]] = None) -> Table:
    """A table from a list of dictionaries, one per sample.

    The column order is the order the keys first appear, and a row missing
    a key gets an empty cell rather than a zero: an empty cell is a
    measurement that was not made, and a zero is a measurement that was.
    """
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return Table(
        columns=columns,
        rows=[[row.get(column) for column in columns] for row in rows],
        title=title, notes=list(notes or []),
    )


def export(obj: Any, path: str | Path, **options) -> Path:
    """Write any measurement to any supported format, chosen by suffix.

    ``.jdx`` and ``.xy`` go through the instrument's own writer so the
    result is a file that instrument's software can read back.
    """
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix in (".jdx", ".dx"):
        from .jcamp import write_jcamp

        table = series_table(obj)
        x = [row[0] for row in table.rows]
        y = [row[1] for row in table.rows]
        return write_jcamp(
            x, y, destination, title=table.title,
            data_type=options.pop("data_type", _jcamp_type(obj)),
            x_units=options.pop("x_units", _jcamp_units(obj)),
            laser_nm=getattr(obj, "laser_nm", None),
        )
    if suffix in (".xy", ".xye"):
        from ..xrd.io import write_pattern

        return write_pattern(obj, destination)
    return series_table(obj).save(destination, **options)


def _jcamp_type(obj: Any) -> str:
    if hasattr(obj, "two_theta"):
        return "X-RAY DIFFRACTION"
    return "RAMAN SPECTRUM"


def _jcamp_units(obj: Any) -> str:
    return "DEGREES" if hasattr(obj, "two_theta") else "1/CM"


__all__ = [
    "FORMATS",
    "Table",
    "export",
    "series_table",
    "significant",
    "summary_table",
    "with_uncertainty",
]
