"""JCAMP-DX: the interchange format every spectrometer can write.

Vendors disagree about everything except this. A Renishaw, a Horiba, a
Bruker and a Thermo all export ``.jdx``/``.dx``, and it is the one format a
user can produce without our having reverse-engineered their binary. It is
also the only text format in common use that carries the *units* and the
excitation wavelength as data rather than as a comment somebody wrote.

Three encodings appear in the wild and all three are read here:

AFFN
    Plain numbers, separated by spaces. What most instruments write.
ASDF/SQZ
    The sign and first digit squeezed into one letter: ``A``–``I`` for
    ``+1``–``+9``, ``a``–``i`` for ``−1``–``−9``, ``@`` for zero. Halves
    the file size and appears without warning in the middle of an
    otherwise plain file.
ASDF/DIF and DUP
    Differences against the previous value (``%JKLMNOPQR`` / ``jklmnopqr``)
    and run-length repeats (``STUVWXYZs``). The compact form, and the one
    that goes wrong silently: a decoder that ignores DIF reads a monotone
    ramp as a spectrum and nothing looks obviously broken.

The X-checksum is verified rather than trusted. Each line of an ``X++(Y..Y)``
block restates its own first X, and a file whose lines disagree with its own
``FIRSTX``/``LASTX``/``NPOINTS`` has been truncated or concatenated — which
is the failure this format actually has, and it is invisible unless checked.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

#: SQZ: sign and first digit in one character.
_SQZ = {"@": 0}
for _i, _c in enumerate("ABCDEFGHI", start=1):
    _SQZ[_c] = _i
for _i, _c in enumerate("abcdefghi", start=1):
    _SQZ[_c] = -_i

#: DIF: the same, but the value is a difference from the one before.
_DIF = {"%": 0}
for _i, _c in enumerate("JKLMNOPQR", start=1):
    _DIF[_c] = _i
for _i, _c in enumerate("jklmnopqr", start=1):
    _DIF[_c] = -_i

#: DUP: repeat the previous value this many times in total.
_DUP = {c: i for i, c in enumerate("STUVWXYZs", start=1)}

_TOKEN = re.compile(
    r"[@A-Ia-i][0-9]*(?:\.[0-9]+)?"      # SQZ
    r"|[%J-Rj-r][0-9]*"                  # DIF
    r"|[S-Zs][0-9]*"                     # DUP
    r"|[-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?"   # plain
)

#: Units as they are spelled in the standard, mapped to what we use.
X_UNITS = {
    "1/CM": "cm-1", "1/ CM": "cm-1", "CM-1": "cm-1", "CM^-1": "cm-1",
    "RAMANSHIFT": "cm-1", "MICROMETERS": "um", "NANOMETERS": "nm",
    "NM": "nm", "SECONDS": "s", "HZ": "Hz", "DEGREES": "deg",
    "2THETA": "deg", "VOLTS": "V",
}


class JCAMPError(ValueError):
    """Raised when a JCAMP-DX file cannot be read as one."""


def _canonical(label: str) -> str:
    """A data label with the spacing the standard says to ignore removed.

    ``##FIRST X``, ``##first-x`` and ``##FIRSTX`` are the same label; the
    standard says so and files rely on it.
    """
    return re.sub(r"[\s/_-]", "", label).upper().lstrip("#").rstrip("=")


def parse_records(text: str) -> list[tuple[str, str]]:
    """Every ``##LABEL= value`` in order, with continuation lines joined.

    Order is kept because a JCAMP file can hold several blocks (``##TITLE``
    … ``##END=``) and the labels only mean anything relative to the block
    they are in.
    """
    records: list[tuple[str, str]] = []
    current: Optional[list[str]] = None
    label = ""
    for raw in text.splitlines():
        line = raw.split("$$", 1)[0].rstrip() if not raw.lstrip().startswith("##") \
            else raw.rstrip()
        if line.lstrip().startswith("##"):
            if current is not None:
                records.append((label, "\n".join(current).strip()))
            head, _, tail = line.strip().partition("=")
            label = _canonical(head)
            current = [tail]
        elif current is not None:
            current.append(line)
    if current is not None:
        records.append((label, "\n".join(current).strip()))
    if not records:
        raise JCAMPError("el archivo no contiene ninguna etiqueta ##")
    return records


def _decode_line(line: str) -> tuple[float, list[float], bool]:
    """One data line: its X, its Y values ASDF-expanded, and whether DIF
    encoding was used on it.

    The X comes back as written — the caller compares it against the X it
    expected rather than using it, because in a compressed file the written
    X is the check and the computed one is the datum. The DIF flag matters
    for a different check: in DIF form the last ordinate of a line is
    repeated as the first ordinate of the next, and a decoder that keeps
    both ends up with one extra point per line.
    """
    tokens = _TOKEN.findall(line)
    if not tokens:
        return math.nan, [], False
    values: list[float] = []
    previous = 0.0
    in_difference = False
    last_difference = 0.0
    used_difference = False
    for index, token in enumerate(tokens):
        head = token[0]
        if index == 0:
            values.append(float(token))
            previous = values[0]
            continue
        if head in _DUP:
            repeats = int(f"{_DUP[head]}{token[1:]}") if token[1:] else _DUP[head]
            if not values:
                raise JCAMPError("DUP al principio de una línea")
            for _ in range(repeats - 1):
                if in_difference:
                    # A repeated DIF repeats the DIFFERENCE, not the value:
                    # this is how a straight ramp is written in two
                    # characters, and reading it as a repeated value turns
                    # the ramp into a plateau.
                    previous = previous + last_difference
                    values.append(previous)
                else:
                    values.append(values[-1])
            continue
        if head in _SQZ and head not in "0123456789":
            values.append(float(f"{_SQZ[head]}{token[1:]}")
                          if _SQZ[head] >= 0
                          else -float(f"{abs(_SQZ[head])}{token[1:]}"))
            previous = values[-1]
            in_difference = False
            continue
        if head in _DIF:
            step = (float(f"{_DIF[head]}{token[1:]}") if _DIF[head] >= 0
                    else -float(f"{abs(_DIF[head])}{token[1:]}"))
            last_difference = step
            previous = previous + step
            values.append(previous)
            in_difference = True
            used_difference = True
            continue
        values.append(float(token))
        previous = values[-1]
        in_difference = False
    return values[0], values[1:], used_difference


def _decode_block(lines: list[str], kind: str) -> tuple[np.ndarray, np.ndarray]:
    """The numbers of an ``XYDATA``/``XYPOINTS``/``PEAK TABLE`` block."""
    if kind == "xydata":
        first_x: list[float] = []
        rows: list[list[float]] = []
        for line in lines:
            if not line.strip():
                continue
            x, ys, used_difference = _decode_line(line)
            if not ys:
                continue
            if used_difference and rows:
                # The Y-check: this line restates the previous line's last
                # ordinate. Verify it and drop it — the disagreement it is
                # there to catch is a line lost in transmission.
                if abs(ys[0] - rows[-1][-1]) > 1e-6 * max(1.0, abs(ys[0])):
                    raise JCAMPError(
                        f"la comprobación Y falla en X={x:.6g}: la línea "
                        f"empieza en {ys[0]:.6g} y la anterior terminó en "
                        f"{rows[-1][-1]:.6g}"
                    )
                ys = ys[1:]
                if not ys:
                    continue
            first_x.append(x)
            rows.append(ys)
        return np.asarray(first_x, dtype=float), rows  # type: ignore[return-value]

    xs: list[float] = []
    ys: list[float] = []
    for line in lines:
        numbers = [float(t) for t in re.findall(
            r"[-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?", line)]
        for index in range(0, len(numbers) - 1, 2):
            xs.append(numbers[index])
            ys.append(numbers[index + 1])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def read_jcamp(path: str | Path, encoding: Optional[str] = None) -> dict[str, Any]:
    """Read a JCAMP-DX file into ``x``, ``y`` and its header.

    Returns a dictionary rather than a :class:`Spectrum` because the same
    format carries diffractograms and chromatograms too; the caller decides
    what it is looking at, using ``data_type`` and ``x_units``.

    Raises
    ------
    JCAMPError
        If the file has no data block, or its own consistency checks fail.
    """
    p = Path(path)
    raw = p.read_bytes()
    if encoding:
        text = raw.decode(encoding, errors="replace")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    return parse_jcamp(text, name=p.stem)


def parse_jcamp(text: str, name: str = "jcamp") -> dict[str, Any]:
    """:func:`read_jcamp` on a string already in memory."""
    records = parse_records(text)
    header: dict[str, str] = {}
    block: Optional[tuple[str, list[str]]] = None
    for label, value in records:
        if label in ("XYDATA", "XYPOINTS", "PEAKTABLE"):
            kind = "xydata" if label == "XYDATA" else "xy"
            body = value.split("\n")[1:] if "\n" in value else []
            block = (kind, body)
            header.setdefault("_form", value.split("\n")[0].strip())
        elif label == "END":
            continue
        else:
            header.setdefault(label, value)
    if block is None:
        raise JCAMPError("no hay bloque de datos (##XYDATA, ##XYPOINTS o ##PEAK TABLE)")

    x_factor = _number(header.get("XFACTOR"), 1.0)
    y_factor = _number(header.get("YFACTOR"), 1.0)
    kind, lines = block

    if kind == "xy":
        x, y = _decode_block(lines, "xy")
        x, y = x * x_factor, y * y_factor
        order = np.argsort(x)
        x, y = x[order], y[order]
    else:
        written_x, rows = _decode_block(lines, "xydata")
        y = np.concatenate([np.asarray(r, dtype=float) for r in rows]) * y_factor
        first_x = _number(header.get("FIRSTX"), None)
        last_x = _number(header.get("LASTX"), None)
        points = int(_number(header.get("NPOINTS"), len(y)) or len(y))
        if first_x is None or last_x is None:
            raise JCAMPError("un bloque X++(Y..Y) necesita ##FIRSTX y ##LASTX")
        if len(y) != points:
            raise JCAMPError(
                f"##NPOINTS dice {points} y el bloque trae {len(y)}: "
                "el archivo está truncado o concatenado"
            )
        x = np.linspace(first_x * x_factor, last_x * x_factor, points)
        _check_x(written_x * x_factor, x, len(rows), rows)

    return {
        "x": x,
        "y": y,
        "title": header.get("TITLE", name),
        "data_type": header.get("DATATYPE", "").strip().upper(),
        "x_units": X_UNITS.get(
            re.sub(r"\s", "", header.get("XUNITS", "")).upper(),
            header.get("XUNITS", "").strip(),
        ),
        "y_units": header.get("YUNITS", "").strip(),
        "laser_nm": _laser(header),
        "header": header,
    }


def _check_x(written: np.ndarray, computed: np.ndarray, rows: int, blocks) -> None:
    """Compare each line's own X against where it should have started.

    The tolerance is one X-step: a rounded FIRSTX in the header is normal
    and harmless, a line in the wrong place is a truncated file.
    """
    if written.size < 2 or computed.size < 2:
        return
    starts = np.cumsum([0] + [len(r) for r in blocks[:-1]])
    expected = computed[np.clip(starts, 0, computed.size - 1)]
    step = abs(computed[1] - computed[0])
    bad = np.abs(written - expected) > max(step * 1.5, 1e-9)
    if bad.any():
        first = int(np.argmax(bad))
        raise JCAMPError(
            f"la línea {first + 1} del bloque dice X={written[first]:.6g} "
            f"y debería empezar en {expected[first]:.6g}: el archivo no es "
            "consistente consigo mismo"
        )


def _number(value: Optional[str], default: Optional[float]) -> Optional[float]:
    if value is None:
        return default
    match = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[Ee][-+]?[0-9]+)?", value)
    return float(match.group()) if match else default


def _laser(header: dict[str, str]) -> Optional[float]:
    """The excitation wavelength, from wherever this vendor put it."""
    for key in ("LASERWAVELENGTH", "EXCITATIONWAVELENGTH", "LASER",
                "SPECTROMETERDATASYSTEM", "ORIGIN", "COMMENTS", "TITLE"):
        value = header.get(key)
        if not value:
            continue
        match = re.search(r"(\d{3,4}(?:\.\d+)?)\s*nm", value, re.IGNORECASE)
        if match:
            laser = float(match.group(1))
            if 200.0 <= laser <= 1200.0:
                return laser
    return None


def write_jcamp(
    x,
    y,
    path: str | Path,
    title: str = "espectro",
    data_type: str = "RAMAN SPECTRUM",
    x_units: str = "1/CM",
    y_units: str = "ARBITRARY UNITS",
    laser_nm: Optional[float] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a JCAMP-DX 4.24 file in the plain ``(XY..XY)`` form.

    Deliberately uncompressed. The compressed forms save space that nobody
    is short of any more, and every one of them is a way for a file to be
    read wrong by somebody else's parser.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x e y deben tener la misma forma")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"##TITLE={title}",
        "##JCAMP-DX=4.24",
        "##DATA TYPE=" + data_type,
        f"##XUNITS={x_units}",
        f"##YUNITS={y_units}",
        f"##FIRSTX={x[0]:.6f}" if x.size else "##FIRSTX=0",
        f"##LASTX={x[-1]:.6f}" if x.size else "##LASTX=0",
        f"##NPOINTS={x.size}",
        "##XFACTOR=1.0",
        "##YFACTOR=1.0",
    ]
    if laser_nm:
        lines.insert(3, f"##LASER WAVELENGTH={laser_nm:g} nm")
    for key, value in (extra or {}).items():
        lines.append(f"##{key}={value}")
    lines.append("##XYPOINTS=(XY..XY)")
    lines.extend(f"{xi:.6f}, {yi:.6f}" for xi, yi in zip(x, y))
    lines.append("##END=")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


__all__ = [
    "JCAMPError",
    "X_UNITS",
    "parse_jcamp",
    "parse_records",
    "read_jcamp",
    "write_jcamp",
]
