"""Readers for photoelectron spectra: PHI ``.spe``, VAMAS, and text.

Three formats, in decreasing order of how much the file tells you:

``.spe`` (PHI MultiPak)
    A hybrid: an ASCII header between ``SOFH`` and ``EOFH`` that names the
    anode, the pass energy, the dwell time and every region, followed by the
    intensities — which in the files the instrument writes are **binary**.
    The header is the valuable part, because it carries the numbers that
    quantification and resolution limits depend on and that nobody
    remembers to write down.

``.vms`` / ``.npl`` (VAMAS, ISO 14976)
    The interchange format. Plain text, one value per line, and rigid: a
    block declares how many ordinate values follow, so a misparse is
    detectable rather than silent.

Plain text
    Two columns. Everything the instrument knew has been thrown away, so
    the photon energy and the pass energy have to be supplied by hand or
    the analysis that needs them refuses to run.

**On the binary part of a ``.spe``.** This package's rule elsewhere is that
binary vendor formats are refused with instructions rather than guessed at,
because a mis-guessed layout produces numbers that look fine. That rule is
kept here in substance: the intensities are recovered only when the
recovery can be *checked* — the block must have exactly the length the
ASCII header declares, be finite and non-negative throughout, and be the
only candidate in the file. If two layouts fit, or none does, the file is
refused with the export that replaces it. What is never done is taking the
first plausible-looking block and calling it a spectrum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .spectrum import DEFAULT_WORK_FUNCTION, XPSError, XPSSpectrum, source_energy

#: What to tell somebody whose file cannot be read.
SPE_EXPORT_ADVICE = (
    "en MultiPak, «File → Export → ASCII» (o «Save As» con tipo texto) "
    "escribe el mismo espectro en dos columnas y se lee sin ambigüedad"
)

_HEADER_END = b"EOFH"
_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


# ----------------------------------------------------------------------
# PHI .spe
# ----------------------------------------------------------------------
@dataclass
class SpeRegion:
    """One ``SpectralRegDef`` line: what the instrument says it measured."""

    number: int
    name: str
    atomic_number: int
    points: int
    step_ev: float
    start_ev: float
    stop_ev: float
    dwell_s: float
    pass_energy: float
    description: str = ""

    @property
    def axis(self) -> np.ndarray:
        """The binding-energy axis, from the endpoints and the point count.

        Built with :func:`numpy.linspace` from ``start``/``stop``/``points``
        rather than by stepping, so a rounding disagreement in the declared
        step cannot accumulate across a 1400 eV survey.
        """
        return np.linspace(self.start_ev, self.stop_ev, self.points)

    @property
    def implied_step(self) -> float:
        """The step the endpoints and point count imply, in eV."""
        if self.points < 2:
            return 0.0
        return abs(self.stop_ev - self.start_ev) / (self.points - 1)

    def check(self) -> Optional[str]:
        """Why this region definition is not self-consistent, if it is not.

        This is the parser's own check on itself. The header gives the
        endpoints, the point count *and* the step, which is one number more
        than is needed; if they disagree, the fields were not where this
        reader thought they were, and going on would mean writing a
        plausible energy axis over the wrong data.
        """
        if self.points < 5:
            return f"la región «{self.name}» declara {self.points} puntos"
        # The step is compared in MAGNITUDE, because the instrument writes
        # it SIGNED and a photoelectron scan normally runs downwards in
        # binding energy: a C 1s from 295 to 280 eV declares -0.05, and
        # that is the usual case, not a broken file. Demanding a positive
        # step refused every such region -- which is most of them -- with
        # a message about the step that read like a corrupt header.
        step = abs(self.step_ev)
        if step <= 0:
            return f"la región «{self.name}» declara un paso de {self.step_ev}"
        implied = self.implied_step
        if abs(implied - step) > 0.02 * step:
            return (
                f"la región «{self.name}» no cuadra consigo misma: "
                f"{self.start_ev:g}→{self.stop_ev:g} eV en {self.points} puntos "
                f"son {implied:.4f} eV por punto y la cabecera declara "
                f"{step:.4f}"
            )
        return None


def parse_spe_header(raw: bytes) -> tuple[dict[str, Any], list[SpeRegion], int]:
    """Split a ``.spe`` into header fields, regions, and the data offset.

    Returns
    -------
    tuple
        ``(fields, regions, offset)`` — the ``key: value`` lines as a dict
        (repeated keys collected into lists), the parsed region
        definitions, and the byte offset just past ``EOFH``.
    """
    # SOFH does not have to be at byte zero. Some exports put a short
    # preamble in front of it, and refusing those on a byte-offset
    # technicality reads to the user as "this program cannot open my
    # files".
    start = raw.find(b"SOFH", 0, 8192)
    end = raw.find(_HEADER_END)
    if start < 0 or end < 0 or end < start:
        raise XPSError(_spe_diagnosis(raw))
    text = raw[start:end].decode("latin-1")
    offset = end + len(_HEADER_END)
    while offset < len(raw) and raw[offset : offset + 1] in (b"\r", b"\n"):
        offset += 1

    fields: dict[str, Any] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if key in fields:
            existing = fields[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                fields[key] = [existing, value]
        else:
            fields[key] = value

    definitions = fields.get("SpectralRegDef", [])
    if isinstance(definitions, str):
        definitions = [definitions]
    regions = [_region_from(line) for line in definitions]
    declared = fields.get("NoSpectralReg")
    if declared is not None and regions:
        try:
            expected = int(float(str(declared).split()[0]))
        except ValueError:
            expected = len(regions)
        if expected != len(regions):
            raise XPSError(
                f"la cabecera declara {expected} regiones y define "
                f"{len(regions)}: el archivo está truncado o mal escrito"
            )
    return fields, regions, offset


def _looks_binary(head: bytes) -> bool:
    """Whether a file is binary, by its control characters.

    Vendors write binary inside ``.txt`` and text inside ``.spe``, so the
    extension decides nothing and the content decides everything.
    """
    sample = head[:512]
    if not sample:
        return False
    control = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return b"\x00" in sample or control > 0.02 * len(sample)


def _spe_diagnosis(raw: bytes) -> str:
    """Say what the file actually is, rather than what it is not.

    "No parece un .spe de PHI" is true and useless: the user cannot act
    on it, and ``.spe`` is not one format. PHI MultiPak writes an ASCII
    SOFH/EOFH header; Princeton Instruments WinSpec writes a completely
    unrelated binary CCD format under the same extension; and some
    instrument software writes plain two-column text and calls it
    ``.spe`` too. Naming which one this is turns a dead end into one
    action.
    """
    head = raw[:64]
    printable = "".join(
        chr(b) if 32 <= b < 127 else "." for b in head[:32]
    )
    binary = _looks_binary(raw)

    if raw[:2] in (b"\x00\x00", b"\x1e\x00", b"\x06\x00") and binary:
        kind = (
            "parece un .spe de WinSpec/Princeton Instruments, que es un "
            "formato de imagen de CCD y no tiene nada que ver con XPS "
            "aunque comparta extensión"
        )
    elif binary:
        kind = (
            "es binario y no lleva la cabecera SOFH/EOFH de PHI, así que no "
            "se puede saber dónde están las intensidades. Adivinar la "
            "disposición de un binario es como se leen mal unos datos sin "
            "que nadie se entere, y por eso este lector no lo intenta"
        )
    else:
        kind = (
            "es texto pero no lleva la cabecera SOFH/EOFH de PHI. Si son dos "
            "columnas (energía e intensidad), renómbralo a .txt y ábrelo: el "
            "lector de texto lo acepta y deduce si el eje es de enlace o "
            "cinético"
        )
    return (
        f"no se reconoce como .spe de PHI MultiPak: {kind}. "
        f"Empieza por: {printable!r}. " + SPE_EXPORT_ADVICE
    )


def _region_from(line: str) -> SpeRegion:
    parts = line.split()
    if len(parts) < 12:
        raise XPSError(
            f"SpectralRegDef con {len(parts)} campos, se esperaban al menos 12: "
            f"{line!r}"
        )
    try:
        return SpeRegion(
            number=int(float(parts[0])),
            name=parts[2],
            atomic_number=int(float(parts[3])),
            points=int(float(parts[4])),
            step_ev=float(parts[5]),
            start_ev=float(parts[6]),
            stop_ev=float(parts[7]),
            dwell_s=float(parts[10]),
            pass_energy=float(parts[11]),
            description=" ".join(parts[12:]).strip('"'),
        )
    except ValueError as exc:
        raise XPSError(f"SpectralRegDef ilegible ({exc}): {line!r}") from None


def _source_from_header(fields: dict[str, Any]) -> tuple[Optional[float], bool, str]:
    """Photon energy, whether monochromated, and the raw string.

    ``XraySource: Al 1486.6 mono`` and ``XraySource: Mg 1253.6`` are both
    seen. The energy is taken from the line when it carries one, and from
    the anode name only as a fallback — a monochromated Al source really is
    at 1486.6 eV, but the instrument's own number is the one to believe.
    """
    text = ""
    for key in ("XraySource", "XrayAnode", "SourceAnode", "XRaySource"):
        value = fields.get(key)
        if value:
            text = value if isinstance(value, str) else value[0]
            break
    if not text:
        return None, True, ""
    mono = "mono" in text.lower()
    numbers = [float(n) for n in _NUMBER.findall(text)]
    energy = next((n for n in numbers if 100.0 < n < 12000.0), None)
    if energy is None:
        anode = text.split()[0] if text.split() else ""
        try:
            energy = source_energy(anode)
        except XPSError:
            energy = None
    return energy, mono, text


def _looks_like_a_spectrum(values: np.ndarray) -> bool:
    """Whether a candidate block behaves like counts rather than like noise.

    Three necessary conditions, all cheap and all things random bytes read
    as floats fail almost immediately: everything finite, nothing negative
    (a count is not negative), and neighbouring points correlated — the
    mean step between adjacent values has to be a small fraction of the
    whole range, which is true of any spectrum and false of garbage.
    """
    if values.size < 5 or not np.all(np.isfinite(values)):
        return False
    if float(values.min()) < 0.0:
        return False
    span = float(values.max() - values.min())
    if span <= 0.0:
        return False
    return float(np.mean(np.abs(np.diff(values)))) < 0.25 * span


def _recover_binary(payload: bytes, regions: list[SpeRegion]) -> list[np.ndarray]:
    """Find the intensities in a ``.spe`` binary payload, or refuse.

    The search is over the two floating layouts PHI writes (little-endian
    float32 and float64) and over byte offsets; a candidate is accepted
    only if **one** offset in the whole file yields, for every region in
    order and back to back, a block of exactly the declared length that
    passes :func:`_looks_like_a_spectrum`. Two surviving candidates are as
    bad as none, and both are refused: an ambiguous layout read one way is
    a silently wrong spectrum.
    """
    total = sum(region.points for region in regions)
    if total <= 0:
        raise XPSError("la cabecera no define ningún punto que leer")

    hits: list[tuple[str, int, list[np.ndarray]]] = []
    for name, dtype in (("float32", np.dtype("<f4")), ("float64", np.dtype("<f8"))):
        need = total * dtype.itemsize
        if need > len(payload):
            continue
        for offset in range(0, len(payload) - need + 1, 4):
            values = np.frombuffer(payload, dtype=dtype, count=total, offset=offset)
            blocks: list[np.ndarray] = []
            cursor = 0
            good = True
            for region in regions:
                block = np.asarray(values[cursor : cursor + region.points], dtype=float)
                cursor += region.points
                if not _looks_like_a_spectrum(block):
                    good = False
                    break
                blocks.append(block)
            if good:
                hits.append((name, offset, blocks))
                if len(hits) > 1:
                    break
        if len(hits) > 1:
            break

    if not hits:
        raise XPSError(
            "la cabecera del .spe se ha leído bien, pero el bloque de datos "
            "binario no encaja con lo que declara "
            f"({total} puntos en {len(regions)} región/regiones). No se va a "
            "adivinar su disposición: unos datos leídos mal no se notan. "
            + SPE_EXPORT_ADVICE
        )
    if len(hits) > 1:
        raise XPSError(
            "el bloque de datos binario del .spe admite más de una lectura "
            "y las dos son igual de plausibles, así que cualquiera de ellas "
            "podría ser la equivocada. " + SPE_EXPORT_ADVICE
        )
    return hits[0][2]


def _ascii_payload(payload: bytes, regions: list[SpeRegion]) -> Optional[list[np.ndarray]]:
    """Intensities when the ``.spe`` was exported as ASCII, else ``None``."""
    if b"\x00" in payload[:512]:
        return None
    try:
        text = payload.decode("latin-1")
    except UnicodeDecodeError:      # pragma: no cover - latin-1 never fails
        return None
    rows: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] not in "-+.0123456789":
            continue
        found = _NUMBER.findall(stripped)
        if len(found) < 2:
            continue
        rows.append([float(value) for value in found])
    if not rows:
        return None
    width = min(len(row) for row in rows)
    table = np.array([row[:width] for row in rows], dtype=float)
    total = sum(region.points for region in regions)
    if table.shape[0] == total and width >= 2:
        # One long two-column block: the regions follow each other.
        out, cursor = [], 0
        for region in regions:
            out.append(table[cursor : cursor + region.points, 1])
            cursor += region.points
        return out
    if len(regions) == 1 and table.shape[0] >= 5:
        return [table[:, 1]]
    if width >= len(regions) + 1 and table.shape[0] == regions[0].points:
        # One shared axis and one intensity column per region.
        return [table[:, i + 1] for i in range(len(regions))]
    return None


def read_spe(
    path: str | Path,
    work_function: Optional[float] = None,
    photon_energy: Optional[float] = None,
) -> list[XPSSpectrum]:
    """Read a PHI MultiPak ``.spe`` into one spectrum per region.

    Parameters
    ----------
    path:
        The file.
    work_function:
        Overrides the analyser work function in the header. Only affects
        the kinetic-energy scale and therefore where Auger lines are
        expected; it does not move the binding-energy axis.
    photon_energy:
        Overrides the anode energy in the header, for a file that does not
        record it.

    Returns
    -------
    list of XPSSpectrum
        In the order the regions appear in the header. The survey, when
        present, is usually first.

    Raises
    ------
    XPSError
        If the header is absent or self-inconsistent, or if the
        intensities cannot be recovered *and checked*.
    """
    path = Path(path)
    raw = path.read_bytes()
    fields, regions, offset = parse_spe_header(raw)
    if not regions:
        raise XPSError(
            f"{path.name}: la cabecera no define ninguna región "
            "(SpectralRegDef). " + SPE_EXPORT_ADVICE
        )
    for region in regions:
        problem = region.check()
        if problem:
            raise XPSError(f"{path.name}: {problem}. " + SPE_EXPORT_ADVICE)

    payload = raw[offset:]
    blocks = _ascii_payload(payload, regions)
    binary = blocks is None
    if binary:
        blocks = _recover_binary(payload, regions)

    energy, mono, source_text = _source_from_header(fields)
    if photon_energy is not None:
        energy = float(photon_energy)
    phi = work_function
    if phi is None:
        for key in ("AnalyserWorkFcn", "AnalyzerWorkFcn", "WorkFunction"):
            if key in fields:
                try:
                    phi = float(_NUMBER.findall(str(fields[key]))[0])
                except (IndexError, ValueError):
                    phi = None
                break
    if phi is None:
        phi = DEFAULT_WORK_FUNCTION

    unit = "cuentas"
    for key in ("YUnit", "IntensityUnit", "AcqUnit"):
        value = str(fields.get(key, ""))
        if "/s" in value or "per second" in value.lower():
            unit = "cuentas/s"
    sweeps = None
    for key in ("SurveyNumCycles", "NumCycles", "NoSweeps"):
        if key in fields:
            try:
                sweeps = int(float(_NUMBER.findall(str(fields[key]))[0]))
            except (IndexError, ValueError):
                sweeps = None
            break

    stem = path.stem
    out: list[XPSSpectrum] = []
    for region, counts in zip(regions, blocks):
        metadata: dict[str, Any] = {
            "archivo": str(path),
            "formato": "PHI .spe" + (" (binario)" if binary else " (ASCII)"),
            "region_declarada": region.name,
            "fuente": source_text,
            "paso_declarado_ev": region.step_ev,
        }
        for key in ("InstrumentModel", "acq_filename", "acq_file_date", "SampleID",
                    "AnalyserMode", "FileDesc"):
            if key in fields:
                metadata[key] = fields[key]
        if binary:
            metadata["aviso_binario"] = (
                "las intensidades se han recuperado del bloque binario y "
                "comprobadas contra las longitudes que declara la cabecera; "
                "si algo no cuadra en el espectro, exporta el ASCII y "
                "compáralo"
            )
        out.append(
            XPSSpectrum(
                binding_energy=region.axis,
                counts=np.asarray(counts, dtype=float),
                photon_energy=energy,
                pass_energy=region.pass_energy,
                dwell_s=region.dwell_s,
                sweeps=sweeps,
                work_function=float(phi),
                monochromated=mono,
                intensity_unit=unit,
                region=region.name,
                name=f"{stem}:{region.name}",
                metadata=metadata,
            )
        )
    return out


# ----------------------------------------------------------------------
# VAMAS / ISO 14976
# ----------------------------------------------------------------------
#: A UTF-8 byte-order mark, and the same three bytes seen through
#: latin-1, which is how this reader decodes the file.
_BYTE_ORDER_MARKS = ("\ufeff", "\u00ef\u00bb\u00bf")


def _without_mark(line: str) -> str:
    for mark in _BYTE_ORDER_MARKS:
        if line.startswith(mark):
            return line[len(mark):]
    return line


def _vamas_lines(text: str) -> list[str]:
    return [line.rstrip("\r") for line in text.split("\n")]


class _Cursor:
    """A position in a VAMAS file, so a misparse says where it happened."""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines
        self.index = 0

    def next(self) -> str:
        if self.index >= len(self.lines):
            raise XPSError(
                f"el archivo VAMAS se acaba en la línea {self.index}: está truncado"
            )
        value = self.lines[self.index].strip()
        self.index += 1
        return value

    def number(self, what: str) -> float:
        raw = self.next()
        try:
            return float(raw)
        except ValueError:
            raise XPSError(
                f"línea {self.index} del VAMAS: se esperaba {what} y hay {raw!r}"
            ) from None

    def integer(self, what: str) -> int:
        return int(self.number(what))

    def skip(self, count: int) -> None:
        self.index += max(0, int(count))


def read_vamas(path: str | Path) -> list[XPSSpectrum]:
    """Read a VAMAS (ISO 14976) file into one spectrum per block.

    The format is positional — every field is one line and the meaning of
    line *n* depends on what line *n−1* said — so this reader checks itself
    at the one place the format makes it possible: each block declares how
    many ordinate values follow, and that count has to be there. A
    misparsed header almost always lands on the wrong count and the file is
    refused instead of silently producing an axis that does not belong to
    the data.

    Only the ``REGULAR`` scan mode is read. An irregular abscissa is legal
    in the standard and rare in practice, and reading it as regular would
    put every point in the wrong place.
    """
    path = Path(path)
    text = path.read_text(encoding="latin-1")
    lines = _vamas_lines(text)
    # Only what comes BEFORE the identifier is skipped, and only if it is
    # empty or a byte-order mark. A blank line further in is a field --
    # an unnamed operator, an empty comment -- and dropping those would
    # shift every value that follows, which in a positional format means
    # reading the wrong numbers rather than failing.
    start = 0
    while start < len(lines) and not _without_mark(lines[start]).strip():
        start += 1
    cursor = _Cursor(lines[start:])

    identifier = _without_mark(cursor.next())
    if "VAMAS" not in identifier.upper():
        raise XPSError(
            f"{path.name}: la primera línea no es el identificador VAMAS "
            f"({identifier[:40]!r})"
        )
    institution = cursor.next()
    instrument = cursor.next()
    cursor.next()                              # operator
    experiment = cursor.next()
    comment_count = cursor.integer("el número de líneas de comentario")
    comments = [cursor.next() for _ in range(max(0, comment_count))]
    # A PHI Quantera writes its whole SOFH/EOFH header into those comment
    # lines, and with it the binding-energy window of every region. That
    # is one number more than the block needs, so it is a check -- the
    # same reason the .spe reader compares the step against the endpoints.
    declared = _declared_windows(comments)
    experiment_mode = cursor.next().upper()
    scan_mode = cursor.next().upper()
    if scan_mode != "REGULAR":
        raise XPSError(
            f"{path.name}: modo de barrido {scan_mode!r}. Este lector solo lee "
            "REGULAR; una abscisa irregular leída como regular coloca cada "
            "punto en un sitio que no es el suyo"
        )
    if experiment_mode in ("MAP", "MAPDP", "NORM", "SDP"):
        cursor.integer("el número de regiones espectrales")
    if experiment_mode in ("MAP", "MAPDP"):
        cursor.integer("el número de posiciones de análisis")
        cursor.integer("el número de coordenadas x del mapa")
        cursor.integer("el número de coordenadas y del mapa")
    variables = cursor.integer("el número de variables experimentales")
    cursor.skip(2 * variables)
    inclusion = cursor.integer("la lista de inclusión/exclusión")
    cursor.skip(abs(inclusion))
    manual = cursor.integer("el número de entradas manuales por bloque")
    cursor.skip(manual)
    cursor.skip(cursor.integer("las entradas futuras de experimento"))
    future_block = cursor.integer("las entradas futuras de bloque")
    blocks = cursor.integer("el número de bloques")

    out: list[XPSSpectrum] = []
    for number in range(blocks):
        out.append(
            _read_vamas_block(
                cursor,
                path=path,
                index=number,
                experiment=experiment,
                institution=institution,
                instrument=instrument,
                experiment_mode=experiment_mode,
                variables=variables,
                future_block=future_block,
                declared=declared,
            )
        )
    return out


def _declared_windows(comments: list[str]) -> dict[str, tuple[float, float]]:
    """Binding-energy windows a PHI header states, by region name.

    ``SpectralRegDef: 2 1 C1s 6 401 -0.0500 298.0000 278.0000 ...`` --
    name at field 2, endpoints at 6 and 7.
    """
    out: dict[str, tuple[float, float]] = {}
    for line in comments:
        if "SpectralRegDef" not in line or ":" not in line:
            continue
        parts = line.partition(":")[2].split()
        if len(parts) < 8:
            continue
        try:
            out.setdefault(parts[2].lower(),
                           (float(parts[6]), float(parts[7])))
        except ValueError:
            continue
    return out


def _declared_for(declared, species: str, transition: str):
    """The window for one block, matched on ``C`` + ``1s`` -> ``c1s``."""
    if not declared:
        return None
    key = f"{species}{transition}".replace(" ", "").lower()
    return declared.get(key)


def _read_vamas_block(
    cursor: _Cursor,
    path: Path,
    index: int,
    experiment: str,
    institution: str,
    instrument: str,
    experiment_mode: str,
    variables: int,
    future_block: int,
    declared: dict[str, tuple[float, float]] | None = None,
) -> XPSSpectrum:
    """One VAMAS block, in the field order of ISO 14976.

    The order matters more than it looks. The values of the experimental
    variables sit near the *top* of the block, just after the technique,
    not at the bottom with the rest of the numbers — put them at the end
    and every field from the source label onwards is read one line late,
    which lands the photon energy on the source strength and still looks
    like a number.
    """
    block_id = cursor.next()
    sample = cursor.next()
    cursor.skip(7)                             # year…seconds, GMT offset
    cursor.skip(cursor.integer("las líneas de comentario del bloque"))
    technique = cursor.next()
    upper = technique.upper()
    if experiment_mode in ("MAP", "MAPDP"):
        cursor.skip(2)                         # x, y of the analysis position
    cursor.skip(variables)                     # values of the experimental variables
    source_label = cursor.next()
    if experiment_mode in ("MAPDP", "MAPSVDP", "SDP", "SDPSV") or upper in (
        "FABMS", "FABMS ENERGY SPEC", "ISS", "SIMS", "SIMS ENERGY SPEC",
        "SNMS", "SNMS ENERGY SPEC",
    ):
        cursor.skip(3)                         # sputtering ion Z, atoms, charge
    photon_energy = cursor.number("la energía característica de la fuente")
    cursor.next()                              # source strength
    cursor.skip(2)                             # beam width x, y
    if experiment_mode in ("MAP", "MAPDP", "MAPSV", "MAPSVDP", "SEM"):
        cursor.skip(2)                         # field of view x, y
    if experiment_mode in ("MAPSV", "MAPSVDP", "SEM"):
        cursor.skip(6)                         # first and last linescan corners
    cursor.skip(2)                             # source polar angle, azimuth
    analyser_mode = cursor.next()
    pass_energy = cursor.number("la energía de paso")
    if upper == "AES DIFF":
        cursor.next()                          # differential width
    cursor.next()                              # magnification of the transfer lens
    work_function = cursor.number("la función de trabajo")
    cursor.next()                              # target bias
    cursor.skip(2)                             # analysis width x, y
    cursor.skip(2)                             # take-off polar angle, azimuth
    species = cursor.next()
    transition = cursor.next()
    cursor.next()                              # charge of the detected particle
    abscissa_label = cursor.next()
    abscissa_units = cursor.next()
    start = cursor.number("el primer valor de la abscisa")
    increment = cursor.number("el incremento de la abscisa")
    corresponding = cursor.integer("el número de variables correspondientes")
    ordinate_labels = []
    for _ in range(corresponding):
        ordinate_labels.append(cursor.next())
        cursor.next()                          # units
    cursor.next()                              # signal mode
    dwell = cursor.number("el tiempo de adquisición por canal")
    scans = cursor.integer("el número de barridos")
    cursor.next()                              # signal time correction
    if experiment_mode in ("MAPDP", "MAPSVDP", "SDP", "SDPSV") and upper in (
        "AES DIFF", "AES DIR", "EDX", "ELS", "UPS", "XPS", "XRF",
    ):
        cursor.skip(7)                         # sputtering source description
    cursor.skip(3)                             # sample tilt polar, azimuth, rotation
    cursor.skip(3 * cursor.integer("los parámetros numéricos adicionales"))
    cursor.skip(future_block)
    total = cursor.integer("el número de valores de ordenada")
    if corresponding <= 0:
        raise XPSError(f"{path.name}: el bloque {index + 1} no declara ordenadas")
    if total % corresponding:
        raise XPSError(
            f"{path.name}: el bloque {index + 1} declara {total} ordenadas para "
            f"{corresponding} variables, que no es múltiplo"
        )
    points = total // corresponding
    cursor.skip(2 * corresponding)             # min and max of each ordinate

    values = np.empty(total, dtype=float)
    for position in range(total):
        values[position] = cursor.number(f"la ordenada {position + 1}")
    counts = values.reshape(points, corresponding)[:, 0] if corresponding > 1 else values

    axis = start + increment * np.arange(points, dtype=float)
    label = f"{abscissa_label} {abscissa_units}".lower()
    if "kinetic" in label or "cinétic" in label:
        # The standard says the abscissa is the kinetic energy as
        # measured, so the work function comes off it. PHI's export has
        # already taken it off: its own header declares the C 1s window as
        # 298.0 to 278.0 eV, and hv - KE reproduces that exactly while
        # hv - KE - phi misses it by 4.05 eV -- the whole work function,
        # in every region, silently. Neither convention can be deduced
        # from the block, so when the file states the window, the file
        # decides; when it does not, the standard does.
        binding = photon_energy - axis - work_function
        original = "energía cinética"
        window = _declared_for(declared, species, transition)
        if window is not None:
            plain = photon_energy - axis
            low, high = min(window), max(window)
            centre = 0.5 * (low + high)
            if abs(np.median(plain) - centre) < abs(np.median(binding) - centre):
                binding = plain
                original = ("energía cinética (ya corregida por la función "
                            "de trabajo en el archivo)")
    else:
        binding = axis
        original = "energía de enlace"
    region = f"{species} {transition}".strip() or (block_id or f"bloque {index + 1}")
    metadata = {
        "archivo": str(path),
        "formato": "VAMAS ISO 14976",
        "experimento": experiment,
        "institución": institution,
        "instrumento": instrument,
        "muestra": sample,
        "técnica": technique,
        "modo_analizador": analyser_mode,
        "fuente": source_label,
        "eje_original": f"{abscissa_label} ({abscissa_units}) — {original}",
        "ordenadas": ordinate_labels,
        "bloque": block_id,
    }
    if upper not in ("XPS", "UPS", "XPS DIFF"):
        metadata["aviso"] = (
            f"el bloque dice que la técnica es {technique!r}, no XPS; lo que "
            "se lea de él se interpretará como si lo fuera"
        )
    return XPSSpectrum(
        binding_energy=binding,
        counts=counts,
        photon_energy=photon_energy if photon_energy > 0 else None,
        pass_energy=pass_energy if pass_energy > 0 else None,
        dwell_s=dwell if dwell > 0 else None,
        sweeps=scans if scans > 0 else None,
        work_function=work_function if work_function else DEFAULT_WORK_FUNCTION,
        monochromated="mono" in source_label.lower(),
        region=region,
        name=f"{path.stem}:{region}",
        metadata=metadata,
    )


# ----------------------------------------------------------------------
# writing
# ----------------------------------------------------------------------
VAMAS_IDENTIFIER = (
    "VAMAS Surface Chemical Analysis Standard Data Transfer Format 1988 May 4"
)


def write_vamas(
    path: str | Path,
    spectra: "XPSSpectrum | list[XPSSpectrum]",
    institution: str = "ramancarbon",
    experiment: str = "ramancarbon",
) -> Path:
    """Write spectra as a VAMAS (ISO 14976) file, one block each.

    Why bother, when the measurement already exists as a file: because what
    comes *out* of this package is not what went in. A spectrum that has
    been charge-referenced has a different axis from the one the instrument
    wrote, and handing a collaborator the original plus a sentence about
    the shift is how the shift gets applied twice. VAMAS is the format
    CasaXPS, KolXPD and Avantage all read.

    The axis is written as binding energy, which is how it is held in
    memory; the block records the photon energy and work function that were
    used, so the kinetic scale can be rebuilt exactly.
    """
    path = Path(path)
    if isinstance(spectra, XPSSpectrum):
        spectra = [spectra]
    if not spectra:
        raise XPSError("no hay ningún espectro que escribir")

    lines: list[str] = [
        VAMAS_IDENTIFIER,
        institution,
        str(spectra[0].metadata.get("instrumento", "desconocido")),
        "",                                    # operator
        experiment,
        "1",
        "escrito por ramancarbon; el eje es energía de enlace",
        "NORM",
        "REGULAR",
        str(len(spectra)),                     # number of spectral regions
        "0",                                   # experimental variables
        "0",                                   # inclusion/exclusion list
        "0",                                   # manually entered items
        "0",                                   # future experiment entries
        "0",                                   # future block entries
        str(len(spectra)),
    ]
    for index, spectrum in enumerate(spectra):
        lines.extend(_vamas_block_lines(spectrum, index))
    lines.append("end of experiment")
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


def _source_label(spectrum: XPSSpectrum) -> str:
    """The source string for a VAMAS block.

    Whether the source was monochromated is not decoration: an
    unmonochromated anode puts X-ray satellites 8–12 eV below every line,
    and a reader that loses the flag will call one of them a chemical
    state. VAMAS has no field for it, so it travels in the source label,
    which is where every other program looks for it too.
    """
    label = str(spectrum.metadata.get("fuente", "") or "")
    if not label:
        label = "Al Ka" if (spectrum.photon_energy or 0) > 1400 else "Mg Ka"
    if spectrum.monochromated and "mono" not in label.lower():
        label = f"{label} mono"
    return label


def _vamas_block_lines(spectrum: XPSSpectrum, index: int) -> list[str]:
    """One block, in the same field order :func:`read_vamas` expects."""
    energy = spectrum.binding_energy
    step = float(energy[1] - energy[0]) if energy.size > 1 else 1.0
    species, _, transition = (spectrum.region or "").partition(" ")
    shift = spectrum.metadata.get("charge_shift_ev")
    comment = [f"región {spectrum.region or spectrum.name}"]
    if shift:
        comment.append(f"eje desplazado {float(shift):+.3f} eV al referenciar la carga")
    if spectrum.history:
        comment.append("historia: " + "; ".join(spectrum.history))
    return [
        spectrum.name or f"bloque {index + 1}",
        str(spectrum.metadata.get("muestra", "")),
        "0", "0", "0", "0", "0", "0", "0",     # date, time, GMT offset
        str(len(comment)), *comment,
        "XPS",
        _source_label(spectrum),
        f"{spectrum.photon_energy if spectrum.photon_energy else 0.0:.4f}",
        "0",                                   # source strength
        "0", "0",                              # beam width x, y
        "0", "0",                              # source polar angle, azimuth
        "FAT",
        f"{spectrum.pass_energy if spectrum.pass_energy else 0.0:.4f}",
        "0",                                   # magnification
        f"{spectrum.work_function:.4f}",
        "0",                                   # target bias
        "0", "0",                              # analysis width x, y
        "0", "0",                              # take-off polar angle, azimuth
        species or (spectrum.region or "?"),
        transition,
        "-1",                                  # charge of the detected particle
        "binding energy",
        "eV",
        f"{float(energy[0]):.6f}",
        f"{step:.6f}",
        "1",                                   # corresponding variables
        "Intensity",
        spectrum.intensity_unit,
        "pulse counting",
        f"{spectrum.dwell_s if spectrum.dwell_s else 0.0:.4f}",
        str(int(spectrum.sweeps or 0)),
        "0",                                   # signal time correction
        "0", "0", "0",                         # sample tilt polar, azimuth, rotation
        "0",                                   # additional numerical parameters
        str(spectrum.counts.size),
        f"{float(spectrum.counts.min()):.6f}",
        f"{float(spectrum.counts.max()):.6f}",
        *[f"{value:.6f}" for value in spectrum.counts],
    ]


# ----------------------------------------------------------------------
# plain text
# ----------------------------------------------------------------------
def read_xps_text(
    path: str | Path,
    photon_energy: Optional[float] = None,
    pass_energy: Optional[float] = None,
    axis: str = "auto",
    work_function: float = DEFAULT_WORK_FUNCTION,
    dwell_s: Optional[float] = None,
    sweeps: Optional[int] = None,
    region: str = "",
) -> XPSSpectrum:
    """Read a two-column text export.

    Parameters
    ----------
    axis:
        ``"binding"``, ``"kinetic"`` or ``"auto"``. ``auto`` reads the
        header text for the words and falls back to binding energy, which
        is what essentially every XPS export writes; the choice made is
        recorded in ``metadata["eje"]`` and, when it was a fallback rather
        than a reading, in ``metadata["aviso"]``. A kinetic axis needs
        ``photon_energy`` to be converted at all.

    Notes
    -----
    A text file has thrown away the pass energy and the dwell time. The
    pass energy sets the resolution floor and the dwell time is what turns
    a counts-per-second axis back into counting statistics, so pass them in
    if the analysis is going to be believed.
    """
    path = Path(path)
    raw = path.read_bytes()
    if b"\x00" in raw[:512]:
        raise XPSError(
            f"{path.name} es binario, no texto. Si es un .spe de PHI, ábrelo "
            "con read_spe; si es de otro equipo, " + SPE_EXPORT_ADVICE
        )
    text = raw.decode("latin-1")
    header: list[str] = []
    rows: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] not in "-+.0123456789":
            header.append(stripped)
            continue
        found = _NUMBER.findall(stripped.replace(";", " ").replace(",", " "))
        if len(found) >= 2:
            rows.append([float(value) for value in found[:2]])
    if len(rows) < 5:
        raise XPSError(
            f"{path.name}: solo {len(rows)} filas de dos números; no es un "
            "espectro de dos columnas"
        )
    table = np.array(rows, dtype=float)
    x, y = table[:, 0], table[:, 1]

    head = " ".join(header).lower()
    chosen, guessed = axis, False
    if axis == "auto":
        if "kinetic" in head or "cinétic" in head or "ke)" in head:
            chosen = "kinetic"
        elif "binding" in head or "enlace" in head or "be)" in head:
            chosen = "binding"
        else:
            chosen, guessed = "binding", True
    if chosen == "kinetic":
        if photon_energy is None:
            raise XPSError(
                f"{path.name}: el eje es cinético y hace falta la energía del "
                "fotón para convertirlo. Suponer aluminio desplaza todo 233 eV "
                "si en realidad era magnesio"
            )
        binding = photon_energy - x - work_function
    else:
        binding = x

    metadata: dict[str, Any] = {
        "archivo": str(path),
        "formato": "texto",
        "eje": "energía cinética" if chosen == "kinetic" else "energía de enlace",
    }
    if header:
        metadata["cabecera"] = header[:20]
    if guessed:
        metadata["aviso"] = (
            "el archivo no dice si la primera columna es energía de enlace o "
            "cinética; se ha supuesto de enlace, que es lo que exporta casi "
            "todo. Si era cinética, pásalo con axis=\"kinetic\""
        )
    if pass_energy is None:
        metadata["aviso_resolución"] = (
            "sin energía de paso no hay suelo de resolución: cualquier FWHM "
            "ajustada por debajo de la resolución del analizador es del "
            "modelo, no de la muestra"
        )
    return XPSSpectrum(
        binding_energy=binding,
        counts=y,
        photon_energy=photon_energy,
        pass_energy=pass_energy,
        dwell_s=dwell_s,
        sweeps=sweeps,
        work_function=work_function,
        region=region or path.stem,
        name=path.stem,
        metadata=metadata,
    )


def read_xps(path: str | Path, **options: Any) -> list[XPSSpectrum]:
    """Read any supported photoelectron file, deciding by its content.

    The extension is a hint and nothing more, the same rule the rest of the
    package follows: instruments write ``.txt`` holding binary and ``.dat``
    holding VAMAS.
    """
    path = Path(path)
    head = path.read_bytes()[:8192]
    try:
        sample = head.decode("latin-1")
    except UnicodeDecodeError:                  # pragma: no cover
        sample = ""
    # VAMAS is decided FIRST, and on the opening lines, because a VAMAS
    # file may legitimately CONTAIN a PHI header: the PHI Quantera writes
    # its whole SOFH/EOFH block into the comment lines of the VAMAS it
    # exports, and the block count on line 6 says how many lines it is.
    # Looking for SOFH anywhere in the first 8 kB therefore found a real
    # VAMAS file and handed it to the PHI reader, which then read the
    # regions out of the comment and tried to recover intensities from a
    # binary block that is not there. The identifier is the only thing
    # that settles what a file IS; SOFH inside it is a quotation.
    opening = sample.upper().split("\n")[:5]
    if any("VAMAS" in line for line in opening):
        return read_vamas(path)
    if path.suffix.lower() in (".vms", ".npl"):
        # The extension says VAMAS and the identifier is not there. Let
        # the VAMAS reader refuse it by name and quote what it found,
        # rather than handing the file to a reader that will describe the
        # wrong problem.
        return read_vamas(path)
    if b"SOFH" in head:
        accepted = {"work_function", "photon_energy"}
        return read_spe(path, **{k: v for k, v in options.items() if k in accepted})
    if path.suffix.lower() == ".spe" and _looks_binary(head):
        # The extension promised PHI and the content is binary and not
        # PHI. Falling through to the text reader produces an error about
        # columns that says nothing about the real problem, and guessing
        # the layout of a binary is how data get read wrong silently.
        # Plain text under a .spe extension is NOT refused: it is read,
        # because telling someone to rename their file is not an answer
        # when the program can simply open it.
        raise XPSError(f"{path.name}: " + _spe_diagnosis(head))
    accepted = {
        "photon_energy", "pass_energy", "axis", "work_function",
        "dwell_s", "sweeps", "region",
    }
    return [read_xps_text(path, **{k: v for k, v in options.items() if k in accepted})]


__all__ = [
    "SPE_EXPORT_ADVICE",
    "VAMAS_IDENTIFIER",
    "SpeRegion",
    "parse_spe_header",
    "read_spe",
    "read_vamas",
    "read_xps",
    "read_xps_text",
    "write_vamas",
]
