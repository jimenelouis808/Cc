"""Reading Quantum ESPRESSO pseudopotential (UPF) headers.

Until now this package guessed a pseudopotential's family from its filename,
and said so as a known limitation: ``C.pbe-n-kjpaw_psl.1.0.0.UPF`` looks like
PAW, but a file renamed or from an unfamiliar table could be anything. The
authoritative answer is in the file, so read it.

Two header layouts exist and both are handled:

* **UPF v2** — XML-ish, with the facts as attributes of ``<PP_HEADER ... />``:
  ``element``, ``pseudo_type``, ``relativistic``, ``functional``,
  ``z_valence``, and sometimes ``wfc_cutoff`` / ``rho_cutoff``.
* **UPF v1** — a fixed-order block of values each followed by a descriptive
  label, e.g. a line reading ``NC  Ultrasoft/Norm-Conserving/PAW``.

Only the header is read: these files run to megabytes, and everything needed
lives in the first few kilobytes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: How many bytes of the file to scan for the header.
_HEADER_BYTES = 65536

_ATTRIBUTE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')

#: Maps the many spellings of a pseudopotential type onto our four families.
_TYPE_ALIASES: dict[str, str] = {
    "nc": "NC",
    "sl": "NC",          # semilocal, still norm-conserving in practice
    "1/r": "NC",
    "us": "USPP",
    "uspp": "USPP",
    "paw": "PAW",
}


@dataclass
class PseudoInfo:
    """What a UPF file says about itself.

    Attributes
    ----------
    path
        Where the file is.
    element
        Chemical symbol it describes.
    family
        ``"NC"``, ``"USPP"``, ``"PAW"`` or ``"unknown"``.
    relativistic
        ``"scalar"``, ``"full"``, ``"nonrelativistic"`` or ``"unknown"``.
        Spin-orbit needs ``"full"``.
    functional
        Exchange-correlation the pseudopotential was generated with, e.g.
        ``"PBE"``. Mixing it with a different ``input_dft`` is legal but
        rarely intended.
    z_valence
        Valence electrons, useful for sizing the band count.
    suggested_wfc_cutoff, suggested_rho_cutoff
        Cutoffs in Ry recommended by whoever generated the file, when it
        records them. Not every table does.
    upf_version
        1 or 2.
    """

    path: Path
    element: str = ""
    family: str = "unknown"
    relativistic: str = "unknown"
    functional: str = ""
    z_valence: Optional[float] = None
    suggested_wfc_cutoff: Optional[float] = None
    suggested_rho_cutoff: Optional[float] = None
    upf_version: int = 0

    @property
    def is_fully_relativistic(self) -> bool:
        """Whether this file carries the spin-orbit information."""
        return self.relativistic.lower().startswith("full")

    @property
    def supports_raman(self) -> bool:
        """Whether QE's DFPT Raman can use it: norm-conserving only."""
        return self.family == "NC"

    def describe(self) -> str:
        """One-line summary for a listing."""
        parts = [f"{self.element or '?'}", self.family]
        if self.functional:
            parts.append(self.functional)
        if self.is_fully_relativistic:
            parts.append("rel")
        if self.suggested_wfc_cutoff:
            parts.append(f"≥{self.suggested_wfc_cutoff:g} Ry")
        return "  ".join(parts)


def _parse_v2(header: str, info: PseudoInfo) -> PseudoInfo:
    """Fill ``info`` from a UPF v2 ``<PP_HEADER .../>`` attribute block."""
    block_match = re.search(r"<PP_HEADER(.*?)/?>", header, re.DOTALL | re.IGNORECASE)
    if block_match is None:
        return info
    attributes = {
        key.lower(): value.strip()
        for key, value in _ATTRIBUTE.findall(block_match.group(1))
    }

    info.upf_version = 2
    info.element = attributes.get("element", "").strip()
    info.functional = attributes.get("functional", "").strip()
    info.relativistic = attributes.get("relativistic", "unknown").strip()

    raw_type = attributes.get("pseudo_type", "").strip().lower()
    info.family = _TYPE_ALIASES.get(raw_type, "unknown")
    # Some files leave pseudo_type vague but set explicit booleans.
    if info.family == "unknown":
        if attributes.get("is_paw", "").upper().startswith("T"):
            info.family = "PAW"
        elif attributes.get("is_ultrasoft", "").upper().startswith("T"):
            info.family = "USPP"

    for key, target in (
        ("z_valence", "z_valence"),
        ("wfc_cutoff", "suggested_wfc_cutoff"),
        ("rho_cutoff", "suggested_rho_cutoff"),
    ):
        raw = attributes.get(key)
        if raw:
            try:
                value = float(raw)
            except ValueError:
                continue
            # A recorded cutoff of zero means "not stated", not "zero Ry".
            if target != "z_valence" and value <= 0:
                continue
            setattr(info, target, value)
    return info


def _parse_v1(header: str, info: PseudoInfo) -> PseudoInfo:
    """Fill ``info`` from a UPF v1 header, whose values carry text labels."""
    block_match = re.search(
        r"<PP_HEADER>(.*?)</PP_HEADER>", header, re.DOTALL | re.IGNORECASE
    )
    if block_match is None:
        return info
    info.upf_version = 1

    for line in block_match.group(1).splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        tokens = stripped.split()
        if not tokens:
            continue
        value = tokens[0]

        if "element" in lowered:
            info.element = value
        elif "norm-conserving" in lowered or "ultrasoft" in lowered:
            info.family = _TYPE_ALIASES.get(value.lower(), "unknown")
        elif "exchange-correlation" in lowered or "functional" in lowered:
            # The functional sits between the value and the label, quoted or
            # not; take the whole prefix and tidy it.
            info.functional = stripped.split("Exchange")[0].strip().strip('"')
        elif "z valence" in lowered:
            try:
                info.z_valence = float(value)
            except ValueError:
                pass
        elif "wavefunctions" in lowered and "cutoff" in lowered:
            try:
                info.suggested_wfc_cutoff = float(value)
            except ValueError:
                pass
        elif "charge density" in lowered and "cutoff" in lowered:
            try:
                info.suggested_rho_cutoff = float(value)
            except ValueError:
                pass

    # v1 headers rarely state relativity; the filename is the only hint left.
    if "rel-" in info.path.name.lower():
        info.relativistic = "full"
    return info


def read_upf_header(path: str | Path) -> PseudoInfo:
    """Read what a UPF file says about itself.

    Parameters
    ----------
    path
        The ``.UPF`` file.

    Returns
    -------
    PseudoInfo
        Fields that could not be determined keep their ``"unknown"`` or
        ``None`` defaults — deliberately, so a caller can tell "this is PAW"
        from "I could not tell".

    Raises
    ------
    ValueError
        If the file does not exist or contains no recognisable UPF header.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"{path}: no existe o no es un archivo.")

    try:
        header = path.read_text(errors="replace")[:_HEADER_BYTES]
    except OSError as exc:
        raise ValueError(f"{path}: no se pudo leer ({exc}).") from exc

    if "PP_HEADER" not in header.upper():
        raise ValueError(
            f"{path.name}: no contiene una cabecera <PP_HEADER>. "
            "¿Es realmente un archivo UPF?"
        )

    info = PseudoInfo(path=path)
    # v2 keeps its facts as attributes; v1 as a labelled block.
    if re.search(r"<PP_HEADER[^>]*\w+\s*=", header, re.IGNORECASE):
        info = _parse_v2(header, info)
    else:
        info = _parse_v1(header, info)

    # Fall back to the filename only for the element, which is unambiguous
    # there, and never for the family, which is the thing worth being sure of.
    if not info.element:
        stem = path.name.split(".")[0]
        if stem[:2].isalpha() and stem[:1].isupper():
            info.element = stem[:2] if stem[1:2].islower() else stem[:1]
    return info
