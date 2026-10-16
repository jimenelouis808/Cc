"""The CasaXPS fit annotation that travels inside a VAMAS block comment.

A VAMAS file carries a spectrum and nothing else: the standard has no
field for "this region was fitted with five components". CasaXPS solves
that the way every program eventually solves it — by writing its own
grammar into the one free-text field the format does have, the block
comment — and every other program that reads CasaXPS files reads *that*,
not the standard.

So a VAMAS this package writes without these lines opens in CasaXPS and
shows the measurement with no fit on it, which is exactly what it is: the
deconvolution lived in the fit result, and the fit result was never
written down. This module writes it down.

The grammar, which is positional and undocumented by its author, was taken
from the only independent implementation that reads it, the ``vms`` parser
of ``pynxtools-xps`` 0.6.3, and the field order below is the field order
that parser indexes::

    Casa Info Follows CasaXPS Version 2.3.22PR1.0
    0                      <- processes (calibrations, smoothing)
    0                      <- annotations, each of which costs TWO lines
    1                      <- regions
    5                      <- components
    CASA region (*C 1s*) (*Shirley*) 280.0 295.0 ...
    CASA comp (*C sp2*) (*GL(30)*) Area ... MFWHM ... Position ...

**Three things are not negotiable** and all three are reasons a file like
this gets read as empty:

* The four counts are read as *the first four lines that are nothing but
  digits*, in order. A prose comment line that happens to be a bare
  number, placed before them, is read as one of the counts.
* The reader then looks at exactly ``5 + processes + 2·annotations +
  regions + components`` lines and stops. A region line past that count
  does not exist.
* Every name is wrapped in ``(*...*)`` and the tokeniser splits on that,
  so a name may hold spaces but never the sequence ``*)``.

**What this writes is the fit, not an invitation to refit.** The values
are ours: position, measured width and area as the fitter converged on
them. The *constraints* are not written as CasaXPS constraints — the
spin-orbit partner of a doublet goes out as its own component with its own
numbers rather than as a component tied to its sibling, and a width shared
between two states goes out as two equal widths. The reason is that a
constraint encoded wrongly does not look wrong: it looks like a fit, drawn
in the right place, that moves somewhere else the moment anybody presses
*Fit* in CasaXPS. Numbers that are merely unconstrained can be re-tied by
hand in a minute; numbers tied to the wrong sibling cannot be noticed.

**And it only makes sense with the background still in the data.** The
component areas and the region definition both sit on top of a background
CasaXPS computes for itself from the region's endpoints. Over a spectrum
this package has already background-subtracted, CasaXPS subtracts a second
one. :func:`casa_comment` therefore describes the fit of a spectrum as
measured, and the exporter refuses to pair it with the subtracted export.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from ..database.loader import DatabaseError
from .elements import XPSDatabase
from .fitting import XPSFitResult, XPSFittedComponent

#: The version CasaXPS writes into the first line. It is a format
#: declaration rather than a claim about who wrote the file: the reader
#: matches the opening words, and a line without a version is a line some
#: readers skip. Who actually wrote the file is said in the prose comment
#: lines that follow the annotation.
CASA_VERSION = "2.3.22PR1.0"

#: Our background names mapped onto the ones CasaXPS knows. These are not
#: free text either: a background type CasaXPS cannot resolve leaves the
#: region with no background and every area in it wrong.
CASA_BACKGROUNDS = {
    "shirley": "Shirley",
    "linear": "Linear",
    "tougaard": "U 3 Tougaard",
}

#: Atomic masses, used by CasaXPS for one thing only: turning atomic per
#: cent into weight per cent. Quantification does not touch them. They
#: cover the elements the XPS database knows; anything else goes out as
#: zero, which is CasaXPS's own "not set" and leaves the atomic per cent
#: untouched.
CASA_MASSES = {
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Na": 22.990,
    "Al": 26.982, "Si": 28.086, "P": 30.974, "S": 32.06, "Cl": 35.45,
    "K": 39.098, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546,
    "Se": 78.971, "Mo": 95.95, "Au": 196.97, "Ag": 107.87,
}


def _name(text: str) -> str:
    """A name safe to wrap in ``(*...*)``.

    The tokeniser ends a name at the first ``*)``, so a name containing
    one swallows the rest of the line. Spaces and commas are fine and are
    what CasaXPS itself writes.
    """
    return str(text).replace("*)", ") ").replace("(*", " (").strip() or "sin nombre"


def _number(value: Optional[float], digits: int = 6) -> str:
    """One numeric field. Non-finite becomes zero, which Casa reads."""
    if value is None:
        return "0"
    number = float(value)
    if not np.isfinite(number):
        return "0"
    return f"{number:.{digits}f}"


def casa_lineshape(profile: str, extra: Sequence[float] = ()) -> str:
    """Our profile name written as a CasaXPS lineshape.

    The two GL forms need no translation at all: this package's ``gl`` and
    ``sgl`` were written as CasaXPS's ``GL(p)`` and ``SGL(p)``, with
    ``mixing`` the same quantity as ``p/100`` and the same convention that
    0 is Gaussian and 100 Lorentzian. A plain Gaussian and a plain
    Lorentzian are the two ends of that same family, so they go out as
    ``GL(0)`` and ``GL(100)`` rather than as a shape CasaXPS would have to
    guess at.

    The Doniach-Sunjic pair is the honest exception. CasaXPS's ``DS(a,n)``
    takes an asymmetry and a second parameter of its own, and the second
    parameter is not this package's Gaussian width: the two programs
    broaden the profile differently. The asymmetry carries over, the
    broadening is written as the Gaussian width in eV because that is the
    closest thing it means, and a metallic component is the one component
    worth checking by eye after the file opens.
    """
    extra = tuple(float(value) for value in extra)
    if profile == "gl":
        return f"GL({round(100 * extra[0])})"
    if profile == "sgl":
        return f"SGL({round(100 * extra[0])})"
    if profile == "gaussian":
        return "GL(0)"
    if profile == "lorentzian":
        return "GL(100)"
    if profile == "ds":
        return f"DS({extra[0]:.3f},0)"
    if profile == "ds_gauss":
        return f"DS({extra[0]:.3f},{extra[1]:.3f})"
    raise ValueError(f"no hay forma CasaXPS para el perfil {profile!r}")


def _bounds(result: XPSFitResult, component: XPSFittedComponent,
            parameter: str, value: float) -> tuple[float, float]:
    """The limits the fit actually used, or the value twice.

    ``result.bounds`` holds only the FREE parameters: a fixed or linked one
    never entered the packing. Writing its value as both limits is not a
    convenience, it is what it was — CasaXPS reads ``min == max`` as fixed,
    which is the same statement.
    """
    found = result.bounds.get(f"{component.name}.{parameter}")
    if found is None:
        return (float(value), float(value))
    return (float(found[0]), float(found[1]))


def _rsf(component: XPSFittedComponent,
         database: Optional[XPSDatabase]) -> float:
    """The sensitivity factor for a component's line, or zero."""
    if database is None or not component.element:
        return 0.0
    try:
        element = database.element(component.element)
    except DatabaseError:
        return 0.0
    # A component's ``line`` is the region it was fitted in, which for a
    # doublet is the component of the line rather than the line: "Se
    # 3d5/2" where the sensitivity factor is tabulated for "Se 3d". The
    # factor belongs to the whole doublet, and giving the same one to both
    # halves is what makes CasaXPS's own quantification right — it sums
    # area/RSF over the components, so the pair contributes its total area
    # over the line's one factor.
    label = component.line or ""
    for candidate in element.lines:
        if candidate.label == label or label.startswith(candidate.label):
            return float(candidate.rsf)
    try:
        return float(element.primary_line.rsf)
    except DatabaseError:
        return 0.0


def _region_line(result: XPSFitResult, rsf: float) -> str:
    """One ``CASA region``: the window, and the background over it.

    Eighteen fields, of which this package has four to say. The rest are
    CasaXPS's own and go out as zero on purpose: the endpoint offsets are
    percentages, and the four cross-section numbers are the Tougaard
    parameters, which are left empty so CasaXPS uses its own universal
    values — they are the same ``B = 2866``, ``C = 1643`` this package
    fits with, so writing them would restate them and writing them wrong
    would quietly change every area in the region.
    """
    low, high = min(result.window), max(result.window)
    name = _name(result.region_label or "region")
    background = CASA_BACKGROUNDS.get(result.background.kind, "Shirley")
    return (
        f"CASA region (*{name}*) (*{background}*) "
        f"{_number(low)} {_number(high)} {_number(rsf, 4)} "
        "2 0 0 0 0 0 0 "
        f"(*{name}*) 0 0 {_number(rsf, 4)}"
    )


def _component_lines(component: XPSFittedComponent,
                     database: Optional[XPSDatabase],
                     result: XPSFitResult) -> list[str]:
    """One ``CASA comp`` per peak that will be drawn — two for a doublet.

    A spin-orbit doublet is one component here, because its splitting and
    its branching ratio are fixed by the degeneracy and fitting them
    separately invents two free parameters that do not exist. CasaXPS has
    no such thing: it draws what it is given. Handing it one component at
    the 2p3/2 position with the area of both would draw one peak where the
    spectrum has two and leave the 2p1/2 as unexplained residual, so the
    pair is split here, at the end, where splitting it costs nothing — the
    areas divide by the same ratio that tied them.
    """
    shape = casa_lineshape(component.profile, component.extra)
    rsf = _rsf(component, database)
    mass = CASA_MASSES.get(component.element or "", 0.0)
    tag = _name(component.line or component.element or component.label)
    width = component.true_fwhm
    width_low, width_high = _bounds(result, component, "fwhm", component.fwhm)
    centre_low, centre_high = _bounds(result, component, "centre", component.centre)
    # The width limits are limits on the width PARAMETER and the width
    # written out is the measured one; for a GL product the two differ by
    # a few per cent. Widening the pair by that same factor keeps the
    # fitted value inside its own limits, which is the property that
    # matters: a value outside them is a fit CasaXPS redraws.
    scale = width / component.fwhm if component.fwhm > 0 else 1.0
    width_low, width_high = width_low * scale, width_high * scale

    if component.doublet is None:
        parts = [(component.label or component.name, component.centre,
                  component.area)]
    else:
        ratio = float(component.doublet.ratio)
        low_share = 1.0 / (1.0 + ratio)
        labels = component.doublet.labels
        parts = [
            (f"{component.label or component.name} {labels[0]}",
             component.centre, component.area * low_share),
            (f"{component.label or component.name} {labels[1]}",
             component.centre + float(component.doublet.splitting_ev),
             component.area * (1.0 - low_share)),
        ]

    lines = []
    for label, centre, area in parts:
        offset = centre - component.centre
        lines.append(
            f"CASA comp (*{_name(label)}*) (*{shape}*) "
            f"Area {_number(max(area, 0.0))} 0 1e+030 -1 1 "
            f"MFWHM {_number(width)} {_number(max(width_low, 0.0))} "
            f"{_number(width_high)} -1 1 "
            f"Position {_number(centre)} {_number(centre_low + offset)} "
            f"{_number(centre_high + offset)} -1 1 "
            f"RSF {_number(rsf, 4)} MASS {_number(mass, 4)} INDEX -1 "
            f"(*{tag}*) CONST (**) UNCORRECTEDRSF {_number(rsf, 4)}"
        )
    return lines


def casa_comment(results: Iterable[XPSFitResult],
                 database: Optional[XPSDatabase] = None) -> list[str]:
    """The block-comment lines describing one block's fits.

    Returns the lines in the order CasaXPS reads them, counts included, and
    an empty list when there is nothing fitted — a block with no fit should
    carry no annotation rather than an annotation that says zero, because a
    reader that finds the header and four zeros has still been told the
    file is a CasaXPS file.

    The caller puts these FIRST in the block comment. Anything of ours goes
    after, where a line of plain prose cannot be mistaken for one of the
    four counts.
    """
    results = [result for result in results if result.components]
    if not results:
        return []
    regions: list[str] = []
    components: list[str] = []
    for result in results:
        primary = max(result.components, key=lambda item: item.area)
        regions.append(_region_line(result, _rsf(primary, database)))
        for component in result.components:
            components.extend(_component_lines(component, database, result))
    return [
        f"Casa Info Follows CasaXPS Version {CASA_VERSION}",
        "0",                                   # processes
        "0",                                   # annotations
        str(len(regions)),
        str(len(components)),
        *regions,
        *components,
    ]


__all__ = [
    "CASA_BACKGROUNDS",
    "CASA_MASSES",
    "CASA_VERSION",
    "casa_comment",
    "casa_lineshape",
]
