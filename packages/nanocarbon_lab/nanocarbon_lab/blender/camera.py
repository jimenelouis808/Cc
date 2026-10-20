"""Where to put the camera, and which way to look from.

No ``bpy``, no ``scipy``: the first because this has to be testable in an
ordinary interpreter, the second because Blender's bundled Python does
not have it and this module runs *inside* Blender. Everything here is
numpy and arithmetic.

Two separate jobs, and the old code did one of them badly and the other
not at all.

**Framing.** The distance used to be computed from the subject's
bounding *sphere*. For a single cage that is right, and for a
composition it is badly wrong: a row of three structures 77 Å long and
12 Å deep has a bounding sphere of radius 39 Å, so the camera backed off
far enough to fit a 78 Å ball and the structures came out filling about a
quarter of the frame with empty space all round. Framing is now done on
the **projected bounding box** in the camera's own plane, separately for
the horizontal and the vertical, which is what actually decides when an
object touches the frame edge.

**Direction.** It used to be the fixed vector ``(1, -1.4, 0.6)``, which
is a reasonable three-quarter view of a compact object and an arbitrary
one of everything else -- it looked down the length of a row and showed
a tube end-on. :func:`best_view` instead *measures*: it projects the
atoms along each candidate direction and keeps the one whose silhouette
has the largest area, which is the direction that shows the most
structure. Ties go to the more three-quarter view, because a silhouette
seen dead-on reads as a flat drawing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = ["NAMED_VIEWS", "basis_for", "best_view", "convex_hull_area",
           "frame_camera", "named_direction", "projected_extent",
           "view_candidates"]

#: Directions a user can ask for by name, as (x, y, z) to look **from**.
#: ``auto`` is not here: it is measured, not tabulated.
NAMED_VIEWS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
    # The classic three-quarter view, and the one the pipeline used for
    # everything before it could measure.
    "iso": (1.0, -1.4, 0.6),
    "hero": (1.0, -1.2, 0.35),
}


def named_direction(name: str) -> np.ndarray:
    """A unit vector for one of :data:`NAMED_VIEWS`."""
    try:
        vector = np.asarray(NAMED_VIEWS[name], dtype=float)
    except KeyError:
        raise ValueError(
            f"unknown view {name!r}; known: {', '.join(sorted(NAMED_VIEWS))}, "
            "auto"
        ) from None
    return vector / np.linalg.norm(vector)


def view_candidates(count: int = 96) -> np.ndarray:
    """Unit vectors spread over the sphere for the automatic search.

    A Fibonacci sphere rather than a latitude-longitude grid, which
    bunches candidates at the poles and would bias the search toward
    looking straight down.

    Three deliberate restrictions, each of which is about what a figure
    wants rather than about what scores highest:

    * **Above the subject only.** A camera underneath is a legitimate
      choice and almost never the one a figure wants, and searching the
      lower hemisphere only finds mirror images of what is already here.
    * **Never from straight overhead** (``z`` stops at 0.9). A view down
      the pole flattens anything that is not a sheet.
    * **The named views are not included.** They are reachable by name;
      putting them in the search means the automatic choice can land on
      an exact elevation -- dead-on front, dead-on top -- which reads as
      a technical drawing rather than a photograph, and an exact optimum
      is not something a tie-break bonus can pull away from.
    """
    if count < 1:
        raise ValueError("view_candidates needs at least one direction.")
    golden = math.pi * (3.0 - math.sqrt(5.0))
    out = []
    for index in range(count):
        # z from just above the equator to near the pole: a view from
        # dead overhead flattens anything that is not a sheet.
        z = 0.05 + 0.85 * (index + 0.5) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * index
        out.append((radius * math.cos(theta), radius * math.sin(theta), z))
    vectors = np.asarray(out, dtype=float)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def basis_for(direction: Sequence[float],
              up: Sequence[float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """A right-handed camera basis for looking from ``direction``.

    Returns the rows ``(right, up, back)``: ``back`` is the normalised
    view direction (pointing *from* the subject *to* the camera, which is
    Blender's ``-Z`` convention seen from the other side), and the other
    two span the image plane.

    The world up is swapped for ``+Y`` when the view is within a few
    degrees of vertical, where the cross product with ``+Z`` degenerates.
    Without that a top view produces a zero-length right vector and every
    projected coordinate comes out ``nan`` -- silently, since nothing
    downstream checks.
    """
    back = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(back)
    if norm < 1e-12:
        raise ValueError("the view direction must not be zero.")
    back = back / norm
    world_up = np.asarray(up, dtype=float)
    if abs(float(np.dot(back, world_up))) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    true_up = np.cross(back, right)
    return np.stack([right, true_up, back])


def projected_extent(points: np.ndarray, direction: Sequence[float],
                     up: Sequence[float] = (0.0, 0.0, 1.0),
                     radii: Sequence[float] | None = None,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Half-extents of ``points`` in the camera frame, and the centre.

    Returns ``(half_widths, centre)`` where ``half_widths`` is
    ``(horizontal, vertical, depth)`` in the camera's own axes and
    ``centre`` is in world coordinates. ``radii`` grows each point by its
    own sphere, so framing accounts for the fact that an atom is drawn as
    a ball and not as a dot -- without it the outermost atom of a
    structure is exactly on the frame edge and half of it is cut off.
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:                        # pragma: no cover - empty
        return np.zeros(3), np.zeros(3)
    basis = basis_for(direction, up)
    local = points @ basis.T
    grow = (np.zeros(len(points)) if radii is None
            else np.asarray(radii, dtype=float))
    low = (local - grow[:, None]).min(axis=0)
    high = (local + grow[:, None]).max(axis=0)
    half = 0.5 * (high - low)
    centre_local = 0.5 * (high + low)
    return half, centre_local @ basis


def convex_hull_area(points_2d: np.ndarray) -> float:
    """Area of the convex hull of 2D points, by monotone chain.

    Written out rather than taken from scipy because Blender's bundled
    Python has no scipy and this runs inside Blender. It is a sort plus
    two linear passes, so a 10 000-atom structure over a hundred
    candidate directions is still well under a second.

    Degenerate inputs -- fewer than three points, or all of them on one
    line -- return 0.0 rather than raising, because that is the honest
    area and a view of a straight chain really does carry none.
    """
    points = np.asarray(points_2d, dtype=float)
    if len(points) < 3:
        return 0.0
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    def turn(a, b, c) -> float:
        """Twice the signed area of the triangle a-b-c.

        Written out rather than ``np.cross``, which is deprecated for
        2-vectors in NumPy 2 and emits a warning *per call* -- and this
        is called once per hull candidate per view direction, which came
        to 1.2 million warnings in one test run.
        """
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    def half(seq):
        chain: list[np.ndarray] = []
        for point in seq:
            while len(chain) >= 2:
                if turn(chain[-2], chain[-1], point) > 1e-12:
                    break
                chain.pop()
            chain.append(point)
        return chain

    hull = half(points)[:-1] + half(points[::-1])[:-1]
    if len(hull) < 3:
        return 0.0
    loop = np.asarray(hull)
    return float(abs(np.dot(loop[:, 0], np.roll(loop[:, 1], -1))
                     - np.dot(loop[:, 1], np.roll(loop[:, 0], -1))) / 2.0)


#: How much a view is rewarded for not being axis-aligned, as a fraction
#: of the silhouette area. Small on purpose: it breaks ties between views
#: that show the same amount of structure, and must never overrule a view
#: that genuinely shows more.
THREE_QUARTER_BONUS = 0.06


def best_view(points: np.ndarray, candidates: np.ndarray | None = None,
              up: Sequence[float] = (0.0, 0.0, 1.0),
              sample: int = 4000, seed: int = 0) -> np.ndarray:
    """The direction whose silhouette shows the most of the structure.

    Measured, not assumed: each candidate direction projects the atoms
    into its own image plane and the convex hull area of that projection
    is the score. Looking down the length of a row of structures gives a
    small hull and looking across it a large one, which is the whole
    content of "the best angle" for a composition.

    A small bonus goes to directions that are not axis-aligned. A view
    exactly down an axis of a symmetric object is often tied with its
    neighbours on area and reads as a flat elevation drawing, so the tie
    is broken toward the three-quarter view a figure wants.

    ``sample`` caps how many atoms are projected: the hull of a random
    few thousand is the hull of the structure to well within the
    precision this decision needs, and it keeps a 100 000-atom
    supernetwork from making the search the slow part of a render.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 3:                         # pragma: no cover - tiny
        return named_direction("iso")
    if len(points) > sample:
        rng = np.random.default_rng(seed)
        points = points[rng.choice(len(points), sample, replace=False)]
    if candidates is None:
        candidates = view_candidates()

    best_score = -np.inf
    best = candidates[0]
    for direction in candidates:
        basis = basis_for(direction, up)
        area = convex_hull_area((points @ basis.T)[:, :2])
        # 0 when the view is down an axis, 1 when it is equally inclined
        # to all three -- the usual three-quarter view.
        obliquity = float(np.min(np.abs(direction)) * math.sqrt(3.0))
        score = area * (1.0 + THREE_QUARTER_BONUS * obliquity)
        if score > best_score:
            best_score, best = score, direction
    return np.asarray(best, dtype=float)


def frame_camera(points: np.ndarray, direction: Sequence[float],
                 lens_mm: float, sensor_mm: float,
                 resolution: Sequence[int], fill: float = 0.86,
                 up: Sequence[float] = (0.0, 0.0, 1.0),
                 radii: Sequence[float] | None = None,
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """Where to stand so the subject fills the frame, and how far that is.

    Returns ``(location, target, distance)``.

    The subject is framed by its **projected bounding box**, horizontally
    and vertically against their own half-angles, rather than by a
    bounding sphere. For anything round the two agree; for a row of
    structures the sphere is the diagonal and backing off far enough to
    fit it leaves the subject a quarter the size it should be. That was
    measured on a three-item composition: sphere framing gave a subject
    covering 23 % of the frame width against the 86 % asked for.

    Blender's automatic sensor fit maps the sensor width onto the
    **longer** image axis, so the shorter one sees a narrower angle and
    is the one that clips first -- a 2000x2400 cover crops vertically
    long before it crops horizontally.

    The depth half-extent is added to the distance because the near face
    of the subject is closer to the camera than its centre, and a camera
    placed as though the whole structure sat at its centre plane frames
    the far half correctly and cuts the near half off.
    """
    half, centre = projected_extent(points, direction, up, radii)
    width, height = (int(resolution[0]), int(resolution[1]))
    long_side, short_side = max(width, height), min(width, height)
    half_long = math.atan(0.5 * sensor_mm / float(lens_mm))
    half_short = math.atan(math.tan(half_long) * short_side / long_side)
    half_h = half_long if width >= height else half_short
    half_v = half_short if width >= height else half_long

    fill = max(1e-3, float(fill))
    needed = max(half[0] / math.tan(half_h), half[1] / math.tan(half_v))
    distance = needed / fill + half[2]
    back = np.asarray(direction, dtype=float)
    back = back / np.linalg.norm(back)
    return centre + back * distance, centre, float(distance)


#: Clearance between the subject's lowest point and the floor, as a
#: fraction of the subject's radius. Small and positive: zero puts the
#: plane through the bottom atom, and a large value levitates.
FLOOR_CLEARANCE = 0.04


def floor_z(bottom: float, radius: float,
            clearance: float = FLOOR_CLEARANCE) -> float:
    """Height of the ground plane under a subject.

    ``bottom`` is the subject's **lowest point**, not its centre, and
    that is the whole content of this function. Placing the floor at
    ``centre - radius`` instead assumes the subject is a sphere, and a
    composition never is: measured on a row of three cages 39.8 A tall
    whose bounding-sphere radius is 65.0 A, the spherical formula put
    the plane **45.1 A** too low -- more than the subject's own height
    below where it belonged -- so the structures levitated out of their
    own reflection and the frame below them was a smooth gradient with
    no trace of them in it (0.0876 rising monotonically to 0.1124).

    The error scales with how far from spherical the subject is, which
    is why it hid for so long: on a single tall tube the same formula is
    only 1.1 A out and the reflection looks perfectly fine.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive.")
    return bottom - radius * clearance
