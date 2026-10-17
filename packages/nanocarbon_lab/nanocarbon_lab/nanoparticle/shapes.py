"""Shapes a nanoparticle can be cut to, written as signed distances.

One abstraction, the same one :mod:`nanocarbon_lab.builders.implicit`
already uses: a shape is a function ``f(points) -> values`` that is
negative inside and positive outside. Everything else here is algebra on
that function -- intersect two shapes, subtract one from another, grow one
outwards -- so a shape nobody anticipated is a composition rather than a
new code path, and the two consumers of a shape need to know nothing
about which one they were handed:

* :mod:`nanocarbon_lab.nanoparticle.carve` keeps the atoms of an imported
  crystal whose ``f <= 0``. That needs only the **sign**, which is exact
  for every shape here including the polyhedra.
* :mod:`nanocarbon_lab.nanoparticle.coat` grows the shape outward by the
  van der Waals gap and meshes the result. That needs the **value** to
  read as a distance in Å, which is why ``exact_outside`` is recorded
  rather than assumed.

Why the distinction is load-bearing. A convex polyhedron's field is
``max_i (n_i . r - d_i)`` over its faces. Outside, that is the true
distance whenever the nearest point lies in the interior of a face, and
an *underestimate* near an edge or a vertex, where the nearest point is
on the edge itself. Offsetting such a field by ``g`` therefore does not
produce a bigger polyhedron with sharp edges: it produces one whose edges
are rounded with radius ``g``. For the job this module exists for that is
not an error to apologise for -- it is the answer. A graphitic sheet
cannot fold along a zero-radius crease; a real carbon-encapsulated cube
is round at the corners, which is what every micrograph of one shows. The
field is honest about the geometry it describes, and the rounding is the
physics.

**The shapes are geometric, and only some of them are crystallographic.**
A sphere, a rod or a cone is a shape in the laboratory frame: cutting a
crystal to one leaves whatever facets the cut happens to expose, which
is a real particle (ball-milled, laser-ablated, melt-quenched) and is not
a faceted one. The polyhedra in :func:`faceted` and :func:`wulff` are the
other kind: their faces are lattice planes, so they need the crystal to
construct and they live behind :mod:`nanocarbon_lab.nanoparticle.crystal`.
Both are offered because both are things people make, and which one you
have is not a matter of taste -- a Wulff polyhedron is what an
equilibrated particle relaxes to and a sphere is what a quench gives you.

Sizes are in Å throughout and are **semi-axes or half-lengths measured
from the centre**, never diameters: a shape is cut around the origin, so
the number that matters is how far it reaches. Each constructor's
docstring says which of its numbers is which, because a "20 Å rod" is
two different rods depending on the answer.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np

#: Same signature as the builders' fields: ``points`` has shape
#: ``(..., 3)`` and the return has shape ``(...)``.
Field = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Shape:
    """A region of space, as the signed distance to its boundary.

    Attributes
    ----------
    field
        ``f(points) -> values``, negative inside, zero on the surface.
    extent
        Half-width of an origin-centred cube that contains the shape, in
        Å. It is what sets the supercell the carver has to build and the
        sampling box the coater has to mesh, so it is part of the shape
        rather than something each caller re-derives. It may be
        generous; it may not be short.
    label
        Human-readable name, carried into ``atoms.info`` so a finished
        structure says what it was cut to.
    exact_outside
        Whether ``field`` is a true Euclidean distance outside the shape.
        ``True`` for the quadrics; ``False`` for the polyhedra and for
        anything built with a smooth boolean, where it underestimates
        near an edge. Only the coater cares, and it cares a lot: see the
        module docstring.
    info
        The constructor's own parameters, for the record.
    """

    field: Field
    extent: float
    label: str
    exact_outside: bool = True
    info: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.field(np.asarray(points, dtype=float))

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of the points inside (or on) the surface."""
        return self(points) <= 0.0

    def volume(self, resolution: int = 96) -> float:
        """Enclosed volume in Å^3, counted on a grid.

        Measured rather than derived, so a composed shape has one too.
        The error is one voxel over the shape's whole **surface**, which
        is about ``1 / resolution`` of the volume for a compact shape and
        a good deal worse for a thin one: a hollow sphere has two
        surfaces and twice the error, and a 5 Å-walled 30 Å cube
        measures 2 % low at ``resolution=200``. Used for reporting and
        for sanity checks, never inside a build.
        """
        axis = np.linspace(-self.extent, self.extent, int(resolution))
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        inside = int(np.count_nonzero(self.field(grid) <= 0.0))
        voxel = (2.0 * self.extent / (resolution - 1)) ** 3
        return inside * voxel

    def bounding_box(self, resolution: int = 96) -> np.ndarray:
        """Tight-ish half-widths along x, y, z, in Å.

        ``extent`` is one number for all three axes, which a plate or a
        long rod makes wasteful: a 200 Å rod of 10 Å radius would have
        the carver build a 400 Å cube. This measures each axis on the
        same grid :meth:`volume` uses and returns ``(3,)`` half-widths,
        rounded up by one voxel so the result never cuts the shape.
        """
        axis = np.linspace(-self.extent, self.extent, int(resolution))
        grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        inside = self.field(grid) <= 0.0
        if not inside.any():
            return np.zeros(3)
        voxel = 2.0 * self.extent / (resolution - 1)
        half = np.empty(3)
        for ax in range(3):
            other = tuple(a for a in range(3) if a != ax)
            occupied = np.flatnonzero(inside.any(axis=other))
            reach = max(abs(axis[occupied[0]]), abs(axis[occupied[-1]]))
            half[ax] = min(reach + voxel, self.extent)
        return half


# ----------------------------------------------------------------------
# algebra
# ----------------------------------------------------------------------
def intersection(*shapes: Shape, label: str = "") -> Shape:
    """Everything inside **all** of them (``max`` of the fields)."""
    if not shapes:
        raise ValueError("intersection needs at least one shape.")
    fields = [shape.field for shape in shapes]

    def combined(points: np.ndarray) -> np.ndarray:
        value = fields[0](points)
        for other in fields[1:]:
            value = np.maximum(value, other(points))
        return value

    return Shape(
        field=combined,
        # The intersection cannot reach further than the nearest bound.
        extent=min(shape.extent for shape in shapes),
        label=label or " & ".join(shape.label for shape in shapes),
        exact_outside=False,
        info={"operation": "intersection",
              "parts": [dict(shape.info) for shape in shapes]},
    )


def union(*shapes: Shape, label: str = "") -> Shape:
    """Everything inside **any** of them (``min`` of the fields)."""
    if not shapes:
        raise ValueError("union needs at least one shape.")
    fields = [shape.field for shape in shapes]

    def combined(points: np.ndarray) -> np.ndarray:
        value = fields[0](points)
        for other in fields[1:]:
            value = np.minimum(value, other(points))
        return value

    return Shape(
        field=combined,
        extent=max(shape.extent for shape in shapes),
        label=label or " + ".join(shape.label for shape in shapes),
        exact_outside=False,
        info={"operation": "union",
              "parts": [dict(shape.info) for shape in shapes]},
    )


def difference(shape: Shape, removed: Shape, label: str = "") -> Shape:
    """``shape`` with ``removed`` taken out of it.

    ``max(f, -g)``: inside the first and outside the second. This is how
    a hollow particle is written -- see :func:`hollow`.
    """
    outer, inner = shape.field, removed.field

    def combined(points: np.ndarray) -> np.ndarray:
        return np.maximum(outer(points), -inner(points))

    return Shape(
        field=combined,
        extent=shape.extent,
        label=label or f"{shape.label} - {removed.label}",
        exact_outside=False,
        info={"operation": "difference", "outer": dict(shape.info),
              "removed": dict(removed.info)},
    )


def smooth_union(first: Shape, second: Shape, blend: float,
                 label: str = "") -> Shape:
    """Union with the crease at the join filled in over ``blend`` Å.

    The same polynomial smooth-min the junction builder uses, and here
    for the same reason: a hard union of two rods leaves a crease of zero
    radius, and a surface meshed through a crease tiles badly whatever
    the remesher does afterwards. A spiked particle -- a star, a tetrapod
    -- needs this at every spike root.
    """
    if blend <= 0:
        return union(first, second, label=label)
    a, b = first.field, second.field

    def combined(points: np.ndarray) -> np.ndarray:
        first_value, second_value = a(points), b(points)
        h = np.clip(
            0.5 + 0.5 * (second_value - first_value) / blend, 0.0, 1.0)
        return (second_value * (1.0 - h) + first_value * h
                - blend * h * (1.0 - h))

    return Shape(
        field=combined,
        extent=max(first.extent, second.extent),
        label=label or f"{first.label} + {second.label} (blended)",
        exact_outside=False,
        info={"operation": "smooth_union", "blend": blend,
              "parts": [dict(first.info), dict(second.info)]},
    )


def offset(shape: Shape, distance: float, label: str = "") -> Shape:
    """Grow (``distance > 0``) or shrink the shape by a constant.

    Exact for a shape whose field is a true distance; for a polyhedron it
    rounds the edges with radius ``distance``, which is the module
    docstring's whole point.
    """
    inner = shape.field

    def grown(points: np.ndarray) -> np.ndarray:
        return inner(points) - distance

    return Shape(
        field=grown,
        extent=shape.extent + max(distance, 0.0),
        label=label or f"{shape.label} offset {distance:+.2f} A",
        exact_outside=shape.exact_outside,
        info={"operation": "offset", "distance": distance,
              "base": dict(shape.info)},
    )


def hollow(shape: Shape, thickness: float, label: str = "") -> Shape:
    """A shell of the given wall thickness, hollow inside.

    The cavity is the shape shrunk by ``thickness``; a wall thicker than
    the shape's own inradius leaves nothing to remove and the result is
    the solid shape again, which is reported rather than refused because
    it is the honest limit of the family.
    """
    return difference(shape, offset(shape, -abs(thickness)),
                      label=label or f"hollow {shape.label} "
                                     f"({thickness:.1f} A wall)")


def translated(shape: Shape, vector: Sequence[float],
               label: str = "") -> Shape:
    """The same shape, moved."""
    move = np.asarray(vector, dtype=float)
    inner = shape.field

    def moved(points: np.ndarray) -> np.ndarray:
        return inner(points - move)

    return Shape(
        field=moved,
        extent=shape.extent + float(np.linalg.norm(move)),
        label=label or shape.label,
        exact_outside=shape.exact_outside,
        info={"operation": "translate", "vector": move.tolist(),
              "base": dict(shape.info)},
    )


def rotated(shape: Shape, matrix: np.ndarray, label: str = "") -> Shape:
    """The same shape under a rotation.

    ``matrix`` maps shape coordinates to laboratory ones, so the field is
    evaluated at ``R^T r``. It must be orthogonal: a general linear map
    would stretch the field and break the distance units every consumer
    of a shape relies on. That is checked rather than trusted.
    """
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError(
            "rotated() needs an orthogonal matrix: a shape's field is a "
            "distance, and a non-rigid map rescales it by a different "
            "factor in every direction."
        )
    inner = shape.field

    def turned(points: np.ndarray) -> np.ndarray:
        return inner(points @ rotation)

    return Shape(
        field=turned,
        extent=shape.extent,
        label=label or shape.label,
        exact_outside=shape.exact_outside,
        info={"operation": "rotate", "base": dict(shape.info)},
    )


def aligned_to(shape: Shape, axis: Sequence[float], label: str = "") -> Shape:
    """Turn a shape built along +z so its axis points along ``axis``.

    Every elongated primitive here is written along z, because writing
    each one three times is how the three drift apart. This is the one
    place that converts.
    """
    target = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(target))
    if norm < 1e-12:
        raise ValueError("axis must be a non-zero vector.")
    target = target / norm
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(z @ target)
    if sine < 1e-12:
        rotation = np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = cross / sine
        kx = np.array([[0.0, -k[2], k[1]],
                       [k[2], 0.0, -k[0]],
                       [-k[1], k[0], 0.0]])
        rotation = np.eye(3) + sine * kx + (1.0 - cosine) * (kx @ kx)
    # `rotation` maps shape coordinates to laboratory ones (it carries +z
    # onto `axis`), which is exactly what `rotated` wants.
    return rotated(shape, rotation, label=label or shape.label)


# ----------------------------------------------------------------------
# quadric primitives: exact distances
# ----------------------------------------------------------------------
def ball(radius: float) -> Shape:
    """A sphere of the given radius. The spherical nanoparticle."""
    _positive("radius", radius)

    def field(points: np.ndarray) -> np.ndarray:
        return np.linalg.norm(points, axis=-1) - radius

    return Shape(field, extent=radius, label=f"sphere r={radius:.1f} A",
                 info={"shape": "sphere", "radius": radius})


def ellipsoid(a: float, b: float, c: float) -> Shape:
    """Semi-axes ``a``, ``b``, ``c`` along x, y, z.

    The family covers more named particles than it looks: ``a == b > c``
    is an oblate **platelet**, ``a == b < c`` a prolate **football**, and
    all three equal is a sphere. The field is the standard first-order
    correction to the implicit form, exact on the surface and a slight
    underestimate elsewhere, so ``exact_outside`` is False -- for a
    strongly eccentric ellipsoid the error reaches a few per cent of the
    offset and a coating grown on it is correspondingly uneven.
    """
    for name, value in (("a", a), ("b", b), ("c", c)):
        _positive(name, value)
    semi = np.array([a, b, c], dtype=float)

    def field(points: np.ndarray) -> np.ndarray:
        scaled = points / semi
        k1 = np.linalg.norm(scaled, axis=-1)
        k2 = np.linalg.norm(scaled / semi, axis=-1)
        return np.where(k2 > 1e-12, k1 * (k1 - 1.0) / np.maximum(k2, 1e-12),
                        k1 - 1.0)

    return Shape(field, extent=float(semi.max()),
                 label=f"ellipsoid {a:.1f}x{b:.1f}x{c:.1f} A",
                 exact_outside=False,
                 info={"shape": "ellipsoid", "a": a, "b": b, "c": c})


def cylinder(radius: float, half_length: float) -> Shape:
    """A flat-ended cylinder along z: the **nanorod** with cut ends.

    ``half_length`` is measured from the centre, so the rod is
    ``2 * half_length`` long. Exact distance everywhere outside.
    """
    _positive("radius", radius)
    _positive("half_length", half_length)

    def field(points: np.ndarray) -> np.ndarray:
        radial = np.linalg.norm(points[..., :2], axis=-1) - radius
        axial = np.abs(points[..., 2]) - half_length
        outside = np.sqrt(np.maximum(radial, 0.0) ** 2
                          + np.maximum(axial, 0.0) ** 2)
        return np.minimum(np.maximum(radial, axial), 0.0) + outside

    return Shape(field, extent=math.hypot(radius, half_length),
                 label=f"cylinder r={radius:.1f} L={2 * half_length:.1f} A",
                 info={"shape": "cylinder", "radius": radius,
                       "half_length": half_length})


def capsule(radius: float, half_length: float) -> Shape:
    """A round-ended rod along z: the **nanorod** as chemistry grows one.

    ``half_length`` is to the centre of the end cap, so the overall
    length is ``2 * (half_length + radius)``. This is the shape a
    solution-grown rod actually has, and it is also the one a graphitic
    coating can follow without a crease; a flat-ended
    :func:`cylinder` has two 90-degree rims.
    """
    _positive("radius", radius)
    if half_length < 0:
        raise ValueError("half_length must be >= 0 for a capsule.")

    def field(points: np.ndarray) -> np.ndarray:
        shifted = np.copy(np.asarray(points, dtype=float))
        shifted[..., 2] = shifted[..., 2] - np.clip(
            shifted[..., 2], -half_length, half_length)
        return np.linalg.norm(shifted, axis=-1) - radius

    return Shape(field, extent=half_length + radius,
                 label=f"capsule r={radius:.1f} "
                       f"L={2 * (half_length + radius):.1f} A",
                 info={"shape": "capsule", "radius": radius,
                       "half_length": half_length})


def box(half_x: float, half_y: float, half_z: float) -> Shape:
    """A rectangular block: cube when the three are equal, **plate**
    when one is small, **wire** when one is large.

    Exact distance outside, including near the edges -- which the
    polyhedral route below cannot claim, and is why a cube gets its own
    primitive instead of being eight half-spaces.
    """
    for name, value in (("half_x", half_x), ("half_y", half_y),
                        ("half_z", half_z)):
        _positive(name, value)
    half = np.array([half_x, half_y, half_z], dtype=float)

    def field(points: np.ndarray) -> np.ndarray:
        q = np.abs(points) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        return outside + np.minimum(np.max(q, axis=-1), 0.0)

    return Shape(field, extent=float(np.linalg.norm(half)),
                 label=f"box {2 * half_x:.1f}x{2 * half_y:.1f}"
                       f"x{2 * half_z:.1f} A",
                 info={"shape": "box", "half_widths": half.tolist()})


def cube(edge: float) -> Shape:
    """A cube of the given **edge length**, faces normal to x, y, z."""
    _positive("edge", edge)
    shape = box(edge / 2.0, edge / 2.0, edge / 2.0)
    return Shape(shape.field, extent=shape.extent,
                 label=f"cube edge={edge:.1f} A", exact_outside=True,
                 info={"shape": "cube", "edge": edge})


def plate(radius: float, half_thickness: float, sides: int = 0) -> Shape:
    """A **platelet**: a disc (``sides=0``) or an n-gon prism, flat on z.

    The thin limit of a prism rather than a separate construction, and
    it is here under its own name because a plate is what people ask
    for and because the aspect ratio is the whole point: a graphitic
    coating on a plate has to turn through 180 degrees at the rim, so
    the rim is where it will tear if the thickness is small.
    """
    if sides and sides < 3:
        raise ValueError("a polygonal plate needs at least 3 sides.")
    shape = (cylinder(radius, half_thickness) if sides == 0
             else prism(radius, half_thickness, sides))
    name = "disc" if sides == 0 else f"{sides}-gon platelet"
    return Shape(shape.field, extent=shape.extent,
                 label=f"{name} r={radius:.1f} t={2 * half_thickness:.1f} A",
                 exact_outside=shape.exact_outside,
                 info={"shape": "plate", "radius": radius,
                       "half_thickness": half_thickness, "sides": sides})


def cone(radius: float, half_height: float, tip_radius: float = 0.0) -> Shape:
    """A cone along z, apex up; ``tip_radius > 0`` truncates it.

    ``radius`` is the base radius and the cone spans ``-half_height`` to
    ``+half_height``. A truncated cone (a **frustum**) is the same
    construction with the apex cut off at ``tip_radius``, which is the
    shape a real "nanocone" has -- a mathematical apex is one atom and
    does not survive a relaxation.
    """
    _positive("radius", radius)
    _positive("half_height", half_height)
    if tip_radius < 0 or tip_radius >= radius:
        raise ValueError("tip_radius must be in [0, radius).")
    height = 2.0 * half_height

    def field(points: np.ndarray) -> np.ndarray:
        z = points[..., 2]
        radial = np.linalg.norm(points[..., :2], axis=-1)
        # Radius the solid is allowed at this height, linear from base to tip.
        fraction = np.clip((z + half_height) / height, 0.0, 1.0)
        allowed = radius + fraction * (tip_radius - radius)
        lateral = radial - allowed
        axial = np.abs(z) - half_height
        outside = np.sqrt(np.maximum(lateral, 0.0) ** 2
                          + np.maximum(axial, 0.0) ** 2)
        return np.minimum(np.maximum(lateral, axial), 0.0) + outside

    name = "cone" if tip_radius == 0 else "frustum"
    return Shape(field, extent=math.hypot(radius, half_height),
                 label=f"{name} r={radius:.1f} h={height:.1f} A",
                 exact_outside=False,
                 info={"shape": name, "radius": radius,
                       "half_height": half_height, "tip_radius": tip_radius})


def torus(major_radius: float, minor_radius: float) -> Shape:
    """A **nanoring**: a torus in the xy plane. Exact distance."""
    _positive("major_radius", major_radius)
    _positive("minor_radius", minor_radius)
    if minor_radius >= major_radius:
        raise ValueError(
            "minor_radius must be smaller than major_radius, or the hole "
            "closes and the shape is a lump, not a ring."
        )

    def field(points: np.ndarray) -> np.ndarray:
        planar = np.linalg.norm(points[..., :2], axis=-1) - major_radius
        return np.hypot(planar, points[..., 2]) - minor_radius

    return Shape(field, extent=major_radius + minor_radius,
                 label=f"torus R={major_radius:.1f} r={minor_radius:.1f} A",
                 info={"shape": "torus", "major_radius": major_radius,
                       "minor_radius": minor_radius})


# ----------------------------------------------------------------------
# polyhedra: intersections of half-spaces
# ----------------------------------------------------------------------
def half_space(normal: Sequence[float], distance: float) -> Shape:
    """``n . r <= d`` -- one face. The building block of every facet."""
    direction = np.asarray(normal, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("normal must be a non-zero vector.")
    direction = direction / norm

    def field(points: np.ndarray) -> np.ndarray:
        return points @ direction - distance

    # A half-space is unbounded, and saying so is not pedantry: `extent`
    # is what sizes the carver's supercell and the coater's grid, and a
    # finite number here would quietly clip whatever it was intersected
    # with. `intersection` takes the minimum, so a half-space costs
    # nothing once it meets something bounded -- and a half-space used on
    # its own fails loudly instead of returning a wrong slab.
    return Shape(field, extent=math.inf, label="half-space",
                 info={"shape": "half_space", "normal": direction.tolist(),
                       "distance": distance})


def polyhedron(normals: Iterable[Sequence[float]],
               distances: Iterable[float] | float,
               label: str = "polyhedron") -> Shape:
    """The convex solid ``n_i . r <= d_i`` for every face.

    Parameters
    ----------
    normals
        Face normals, pointing outward. Not required to be unit vectors.
    distances
        One distance per normal, or a single number for all of them.
        **This is the distance from the centre to the face**, not to a
        vertex: for a cube of edge ``a`` it is ``a/2``, and the
        difference is a factor of sqrt(3) on the diagonal.

    Notes
    -----
    The field is ``max_i (n_i . r - d_i)``, which gives the exact sign
    everywhere and the exact distance outside only away from the edges.
    That is recorded in ``exact_outside=False``; see the module docstring
    for why the caller needs to know.

    A face whose distance exceeds the inradius of all the others never
    touches the solid. That is not an error -- it is how a Wulff
    construction drops a facet whose surface energy is too high -- so it
    is counted in ``info["inactive_faces"]`` instead of being refused.
    """
    stack = np.asarray([np.asarray(n, dtype=float) for n in normals])
    if stack.ndim != 2 or stack.shape[1] != 3 or len(stack) == 0:
        raise ValueError("normals must be a non-empty sequence of 3-vectors.")
    lengths = np.linalg.norm(stack, axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError("a face normal is the zero vector.")
    stack = stack / lengths[:, None]
    if np.isscalar(distances):
        offsets = np.full(len(stack), float(distances))
    else:
        offsets = np.asarray(list(distances), dtype=float)
    if offsets.shape != (len(stack),):
        raise ValueError(
            f"got {len(stack)} normals and {offsets.size} distances."
        )
    if np.any(offsets <= 0):
        raise ValueError(
            "every face distance must be positive: a face at or through "
            "the centre cuts the particle in half rather than truncating it."
        )

    def field(points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=float) @ stack.T - offsets
        return np.max(values, axis=-1)

    extent = _circumradius(stack, offsets)
    inactive = _inactive_faces(stack, offsets)
    return Shape(field, extent=extent, label=label, exact_outside=False,
                 info={"shape": "polyhedron", "n_faces": len(stack),
                       "distances": offsets.tolist(),
                       "normals": stack.tolist(),
                       "inactive_faces": inactive})


def polyhedron_vertices(normals: np.ndarray, offsets: np.ndarray,
                        tolerance: float = 1e-7) -> np.ndarray:
    """Every corner of ``n_i . r <= d_i``, by enumerating face triples.

    Each vertex of a convex polyhedron is the intersection of three of
    its faces, so solving every triple and keeping the solutions that
    satisfy all the other constraints enumerates them exactly. With at
    most a few dozen faces that is thousands of 3x3 solves -- fast
    enough to do once per shape, and the alternative is a *sampled*
    bound, which is where this started and which was wrong in the
    direction that matters.

    A regular tetrahedron built from four {111} planes at 16 Å has its
    vertices at 48 Å: three times the face distance, because a regular
    tetrahedron's circumradius is three times its inradius. A sampled
    bound over 512 directions returned 44.4 Å for it -- short of the
    real solid, so the sampling box would have cut the four corners off
    and nothing downstream would have said so. Sharp vertices are
    exactly what sampling misses, and a facetted particle is nothing but
    sharp vertices.
    """
    count = len(normals)
    found: list[np.ndarray] = []
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                matrix = normals[[i, j, k]]
                determinant = float(np.linalg.det(matrix))
                if abs(determinant) < 1e-9:
                    continue            # three planes with a common line
                point = np.linalg.solve(matrix, offsets[[i, j, k]])
                if np.all(point @ normals.T - offsets <= tolerance
                          * max(1.0, float(np.abs(point).max()))):
                    found.append(point)
    if not found:
        return np.empty((0, 3))
    return np.unique(np.round(np.asarray(found), 6), axis=0)


def _circumradius(normals: np.ndarray, offsets: np.ndarray) -> float:
    """Distance from the centre to the furthest corner, in Å.

    ``inf`` when the face set does not bound a solid at all: the normals
    have to positively span space, and if some direction has a
    non-positive projection on every face then the solid runs to
    infinity that way. Returning a finite number for that case would
    produce a sampling box that silently cut a slab out of an unbounded
    region and called it a particle.
    """
    projections = _fibonacci_sphere(2048) @ normals.T
    if np.any(np.max(projections, axis=1) <= 1e-9):
        return math.inf
    vertices = polyhedron_vertices(normals, offsets)
    if len(vertices) == 0:                      # pragma: no cover - degenerate
        return math.inf
    return float(np.linalg.norm(vertices, axis=1).max())


def _inactive_faces(normals: np.ndarray, offsets: np.ndarray) -> int:
    """Count faces that do not touch the solid the others already bound.

    A face is inactive when the point closest to the centre on its own
    plane is already outside some other face. Checked at that point
    only, which is exact for a convex body: if the nearest point of the
    plane is excluded, so is everything further out on it.
    """
    count = 0
    for index, (normal, offset_value) in enumerate(
            zip(normals, offsets, strict=True)):
        point = normal * offset_value
        others = np.delete(np.arange(len(normals)), index)
        if np.any(point @ normals[others].T - offsets[others] > 1e-9):
            count += 1
    return count


def _fibonacci_sphere(count: int) -> np.ndarray:
    """``count`` roughly equidistant unit vectors. Deterministic."""
    index = np.arange(count, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * index / count)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * index
    return np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=-1)


def prism(radius: float, half_height: float, sides: int = 6) -> Shape:
    """A regular n-gon prism along z.

    ``radius`` is the **inradius** -- the centre-to-face distance -- so
    that it means the same thing as every other face distance here and
    so that a hexagonal prism of inradius ``r`` has the same waist as a
    cylinder of radius ``r``. The circumradius is
    ``radius / cos(pi/sides)``.
    """
    _positive("radius", radius)
    _positive("half_height", half_height)
    if sides < 3:
        raise ValueError("a prism needs at least 3 sides.")
    angles = 2.0 * np.pi * np.arange(sides) / sides
    normals = [(math.cos(a), math.sin(a), 0.0) for a in angles]
    normals += [(0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    distances = [radius] * sides + [half_height, half_height]
    shape = polyhedron(normals, distances)
    return Shape(shape.field,
                 extent=math.hypot(radius / math.cos(math.pi / sides),
                                   half_height),
                 label=f"{sides}-gon prism r={radius:.1f} "
                       f"h={2 * half_height:.1f} A",
                 exact_outside=False,
                 info={"shape": "prism", "radius": radius, "sides": sides,
                       "half_height": half_height})


def bipyramid(radius: float, half_height: float, sides: int = 6) -> Shape:
    """Two n-gon pyramids base to base: the **bipyramid** habit.

    ``radius`` is the equatorial inradius and ``half_height`` the
    apex-to-centre distance. Common for gold (hexagonal bipyramids),
    for ZnO and for CaCO3, and the one habit in this list whose two
    apices are genuinely sharp -- so it is the one to expect a coating
    to round off most.
    """
    _positive("radius", radius)
    _positive("half_height", half_height)
    if sides < 3:
        raise ValueError("a bipyramid needs at least 3 sides.")
    angles = 2.0 * np.pi * np.arange(sides) / sides
    # Each face contains an equatorial point at the inradius and the
    # apex. Asking the plane `n . r = d` to pass through both
    # `(radius cos, radius sin, 0)` and `(0, 0, half_height)` gives
    # `n = (half_height cos, half_height sin, +-radius)` with
    # `d = radius * half_height` -- note the two lengths swap places
    # between the normal and the point, which is the step that is easy
    # to get backwards and gives a solid of the wrong aspect ratio.
    # `polyhedron` takes a DIRECTION plus a centre-to-face distance in A,
    # so both are divided by the normal's length here.
    slant = math.hypot(half_height, radius)
    normals = []
    for angle in angles:
        for sign in (1.0, -1.0):
            normals.append((half_height * math.cos(angle) / slant,
                            half_height * math.sin(angle) / slant,
                            sign * radius / slant))
    distance = radius * half_height / slant
    shape = polyhedron(normals, [distance] * len(normals))
    return Shape(shape.field,
                 extent=max(radius / math.cos(math.pi / sides), half_height),
                 label=f"{sides}-gon bipyramid r={radius:.1f} "
                       f"h={2 * half_height:.1f} A",
                 exact_outside=False,
                 info={"shape": "bipyramid", "radius": radius, "sides": sides,
                       "half_height": half_height})


#: Face-normal directions of the cubic Miller families, as the plain
#: permutations-with-signs of the index triple. These are the *cubic*
#: families and nothing else: for a hexagonal or tetragonal crystal the
#: symmetry-equivalent set is different and has to come from the lattice
#: itself -- which is what
#: :func:`nanocarbon_lab.nanoparticle.crystal.miller_family` does. Used
#: here only by the lattice-free polyhedra below, whose geometry is
#: defined in the laboratory frame.
CUBIC_FAMILIES: dict[str, tuple[int, int, int]] = {
    "100": (1, 0, 0),
    "110": (1, 1, 0),
    "111": (1, 1, 1),
    "210": (2, 1, 0),
    "211": (2, 1, 1),
    "311": (3, 1, 1),
    "331": (3, 3, 1),
}


def cubic_family_normals(indices: Sequence[int]) -> np.ndarray:
    """Every distinct permutation-with-signs of an index triple.

    For a cubic lattice this is the symmetry-equivalent set: {100} gives
    6 directions, {110} gives 12, {111} gives 8, {211} gives 24. For any
    other lattice it is **not**, and the caller must not pretend
    otherwise.
    """
    h, k, ell = (int(round(value)) for value in indices)
    seen: set[tuple[int, int, int]] = set()
    base = (h, k, ell)
    for permutation in _permutations_of_three(base):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    candidate = (permutation[0] * sx, permutation[1] * sy,
                                 permutation[2] * sz)
                    if candidate != (0, 0, 0):
                        seen.add(candidate)
    return np.asarray(sorted(seen), dtype=float)


def _permutations_of_three(triple: tuple[int, int, int]
                           ) -> list[tuple[int, int, int]]:
    a, b, c = triple
    return [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]


def octahedron(face_distance: float) -> Shape:
    """Eight {111} faces. The habit of fcc metals grown slowly, of
    magnetite and of Cu2O; ``face_distance`` is centre to face."""
    return _cubic_solid({"111": face_distance}, "octahedron")


def cuboctahedron(face_distance: float, truncation: float = 1.0) -> Shape:
    """{111} plus {100}, the equilibrium habit of many fcc metals.

    ``truncation`` scales the {100} distance relative to the {111} one.
    At ``1.0`` the solid is the Archimedean cuboctahedron; raising it
    toward sqrt(3) retires the cube faces and leaves the octahedron,
    lowering it toward 1/sqrt(3) leaves the cube. The two named solids
    are therefore the ends of one parameter rather than three separate
    constructions, which is also how a real particle moves between them
    as the growth conditions change.
    """
    _positive("face_distance", face_distance)
    _positive("truncation", truncation)
    return _cubic_solid(
        {"111": face_distance, "100": face_distance * truncation},
        f"cuboctahedron (trunc {truncation:.2f})",
    )


def truncated_octahedron(face_distance: float) -> Shape:
    """The Kelvin solid: {111} hexagons and {100} squares, all edges equal.

    ``face_distance`` is the {111} distance; the {100} one follows from
    the geometry rather than from a remembered constant. Take the
    octahedron with vertices at ``(+-a, 0, 0)`` and permutations: its
    {111} planes sit at ``a / sqrt(3)``, and cutting each vertex off a
    third of the way along every edge -- the truncation that makes the
    hexagons regular -- puts the {100} planes at ``2a / 3``. The ratio is
    therefore ``(2a/3) / (a/sqrt(3)) = 2 / sqrt(3) = 1.1547``.

    It is the Wigner-Seitz cell of the bcc lattice and the shape that
    tiles space with the least surface per volume of any known tiling,
    which is why Kelvin proposed it for foams and why it is the habit
    annealed fcc particles tend to.
    """
    _positive("face_distance", face_distance)
    ratio = 2.0 / math.sqrt(3.0)
    return _cubic_solid(
        {"111": face_distance, "100": face_distance * ratio},
        "truncated octahedron (Kelvin)",
    )


def tetrahedron(face_distance: float) -> Shape:
    """Four of the eight {111} faces: the tetrahedral habit.

    Half of the {111} family -- the four whose index product is
    positive -- which is the alternation that makes a tetrahedron rather
    than an octahedron.
    """
    _positive("face_distance", face_distance)
    normals = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    shape = polyhedron(normals, [face_distance] * 4, label="tetrahedron")
    return Shape(shape.field, extent=shape.extent,
                 label=f"tetrahedron d={face_distance:.1f} A",
                 exact_outside=False,
                 info={"shape": "tetrahedron",
                       "face_distance": face_distance})


def rhombic_dodecahedron(face_distance: float) -> Shape:
    """Twelve {110} faces. The habit of garnet, of many MOFs and of
    Cu2O grown in the other regime from its octahedra."""
    return _cubic_solid({"110": face_distance}, "rhombic dodecahedron")


def _cubic_solid(families: Mapping[str, float], name: str) -> Shape:
    """A convex solid from cubic Miller families at given distances."""
    normals: list[Sequence[float]] = []
    distances: list[float] = []
    for key, distance in families.items():
        _positive(f"d({key})", distance)
        indices = CUBIC_FAMILIES.get(key)
        if indices is None:
            raise ValueError(
                f"unknown cubic family {key!r}; known: "
                f"{sorted(CUBIC_FAMILIES)}"
            )
        for normal in cubic_family_normals(indices):
            normals.append(normal)
            distances.append(distance)
    shape = polyhedron(normals, distances, label=name)
    return Shape(shape.field, extent=shape.extent, label=name,
                 exact_outside=False,
                 info={"shape": name, "families": dict(families),
                       "inactive_faces": shape.info["inactive_faces"]})


def wulff(energies: Mapping[str, float], scale: float,
          normals_for: Callable[[str], np.ndarray] | None = None) -> Shape:
    """The Wulff construction: face distance proportional to surface energy.

    Parameters
    ----------
    energies
        Surface energy per family, in any consistent unit (J/m^2, eV/A^2
        -- only the ratios matter). A family whose energy is high enough
        relative to the others simply does not appear on the particle,
        which is the construction doing its job and is counted in
        ``info["inactive_faces"]``.
    scale
        Multiplies every energy to give the distances in Å, so it sets
        the particle's size. The smallest face distance is
        ``scale * min(energies)``.
    normals_for
        Maps a family key to its face normals. ``None`` uses the **cubic**
        families, which is right for an fcc or bcc metal and wrong for
        anything else; pass
        :func:`nanocarbon_lab.nanoparticle.crystal.miller_family` bound to
        the imported crystal to get the lattice's own set.

    Notes
    -----
    Wulff's theorem is about the *equilibrium* shape -- the one that
    minimises total surface energy at fixed volume -- so this is the
    habit of a particle that was allowed to equilibrate, typically by
    annealing. A particle grown fast is kinetically shaped and is
    usually a different, more extreme member of the same family; that is
    a statement about the synthesis, not about the construction.

    The energies are the user's to supply. This function does not ship a
    table of them: surface energies depend on the facet's termination,
    on adsorbates and on the method they were computed with, and a
    builder that quietly supplied its own would be making the most
    important decision in the construction on the user's behalf.
    """
    if not energies:
        raise ValueError("wulff() needs at least one surface energy.")
    _positive("scale", scale)
    chooser = normals_for or (
        lambda key: cubic_family_normals(
            CUBIC_FAMILIES[key] if key in CUBIC_FAMILIES
            else _parse_indices(key)))
    collected: list[Sequence[float]] = []
    distances: list[float] = []
    for key, energy in energies.items():
        if energy <= 0:
            raise ValueError(
                f"surface energy for {key!r} must be positive; got {energy}."
            )
        for normal in chooser(key):
            collected.append(normal)
            distances.append(float(scale * energy))
    shape = polyhedron(collected, distances, label="Wulff polyhedron")
    return Shape(shape.field, extent=shape.extent,
                 label="Wulff polyhedron", exact_outside=False,
                 info={"shape": "wulff", "energies": dict(energies),
                       "scale": scale, "n_faces": shape.info["n_faces"],
                       "inactive_faces": shape.info["inactive_faces"]})


def _parse_indices(key: str) -> tuple[int, int, int]:
    """``"111"`` or ``"2 1 0"`` or ``"(1,1,-1)"`` -> an index triple."""
    cleaned = key.replace("(", " ").replace(")", " ").replace(",", " ")
    cleaned = cleaned.replace("{", " ").replace("}", " ").strip()
    parts = cleaned.split()
    if len(parts) == 1 and all(c in "-0123456789" for c in parts[0]):
        text = parts[0]
        if "-" in text:
            raise ValueError(
                f"cannot read {key!r}: write indices with a minus sign as "
                "separate numbers, e.g. '1 1 -1'."
            )
        if len(text) != 3:
            raise ValueError(f"cannot read Miller indices from {key!r}.")
        parts = list(text)
    if len(parts) != 3:
        raise ValueError(f"cannot read Miller indices from {key!r}.")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


# ----------------------------------------------------------------------
# five-fold habits, which are not crystal planes at all
# ----------------------------------------------------------------------
def decahedron(radius: float) -> Shape:
    """The **decahedral** habit: a pentagonal bipyramid.

    Ten {111}-like faces arranged with a five-fold axis. There is no
    five-fold axis in any crystal lattice, and that is the point: a real
    decahedral particle is five tetrahedral crystals twinned about a
    common edge, with a 7.35-degree angular gap shared out as strain.
    **Carving one out of a single crystal gives the right outline and
    the wrong interior** -- the twin boundaries are not there, so what
    comes out is a single crystal cut to a decahedral envelope. For a
    shell, a coating or a sizing study that is exactly what is wanted;
    for anything that depends on the twins it is not the structure, and
    this builder cannot make that one.

    ``radius`` is the centre-to-face distance of the ten faces.
    """
    _positive("radius", radius)
    normals = []
    # Two pyramids of five faces each, meeting at the equator. The
    # 54.7-degree tilt is the one that makes the faces {111}-like in the
    # twinned-tetrahedron picture.
    tilt = math.radians(54.735610317245346)
    for index in range(5):
        angle = 2.0 * math.pi * index / 5.0
        for sign in (1.0, -1.0):
            normals.append((math.cos(angle) * math.cos(tilt),
                            math.sin(angle) * math.cos(tilt),
                            sign * math.sin(tilt)))
    shape = polyhedron(normals, [radius] * len(normals))
    return Shape(shape.field, extent=shape.extent,
                 label=f"decahedron d={radius:.1f} A", exact_outside=False,
                 info={"shape": "decahedron", "face_distance": radius,
                       "twinned": False,
                       "note": "single-crystal cut to a decahedral envelope; "
                               "the five twin boundaries are not present"})


def icosahedron(radius: float) -> Shape:
    """The **icosahedral** (Mackay) habit: twenty faces, five-fold axes.

    The same caveat as :func:`decahedron`, more so: a Mackay icosahedron
    is twenty twinned tetrahedra. What this cuts is the envelope.
    ``radius`` is centre to face.
    """
    _positive("radius", radius)
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    # The 12 vertices of an icosahedron are the face normals of its dual,
    # the dodecahedron; the 20 face normals we want are the dodecahedron's
    # vertices, i.e. the cube's 8 corners plus the 12 golden-ratio points.
    directions = [(s1, s2, s3) for s1 in (1, -1) for s2 in (1, -1)
                  for s3 in (1, -1)]
    for s1 in (1, -1):
        for s2 in (1, -1):
            directions.append((0.0, s1 / golden, s2 * golden))
            directions.append((s1 / golden, s2 * golden, 0.0))
            directions.append((s1 * golden, 0.0, s2 / golden))
    shape = polyhedron(directions, [radius] * len(directions))
    return Shape(shape.field, extent=shape.extent,
                 label=f"icosahedron d={radius:.1f} A", exact_outside=False,
                 info={"shape": "icosahedron", "face_distance": radius,
                       "twinned": False,
                       "note": "single-crystal cut to an icosahedral "
                               "envelope; the twin boundaries of a Mackay "
                               "icosahedron are not present"})


# ----------------------------------------------------------------------
# the exotic end: shapes that are compositions
# ----------------------------------------------------------------------
def star(core_radius: float, spike_length: float, spike_radius: float,
         directions: str | np.ndarray = "111", blend: float = 2.0) -> Shape:
    """A **nanostar**: a core with spikes, the urchin's small sibling.

    Parameters
    ----------
    core_radius
        Radius of the central ball, Å.
    spike_length
        How far each spike reaches beyond the core's surface, Å.
    spike_radius
        Spike radius at its base, Å. The spikes are capsules, so they are
        rounded at the tip; a conical spike is
        :func:`union` of a core with :func:`cone` shapes if that is what
        is wanted.
    directions
        A cubic family key (``"100"``, ``"111"``, ...) or an ``(n, 3)``
        array of directions. Six spikes along {100} is the classic
        concave-cube star; eight along {111} the branched one.
    blend
        Smooth-union radius at the spike roots, Å. **Not optional in
        practice**: a hard union leaves a crease of zero radius at every
        root, and a coating meshed through one tears there. Pass ``0`` to
        get the hard union anyway.
    """
    _positive("core_radius", core_radius)
    _positive("spike_length", spike_length)
    _positive("spike_radius", spike_radius)
    axes = _direction_set(directions)
    shape = ball(core_radius)
    reach = core_radius + spike_length
    if reach <= spike_radius:
        raise ValueError(
            f"a {spike_radius:.1f} A spike radius is wider than the "
            f"{reach:.1f} A it has to reach: there is no spike, only a "
            "bulge on the core."
        )
    for axis in axes:
        # Each spike is ONE-SIDED. A capsule centred on the origin would
        # reach out along -axis as well, and since a cubic family contains
        # both of every direction every spike would be built twice --
        # harmless for {111}, and silently eight spikes instead of four
        # for a direction set the caller wrote out by hand.
        half = 0.5 * (reach - spike_radius)
        unit = np.asarray(axis, dtype=float)
        unit = unit / np.linalg.norm(unit)
        spike = translated(aligned_to(capsule(spike_radius, half), unit),
                           unit * half)
        shape = smooth_union(shape, spike, blend)
    return Shape(shape.field, extent=reach + spike_radius,
                 label=f"star core={core_radius:.1f} "
                       f"spikes={len(axes)}x{spike_length:.1f} A",
                 exact_outside=False,
                 info={"shape": "star", "core_radius": core_radius,
                       "spike_length": spike_length,
                       "spike_radius": spike_radius,
                       "n_spikes": len(axes), "blend": blend})


def tetrapod(arm_length: float, arm_radius: float,
             blend: float = 2.0) -> Shape:
    """Four arms along the tetrahedral directions: the **tetrapod**.

    The habit of ZnO, CdSe and CdTe grown with a zinc-blende core and
    wurtzite arms. As with the five-fold habits, carving one from a
    single crystal reproduces the shape and not the polytype change at
    the arm roots -- which in a real tetrapod is the reason the arms grow
    at all.
    """
    _positive("arm_length", arm_length)
    _positive("arm_radius", arm_radius)
    axes = np.asarray([(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
                      dtype=float)
    shape = ball(arm_radius)
    for axis in axes:
        arm = aligned_to(capsule(arm_radius, arm_length / 2.0), axis)
        shape = smooth_union(shape, translated(arm, axis / np.linalg.norm(axis)
                                               * arm_length / 2.0), blend)
    return Shape(shape.field, extent=arm_length + arm_radius,
                 label=f"tetrapod arms=4x{arm_length:.1f} A",
                 exact_outside=False,
                 info={"shape": "tetrapod", "arm_length": arm_length,
                       "arm_radius": arm_radius, "blend": blend,
                       "note": "single-crystal cut; the wurtzite arms of a "
                               "real tetrapod are a polytype change this "
                               "does not reproduce"})


def dumbbell(lobe_radius: float, separation: float,
             neck_radius: float | None = None, blend: float = 2.0) -> Shape:
    """Two lobes joined by a neck: the **dumbbell** or peanut.

    ``separation`` is centre to centre. ``neck_radius`` defaults to half
    the lobe radius; a neck as wide as the lobes gives a capsule and a
    neck of zero gives two separate particles, which is refused because
    a disconnected "nanoparticle" is two of them.
    """
    _positive("lobe_radius", lobe_radius)
    _positive("separation", separation)
    neck = lobe_radius / 2.0 if neck_radius is None else float(neck_radius)
    _positive("neck_radius", neck)
    if separation >= 2.0 * lobe_radius + 2.0 * neck:
        raise ValueError(
            f"separation {separation:.1f} A is too large for lobes of "
            f"{lobe_radius:.1f} A joined by a {neck:.1f} A neck: the two "
            "lobes would not touch, and two disconnected lumps are two "
            "particles rather than a dumbbell."
        )
    half = separation / 2.0
    shape = smooth_union(
        translated(ball(lobe_radius), (0.0, 0.0, half)),
        translated(ball(lobe_radius), (0.0, 0.0, -half)), blend)
    shape = smooth_union(shape, cylinder(neck, half), blend)
    return Shape(shape.field, extent=half + lobe_radius,
                 label=f"dumbbell r={lobe_radius:.1f} d={separation:.1f} A",
                 exact_outside=False,
                 info={"shape": "dumbbell", "lobe_radius": lobe_radius,
                       "separation": separation, "neck_radius": neck,
                       "blend": blend})


def concave_cube(edge: float, bite: float) -> Shape:
    """A cube with a sphere bitten out of each face: the **concave cube**.

    A real habit (platinum, palladium, gold grown with face-blocking
    surfactants) and a useful stress test for a coating: the six
    concavities have negative Gaussian curvature, so a graphitic shell
    over one needs heptagons there, exactly as a schwarzite does.

    ``bite`` is how deep each concavity cuts, measured from the face.
    """
    _positive("edge", edge)
    _positive("bite", bite)
    half = edge / 2.0
    if bite >= half:
        raise ValueError(
            f"a {bite:.1f} A bite goes through a cube of edge {edge:.1f} A."
        )
    # A sphere of radius R centred at distance (half + R - bite) from the
    # centre cuts `bite` deep into the face it faces.
    sphere_radius = edge * 0.75
    centre_distance = half + sphere_radius - bite
    shape = cube(edge)
    for axis in range(3):
        for sign in (1.0, -1.0):
            offset_vector = np.zeros(3)
            offset_vector[axis] = sign * centre_distance
            shape = difference(
                shape, translated(ball(sphere_radius), offset_vector))
    return Shape(shape.field, extent=half * math.sqrt(3.0),
                 label=f"concave cube edge={edge:.1f} bite={bite:.1f} A",
                 exact_outside=False,
                 info={"shape": "concave_cube", "edge": edge, "bite": bite})


def _direction_set(directions: str | np.ndarray) -> np.ndarray:
    if isinstance(directions, str):
        indices = CUBIC_FAMILIES.get(directions) or _parse_indices(directions)
        return cubic_family_normals(indices)
    stack = np.asarray(directions, dtype=float).reshape(-1, 3)
    if len(stack) == 0:
        raise ValueError("directions is empty.")
    return stack


def _positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive, finite length; "
                         f"got {value!r}.")


# ----------------------------------------------------------------------
# the catalogue
# ----------------------------------------------------------------------
#: Every shape the CLI and the GUI offer, as ``name -> (constructor,
#: parameter names with defaults)``. The defaults are a *size* that
#: builds in seconds rather than a recommendation: a nanoparticle's size
#: is the one number nobody else can choose for you.
#:
#: Grouped by what the shape is, because the groups mean different
#: things: the quadrics are laboratory-frame cuts, the polyhedra are
#: facetted habits, the five-fold habits are envelopes of twinned
#: particles, and the compositions are unions and differences of the
#: rest.
SHAPES: dict[str, dict[str, Any]] = {
    # -- round cuts --------------------------------------------------
    "sphere": {"build": ball, "group": "round",
               "params": {"radius": 15.0}},
    "ellipsoid": {"build": ellipsoid, "group": "round",
                  "params": {"a": 20.0, "b": 15.0, "c": 10.0}},
    "rod": {"build": capsule, "group": "round",
            "params": {"radius": 10.0, "half_length": 20.0}},
    "cylinder": {"build": cylinder, "group": "round",
                 "params": {"radius": 10.0, "half_length": 20.0}},
    "disc": {"build": lambda radius=18.0, half_thickness=5.0:
             plate(radius, half_thickness, sides=0),
             "group": "round", "params": {"radius": 18.0,
                                          "half_thickness": 5.0}},
    "cone": {"build": cone, "group": "round",
             "params": {"radius": 14.0, "half_height": 16.0,
                        "tip_radius": 2.0}},
    "torus": {"build": torus, "group": "round",
              "params": {"major_radius": 18.0, "minor_radius": 6.0}},
    # -- blocks and prisms -------------------------------------------
    "cube": {"build": cube, "group": "block", "params": {"edge": 24.0}},
    "box": {"build": box, "group": "block",
            "params": {"half_x": 16.0, "half_y": 12.0, "half_z": 8.0}},
    "prism": {"build": prism, "group": "block",
              "params": {"radius": 14.0, "half_height": 16.0, "sides": 6}},
    "platelet": {"build": lambda radius=18.0, half_thickness=5.0, sides=6:
                 plate(radius, half_thickness, sides=sides),
                 "group": "block",
                 "params": {"radius": 18.0, "half_thickness": 5.0,
                            "sides": 6}},
    "bipyramid": {"build": bipyramid, "group": "block",
                  "params": {"radius": 14.0, "half_height": 20.0,
                             "sides": 6}},
    # -- facetted habits ---------------------------------------------
    "octahedron": {"build": octahedron, "group": "facetted",
                   "params": {"face_distance": 14.0}},
    "cuboctahedron": {"build": cuboctahedron, "group": "facetted",
                      "params": {"face_distance": 14.0, "truncation": 1.0}},
    "truncated octahedron": {"build": truncated_octahedron,
                             "group": "facetted",
                             "params": {"face_distance": 14.0}},
    "tetrahedron": {"build": tetrahedron, "group": "facetted",
                    "params": {"face_distance": 16.0}},
    "rhombic dodecahedron": {"build": rhombic_dodecahedron,
                             "group": "facetted",
                             "params": {"face_distance": 14.0}},
    # -- five-fold envelopes -----------------------------------------
    "decahedron": {"build": decahedron, "group": "five-fold",
                   "params": {"radius": 14.0}},
    "icosahedron": {"build": icosahedron, "group": "five-fold",
                    "params": {"radius": 14.0}},
    # -- compositions ------------------------------------------------
    "star": {"build": star, "group": "composite",
             "params": {"core_radius": 10.0, "spike_length": 10.0,
                        "spike_radius": 4.0, "directions": "111",
                        "blend": 2.0}},
    "tetrapod": {"build": tetrapod, "group": "composite",
                 "params": {"arm_length": 20.0, "arm_radius": 6.0,
                            "blend": 2.0}},
    "dumbbell": {"build": dumbbell, "group": "composite",
                 "params": {"lobe_radius": 10.0, "separation": 16.0,
                            "neck_radius": 5.0, "blend": 2.0}},
    "concave cube": {"build": concave_cube, "group": "composite",
                     "params": {"edge": 26.0, "bite": 5.0}},
}

#: Shape names grouped, for a window that wants to present them in
#: sections rather than as one list of twenty-three.
SHAPE_GROUPS: dict[str, tuple[str, ...]] = {
    group: tuple(name for name, spec in SHAPES.items()
                 if spec["group"] == group)
    for group in ("round", "block", "facetted", "five-fold", "composite")
}


def make_shape(name: str, **params: Any) -> Shape:
    """Build a catalogue shape by name, with its defaults filled in.

    The one entry point the CLI and the GUI use, so a shape added to
    :data:`SHAPES` appears in both without either being edited.
    """
    try:
        spec = SHAPES[name]
    except KeyError:
        raise ValueError(
            f"Unknown shape {name!r}; the catalogue has: "
            f"{', '.join(sorted(SHAPES))}."
        ) from None
    arguments = dict(spec["params"])
    unknown = set(params) - set(arguments)
    if unknown:
        raise ValueError(
            f"shape {name!r} takes {sorted(arguments)}; got unexpected "
            f"{sorted(unknown)}."
        )
    arguments.update(params)
    shape = spec["build"](**arguments)
    return Shape(shape.field, extent=shape.extent, label=shape.label,
                 exact_outside=shape.exact_outside,
                 info={**dict(shape.info), "catalogue_name": name,
                       "requested": arguments})


def hollowed(name: str, thickness: float, **params: Any) -> Shape:
    """A catalogue shape as a hollow shell: the **nanocage** family.

    Every habit has a hollow version -- a cage, a box, a rattle -- and
    they are made the same way (galvanic replacement, Kirkendall
    hollowing), so this is one function rather than a dozen catalogue
    entries. ``thickness`` is the wall, in Å.
    """
    return hollow(make_shape(name, **params), thickness)


__all__ = [
    "CUBIC_FAMILIES",
    "Field",
    "SHAPES",
    "SHAPE_GROUPS",
    "Shape",
    "aligned_to",
    "ball",
    "bipyramid",
    "box",
    "capsule",
    "concave_cube",
    "cone",
    "cube",
    "cuboctahedron",
    "cubic_family_normals",
    "cylinder",
    "decahedron",
    "difference",
    "dumbbell",
    "ellipsoid",
    "half_space",
    "hollow",
    "hollowed",
    "icosahedron",
    "intersection",
    "make_shape",
    "octahedron",
    "offset",
    "plate",
    "polyhedron",
    "polyhedron_vertices",
    "prism",
    "rhombic_dodecahedron",
    "rotated",
    "smooth_union",
    "star",
    "tetrahedron",
    "tetrapod",
    "torus",
    "translated",
    "truncated_octahedron",
    "union",
    "wulff",
]
