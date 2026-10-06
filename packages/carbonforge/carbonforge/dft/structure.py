"""Qué hay que ver de una estructura antes de lanzarle un cálculo.

Tres preguntas, y ninguna es decorativa:

**¿Cómo es la caja?** Qué ejes se repiten, cuánto vacío queda por lado en
los que no, y si ese vacío basta para el modo elegido. En ondas planas
Hartree se resuelve con condiciones periódicas aunque el eje no lo sea,
así que un vacío que sobra en LCAO se queda corto en PW.

**¿Qué átomo es cuál?** El índice, el símbolo, dónde está, con quién se
enlaza y en qué anillos entra. Sin eso no se puede decir "la LDOS sobre
estos tres" ni "el hueco de core sobre aquél".

**¿Qué entornos distintos hay?** Es la pregunta que de verdad decide un
XPS. Calcular el nivel de core de los 204 carbonos de un coil es tirar
tiempo: lo que se mide son los corrimientos entre entornos, y los átomos
equivalentes dan el mismo número. Así que se agrupan por entorno químico
--especie, coordinación, vecinos y anillos-- y se ofrece un representante
por grupo. Un nitrógeno piridínico, uno pirrólico y uno grafítico salen
como tres grupos, que son exactamente los tres cálculos que hay que
hacer.

Sin Tkinter y sin GPAW: esto se puede probar entero. El widget que lo
pinte irá encima y será delgado.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

import networkx as nx
from ase import Atoms

from ..topology.graph import build_bond_graph

__all__ = [
    "AtomRow",
    "CellReport",
    "Environment",
    "StructureView",
    "describe",
    "representatives",
    "select",
]

#: Vacío por lado que pide cada modo, en Å. LCAO y FD resuelven Hartree con
#: condiciones de contorno nulas y se conforman con menos; PW lo resuelve
#: periódico aunque el eje no lo sea, y ahí las imágenes de un grupo polar
#: se hablan por sus dipolos.
VACUUM_PER_SIDE: dict[str, float] = {"lcao": 5.0, "fd": 5.0, "pw": 8.0}

#: Anillos que se buscan al clasificar entornos.
MAX_RING = 8


@dataclass(frozen=True)
class CellReport:
    """La caja, y si sirve para lo que se le va a pedir."""

    lengths: tuple[float, float, float]
    angles: tuple[float, float, float]
    volume: float
    pbc: tuple[bool, bool, bool]
    #: Vacío por LADO en cada eje (la mitad del hueco total), Å.
    vacuum_per_side: tuple[float, float, float]
    #: Extensión de los átomos en cada eje, Å.
    span: tuple[float, float, float]
    dimensionality: int

    @property
    def periodic_axes(self) -> tuple[int, ...]:
        return tuple(i for i, p in enumerate(self.pbc) if p)

    def image_separation(self, axis: int) -> float:
        """Cuánto hay entre un átomo y su imagen más cercana en ``axis``."""
        return float(self.lengths[axis] - self.span[axis])

    def vacuum_verdict(self, mode: str) -> list[str]:
        """Qué ejes no tienen bastante vacío para ``mode``."""
        needed = VACUUM_PER_SIDE.get(mode, 5.0)
        out = []
        for axis, periodic in enumerate(self.pbc):
            if periodic:
                continue
            have = self.vacuum_per_side[axis]
            if have < needed:
                out.append(
                    f"Eje {'xyz'[axis]}: {have:.1f} Å de vacío por lado, y "
                    f"{mode} pide {needed:.0f}. "
                    + ("En ondas planas las imágenes se hablan por sus dipolos."
                       if mode == "pw" else
                       "La densidad llega al borde de la caja."))
        return out


@dataclass(frozen=True)
class AtomRow:
    """Una fila de la tabla de átomos: lo que hace falta para elegirlo."""

    index: int
    symbol: str
    position: tuple[float, float, float]
    scaled: Optional[tuple[float, float, float]]
    coordination: int
    neighbours: tuple[int, ...]
    neighbour_symbols: tuple[str, ...]
    rings: tuple[int, ...]
    environment: str


@dataclass(frozen=True)
class Environment:
    """Un entorno químico y los átomos que lo comparten."""

    key: str
    label: str
    members: tuple[int, ...]

    @property
    def representative(self) -> int:
        """Uno cualquiera: por definición los demás dan el mismo número."""
        return self.members[0]

    @property
    def count(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class StructureView:
    """Todo lo que el panel de estructura necesita saber."""

    cell: CellReport
    rows: tuple[AtomRow, ...]
    environments: tuple[Environment, ...]
    formula: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def row(self, index: int) -> AtomRow:
        return self.rows[index]

    def environment_of(self, index: int) -> Environment:
        for environment in self.environments:
            if index in environment.members:
                return environment
        raise KeyError(f"El átomo {index} no está en ningún entorno.")


# ------------------------------------------------------------------ entornos


def _ring_membership(graph: nx.Graph, n: int) -> dict[int, tuple[int, ...]]:
    """Tamaños de anillo a los que pertenece cada átomo.

    La base de ciclos de networkx no es única en un sólido desordenado, así
    que esto es indicativo y no una lista de caras. Para lo que se usa aquí
    --distinguir un carbono de pentágono de uno de hexágono, un nitrógeno
    piridínico de uno pirrólico-- es suficiente y es lo que se puede
    sostener.
    """
    sizes: dict[int, set[int]] = {i: set() for i in range(n)}
    for cycle in nx.cycle_basis(graph):
        if 3 <= len(cycle) <= MAX_RING:
            for atom in cycle:
                sizes[atom].add(len(cycle))
    return {i: tuple(sorted(s)) for i, s in sizes.items()}


def _label(symbol: str, neighbours: tuple[str, ...],
           rings: tuple[int, ...]) -> str:
    """Nombre químico del entorno, con los que importan dichos por su nombre.

    Los casos con nombre propio son los que el usuario va a dopar y
    funcionalizar; el resto se describe por coordinación y anillo, que es
    todo lo que se puede afirmar.
    """
    counts = Counter(neighbours)
    carbons, hydrogens = counts.get("C", 0), counts.get("H", 0)
    degree = len(neighbours)
    ring = f"anillo {'/'.join(str(r) for r in rings)}" if rings else "sin anillo"

    if symbol == "N":
        if degree == 2 and carbons == 2 and 6 in rings:
            return "N piridínico"
        if degree == 3 and carbons == 2 and hydrogens == 1 and 5 in rings:
            return "N pirrólico"
        if degree == 3 and carbons == 3:
            return "N grafítico (sustitucional)"
        if degree == 3 and hydrogens == 2:
            return "N de amina (-NH2)"
        if degree == 1:
            return "N nitrilo o terminal"
        return f"N {degree}-coordinado, {ring}"

    if symbol == "O":
        if degree == 1:
            return "O carbonilo (C=O)"
        if degree == 2 and hydrogens == 1:
            return "O hidroxilo (-OH)"
        if degree == 2 and carbons == 2:
            return "O éter"
        return f"O {degree}-coordinado, {ring}"

    if symbol == "C":
        if degree == 4:
            return "C sp3"
        if degree == 3 and hydrogens:
            return f"C sp2 con H, {ring}"
        if degree == 3:
            return f"C sp2, {ring}"
        if degree == 2:
            return "C sp (cadena o terminal)"
        if degree == 1:
            return "C terminal"
        return f"C {degree}-coordinado, {ring}"

    if symbol == "H":
        partner = neighbours[0] if neighbours else "?"
        return f"H sobre {partner}"

    return f"{symbol} {degree}-coordinado, {ring}"


def describe(atoms: Atoms, tolerance: float = 0.30) -> StructureView:
    """Caja, átomos y entornos de ``atoms``.

    ``tolerance`` es la holgura con que se deciden los enlaces, la misma
    que usa :func:`~carbonforge.topology.graph.build_bond_graph`.
    """
    n = len(atoms)
    positions = atoms.get_positions()
    lengths = tuple(float(x) for x in atoms.cell.lengths())
    angles = tuple(float(x) for x in atoms.cell.angles())
    pbc = tuple(bool(p) for p in atoms.get_pbc())

    span, vacuum = [], []
    for axis in range(3):
        extent = (float(positions[:, axis].max() - positions[:, axis].min())
                  if n else 0.0)
        span.append(extent)
        gap = lengths[axis] - extent
        vacuum.append(max(0.0, gap / 2.0) if lengths[axis] > 0 else 0.0)

    cell = CellReport(
        lengths=lengths, angles=angles,  # type: ignore[arg-type]
        volume=float(atoms.get_volume()) if atoms.cell.rank == 3 else 0.0,
        pbc=pbc,  # type: ignore[arg-type]
        vacuum_per_side=tuple(vacuum),  # type: ignore[arg-type]
        span=tuple(span),  # type: ignore[arg-type]
        dimensionality=sum(pbc),
    )

    graph = build_bond_graph(atoms, tolerance=tolerance)
    rings = _ring_membership(graph, n)
    symbols = atoms.get_chemical_symbols()
    scaled = (atoms.get_scaled_positions(wrap=False)
              if atoms.cell.rank == 3 else None)

    rows, groups = [], {}
    for index in range(n):
        neighbours = tuple(sorted(graph.neighbors(index)))
        neighbour_symbols = tuple(sorted(symbols[j] for j in neighbours))
        label = _label(symbols[index], neighbour_symbols, rings[index])
        key = f"{symbols[index]}|{len(neighbours)}|" \
              f"{''.join(neighbour_symbols)}|{rings[index]}"
        rows.append(AtomRow(
            index=index, symbol=symbols[index],
            position=tuple(float(x) for x in positions[index]),  # type: ignore[arg-type]
            scaled=(tuple(float(x) for x in scaled[index])  # type: ignore[arg-type]
                    if scaled is not None else None),
            coordination=len(neighbours), neighbours=neighbours,
            neighbour_symbols=neighbour_symbols, rings=rings[index],
            environment=label))
        groups.setdefault(key, (label, []))[1].append(index)

    environments = tuple(
        Environment(key=key, label=label, members=tuple(members))
        for key, (label, members) in sorted(
            groups.items(), key=lambda item: (-len(item[1][1]), item[0])))

    notes = []
    loose = [r.index for r in rows if r.coordination == 0]
    if loose:
        notes.append(
            f"{len(loose)} átomo(s) sin ningún enlace detectado "
            f"(el primero es el {loose[0]}): o están sueltos, o la tolerancia "
            "de enlace es demasiado estrecha para esta estructura.")
    if n and cell.dimensionality == 0 and max(cell.lengths) <= 0:
        notes.append(
            "Sin celda: para GPAW hace falta una caja con vacío. Centra la "
            "estructura y dale vacío antes de calcular.")

    return StructureView(
        cell=cell, rows=tuple(rows), environments=environments,
        formula=atoms.get_chemical_formula(), notes=tuple(notes))


# ------------------------------------------------------------------ elección


def representatives(view: StructureView,
                    only: Optional[Iterable[str]] = None) -> tuple[int, ...]:
    """Un átomo por entorno, que es lo que un XPS necesita calcular.

    Con ``only`` se restringe a ciertas especies, p.ej. ``("N",)`` para
    quedarse con los tres nitrógenos distintos de una muestra dopada.
    """
    wanted = set(only) if only is not None else None
    out = []
    for environment in view.environments:
        index = environment.representative
        if wanted is None or view.row(index).symbol in wanted:
            out.append(index)
    return tuple(out)


def select(view: StructureView, *, element: Optional[str] = None,
           environment: Optional[str] = None,
           coordination: Optional[int] = None,
           ring: Optional[int] = None) -> tuple[int, ...]:
    """Los átomos que cumplen todo lo que se pida.

    Los criterios se combinan con Y: ``select(view, element="C", ring=5)``
    son los carbonos que están en algún pentágono.
    """
    out = []
    for row in view.rows:
        if element is not None and row.symbol != element:
            continue
        if environment is not None and environment not in row.environment:
            continue
        if coordination is not None and row.coordination != coordination:
            continue
        if ring is not None and ring not in row.rings:
            continue
        out.append(row.index)
    return tuple(out)
