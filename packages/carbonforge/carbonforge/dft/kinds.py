"""Qué cálculos sabe pedir carbonforge, y qué decide cada número.

Un :class:`Kind` es un tipo de cálculo -- relajación, estado base, bandas,
DOS, LDOS, fonones, XPS, Raman resonante -- descrito como datos y no como
código: sus parámetros, sus valores por defecto, qué tiene que haber
corrido antes, qué deja escrito, y un texto de ayuda por parámetro. La GUI
construye sus paneles leyendo esto, la CLI construye sus banderas leyendo
esto, y el registro de procedencia guarda esto. Añadir un tipo de cálculo
es añadir una entrada, no tocar tres capas.

**La validación es física, no de formato.** Un entero positivo puede ser
una barbaridad: 8 puntos k a lo largo de un eje no periódico, un hueco de
core sin decir en qué átomo, bandas pedidas sin un estado base del que
leer la densidad. Lo que distingue a un frente útil de un formulario es
que esas cosas se digan antes de lanzar, no después de seis horas de cola.

Lo que **no** hay aquí: ninguna llamada a GPAW. Este módulo describe; los
ejecutores traducen. Así se puede construir, validar y probar una receta
entera en una máquina sin GPAW instalado, que es como se trabaja la mitad
del tiempo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
from ase import Atoms

__all__ = [
    "KINDS",
    "Kind",
    "Parameter",
    "kind",
    "kind_names",
    "validate_chain",
]

ParamType = Literal["float", "int", "bool", "choice", "text", "indices", "path"]


@dataclass(frozen=True)
class Parameter:
    """Un ajuste, con lo que hace falta para pintarlo y para validarlo."""

    name: str
    label: str
    type: ParamType
    default: Any
    help: str
    choices: tuple[str, ...] = ()
    lo: Optional[float] = None
    hi: Optional[float] = None
    unit: str = ""
    advanced: bool = False

    def check(self, value: Any) -> Optional[str]:
        """Mensaje de error, o ``None`` si el valor es admisible.

        Sólo mira el valor en sí; lo que depende de la estructura o de los
        otros parámetros lo mira :meth:`Kind.check`.
        """
        if self.type == "choice":
            if value not in self.choices:
                return (f"{self.label}: {value!r} no está entre "
                        f"{', '.join(map(str, self.choices))}.")
            return None
        if self.type == "bool":
            return None if isinstance(value, bool) else f"{self.label}: debe ser sí o no."
        if self.type == "indices":
            try:
                items = [int(x) for x in value]
            except (TypeError, ValueError):
                return f"{self.label}: debe ser una lista de índices de átomos."
            if any(i < 0 for i in items):
                return f"{self.label}: los índices empiezan en 0 y no son negativos."
            return None
        if self.type in ("float", "int"):
            try:
                number = float(value)
            except (TypeError, ValueError):
                return f"{self.label}: debe ser un número."
            if self.type == "int" and float(value) != int(float(value)):
                return f"{self.label}: debe ser entero."
            if self.lo is not None and number < self.lo:
                return f"{self.label}: {number:g} está por debajo de {self.lo:g} {self.unit}".strip() + "."
            if self.hi is not None and number > self.hi:
                return f"{self.label}: {number:g} pasa de {self.hi:g} {self.unit}".strip() + "."
        return None


# --------------------------------------------------------------- compartidos

#: Ajustes que deciden el hamiltoniano y que por tanto comparten todos los
#: tipos de cálculo. Van juntos porque cambiar cualquiera de ellos invalida
#: la comparación entre dos corridas: un DOS con otro funcional no es el
#: mismo DOS.
BASE_PARAMETERS: tuple[Parameter, ...] = (
    Parameter(
        "mode", "Modo", "choice", "lcao",
        choices=("lcao", "fd", "pw"),
        help="Cómo se representan las funciones de onda.\n"
             "• lcao: base localizada (dzp). El más rápido; para barrer muchas "
             "estructuras. Su malla real debe ser fina (h ≤ 0.18 Å) o el efecto "
             "de caja de huevos rompe la simetría traslacional y contamina los "
             "modos de baja frecuencia.\n"
             "• fd: diferencias finitas en malla real. Mejorable sistemáticamente "
             "con h; el modo para confirmar lo que asignes.\n"
             "• pw: ondas planas (ecut). Converge con un solo número, pero "
             "resuelve Hartree con condiciones periódicas aunque pbc sea falso: "
             "las imágenes de una molécula polar interactúan por sus dipolos y "
             "eso mueve fuerzas y derivadas del dipolo. Con pw, más vacío."),
    Parameter(
        "xc", "Funcional", "choice", "PBE",
        choices=("PBE", "RPBE", "revPBE", "PBEsol", "LDA", "BEEF-vdW"),
        help="PBE es el punto de partida razonable para carbono. LDA sobreliga "
             "y acorta enlaces. Para interacciones de van der Waals entre capas "
             "ninguno de los GGA puros sirve: ahí hace falta BEEF-vdW u otra "
             "corrección de dispersión, y eso cambia las distancias entre planos "
             "pero poco las frecuencias intracapa."),
    Parameter(
        "h", "Espaciado de malla", "float", 0.18, unit="Å", lo=0.10, hi=0.30,
        help="Separación de la malla real, en lcao y fd. 0.18 Å es el límite "
             "superior razonable para carbono; por encima el efecto de caja de "
             "huevos mete fuerzas espurias que dependen de dónde caiga el átomo "
             "respecto a la malla. Bajarlo cuesta como h⁻³."),
    Parameter(
        "ecut", "Corte de ondas planas", "float", 500.0, unit="eV", lo=200.0, hi=1200.0,
        help="Sólo en modo pw. Por debajo de 400 eV las fuerzas sobre átomos de "
             "la primera fila no están convergidas. 500 eV es un punto de partida "
             "sensato para C, N y O."),
    Parameter(
        "basis", "Base LCAO", "choice", "dzp",
        choices=("sz", "szp", "dz", "dzp"),
        help="Sólo en modo lcao. dzp (doble zeta más polarización) es el mínimo "
             "para geometría y vibraciones; sz sirve para una ojeada y poco más.",
        advanced=True),
    Parameter(
        "kpts", "Puntos k", "text", "1,1,1",
        help="Malla de Monkhorst-Pack, un número por eje. Un eje NO periódico "
             "debe llevar 1: pedir más es muestrear una dirección que no se "
             "repite, y sólo gasta tiempo. Para una celda de ~15 Å a lo largo "
             "del eje periódico, 8 suele bastar para el estado base y hacen "
             "falta más para el DOS."),
    Parameter(
        "spinpol", "Polarización de espín", "bool", False,
        help="Enciéndela si el sistema tiene electrones desapareados: radicales, "
             "bordes zigzag, vacantes, metales de transición. Un cálculo sin "
             "espín sobre un sistema magnético da energías y DOS equivocados, y "
             "no avisa."),
    Parameter(
        "occupations_width", "Ensanchamiento", "float", 0.05, unit="eV", lo=0.0, hi=0.5,
        help="Ancho de Fermi-Dirac. En una molécula con hueco debe ser 0 o casi; "
             "en algo metálico (grafeno, un coil sin hueco) hace falta 0.05-0.1 eV "
             "para que el SCF converja, y las energías se extrapolan a 0.",
        advanced=True),
    Parameter(
        "convergence_density", "Convergencia de densidad", "float", 1e-6,
        lo=1e-10, hi=1e-3,
        help="Criterio del SCF. Las diferencias finitas de fuerzas sobre un "
             "desplazamiento de 0.01 Å necesitan fuerzas mucho mejor convergidas "
             "que lo que basta para una energía total: 1e-6 para vibraciones, "
             "1e-5 es lo más flojo que sigue dando diferencias limpias.",
        advanced=True),
)


def _p(*parameters: Parameter) -> tuple[Parameter, ...]:
    return tuple(parameters)


# ------------------------------------------------------------------- tipos


@dataclass(frozen=True)
class Kind:
    """Un tipo de cálculo: qué pide, qué necesita antes y qué deja."""

    name: str
    label: str
    summary: str
    help: str
    own: tuple[Parameter, ...] = ()
    requires: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    needs_sites: bool = False
    base: bool = True
    #: Validación que sólo tiene sentido para este tipo. Se le pasan los
    #: valores y la estructura, y devuelve los problemas que vea.
    checker: Optional[Callable[[dict, Optional[Atoms]], list[str]]] = None

    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Los propios detrás de los compartidos, en orden de panel."""
        return (BASE_PARAMETERS if self.base else ()) + self.own

    def parameter(self, name: str) -> Parameter:
        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise KeyError(f"{self.name} no tiene el parámetro {name!r}.")

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.parameters}

    def check(self, values: dict[str, Any],
              atoms: Optional[Atoms] = None) -> list[str]:
        """Todo lo que está mal, dicho antes de lanzar.

        Primero cada valor por su cuenta, después lo que sólo se ve con la
        estructura delante: puntos k sobre ejes que no se repiten, vacío
        insuficiente en ondas planas, sitios sin elegir.
        """
        problems: list[str] = []
        for parameter in self.parameters:
            if parameter.name not in values:
                continue
            message = parameter.check(values[parameter.name])
            if message:
                problems.append(message)

        mode = values.get("mode")
        if mode == "pw" and atoms is not None and not all(atoms.pbc):
            gaps = _vacuum_per_axis(atoms)
            thin = [ax for ax, (periodic, gap) in enumerate(zip(atoms.pbc, gaps))
                    if not periodic and gap < 8.0]
            if thin:
                names = ", ".join("xyz"[a] for a in thin)
                problems.append(
                    f"Modo pw con poco vacío en {names}: Hartree se resuelve con "
                    "condiciones periódicas aunque el eje no lo sea, así que las "
                    "imágenes se hablan por sus dipolos. Sube a 8 Å por lado o usa "
                    "fd o lcao, que no tienen el problema.")

        kpts = _parse_kpts(values.get("kpts", "1,1,1"))
        if kpts is None:
            problems.append("Puntos k: escribe tres enteros separados por comas.")
        elif atoms is not None:
            for axis, (n, periodic) in enumerate(zip(kpts, atoms.pbc)):
                if n > 1 and not periodic:
                    problems.append(
                        f"{n} puntos k sobre el eje {'xyz'[axis]}, que no es "
                        "periódico: esa dirección no se repite, así que sólo "
                        "gastas tiempo. Pon 1.")

        if self.needs_sites and not list(values.get("sites", []) or []):
            problems.append(
                f"{self.label} necesita que digas sobre qué átomos va. "
                "Elígelos en el panel de estructura.")
        if atoms is not None:
            for index in values.get("sites", []) or []:
                if int(index) >= len(atoms):
                    problems.append(
                        f"El átomo {index} no existe: la estructura tiene "
                        f"{len(atoms)} (0 a {len(atoms) - 1}).")
                    break
        problems.extend(self._own_check(values, atoms))
        return problems

    def _own_check(self, values: dict[str, Any],
                   atoms: Optional[Atoms]) -> list[str]:
        return list(self.checker(values, atoms)) if self.checker else []


def _vacuum_per_axis(atoms: Atoms) -> list[float]:
    """Vacío por lado en cada eje, en Å."""
    if atoms.cell.rank == 0:
        return [0.0, 0.0, 0.0]
    positions = atoms.get_positions()
    lengths = atoms.cell.lengths()
    out = []
    for axis in range(3):
        span = float(positions[:, axis].max() - positions[:, axis].min())
        out.append(max(0.0, (float(lengths[axis]) - span) / 2.0))
    return out


def _parse_kpts(text: Any) -> Optional[tuple[int, int, int]]:
    if isinstance(text, (list, tuple)) and len(text) == 3:
        try:
            return tuple(int(x) for x in text)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return None
    try:
        parts = [int(p) for p in str(text).replace(" ", "").split(",")]
    except ValueError:
        return None
    return tuple(parts) if len(parts) == 3 else None  # type: ignore[return-value]


# ------------------------------------------------------------- validaciones


def _check_relax(values: dict, atoms: Optional[Atoms]) -> list[str]:
    out: list[str] = []
    if values.get("relax_cell") and atoms is not None and not all(atoms.pbc):
        axes = ", ".join("xyz"[a] for a, p in enumerate(atoms.pbc) if not p)
        out.append(
            f"Relajar la celda con los ejes {axes} no periódicos colapsaría el "
            "vacío: no hay nada que lo sostenga y el optimizador lo lee como "
            "energía a ganar. Relaja sólo las posiciones, o haz la celda "
            "periódica en los tres ejes.")
    if float(values.get("fmax", 0.05)) > 0.05 and values.get("for_vibrations"):
        out.append(
            "Para vibraciones, fmax por encima de 0.05 eV/Å deja fuerzas "
            "residuales que salen como frecuencias imaginarias. Baja a 0.01-0.02.")
    return out


def _check_bands(values: dict, atoms: Optional[Atoms]) -> list[str]:
    out: list[str] = []
    if atoms is not None and not any(atoms.pbc):
        out.append(
            "Una estructura sin ningún eje periódico no tiene estructura de "
            "bandas: sus estados son niveles discretos. Lo que buscas es el "
            "espectro de autovalores, no un camino en k.")
    if not str(values.get("path", "")).strip():
        out.append("Camino de bandas vacío: escribe uno (p.ej. 'GXM G') o deja "
                   "que ASE proponga el de la red con 'auto'.")
    return out


def _check_dos(values: dict, atoms: Optional[Atoms]) -> list[str]:
    out: list[str] = []
    kpts = _parse_kpts(values.get("kpts", "1,1,1"))
    if kpts and atoms is not None:
        periodic = [n for n, p in zip(kpts, atoms.pbc) if p]
        if periodic and max(periodic) < 8:
            out.append(
                f"Para un DOS, {max(periodic)} puntos k en el eje periódico es "
                "poco: el DOS es una integral sobre la zona de Brillouin y con "
                "malla gruesa salen picos que son artefactos del muestreo. "
                "Suele hacer falta 2-4 veces la malla del estado base.")
    return out


def _check_phonons(values: dict, atoms: Optional[Atoms]) -> list[str]:
    out: list[str] = []
    supercell = _parse_kpts(values.get("supercell", "1,1,1"))
    if supercell is None:
        out.append("Supercelda: escribe tres enteros separados por comas.")
        return out
    if atoms is not None:
        displacements = 6 * len(atoms) * int(np.prod(supercell))
        out.append(
            f"Nota de coste: {displacements} desplazamientos "
            f"({len(atoms)} átomos x 6 x supercelda {np.prod(supercell):.0f}). "
            "Cada uno es un SCF completo.")
        for axis, (n, p) in enumerate(zip(supercell, atoms.pbc)):
            if n > 1 and not p:
                out.append(
                    f"Supercelda {n} sobre el eje {'xyz'[axis]}, que no es "
                    "periódico: repetir vacío no añade física.")
    return out


def _check_xps(values: dict, atoms: Optional[Atoms]) -> list[str]:
    out: list[str] = []
    if atoms is not None:
        for index in values.get("sites", []) or []:
            if int(index) < len(atoms) and atoms[int(index)].symbol == "H":
                out.append(
                    f"El átomo {index} es hidrógeno y no tiene nivel 1s de core "
                    "que ionizar: elige C, N u O.")
                break
    if values.get("spinpol") is False:
        out.append(
            "Un hueco de core deja el sistema con un electrón desapareado: el "
            "cálculo final tiene que ir con espín. Enciende la polarización.")
    return out


def _check_resonant_raman(values: dict, atoms: Optional[Atoms]) -> list[str]:
    out: list[str] = []
    if atoms is not None and any(atoms.pbc):
        out.append(
            "El Raman resonante por LrTDDFT de ASE está hecho para sistemas "
            "finitos: pide excitaciones de un espectro discreto. Para un sólido "
            "periódico el camino es el de electrón-fonón, no éste.")
    try:
        laser = float(values.get("laser_energy", 0.0))
    except (TypeError, ValueError):
        laser = 0.0
    if laser <= 0.0:
        out.append("Energía del láser: ponla en eV (532 nm son 2.33 eV).")
    if int(values.get("excitations", 0) or 0) < 10:
        out.append(
            "Pocas excitaciones: la suma sobre estados intermedios necesita "
            "cubrir por encima de la energía del láser, no sólo hasta ella.")
    return out


# ------------------------------------------------------------------ catálogo

KINDS: dict[str, Kind] = {}


def _add(kind_object: Kind) -> Kind:
    KINDS[kind_object.name] = kind_object
    return kind_object


_add(Kind(
    name="relax",
    label="Relajación de geometría",
    summary="Mueve los átomos hasta que las fuerzas bajan del criterio.",
    help="El paso del que depende todo lo demás. Una geometría que no es "
         "mínimo del MISMO modelo que luego calcula no sirve: los fonones "
         "salen imaginarios y las energías no son comparables. Si vas a "
         "calcular vibraciones, relaja con el mismo funcional, la misma base "
         "y la misma malla.",
    own=_p(
        Parameter("optimizer", "Optimizador", "choice", "BFGS",
                  choices=("BFGS", "LBFGS", "FIRE", "GPMin"),
                  help="BFGS converge en pocos pasos cerca del mínimo. FIRE "
                       "aguanta mejor un arranque muy lejos del mínimo o una "
                       "superficie de energía accidentada. GPMin suele ser el "
                       "que menos evaluaciones gasta en sistemas medianos."),
        Parameter("fmax", "Fuerza máxima", "float", 0.02, unit="eV/Å",
                  lo=0.001, hi=1.0,
                  help="Cuándo parar. 0.05 eV/Å basta para una energía; para "
                       "vibraciones hace falta 0.01-0.02, porque la fuerza "
                       "residual se cuela como frecuencia imaginaria."),
        Parameter("steps", "Pasos máximos", "int", 300, lo=1, hi=5000,
                  help="Tope de iteraciones. Si lo alcanza sin converger, lo "
                       "que hay no es un mínimo y hay que decirlo."),
        Parameter("relax_cell", "Relajar también la celda", "bool", False,
                  help="Sólo tiene sentido con los tres ejes periódicos. Con "
                       "vacío, el optimizador lo colapsa.",
                  advanced=True),
        Parameter("for_vibrations", "Va a usarse para vibraciones", "bool", False,
                  help="Marca esto y el criterio de fuerza se juzga con la vara "
                       "estricta que piden las diferencias finitas."),
    ),
    produces=("relaxed.traj", "relaxed.xyz"),
    checker=_check_relax,
))

_add(Kind(
    name="scf",
    label="Estado base (SCF)",
    summary="Resuelve la densidad autoconsistente y la guarda.",
    help="El cálculo del que cuelgan bandas, DOS, LDOS y XPS. Se guarda el "
         "fichero .gpw con la densidad convergida, y los demás lo leen en vez "
         "de repetir el SCF: bandas y DOS se hacen a densidad fija.",
    own=_p(
        Parameter("nbands", "Bandas", "text", "auto",
                  help="'auto' deja que GPAW cuente las ocupadas y añada "
                       "margen. Para DOS o bandas por encima del nivel de Fermi "
                       "hacen falta bandas vacías de más: pon un número.",
                  advanced=True),
        Parameter("maxiter", "Iteraciones máximas del SCF", "int", 333,
                  lo=10, hi=5000, advanced=True,
                  help="Si no converge, el problema casi nunca es el tope: "
                       "mira el ensanchamiento y la polarización de espín."),
    ),
    produces=("gs.gpw",),
))

_add(Kind(
    name="bands",
    label="Estructura de bandas",
    summary="Autovalores a lo largo de un camino de la zona de Brillouin.",
    help="Se calcula a densidad fija leyendo el .gpw del estado base, así que "
         "no vuelve a hacer el SCF. Pide un camino en k: sólo tiene sentido "
         "en direcciones periódicas.",
    own=_p(
        Parameter("path", "Camino", "text", "auto",
                  help="Puntos de alta simetría, p.ej. 'GXM G'. Con 'auto', ASE "
                       "propone el camino estándar de la red que detecte."),
        Parameter("npoints", "Puntos del camino", "int", 120, lo=10, hi=2000,
                  help="Cuántos k se reparten por el camino. 120 da curvas "
                       "limpias; más no añade física, sólo tiempo."),
        Parameter("nbands_extra", "Bandas vacías de más", "int", 10, lo=0, hi=200,
                  help="Bandas por encima del nivel de Fermi. Sin ellas la "
                       "gráfica se corta justo donde empieza lo interesante."),
    ),
    requires=("scf",),
    produces=("bands.json",),
    checker=_check_bands,
))

_add(Kind(
    name="dos",
    label="Densidad de estados",
    summary="DOS total y proyectada por elemento u orbital.",
    help="Integral sobre la zona de Brillouin, así que es mucho más sensible "
         "a la malla de puntos k que la energía total: con malla gruesa salen "
         "picos que son del muestreo y no del material.",
    own=_p(
        Parameter("width", "Ensanchamiento", "float", 0.1, unit="eV",
                  lo=0.0, hi=2.0,
                  help="Gaussiana con que se suaviza. Demasiado ancho borra "
                       "estructura real; demasiado estrecho deja ver el "
                       "muestreo en k."),
        Parameter("npts", "Puntos de energía", "int", 801, lo=50, hi=10000,
                  help="Cuántos puntos de energía se evalúan. Más puntos no añaden "
                       "información si el ensanchamiento es ancho: la "
                       "resolución real la fija width, no esto."),
        Parameter("projection", "Proyección", "choice", "total",
                  choices=("total", "elemento", "orbital"),
                  help="'total' es el DOS a secas. 'elemento' lo separa por "
                       "especie química. 'orbital' por s, p, d, que es lo que "
                       "distingue sp2 de sp3 en carbono."),
    ),
    requires=("scf",),
    produces=("dos.json",),
    checker=_check_dos,
))

_add(Kind(
    name="ldos",
    label="DOS local (por átomos)",
    summary="Densidad de estados proyectada sobre los átomos que elijas.",
    help="Lo mismo que la proyectada, pero sobre átomos concretos en vez de "
         "sobre una especie entera. Es lo que separa un nitrógeno piridínico "
         "de uno grafítico, o un carbono de pentágono de uno de hexágono. "
         "Hay que elegir los átomos en el panel de estructura.",
    own=_p(
        Parameter("sites", "Átomos", "indices", (),
                  help="Índices de los átomos sobre los que proyectar. Se "
                       "eligen pinchándolos en el visor de estructura."),
        Parameter("width", "Ensanchamiento", "float", 0.1, unit="eV",
                  lo=0.0, hi=2.0,
                  help="Gaussiana con que se suaviza, igual que en el DOS "
                       "total. Para comparar un LDOS con otro hay que usar "
                       "el mismo valor en los dos."),
        Parameter("orbitals", "Orbitales", "choice", "todos",
                  choices=("todos", "s", "p", "d"),
                  help="Separar por momento angular permite ver cuánto carácter "
                       "p queda en un carbono que la curvatura ha piramidalizado."),
    ),
    requires=("scf",),
    produces=("ldos.json",),
    needs_sites=True,
))

_add(Kind(
    name="phonons",
    label="Fonones",
    summary="Constantes de fuerza por desplazamientos finitos, en supercelda.",
    help="El coste manda: son 6N desplazamientos por celda, cada uno un SCF. "
         "La supercelda hace falta para fonones fuera de Γ; si sólo quieres "
         "los de Γ (que es lo que da el Raman de primer orden), déjala en "
         "1,1,1. La geometría tiene que venir relajada con estos mismos "
         "ajustes o salen frecuencias imaginarias.",
    own=_p(
        Parameter("supercell", "Supercelda", "text", "1,1,1",
                  help="Repeticiones por eje. 1,1,1 da sólo Γ. Para una curva "
                       "de dispersión hacen falta al menos 3 en la dirección "
                       "periódica, y el coste se multiplica por el volumen."),
        Parameter("delta", "Desplazamiento", "float", 0.01, unit="Å",
                  lo=0.001, hi=0.05,
                  help="Tamaño del desplazamiento finito. Muy pequeño y el "
                       "ruido del SCF domina; muy grande y entra anarmonicidad."),
        Parameter("central", "Diferencias centradas", "bool", True,
                  help="±δ en vez de sólo +δ. Dobla el coste y quita el error "
                       "de primer orden: para fonones vale la pena.",
                  advanced=True),
    ),
    requires=("relax",),
    produces=("phonons.json",),
    checker=_check_phonons,
))

_add(Kind(
    name="xps",
    label="XPS (ΔSCF con hueco de core)",
    summary="Energías de ligadura y corrimientos químicos por sitio.",
    help="La energía de ligadura sale como diferencia de energías totales "
         "entre el estado base y uno con un hueco en el nivel 1s del átomo "
         "elegido, que es el método de estado final y no la aproximación de "
         "Koopmans. Hay que hacer un cálculo POR SITIO inequivalente, así que "
         "elige los átomos con cabeza: el interés está en los corrimientos "
         "entre entornos (piridínico contra pirrólico contra grafítico), no "
         "en el valor absoluto, que arrastra el error del funcional.\n\n"
         "El nombre del setup con hueco depende de tu instalación de GPAW y "
         "hay que generarlo con su generador de setups. Déjalo por defecto si "
         "no lo has cambiado, y comprueba que GPAW lo encuentra antes de "
         "lanzar una tanda larga.",
    own=_p(
        Parameter("sites", "Átomos a ionizar", "indices", (),
                  help="Un cálculo por cada uno. Elige un representante de cada "
                       "entorno químico, no todos los átomos equivalentes."),
        Parameter("core_level", "Nivel", "choice", "1s",
                  choices=("1s",),
                  help="1s es lo que mide un XPS de laboratorio en C, N y O."),
        Parameter("setup_name", "Setup con hueco", "text", "fch1s",
                  help="Nombre del setup de GPAW con el hueco de core. El "
                       "convenio habitual es 'fch1s' (full core hole) para XPS "
                       "y 'hch1s' (half) para XAS. SIN VERIFICAR contra tu "
                       "instalación: compruébalo antes de una tanda larga."),
        Parameter("reference", "Átomo de referencia", "text", "",
                  help="Índice del átomo cuyo valor se toma como cero para los "
                       "corrimientos. Vacío deja las energías absolutas, que "
                       "son menos fiables que las diferencias."),
    ),
    requires=("scf",),
    produces=("xps.json",),
    needs_sites=True,
    checker=_check_xps,
))

_add(Kind(
    name="raman_resonant",
    label="Raman resonante",
    summary="Intensidades a la energía del láser, por LrTDDFT.",
    help="Usa ResonantRamanCalculator de ASE con excitaciones de LrTDDFT en "
         "cada desplazamiento: por eso cuesta mucho más que el Raman no "
         "resonante, que sólo necesita la polarizabilidad estática. Es el "
         "camino para comparar con un espectro medido a una longitud de onda "
         "concreta, donde la resonancia decide qué bandas se ven.\n\n"
         "Está pensado para sistemas finitos. Para un sólido periódico el "
         "camino es el de acoplamiento electrón-fonón, no éste.",
    own=_p(
        Parameter("laser_energy", "Energía del láser", "float", 2.33, unit="eV",
                  lo=0.5, hi=6.0,
                  help="532 nm = 2.33 eV; 633 nm = 1.96 eV; 785 nm = 1.58 eV."),
        Parameter("excitations", "Excitaciones", "int", 40, lo=1, hi=1000,
                  help="Cuántos estados intermedios entran en la suma. Tienen "
                       "que llegar por encima de la energía del láser."),
        Parameter("approximation", "Aproximación", "choice", "Profeta",
                  choices=("Placzek", "Profeta", "Albrecht"),
                  help="Placzek es el límite no resonante. Profeta y Albrecht "
                       "incluyen la resonancia; Albrecht separa los términos A "
                       "y B, que es lo que explica por qué unas bandas se "
                       "realzan y otras no."),
        Parameter("delta", "Desplazamiento", "float", 0.01, unit="Å",
                  lo=0.001, hi=0.05, advanced=True,
                  help="Desplazamiento finito para las derivadas. Igual que "
                       "en fonones: muy pequeño y manda el ruido del SCF, "
                       "muy grande y entra anarmonicidad."),
    ),
    requires=("relax",),
    produces=("raman_resonant.json",),
    checker=_check_resonant_raman,
))


def kind(name: str) -> Kind:
    """El tipo de cálculo llamado ``name``."""
    try:
        return KINDS[name]
    except KeyError:
        raise KeyError(
            f"No hay un cálculo llamado {name!r}. Hay: "
            f"{', '.join(sorted(KINDS))}.") from None


def kind_names() -> tuple[str, ...]:
    """Los tipos, en el orden en que tiene sentido correrlos."""
    return tuple(KINDS)


def validate_chain(names: Sequence[str]) -> list[str]:
    """Qué le falta a una receta para poder correrse en ese orden.

    Un DOS sin estado base delante no es un error de valor: es una receta
    que no cierra, y conviene decirlo al editarla y no al lanzarla.
    """
    problems: list[str] = []
    done: set[str] = set()
    for name in names:
        if name not in KINDS:
            problems.append(f"{name!r} no es un tipo de cálculo.")
            continue
        for needed in KINDS[name].requires:
            if needed not in done:
                problems.append(
                    f"{KINDS[name].label} necesita {KINDS[needed].label} antes, "
                    "y no está en la receta.")
        done.add(name)
    return problems
