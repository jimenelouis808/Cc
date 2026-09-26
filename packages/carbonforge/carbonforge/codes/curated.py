"""The curated core of each code's catalogue: the keywords worth knowing.

Short, written here, and deliberately incomplete: the parameters a
nanocarbon calculation commonly needs to adjust, plus every keyword
carbonforge sets itself (marked ``managed``, so overriding one is reported).
The code's own manual, imported with :mod:`carbonforge.codes.importers`,
completes and supersedes these entries.
"""

from __future__ import annotations

from .catalog import Catalog, Parameter


def _qe() -> list[Parameter]:
    P = Parameter  # noqa: N806
    m = {"managed": True}
    return [
        # &CONTROL
        P("qe", "calculation", "control", "string", "'scf'",
          ("scf", "nscf", "bands", "relax", "md", "vc-relax", "vc-md"),
          description="Tipo de cálculo.", **m),
        P("qe", "prefix", "control", "string", "'pwscf'", description="Prefijo de los archivos.", **m),
        P("qe", "pseudo_dir", "control", "string", description="Carpeta de pseudopotenciales.", **m),
        P("qe", "outdir", "control", "string", description="Carpeta de archivos temporales.", **m),
        P("qe", "verbosity", "control", "string", "'low'", ("low", "high"),
          description="Detalle de la salida.", **m),
        P("qe", "tprnfor", "control", "logical", description="Calcular fuerzas.", **m),
        P("qe", "tstress", "control", "logical", description="Calcular el tensor de tensiones.", **m),
        P("qe", "nstep", "control", "integer", description="Pasos máximos de relajación o MD."),
        P("qe", "etot_conv_thr", "control", "real", "1.0D-4", units="Ry",
          description="Convergencia de la energía entre pasos iónicos."),
        P("qe", "forc_conv_thr", "control", "real", "1.0D-3", units="Ry/Bohr",
          description="Convergencia de las fuerzas en la relajación."),
        P("qe", "disk_io", "control", "string", choices=("high", "medium", "low", "nowf", "none"),
          description="Cuánto escribe a disco; 'low' ahorra espacio."),
        P("qe", "max_seconds", "control", "real", units="s",
          description="Tiempo máximo: el cálculo se detiene limpio para reanudarlo."),
        P("qe", "tefield", "control", "logical",
          description="Campo eléctrico en diente de sierra (ver edir, eamp)."),
        P("qe", "dipfield", "control", "logical",
          description="Corrección de dipolo en slabs (requiere tefield)."),
        # &SYSTEM
        P("qe", "ibrav", "system", "integer", "0", description="Red de Bravais.", **m),
        P("qe", "nat", "system", "integer", description="Número de átomos.", **m),
        P("qe", "ntyp", "system", "integer", description="Número de especies.", **m),
        P("qe", "ecutwfc", "system", "real", units="Ry",
          description="Cutoff de funciones de onda.", **m),
        P("qe", "ecutrho", "system", "real", "4 * ecutwfc", units="Ry",
          description="Cutoff de densidad: 4x con norm-conserving, 8-12x con PAW/ultrasoft.", **m),
        P("qe", "occupations", "system", "string", "'fixed'",
          ("smearing", "tetrahedra", "tetrahedra_opt", "fixed", "from_input"),
          description="Ocupaciones electrónicas.", **m),
        P("qe", "smearing", "system", "string", "'gaussian'",
          ("gaussian", "mv", "m-v", "cold", "mp", "m-p", "fd", "f-d"),
          description="Tipo de smearing.", **m),
        P("qe", "degauss", "system", "real", "0", units="Ry", description="Ancho del smearing.",
          **m),
        P("qe", "nspin", "system", "integer", "1", ("1", "2"),
          description="1 sin espín, 2 colineal.", **m),
        P("qe", "tot_magnetization", "system", "real",
          description="Magnetización total fija (fija la diferencia de electrones up-down).",
          **m),
        P("qe", "tot_charge", "system", "real", "0.0",
          description="Carga total del sistema (+ = faltan electrones)."),
        P("qe", "nbnd", "system", "integer",
          description="Número de bandas; súbelo para ver estados vacíos."),
        P("qe", "nosym", "system", "logical",
          description="Desactivar la simetría (útil con estructuras distorsionadas)."),
        P("qe", "noncolin", "system", "logical", description="Espín no colineal.", **m),
        P("qe", "lspinorb", "system", "logical", description="Acoplamiento espín-órbita.", **m),
        P("qe", "input_dft", "system", "string", description="Funcional, si no el del pseudo.",
          **m),
        P("qe", "vdw_corr", "system", "string", choices=(
            "none", "grimme-d2", "dft-d", "grimme-d3", "dft-d3", "ts", "tkatchenko-scheffler",
            "xdm", "mbd"), description="Corrección de van der Waals.", **m),
        P("qe", "assume_isolated", "system", "string",
          choices=("none", "makov-payne", "m-p", "mp", "martyna-tuckerman", "m-t", "mt",
                   "esm", "2D"),
          description="Corrección para sistemas no periódicos (0D, 2D).", **m),
        P("qe", "exxdiv_treatment", "system", "string",
          choices=("gygi-baldereschi", "vcut_spherical", "vcut_ws", "none"),
          description="Divergencia del intercambio exacto en híbridos."),
        P("qe", "ecutfock", "system", "real", units="Ry",
          description="Cutoff del operador de Fock en híbridos; bajarlo abarata mucho."),
        P("qe", "edir", "system", "integer", choices=("1", "2", "3"),
          description="Dirección del campo eléctrico (tefield)."),
        P("qe", "eamp", "system", "real", units="Ha a.u.", description="Amplitud del campo."),
        P("qe", "esm_bc", "system", "string", choices=("pbc", "bc1", "bc2", "bc3"),
          description="Condiciones de contorno ESM para slabs cargados."),
        # &ELECTRONS
        P("qe", "conv_thr", "electrons", "real", "1.D-6", units="Ry",
          description="Convergencia SCF.", **m),
        P("qe", "mixing_beta", "electrons", "real", "0.7",
          description="Mezcla de densidad; bájala (0.1-0.3) si el SCF oscila.", **m),
        P("qe", "mixing_mode", "electrons", "string", "'plain'",
          ("plain", "TF", "local-TF"),
          description="'local-TF' ayuda en sistemas inhomogéneos (slabs, cintas)."),
        P("qe", "electron_maxstep", "electrons", "integer", "100",
          description="Iteraciones SCF máximas."),
        P("qe", "diagonalization", "electrons", "string", "'david'",
          ("david", "cg", "ppcg", "paro", "rmm-davidson", "rmm-paro"),
          description="Diagonalizador; 'cg' más lento pero robusto."),
        P("qe", "startingwfc", "electrons", "string", choices=("atomic", "atomic+random",
                                                              "random", "file"),
          description="Funciones de onda iniciales."),
        # &IONS / &CELL
        P("qe", "ion_dynamics", "ions", "string", choices=("bfgs", "damp", "fire", "verlet",
                                                          "langevin"),
          description="Algoritmo de relajación o MD.", **m),
        P("qe", "cell_dynamics", "cell", "string", choices=("none", "sd", "damp-pr", "damp-w",
                                                           "bfgs", "pr", "w"),
          description="Algoritmo de la celda.", **m),
        P("qe", "cell_dofree", "cell", "string",
          choices=("all", "ibrav", "a", "b", "c", "fixa", "fixb", "fixc", "x", "y", "z",
                   "xy", "xz", "yz", "xyz", "shape", "volume", "2Dxy", "2Dshape",
                   "epitaxial_ab", "epitaxial_ac", "epitaxial_bc"),
          description="Qué grados de libertad de la celda relajan.", **m),
        P("qe", "press", "cell", "real", "0.D0", units="kbar", description="Presión externa."),
        P("qe", "press_conv_thr", "cell", "real", "0.5D0", units="kbar",
          description="Convergencia de la presión."),
    ]


def _siesta() -> list[Parameter]:
    P = Parameter  # noqa: N806
    m = {"managed": True}
    return [
        P("siesta", "SystemLabel", kind="string", default="siesta",
          description="Prefijo de los archivos.", **m),
        P("siesta", "SystemName", kind="string", description="Título.", **m),
        P("siesta", "NumberOfAtoms", kind="integer", description="Número de átomos.", **m),
        P("siesta", "NumberOfSpecies", kind="integer", description="Número de especies.", **m),
        P("siesta", "MeshCutoff", kind="quantity", default="300 Ry", units="Ry",
          description="Cutoff de la malla real (no es el de ondas planas).", **m),
        P("siesta", "PAO.BasisSize", kind="string", default="DZP",
          choices=("SZ", "SZP", "DZ", "DZP", "TZP"), description="Tamaño de la base.", **m),
        P("siesta", "PAO.EnergyShift", kind="quantity", default="0.02 Ry", units="Ry",
          description="Confinamiento de los orbitales; menor = orbitales más extensos.", **m),
        P("siesta", "XC.functional", kind="string", default="LDA",
          choices=("LDA", "GGA", "VDW"), description="Familia del funcional.", **m),
        P("siesta", "XC.authors", kind="string", default="PZ",
          description="Funcional concreto (PBE, revPBE, DRSLL...).", **m),
        P("siesta", "DM.MixingWeight", kind="real", default="0.25",
          description="Mezcla de la matriz densidad.", **m),
        P("siesta", "DM.Tolerance", kind="real", default="1e-4",
          description="Convergencia SCF en la matriz densidad.", **m),
        P("siesta", "MD.MaxForceTol", kind="quantity", default="0.04 eV/Ang", units="eV/Ang",
          description="Fuerza máxima al relajar.", **m),
        P("siesta", "MD.FClast", kind="integer", description="Constantes de fuerza.", **m),
        P("siesta", "Spin", kind="string", default="non-polarized",
          choices=("non-polarized", "polarized", "non-colinear", "spin-orbit"),
          description="Componentes de espín.", **m),
        P("siesta", "ElectronicTemperature", kind="quantity", default="300 K", units="K",
          description="Smearing de Fermi-Dirac.", **m),
        P("siesta", "OccupationFunction", kind="string", default="FD", choices=("FD", "MP"),
          description="Función de ocupación."),
        P("siesta", "MaxSCFIterations", kind="integer", default="1000",
          description="Iteraciones SCF máximas.", **m),
        P("siesta", "SCF.Mixer.Method", kind="string", choices=("Pulay", "Broyden", "Linear"),
          description="Método de mezcla."),
        P("siesta", "NetCharge", kind="real", default="0.0",
          description="Carga neta del sistema."),
        P("siesta", "MD.TypeOfRun", kind="string",
          choices=("CG", "Broyden", "FIRE", "Verlet", "Nose", "FC", "LUA"),
          description="Relajación, MD o constantes de fuerza.", **m),
        P("siesta", "MD.Steps", kind="integer", description="Pasos máximos de relajación o MD."),
        P("siesta", "SolutionMethod", kind="string", choices=("diagon", "OMM", "transiesta"),
          description="Diagonalización u orden N.", **m),
        P("siesta", "WriteForces", kind="logical",
          description="Escribir fuerzas en la salida.", **m),
        P("siesta", "SaveRho", kind="logical", description="Guardar la densidad."),
        P("siesta", "DM.UseSaveDM", kind="logical",
          description="Reanudar desde una matriz densidad guardada."),
        P("siesta", "Slab.DipoleCorrection", kind="string",
          description="Corrección de dipolo en slabs."),
    ]


def _gpaw() -> list[Parameter]:
    P = Parameter  # noqa: N806
    m = {"managed": True}
    return [
        P("gpaw", "mode", kind="any", default="fd",
          description="lcao, fd o pw (con ecut).", **m),
        P("gpaw", "xc", kind="string", default="LDA", description="Funcional.", **m),
        P("gpaw", "h", kind="real", units="Ang", description="Paso de la rejilla real.", **m),
        P("gpaw", "basis", kind="any", description="Base LCAO (dzp, szp...).", **m),
        P("gpaw", "spinpol", kind="logical", description="Cálculo con espín.", **m),
        P("gpaw", "charge", kind="real", default="0", description="Carga total.", **m),
        P("gpaw", "convergence", kind="any", description="Criterios SCF (energy, density, "
          "forces, eigenstates, bands).", **m),
        P("gpaw", "symmetry", kind="any", description="Uso de la simetría.", **m),
        P("gpaw", "kpts", kind="any", description="Malla de puntos k (Γ en sistemas finitos)."),
        P("gpaw", "nbands", kind="any", description="Número de bandas ('150%' o un entero)."),
        P("gpaw", "occupations", kind="any",
          description="Ocupaciones: {'name': 'fermi-dirac', 'width': 0.05}."),
        P("gpaw", "mixer", kind="any", description="Mezclador de densidad."),
        P("gpaw", "maxiter", kind="integer", default="333", description="Iteraciones SCF."),
        P("gpaw", "eigensolver", kind="any", description="Diagonalizador."),
        P("gpaw", "poissonsolver", kind="any",
          description="Solver de Poisson; corrección de dipolo en slabs."),
        P("gpaw", "setups", kind="any", description="Datasets PAW por elemento."),
        P("gpaw", "parallel", kind="any", description="Reparto MPI (domain, band, kpt)."),
    ]


_BUILDERS = {"qe": _qe, "siesta": _siesta, "gpaw": _gpaw}


def curated_catalog(code: str) -> Catalog:
    """The curated core for ``code``."""
    catalog = Catalog(code, sources=["curado"])
    for parameter in _BUILDERS[code]():
        catalog.add(parameter)
    return catalog
