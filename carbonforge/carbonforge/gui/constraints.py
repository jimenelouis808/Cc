"""Constraints between parameters, checked before anything is built.

Per-field bounds (``ParamSpec.minimum`` / ``maximum``) catch a nonsense
number in isolation. They cannot catch the more common failure: two numbers
that are each reasonable but wrong *together*. A 60 Ry wavefunction cutoff is
fine and a 200 Ry density cutoff is fine, but the pair is rejected outright by
Quantum ESPRESSO, which requires the density cutoff to be at least four times
the wavefunction one.

Each constraint here states the rule, why it exists, and what to do — and is
evaluated live, so the feedback arrives while the user is still deciding
rather than after a job dies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ase import Atoms


@dataclass
class ConstraintViolation:
    """One rule that the current parameter combination breaks."""

    severity: str  # "error" blocks; "warning" informs
    message: str
    fields: tuple[str, ...] = ()

    @property
    def blocking(self) -> bool:
        return self.severity == "error"


def _as_float(values: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Read a possibly-string widget value as a float, tolerating commas."""
    raw = values.get(key, default)
    if isinstance(raw, (int, float)):
        return float(raw)
    try:
        return float(str(raw).strip().replace(",", "."))
    except ValueError:
        return default


def _as_int(values: dict[str, Any], key: str, default: int = 0) -> int:
    raw = values.get(key, default)
    if isinstance(raw, int):
        return raw
    try:
        return int(float(str(raw).strip().replace(",", ".")))
    except ValueError:
        return default


def _as_bool(values: dict[str, Any], key: str) -> bool:
    raw = values.get(key, False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "sí", "si", "yes", "on"}


# --------------------------------------------------------------------------
# Individual rules. Each takes the merged parameter dict plus the structure
# (when one has been built) and returns any violations.
# --------------------------------------------------------------------------


def _check_cutoff_ratio(values: dict[str, Any], atoms: Optional[Atoms]):
    """``ecutrho`` must be at least 4x ``ecutwfc``; 8x for PAW and ultrasoft."""
    wfc = _as_float(values, "ecutwfc", 60.0)
    rho = _as_float(values, "ecutrho", wfc * 8.0)
    if wfc <= 0:
        return []
    ratio = rho / wfc
    if ratio < 4.0:
        return [ConstraintViolation(
            "error",
            f"El cutoff de densidad ({rho:g} Ry) es solo {ratio:.1f}x el de "
            f"funciones de onda ({wfc:g} Ry). Quantum ESPRESSO exige al menos "
            "4x, y con PAW o ultrasoft hace falta 8x. Sube ecutrho o baja "
            "ecutwfc.",
            ("ecutwfc", "ecutrho"),
        )]
    if ratio < 8.0:
        return [ConstraintViolation(
            "warning",
            f"ecutrho es {ratio:.1f}x ecutwfc. Con norm-conserving basta 4x, "
            "pero los pseudopotenciales PAW y ultrasoft (los de por defecto "
            "aquí) necesitan 8x para converger la densidad.",
            ("ecutwfc", "ecutrho"),
        )]
    return []


def _check_dos_sampling(values: dict[str, Any], atoms: Optional[Atoms]):
    """The DOS energy step must be finer than the smearing it resolves."""
    if values.get("task") not in ("dos",) and values.get("preset") != "dos":
        return []
    delta_e = _as_float(values, "delta_e", 0.02)
    degauss_ry = _as_float(values, "degauss", 0.01)
    degauss_ev = degauss_ry * 13.6057
    if delta_e > degauss_ev:
        return [ConstraintViolation(
            "warning",
            f"El paso de energía del DOS ({delta_e:g} eV) es mayor que el "
            f"ensanchamiento ({degauss_ev:.3f} eV). Estarás submuestreando la "
            "curva: baja delta_e o sube degauss.",
            ("delta_e", "degauss"),
        )]
    return []


def _check_raman_feasibility(values: dict[str, Any], atoms: Optional[Atoms]):
    """Raman needs a band gap; metallic carbons cannot do it."""
    task = values.get("task")
    preset = values.get("preset")
    if task not in ("raman", "ir") and preset != "raman":
        return []
    if atoms is None:
        return []

    from ..validation.calculations import is_likely_metallic

    metallic, reason = is_likely_metallic(atoms)
    if metallic:
        return [ConstraintViolation(
            "error",
            f"Raman/IR necesita un gap y {reason} Quantum ESPRESSO no puede "
            "calcular epsil en un sistema metálico. Usa 'fonones' para tener "
            "frecuencias sin intensidades, o elige una quiralidad o borde "
            "semiconductor.",
            ("task", "preset"),
        )]
    return []


def _check_spin_needed(values: dict[str, Any], atoms: Optional[Atoms]):
    """A zigzag ribbon must be spin-polarised or its ground state is missed."""
    if atoms is None:
        return []
    kind = str(atoms.info.get("structure_type", ""))
    edge = str(atoms.info.get("edge", ""))
    if kind != "nanoribbon" or edge != "zigzag":
        return []

    # A preset handles this automatically; only the manual path can get it
    # wrong.
    if values.get("preset", "ninguna") != "ninguna":
        return []
    return [ConstraintViolation(
        "error",
        "Esta cinta zigzag tiene bordes magnéticos acoplados "
        "antiferromagnéticamente, y ese es su estado fundamental. Sin "
        "polarización de espín el SCF converge a otro estado sin dar ningún "
        "error, y las bandas salen mal. Usa una receta (la activa sola) o "
        "configura el espín a mano.",
        ("preset",),
    )]


def _check_hybrid_cost(values: dict[str, Any], atoms: Optional[Atoms]):
    """A hybrid on a large cell is a days-long job; say so before queueing."""
    if values.get("preset") != "bands-hse":
        return []
    if atoms is None:
        return []
    n_atoms = len(atoms)
    if n_atoms > 100:
        return [ConstraintViolation(
            "warning",
            f"HSE06 con {n_atoms} átomos cuesta del orden de 30x un PBE, que "
            "en una celda de este tamaño puede significar días. Converge "
            "primero con PBE y deja el híbrido para el gap final, o reduce "
            "la supercelda.",
            ("preset",),
        )]
    return []


def _check_group_capacity(values: dict[str, Any], atoms: Optional[Atoms]):
    """You cannot attach more groups than there are sites to attach them to."""
    group = values.get("group", "ninguno")
    if group in ("ninguno", None) or atoms is None:
        return []
    count = _as_int(values, "group_count", 1)
    site_kind = values.get("group_site", "edge")

    from ..functionalization.sites import find_sites

    try:
        available = len(find_sites(atoms, kind=site_kind))
    except Exception:
        return []

    if available == 0:
        return [ConstraintViolation(
            "error",
            f"No hay sitios de tipo '{site_kind}' en esta estructura. Una "
            "lámina periódica no tiene bordes: usa 'basal', o construye una "
            "cinta o un fragmento finito.",
            ("group", "group_site"),
        )]
    if count > available:
        return [ConstraintViolation(
            "error",
            f"Se piden {count} grupos pero solo hay {available} sitios "
            f"'{site_kind}'.",
            ("group_count",),
        )]
    if site_kind == "edge" and count > available // 2:
        return [ConstraintViolation(
            "warning",
            f"{count} grupos sobre {available} sitios de borde: quedarán en "
            "carbonos vecinos y podrían solaparse. El constructor lo "
            "rechazará si ocurre.",
            ("group_count",),
        )]
    return []


def _check_nitrogen_capacity(values: dict[str, Any], atoms: Optional[Atoms]):
    """Lattice nitrogen needs enough carbons, and vacancies need room."""
    nitrogen = values.get("nitrogen", "ninguno")
    if nitrogen in ("ninguno", None) or atoms is None:
        return []
    count = _as_int(values, "nitrogen_count", 1)
    n_carbon = sum(1 for s in atoms.get_chemical_symbols() if s == "C")

    if nitrogen == "graphitic" and count > n_carbon:
        return [ConstraintViolation(
            "error",
            f"Se piden {count} nitrógenos grafíticos pero solo hay "
            f"{n_carbon} carbonos.",
            ("nitrogen_count",),
        )]
    # A vacancy-based configuration needs space between defects.
    if nitrogen in ("pyridinic", "pyrrolic", "n-oxide") and count > 1:
        if n_carbon < 40 * count:
            return [ConstraintViolation(
                "warning",
                f"{count} sitios {nitrogen} en una celda de {n_carbon} "
                "carbonos quedan muy juntos. Sus vacantes interaccionarán "
                "entre sí y con las imágenes periódicas: usa una supercelda "
                "mayor si quieres sitios aislados.",
                ("nitrogen_count",),
            )]
    return []


def _check_vacuum_for_dft(values: dict[str, Any], atoms: Optional[Atoms]):
    """Vacuum below ~10 Å lets a slab interact with its own image."""
    vacuum = _as_float(values, "vacuum", 15.0)
    if vacuum <= 0:
        return []
    if vacuum < 10.0:
        return [ConstraintViolation(
            "warning",
            f"{vacuum:g} Å de vacío es poco para DFT: por debajo de ~10 Å la "
            "estructura interacciona con su propia imagen periódica y las "
            "energías dejan de ser las del sistema aislado.",
            ("vacuum",),
        )]
    return []


def _check_cores_vs_kpoints(values: dict[str, Any], atoms: Optional[Atoms]):
    """Warn when the core count wastes Quantum ESPRESSO's k-point pools."""
    if atoms is None or values.get("preset", "ninguna") == "ninguna":
        return []
    cores = _as_int(values, "cores", 8)
    if cores < 2:
        return []

    from ..exports.qe import QESettings
    from ..workflows.pipeline import count_kpoints, suggest_pools

    density = _as_float(values, "kpoint_density", QESettings().kpoint_density)
    try:
        n_kpoints = count_kpoints(atoms, density)
    except Exception:
        return []
    pools = suggest_pools(cores, n_kpoints)
    if pools == 1 and n_kpoints > 1:
        return [ConstraintViolation(
            "warning",
            f"Con {cores} núcleos y ~{n_kpoints} puntos k no sale ninguna "
            "división en pools. Quantum ESPRESSO escala mucho mejor sobre "
            "puntos k que sobre ondas planas: si puedes elegir, usa un "
            f"número de núcleos divisible por {n_kpoints}.",
            ("cores",),
        )]
    return []


#: Every rule, in the order they are reported.
CONSTRAINTS: tuple[Callable[[dict[str, Any], Optional[Atoms]], list], ...] = (
    _check_cutoff_ratio,
    _check_dos_sampling,
    _check_spin_needed,
    _check_raman_feasibility,
    _check_hybrid_cost,
    _check_group_capacity,
    _check_nitrogen_capacity,
    _check_vacuum_for_dft,
    _check_cores_vs_kpoints,
)


def check_constraints(
    values: dict[str, Any],
    atoms: Optional[Atoms] = None,
) -> list[ConstraintViolation]:
    """Evaluate every cross-parameter rule.

    Parameters
    ----------
    values
        All widget values merged into one dict.
    atoms
        The built structure, when there is one. Several rules cannot say
        anything useful without it — whether a ribbon is zigzag, how many
        edge sites exist — and are skipped rather than guessing.

    Returns
    -------
    list[ConstraintViolation]
        Errors first, then warnings.
    """
    violations: list[ConstraintViolation] = []
    for rule in CONSTRAINTS:
        try:
            violations.extend(rule(values, atoms))
        except Exception:
            # A broken rule must never block the user from working.
            continue
    return sorted(violations, key=lambda v: 0 if v.blocking else 1)


def format_violations(violations: list[ConstraintViolation]) -> str:
    """Render the violations for the GUI panel."""
    if not violations:
        return "✅ Los parámetros son coherentes entre sí."

    errors = [v for v in violations if v.blocking]
    warnings = [v for v in violations if not v.blocking]
    lines: list[str] = []
    if errors:
        lines.append("Incompatibilidades que impiden calcular:")
        lines.extend(f"  ❌ {v.message}" for v in errors)
    if warnings:
        if errors:
            lines.append("")
        lines.append("Avisos:")
        lines.extend(f"  ⚠️  {v.message}" for v in warnings)
    return "\n".join(lines)
