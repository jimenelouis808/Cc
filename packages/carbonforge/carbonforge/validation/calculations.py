"""Physics sanity checks for calculation setups.

:mod:`carbonforge.validation.checks` asks "is this structure geometrically
sane?". This module asks the complementary question: "will this *calculation*
actually run, and will the number it produces mean what you think?"

Every check here encodes a failure that is cheap to catch now and expensive
to discover later — either because the job dies hours into a queue, or worse,
because it completes and returns something quietly wrong. The four in the
last category are:

* **Spin-orbit with scalar-relativistic pseudopotentials.** The run succeeds
  and shows zero splitting, which is indistinguishable from "SOC is
  negligible here".
* **Variable-cell relaxation on a slab or wire.** ``vc-relax`` will happily
  compress the vacuum you carefully added, because vacuum costs energy to
  maintain; you get a converged run of a completely different system.
* **A metal with fixed occupations.** Converges to nonsense rather than
  failing outright.
* **A band path from the fallback generator.** Plots fine and looks
  publishable, but the labels are not the standard ones.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from ase import Atoms

from ..calculations.kpaths import BandPathSpec
from ..calculations.electronic import ElectronicSpec
from ..calculations.spectroscopy import SpectroscopySpec
from ..calculations.spinorbit import (
    SpinOrbitSpec,
    heaviest_element,
    soc_is_physically_relevant,
)
from .checks import ValidationReport

# Substrings identifying pseudopotential families in PSLibrary / SSSP /
# PseudoDojo filenames.
_PAW_MARKERS = ("kjpaw", "paw")
_USPP_MARKERS = ("rrkjus", "uspp")
_NC_MARKERS = ("oncv", "sg15", "nc-", "-mt", "vbc", "bhs", "hgh", "_nc_")
_REL_MARKERS = ("rel-", "_rel", "fr-")


def pseudo_family(filename: str) -> str:
    """Classify a pseudopotential file by name.

    Returns one of ``"NC"`` (norm-conserving), ``"USPP"`` (ultrasoft),
    ``"PAW"`` or ``"unknown"``.

    This is a filename heuristic, not a file parse: the authoritative answer
    lives in the UPF header. It is good enough to catch the common mistake of
    running Raman with the PSLibrary PAW defaults, and deliberately returns
    ``"unknown"`` rather than guessing when the name is unfamiliar.
    """
    name = filename.lower()
    # Order matters: "kjpaw" contains neither NC nor USPP markers, but some
    # NC names embed other substrings, so test the specific families first.
    if any(marker in name for marker in _PAW_MARKERS):
        return "PAW"
    if any(marker in name for marker in _USPP_MARKERS):
        return "USPP"
    if any(marker in name for marker in _NC_MARKERS):
        return "NC"
    return "unknown"


def is_fully_relativistic(filename: str) -> bool:
    """Whether a pseudopotential filename indicates a fully-relativistic set."""
    return any(marker in filename.lower() for marker in _REL_MARKERS)


def is_likely_metallic(atoms: Atoms) -> tuple[bool, str]:
    """Predict whether a nanocarbon is metallic, from its structural metadata.

    Uses the standard zone-folding rules rather than any electronic
    calculation:

    * A ``(n, m)`` nanotube is metallic when ``(n - m) % 3 == 0``; armchair
      ``(n, n)`` tubes are the always-metallic special case. Tubes with
      ``(n - m) % 3 == 0`` but ``n != m`` open a small curvature-induced gap
      (tens of meV), so they behave as near-metals.
    * Pristine graphene is a zero-gap semimetal.
    * Zigzag nanoribbons carry metallic edge states without spin
      polarisation.
    * Armchair nanoribbons are semiconducting, except the ``N = 3p + 2``
      family whose gap nearly closes.

    Returns
    -------
    (bool, str)
        The verdict and a short justification. Falls back to ``True`` with an
        "unknown" reason for structures carrying no usable metadata, since
        assuming metallic is the conservative choice: it only ever adds a
        warning, never suppresses one.
    """
    info = atoms.info
    kind = str(info.get("structure_type", ""))

    if kind in ("CNT", "nanocoil"):
        n, m = info.get("n"), info.get("m")
        if isinstance(n, int) and isinstance(m, int):
            if n == m:
                return True, f"CNT armchair ({n},{m}): metálico."
            if (n - m) % 3 == 0:
                return True, (
                    f"CNT ({n},{m}): (n-m) divisible por 3 → metálico o "
                    "casi metálico (pequeño gap por curvatura)."
                )
            return False, f"CNT ({n},{m}): semiconductor."
        return True, "CNT sin índices quirales conocidos: se asume metálico."

    if kind in ("graphene", "graphene_supercell"):
        return True, "Grafeno prístino: semimetal de gap nulo."

    if kind == "nanoribbon":
        edge = str(info.get("edge", ""))
        width = info.get("width")
        if edge == "zigzag":
            return True, "Nanocinta zigzag: estados de borde metálicos."
        if edge == "armchair" and isinstance(width, int):
            if width % 3 == 2:
                return True, (
                    f"Nanocinta armchair N={width} (familia 3p+2): "
                    "gap casi nulo."
                )
            return False, f"Nanocinta armchair N={width}: semiconductora."
        return False, "Nanocinta armchair: semiconductora."

    if kind == "carbon_foam":
        return True, "Espuma desordenada: se asume metálica (sin gap definido)."

    return True, "Tipo de estructura desconocido: se asume metálico."


def check_spectroscopy(
    atoms: Atoms,
    spec: SpectroscopySpec,
    pseudopotentials: Optional[dict[str, str]] = None,
) -> ValidationReport:
    """Validate a phonon / IR / Raman setup against QE's real requirements."""
    report = ValidationReport()

    if spec.needs_epsil:
        metallic, reason = is_likely_metallic(atoms)
        report.info["metallic_prediction"] = reason
        if metallic:
            report.errors.append(
                f"IR/Raman requiere epsil=.true., que Quantum ESPRESSO solo "
                f"puede calcular en sistemas con gap. {reason} "
                "El cálculo fallará. Alternativas: usa mode='phonon' (solo "
                "frecuencias, sin intensidades), o elige una quiralidad "
                "semiconductora."
            )

    if spec.needs_raman and pseudopotentials:
        offenders = {
            symbol: filename
            for symbol, filename in pseudopotentials.items()
            if pseudo_family(filename) in ("PAW", "USPP")
        }
        if offenders:
            detail = ", ".join(f"{s}: {f}" for s, f in offenders.items())
            report.errors.append(
                "Raman por DFPT en Quantum ESPRESSO NO admite pseudopotenciales "
                f"PAW ni ultrasoft, y estos lo son ({detail}). Necesitas "
                "norm-conserving (ONCV, SG15). Sin ellos ph.x se detiene."
            )
        unknown = {
            symbol: filename
            for symbol, filename in pseudopotentials.items()
            if pseudo_family(filename) == "unknown"
        }
        if unknown:
            report.warnings.append(
                "No se pudo determinar la familia de estos pseudopotenciales: "
                f"{', '.join(unknown)}. Confirma que son norm-conserving "
                "antes de lanzar el Raman."
            )

    if spec.needs_epsil and not spec.is_gamma:
        report.errors.append(
            f"Las intensidades IR/Raman solo están definidas en Γ, pero "
            f"q = {spec.qpoint}. Usa qpoint=(0,0,0)."
        )

    if spec.ldisp and spec.needs_epsil:
        report.errors.append(
            "ldisp=.true. (dispersión en una malla de q) es incompatible con "
            "intensidades IR/Raman, que son magnitudes de Γ."
        )

    if spec.asr == "no":
        report.warnings.append(
            "Sin regla de suma acústica (asr='no') los tres modos acústicos "
            "no salen a frecuencia cero y contaminan la zona de bajo número "
            "de onda. Recomendado: asr='crystal'."
        )

    dim = int(sum(atoms.get_pbc()))
    if dim in (1, 2) and spec.needs_epsil:
        report.warnings.append(
            f"En sistemas {dim}D con vacío, ε∞ depende del vacío que hayas "
            "puesto y no es una constante dieléctrica bien definida. Las "
            "frecuencias son fiables; las intensidades absolutas no. Compara "
            "intensidades relativas dentro del mismo espectro."
        )
    return report


def check_spinorbit(
    atoms: Atoms,
    spec: SpinOrbitSpec,
    pseudopotentials: Optional[dict[str, str]] = None,
) -> ValidationReport:
    """Validate a spin-orbit setup."""
    report = ValidationReport()
    if not spec.enabled:
        return report

    symbols = sorted(set(atoms.get_chemical_symbols()))
    element, z = heaviest_element(symbols)
    report.info["heaviest_element"] = f"{element} (Z={z})"

    if pseudopotentials:
        scalar = {
            symbol: filename
            for symbol, filename in pseudopotentials.items()
            if not is_fully_relativistic(filename)
        }
        if scalar:
            detail = ", ".join(f"{s}: {f}" for s, f in scalar.items())
            report.errors.append(
                "El acoplamiento espín-órbita exige pseudopotenciales "
                f"totalmente relativistas ('rel-'), y estos no lo son ({detail}). "
                "Con pseudos escalares el cálculo TERMINA SIN ERROR pero el "
                "desdoblamiento sale exactamente cero, que es indistinguible "
                "de 'aquí el SOC es despreciable'. Descarga las versiones "
                "rel- de PSLibrary."
            )

    if not soc_is_physically_relevant(symbols):
        report.warnings.append(
            f"El elemento más pesado presente es {element} (Z={z}). El "
            "acoplamiento espín-órbita crece aproximadamente como Z⁴: en "
            "grafeno prístino el gap SOC es del orden de 10⁻² meV, muy por "
            "debajo de la precisión de un DFT rutinario y de k_BT a "
            "temperatura ambiente (~25 meV). Activar SOC aquí duplica el "
            "coste para un efecto que probablemente no puedas resolver. "
            "Tiene sentido con adátomos pesados (Au, Bi, Pb) o un sustrato "
            "pesado."
        )

    if spec.enabled and not spec.noncolin:
        report.errors.append(
            "lspinorb=.true. requiere noncolin=.true. en Quantum ESPRESSO."
        )
    return report


def check_electronic_setup(
    atoms: Atoms,
    spec: Optional[ElectronicSpec] = None,
) -> ValidationReport:
    """Check the electronic-structure physics against the structure.

    The one that matters most: a **zigzag graphene nanoribbon has magnetic
    edge states**, antiferromagnetically coupled across the ribbon. Run it
    without spin polarisation and the SCF converges to a non-magnetic,
    metallic state that is not the ground state — no error, no warning from
    the code, just a wrong band structure and a gap of zero where the real
    answer is a few tenths of an eV.

    Also checks that dispersion is present when it is likely to matter, and
    warns about the cost of a hybrid before it is queued rather than after.
    """
    report = ValidationReport()
    kind = str(atoms.info.get("structure_type", ""))
    edge = str(atoms.info.get("edge", ""))
    spin_on = spec is not None and spec.is_spin_polarized

    if kind == "nanoribbon" and edge == "zigzag" and not spin_on:
        report.errors.append(
            "Nanocinta ZIGZAG sin polarización de espín. Sus bordes tienen "
            "estados magnéticos acoplados antiferromagnéticamente, y ese es "
            "el estado fundamental. Sin nspin=2 el SCF converge a un estado "
            "no magnético que NO es el fundamental: obtendrás bandas y gap "
            "equivocados, sin ningún aviso del código. Usa "
            "setup_antiferromagnetic_edges(), o --spin afm en la terminal."
        )

    if spec is None:
        return report

    report.info["electronic_cost_multiplier"] = spec.cost_multiplier()

    if spec.spin == "collinear" and not spec.starting_magnetization:
        report.warnings.append(
            "nspin=2 sin magnetizaciones iniciales: el SCF arranca desde un "
            "estado no magnético y probablemente se quede ahí. Da un patrón "
            "de signos con starting_magnetization."
        )

    # Dispersion matters wherever pieces are held together by nothing else.
    if spec.vdw_correction == "none":
        if kind == "carbon_foam":
            report.warnings.append(
                "Espuma sin corrección de van der Waals. Los fragmentos "
                "grafíticos se mantienen unidos por dispersión, que PBE no "
                "tiene: la estructura se desmoronará al relajar. Usa "
                "vdw_correction='grimme-d3'."
            )
        elif kind == "nanocoil":
            report.warnings.append(
                "Nanoespiral sin corrección de van der Waals. Las vueltas "
                "vecinas interaccionan por dispersión; sin ella el paso de "
                "hélice relajado saldrá demasiado grande."
            )

    if spec.is_hybrid:
        n_atoms = len(atoms)
        report.warnings.append(
            f"Funcional híbrido con {n_atoms} átomos: del orden de "
            f"{spec.cost_multiplier():.0f}x el coste de PBE. Con una celda de "
            "este tamaño puede pasar de días. Converge primero con PBE y usa "
            "el híbrido solo para el gap final."
        )
        if int(sum(atoms.get_pbc())) == 2:
            report.warnings.append(
                "Además, en sistemas 2D con vacío el intercambio exacto "
                "converge despacio con el vacío: comprueba que el gap sea "
                "estable al aumentarlo."
            )

    if spec.hubbard_u:
        light = {"C", "N", "B", "O", "H"}
        applied = set(spec.hubbard_u) & light
        if applied:
            report.warnings.append(
                f"Hubbard U sobre {', '.join(sorted(applied))}: son elementos "
                "ligeros sin electrones d o f localizados. DFT+U está pensado "
                "para metales de transición; aquí es difícil de justificar."
            )
    return report


def check_calculation_type(
    atoms: Atoms,
    calculation: str,
    occupations: str = "smearing",
    cell_dofree: Optional[str] = None,
) -> ValidationReport:
    """Validate the calculation type against the structure's dimensionality.

    The important one is ``vc-relax`` on a system with vacuum.
    """
    report = ValidationReport()
    pbc = atoms.get_pbc()
    dim = int(sum(pbc))

    if calculation == "vc-relax" and dim < 3:
        if cell_dofree is None:
            frozen = {
                1: "'z'  (solo el eje periódico)",
                2: "'2Dxy'  (solo el plano)",
                0: "ninguno: un sistema 0D no admite vc-relax",
            }
            suggestion = frozen.get(dim, "el subconjunto adecuado")
            report.errors.append(
                f"vc-relax en un sistema {dim}D relajará también las "
                "direcciones con vacío y lo comprimirá, porque mantener vacío "
                "cuesta energía. El resultado converge, pero de un sistema "
                "distinto al que querías. Fija cell_dofree = "
                f"{suggestion}."
            )
        else:
            report.info["cell_dofree"] = cell_dofree

    if calculation == "vc-relax" and dim == 0:
        report.errors.append(
            "vc-relax no tiene sentido en un sistema aislado (0D): no hay "
            "celda física que optimizar. Usa 'relax'."
        )

    metallic, reason = is_likely_metallic(atoms)
    if metallic and occupations == "fixed":
        report.errors.append(
            f"occupations='fixed' en un sistema sin gap. {reason} "
            "Converge a un resultado sin sentido en lugar de fallar. Usa "
            "occupations='smearing'."
        )
    if not metallic and occupations == "smearing":
        report.warnings.append(
            f"{reason} Con gap puedes usar occupations='fixed', que da "
            "energías algo más limpias, aunque 'smearing' también es válido."
        )
    return report


def check_band_path(spec: BandPathSpec) -> ValidationReport:
    """Validate a band path."""
    report = ValidationReport()
    report.info["band_path"] = spec.path_string
    report.info["band_path_source"] = spec.source

    if spec.dimensionality == 0:
        report.warnings.append(
            "Sistema 0D: no hay estructura de bandas, solo niveles discretos "
            "en Γ. Para una molécula o un cúmulo mira el espectro de "
            "autovalores, no una dispersión."
        )
    elif spec.source == "fallback":
        report.warnings.append(
            "No se reconoció la red de Bravais, así que el camino es genérico "
            "(Γ → borde de zona) y sus etiquetas no son las convencionales. "
            "Revísalo antes de publicar la figura."
        )
    if spec.npoints_per_segment < 10 and spec.dimensionality > 0:
        report.warnings.append(
            f"Solo {spec.npoints_per_segment} puntos por segmento: la "
            "dispersión saldrá angulosa. 30-50 es lo habitual."
        )
    return report


def check_full_setup(
    atoms: Atoms,
    calculation: str = "scf",
    occupations: str = "smearing",
    cell_dofree: Optional[str] = None,
    spectroscopy: Optional[SpectroscopySpec] = None,
    spinorbit: Optional[SpinOrbitSpec] = None,
    band_path: Optional[BandPathSpec] = None,
    pseudopotentials: Optional[dict[str, str]] = None,
    electronic: Optional[ElectronicSpec] = None,
) -> ValidationReport:
    """Run every applicable calculation check and merge the results."""
    report = ValidationReport()
    report.merge(
        check_calculation_type(atoms, calculation, occupations, cell_dofree)
    )
    report.merge(check_electronic_setup(atoms, electronic))
    if spectroscopy is not None:
        report.merge(check_spectroscopy(atoms, spectroscopy, pseudopotentials))
    if spinorbit is not None:
        report.merge(check_spinorbit(atoms, spinorbit, pseudopotentials))
    if band_path is not None:
        report.merge(check_band_path(band_path))
    return report
