"""Synthetic spectra for trying the program out and for testing it.

These are **generated**, not measured. They are built by summing lineshapes
at literature positions and adding noise and a fluorescence tail, so they
have the right shape and the right relationships between bands, and they let
the interface and the test suite be exercised without shipping somebody
else's data.

What they are good for: checking that the pipeline runs, that the classifier
separates the material classes, that the fitter recovers parameters it was
given, and — because the laser wavelength drives both the band positions and
the defect-band intensities — that the dispersion and λ⁴ corrections are
applied correctly. What they are **not** good for: validating the package against
reality. Real spectra have correlated noise, instrument response,
substrate lines, sample inhomogeneity and non-Lorentzian wings, none of
which are here. Every claim this package makes about accuracy has been
checked against synthetic data only, and the documentation says so.

The parameters below come from the ranges in ``database/data/materials.json``.
"""

from __future__ import annotations


import math
from typing import Optional

import numpy as np

from ..core.spectrum import Spectrum, laser_energy_ev
from ..models.lineshapes import bwf, gaussian, lorentzian

#: Reference excitation the band positions below are quoted at, in eV.
REFERENCE_EV = 2.33

#: The demo materials, in the order the GUI loads them.
DEMO_KINDS = (
    "SWCNT",
    "SWCNT_metalico",
    "DWCNT",
    "MWCNT",
    "grafeno_1L",
    "GO",
    "MWCNT_FeSe",
    "MWCNT_Se",
)


def _dispersed(position: float, dispersion: float, laser_ev: float) -> float:
    """Move a band to where it would sit at this excitation."""
    return position + dispersion * (laser_ev - REFERENCE_EV)


def make_demo(
    kind: str = "SWCNT",
    laser_nm: float = 532.0,
    seed: int = 0,
    low: float = 90.0,
    high: float = 3200.0,
    step: float = 1.0,
    noise: float = 2.0,
    fluorescence: float = 260.0,
) -> Spectrum:
    """Build one synthetic spectrum of a named material.

    Parameters
    ----------
    kind:
        One of :data:`DEMO_KINDS`.
    laser_nm:
        Excitation wavelength. Two things follow from it, so that changing
        it is a genuine test rather than a decorative parameter: the
        dispersive bands (D, 2D, D+D') move, so a 785 nm demo really does
        show its D band near 1312 cm⁻¹; and the defect-activated bands
        (D, D', D+D') scale in intensity as λ⁴ relative to G, so I_D/I_G
        really is about twice as large at 633 nm as at 532 nm.
    seed:
        Seed for the noise, so a demo is reproducible.
    low, high, step:
        Spectral range and sampling in cm⁻¹. Raising ``low`` above 400
        simulates an instrument that cannot reach the RBM region, which is
        the case the classifier has to refuse to over-interpret.
    noise:
        Gaussian noise standard deviation, in the same arbitrary units as
        the peak heights (which run to ~900).
    fluorescence:
        Amplitude of an exponentially decaying background.

    Returns
    -------
    Spectrum

    Raises
    ------
    ValueError
        If ``kind`` is not one of :data:`DEMO_KINDS`.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(float(low), float(high), float(step))
    y = np.zeros_like(x)
    ev = laser_energy_ev(laser_nm)

    # The defect-activated bands get stronger, relative to G, as the fourth
    # power of the excitation wavelength: I_D/I_G at 633 nm is about twice
    # its value at 532 nm on the SAME sample. Without this the demo spectra
    # would give a crystallite size that depends on which laser measured
    # them, which is exactly the error the lambda^4 correction exists to
    # remove — so leaving it out would make the two-wavelength consistency
    # check pass for the wrong reason.
    defect_gain = (laser_nm / 532.0) ** 4

    d = _dispersed(1350.0, 50.0, ev)
    two_d = _dispersed(2690.0, 100.0, ev)
    d_prime = _dispersed(1620.0, 10.0, ev)
    d_plus_dp = _dispersed(2940.0, 60.0, ev)

    if kind == "SWCNT":
        # Three resonant diameters, a narrow split G, very little disorder.
        y += lorentzian(x, 165.0, 90.0, 9.0)
        y += lorentzian(x, 187.0, 60.0, 8.0)
        y += lorentzian(x, 254.0, 40.0, 10.0)
        y += lorentzian(x, d, 55.0 * defect_gain, 32.0)
        y += lorentzian(x, 1570.0, 220.0, 22.0)
        y += lorentzian(x, 1591.0, 900.0, 16.0)
        y += lorentzian(x, d_prime, 45.0 * defect_gain, 18.0)
        y += lorentzian(x, two_d, 260.0, 45.0)
    elif kind == "SWCNT_metalico":
        # One diameter, and a Breit-Wigner-Fano G- from the electronic
        # continuum: the metallic signature.
        y += lorentzian(x, 195.0, 70.0, 10.0)
        y += lorentzian(x, d, 70.0 * defect_gain, 35.0)
        y += bwf(x, 1545.0, 300.0, 85.0, -0.22)
        y += lorentzian(x, 1592.0, 800.0, 17.0)
        y += lorentzian(x, two_d, 200.0, 50.0)
    elif kind == "DWCNT":
        # Two RBM clusters whose diameters differ by twice the wall spacing.
        y += lorentzian(x, 158.0, 70.0, 10.0)
        y += lorentzian(x, 178.0, 55.0, 11.0)
        y += lorentzian(x, 265.0, 65.0, 8.0)
        y += lorentzian(x, 291.0, 45.0, 9.0)
        y += lorentzian(x, d, 130.0 * defect_gain, 40.0)
        y += lorentzian(x, 1572.0, 180.0, 26.0)
        y += lorentzian(x, 1590.0, 850.0, 20.0)
        y += lorentzian(x, d_prime, 70.0 * defect_gain, 20.0)
        y += lorentzian(x, two_d, 220.0, 60.0)
    elif kind == "MWCNT":
        # No RBM, broad G with a D' shoulder, I_D/I_G near 1.
        y += lorentzian(x, d, 780.0 * defect_gain, 60.0)
        y += lorentzian(x, 1580.0, 900.0, 48.0)
        y += lorentzian(x, d_prime, 190.0 * defect_gain, 28.0)
        y += gaussian(x, 1500.0, 90.0, 180.0)
        y += lorentzian(x, two_d + 20.0, 180.0, 110.0)
        y += lorentzian(x, d_plus_dp, 90.0 * defect_gain, 130.0)
    elif kind == "grafeno_1L":
        # A single narrow 2D band, three times the G, almost no D.
        y += lorentzian(x, 1583.0, 300.0, 15.0)
        y += lorentzian(x, two_d - 12.0, 950.0, 28.0)
        y += lorentzian(x, d, 12.0 * defect_gain, 30.0)
    elif kind == "GO":
        # Broad D and G of comparable size, G upshifted, 2D essentially gone.
        y += lorentzian(x, d + 5.0, 900.0 * defect_gain, 130.0)
        y += lorentzian(x, 1598.0, 950.0, 85.0)
        y += gaussian(x, 1510.0, 260.0, 220.0)
    elif kind == "MWCNT_FeSe":
        # Multi-walled tubes decorated with tetragonal beta-FeSe, plus a
        # little residual trigonal selenium. This is the case the phase
        # scan exists for: the FeSe A1g/B1g pair at 181/196 and the Se A1
        # at 237 sit squarely inside the radial-breathing-mode window, and
        # a diameter analysis that does not know what they are converts
        # them into three confident, fictitious tube diameters.
        y += lorentzian(x, 181.0, 210.0, 7.0)
        y += lorentzian(x, 196.0, 130.0, 8.0)
        y += lorentzian(x, 102.0, 45.0, 9.0)
        y += lorentzian(x, 237.0, 120.0, 8.0)
        y += lorentzian(x, 143.0, 40.0, 10.0)
        y += lorentzian(x, d, 760.0 * defect_gain, 62.0)
        y += lorentzian(x, 1580.0, 880.0, 50.0)
        y += lorentzian(x, d_prime, 185.0 * defect_gain, 28.0)
        y += gaussian(x, 1500.0, 85.0, 180.0)
        y += lorentzian(x, two_d + 20.0, 170.0, 115.0)
    elif kind == "MWCNT_Se":
        # The same tubes with amorphous/ring selenium only: one broad band
        # near 252 cm-1 and no iron phase. Included so the polymorph logic
        # has a case where the family is present but the chain/ring call
        # is what carries the information.
        y += lorentzian(x, 252.0, 150.0, 22.0)
        y += lorentzian(x, 235.0, 70.0, 20.0)
        y += lorentzian(x, 112.0, 45.0, 12.0)
        y += lorentzian(x, d, 700.0 * defect_gain, 60.0)
        y += lorentzian(x, 1581.0, 860.0, 47.0)
        y += lorentzian(x, d_prime, 175.0 * defect_gain, 27.0)
        y += lorentzian(x, two_d + 20.0, 165.0, 110.0)
    else:
        raise ValueError(
            f"unknown demo material {kind!r}; available: {', '.join(DEMO_KINDS)}"
        )

    y += fluorescence * np.exp(-(x - x[0]) / 900.0) + 15.0
    y += rng.normal(0.0, noise, x.size)
    return Spectrum(
        shift=x,
        intensity=y,
        laser_nm=laser_nm,
        name=f"demo_{kind}_{laser_nm:g}nm",
        metadata={
            "synthetic": True,
            "warning": (
                "Espectro sintético generado por ramancarbon. No son datos "
                "medidos; sirven para probar el programa."
            ),
        },
    )


def demo_spectra(laser_nm: float = 532.0, seed: int = 0) -> list[Spectrum]:
    """One synthetic spectrum of each demo material."""
    return [make_demo(kind, laser_nm=laser_nm, seed=seed + i)
            for i, kind in enumerate(DEMO_KINDS)]


def add_doping(
    spectrum: Spectrum,
    delta_g: float = 8.0,
    delta_2d: float = -12.0,
    extra_disorder: float = 0.0,
) -> Spectrum:
    """Shift a demo spectrum's G and 2D bands, as doping would.

    Used by the examples and tests to check that the shift analysis
    recovers a perturbation that was deliberately put in. Implemented by
    resampling the spectrum on a locally stretched axis rather than by
    rebuilding it, so the band shapes are untouched and only the positions
    move — which is what the analysis is supposed to detect.

    Parameters
    ----------
    spectrum:
        A demo spectrum.
    delta_g:
        Shift applied around the G band, cm⁻¹. Positive is a stiffening,
        which is what both electron and hole doping produce.
    delta_2d:
        Shift applied around the 2D band. Negative with a positive
        ``delta_g`` is the n-type signature; positive is p-type.
    extra_disorder:
        Fractional increase of the D band's intensity, to mimic the
        defects that substitutional doping introduces.

    Returns
    -------
    Spectrum
    """
    import numpy as np

    x = spectrum.shift
    warp = np.zeros_like(x)
    warp += delta_g * np.exp(-0.5 * ((x - 1585.0) / 90.0) ** 2)
    warp += delta_2d * np.exp(-0.5 * ((x - 2690.0) / 160.0) ** 2)
    shifted = np.interp(x, x + warp, spectrum.intensity)

    if extra_disorder:
        boost = 1.0 + extra_disorder * np.exp(-0.5 * ((x - 1350.0) / 45.0) ** 2)
        shifted = shifted * boost

    out = spectrum.copy()
    out.intensity = shifted
    out.name = spectrum.name + "_dopado"
    out.history.append(
        f"synthetic_doping(delta_G={delta_g:+g}, delta_2D={delta_2d:+g})"
    )
    return out


__all__ = [
    "DEMO_KINDS",
    "TMD_DEMOS",
    "TMD_OXIDE_DEMOS",
    "ECHEM_DEMOS",
    "XRD_DEMOS",
    "add_doping",
    "demo_spectra",
    "make_demo",
    "make_tmd_demo",
    "cv_rate_series",
    "make_cv_demo",
    "make_eis_demo",
    "make_gcd_demo",
    "make_lsv_demo",
    "make_xrd_demo",
    "tmd_demo_spectra",
    "xrd_demo_spectra",
]


#: Synthetic TMD samples, as ``(material, layers)``.
TMD_DEMOS = (
    ("MoS2", "1"),
    ("MoS2", "2"),
    ("MoS2", "bulk"),
    ("WS2", "1"),
    ("MoSe2", "1"),
    ("WSe2", "2"),
)

#: Synthetic oxide/chalcogenide composites, as ``(material, layers, oxide)``.
#: These are the samples the heterostructure module exists for: the oxide
#: bands land partly on top of the dichalcogenide's own modes, and only the
#: signature lines above 400 cm⁻¹ tell them apart.
TMD_OXIDE_DEMOS = (
    ("MoSe2", "bulk", "MoO3_alpha"),
    ("MoSe2", "bulk", "MoO2"),
    ("MoS2", "bulk", "MoO3_alpha"),
)


def make_tmd_demo(
    material: str = "MoS2",
    layers: str = "1",
    laser_nm: float = 532.0,
    seed: int = 0,
    low: float = 100.0,
    high: float = 800.0,
    step: float = 0.5,
    noise: float = 3.0,
    phase_1t: bool = False,
    oxide: str | None = None,
    oxide_scale: float = 0.45,
    interface_shift: float = 0.0,
) -> Spectrum:
    """Build a synthetic dichalcogenide spectrum.

    Positions come from ``database/data/tmd.json``, and the layer count is
    imposed by placing the two main modes at a separation drawn from that
    material's own table — so a demo labelled "2 layers" really does have
    the separation the database says two layers have, and analysing it is a
    genuine round trip rather than a tautology.

    Bands are narrow here (3 cm⁻¹) as real dichalcogenide modes are, which
    is why the default sampling step is 0.5 cm⁻¹ rather than the 1 cm⁻¹
    used for carbon: at 1 cm⁻¹ these bands are only three points wide.

    Parameters
    ----------
    material:
        ``"MoS2"``, ``"WS2"``, ``"MoSe2"``, ``"WSe2"`` or ``"MoTe2"``.
    layers:
        ``"1"``, ``"2"``, ``"3"``, ``"4"`` or ``"bulk"``, where the
        material's table has an entry for it.
    laser_nm, seed, low, high, step, noise:
        As for :func:`make_demo`.
    phase_1t:
        Add the J1–J3 modes of the metallic 1T′ phase.
    oxide:
        Key of an oxide from ``tmd.json`` to superimpose, e.g.
        ``"MoO3_alpha"``. Its lines are scaled by the catalogued relative
        weights so the signature bands dominate, as they do in practice.
    oxide_scale:
        Height of the oxide's strongest line relative to the host's.
    interface_shift:
        Displacement in cm⁻¹ applied to the host's out-of-plane A₁g mode
        only, simulating charge transfer across an interface. Used to
        exercise the strain-versus-doping logic.

    Returns
    -------
    Spectrum

    Raises
    ------
    ValueError
        If the material is unknown.
    """
    from ..analysis.tmd import load_tmd_database

    payload, materials = load_tmd_database()
    catalogue = {m.key: m for m in materials}
    if material not in catalogue:
        raise ValueError(
            f"unknown TMD {material!r}; available: {', '.join(sorted(catalogue))}"
        )
    entry = catalogue[material]
    rng = np.random.default_rng(seed)
    x = np.arange(float(low), float(high), float(step))
    y = np.zeros_like(x)

    table = entry.separation_by_layers
    if table and layers in table:
        wanted = 0.5 * sum(table[layers])
        e2g = entry.modes.get("E2g")
        a1g = entry.modes.get("A1g")
        if e2g and a1g:
            # Split the required change symmetrically about the catalogue
            # positions, which is what stacking actually does: E2g softens
            # and A1g stiffens by comparable amounts.
            nominal = abs(a1g.position - e2g.position)
            delta = 0.5 * (wanted - nominal)
            sign = 1.0 if a1g.position > e2g.position else -1.0
            y += lorentzian(x, e2g.position - sign * delta, 620.0, 3.2)
            y += lorentzian(x, a1g.position + sign * delta, 900.0, 3.0)
    for key, mode in entry.modes.items():
        if key in {"E2g", "A1g"} and table and layers in table:
            continue
        if key == "B12g" and layers == "1":
            continue  # forbidden in a monolayer, which is the point
        y += lorentzian(x, mode.position, 220.0 if key != "E2g_A1g" else 900.0, 3.5)

    if phase_1t:
        for line in payload["phases"]["1T_prime"]["marker_bands"].get(material, ()):
            y += lorentzian(x, line, 260.0, 5.0)

    if interface_shift:
        a1g = entry.modes.get("A1g")
        if a1g is not None:
            # Move the out-of-plane mode alone. That asymmetry is the whole
            # point: charge transfer moves A1g and leaves E2g where it was,
            # while biaxial strain moves both.
            centre = a1g.position
            y -= lorentzian(x, centre, 900.0, 3.0)
            y += lorentzian(x, centre + interface_shift, 900.0, 3.0)

    if oxide is not None:
        catalogue_oxides = {o["key"]: o for o in payload.get("oxides", [])}
        if oxide not in catalogue_oxides:
            raise ValueError(
                f"unknown oxide {oxide!r}; available: "
                + ", ".join(sorted(catalogue_oxides))
            )
        entry_oxide = catalogue_oxides[oxide]
        strong = set(entry_oxide.get("strong", ()))
        peak_height = 900.0 * float(oxide_scale)
        broad = entry_oxide.get("min_fwhm")
        for line in entry_oxide["bands"]:
            weight = 1.0 if line in strong else 0.28
            width = float(broad) if broad else 8.0
            y += lorentzian(x, float(line), peak_height * weight, width)

    y += 120.0 * np.exp(-(x - low) / 400.0) + 20.0
    y += rng.normal(0.0, noise, x.size)
    return Spectrum(
        shift=x,
        intensity=y,
        laser_nm=laser_nm,
        name=(
            f"demo_{oxide.split('_')[0]}@{material}_{laser_nm:g}nm"
            if oxide
            else f"demo_{material}_{layers}capa_{laser_nm:g}nm"
        ),
        metadata={
            "synthetic": True,
            "material": material,
            "layers": layers,
            **({"oxide": oxide} if oxide else {}),
        },
    )


def tmd_demo_spectra(laser_nm: float = 532.0, seed: int = 0) -> list[Spectrum]:
    """One synthetic spectrum for each entry in :data:`TMD_DEMOS` and
    :data:`TMD_OXIDE_DEMOS`.

    The composites are generated over a wider range (up to 1100 cm⁻¹)
    because the lines that identify an oxide unambiguously — 819 and 995 of
    MoO₃, 744 of MoO₂ — lie above where a dichalcogenide measurement
    usually stops.
    """
    out = [
        make_tmd_demo(material, layers, laser_nm=laser_nm, seed=seed + i)
        for i, (material, layers) in enumerate(TMD_DEMOS)
    ]
    out.extend(
        make_tmd_demo(
            material,
            layers,
            laser_nm=laser_nm,
            seed=seed + 100 + i,
            high=1100.0,
            oxide=oxide,
            interface_shift=1.8 if oxide == "MoO2" else 0.0,
        )
        for i, (material, layers, oxide) in enumerate(TMD_OXIDE_DEMOS)
    )
    return out


# -- diffraction ------------------------------------------------------

#: Synthetic diffractograms, as ``(name, [(phase, scale)], texture axis)``.
#: Chosen to exercise the cases that go wrong: a mixture whose phases
#: overlap, a layered material with severe preferred orientation, an
#: iron-bearing sample that would fluoresce under a copper tube, and a
#: pattern with a phase deliberately left out of the reference library so
#: that something is genuinely unexplained.
XRD_DEMOS: tuple[tuple[str, tuple[tuple[str, float], ...], object], ...] = (
    ("CNT_FeSe", (("grafito_2H", 0.55), ("FeSe_tetragonal", 1.0),
                  ("Se_trigonal", 0.18)), None),
    ("CNT_Fe3O4", (("grafito_2H", 1.0), ("Fe3O4_magnetita", 0.35),
                   ("Fe_alfa", 0.10)), None),
    ("MoS2_texturado", (("MoS2_2H", 1.0),), (0, 0, 1)),
    ("FeSe_dos_fases", (("FeSe_tetragonal", 1.0), ("FeSe_hexagonal", 0.45)), None),
)


def make_xrd_demo(
    kind: str = "CNT_FeSe",
    two_theta_range: tuple[float, float] = (10.0, 80.0),
    step: float = 0.02,
    background: float = 260.0,
    counts_at_max: float = 15000.0,
    seed: int = 0,
    texture_r: float = 0.65,
    fwhm: float = 0.09,
):
    """Build a synthetic powder pattern from the bundled structures.

    Calculated from real crystal structures rather than drawn as a set of
    Gaussians, so analysing one is a genuine round trip: the lattice
    parameters that come out of a refinement are the ones that went in,
    and any disagreement is a bug rather than a mismatch of conventions.

    The noise is genuinely Poisson, because everything downstream —
    detection thresholds, refinement weights, the goodness of fit — is
    built on σ = √N and calibrating it against anything else calibrates it
    against the wrong thing.

    Parameters
    ----------
    kind:
        One of the names in :data:`XRD_DEMOS`.
    two_theta_range, step:
        Scan range and step in degrees.
    background, counts_at_max:
        Background level and peak height, both in counts.
    texture_r:
        March–Dollase parameter applied when the demo declares a texture
        axis. 0.65 is severe but entirely ordinary for a layered powder
        pressed into a front-loaded holder.
    fwhm:
        Peak width in degrees at low angle, through the Caglioti W term.

    Returns
    -------
    ramancarbon.xrd.pattern.Pattern
    """
    from ..xrd.powder import Profile, simulate
    from ..xrd.reference import find_phase

    entries = {name: (phases, axis) for name, phases, axis in XRD_DEMOS}
    if kind not in entries:
        raise ValueError(
            f"unknown XRD demo {kind!r}; available: {', '.join(sorted(entries))}"
        )
    phases, axis = entries[kind]
    crystals = []
    scales = []
    for name, scale in phases:
        crystal = find_phase(name)
        if crystal is None:  # pragma: no cover - bundled library is complete
            raise ValueError(f"falta la fase de referencia {name!r}")
        crystals.append(crystal)
        scales.append(scale)

    pattern = simulate(
        crystals,
        two_theta_range=two_theta_range,
        step=step,
        scales=scales,
        profile=Profile(u=0.006, v=-0.001, w=fwhm**2, eta0=0.55),
        background=background,
        counts_at_max=counts_at_max,
        preferred_axis=axis,
        preferred_r=texture_r if axis is not None else 1.0,
        seed=seed,
        name=f"demo_drx_{kind}",
    )
    pattern.metadata.update(
        {
            "synthetic": True,
            "phases": [name for name, _ in phases],
            "texture_axis": axis,
            "warning": (
                "Difractograma calculado, no medido. Sirve para probar el "
                "programa; no son datos experimentales."
            ),
        }
    )
    return pattern


def xrd_demo_spectra(seed: int = 0):
    """One synthetic diffractogram for each entry in :data:`XRD_DEMOS`."""
    return [
        make_xrd_demo(name, seed=seed + i)
        for i, (name, _, _) in enumerate(XRD_DEMOS)
    ]


# -- electrochemistry -------------------------------------------------

#: Synthetic electrodes, as ``(name, mechanism)``. The three mechanisms
#: are the ones the classifier has to tell apart, and they are generated
#: from different physics rather than from the same curve with different
#: parameters: the capacitor from a constant differential capacitance, the
#: pseudocapacitor from broad surface redox, the battery from a narrow
#: two-phase plateau.
ECHEM_DEMOS = ("condensador", "pseudocondensador", "bateria")


def _cv_current(
    potential, scan_rate: float, kind: str, capacitance: float, rng
):
    """Current of one synthetic electrode over a potential sweep."""
    direction = np.sign(np.gradient(potential))
    direction[direction == 0] = 1.0
    current = capacitance * scan_rate * direction

    if kind == "pseudocondensador":
        # Broad, strongly overlapping surface redox on top of the double
        # layer: the peaks are there but they do not resolve.
        for centre, height, width, sign in (
            (0.35, 2.2, 0.11, 1.0),
            (0.28, 2.2, 0.11, -1.0),
        ):
            peak = np.exp(-0.5 * ((potential - centre) / width) ** 2)
            current += sign * height * capacitance * scan_rate * peak * (direction == sign)
    elif kind == "bateria":
        # A narrow pair of redox peaks, well separated: a phase transition.
        # The current scales as sqrt(scan rate) because it is diffusion
        # limited, which is what makes b come out near 0.5.
        diffusive = math.sqrt(scan_rate / 0.005)
        for centre, sign in ((0.42, 1.0), (0.30, -1.0)):
            peak = np.exp(-0.5 * ((potential - centre) / 0.022) ** 2)
            current += (
                sign * 9.0 * capacitance * 0.005 * diffusive * peak * (direction == sign)
            )
    return current


def make_cv_demo(
    kind: str = "condensador",
    scan_rate: float = 0.02,
    window: tuple[float, float] = (0.0, 0.6),
    capacitance: float = 0.05,
    cycles: int = 3,
    points: int = 600,
    noise: float = 2e-6,
    resistance: float = 2.0,
    seed: int = 0,
):
    """Build a synthetic cyclic voltammogram.

    ``capacitance`` is the differential capacitance in farads that the
    double-layer part is generated from, so a round trip through
    :func:`~ramancarbon.echem.cv.capacitance` must return it — which is
    what makes the demo a test rather than a picture.

    ``resistance`` is a series resistance in ohms, applied as a distortion
    of the potential axis, so the iR correction has something real to
    correct.
    """
    from ..echem.curve import Electrode, Voltammogram

    if kind not in ECHEM_DEMOS:
        raise ValueError(
            f"unknown demo electrode {kind!r}; available: {', '.join(ECHEM_DEMOS)}"
        )
    rng = np.random.default_rng(seed)
    low, high = window
    ramp = np.linspace(low, high, points // 2)
    one = np.concatenate([ramp, ramp[::-1]])
    potential = np.tile(one, cycles)
    cycle = np.repeat(np.arange(1, cycles + 1), one.size)

    current = _cv_current(potential, scan_rate, kind, capacitance, rng)
    current = current + rng.normal(0.0, noise, current.size)
    measured = potential + resistance * current

    return Voltammogram(
        potential=measured,
        current=current,
        scan_rate=scan_rate,
        electrode=Electrode(
            mass_mg=2.0, area_cm2=1.0, reference="Ag/AgCl_3M", ph=14.0,
            resistance_ohm=resistance, label=f"demo {kind}",
        ),
        cycle=cycle,
        name=f"demo_cv_{kind}_{1e3 * scan_rate:g}mVs",
        metadata={
            "synthetic": True,
            "mechanism": kind,
            "true_capacitance_f": capacitance,
            "true_resistance_ohm": resistance,
        },
    )


def cv_rate_series(
    kind: str = "condensador",
    rates: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    seed: int = 0,
):
    """One synthetic voltammogram per scan rate, for a rate study."""
    return [
        make_cv_demo(kind, scan_rate=rate, seed=seed + i)
        for i, rate in enumerate(rates)
    ]


def make_gcd_demo(
    kind: str = "condensador",
    current: float = 1e-3,
    capacitance: float = 0.05,
    window: tuple[float, float] = (0.0, 0.6),
    resistance: float = 2.0,
    cycles: int = 3,
    points_per_branch: int = 300,
    efficiency: float = 0.98,
    noise: float = 3e-5,
    seed: int = 0,
):
    """Build a synthetic galvanostatic charge–discharge curve.

    The IR jump is put in explicitly as ``I·R`` at every current reversal,
    so removing it is a real correction and not a formality, and the
    coulombic efficiency is imposed by shortening the discharge.
    """
    from ..echem.curve import ChargeDischarge, Electrode

    if kind not in ECHEM_DEMOS:
        raise ValueError(f"unknown demo electrode {kind!r}")
    rng = np.random.default_rng(seed)
    low, high = window
    span = high - low
    charge_time = capacitance * span / current

    times, potentials, currents, labels = [], [], [], []
    clock = 0.0
    for cycle in range(1, cycles + 1):
        for direction in (1.0, -1.0):
            duration = charge_time * (1.0 if direction > 0 else efficiency)
            t = np.linspace(0.0, duration, points_per_branch)
            fraction = t / charge_time
            if kind == "condensador":
                shape = fraction * span
            elif kind == "pseudocondensador":
                # Slight curvature: the differential capacitance rises in
                # the middle of the window.
                shape = span * (fraction - 0.10 * np.sin(2 * np.pi * fraction) / (2 * np.pi))
            else:
                # A plateau: most of the charge enters at one potential, and
                # NOT at the middle of the window. Real battery plateaus are
                # off-centre (LiFePO4 sits at 3.4 V in a 2.5-4.0 V window),
                # and that asymmetry is what makes 1/2 CV^2 wrong for them:
                # for a symmetric plateau it happens to give the right
                # answer, so a centred demo would hide the error.
                centre = 0.32
                shape = span * (
                    0.18 * fraction
                    + 0.82 / (1.0 + np.exp(-(fraction - centre) / 0.06))
                    - 0.82 / (1.0 + np.exp(centre / 0.06))
                )
                shape = shape / shape[-1] * span
            base = low + shape if direction > 0 else high - shape * (span / max(shape[-1], 1e-12)) * 0 - shape
            if direction < 0:
                base = high - shape
            base = base - direction * current * resistance
            times.append(clock + t)
            potentials.append(base)
            currents.append(np.full(t.size, direction * current))
            labels.append(np.full(t.size, cycle))
            clock += duration

    time = np.concatenate(times)
    potential = np.concatenate(potentials) + rng.normal(0.0, noise, time.size)
    return ChargeDischarge(
        time=time,
        potential=potential,
        current=np.concatenate(currents),
        electrode=Electrode(
            mass_mg=2.0, area_cm2=1.0, reference="Ag/AgCl_3M", ph=14.0,
            label=f"demo {kind}",
        ),
        cycle=np.concatenate(labels),
        name=f"demo_gcd_{kind}",
        metadata={
            "synthetic": True,
            "mechanism": kind,
            "true_capacitance_f": capacitance,
            "true_resistance_ohm": resistance,
            "true_efficiency": efficiency,
        },
    )


def make_eis_demo(
    circuit: str = "R0-(R1|Q1)-Q2",
    values: Optional[dict] = None,
    frequencies: tuple[float, float] = (0.01, 1e5),
    points: int = 60,
    noise_pct: float = 0.5,
    seed: int = 0,
):
    """Build a synthetic impedance spectrum from a known circuit.

    Generated from the same circuit code the fitter uses, so a round trip
    recovers the parameters that went in — which makes the demo a test of
    the fit rather than of two independent implementations agreeing.
    """
    from ..echem.curve import Electrode, Impedance
    from ..echem.eis import LIBRARY, parse_circuit

    text = LIBRARY.get(circuit, circuit)
    tree = parse_circuit(text)
    defaults = {
        "R0.R": 5.0, "R1.R": 40.0, "Q1.Q": 2e-4, "Q1.n": 0.88,
        "Q2.Q": 0.02, "Q2.n": 0.92, "C1.C": 1e-4, "W1.sigma": 25.0,
        "Wo1.R": 60.0, "Wo1.tau": 12.0, "L0.L": 1e-6,
    }
    chosen = {**defaults, **(values or {})}
    for element in tree.elements():
        for index, label in enumerate(element.labels):
            if label in chosen:
                element.values[index] = float(chosen[label])

    frequency = np.logspace(
        math.log10(frequencies[1]), math.log10(frequencies[0]), points
    )
    z = tree.impedance(2.0 * math.pi * frequency)
    rng = np.random.default_rng(seed)
    scale = noise_pct / 100.0
    z = z * (1.0 + rng.normal(0.0, scale, z.size)) + 1j * np.abs(z) * rng.normal(
        0.0, scale, z.size
    )
    return Impedance(
        frequency=frequency,
        z=z,
        electrode=Electrode(mass_mg=2.0, area_cm2=1.0),
        name=f"demo_eis_{text}",
        metadata={"synthetic": True, "circuit": text, "true_values": chosen},
    )


def make_lsv_demo(
    reaction: str = "OER",
    tafel_mv_per_decade: float = 60.0,
    exchange_ma_cm2: float = 1e-5,
    resistance: float = 3.0,
    area_cm2: float = 1.0,
    points: int = 600,
    limit_ma_cm2: float = 200.0,
    noise: float = 0.02,
    seed: int = 0,
):
    """A synthetic linear-sweep polarisation curve with a known Tafel slope.

    Built from the Tafel law itself, ``η = b·log₁₀(j/j₀)``, so a round trip
    through :func:`~ramancarbon.echem.evaluate.tafel_analysis` has to give
    back the slope that went in — which makes the demo a test of the
    fitting, not an illustration of it.

    A mass-transport limit and a series resistance are added on top,
    because both are what make a real Tafel fit hard: the first curves the
    line over at high current and the second tilts it.
    """
    from ..echem.curve import Electrode, Voltammogram

    rng = np.random.default_rng(seed)
    slope = tafel_mv_per_decade * 1e-3
    eta = np.linspace(0.02, 0.42, points)
    kinetic = exchange_ma_cm2 * 10.0 ** (eta / slope)
    # Koutecky-Levich style saturation towards the transport limit.
    density = 1.0 / (1.0 / kinetic + 1.0 / limit_ma_cm2)
    density = density * (1.0 + rng.normal(0.0, noise, density.size))

    equilibrium = 1.23 if reaction == "OER" else 0.0
    sign = 1.0 if reaction == "OER" else -1.0
    rhe = equilibrium + sign * eta
    current = sign * density * 1e-3 * area_cm2
    measured = rhe + resistance * current   # uncompensated ohmic drop

    return Voltammogram(
        potential=measured,
        current=current,
        scan_rate=0.005,
        electrode=Electrode(
            mass_mg=0.2, area_cm2=area_cm2, reference="RHE",
            resistance_ohm=resistance, label=f"demo {reaction}",
        ),
        name=f"demo_lsv_{reaction}",
        metadata={
            "synthetic": True,
            "reaction": reaction,
            "true_tafel_mv_dec": tafel_mv_per_decade,
            "true_j0_mA_cm2": exchange_ma_cm2,
            "true_resistance_ohm": resistance,
        },
    )


#: Named synthetic maps, each built to exercise one thing a map does.
MAP_DEMOS = ("dos_fases", "escalon", "hojuela")


def make_map_demo(
    kind: str = "dos_fases",
    rows: int = 20,
    columns: int = 24,
    step_um: float = 0.5,
    laser_nm: float = 532.0,
    seed: int = 0,
    low: float = 1100.0,
    high: float = 2900.0,
    spectral_step: float = 2.0,
    noise: float = 3.0,
    fluorescence_gradient: float = 400.0,
    cosmic_rays: int = 4,
    missing: int = 3,
):
    """Build a synthetic Raman map.

    Everything that makes a real map hard is in here on purpose, because
    the analysis has to survive all of it and a clean cube proves nothing:

    - a **fluorescence gradient across the sample**, which is what turns a
      raw band area into a picture of the background rather than of the
      material;
    - **cosmic rays**, single-pixel single-channel spikes that dominate
      the variance of the whole cube and therefore the first principal
      component, if they are not removed;
    - **missing pixels**, because scans get aborted;
    - a **spatial structure that is not aligned to the grid**, so that a
      cluster map has something to find that is not the raster.

    ``kind`` chooses the structure: ``dos_fases`` is two materials meeting
    along a diagonal, ``escalon`` is monolayer against bilayer graphene
    (the 2D band changes shape, not just size), and ``hojuela`` is a flake
    on a bare substrate, which is the case where most of the map is
    nothing and every ratio there is noise.
    """
    from ..mapping.cube import RamanMap

    if kind not in MAP_DEMOS:
        raise ValueError(
            f"mapa desconocido: {kind!r}; hay: {', '.join(MAP_DEMOS)}")

    rng = np.random.default_rng(seed)
    x_um = np.arange(columns, dtype=float) * step_um
    y_um = np.arange(rows, dtype=float) * step_um
    shift = np.arange(float(low), float(high), float(spectral_step))
    cube = np.zeros((rows, columns, shift.size))

    grid_x, grid_y = np.meshgrid(x_um, y_um)
    span_x = max(x_um[-1], 1e-9)
    span_y = max(y_um[-1], 1e-9)

    if kind == "dos_fases":
        # A diagonal boundary, deliberately not along a row or a column.
        fraction = np.clip(
            0.5 + 2.0 * ((grid_x / span_x) + (grid_y / span_y) - 1.0), 0.0, 1.0)
        first = _map_component(shift, [(1350.0, 40.0, 55.0),
                                       (1580.0, 480.0, 22.0),
                                       (2700.0, 210.0, 45.0)])
        second = _map_component(shift, [(1350.0, 300.0, 90.0),
                                        (1590.0, 420.0, 60.0),
                                        (2700.0, 60.0, 120.0)])
        cube = (fraction[..., None] * second + (1 - fraction)[..., None] * first)
        truth = {"componentes": 2, "frontera": "diagonal"}

    elif kind == "escalon":
        # Monolayer on the left, bilayer on the right: the 2D band goes
        # from one narrow line to a wider, weaker one. Amplitude alone
        # does not separate them; the shape does.
        bilayer = (grid_x > span_x * 0.45).astype(float)
        mono = _map_component(shift, [(1580.0, 300.0, 14.0),
                                      (2680.0, 900.0, 26.0)])
        bi = _map_component(shift, [(1582.0, 320.0, 15.0),
                                    (2690.0, 420.0, 52.0)])
        cube = bilayer[..., None] * bi + (1 - bilayer)[..., None] * mono
        truth = {"componentes": 2, "frontera": "vertical"}

    else:   # hojuela
        centre_x, centre_y = span_x * 0.45, span_y * 0.55
        radius = min(span_x, span_y) * 0.32
        distance = np.hypot(grid_x - centre_x, grid_y - centre_y)
        flake = 1.0 / (1.0 + np.exp((distance - radius) / (0.6 * step_um)))
        material = _map_component(shift, [(1350.0, 120.0, 60.0),
                                          (1580.0, 500.0, 30.0),
                                          (2700.0, 240.0, 60.0)])
        cube = flake[..., None] * material
        truth = {"componentes": 1, "cobertura": float(np.mean(flake > 0.5))}

    # A fluorescence background that varies across the sample. This is the
    # thing that makes an uncorrected area map useless.
    slope = (grid_x / span_x + 0.4 * grid_y / span_y) / 1.4
    background = fluorescence_gradient * (0.3 + slope)[..., None] * np.exp(
        -(shift - shift[0]) / 1800.0)
    cube = cube + background + 120.0

    cube = cube + rng.normal(0.0, noise, size=cube.shape)

    for _ in range(int(cosmic_rays)):
        row = int(rng.integers(rows))
        column = int(rng.integers(columns))
        channel = int(rng.integers(2, shift.size - 2))
        cube[row, column, channel] += rng.uniform(3000.0, 9000.0)

    dropped = []
    for _ in range(int(missing)):
        row = int(rng.integers(rows))
        column = int(rng.integers(columns))
        cube[row, column] = np.nan
        dropped.append((row, column))

    return RamanMap(
        shift=shift, intensity=cube, x=x_um, y=y_um, laser_nm=laser_nm,
        spot_um=1.0, name=f"demo_mapa_{kind}",
        metadata={"synthetic": True, "kind": kind, "rayos_cosmicos": cosmic_rays,
                  "pixeles_perdidos": dropped, **truth},
    )


def _map_component(shift: np.ndarray, bands) -> np.ndarray:
    """A spectrum from ``(centre, height, fwhm)`` triples, as Lorentzians."""
    out = np.zeros_like(shift)
    for centre, height, fwhm in bands:
        half = fwhm / 2.0
        out = out + height * half**2 / ((shift - centre) ** 2 + half**2)
    return out


def make_plateau_gcd_demo(
    plateaus: tuple[tuple[float, float, float], ...] = (
        (3.62, 0.020, 0.9), (3.20, 0.035, 0.6)),
    window: tuple[float, float] = (2.90, 4.10),
    baseline: float = 0.25,
    current: float = -1e-3,
    points: int = 2000,
    seed: int = 0,
    noise_v: float = 2e-4,
):
    """A galvanostatic curve built FROM its differential capacity.

    The honest way to test dQ/dV: rather than drawing a voltage curve that
    looks as though it has plateaus, this specifies dQ/dV directly as a
    baseline plus Gaussian peaks at named potentials, integrates it to get
    Q(V), and inverts that at constant current to get V(t). Whatever comes
    out of a dQ/dV analysis can then be compared with the peaks that went
    in, which is a round trip rather than an impression.

    ``plateaus`` are ``(potential, width in V, relative height)``.
    """
    from ..echem.curve import ChargeDischarge, Electrode

    rng = np.random.default_rng(seed)
    low, high = float(min(window)), float(max(window))
    grid = np.linspace(high, low, 4000)

    dq_dv = np.full_like(grid, float(baseline))
    for centre, width, height in plateaus:
        dq_dv = dq_dv + height * np.exp(-0.5 * ((grid - centre) / width) ** 2)

    # Q(V) by integrating downwards from the top of the window.
    charge = np.concatenate([[0.0], np.cumsum(
        0.5 * (dq_dv[1:] + dq_dv[:-1]) * np.abs(np.diff(grid)))])
    total = float(charge[-1])
    time_total = total * 3600.0 / abs(current)

    time = np.linspace(0.0, time_total, int(points))
    wanted = time * abs(current) / 3600.0
    potential = np.interp(wanted, charge, grid)
    potential = potential + rng.normal(0.0, noise_v, potential.shape)

    return ChargeDischarge(
        time=time, potential=potential,
        current=np.full_like(time, float(current)),
        electrode=Electrode(mass_mg=5.0, area_cm2=1.0, label="demo mesetas"),
        name="demo_gcd_mesetas",
        metadata={
            "synthetic": True,
            "true_plateaus_v": tuple(p[0] for p in plateaus),
            "true_capacity_ah": total,
        },
    )
