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
    "add_doping",
    "demo_spectra",
    "make_demo",
    "make_tmd_demo",
    "tmd_demo_spectra",
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

    y += 120.0 * np.exp(-(x - low) / 400.0) + 20.0
    y += rng.normal(0.0, noise, x.size)
    return Spectrum(
        shift=x,
        intensity=y,
        laser_nm=laser_nm,
        name=f"demo_{material}_{layers}capa_{laser_nm:g}nm",
        metadata={"synthetic": True, "material": material, "layers": layers},
    )


def tmd_demo_spectra(laser_nm: float = 532.0, seed: int = 0) -> list[Spectrum]:
    """One synthetic spectrum for each entry in :data:`TMD_DEMOS`."""
    return [
        make_tmd_demo(material, layers, laser_nm=laser_nm, seed=seed + i)
        for i, (material, layers) in enumerate(TMD_DEMOS)
    ]
