"""Maps: the cube, its files, per-pixel images and chemometrics."""

from __future__ import annotations

import numpy as np
import pytest

from ramancarbon.core.baseline import snip_baseline
from ramancarbon.core.spectrum import Spectrum
from ramancarbon.examples.demo_data import MAP_DEMOS, make_demo, make_map_demo
from ramancarbon.mapping.chemometrics import (
    despike_map,
    kmeans,
    mcr_als,
    pca,
    suggested_components,
)
from ramancarbon.mapping.cube import MapError, RamanMap, from_spectra
from ramancarbon.mapping.images import (
    band_intensity,
    band_position,
    band_ratio,
    band_width,
    coverage,
    noise_level,
    signal_to_noise,
)
from ramancarbon.mapping.io import read_map, write_map


@pytest.fixture(scope="module")
def dos_fases():
    return make_map_demo("dos_fases", seed=1)


@pytest.fixture(scope="module")
def limpio(dos_fases):
    return despike_map(dos_fases)[0]


# -- the cube ----------------------------------------------------------

def test_a_map_knows_its_own_shape(dos_fases):
    rows, columns, points = dos_fases.shape
    assert (rows, columns) == (20, 24)
    assert points == dos_fases.shift.size
    assert dos_fases.n_pixels == rows * columns


def test_the_axes_have_to_match_the_cube():
    with pytest.raises(MapError, match="ejes"):
        RamanMap(shift=np.arange(10.0), intensity=np.zeros((3, 4, 9)),
                 x=np.arange(4.0), y=np.arange(3.0))


def test_something_that_is_not_a_cube_is_refused():
    with pytest.raises(MapError, match="dimensiones"):
        RamanMap(shift=np.arange(10.0), intensity=np.zeros((3, 10)),
                 x=np.arange(3.0), y=np.arange(1.0))


def test_a_reversed_spectral_axis_is_put_back_in_order():
    shift = np.arange(200.0, 100.0, -10.0)
    cube = np.tile(np.arange(10.0), (2, 2, 1))
    ordered = RamanMap(shift=shift, intensity=cube,
                       x=np.arange(2.0), y=np.arange(2.0))
    assert np.all(np.diff(ordered.shift) > 0)
    assert ordered.intensity[0, 0, 0] == 9.0


def test_missing_pixels_are_known_and_not_averaged_in(dos_fases):
    assert np.count_nonzero(dos_fases.missing) >= 1
    mean = dos_fases.mean_spectrum()
    assert np.all(np.isfinite(mean.intensity))


def test_a_pixel_comes_out_as_a_normal_spectrum(dos_fases):
    row, column = map(int, np.argwhere(~dos_fases.missing)[0])
    spectrum = dos_fases.spectrum_at(row, column)
    assert isinstance(spectrum, Spectrum)
    assert spectrum.laser_nm == dos_fases.laser_nm
    assert spectrum.metadata["fila"] == row


def test_asking_for_a_missing_pixel_says_so(dos_fases):
    row, column = np.argwhere(dos_fases.missing)[0]
    with pytest.raises(MapError, match="no tiene espectro"):
        dos_fases.spectrum_at(int(row), int(column))


def test_the_nearest_pixel_to_a_position(dos_fases):
    row, column = dos_fases.nearest(dos_fases.x[5] + 0.1, dos_fases.y[2] - 0.1)
    assert (row, column) == (2, 5)


def test_oversampling_is_reported_because_it_makes_statistics_meaningless():
    cube = make_map_demo("dos_fases", step_um=0.2)
    cube.spot_um = 1.0
    warning = cube.sampling_warning()
    assert warning and "óptica" in warning


def test_an_undeclared_spot_size_is_said_rather_than_guessed(dos_fases):
    cube = make_map_demo("dos_fases")
    cube.spot_um = None
    assert "no está declarado" in cube.sampling_warning()


def test_flattening_and_unflattening_puts_values_back_where_they_were(dos_fases):
    matrix, indices = dos_fases.flat()
    assert matrix.shape[0] == indices.shape[0]
    assert matrix.shape[0] == dos_fases.n_pixels - np.count_nonzero(dos_fases.missing)
    image = dos_fases.unflatten(matrix[:, 0], indices)
    for (row, column), value in zip(indices[:5], matrix[:5, 0]):
        assert image[row, column] == value


def test_cropping_keeps_the_grid_and_narrows_the_spectrum(dos_fases):
    narrow = dos_fases.crop(1500.0, 1700.0)
    assert narrow.shape[:2] == dos_fases.shape[:2]
    assert narrow.shift[0] >= 1500.0 and narrow.shift[-1] <= 1700.0
    assert "crop" in narrow.history[-1]


def test_a_window_with_nothing_in_it_is_refused(dos_fases):
    with pytest.raises(MapError, match="menos de dos puntos"):
        dos_fases.crop(10.0, 11.0)


def test_a_region_is_a_smaller_map(dos_fases):
    piece = dos_fases.region(slice(2, 6), slice(3, 9))
    assert piece.shape[:2] == (4, 6)
    assert np.allclose(piece.x, dos_fases.x[3:9])


def test_a_map_can_be_built_from_spectra_with_coordinates():
    spectra, positions = [], []
    for row in range(3):
        for column in range(4):
            spectrum = make_demo("MWCNT", seed=row * 4 + column, low=1200.0,
                                 high=1700.0)
            spectra.append(spectrum)
            positions.append((column * 0.5, row * 0.5))
    cube = from_spectra(spectra, positions, spot_um=1.0)
    assert cube.shape[:2] == (3, 4)
    assert np.allclose(cube.step_um, (0.5, 0.5))


def test_spectra_on_different_axes_cannot_make_a_map():
    a = make_demo("MWCNT", low=1200.0, high=1700.0)
    b = make_demo("MWCNT", low=1200.0, high=1800.0)
    with pytest.raises(MapError, match="eje espectral"):
        from_spectra([a, b], [(0.0, 0.0), (1.0, 0.0)])


def test_positions_that_are_not_a_grid_are_refused_rather_than_snapped():
    """Snapping puts a spectrum in the wrong pixel and the image still
    looks fine, which is why this is an error and not a warning."""
    spectra = [make_demo("MWCNT", low=1200.0, high=1700.0) for _ in range(5)]
    scattered = [(0.0, 0.0), (1.0, 0.3), (0.37, 0.61), (0.9, 0.05), (0.2, 0.8)]
    with pytest.raises(MapError, match="rejilla"):
        from_spectra(spectra, scattered)


@pytest.mark.parametrize("kind", MAP_DEMOS)
def test_every_demo_map_builds_and_describes_itself(kind):
    cube = make_map_demo(kind, rows=6, columns=6)
    assert cube.shape[:2] == (6, 6)
    assert "píxeles" in cube.describe()


# -- files -------------------------------------------------------------

@pytest.mark.parametrize("layout", ["ancho", "largo"])
def test_a_map_survives_a_file_round_trip(tmp_path, layout):
    cube = make_map_demo("escalon", rows=4, columns=5, missing=0, cosmic_rays=0)
    path = write_map(cube, tmp_path / f"mapa_{layout}.txt", layout=layout)
    back = read_map(path, laser_nm=cube.laser_nm)
    assert back.shape == cube.shape
    assert np.allclose(back.shift, cube.shift, atol=1e-3)
    assert np.allclose(back.intensity, cube.intensity, rtol=1e-5)


def test_the_layout_is_detected_not_asked_for(tmp_path):
    cube = make_map_demo("escalon", rows=3, columns=3, missing=0, cosmic_rays=0)
    wide = read_map(write_map(cube, tmp_path / "a.txt", layout="ancho"))
    long = read_map(write_map(cube, tmp_path / "b.txt", layout="largo"))
    assert wide.shape == long.shape == cube.shape


def test_the_laser_is_read_from_the_header(tmp_path):
    cube = make_map_demo("escalon", rows=3, columns=3, missing=0, cosmic_rays=0)
    path = write_map(cube, tmp_path / "m.txt")
    assert read_map(path).laser_nm == 532.0


def test_a_file_with_no_numbers_is_refused(tmp_path):
    path = tmp_path / "vacio.txt"
    path.write_text("# solo comentarios\n# nada más\n", encoding="utf-8")
    with pytest.raises(MapError, match="ninguna fila"):
        read_map(path)


def test_a_missing_file_is_refused_by_name(tmp_path):
    with pytest.raises(MapError, match="no existe"):
        read_map(tmp_path / "no_esta.txt")


# -- per-pixel images --------------------------------------------------

def test_a_band_area_map_has_one_value_per_pixel(dos_fases):
    image = band_intensity(dos_fases, 1500.0, 1660.0)
    assert image.values.shape == dos_fases.shape[:2]
    assert np.all(np.isnan(image.values[dos_fases.missing]))


def test_the_local_baseline_is_what_separates_signal_from_background(dos_fases):
    """The demo has a fluorescence gradient across it on purpose."""
    corrected = band_intensity(dos_fases, 1500.0, 1660.0, baseline=True)
    raw = band_intensity(dos_fases, 1500.0, 1660.0, baseline=False)
    assert np.nanmean(raw.values) > 2 * np.nanmean(corrected.values)
    assert any("fondo" in w for w in raw.warnings)


def test_area_and_height_are_both_offered(dos_fases):
    assert band_intensity(dos_fases, 1500.0, 1660.0, mode="altura").units == "cuentas"
    with pytest.raises(MapError, match="modo desconocido"):
        band_intensity(dos_fases, 1500.0, 1660.0, mode="volumen")


def test_a_window_too_narrow_to_integrate_is_refused(dos_fases):
    with pytest.raises(MapError, match="puntos del espectro"):
        band_intensity(dos_fases, 1580.0, 1581.0)


def test_a_ratio_map_finds_the_boundary_that_was_put_there(dos_fases):
    ratio = band_ratio(dos_fases, (1280.0, 1420.0), (1500.0, 1660.0))
    left = np.nanmean(ratio.values[:, :5])
    right = np.nanmean(ratio.values[:, -5:])
    assert right > 2 * left, "el lado desordenado tiene que dar I_D/I_G mayor"


def test_a_ratio_is_masked_where_its_denominator_is_noise():
    flake = make_map_demo("hojuela", seed=2)
    ratio = band_ratio(flake, (1280.0, 1420.0), (1500.0, 1660.0))
    assert ratio.masked > 0.2 * flake.n_pixels
    assert any("enmascarado" in w for w in ratio.warnings)


def test_a_position_map_is_not_quantised_to_the_spectral_step(dos_fases):
    position = band_position(dos_fases, 1500.0, 1660.0)
    distinct = np.unique(np.round(position.finite, 4))
    step = float(np.median(np.diff(dos_fases.shift)))
    assert distinct.size > 50, "sin interpolación el mapa sale en terrazas"
    assert np.all(np.abs(position.finite - 1585.0) < 30.0)
    assert position.units == "cm⁻¹"
    assert step > 0


def test_a_position_map_finds_the_shift_that_was_put_there(dos_fases):
    """The two phases have their G band 10 cm-1 apart."""
    position = band_position(dos_fases, 1500.0, 1660.0)
    assert np.nanmean(position.values[:, -5:]) > np.nanmean(position.values[:, :5])


def test_a_width_map_says_where_it_does_not_apply(dos_fases):
    width = band_width(dos_fases, 1500.0, 1660.0)
    assert any("solapan" in w for w in width.warnings)
    assert np.nanmedian(width.finite) > 0


def test_noise_is_estimated_per_pixel(dos_fases):
    noise = noise_level(dos_fases)
    assert noise.values.shape == dos_fases.shape[:2]
    assert 1.0 < np.nanmedian(noise.finite) < 8.0


def test_signal_to_noise_and_coverage_agree_about_an_empty_substrate():
    flake = make_map_demo("hojuela", seed=3)
    ratio = signal_to_noise(flake, 1500.0, 1660.0)
    mask, fraction = coverage(flake, 1500.0, 1660.0)
    assert 0.2 < fraction < 0.8
    assert np.nanmean(ratio.values[mask.values == 1]) > \
        np.nanmean(ratio.values[mask.values == 0])


def test_a_property_map_summarises_itself(dos_fases):
    image = band_intensity(dos_fases, 1500.0, 1660.0)
    stats = image.statistics()
    assert stats["pixeles"] > 400
    assert "mediana" in image.describe()


# -- chemometrics ------------------------------------------------------

def test_cosmic_rays_are_removed_and_band_maxima_are_not(dos_fases):
    cleaned, replaced = despike_map(dos_fases)
    injected = dos_fases.metadata["rayos_cosmicos"]
    assert injected <= replaced <= 4 * injected, (
        "un despicado que toca miles de canales está aplanando bandas")
    assert np.nanmax(cleaned.intensity) < 0.5 * np.nanmax(dos_fases.intensity)


def test_despiking_is_recorded_in_the_history(limpio):
    assert any(step.startswith("despike") for step in limpio.history)


def test_pca_on_a_two_phase_map_finds_two_components(limpio):
    result = pca(limpio, components=5)
    assert suggested_components(result) == 2
    assert result.explained[0] > result.explained[1]
    assert result.residual < 0.1


def test_pca_says_a_component_is_not_a_substance(limpio):
    assert any("sustancia" in w for w in pca(limpio, 3).warnings)


def test_an_undespiked_map_is_warned_about(dos_fases):
    assert any("rayos cósmicos" in w for w in pca(dos_fases, 3).warnings)


def test_a_component_comes_out_as_a_spectrum_marked_as_not_a_measurement(limpio):
    spectrum = pca(limpio, 2).component_spectrum(0)
    assert spectrum.shift.size == limpio.shift.size
    assert spectrum.metadata["no_es_una_medida"]


def test_kmeans_separates_the_two_halves(limpio):
    groups = kmeans(limpio, k=2, seed=0)
    labels = groups.labels
    left = labels[:, :6][labels[:, :6] >= 0]
    right = labels[:, -6:][labels[:, -6:] >= 0]
    assert np.bincount(left, minlength=2).argmax() != \
        np.bincount(right, minlength=2).argmax()
    assert groups.silhouette > 0.4


def test_the_cluster_centres_are_in_the_original_units(limpio):
    groups = kmeans(limpio, k=2)
    centre = groups.centre_spectrum(0)
    assert np.nanmax(centre.intensity) > 100.0


def test_clustering_a_map_with_no_structure_says_the_groups_are_arbitrary():
    rng = np.random.default_rng(0)
    shift = np.arange(1200.0, 1800.0, 2.0)
    base = 500.0 * np.exp(-0.5 * ((shift - 1580.0) / 20.0) ** 2) + 100.0
    cube = RamanMap(
        shift=shift,
        intensity=base + rng.normal(0.0, 5.0, size=(10, 10, shift.size)),
        x=np.arange(10.0), y=np.arange(10.0), laser_nm=532.0,
        history=["despike(manual)"],
    )
    groups = kmeans(cube, k=3, seed=0)
    assert groups.silhouette < 0.25
    assert any("nube" in w for w in groups.warnings)


def test_asking_for_one_group_is_refused(limpio):
    with pytest.raises(MapError, match="dos grupos"):
        kmeans(limpio, k=1)


def test_not_normalising_before_clustering_is_warned_about(limpio):
    assert any("brillo" in w for w in kmeans(limpio, k=2, normalise=None).warnings)


def test_mcr_resolves_two_non_negative_spectra(limpio):
    result = mcr_als(limpio, components=2)
    assert result.components.shape == (2, limpio.shift.size)
    assert np.all(result.components >= -1e-9), "una componente negativa no es un espectro"
    assert result.residual < 0.05


def test_mcr_puts_each_component_on_its_own_side(limpio):
    result = mcr_als(limpio, components=2)
    contrasts = [
        np.nanmean(result.scores[:, -6:, i]) - np.nanmean(result.scores[:, :6, i])
        for i in range(2)
    ]
    assert contrasts[0] * contrasts[1] < 0, (
        "las dos componentes tienen que repartirse el mapa, no seguirse")


def test_mcr_admits_the_solution_is_not_unique(limpio):
    warnings = mcr_als(limpio, components=2).warnings
    assert any("no es única" in w.lower() or "NO es única" in w for w in warnings)
    assert any("fracciones másicas" in w for w in warnings)


def test_closure_is_offered_and_its_cost_stated(limpio):
    result = mcr_als(limpio, components=2, closure=True)
    assert any("cierre" in w for w in result.warnings)


# -- SNIP --------------------------------------------------------------

def test_snip_recovers_a_pure_background():
    x = np.arange(100.0, 3000.0, 1.0)
    background = 500.0 * np.exp(-(x - 100.0) / 900.0) + 50.0
    estimated = snip_baseline(background, iterations=100)
    assert np.max(np.abs(estimated - background) / background) < 0.01


def test_snip_leaves_the_peak_alone():
    x = np.arange(100.0, 3000.0, 1.0)
    peak = 900.0 * np.exp(-0.5 * ((x - 1580.0) / 15.0) ** 2)
    background = 500.0 * np.exp(-(x - 100.0) / 900.0) + 50.0
    corrected = (peak + background) - snip_baseline(peak + background,
                                                    iterations=80)
    assert abs(np.max(corrected) - 900.0) / 900.0 < 0.05


def test_snip_is_reachable_by_name():
    from ramancarbon.core.baseline import estimate_baseline

    spectrum = make_demo("MWCNT", fluorescence=600.0)
    assert estimate_baseline(spectrum, method="snip").shape == spectrum.shift.shape


def test_snip_refuses_what_it_cannot_work_on():
    with pytest.raises(ValueError, match="cinco puntos"):
        snip_baseline([1.0, 2.0])
    with pytest.raises(ValueError, match="no finitos"):
        snip_baseline([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
