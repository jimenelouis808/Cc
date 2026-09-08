"""Finding the components of a map without being told what they are.

A map of forty thousand spectra cannot be looked at. The three methods
here are what people actually use to reduce it to something that can be:

**PCA** answers "how many different things are in here?" It is a rotation,
nothing more — the components are orthogonal by construction and there is
no reason a real chemical component should be orthogonal to another. A
principal component is a direction of variance, not a substance, and
reading loadings as spectra is the most common misuse of this method.

**Clustering** answers "which pixels are alike?" It gives a picture that
looks like a phase map and is not one: k-means always returns exactly the
k clusters it was asked for, whether or not the sample has k phases.

**MCR-ALS** answers "what spectra add up to this?" It is the one that
returns things that can be read as spectra, because it constrains them to
be non-negative. What it cannot give is a unique answer: any pair of
factors that multiply back to the data is a solution, and the rotational
ambiguity is a property of the problem, not of the implementation.

Two preparations are not optional and both are done here rather than
suggested:

**Cosmic rays first.** A single spike in a single pixel is the largest
excursion in the entire cube, so it becomes the first principal component
and its own cluster. Un-despiked map decomposition finds cosmic rays.

**Per-spectrum normalisation before clustering.** Without it, k-means
sorts pixels by brightness, and brightness is set by focus and topography.
The clusters come out as a picture of the surface's height.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.spectrum import Spectrum
from .cube import MapError, RamanMap


@dataclass
class Decomposition:
    """The result of a PCA or an MCR-ALS on a map."""

    components: np.ndarray
    """``(k, n_shift)``: the spectra, or the loadings."""
    scores: np.ndarray
    """``(n_y, n_x, k)``: how much of each component is at each pixel."""
    shift: np.ndarray
    explained: np.ndarray = field(default_factory=lambda: np.zeros(0))
    """Fraction of the variance each component accounts for. Empty for
    methods where it is not defined."""
    method: str = ""
    mean: Optional[np.ndarray] = None
    """The spectrum that was subtracted, for PCA. Needed to reconstruct."""
    warnings: list[str] = field(default_factory=list)
    residual: float = 0.0
    """Fraction of the data not reproduced by the model."""

    @property
    def k(self) -> int:
        return int(self.components.shape[0])

    def component_spectrum(self, index: int, name: str = "") -> Spectrum:
        """One component as a :class:`Spectrum`, for plotting or fitting."""
        return Spectrum(
            shift=self.shift, intensity=self.components[index],
            name=name or f"{self.method} {index + 1}",
            metadata={"metodo": self.method, "componente": index + 1,
                      "no_es_una_medida": True},
        )

    def score_map(self, index: int) -> np.ndarray:
        return self.scores[:, :, index]

    def describe(self) -> str:
        text = f"{self.method}: {self.k} componentes"
        if self.explained.size:
            total = 100 * float(np.sum(self.explained[:self.k]))
            text += f", {total:.1f} % de la varianza"
        text += f", residuo {100 * self.residual:.2f} %"
        return text


@dataclass
class Clustering:
    """Pixels grouped by similarity."""

    labels: np.ndarray
    """``(n_y, n_x)`` integers, −1 where the pixel has no data."""
    centres: np.ndarray
    """``(k, n_shift)``: the mean spectrum of each cluster."""
    shift: np.ndarray
    sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    silhouette: Optional[float] = None
    """Mean silhouette, between −1 and 1. Above ~0.5 the clusters are
    separated; near 0 they are one cloud cut arbitrarily."""
    warnings: list[str] = field(default_factory=list)
    inertia: float = 0.0

    @property
    def k(self) -> int:
        return int(self.centres.shape[0])

    def centre_spectrum(self, index: int) -> Spectrum:
        return Spectrum(shift=self.shift, intensity=self.centres[index],
                        name=f"grupo {index + 1}",
                        metadata={"pixeles": int(self.sizes[index])
                                  if self.sizes.size else None})

    def describe(self) -> str:
        text = f"{self.k} grupos"
        if self.sizes.size:
            text += " (" + ", ".join(f"{n}" for n in self.sizes) + " píxeles)"
        if self.silhouette is not None:
            text += f", silueta {self.silhouette:.2f}"
        return text


# -- preparation --------------------------------------------------------

def despike_map(cube: RamanMap, threshold: float = 8.0,
                window: int = 5, width: int = 3) -> tuple[RamanMap, int]:
    """Remove cosmic rays without clipping the tops of real bands.

    A cosmic ray is a **narrow** excursion — one to three channels — that
    a real Raman band never is. So the test has two parts, and both are
    needed:

    - the channel is far above the median of its own spectral
      neighbourhood, measured in units of **that pixel's own noise**;
    - the excursion is no wider than ``width`` channels.

    The noise is estimated per pixel from the median absolute second
    difference, which kills any smooth curvature. Estimating it locally
    instead — the obvious thing — makes it tiny wherever the spectrum is
    smooth, so the top of every well-sampled band exceeds the threshold
    and gets flattened. That mistake replaced half a percent of all
    channels on a clean synthetic map, most of them band maxima.

    Returns the cleaned map and how many channels were replaced.
    """
    if window % 2 == 0:
        window += 1
    data = cube.intensity
    if data.shape[-1] < window + 2:
        return cube, 0

    import warnings as _warnings

    padded = np.pad(data, ((0, 0), (0, 0), (window // 2, window // 2)),
                    mode="edge")
    stack = np.stack([padded[:, :, i:i + data.shape[2]] for i in range(window)],
                     axis=-1)
    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message="All-NaN")
        median = np.nanmedian(stack, axis=-1)
        # Per-pixel noise: the median absolute second difference, which is
        # blind to any smooth shape and therefore to real bands.
        second = np.diff(np.where(np.isfinite(data), data, np.nan), n=2, axis=-1)
        sigma = np.nanmedian(np.abs(second), axis=-1) / (0.6745 * np.sqrt(6.0))
    sigma = np.maximum(sigma, 1e-9)[..., None]

    excess = data - median
    candidate = np.isfinite(data) & (excess > threshold * sigma)

    # Narrowness: a run of flagged channels longer than `width` is a band
    # the estimate got wrong, not a cosmic ray.
    spikes = candidate & ~_wide_runs(candidate, width)

    cleaned = np.where(spikes, median, data)
    return (
        cube.with_intensity(cleaned, step=f"despike({threshold})"),
        int(np.count_nonzero(spikes)),
    )


def _wide_runs(flags: np.ndarray, width: int) -> np.ndarray:
    """Mark the flagged channels that belong to a run longer than ``width``."""
    if width < 1:
        return np.zeros_like(flags)
    kernel = np.ones(width + 1, dtype=int)
    counts = np.apply_along_axis(
        lambda row: np.convolve(row.astype(int), kernel, mode="same"),
        -1, flags)
    return flags & (counts > width)


def _matrix(cube: RamanMap, normalise: Optional[str]) -> tuple[np.ndarray, np.ndarray]:
    matrix, indices = cube.flat(drop_missing=True)
    if matrix.shape[0] < 3:
        raise MapError("hacen falta al menos tres píxeles con datos")
    matrix = np.where(np.isfinite(matrix), matrix, 0.0)
    if normalise == "area":
        scale = np.sum(np.abs(matrix), axis=1, keepdims=True)
    elif normalise == "max":
        scale = np.max(np.abs(matrix), axis=1, keepdims=True)
    elif normalise in (None, "no", "ninguna"):
        scale = np.ones((matrix.shape[0], 1))
    elif normalise == "vector":
        scale = np.linalg.norm(matrix, axis=1, keepdims=True)
    else:
        raise MapError(f"normalización desconocida: {normalise!r}")
    scale = np.where(scale > 0, scale, 1.0)
    return matrix / scale, indices


# -- PCA ----------------------------------------------------------------

def pca(cube: RamanMap, components: int = 5,
        normalise: Optional[str] = None) -> Decomposition:
    """Principal components of a map.

    The data are **mean-centred** first, which is what makes the result
    principal components rather than a description of the mean spectrum:
    without centring, the first component is just the average and every
    map looks the same.

    Use it to decide how many distinct things are in the map — the point
    where the explained variance stops dropping — not to identify them.
    """
    matrix, indices = _matrix(cube, normalise)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    k = int(min(components, centred.shape[0] - 1, centred.shape[1]))
    if k < 1:
        raise MapError("no hay bastantes píxeles para un análisis de componentes")

    _, singular, right = np.linalg.svd(centred, full_matrices=False)
    loadings = right[:k]
    scores = centred @ loadings.T
    variance = singular ** 2
    explained = variance / max(float(np.sum(variance)), 1e-30)

    reconstruction = scores @ loadings
    residual = float(np.linalg.norm(centred - reconstruction)
                     / max(np.linalg.norm(centred), 1e-30))

    warnings = [
        "una componente principal es una dirección de varianza, no una "
        "sustancia: son ortogonales por construcción y nada obliga a que "
        "dos componentes químicas reales lo sean"
    ]
    warnings += _despike_warning(cube)
    return Decomposition(
        components=loadings, scores=cube.unflatten(scores, indices),
        shift=cube.shift, explained=explained, method="PCA", mean=mean,
        warnings=warnings, residual=residual,
    )


def suggested_components(decomposition: Decomposition,
                         floor: float = 0.01) -> int:
    """How many components carry more than noise.

    The rule is the elbow of the explained variance, taken as the last
    component above ``floor`` of the total. It is a suggestion: the number
    of chemical species is not a property of a singular value.
    """
    explained = decomposition.explained
    if explained.size == 0:
        return decomposition.k
    above = int(np.count_nonzero(explained > floor))
    return max(1, min(above, decomposition.k))


# -- clustering ---------------------------------------------------------

def kmeans(
    cube: RamanMap,
    k: int = 3,
    normalise: Optional[str] = "area",
    seed: int = 0,
    iterations: int = 100,
    silhouette_sample: int = 400,
) -> Clustering:
    """Group pixels by the shape of their spectra.

    ``normalise="area"`` by default and deliberately: on raw intensities
    k-means sorts by brightness, and brightness is set by focus and by how
    much material happens to be under the spot. The clusters then map the
    topography of the sample, which is a real picture of the wrong thing.

    Initialised with k-means++ and run to convergence. The silhouette says
    whether the groups are separated at all — k-means returns exactly k
    groups whether or not the sample has k of anything.
    """
    if k < 2:
        raise MapError("agrupar en menos de dos grupos no dice nada")
    matrix, indices = _matrix(cube, normalise)
    if matrix.shape[0] < k:
        raise MapError(
            f"hay {matrix.shape[0]} píxeles con datos y se piden {k} grupos")

    rng = np.random.default_rng(seed)
    centres = _kmeans_plus_plus(matrix, k, rng)
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(iterations):
        distances = _square_distances(matrix, centres)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for index in range(k):
            members = matrix[labels == index]
            if members.size:
                centres[index] = members.mean(axis=0)
            else:
                centres[index] = matrix[rng.integers(matrix.shape[0])]

    distances = _square_distances(matrix, centres)
    labels = np.argmin(distances, axis=1)
    inertia = float(np.sum(distances[np.arange(labels.size), labels]))
    sizes = np.bincount(labels, minlength=k)

    # The centres in the ORIGINAL units, not the normalised ones: a
    # cluster centre is shown as a spectrum and has to be comparable with
    # the data.
    raw, _ = cube.flat(drop_missing=True)
    raw = np.where(np.isfinite(raw), raw, np.nan)
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.filterwarnings("ignore", message="Mean of empty slice")
        centre_spectra = np.stack([
            np.nanmean(raw[labels == index], axis=0) if sizes[index]
            else np.full(cube.shift.size, np.nan)
            for index in range(k)
        ])

    label_image = cube.unflatten(labels.astype(float), indices)
    label_image = np.where(np.isfinite(label_image), label_image, -1)

    score = _silhouette(matrix, labels, rng, silhouette_sample)
    warnings = _despike_warning(cube)
    if score is not None and score < 0.25:
        warnings.append(
            f"la silueta media es {score:.2f}: los píxeles forman una sola "
            f"nube y los {k} grupos son un corte arbitrario de ella, no fases"
        )
    if np.any(sizes < max(3, 0.01 * matrix.shape[0])):
        warnings.append(
            "algún grupo tiene muy pocos píxeles: suele ser un rayo cósmico "
            "o un borde, no una fase"
        )
    if normalise in (None, "no", "ninguna"):
        warnings.append(
            "sin normalizar, k-medias ordena por brillo, que lo fija el "
            "enfoque y la topografía, no la química"
        )
    return Clustering(
        labels=label_image.astype(int), centres=centre_spectra,
        shift=cube.shift, sizes=sizes, silhouette=score,
        warnings=warnings, inertia=inertia,
    )


def _kmeans_plus_plus(matrix: np.ndarray, k: int, rng) -> np.ndarray:
    centres = [matrix[rng.integers(matrix.shape[0])]]
    for _ in range(1, k):
        # Clipped at zero: the expansion of ‖x−c‖² is exact in algebra and
        # not in floating point, and a distance of −1e−13 makes the draw
        # below refuse the whole probability vector.
        distances = np.maximum(
            np.min(_square_distances(matrix, np.stack(centres)), axis=1), 0.0)
        total = float(np.sum(distances))
        if total <= 0:
            centres.append(matrix[rng.integers(matrix.shape[0])])
            continue
        centres.append(matrix[rng.choice(matrix.shape[0], p=distances / total)])
    return np.stack(centres)


def _square_distances(matrix: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """‖x − c‖² for every point and centre, without building the difference."""
    return (np.sum(matrix ** 2, axis=1)[:, None]
            - 2 * matrix @ centres.T
            + np.sum(centres ** 2, axis=1)[None, :])


def _silhouette(matrix, labels, rng, sample: int) -> Optional[float]:
    """Mean silhouette on a random subsample.

    Subsampled because the exact figure needs every pairwise distance —
    forty thousand pixels is 1.6 billion of them — and the mean over a few
    hundred is the same number to two decimals.
    """
    unique = np.unique(labels)
    if unique.size < 2:
        return None
    size = min(sample, matrix.shape[0])
    chosen = rng.choice(matrix.shape[0], size=size, replace=False)
    subset, sub_labels = matrix[chosen], labels[chosen]
    distances = np.sqrt(np.maximum(_square_distances(subset, subset), 0.0))
    scores = []
    for index in range(size):
        own = sub_labels == sub_labels[index]
        own[index] = False
        if not np.any(own):
            continue
        a = float(np.mean(distances[index, own]))
        others = [float(np.mean(distances[index, sub_labels == other]))
                  for other in unique if other != sub_labels[index]
                  and np.any(sub_labels == other)]
        if not others:
            continue
        b = min(others)
        scores.append((b - a) / max(a, b, 1e-30))
    return float(np.mean(scores)) if scores else None


# -- MCR-ALS ------------------------------------------------------------

def mcr_als(
    cube: RamanMap,
    components: int = 2,
    iterations: int = 200,
    tolerance: float = 1e-6,
    closure: bool = False,
    seed: int = 0,
) -> Decomposition:
    """Multivariate curve resolution by alternating least squares.

    Alternates between solving for the concentrations given the spectra
    and for the spectra given the concentrations, with **both constrained
    to be non-negative**. That constraint is what makes the result
    readable as spectra: a component with negative intensity is not a
    substance, and unconstrained factorisation produces them freely.

    ``closure`` additionally forces the concentrations at each pixel to
    sum to one. Use it only when the components really are the whole of
    what is there; on a map with bare substrate it forces the empty pixels
    to be made of something.

    The solution is **not unique**. Any invertible rotation of the pair
    that keeps both non-negative is another solution, and the ambiguity
    is a property of the problem. Two runs from different starts can give
    different spectra that fit equally well; that is the method, not a bug.
    """
    matrix, indices = _matrix(cube, None)
    matrix = np.maximum(matrix, 0.0)
    k = int(components)
    if k < 1:
        raise MapError("hacen falta al menos dos componentes para resolver")
    if k > matrix.shape[1]:
        raise MapError("más componentes que puntos espectrales")

    # Start from the pixels that are least like each other: a pure-variable
    # start (SIMPLISMA-like) beats random, and beats the mean spectrum,
    # which sends every component to the same place.
    spectra = _purest(matrix, k, seed)

    previous = np.inf
    concentrations = np.zeros((matrix.shape[0], k))
    for _ in range(iterations):
        concentrations = _nnls_rows(spectra, matrix)
        if closure:
            total = np.sum(concentrations, axis=1, keepdims=True)
            concentrations = concentrations / np.where(total > 0, total, 1.0)
        spectra = _nnls_rows(concentrations.T, matrix.T).T
        residual = float(np.linalg.norm(matrix - concentrations @ spectra)
                         / max(np.linalg.norm(matrix), 1e-30))
        if abs(previous - residual) < tolerance:
            break
        previous = residual

    order = np.argsort(-np.sum(concentrations, axis=0))
    spectra, concentrations = spectra[order], concentrations[:, order]
    scale = np.max(spectra, axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    spectra = spectra / scale
    concentrations = concentrations * scale.T

    warnings = [
        "la solución NO es única: cualquier rotación del par que mantenga "
        "las dos partes no negativas ajusta igual de bien, y eso es una "
        "propiedad del problema, no del programa",
        "las concentraciones son relativas: no son fracciones másicas ni "
        "molares, porque las secciones eficaces Raman de dos especies "
        "difieren en órdenes de magnitud",
    ]
    warnings += _despike_warning(cube)
    if closure:
        warnings.append(
            "con cierre, las concentraciones suman uno en cada píxel: en un "
            "mapa con sustrato desnudo eso obliga a que el vacío esté hecho "
            "de algo"
        )
    return Decomposition(
        components=spectra, scores=cube.unflatten(concentrations, indices),
        shift=cube.shift, method="MCR-ALS", warnings=warnings,
        residual=residual,
    )


def _purest(matrix: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Starting spectra: the pixels furthest from each other."""
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(matrix, axis=1)
    chosen = [int(np.argmax(norms))]
    for _ in range(1, k):
        distances = np.maximum(
            np.min(_square_distances(matrix, matrix[chosen]), axis=1), 0.0)
        distances[chosen] = -1.0
        candidate = int(np.argmax(distances))
        if candidate in chosen:
            candidate = int(rng.integers(matrix.shape[0]))
        chosen.append(candidate)
    start = matrix[chosen].copy()
    return np.maximum(start, 1e-9)


def _nnls_rows(design: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Solve ``targets ≈ coefficients @ design`` row-wise, all non-negative.

    ``scipy.optimize.nnls`` one row at a time is exact and far too slow for
    a map, so this is projected gradient descent with a Lipschitz step —
    a few hundred iterations on the whole matrix at once, which for this
    purpose lands in the same place.
    """
    gram = design @ design.T
    cross = targets @ design.T
    step = 1.0 / max(float(np.linalg.norm(gram, 2)), 1e-12)
    coefficients = np.maximum(cross @ np.linalg.pinv(gram), 0.0)
    for _ in range(200):
        gradient = coefficients @ gram - cross
        updated = np.maximum(coefficients - step * gradient, 0.0)
        if np.max(np.abs(updated - coefficients)) < 1e-10 * (
                1.0 + np.max(np.abs(coefficients))):
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def _despike_warning(cube: RamanMap) -> list[str]:
    if any(step.startswith("despike") for step in cube.history):
        return []
    return [
        "el mapa no ha pasado por el filtro de rayos cósmicos: un pico de un "
        "solo canal en un solo píxel es la mayor excursión del cubo entero, "
        "así que sale como primera componente y como grupo propio"
    ]


__all__ = [
    "Clustering",
    "Decomposition",
    "despike_map",
    "kmeans",
    "mcr_als",
    "pca",
    "suggested_components",
]
