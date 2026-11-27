"""The catalogue example renders a cached entry, tiled when periodic."""

import json

import numpy as np

from nanocarbon_lab.examples import ex12_catalogue as cat


def test_a_ring_across_the_seam_is_joined_not_split():
    cell = np.array([10.0, 10.0, 0.0])
    ring = np.array([[9.5, 1, 0], [0.5, 1, 0], [0.5, 2, 0]], float)
    joined = cat._unwrap(ring, cell, [True, True, False])
    assert np.ptp(joined[:, 0]) < 1.5


def test_the_catalogue_renders_a_cached_entry(tmp_path):
    entry = next(e for e in cat.ENTRIES if e[0] == "pchiral")
    cache = tmp_path / "cache"
    cache.mkdir()
    path = cat._build(entry, cache)
    record = json.loads(path.read_text())
    assert record["census"] == {"6": 296, "8": 12}
    image = cat.render([entry], cache, tmp_path / "c.png", columns=1)
    assert image.exists() and image.stat().st_size > 10_000
