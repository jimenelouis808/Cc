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


def test_every_entry_has_an_english_label():
    assert {e[0] for e in cat.ENTRIES} == set(cat.LABELS_EN)
    assert len({e[0] for e in cat.ENTRIES}) == len(cat.ENTRIES)


def test_a_knee_entry_renders_in_english_and_as_a_4k_panel(tmp_path):
    from matplotlib.image import imread

    entry = next(e for e in cat.ENTRIES if e[0] == "kneess")
    cache = tmp_path / "cache"
    cache.mkdir()
    record = json.loads(cat._build(entry, cache).read_text())
    assert record["census"] == {"5": 4, "6": 68, "7": 16}
    assert cat.label(entry, "en") == "Super-square (knees, no spheres)"
    image = cat.render([entry], cache, tmp_path / "c.png", columns=1,
                       lang="en", dpi=300)
    assert image.exists()
    (panel,) = cat.render_panels([entry], cache, tmp_path / "p", "en")
    assert imread(panel).shape[:2] == (2160, 3840)
