"""Smoke test for the Streamlit GUI.

Only checks that the script executes without raising and renders its structure —
enough to catch an import error or a renamed function, which is the failure mode
that actually happens when the library underneath changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

#: AppTest resolves a relative path against the *test* file, not the cwd.
APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_runs_clean_from_empty_state() -> None:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    assert not app.exception
    assert app.title[0].value.endswith("nanocarbon_biblio")
    assert len(app.tabs) == 7
    # Every downstream tab must tell the user what to do first.
    assert any("pestaña 1" in info.value for info in app.info)
