"""Shared fixtures."""

from __future__ import annotations

import pytest

from ramancarbon.database import load_database
from ramancarbon.examples.demo_data import make_demo


@pytest.fixture(scope="session")
def db():
    """The literature database, loaded once."""
    return load_database()


@pytest.fixture
def swcnt():
    return make_demo("SWCNT", seed=1)


@pytest.fixture
def dwcnt():
    return make_demo("DWCNT", seed=2)


@pytest.fixture
def mwcnt():
    return make_demo("MWCNT", seed=3)


@pytest.fixture
def graphene():
    return make_demo("grafeno_1L", seed=4)
