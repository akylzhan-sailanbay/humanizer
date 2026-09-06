"""Shared fixtures.

The cache is per-test (``tmp_path``) so hit/miss counts are meaningful; the
model manager is session-scoped because a gpt2-large load is 3 GB off disk and
re-paying it per test makes the slow suite unusable.
"""

import pytest

from humanizer.core.cache import Cache
from humanizer.core.logprob import LogProbService
from humanizer.core.models import ModelManager


@pytest.fixture
def cache(tmp_path):
    return Cache(tmp_path / "cache")


@pytest.fixture(scope="session")
def manager():
    return ModelManager(cap=2)


@pytest.fixture
def lps(manager, cache):
    return LogProbService(manager, cache)
