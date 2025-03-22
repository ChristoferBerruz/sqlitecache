import os

import pytest

from sqlitecache.cache import CacheSettings, FauxStorage, PersistentCache


@pytest.fixture
def persistent_cache():
    cache = PersistentCache.new("cache.db", CacheSettings(), FauxStorage())
    try:
        yield cache
    finally:
        # delete the cache file
        os.remove("cache.db")


def test_cache(persistent_cache: PersistentCache):
    persistent_cache.put(1, 2)
    assert persistent_cache.get(1) == 2
