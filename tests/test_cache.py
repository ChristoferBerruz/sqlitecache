import os

import pytest

from sqlitecache.cache import CacheSettings, DiskStorage, LRUCache


@pytest.fixture
def disk_storage(tmpdir) -> DiskStorage:
    return DiskStorage(tmpdir)


@pytest.fixture
def max_size() -> int:
    return 100


@pytest.fixture
def cache_settings(max_size) -> CacheSettings:
    return CacheSettings(max_size)


@pytest.fixture
def lru_cache(disk_storage, cache_settings, tmp_path_factory) -> LRUCache:
    file = tmp_path_factory.mktemp("data") / "test.db"
    return LRUCache(db=file, storage=disk_storage, settings=cache_settings)


class TestLRUCache:
    def test_lru_put(self, lru_cache):
        lru_cache.put("key", "value")
        assert lru_cache.get("key") == "value"

    def test_lru_eviction(self, lru_cache: LRUCache, disk_storage: DiskStorage, mocker):
        def put_and_get(key, value):
            lru_cache.put(key, value)
            _cache_val = lru_cache.get(key)
            assert _cache_val == value

        put_and_get("key1", "value1")
        put_and_get("key2", "value2")
        put_and_get("key3", "value3")
        put_and_get("key4", "value4")
        put_and_get("key4", "value5")
        put_and_get("key5", "value5")
        assert lru_cache.get("key1") is None
