import os
import time

import pytest

from sqlitecache.cache import (
    CacheSettings,
    DiskStorage,
    HybridCache,
    LFUCache,
    LRUCache,
    TTLSettings,
)


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
def db(tmp_path_factory):
    return tmp_path_factory.mktemp("data") / "test.db"


@pytest.fixture
def lru_cache(disk_storage, cache_settings, db) -> LRUCache:
    return LRUCache(db=db, storage=disk_storage, settings=cache_settings)


@pytest.fixture
def lfu_cache(disk_storage, cache_settings, db) -> LFUCache:
    return LFUCache(db=db, storage=disk_storage, settings=cache_settings)


@pytest.fixture
def ttl() -> int:
    return 10


@pytest.fixture
def treshold() -> int:
    return 5


@pytest.fixture
def hybrid_cache(lru_cache, lfu_cache, ttl, treshold, db):
    return HybridCache(
        db=db, ttl=ttl, treshold=treshold, lru_cache=lru_cache, lfu_cache=lfu_cache
    )


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

    def test_persistance(self, lru_cache: LRUCache, disk_storage, cache_settings, db):
        lru_cache.put("key1", "value1")
        # Simulate the cache being closed and reopened
        new_lru_cache = LRUCache(db=db, storage=disk_storage, settings=cache_settings)
        assert new_lru_cache.get("key1") == "value1"

    def test_get_rates(self, lru_cache: LRUCache):
        lru_cache.put("key1", "value1")
        lru_cache.get("key1")
        lru_cache.get("key2")
        hit_rate, miss_rate = lru_cache.get_rates()
        pytest.approx(hit_rate, 0.5)
        pytest.approx(miss_rate, 0.5)


class TestLFUCache:
    def test_lfu_put(self, lfu_cache: LFUCache):
        lfu_cache.put("key", "value")
        assert lfu_cache.get("key") == "value"

    def test_lfu_eviction(self, lfu_cache: LFUCache, disk_storage: DiskStorage, mocker):
        key_to_hashed_key = {}

        def put_and_get(key, value):
            hashed_key = lfu_cache.put(key, value)
            key_to_hashed_key[key] = hashed_key
            _cache_val = lfu_cache.get(key)
            assert _cache_val == value

        def get_from_cache(key):
            return lfu_cache.get(key)

        put_and_get("key1", "value1")
        put_and_get("key2", "value2")
        put_and_get("key3", "value3")
        put_and_get("key4", "value4")
        put_and_get("key4", "value5")
        # Make reads to increase frequency of keys
        get_from_cache("key1")
        get_from_cache("key2")
        get_from_cache("key3")
        # this will cause an eviction of the LFU key, which should be key4
        put_and_get("key5", "value5")
        assert lfu_cache.get("key4") is None

    def test_persistance(self, lfu_cache: LFUCache, disk_storage, cache_settings, db):
        lfu_cache.put("key1", "value1")
        # Simulate the cache being closed and reopened
        new_lfu_cache = LFUCache(db=db, storage=disk_storage, settings=cache_settings)
        assert new_lfu_cache.get("key1") == "value1"

    def test_get_rates(self, lfu_cache: LFUCache):
        lfu_cache.put("key1", "value1")
        lfu_cache.get("key1")
        lfu_cache.get("key2")
        hit_rate, miss_rate = lfu_cache.get_rates()
        pytest.approx(hit_rate, 0.5)
        pytest.approx(miss_rate, 0.5)

    @pytest.mark.slow
    def test_lfu_eviction_with_ttl(self, db, disk_storage, cache_settings):
        ttl_settings = TTLSettings(use_ttl=True, default_ttl=1)
        cache = LFUCache(
            db=db,
            storage=disk_storage,
            settings=cache_settings,
            ttl_settings=ttl_settings,
        )

        def put_and_get(key, value):
            cache.put(key, value)
            _cache_val = cache.get(key)
            assert _cache_val == value

        put_and_get("key1", "value1")
        put_and_get("key2", "value2")
        put_and_get("key3", "value3")
        put_and_get("key4", "value4")
        put_and_get("key4", "value5")
        # sleep as to make time pass
        time.sleep(3)
        put_and_get("key5", "value5")  # key5 triggers evictions
        assert cache.get("key1") is None


class TestHybridCache:
    def test_hybrid_put(self, hybrid_cache):
        hybrid_cache.put("key", "value")
        assert hybrid_cache.get("key") == "value"
