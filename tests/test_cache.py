import os
import shutil
import tempfile
import pytest

from sqlitecache.cache import CacheSettings, DiskStorage, LFUCache, LRUCache
from cryptography.fernet import Fernet


@pytest.fixture(scope="session")
def encryption_key():
    return Fernet.generate_key()


@pytest.fixture
def disk_storage(tmpdir, encryption_key) -> DiskStorage:
    return DiskStorage(tmpdir, encryption_key=encryption_key)


@pytest.fixture
def max_size() -> int:
    return 200


@pytest.fixture
def cache_settings(max_size) -> CacheSettings:
    return CacheSettings(max_size)


@pytest.fixture
def lru_cache(disk_storage, cache_settings, tmp_path_factory) -> LRUCache:
    file = tmp_path_factory.mktemp("data") / "test.db"
    return LRUCache(db=file, storage=disk_storage, settings=cache_settings)


@pytest.fixture
def lfu_cache(disk_storage, cache_settings, tmp_path_factory) -> LFUCache:
    file = tmp_path_factory.mktemp("data") / "test.db"
    return LFUCache(db=file, storage=disk_storage, settings=cache_settings)


class TestLRUCache:
    def test_lru_put(self, lru_cache):
        lru_cache.put("key", "v1")
        assert lru_cache.get("key") == "v1"

    def test_lru_eviction(self, lru_cache: LRUCache, disk_storage: DiskStorage, mocker):
        def put_and_get(key, value):
            lru_cache.put(key, value)
            _cache_val = lru_cache.get(key)
            assert _cache_val == value

        put_and_get("key1", "v1")
        put_and_get("key2", "v2")
        put_and_get("key3", "v3")
        put_and_get("key4", "v4")
        put_and_get("key4", "v5")
        put_and_get("key5", "v6")
        assert lru_cache.get("key1") is None


class TestLFUCache:
    def test_lfu_put(self, lfu_cache: LFUCache):
        lfu_cache.put("key", "v1")
        assert lfu_cache.get("key") == "v1"

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
