import os
import random
import time
from collections import defaultdict

import pandas as pd
import pytest

from sqlitecache.cache import (
    BaseCache,
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
        hit_rate, miss_rate, *_ = lru_cache.get_rates()
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
        hit_rate, miss_rate, *_ = lfu_cache.get_rates()
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


p_values = [100, 200, 300, 500]
q_values = [100, 200, 300, 500]
p_ids = [f"p={p}" for p in p_values]
q_ids = [f"q={q}" for q in q_values]
n_request_values = [100, 200, 300, 500]
n_request_ids = [f"n_requests={n_requests}" for n_requests in n_request_values]


@pytest.mark.simulation
class TestSimulation:
    @pytest.fixture(scope="class")
    def recorder(self):
        # Use the recorder such that we can collect the session stats
        _recorder = defaultdict(list)
        yield _recorder
        # convert the stats to multiple dataframes, depending on they key
        for key, values in _recorder.items():
            df = pd.DataFrame(values)
            df.to_csv(f"results/{key}.csv", index=False)
            print(f"Stats for {key}:")
            print(df.describe())

    @pytest.fixture
    def cache(self, request, p, q, db, disk_storage, n_requests, recorder):
        _cache = None
        if request.param == "lru_cache":
            size_settings = CacheSettings(max_size_in_bytes=p)
            _cache = LRUCache(db=db, storage=disk_storage, settings=size_settings)
        elif request.param == "lfu_cache_no_ttl":
            size_settings = CacheSettings(max_size_in_bytes=q)
            _cache = LFUCache(db=db, storage=disk_storage, settings=size_settings)
        else:
            raise ValueError("Invalid cache type")
        yield _cache
        # collection of stats
        hit_rate, miss_rate, total = _cache.get_rates()
        recorder[request.param].append(
            {
                "size": p if request.param == "lru_cache" else q,
                "n_requests": n_requests,
                "hit_rate": hit_rate,
                "miss_rate": miss_rate,
                "total": total,
            }
        )
        print(f"Hit rate: {hit_rate}, Miss rate: {miss_rate}, Total: {total}")

    @pytest.mark.parametrize("p", p_values, ids=p_ids)
    @pytest.mark.parametrize("q", q_values, ids=q_ids)
    @pytest.mark.parametrize("cache", ["lru_cache", "lfu_cache_no_ttl"], indirect=True)
    @pytest.mark.parametrize("n_requests", n_request_values, ids=n_request_ids)
    def test_simulation(self, p, q, cache: BaseCache, n_requests):
        """This is basically a simulation of doing random requests to the cache.
        Because our cache is "key" oriented rather than value oriented,
        we can insert the same value under random keys.

        The probability of reading and writing to the cache is 50% each.
        """

        def get_random_key():
            return f"key{random.randint(1, n_requests)}"

        inserted_keys_set = set()
        inserted_keys_lst = []

        # Penalty on memory, but easy lookup and avoid
        # overhead of converting the set into a list
        def perform_insert():
            key = get_random_key()
            value = "value"
            if key not in inserted_keys_set:
                inserted_keys_set.add(key)
                inserted_keys_lst.append(key)
            cache.put(key, value)

        def perform_read():
            if not inserted_keys_lst:
                return
            key = random.choice(inserted_keys_lst)
            value = cache.get(key)
            if value is None:
                # It is possible to have a hit miss,
                # but we need to update these two such that we can
                # always pick a read from the current values
                inserted_keys_set.remove(key)
                inserted_keys_lst.remove(key)

        for _ in range(n_requests):
            prob = random.random()
            if prob < 0.5:
                # Write to the cache
                perform_insert()
            else:
                perform_read()
