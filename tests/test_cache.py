import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from attrs import define, field

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


@define
class SimulationResult:
    """This is the result of the simulation.

    It contains the hit rate, miss rate, and total number of requests.
    """

    cache_name: str
    cache_size: int
    hit_rates: List[float] = field(factory=list)
    miss_rates: List[float] = field(factory=list)
    requests: List[int] = field(factory=list)
    T: Optional[int] = None


@pytest.mark.simulation
class TestSimulation:
    """This is the simulation harness following the paper.

    Assume the cache size is of 100 elements.
    There are 500 unique elements that can be inserted.

    Because the paper is element oriented, we need to adapt it to key
    oriented.

    We can use integer key-value pairs (k, k) where
    k in [1, 500]. Each value occupy 1 byte.
    Therefore, the cache size is 100 bytes.

    One hyperparameter to tune is T = the counter before
    the elements are evicted given that P, Q (sizes) are already
    fixed to be 100.

    The simulation is a sustained load of N requests.
    At every request, we track the hit rate and miss rate.
    Note that although the caches do keep track
    of the hit rate and miss rates, accessing them
    N times will incur a penalty on the TTL. Therefore,
    we can use some aux structures in memory
    to keep track of the hit rate and miss rate -
    for simulation sake.
    """

    @pytest.fixture(scope="class")
    def p(self):
        # LRU cache size
        yield 100

    @pytest.fixture(scope="class")
    def q(self):
        # LFU cache size
        yield 100

    @pytest.fixture(scope="class")
    def recorder(self):
        # Use the recorder such that we can collect the session stats
        _recorder: Dict[str, SimulationResult] = {}
        yield _recorder
        # convert the stats to multiple dataframes, depending on they key
        print("Simulation results:")
        cache_names_to_df = {}
        for key, result in _recorder.items():
            df = pd.DataFrame(
                {
                    "hit_rate": result.hit_rates,
                    "miss_rate": result.miss_rates,
                    "requests": result.requests,
                }
            )
            cache_names_to_df[key] = df
            print(f"Cache: {key}")
            print(df)
            # Save the dataframe to a CSV file
            filename = f"results/{key}_simulation_results.csv"
            df.to_csv(filename, index=False)
        print("Generating all plots")

        # Plot all hit rates in the same subplot
        def generate_plot_for_attr(attr: str = "hit_rate"):
            plt.figure(figsize=(10, 6))
            for key, df in cache_names_to_df.items():
                plt.plot(df["requests"], df[attr], label=f"{key} Hit Rate")

            plt.title(f"{attr} for different Cache")
            plt.xlabel("Requests")
            plt.ylabel(attr)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"results/{attr}_plot.png")  # Save the plot as a PNG file

        generate_plot_for_attr(attr="hit_rate")
        generate_plot_for_attr(attr="miss_rate")

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

    @pytest.mark.parametrize("cache", ["lru_cache", "lfu_cache_no_ttl"], indirect=True)
    def test_simulation(self, p, q, cache: BaseCache, n_requests, recorder):
        """See the class docstring for the simulation."""
        hit_miss_rates_tracker = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
        }
        simulation_result = SimulationResult(
            cache_name=cache.__class__.__name__,
            cache_size=cache.settings.max_size_in_bytes,
            T=None,
        )
        # 500 unique elements that can be inserted
        key_values_pairs = [(i, i) for i in range(1, 501)]
        recorder[cache.__class__.__name__] = simulation_result
        # sustained load of N requests
        for i in range(n_requests):
            key = random.choice(key_values_pairs)
            result = cache.get(key, default=None)
            hit_miss_rates_tracker["total_requests"] += 1
            if result is not None:
                hit_miss_rates_tracker["hits"] += 1
            else:
                hit_miss_rates_tracker["misses"] += 1
            # hits and misses only make sense for reads
            cur_hit_rate = (
                hit_miss_rates_tracker["hits"]
                / hit_miss_rates_tracker["total_requests"]
            )
            cur_miss_rate = (
                hit_miss_rates_tracker["misses"]
                / hit_miss_rates_tracker["total_requests"]
            )
            cur_requests = hit_miss_rates_tracker["total_requests"]
            simulation_result.hit_rates.append(cur_hit_rate)
            simulation_result.miss_rates.append(cur_miss_rate)
            simulation_result.requests.append(cur_requests)
            # Bear with me here. In the paper, if the element is
            # not in the cache, we insert it. This is weird
            # because in concrete, that is not a real cache.
            # Regardless, we will do it ONLY during the simulation
            # and not ship it in the actual classes.
            if result is None:
                cache.put(key, key)
