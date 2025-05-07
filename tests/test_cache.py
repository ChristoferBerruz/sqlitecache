import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from attrs import define, field
from cryptography.hazmat.primitives.asymmetric import rsa

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
def threshold() -> int:
    return 5


@pytest.fixture
def hybrid_cache(lru_cache, lfu_cache, ttl, threshold, db):
    return HybridCache(
        db=db, ttl=ttl, threshold=threshold, lru_cache=lru_cache, lfu_cache=lfu_cache
    )


class TestCompressionAndEncryption:
    # This works like a hardness where we test
    # the combinations of compression and encryption
    # with the LRU and LFU caches.
    @pytest.fixture
    def max_size(self):
        # bump up the size to 512 bytes because
        # it is possible that encryption bloats the byte size
        return 512

    @pytest.fixture
    def private_rsa_key(self):
        # Generate a private RSA key for testing
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        return private_key

    @pytest.fixture
    def public_rsa_key(self, private_rsa_key):
        # Generate a public RSA key from the private key
        public_key = private_rsa_key.public_key()
        return public_key

    @pytest.fixture(params=[True, False], ids=["compression", "no_compression"])
    def compression(self, request):
        return request.param

    @pytest.fixture(params=[True, False], ids=["encryption", "no_encryption"])
    def pub_key(self, request, public_rsa_key):
        if request.param:
            return public_rsa_key
        return None

    @pytest.fixture
    def cache_settings(self, max_size, compression, pub_key):
        return CacheSettings(max_size, compression=compression, public_key=pub_key)

    @pytest.fixture(params=["lru", "lfu"])
    def cache(self, request, lru_cache: LRUCache, lfu_cache: LFUCache, private_rsa_key):
        cache_to_return = lru_cache if request.param == "lru" else lfu_cache
        if cache_to_return.settings.public_key is not None:
            cache_to_return._private_key = private_rsa_key
        return cache_to_return

    def test_feature(self, cache: BaseCache, disk_storage: DiskStorage):
        # Test that putting and getting a value works
        # regardless of compression and encryption being active
        cache.put("key", "value")
        assert cache.get("key") == "value"


class TestEncryption:
    @pytest.fixture
    def cache_settings(self, max_size, public_rsa_key):
        # Use the public key for encryption
        return CacheSettings(max_size, encryption=True, encryption_key=public_rsa_key)


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

    @pytest.mark.slow
    def test_hybrid_eviction(
        self, hybrid_cache: HybridCache, disk_storage: DiskStorage
    ):
        def put_and_get_with_delay(key, value):
            hybrid_cache.put(key, value)
            time.sleep(0.1)  # Simulate some delay
            _cache_val = hybrid_cache.get(key)
            assert _cache_val == value

        # put and get with delay to simulate the time passing
        put_and_get_with_delay("key1", "value1")
        put_and_get_with_delay("key2", "value2")
        put_and_get_with_delay("key3", "value3")
        put_and_get_with_delay("key4", "value4")
        put_and_get_with_delay("key4", "value5")
        # key5 should trigger eviction
        put_and_get_with_delay("key5", "value5")
        # Make sure one of the previous keys was actually evicted
        assert any(
            hybrid_cache.lru_cache.get(key) is None
            for key in ["key1", "key2", "key3", "key4"]
        )


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


@pytest.mark.slow
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
        yield 1_800

    @pytest.fixture(scope="class")
    def q(self):
        # LFU cache size
        yield 1_800

    @pytest.fixture
    def ttl_settings(self, ttl):
        # TTL settings
        yield TTLSettings(use_ttl=True, default_ttl=ttl)

    @pytest.fixture
    def ttl(self):
        yield 5

    @pytest.fixture
    def threshold(self):
        yield 5

    @pytest.fixture(scope="class")
    def recorder(self):
        # Use the recorder such that we can collect the session stats
        _recorder: Dict[str, SimulationResult] = {}
        yield _recorder
        cache_names_to_df = {}
        # Saving the results to a CSV file
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
            filename = f"results/{key}SimulationResults.csv"
            df.to_csv(filename, index=False)

    @pytest.fixture
    def lru_cache(self, db, disk_storage, p) -> LRUCache:
        size_settings = CacheSettings(max_size_in_bytes=p)
        return LRUCache(db=db, storage=disk_storage, settings=size_settings)

    @pytest.fixture
    def lfu_cache(self, db, disk_storage, q) -> LFUCache:
        size_settings = CacheSettings(max_size_in_bytes=q)
        return LFUCache(db=db, storage=disk_storage, settings=size_settings)

    @pytest.fixture
    def lfu_cache_ttl(self, db, disk_storage, q, ttl_settings) -> LFUCache:
        size_settings = CacheSettings(max_size_in_bytes=q)
        return LFUCache(
            db=db,
            storage=disk_storage,
            settings=size_settings,
            ttl_settings=ttl_settings,
        )

    @pytest.fixture
    def hybrid_cache(self, lru_cache, lfu_cache, threshold, ttl, db) -> HybridCache:
        return HybridCache(
            db=db,
            lru_cache=lru_cache,
            lfu_cache=lfu_cache,
            threshold=threshold,
            ttl=ttl,
        )

    @pytest.fixture(scope="class")
    def dataset(self):
        yield [f"{i}".zfill(3) for i in range(1, 501)]

    def run_simulation(self, cache: BaseCache, dataset: List[str], n_requests: int):
        """Run the simulation for a given cache and dataset."""
        hit_miss_rates_tracker = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0,
        }
        miss_rates = []
        hit_rates = []
        all_requests = []

        # sustained load of N requests
        for _ in range(n_requests):
            key = random.choice(dataset)
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
            miss_rates.append(cur_miss_rate)
            hit_rates.append(cur_hit_rate)
            all_requests.append(cur_requests)
            # Bear with me here. In the paper, if the element is
            # not in the cache, we insert it. This is weird
            # because in concrete, that is not a real cache.
            # Regardless, we will do it ONLY during the simulation
            # and not ship it in the actual classes.
            if result is None:
                cache.put(key, key)
        return hit_rates, miss_rates, all_requests

    def test_simulation_lru_cache(self, lru_cache, n_requests, recorder, dataset):
        """See the class docstring for the simulation."""
        hit_rates, miss_rates, all_requests = self.run_simulation(
            lru_cache, dataset, n_requests
        )
        cache_name = "LRUCache"
        recorder[cache_name] = SimulationResult(
            cache_name=cache_name,
            cache_size=lru_cache.settings.max_size_in_bytes,
            hit_rates=hit_rates,
            miss_rates=miss_rates,
            requests=all_requests,
        )

    def test_simulation_lfu_cache(self, lfu_cache, n_requests, recorder, dataset):
        """See the class docstring for the simulation."""
        hit_rates, miss_rates, all_requests = self.run_simulation(
            lfu_cache, dataset, n_requests
        )
        cache_name = "LFUCache"
        recorder[cache_name] = SimulationResult(
            cache_name=cache_name,
            cache_size=lfu_cache.settings.max_size_in_bytes,
            hit_rates=hit_rates,
            miss_rates=miss_rates,
            requests=all_requests,
        )

    def test_simulation_lfu_cache_ttl(
        self, lfu_cache_ttl, n_requests, recorder, dataset
    ):
        """See the class docstring for the simulation."""
        hit_rates, miss_rates, all_requests = self.run_simulation(
            lfu_cache_ttl, dataset, n_requests
        )
        cache_name = "LFUCacheTTL"
        recorder[cache_name] = SimulationResult(
            cache_name=cache_name,
            cache_size=lfu_cache_ttl.settings.max_size_in_bytes,
            hit_rates=hit_rates,
            miss_rates=miss_rates,
            requests=all_requests,
        )

    def test_simulation_hybrid_cache(self, hybrid_cache, n_requests, recorder, dataset):
        """See the class docstring for the simulation."""
        hit_rates, miss_rates, all_requests = self.run_simulation(
            hybrid_cache, dataset, n_requests
        )
        cache_name = "HybridCache"
        recorder[cache_name] = SimulationResult(
            cache_name=cache_name,
            cache_size=hybrid_cache.lru_cache.settings.max_size_in_bytes,
            hit_rates=hit_rates,
            miss_rates=miss_rates,
            requests=all_requests,
        )
