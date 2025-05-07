"""Persistent cache.

Let's understand what a cache is using a top-down approach.

For a user, a cache is some container that supports 3 operations.

put(key, value)
get(key)
delete(key)

Behaviorally, a cache has a fixed capacity. Capacity can be
upper bounded by either the number of elements, or the total
byte space they occupy. Why? Because a cache will eventually
run out of space, forcing an eviction policy to kick in.

The goal of this module is to support "persistent cache".
In practice, this means that the data is still on the cache
even after a reboot. How do you achieve persistency? Using
the filesystem*.

This module can be decomposed into two parts.

1) Cache housekeeping. This keeps track of elements inside the cache, expirations,
usage, etc.
2) *Element storage. This is where the elements actually live in.

We know for a fact that for 1), we want to use SQLite.

However, for 2), can use a database table to store the cache
elements (as entries). Or we can use a filesystem ~ somehow. 
One approach for using the filesystem is simply store
each value in a different file and the "cache storage"
is basically a directory.

Insert operation:

1. User submits a (key, value) into the Cache
2. Cache checks whether there's enough space. If enough space, Cache forwards
request to Storage. Otherwise, first evict using policy and forward request.
3. Storage stores (key, value) and returns a storageKey to Cache.
4. Cache saves (key, storageKey) into its cache + accompanying metadata.

Get operation:

1. Users submits a key into the Cache
2. Cache lookups a corresponding (key, storageKey). If no defined, return some default (None).
3. Cache request Storage for a lookup of storageKey.
4. Storage returns the value to Cache
5. Cache returns the value

Delete operation:
1. Users submits a key into the Cache
2. If key not present, no-OP. Else, find a corresponding (key, storageKey) in the table.
3. Forward the delete request to Storage.

NOTE: The above algorithms are general pseudocode, the actual implementation
lives in the code. However, it is a nice mental model to have when reading this module.
"""
import os
import pickle
import sqlite3
import uuid
import zlib
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import contextmanager
from typing import Any, Generator, Hashable, Optional, Tuple

import pendulum
from attrs import define, field

HashedKey = int


class Cache(ABC):
    @abstractmethod
    def put(key: Hashable, value: Any) -> HashedKey:
        pass

    @abstractmethod
    def get(key: Hashable, default: Any = None) -> Any:
        pass

    @abstractmethod
    def delete(key: Hashable):
        pass

    @abstractmethod
    def fits(size: int) -> bool:
        """Check whether we can insert something of size `size` into the cache."""
        pass

    @abstractmethod
    def exists(key: Hashable) -> bool:
        pass

    @abstractmethod
    def get_rates(self) -> Tuple[float, float, int]:
        """Get the hit and miss rates of the cache."""
        pass


@define(frozen=False, slots=False)
class DiskStorage:
    storage_dir: str

    def put(
        self, key: Hashable, value: Any, compression: bool = False
    ) -> Tuple[str, int]:
        filename = f"{uuid.uuid4().hex}.cache"
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "wb") as f:
            if compression:
                compressed_value = zlib.compress(pickle.dumps(value))
                f.write(compressed_value)
            else:
                pickle.dump(value, f)
        return filename, os.path.getsize(filepath)

    def get(self, filename: str, compression: bool = False) -> Any:
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "rb") as f:
            if compression:
                compressed_value = f.read()
                return pickle.loads(zlib.decompress(compressed_value))
            else:
                # If not compressed, just load the value
                # and return it.
                return pickle.load(f)

    def delete(self, filename: str):
        filepath = os.path.join(self.storage_dir, filename)
        os.remove(filepath)

    def predict_size(self, value: Any, compression: bool = False) -> int:
        if compression:
            # If we are using compression, we need to compress the value
            # and then get the size.
            return len(zlib.compress(pickle.dumps(value)))
        # If not, we just need to get the size of the pickled value.
        else:
            return len(pickle.dumps(value))

    @contextmanager
    def defer_delete(self, filename: str):
        try:
            yield filename
        finally:
            self.delete(filename)


def get_hash_for_key(key: Hashable) -> int:
    return hash(key) & 0xFFFFFFFF


@define
class CacheSettings:
    max_size_in_bytes: int
    compression: bool = False


@define
class BaseCache(Cache):
    db: str

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.db)
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def commit_connection(self) -> Generator[sqlite3.Connection, None, None]:
        with self.connection() as connection:
            with connection:
                yield connection

    @contextmanager
    def cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        with self.connection() as connection:
            cursor = connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()


def auto_hash_key(func):
    def wrapper(self, key: Hashable, *args, **kwargs):
        key = get_hash_for_key(key)
        return func(self, key, *args, **kwargs)

    return wrapper


@define
class FunctionalCache(BaseCache):
    storage: DiskStorage
    settings: CacheSettings

    @property
    @abstractmethod
    def size_table(self) -> str:
        pass

    @property
    @abstractmethod
    def cache_table(self) -> str:
        pass

    @property
    @abstractmethod
    def metadata_table(self) -> str:
        pass

    def _create_triggers_for_size_tracking(self):
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS insert_{self.size_table}
                AFTER INSERT ON {self.cache_table} FOR EACH ROW
                BEGIN
                    UPDATE {self.metadata_table}
                    SET value = value + NEW.size
                    WHERE setting = 'current_size';
                END
                """
            )
            cursor.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS delete_{self.size_table}
                AFTER DELETE ON {self.cache_table} FOR EACH ROW
                BEGIN
                    UPDATE {self.metadata_table}
                    SET value = value - OLD.size
                    WHERE setting = 'current_size';
                END
                """
            )

            cursor.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS update_{self.size_table}
                AFTER UPDATE ON {self.cache_table} FOR EACH ROW
                BEGIN
                    UPDATE {self.metadata_table}
                    SET value = value + NEW.size - OLD.size
                    WHERE setting = 'current_size';
                END
                """
            )

    def _set_settings(self):
        # To set the settings, first double check that the settings exists.
        # Let's use ON CONFLICT to avoid duplicates by simply ignoring the request.
        # Also, all setting values are simply TEXT strings!
        with self.commit_connection() as con:
            con.execute(
                f"""
                INSERT INTO {self.metadata_table} (setting, value)
                VALUES (?, ?)
                ON CONFLICT(setting) DO NOTHING
                """,
                ("current_size", 0),
            )
            con.execute(
                f"""
                INSERT INTO {self.metadata_table} (setting, value)
                VALUES (?, ?)
                ON CONFLICT(setting) DO NOTHING
                """,
                ("max_size", self.settings.max_size_in_bytes),
            )
            # compression is a boolean, so we need to convert it to an int.
            # 0 for False, 1 for True.
            con.execute(
                f"""
                INSERT INTO {self.metadata_table} (setting, value)
                VALUES (?, ?)
                ON CONFLICT(setting) DO NOTHING
                """,
                ("compression", int(self.settings.compression)),
            )
            # Let's also add tracking for hit and miss rates.
            con.execute(
                f"""
                INSERT INTO {self.metadata_table} (setting, value)
                VALUES (?, ?)
                ON CONFLICT(setting) DO NOTHING
                """,
                ("hit_rate", 0),
            )
            con.execute(
                f"""
                INSERT INTO {self.metadata_table} (setting, value)
                VALUES (?, ?)
                ON CONFLICT(setting) DO NOTHING
                """,
                ("miss_rate", 0),
            )
        # Override the in-memory settings with the ones from the database.
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT value
                FROM {self.metadata_table}
                WHERE setting = 'compression'
                """
            )
            compression = int(cursor.fetchone()[0])
            if compression == 1:
                self.settings.compression = True
            else:
                self.settings.compression = False
            cursor.execute(
                f"""
                SELECT value
                FROM {self.metadata_table}
                WHERE setting = 'max_size'
                """
            )
            max_size = cursor.fetchone()[0]
            self.settings.max_size_in_bytes = int(max_size)

    def _update_hit_rate(self, hit: bool):
        """
        Update the hit rate in the metadata table.
        Arguments:
            hit: A boolean indicating whether the cache hit or miss.
        """
        with self.commit_connection() as con:
            if hit:
                con.execute(
                    f"""
                    UPDATE {self.metadata_table}
                    SET value = value + 1
                    WHERE setting = 'hit_rate'
                    """
                )
            else:
                con.execute(
                    f"""
                    UPDATE {self.metadata_table}
                    SET value = value + 1
                    WHERE setting = 'miss_rate'
                    """
                )

    def get_rates(self) -> Tuple[float, float, int]:
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT value
                FROM {self.metadata_table}
                WHERE setting = 'hit_rate'
                """
            )
            hit_rate: int = int(cursor.fetchone()[0])
            cursor.execute(
                f"""
                SELECT value
                FROM {self.metadata_table}
                WHERE setting = 'miss_rate'
                """
            )
            miss_rate: int = int(cursor.fetchone()[0])
            total_requests = hit_rate + miss_rate
            if total_requests == 0:
                return 0.0, 0.0
            return hit_rate / total_requests, miss_rate / total_requests, total_requests

    def fits(self, size: int) -> bool:
        with self.cursor() as cursor:
            # We are currently tracking the size of the cache, and the max size
            # is stored in the settings table.
            cursor.execute(
                f"""
                SELECT value
                FROM {self.metadata_table}
                WHERE setting = 'current_size'
                """
            )
            current_size: int = int(cursor.fetchone()[0])
            cursor.execute(
                f"""
                SELECT value
                FROM {self.metadata_table}
                WHERE setting = 'max_size'
                """
            )
            max_size: int = int(cursor.fetchone()[0])
            return current_size + size <= max_size

    @auto_hash_key
    def exists(self, key: Hashable) -> bool:
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT COUNT(*)
                FROM {self.cache_table}
                WHERE key = ?
                """,
                (key,),
            )
            fetched = cursor.fetchone()
            if fetched is None:
                return False
            return fetched[0] > 0

    def _get_filename_for_key(
        self, key: HashedKey, default: str = None
    ) -> Optional[str]:
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT filename
                FROM {self.cache_table}
                WHERE key = ?
                """,
                (key,),
            )
            filename_res = cursor.fetchone()
            if filename_res is None:
                return default
            return filename_res[0]

    @auto_hash_key
    def _get_size_for_key(self, key: Hashable) -> int:
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT size
                FROM {self.cache_table}
                WHERE key = ?
                """,
                (key,),
            )
            return cursor.fetchone()[0]


@define
class LRUCache(FunctionalCache):
    def __attrs_post_init__(self):
        self._create_cache_table_and_metadata()

    @auto_hash_key
    def put(self, key: Hashable, value: Any) -> HashedKey:
        predicted_size = self.storage.predict_size(
            value, compression=self.settings.compression
        )
        if not self.exists(key):
            self._evict_until_satified(predicted_size)
            self._put_new(key, value)
        else:
            current_size = self._get_size_for_key(key)
            self._evict_until_satified(predicted_size - current_size)
            self._put_update(key, value)

    def _evict_until_satified(self, size: int):
        while not self.fits(size):
            # Evict the least recently used element
            with self.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT key
                    FROM {self.cache_table}
                    ORDER BY accessed_at ASC
                    LIMIT 1
                    """
                )
                evict_key = cursor.fetchone()[0]
                self.delete(evict_key)

    @auto_hash_key
    def _put_new(self, key: Hashable, value: Any) -> HashedKey:
        with self.commit_connection() as con:
            now = pendulum.now().timestamp()
            filename, size = self.storage.put(
                key, value, compression=self.settings.compression
            )
            con.execute(
                f"""
                INSERT INTO {self.cache_table} (key, filename, size, stored_at, accessed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, filename, size, now, now),
            )
            return key

    @auto_hash_key
    def _put_update(self, key: Hashable, value: Any) -> HashedKey:
        prev_filename = self._get_filename_for_key(key)
        with self.storage.defer_delete(
            prev_filename
        ) as _, self.commit_connection() as con:
            now = pendulum.now().timestamp()
            filename, size = self.storage.put(
                key, value, compression=self.settings.compression
            )
            con.execute(
                f"""
                UPDATE {self.cache_table}
                SET filename = ?, size = ?, stored_at = ?, accessed_at = ?
                WHERE key = ?
                """,
                (filename, size, now, now, key),
            )
            return key

    @auto_hash_key
    def get(self, key: Hashable, default: Any = None) -> Any:
        if not self.exists(key):
            self._update_hit_rate(hit=False)
            return default

        self._update_hit_rate(hit=True)
        filename = self._get_filename_for_key(key)
        with self.commit_connection() as cursor:
            now = pendulum.now().timestamp()
            cursor.execute(
                f"""
                UPDATE {self.cache_table}
                SET accessed_at = ?
                WHERE key = ?
                """,
                (now, key),
            )
            return self.storage.get(filename, compression=self.settings.compression)

    @auto_hash_key
    def delete(self, key: Hashable):
        if not self.exists(key):
            return
        filename = self._get_filename_for_key(key)
        with self.storage.defer_delete(filename) as _, self.commit_connection() as con:
            con.execute(
                f"""
                DELETE FROM {self.cache_table}
                WHERE key = ?
                """,
                (key,),
            )

    def _create_cache_table_and_metadata(self):
        with self.cursor() as cursor:
            # Creates metadata table
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.metadata_table} (
                    setting TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL
                )
                """
            )
            # Creates the main cache table
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.cache_table} (
                    key INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    stored_at TIMESTAMP NOT NULL,
                    accessed_at INTEGER NOT NULL
                )
                """
            )

            # Let's make an index on the key
            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS LRUPrimaryCache_key
                ON {self.cache_table} (key)
                """
            )
        self._create_triggers_for_size_tracking()
        self._set_settings()

    @property
    def size_table(self) -> str:
        return "LRUTotalSize"

    @property
    def cache_table(self) -> str:
        return "LRUPrimaryCache"

    @property
    def metadata_table(self) -> str:
        return "LRUMetadata"

    def show_cache(self):
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM {self.cache_table}
                """
            )
            print(cursor.fetchall())


@define
class TTLSettings:
    use_ttl: bool = False
    default_ttl: float = 0

    def __attrs_post_init__(self):
        if self.use_ttl and self.default_ttl <= 0:
            raise ValueError("TTL must be greater than 0 if use_ttl is True.")


@define
class LFUCache(FunctionalCache):
    ttl_settings: Optional[TTLSettings] = None

    def __attrs_post_init__(self):
        self._create_cache_table_and_metadata()

    def _set_settings(self):
        # Run the parent's settings, which basically sets all the "shared" settings.
        super()._set_settings()
        # Now set the TTL settings.
        if self.ttl_settings is not None:
            with self.commit_connection() as con:
                con.execute(
                    f"""
                    INSERT INTO {self.metadata_table} (setting, value)
                    VALUES (?, ?)
                    ON CONFLICT(setting) DO NOTHING
                    """,
                    ("use_ttl", int(self.ttl_settings.use_ttl)),
                )
                con.execute(
                    f"""
                    INSERT INTO {self.metadata_table} (setting, value)
                    VALUES (?, ?)
                    ON CONFLICT(setting) DO NOTHING
                    """,
                    ("default_ttl", self.ttl_settings.default_ttl),
                )
        # read back the settings
        disk_ttl_settings = None
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT value
                FROM {self.metadata_table}
                WHERE setting = 'use_ttl'
                """
            )
            maybe_use_ttl = cursor.fetchone()
            if maybe_use_ttl is None:
                disk_ttl_settings = None
                return
            use_ttl = int(maybe_use_ttl[0])
            if use_ttl == 1:
                use_ttl = True
            else:
                use_ttl = False
            cursor.execute(
                f"""
                SELECT value
                FROM {self.metadata_table}
                WHERE setting = 'default_ttl'
                """
            )
            # if ttl was defined, we are guaranteed to have a value.
            default_ttl = float(cursor.fetchone()[0])
            disk_ttl_settings = TTLSettings(use_ttl=use_ttl, default_ttl=default_ttl)
        self.ttl_settings = disk_ttl_settings

    @auto_hash_key
    def put(self, key: Hashable, value: Any, ttl: Optional[int] = None) -> HashedKey:
        # By doing this, we are allowing user to insert a ttl value into the
        # cache but it will not be used if the cache is not configured to use it.
        _ttl = ttl or self.ttl_settings.default_ttl if self.ttl_settings else ttl
        predicted_size = self.storage.predict_size(
            value, compression=self.settings.compression
        )
        if not self.exists(key):
            self._evict_until_satified(predicted_size)
            return self._put_new(key, value, _ttl)
        else:
            current_size = self._get_size_for_key(key)
            self._evict_until_satified(predicted_size - current_size)
            return self._put_update(key, value, _ttl)

    def _evict_until_satified(self, size: int):
        if self.ttl_settings is not None and self.ttl_settings.use_ttl:
            self._evict_until_satified_ttl(size)
        else:
            self._evict_until_satified_countbased(size)

    def _evict_all_expired(self):
        if self.ttl_settings is not None and self.ttl_settings.use_ttl:
            with self.commit_connection() as cursor:
                now = pendulum.now().timestamp()
                # an element is considered to be expired if the current time is greater
                # than the stored_at + ttl
                cursor.execute(
                    f"""
                    DELETE FROM {self.cache_table}
                    WHERE stored_at + ttl < ?
                    """,
                    (now,),
                )

    def _evict_until_satified_ttl(self, size: int):
        self._evict_all_expired()
        if self.fits(size):
            return
        # If we need more space, we need to evict the elements based on ttl.
        # Now, because we never updated the ttl, we don't know which is the lowest.
        # An example, assume at t = 0, we inserted with ttl = 6, and at t = 5, we inserted
        # with ttl = 2. The first one will be evicted first, but the second one is the one
        # that is going to expire first. So we need to update the ttl of all the elements first
        with self.commit_connection() as cursor:
            # update is simply the delta = now - stored_at
            # and new_ttl = ttl - delta
            now = pendulum.now().timestamp()
            cursor.execute(
                f"""
                    UPDATE {self.cache_table}
                    SET ttl = ttl - (? - stored_at)
                    WHERE stored_at + ttl < ?
                    """,
                (now, now),
            )
        while not self.fits(size):
            with self.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT key
                    FROM {self.cache_table}
                    ORDER BY ttl ASC
                    LIMIT 1
                    """
                )
                evict_key = cursor.fetchone()[0]
                self.delete(evict_key)

    def _evict_until_satified_countbased(self, size: int):
        while not self.fits(size):
            # Evict the least frequently used element.
            # TODO: Consider handling breaking ties.
            with self.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT key
                    FROM {self.cache_table}
                    ORDER BY accessed_count ASC
                    LIMIT 1
                    """
                )
                evict_key = cursor.fetchone()[0]
                self.delete(evict_key)

    @auto_hash_key
    def _put_new(
        self, key: Hashable, value: Any, ttl: Optional[int] = None
    ) -> HashedKey:
        with self.commit_connection() as con:
            now = pendulum.now().timestamp()
            filename, size = self.storage.put(
                key, value, compression=self.settings.compression
            )
            con.execute(
                f"""
                INSERT INTO {self.cache_table} (key, filename, size, stored_at, accessed_count, ttl)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, filename, size, now, 0, ttl),
            )
            return key

    @auto_hash_key
    def _put_update(
        self, key: Hashable, value: Any, ttl: Optional[int] = None
    ) -> HashedKey:
        prev_filename = self._get_filename_for_key(key)
        with self.storage.defer_delete(
            prev_filename
        ) as _, self.commit_connection() as con:
            now = pendulum.now().timestamp()
            filename, size = self.storage.put(
                key, value, compression=self.settings.compression
            )
            # We reset the accessed_count to 0 because we are updating the value.
            con.execute(
                f"""
                UPDATE {self.cache_table}
                SET filename = ?, size = ?, stored_at = ?, accessed_count = ?, ttl = ?
                WHERE key = ?
                """,
                (filename, size, now, 0, ttl, key),
            )
            return key

    @auto_hash_key
    def get(self, key: Hashable, default: Any = None) -> Any:
        self._evict_all_expired()
        if not self.exists(key):
            self._update_hit_rate(hit=False)
            return default
        self._update_hit_rate(hit=True)
        filename = self._get_filename_for_key(key)
        with self.commit_connection() as cursor:
            cursor.execute(
                f"""
                UPDATE {self.cache_table}
                SET accessed_count = accessed_count + 1
                WHERE key = ?
                """,
                (key,),
            )
            return self.storage.get(filename, self.settings.compression)

    @auto_hash_key
    def delete(self, key: Hashable):
        if not self.exists(key):
            return
        filename = self._get_filename_for_key(key)
        with self.storage.defer_delete(filename) as _, self.commit_connection() as con:
            con.execute(
                f"""
                DELETE FROM {self.cache_table}
                WHERE key = ?
                """,
                (key,),
            )

    @auto_hash_key
    def reset_ttl(self, key: Hashable, ttl: float):
        with self.commit_connection() as con:
            con.execute(
                f"""
                UPDATE {self.cache_table}
                SET ttl = ?
                WHERE key = ?
                """,
                (ttl, key),
            )

    def _create_cache_table_and_metadata(self):
        with self.cursor() as cursor:
            # Creates metadata table
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.metadata_table} (
                    setting TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL
                )
                """
            )
            # Creates the main cache table, note that we don't have a expires_at
            # because we can calculate it using the stored_at and ttl
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.cache_table} (
                    key INTEGER PRIMARY KEY,
                    filename TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    stored_at TIMESTAMP NOT NULL,
                    accessed_count INTEGER NOT NULL,
                    ttl REAL
                )
                """
            )

            cursor.execute(
                f"""
                CREATE INDEX IF NOT EXISTS LFUPrimaryCache_key
                ON {self.cache_table} (key)
                """
            )
        self._create_triggers_for_size_tracking()
        self._set_settings()

    @property
    def size_table(self) -> str:
        return "LFUTotalSize"

    @property
    def cache_table(self) -> str:
        return "LFUPrimaryCache"

    @property
    def metadata_table(self) -> str:
        return "LFUMetadata"

    def show_cache(self):
        with self.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT *
                FROM {self.cache_table}
                """
            )
            print(cursor.fetchall())


@define
class HybridCache(BaseCache):
    """A hybrid cache that uses both an LRU and LFU cache.
    Based on: https://ieeexplore.ieee.org/document/10454976
    """

    ttl: float
    threshold: int
    lru_cache: LRUCache
    lfu_cache: LFUCache
    counter: Counter = field(factory=Counter)

    def __attrs_post_init__(self):
        # Initialize a table to keep track of the hit and miss rates.
        with self.commit_connection() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS HybridCacheMetadata (
                    setting TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                INSERT INTO HybridCacheMetadata (setting, value)
                VALUES (?, ?)
                ON CONFLICT(setting) DO NOTHING
                """,
                ("hit_rate", 0),
            )
            con.execute(
                """
                INSERT INTO HybridCacheMetadata (setting, value)
                VALUES (?, ?)
                ON CONFLICT(setting) DO NOTHING
                """,
                ("miss_rate", 0),
            )

    def _update_hit_rate(self, hit: bool):
        """
        Update the hit rate in the metadata table.
        Arguments:
            hit: A boolean indicating whether the cache hit or miss.
        """
        with self.commit_connection() as con:
            if hit:
                con.execute(
                    """
                    UPDATE HybridCacheMetadata
                    SET value = value + 1
                    WHERE setting = 'hit_rate'
                    """
                )
            else:
                con.execute(
                    """
                    UPDATE HybridCacheMetadata
                    SET value = value + 1
                    WHERE setting = 'miss_rate'
                    """
                )

    def get_rates(self) -> Tuple[float, float, int]:
        with self.cursor() as cursor:
            cursor.execute(
                """
                SELECT value
                FROM HybridCacheMetadata
                WHERE setting = 'hit_rate'
                """
            )
            hit_rate: int = int(cursor.fetchone()[0])
            cursor.execute(
                """
                SELECT value
                FROM HybridCacheMetadata
                WHERE setting = 'miss_rate'
                """
            )
            miss_rate: int = int(cursor.fetchone()[0])
            total_requests = hit_rate + miss_rate
            if total_requests == 0:
                return 0.0, 0.0
            return hit_rate / total_requests, miss_rate / total_requests, total_requests

    @auto_hash_key
    def put(self, key: Hashable, value: Any):
        # LRU's cache put handles evictions by least recently used already
        self.lru_cache.put(key, value)
        self.counter[key] = 0

    @auto_hash_key
    def get(self, key: Hashable, default: Any = None) -> Any:
        value_to_return = None
        if self.lru_cache.exists(key):
            self.counter[key] += 1
            value = self.lru_cache.get(key)
            if self.counter[key] >= self.threshold:
                self.move_element_to_lfu(key, value)
            value_to_return = value
        elif self.lfu_cache.exists(key):
            self.lfu_cache.reset_ttl(key, self.ttl)
            value_to_return = self.lfu_cache.get(key)
        # TODO: Different from the paper, we treat gets as plain lookups
        # without any side effects.
        if value_to_return is not None:
            self._update_hit_rate(hit=True)
            return value_to_return
        else:
            # If we are here, it means that the element was not found in either cache.
            # We should update the miss rate.
            self._update_hit_rate(hit=False)
            return default

    @auto_hash_key
    def move_element_to_lfu(self, key: Hashable, value: Any):
        # Remove element from LRU cache
        self.lru_cache.delete(key)
        # Add element to LFU Cache, but evict the ones with the lowest TTL
        # We need to somehow pass this as a param.
        self.lfu_cache.put(key, value, ttl=self.ttl)

    @auto_hash_key
    def delete(self, key: Hashable):
        if self.lru_cache.exists(key):
            self.lru_cache.delete(key)
            self.counter.pop(key, None)
        elif self.lfu_cache.exists(key):
            self.lfu_cache.delete(key)
            self.counter.pop(key, None)

    @auto_hash_key
    def exists(self, key: Hashable) -> bool:
        return self.lru_cache.exists(key) or self.lfu_cache.exists(key)

    @auto_hash_key
    def fits(self, size: int) -> bool:
        return self.lru_cache.fits(size)
