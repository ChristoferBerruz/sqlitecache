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
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, Hashable, Optional, Tuple

import pendulum
from attrs import define


class Cache(ABC):
    @abstractmethod
    def put(key: Hashable, value: Any):
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


@define(frozen=False)
class DiskStorage:
    storage_dir: str

    def put(self, key: Hashable, value: Any) -> Tuple[str, int]:
        filename = f"{uuid.uuid4().hex}.cache"
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "wb") as f:
            pickle.dump(value, f)
        return filename, os.path.getsize(filepath)

    def get(self, filename: str) -> Any:
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def delete(self, filename: str):
        filepath = os.path.join(self.storage_dir, filename)
        os.remove(filepath)

    def predict_size(self, value: Any) -> int:
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


@define
class BaseCache(Cache):
    db: str
    storage: DiskStorage
    settings: CacheSettings

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


@define
class FunctionalCache(BaseCache):
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
        with self.commit_connection() as con:
            con.execute(
                f"""
                INSERT INTO {self.metadata_table} (setting, value)
                VALUES (?, ?)
                """,
                ("current_size", 0),
            )
            con.execute(
                f"""
                INSERT INTO {self.metadata_table} (setting, value)
                VALUES (?, ?)
                """,
                ("max_size", self.settings.max_size_in_bytes),
            )


HashedKey = int


def auto_hash_key(func):
    def wrapper(self, key: Hashable, *args, **kwargs):
        key = get_hash_for_key(key)
        return func(self, key, *args, **kwargs)

    return wrapper


@define
class LRUCache(FunctionalCache):
    def __attrs_post_init__(self):
        self._create_cache_table_and_metadata()

    @auto_hash_key
    def put(self, key: Hashable, value: Any):
        predicted_size = self.storage.predict_size(value)
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

    @auto_hash_key
    def _put_new(self, key: Hashable, value: Any):
        with self.commit_connection() as con:
            now = pendulum.now().timestamp()
            filename, size = self.storage.put(key, value)
            con.execute(
                f"""
                INSERT INTO {self.cache_table} (key, filename, size, stored_at, accessed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, filename, size, now, now),
            )

    @auto_hash_key
    def _put_update(self, key: Hashable, value: Any):
        prev_filename = self._get_filename_for_key(key)
        with self.storage.defer_delete(
            prev_filename
        ) as _, self.commit_connection() as con:
            now = pendulum.now().timestamp()
            filename, size = self.storage.put(key, value)
            con.execute(
                f"""
                UPDATE {self.cache_table}
                SET filename = ?, size = ?, stored_at = ?, accessed_at = ?
                WHERE key = ?
                """,
                (filename, size, now, now, key),
            )

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

    @auto_hash_key
    def get(self, key: Hashable, default: Any = None) -> Any:
        if not self.exists(key):
            return default
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
            return self.storage.get(filename)

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
            print(current_size, size, max_size)
            return current_size + size <= max_size

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
class LFUCache(FunctionalCache):
    def put(self, key: Hashable, value: Any):
        with self.commit_connection() as con:
            now = pendulum.now().timestamp()
            con.execute(
                """
                INSERT INTO LFUCache (key, value, stored_at, accessed_at, ttl)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, value, now, now, None),
            )

    def get(self, key: Hashable) -> Any:
        with self.cursor() as cursor:
            cursor.execute(
                """
                SELECT value
                FROM LFUCache
                WHERE key = ?
                """,
                (key,),
            )
            return cursor.fetchone()[0]

    def delete(self, key: Hashable):
        with self.commit_connection() as con:
            con.execute(
                """
                DELETE FROM LFUCache
                WHERE key = ?
                """,
                (key,),
            )

    def is_full(self) -> bool:
        with self.cursor() as cursor:
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM LFUCache
                """
            )
            return cursor.fetchone()[0] == self.settings

    def _create_cache_table_and_metadata(self):
        with self.cursor() as cursor:
            # Creates settings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS LFUCacheSettings (
                    setting TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL
                )
                """
            )
            # Creates the main cache table, note that we don't have a expires_at
            # because we can calculate it using the stored_at and ttl
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS LFUPrimaryCache (
                    rowid INTEGER PRIMARY KEY,
                    key BLOB NOT NULL,
                    value BLOB NOT NULL,
                    stored_at TIMESTAMP,
                    accessed_at INTEGER,
                    ttl INTEGER
                )
                """
            )

            # Create the total size table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS LFUTotalSize (
                    size INTEGER
                )
                """
            )

            # Create a simple trigger that updates the access_at
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_access_at_LFU
                AFTER UPDATE ON LFUPrimaryCache
                BEGIN
                    UPDATE LFUPrimaryCache
                    SET accessed_at = CURRENT_TIMESTAMP
                    WHERE rowid = NEW.rowid;
                END
                """
            )


@define
class HybridCache(BaseCache):
    """A hybrid cache that uses both an LRU and LFU cache.
    Based on: https://ieeexplore.ieee.org/document/10454976
    """

    def put(self, key: Hashable, value: Any):
        pass

    def get(self, key: Hashable) -> Any:
        pass

    def delete(self, key: Hashable):
        pass

    def is_full(self) -> bool:
        pass
