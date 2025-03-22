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
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, Hashable

import pendulum
from attrs import define


class Cache(ABC):
    @abstractmethod
    def put(key: Hashable, value: Any):
        pass

    @abstractmethod
    def get(key: Hashable, value: Any):
        pass

    @abstractmethod
    def delete(key: Hashable, value: Any):
        pass

    @abstractmethod
    def is_full() -> bool:
        pass


class Storage(ABC):
    pass


class FauxStorage(Storage):
    pass


@define
class CacheSettings:
    capacity_gb: float = 1.0
    eviction_policy: str = "LRU"


@define
class PersistentCache(Cache):
    settings: CacheSettings
    backend: Storage
    db: str = "cache.db"

    def __attrs_post_init__(self):
        self._create_cache_table_and_metadata()

    @classmethod
    def new(cls, db: str, settings: CacheSettings, backend: Storage):
        return cls(db=db, settings=settings, backend=backend)

    def put(self, key: Hashable, value: Any):
        # Assume integers
        now = pendulum.now().timestamp()
        with self.commit_connection() as con:
            con.execute(
                """
                INSERT INTO PrimaryCache (key, value, stored_at, accessed_at, accessed_count, ttl, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (key, value, now, now, 0, None, None),
            )

    def get(self, key: Hashable) -> Any:
        with self.cursor() as cursor:
            cursor.execute(
                """
                SELECT value
                FROM PrimaryCache
                WHERE key = ?
                """,
                (key,),
            )
            return cursor.fetchone()[0]

    def get_all_rows(self):
        with self.cursor() as cursor:
            cursor.execute(
                """
                SELECT *
                FROM PrimaryCache
                """
            )
            return cursor.fetchall()

    def delete(self, key: Hashable):
        with self.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM PrimaryCache
                WHERE key = ?
                """,
                (key,),
            )

    def is_full(self) -> bool:
        return False

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

    def _create_cache_table_and_metadata(self):
        # To support LRU + LFU, we need the number of times a key is accessed
        # and the last time it was accessed. We will also have a TTL (time-to-live)
        # for each key. If the key is not accessed within the TTL, it will be evicted.
        # We also have to keep track of the settings set for the cache, which we keep
        # in a separate table.
        # Also, to speed up "is_full" check, we can keep track of the total size
        # of the cache in a separate table and update it whenever we insert, modify
        # or delete an entry.
        with self.cursor() as cursor:
            # Creates settings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS CacheSettings (
                    setting TEXT NOT NULL UNIQUE,
                    value TEXT NOT NULL
                )
                """
            )
            # Creates the main cache table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS PrimaryCache (
                    rowid INTEGER PRIMARY KEY,
                    key BLOB NOT NULL,
                    value BLOB NOT NULL,
                    stored_at TIMESTAMP,
                    accessed_at TIMESTAMP,
                    accessed_count INTEGER,
                    ttl INTEGER,
                    expires_at TIMESTAMP
                )
                """
            )

            # Create the total size table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS TotalSize (
                    size INTEGER
                )
                """
            )

    def _create_triggers_for_bookeeping(self):
        with self.cursor() as cursor:
            # Trigger to update the total size table
            # Triggers for bookeeping the cache table
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_accessed_at
                AFTER UPDATE ON PrimaryCache
                BEGIN
                    UPDATE PrimaryCache
                    SET accessed_at = CURRENT_TIMESTAMP
                    WHERE rowid = NEW.rowid;
                END
                """
            )

            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_accessed_count
                AFTER UPDATE ON PrimaryCache
                BEGIN
                    UPDATE PrimaryCache
                    SET accessed_count = accessed_count + 1
                    WHERE rowid = NEW.rowid;
                END
                """
            )

            # Add triggers to update the total size. We have to handle
            # 3 scenarios: insert, update, delete
            # Insertion directly updates the total size
            # update: new size - old size
            # delete: old size - new size
            # However, let's define a "size" function that returns the size
            # of a row. This way, we can use it in the triggers.
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_total_size_insert
                AFTER INSERT ON PrimaryCache
                BEGIN
                    UPDATE TotalSize
                    SET size = size + size(NEW);
                END
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_total_size_delete
                AFTER DELETE ON PrimaryCache
                BEGIN
                    UPDATE TotalSize
                    SET size = size - size(OLD);
                END
                """
            )
            cursor.execute(
                """
                CREATE TRIGGER IF NOT EXISTS update_total_size_update
                AFTER UPDATE ON PrimaryCache
                BEGIN
                    UPDATE TotalSize
                    SET size = size + size(NEW) - size(OLD);
                END
                """
            )
