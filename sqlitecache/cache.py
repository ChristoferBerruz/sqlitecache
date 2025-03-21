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
lives in the code. However, it is a nice mental model to have when reading this model.
"""
from abc import ABC, abstractmethod
from typing import Any, Hashable


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
