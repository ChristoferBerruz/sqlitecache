# Introduction

This python package is a persistent key-value cache
using SQlite as its backend. It supports three eviction
policies: Least Recently Used (LRU), Least Frequently Used (LFU)
and a hybrid eviction policy defined in this
paper: [J. Shah and A. A. Siddiqui, “An improved cache eviction
strategy: Combining least recently used and least frequently used
policies,” in 2023 6th International Conference on Advances in
Science and Technology (ICAST), 2023, pp. 1–6.](https://ieeexplore.ieee.org/document/10454976)

# Requirements

## Python version
Minimal required version is Python 3.10

## Poetry
We will be using [Poetry](https://python-poetry.org/docs/#installation) to handle
the virtual environment.

# Running the testcases and simulation

Once you have all the requirements installed, you can install the project
by doing

```bash
poetry install --with=dev
```

This will create a custom virtual environment. Make sure to **enter**
the virtual environment by running `poetry shell`. If any of
the following commands fail because a missing package, **prepend**
each command with `poetry run`. For example, `poetry run pytest -svv`.

To run the testcases,
simply do

```bash
pytest -svv
```

To run the simulation, simply do
```bash
pytest --simulation -vv --n-requests=100 --durations=0 -n 4
```
Note that you can override the number of requests by passing a different value
of `--n-requests`. All other flags are required and it will distribute
the load to 4 CPUs. If you don't know how many CPUs to use, you can pass
`-n auto` instead for automatic detection.

To visualize the results of the simulation, simply do
```bash
python3 ./scripts/analyze.py ./results
```

Under the results/ directory, you should see the results
as `.eps` and `.pdf` files.

# Installing as a python package
Note that our project ships three things: a python package,
a functional test suite, and a simulation. The above instructions
specify how to run the tests and simulation, which under the
hood automatically use the python package.

If you would like to try `sqlitecache` in your own Python project,
you can build a `.whl` executing

```bash
poetry build
```

The `.whl` will be built under the the name `sqlitecache-0.1.0-py3-none-any.whl`.
Then, activate your python environment and you can install our package
by

```bash
pip install <path_to_wheel>/sqlitecache-0.1.0-py3-none-any.whl
```

Once that is done, here's a code snippet how to use our package in your project.

```python3
import os
from sqlitecache.cache import (
    CacheSettings,
    DiskStorage,
    HybridCache,
    LFUCache,
    LRUCache,
    TTLSettings,
)
storage_path = "storage"
os.makedirs(storage_path)
disk_storage = DiskStorage(storage_path)
cache_settings = CacheSettings(100) # 100 bytes of cache, no compression, no encryption
lru_cache = LRUCache(db="cache", storage=disk_storage, settings=cache_settings)
# lru_cache fully initialized and ready to use.
```

# Contributions
