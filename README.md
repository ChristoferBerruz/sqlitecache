# Introduction

This python package is a persistent key-value cache
using SQlite as its backend. It supports three eviction
policies: Least Recently Used (LRU), Least Frequently Used (LFU)
and a hybrid eviction policy defined in this [paper](https://ieeexplore.ieee.org/document/10454976).

# Developer Setup

## Python version
Minimal required version is Python 3.10

## Poetry
We will be using [Poetry](https://python-poetry.org/docs/#installation) to handle
the virtual environment.

Once you have poetry installed, run

```bash
poetry install --with=dev
```

This will create all the python dependencies in your project.

You can enter the virtualenv by running
```bash
poetry shell
```

You can also execute `python <something>` commands without entering
the environment by simply doing
```bash
poetry run python <something>
```

## Pre-commit
To avoid having to manually format files, this project
also uses pre-commit. Once you ran `poetry install`, you can run

```bash
poetry run pre-commit run --all-files
```

This will automatically format your files.

This pre-commit hook will also run on push on the branch, so if
your push gets rejected, just run the above command
locally on your computer.