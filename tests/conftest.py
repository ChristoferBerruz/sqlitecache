import os

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "simulation: tests used to bootstrap the simulation"
    )


def pytest_addoption(parser):
    parser.addoption("--n-requests", action="store", type=int, default=100)
    parser.addoption("--simulation", action="store_true", default=False)


@pytest.fixture
def n_requests(request):
    """Fixture to get the number of requests from the command line."""
    yield request.config.getoption("--n-requests")


def pytest_collection_modifyitems(config, items):
    """Modifying collection for better streamlining of simulation tests."""
    simulation_okay = config.getoption("--simulation")
    kept_items = []
    deselected_items = []
    for item in items:
        is_simulation_marked = item.get_closest_marker("simulation")
        # simulation marked tests are only kept if --simulation is given
        # regular tests are kept only when no --simulation is given
        if is_simulation_marked and not simulation_okay:
            deselected_items.append(item)
        elif is_simulation_marked and simulation_okay:
            kept_items.append(item)
        elif not is_simulation_marked and not simulation_okay:
            kept_items.append(item)
        else:
            deselected_items.append(item)

    config.hook.pytest_deselected(items=deselected_items)
    items[:] = kept_items


@pytest.fixture(autouse=True, scope="session")
def create_results_dir():
    """Fixture to create the results directory if it doesn't exist."""
    if not os.path.exists("results"):
        os.makedirs("results")
    yield
