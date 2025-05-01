import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "simulation: tests used to bootstrap the simulation"
    )


def pytest_addoption(parser):
    parser.addoption("--n-requests", action="store", type=int, default=100)


@pytest.fixture
def n_requests(request):
    """Fixture to get the number of requests from the command line."""
    yield request.config.getoption("--n-requests")
