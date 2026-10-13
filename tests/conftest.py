import pytest

from recsys.data import make_synthetic_events
from recsys.pipeline import prepare_data


@pytest.fixture(scope="session")
def synthetic():
    """(events, item_categories) from a small, fast synthetic world."""
    return make_synthetic_events(n_users=1500, n_items=600, n_clusters=8, days=45, seed=7)


@pytest.fixture(scope="session")
def prepared(synthetic):
    events, _ = synthetic
    return prepare_data(events, test_days=7)
