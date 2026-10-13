"""Project-wide configuration.

Everything a learner might want to tweak lives here so that notebooks and
scripts stay in sync. Override the data location with the ``RECSYS_DATA_DIR``
environment variable if you keep the Kaggle files somewhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RECSYS_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

KAGGLE_DATASET = "retailrocket/ecommerce-dataset"

# One seed for every source of randomness (data generation, model init, sampling).
SEED = 42

# How much evidence of interest each behaviour carries. A purchase says far more
# than a view. These numbers are a modelling choice, not ground truth: notebook
# 02 discusses them and the exercises ask you to tune them.
EVENT_WEIGHTS: dict[str, float] = {
    "view": 1.0,
    "addtocart": 4.0,
    "transaction": 8.0,
}

# Preprocessing defaults.
TEST_DAYS = 7  # the last N days of the log are held out as the test period
MIN_USER_ITEMS = 2  # k-core: keep users with at least this many distinct items
MIN_ITEM_USERS = 2  # k-core: keep items with at least this many distinct users
MAX_USER_EVENTS = 1000  # users above this are treated as bots / crawlers

# Evaluation defaults.
K = 10
MAX_EVAL_USERS = 20_000  # sample test users above this to keep evaluation fast
