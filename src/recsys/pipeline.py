"""One call from raw events to "ready to model".

Notebooks 01 and 02 walk through each step by hand. From notebook 03 on, we
call :func:`prepare_data` so that every model is trained and evaluated on
exactly the same data.

The order of operations matters:

1. clean and weight the events;
2. remove bots, *using the whole log*. This is a data-quality step rather
   than learning, and a bot is a bot in both periods;
3. split by time;
4. k-core filter **only the training period**. Filtering on the test
   period would use future information to decide what the model gets to see;
5. fit id mappers on the training data. Anything new in the test period is
   cold start and is reported, not silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from scipy import sparse

from . import config
from .preprocess import IdMapper, add_weights, build_interaction_matrix, clean_events, filter_outlier_users, k_core_filter
from .split import GroundTruth, build_ground_truth, temporal_split


@dataclass
class PreparedData:
    train_events: pd.DataFrame
    test_events: pd.DataFrame
    user_map: IdMapper
    item_map: IdMapper
    X_train: sparse.csr_matrix
    ground_truth: GroundTruth
    cutoff: pd.Timestamp
    report: dict = field(default_factory=dict)

    def summary(self) -> pd.Series:
        n_users, n_items = self.X_train.shape
        density = self.X_train.nnz / max(n_users * n_items, 1)
        return pd.Series(
            {
                "train users": n_users,
                "train items": n_items,
                "train interactions (user-item pairs)": self.X_train.nnz,
                "matrix density": f"{density:.5%}",
                "cutoff": self.cutoff,
                **{k: v for k, v in self.report.items()},
                **{f"test: {k}": v for k, v in self.ground_truth.stats.items()},
            }
        )


def prepare_data(
    events: pd.DataFrame,
    test_days: float = config.TEST_DAYS,
    min_user_items: int = config.MIN_USER_ITEMS,
    min_item_users: int = config.MIN_ITEM_USERS,
    max_user_events: int = config.MAX_USER_EVENTS,
    weights: dict[str, float] | None = None,
    transform: str = "log1p",
    target_events: tuple[str, ...] | None = None,
) -> PreparedData:
    ev = add_weights(clean_events(events), weights)
    ev, bots = filter_outlier_users(ev, max_user_events)
    split = temporal_split(ev, test_days=test_days)
    train = k_core_filter(split.train, min_user_items, min_item_users)

    user_map = IdMapper(train["visitorid"])
    item_map = IdMapper(train["itemid"])
    X = build_interaction_matrix(train, user_map, item_map, transform=transform)
    gt = build_ground_truth(split.test, user_map, item_map, train_matrix=X, target_events=target_events)

    report = {
        "raw events": len(events),
        "bot users removed": len(bots),
        "train events (after k-core)": len(train),
        "test events": len(split.test),
    }
    return PreparedData(train, split.test, user_map, item_map, X, gt, split.cutoff, report)
