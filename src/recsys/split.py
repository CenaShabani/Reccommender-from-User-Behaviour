"""Train/test splitting for behavioural data.

The golden rule: **evaluate the way the model will be used.** In production a
recommender is trained on the past and serves the future. A random split
lets the model "see the future" (e.g. it learns an item became popular next
week) and inflates offline metrics. Notebook 02 measures how big that
inflation is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .preprocess import IdMapper


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp


def temporal_split(
    events: pd.DataFrame, test_days: float = config.TEST_DAYS, cutoff: pd.Timestamp | None = None
) -> TemporalSplit:
    """Everything before ``cutoff`` is train and everything at or after it is test.

    By default the cutoff is ``test_days`` before the last event.
    """
    if cutoff is None:
        cutoff = events["timestamp"].max() - pd.Timedelta(days=test_days)
    is_test = events["timestamp"] >= cutoff
    return TemporalSplit(
        train=events[~is_test].reset_index(drop=True),
        test=events[is_test].reset_index(drop=True),
        cutoff=pd.Timestamp(cutoff),
    )


def random_split(events: pd.DataFrame, test_frac: float = 0.2, seed: int = config.SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random row-level split. **Leaks the future.** Here for comparison only."""
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(events)) < test_frac
    return events[~is_test].reset_index(drop=True), events[is_test].reset_index(drop=True)


def leave_last_out(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent event (a common academic protocol).

    It respects per-user order but not global time: one user's test event can
    pre-date another user's training events. Prefer :func:`temporal_split`.
    """
    ordered = events.sort_values("timestamp", kind="stable")
    last = ordered.groupby("visitorid").tail(1).index
    return ordered.drop(index=last).reset_index(drop=True), ordered.loc[last].reset_index(drop=True)


@dataclass
class GroundTruth:
    """Per-user sets of relevant items, in matrix index space."""

    relevant: dict[int, np.ndarray]
    stats: dict = field(default_factory=dict)

    @property
    def users(self) -> np.ndarray:
        return np.fromiter(self.relevant.keys(), dtype=np.int64, count=len(self.relevant))

    def __len__(self) -> int:
        return len(self.relevant)


def build_ground_truth(
    test: pd.DataFrame,
    user_map: IdMapper,
    item_map: IdMapper,
    train_matrix=None,
    target_events: tuple[str, ...] | None = None,
    exclude_seen: bool = True,
) -> GroundTruth:
    """Turn test-period events into ``{user_index: relevant_item_indices}``.

    Parameters
    ----------
    target_events
        Which behaviours count as "relevant". ``None`` means any event. Use
        ``("addtocart", "transaction")`` to evaluate for commercial intent.
    exclude_seen
        Drop test items the user already interacted with in training. Our
        recommenders never re-recommend seen items, so keeping them would
        penalise every model for something none of them attempts. (Re-engaging
        with seen items is a separate product: "recently viewed".)

    Users and items that are not in the training mappers are *cold start*.
    We cannot score them with collaborative filtering, so they are counted
    in ``stats`` and excluded.
    """
    t = test if target_events is None else test[test["event"].astype(str).isin(target_events)]
    u = user_map.to_index(t["visitorid"])
    i = item_map.to_index(t["itemid"])

    stats = {
        "test_users": int(t["visitorid"].nunique()),
        "cold_users": int(t.loc[u < 0, "visitorid"].nunique()),
        "test_events": int(len(t)),
        "events_with_cold_item": int(((u >= 0) & (i < 0)).sum()),
    }

    pairs = pd.DataFrame({"u": u, "i": i})
    pairs = pairs[(pairs["u"] >= 0) & (pairs["i"] >= 0)].drop_duplicates()
    if exclude_seen and train_matrix is not None and len(pairs):
        seen = np.asarray(train_matrix[pairs["u"].to_numpy(), pairs["i"].to_numpy()]).ravel() > 0
        stats["pairs_already_seen"] = int(seen.sum())
        pairs = pairs[~seen]

    relevant = {int(user): grp.to_numpy(dtype=np.int64) for user, grp in pairs.groupby("u")["i"]}
    stats["eval_users"] = len(relevant)
    return GroundTruth(relevant=relevant, stats=stats)
