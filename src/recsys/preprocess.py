"""Turning a raw event log into a user x item interaction matrix.

The steps, in order, and why each one exists:

1. :func:`clean_events`: drop exact duplicates and rows with missing ids.
2. :func:`add_weights`: map each behaviour to a confidence weight
   (a purchase is stronger evidence of interest than a view).
3. :func:`filter_outlier_users`: remove crawlers / bots whose thousands of
   events would otherwise dominate item similarities and popularity counts.
4. :func:`k_core_filter`: iteratively drop users and items with too little
   data to learn anything from.
5. :class:`IdMapper` + :func:`build_interaction_matrix`: re-index raw ids to
   ``0..n-1`` and aggregate events into a sparse CSR matrix.
"""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd
from scipy import sparse

from . import config


def clean_events(events: pd.DataFrame) -> pd.DataFrame:
    out = events.dropna(subset=["visitorid", "itemid"]).drop_duplicates()
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


def add_weights(events: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    weights = weights or config.EVENT_WEIGHTS
    out = events.copy()
    out["weight"] = out["event"].astype(str).map(weights).astype("float32")
    if out["weight"].isna().any():
        raise ValueError(f"no weight for event types {set(out.loc[out['weight'].isna(), 'event'])}")
    return out


def filter_outlier_users(
    events: pd.DataFrame, max_events: int = config.MAX_USER_EVENTS
) -> tuple[pd.DataFrame, pd.Index]:
    """Drop users with more than ``max_events`` events. Returns (filtered, removed_user_ids)."""
    counts = events["visitorid"].value_counts()
    removed = counts.index[counts > max_events]
    return events[~events["visitorid"].isin(removed)].reset_index(drop=True), removed


def k_core_filter(
    events: pd.DataFrame,
    min_user_items: int = config.MIN_USER_ITEMS,
    min_item_users: int = config.MIN_ITEM_USERS,
    max_iter: int = 50,
) -> pd.DataFrame:
    """Keep the largest sub-log where every user touched >= ``min_user_items``
    distinct items and every item was touched by >= ``min_item_users`` distinct users.

    Removing sparse items can push a user below the threshold (and vice versa),
    so we repeat until nothing changes.
    """
    out = events
    for _ in range(max_iter):
        pairs = out[["visitorid", "itemid"]].drop_duplicates()
        user_n = pairs["visitorid"].value_counts()
        item_n = pairs["itemid"].value_counts()
        keep_users = user_n.index[user_n >= min_user_items]
        keep_items = item_n.index[item_n >= min_item_users]
        mask = out["visitorid"].isin(keep_users) & out["itemid"].isin(keep_items)
        if mask.all():
            break
        out = out[mask]
    return out.reset_index(drop=True)


class IdMapper:
    """Bidirectional mapping between raw ids and contiguous indices ``0..n-1``."""

    def __init__(self, ids: Iterable):
        self.index = pd.Index(pd.unique(np.asarray(ids)))

    def __len__(self) -> int:
        return len(self.index)

    def to_index(self, ids) -> np.ndarray:
        """Raw ids -> indices. Unknown ids map to -1."""
        return self.index.get_indexer(np.asarray(ids))

    def to_raw(self, idx) -> np.ndarray:
        return self.index.to_numpy()[np.asarray(idx)]


Transform = Literal["log1p", "binary", "none"]


def build_interaction_matrix(
    events: pd.DataFrame,
    user_map: IdMapper,
    item_map: IdMapper,
    value_col: str = "weight",
    transform: Transform = "log1p",
) -> sparse.csr_matrix:
    """Aggregate events into a ``(n_users, n_items)`` CSR matrix.

    Repeated events on the same (user, item) pair are summed, then transformed:

    * ``"log1p"`` (default): diminishing returns, so the 50th view of an item
      counts for less than the 2nd. This stops a few obsessive users or items
      dominating the loss.
    * ``"binary"``: 1 if there was any interaction.
    * ``"none"``: the raw summed weight.

    Rows whose user or item is not in the mappers are silently dropped.
    """
    u = user_map.to_index(events["visitorid"])
    i = item_map.to_index(events["itemid"])
    v = events[value_col].to_numpy(dtype=np.float32) if value_col in events else np.ones(len(events), np.float32)
    ok = (u >= 0) & (i >= 0)
    X = sparse.csr_matrix((v[ok], (u[ok], i[ok])), shape=(len(user_map), len(item_map)), dtype=np.float32)
    X.sum_duplicates()
    if transform == "log1p":
        X.data = np.log1p(X.data)
    elif transform == "binary":
        X.data = np.ones_like(X.data)
    elif transform != "none":
        raise ValueError(f"unknown transform {transform!r}")
    X.eliminate_zeros()
    return X
