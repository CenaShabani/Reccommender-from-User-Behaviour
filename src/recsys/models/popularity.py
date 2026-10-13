"""Non-personalised baselines: random, popularity and time-decayed "trending".

Never skip these. Popularity is embarrassingly hard to beat on many
e-commerce datasets, and a fancy model that cannot beat it is not ready to ship.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from ..preprocess import IdMapper
from .base import Recommender


class RandomRecommender(Recommender):
    """The floor. Any model that cannot beat this is broken."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def score(self, users):
        rng = np.random.default_rng(self.seed + int(np.asarray(users)[0]))
        return rng.random((len(users), self.n_items_), dtype=np.float32)


class PopularityRecommender(Recommender):
    """Recommend the items the most users interacted with (same list for everyone)."""

    name = "popularity"

    def _fit(self, X, **kwargs):
        # Count distinct users rather than summing weights, so that one user
        # viewing an item 500 times does not make it "popular".
        self.item_scores_ = np.asarray((X > 0).sum(axis=0), dtype=np.float32).ravel()

    def score(self, users):
        return np.broadcast_to(self.item_scores_, (len(users), self.n_items_))


class TrendingRecommender(Recommender):
    """Popularity where recent interactions count more (exponential decay).

    An event that happened ``half_life_days`` before the end of training counts
    half as much as one at the very end. It needs timestamps, which the
    interaction matrix does not have, so it takes the training events too.
    """

    name = "trending"

    def __init__(self, half_life_days: float = 7.0):
        self.half_life_days = half_life_days

    def _fit(self, X, train_events: pd.DataFrame | None = None, item_map: IdMapper | None = None, **kwargs):
        if train_events is None or item_map is None:
            raise ValueError("TrendingRecommender.fit needs train_events= and item_map=")
        ev = train_events[["timestamp", "visitorid", "itemid"]].drop_duplicates(["visitorid", "itemid"], keep="last")
        idx = item_map.to_index(ev["itemid"])
        age_days = (ev["timestamp"].max() - ev["timestamp"]).dt.total_seconds().to_numpy() / 86_400
        decay = 0.5 ** (age_days / self.half_life_days)
        ok = idx >= 0
        self.item_scores_ = np.bincount(idx[ok], weights=decay[ok], minlength=self.n_items_).astype(np.float32)

    def score(self, users):
        return np.broadcast_to(self.item_scores_, (len(users), self.n_items_))
