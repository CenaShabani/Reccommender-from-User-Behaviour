"""Two-stage recommendation: candidate generation followed by learning-to-rank.

This is the shape of most industrial recommenders (YouTube, Pinterest,
Amazon...):

1. **Candidate generation**: several cheap, high-recall models each propose
   their top-N items. Each one sees the user differently: item-kNN catches
   "similar to what you looked at", ALS catches latent taste, and trending
   catches what is hot right now. Their union is a few hundred items rather
   than the whole catalogue.
2. **Ranking**: a supervised model scores each (user, candidate) pair using
   rich features: every generator's score and rank, item popularity and
   freshness, user activity and recency, and user-category affinity. Then it
   re-orders the list.

The subtle part is **where the ranker's labels come from**. We cannot use the
test period (that is cheating), and we cannot label the training period with
itself (every candidate excludes seen items, so there would be no positives).
So we repeat the train/test protocol *inside* the training data:

    |---------------- train period ----------------|---- test ----|
    |------ inner train ------|--- validation ---|
      fit generators here       labels come from here

The ranker learns "which candidates turn out to be relevant next week". For
serving, the generators are re-fitted on the full training period and the
ranker is applied unchanged.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier

from .. import config
from ..preprocess import IdMapper, build_interaction_matrix
from ..split import build_ground_truth, temporal_split
from .als import ALSRecommender
from .base import Recommender, top_k
from .itemknn import ItemKNNRecommender
from .popularity import PopularityRecommender, TrendingRecommender

GeneratorFactory = Callable[[], Recommender]

DEFAULT_GENERATORS: dict[str, GeneratorFactory] = {
    "item_knn": ItemKNNRecommender,
    "als": ALSRecommender,
    "popularity": PopularityRecommender,
    "trending": TrendingRecommender,
}


# --------------------------------------------------------------------------- #
# Context: user- and item-level facts at a point in time
# --------------------------------------------------------------------------- #
class FeatureContext:
    """Precomputed user and item features for one period of events."""

    def __init__(
        self,
        X: sparse.csr_matrix,
        events: pd.DataFrame,
        user_map: IdMapper,
        item_map: IdMapper,
        item_categories: pd.DataFrame | None = None,
        half_life_days: float = 7.0,
    ):
        n_users, n_items = X.shape
        end = events["timestamp"].max()
        self.X_binary = (X > 0).astype(np.float32).tocsr()

        self.item_pop = np.log1p(np.asarray(self.X_binary.sum(axis=0)).ravel())
        trend = TrendingRecommender(half_life_days).fit(X, train_events=events, item_map=item_map)
        self.item_trend = np.log1p(trend.item_scores_)

        i_idx = item_map.to_index(events["itemid"])
        first_seen = events["timestamp"].groupby(i_idx).min()
        first_seen = first_seen[first_seen.index >= 0]
        self.item_age_days = np.full(n_items, 999.0, dtype=np.float32)
        self.item_age_days[first_seen.index.to_numpy()] = (end - first_seen).dt.total_seconds().to_numpy() / 86_400

        self.user_activity = np.log1p(np.diff(X.indptr)).astype(np.float32)
        u_idx = user_map.to_index(events["visitorid"])
        last_seen = events["timestamp"].groupby(u_idx).max()
        last_seen = last_seen[last_seen.index >= 0]
        self.user_recency_days = np.full(n_users, 999.0, dtype=np.float32)
        self.user_recency_days[last_seen.index.to_numpy()] = (end - last_seen).dt.total_seconds().to_numpy() / 86_400

        # Category one-hot (with an "unknown" bucket) for user-category affinity.
        self.item_cat = None
        if item_categories is not None:
            cat_idx = np.full(n_items, -1, dtype=np.int64)
            idx = item_map.to_index(item_categories["itemid"])
            codes, _ = pd.factorize(item_categories["categoryid"])
            ok = idx >= 0
            cat_idx[idx[ok]] = codes[ok]
            n_cats = codes.max() + 2 if len(codes) else 1
            cat_idx[cat_idx < 0] = n_cats - 1
            self.item_cat = cat_idx
            self.cat_onehot = sparse.csr_matrix(
                (np.ones(n_items, np.float32), (np.arange(n_items), cat_idx)), shape=(n_items, n_cats)
            )


# --------------------------------------------------------------------------- #
# Candidate generation + features
# --------------------------------------------------------------------------- #
def generate_candidates(
    users: np.ndarray,
    X: sparse.csr_matrix,
    generators: dict[str, Recommender],
    n_candidates: int = 50,
    context: FeatureContext | None = None,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Union of each generator's top-N unseen items, one row per (user, item).

    With ``context`` it also adds ranking features. The ``<gen>_rank`` column
    is 0-based within that generator's top-N, or N if it did not propose the item.
    """
    n_items = X.shape[1]
    frames = []
    for start in range(0, len(users), batch_size):
        batch = np.asarray(users[start : start + batch_size], dtype=np.int64)
        seen = X[batch]
        seen_rc = seen.nonzero()
        scores, tops = {}, {}
        for name, gen in generators.items():
            S = np.array(gen.score(batch), dtype=np.float32, copy=True)
            S[seen_rc] = -np.inf
            scores[name], tops[name] = S, top_k(S, n_candidates)

        all_tops = np.concatenate(list(tops.values()), axis=1)
        local_rows = np.repeat(np.arange(len(batch)), all_tops.shape[1])
        keys = np.unique(local_rows * n_items + all_tops.ravel())
        rows, items = keys // n_items, keys % n_items
        not_seen = np.asarray(seen[rows, items]).ravel() == 0
        keys, rows, items = keys[not_seen], rows[not_seen], items[not_seen]

        frame = {"user": batch[rows], "item": items}
        if context is not None:
            for name in generators:
                frame[f"{name}_score"] = scores[name][rows, items]
                rank = np.full(len(keys), n_candidates, dtype=np.int32)
                top = tops[name]
                top_keys = np.repeat(np.arange(len(batch)), top.shape[1]) * n_items + top.ravel()
                pos = np.searchsorted(keys, top_keys)
                found = (pos < len(keys)) & (keys[np.minimum(pos, len(keys) - 1)] == top_keys)
                rank[pos[found]] = np.tile(np.arange(top.shape[1]), len(batch))[found]
                frame[f"{name}_rank"] = rank
            frame["item_popularity"] = context.item_pop[items]
            frame["item_trend"] = context.item_trend[items]
            frame["item_age_days"] = context.item_age_days[items]
            frame["user_activity"] = context.user_activity[batch[rows]]
            frame["user_recency_days"] = context.user_recency_days[batch[rows]]
            if context.item_cat is not None:
                hist = context.X_binary[batch] @ context.cat_onehot
                hist = np.asarray(hist.todense())
                hist /= np.maximum(hist.sum(axis=1, keepdims=True), 1.0)
                frame["user_category_affinity"] = hist[rows, context.item_cat[items]]
        frames.append(pd.DataFrame(frame))
    return pd.concat(frames, ignore_index=True)


def label_candidates(candidates: pd.DataFrame, relevant: dict[int, np.ndarray]) -> np.ndarray:
    truth = pd.DataFrame(
        [(u, i) for u, items in relevant.items() for i in items], columns=["user", "item"]
    ).assign(label=1)
    merged = candidates[["user", "item"]].merge(truth, on=["user", "item"], how="left")
    return merged["label"].fillna(0).to_numpy(dtype=np.int8)


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #
class TwoStageRecommender(Recommender):
    """Parameters
    ----------
    generators
        ``{name: factory}``. A factory returns an *unfitted* recommender. We need
        factories rather than instances because each generator is fitted twice
        (inner train, then full train).
    n_candidates
        How many items each generator proposes per user.
    val_days
        Length of the inner validation window that produces ranker labels.
    max_train_users
        Cap on validation users used to train the ranker (keeps memory bounded).
    """

    name = "two_stage"

    def __init__(
        self,
        generators: dict[str, GeneratorFactory] | None = None,
        n_candidates: int = 50,
        val_days: float = config.TEST_DAYS,
        max_train_users: int = 20_000,
        transform: str = "log1p",
        ranker=None,
        seed: int = config.SEED,
        verbose: bool = False,
    ):
        self.generators = generators or DEFAULT_GENERATORS
        self.n_candidates = n_candidates
        self.val_days = val_days
        self.max_train_users = max_train_users
        self.transform = transform
        self.ranker = ranker
        self.seed = seed
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _fit_generators(self, X, events, item_map) -> dict[str, Recommender]:
        fitted = {}
        for name, factory in self.generators.items():
            self._log(f"  fitting generator: {name}")
            fitted[name] = factory().fit(X, train_events=events, item_map=item_map)
        return fitted

    def _fit(
        self,
        X,
        train_events: pd.DataFrame | None = None,
        user_map: IdMapper | None = None,
        item_map: IdMapper | None = None,
        item_categories: pd.DataFrame | None = None,
        **kwargs,
    ):
        if train_events is None or user_map is None or item_map is None:
            raise ValueError("TwoStageRecommender.fit needs train_events=, user_map= and item_map=")

        # 1) Inner split: generators see "inner train"; labels come from "validation".
        inner = temporal_split(train_events, test_days=self.val_days)
        X_inner = build_interaction_matrix(inner.train, user_map, item_map, transform=self.transform)
        gt_val = build_ground_truth(inner.test, user_map, item_map, train_matrix=X_inner)
        users = gt_val.users
        users = users[np.diff(X_inner.indptr)[users] > 0]  # need some history to personalise
        rng = np.random.default_rng(self.seed)
        if len(users) > self.max_train_users:
            users = rng.choice(users, self.max_train_users, replace=False)
        if len(users) == 0:
            raise ValueError("no validation users with history: increase val_days or the data size")

        self._log(f"Stage 1: generators on inner train ({inner.train['timestamp'].max():%Y-%m-%d} cutoff)")
        gens_inner = self._fit_generators(X_inner, inner.train, item_map)
        ctx_inner = FeatureContext(X_inner, inner.train, user_map, item_map, item_categories)
        cands = generate_candidates(users, X_inner, gens_inner, self.n_candidates, ctx_inner)
        y = label_candidates(cands, gt_val.relevant)

        # Users with no relevant candidate teach the ranker nothing about ordering.
        has_pos = pd.Series(y).groupby(cands["user"].to_numpy()).transform("max").to_numpy() > 0
        cands, y = cands[has_pos], y[has_pos]
        self.feature_names_ = [c for c in cands.columns if c not in ("user", "item")]

        n_relevant = sum(len(gt_val.relevant[int(u)]) for u in users)
        self.train_report_ = {
            "val_users": len(users),
            "candidate_recall": float(y.sum() / max(n_relevant, 1)),
            "ranker_rows": len(cands),
            "ranker_users": int(cands["user"].nunique()),
            "positive_rate": float(y.mean()) if len(y) else 0.0,
        }
        self._log(f"Stage 2: ranker on {self.train_report_}")

        self.ranker_ = self.ranker or HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=self.seed,
        )
        self.ranker_.fit(cands[self.feature_names_], y)
        self.ranker_train_frame_ = cands.assign(label=y)

        # 2) Re-fit generators on the full training period for serving.
        self._log("Stage 3: re-fitting generators on the full training period")
        self.generators_ = self._fit_generators(X, train_events, item_map)
        self.context_ = FeatureContext(X, train_events, user_map, item_map, item_categories)

    def candidates(self, users) -> pd.DataFrame:
        """Candidates + features + ranker score for ``users`` (useful for inspection)."""
        users = np.atleast_1d(np.asarray(users, dtype=np.int64))
        cands = generate_candidates(users, self.X_, self.generators_, self.n_candidates, self.context_)
        cands["ranker_score"] = self.ranker_.predict_proba(cands[self.feature_names_])[:, 1]
        return cands

    def score(self, users):
        users = np.atleast_1d(np.asarray(users, dtype=np.int64))
        cands = self.candidates(users)
        out = np.full((len(users), self.n_items_), -np.inf, dtype=np.float32)
        row_of = pd.Series(np.arange(len(users)), index=users)
        out[row_of[cands["user"]].to_numpy(), cands["item"].to_numpy()] = cands["ranker_score"].to_numpy()
        return out
