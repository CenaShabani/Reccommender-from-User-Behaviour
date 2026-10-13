"""Offline ranking metrics.

Every accuracy metric here starts from the same object: a *hit vector*
for one user, e.g. ``[0, 1, 0, 0, 1]`` means the 2nd and 5th recommended
items were relevant. The metrics differ only in how they reward hits and
where in the list those hits sit.

=================  ============================================================
metric             question it answers
=================  ============================================================
hit_rate@k         Did we get *at least one* thing right? (easy to explain)
precision@k        What fraction of the k slots were useful?
recall@k           What fraction of what the user wanted did we surface?
MAP@k              Precision, averaged at each hit: rewards early hits.
NDCG@k             Discounted gain: a hit at rank 1 is worth more than at rank 10.
MRR                1 / rank of the first hit.
coverage           What share of the catalogue do we ever recommend?
novelty            How un-obvious are the recommendations (mean self-information)?
=================  ============================================================
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from . import config

if TYPE_CHECKING:
    from .models.base import Recommender
    from .split import GroundTruth


# --------------------------------------------------------------------------- #
# Per-user metrics, from a hit vector
# --------------------------------------------------------------------------- #
def hits_vector(recommended: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    return np.isin(recommended, relevant).astype(np.float64)


def precision_at_k(hits: np.ndarray, k: int) -> float:
    return float(hits[:k].sum() / k)


def recall_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    return float(hits[:k].sum() / n_relevant) if n_relevant else 0.0


def hit_rate_at_k(hits: np.ndarray, k: int) -> float:
    return float(hits[:k].any())


def average_precision_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    h = hits[:k]
    if not h.any():
        return 0.0
    precisions = np.cumsum(h) / np.arange(1, len(h) + 1)
    return float((precisions * h).sum() / min(n_relevant, k))


def ndcg_at_k(hits: np.ndarray, n_relevant: int, k: int) -> float:
    h = hits[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(h) + 2))
    dcg = float((h * discounts).sum())
    idcg = float(discounts[: min(n_relevant, k)].sum())
    return dcg / idcg if idcg > 0 else 0.0


def reciprocal_rank(hits: np.ndarray) -> float:
    nz = np.flatnonzero(hits)
    return 1.0 / (nz[0] + 1) if len(nz) else 0.0


# --------------------------------------------------------------------------- #
# Beyond-accuracy metrics, over the whole set of recommendation lists
# --------------------------------------------------------------------------- #
def catalog_coverage(recommendations: np.ndarray, n_items: int) -> float:
    return len(np.unique(recommendations[recommendations >= 0])) / n_items


def novelty(recommendations: np.ndarray, item_popularity: np.ndarray, n_users: int) -> float:
    """Mean self-information ``-log2(p(i))`` where ``p(i)`` is the share of users who touched item i.

    Higher means the model recommends less obvious items. A pure popularity
    model has the lowest novelty possible.
    """
    p = np.clip(item_popularity / max(n_users, 1), 1e-12, 1.0)
    recs = recommendations[recommendations >= 0]
    return float(np.mean(-np.log2(p[recs]))) if len(recs) else 0.0


# --------------------------------------------------------------------------- #
# Evaluating a model
# --------------------------------------------------------------------------- #
def score_lists(recommendations: np.ndarray, users: np.ndarray, ground_truth: "GroundTruth", k: int) -> pd.DataFrame:
    """Per-user metrics for precomputed recommendation lists."""
    rows = []
    for user, recs in zip(users, recommendations):
        relevant = ground_truth.relevant[int(user)]
        h = hits_vector(recs[:k], relevant)
        n_rel = len(relevant)
        rows.append(
            (
                int(user),
                n_rel,
                precision_at_k(h, k),
                recall_at_k(h, n_rel, k),
                hit_rate_at_k(h, k),
                average_precision_at_k(h, n_rel, k),
                ndcg_at_k(h, n_rel, k),
                reciprocal_rank(h),
            )
        )
    cols = ["user", "n_relevant", f"precision@{k}", f"recall@{k}", f"hit_rate@{k}", f"map@{k}", f"ndcg@{k}", "mrr"]
    return pd.DataFrame(rows, columns=cols)


def sample_eval_users(ground_truth: "GroundTruth", max_users: int | None, seed: int = config.SEED) -> np.ndarray:
    users = np.sort(ground_truth.users)
    if max_users is not None and len(users) > max_users:
        users = np.sort(np.random.default_rng(seed).choice(users, size=max_users, replace=False))
    return users


def evaluate(
    model: "Recommender",
    X_train,
    ground_truth: "GroundTruth",
    k: int = config.K,
    max_users: int | None = config.MAX_EVAL_USERS,
    seed: int = config.SEED,
    batch_size: int = 256,
    return_per_user: bool = False,
):
    """Recommend for every evaluation user and average the per-user metrics.

    Returns a dict of metrics, or ``(dict, per_user_frame)`` if
    ``return_per_user=True``. The per-user frame is what you need for
    segment analysis and significance tests (notebook 07).
    """
    users = sample_eval_users(ground_truth, max_users, seed)
    recs = model.recommend(users, k=k, X=X_train, batch_size=batch_size)
    per_user = score_lists(recs, users, ground_truth, k)

    summary = per_user.drop(columns=["user", "n_relevant"]).mean().to_dict()
    item_pop = np.asarray((X_train > 0).sum(axis=0)).ravel()
    summary["coverage"] = catalog_coverage(recs, X_train.shape[1])
    summary["novelty"] = novelty(recs, item_pop, X_train.shape[0])
    summary["n_users"] = len(users)
    return (summary, per_user) if return_per_user else summary


def evaluate_many(models: dict[str, "Recommender"], X_train, ground_truth: "GroundTruth", **kwargs) -> pd.DataFrame:
    """Evaluate several fitted models into a leaderboard sorted by NDCG."""
    k = kwargs.get("k", config.K)
    rows = {name: evaluate(m, X_train, ground_truth, **kwargs) for name, m in models.items()}
    board = pd.DataFrame(rows).T
    board["n_users"] = board["n_users"].astype(int)
    return board.sort_values(f"ndcg@{k}", ascending=False)
