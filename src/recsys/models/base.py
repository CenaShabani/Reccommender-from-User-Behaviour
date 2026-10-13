"""The interface every recommender in this project implements.

A recommender is anything that can turn a user into a score for every item.
Ranking, excluding already-seen items and batching are the same for every
model, so they live here once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse


class Recommender(ABC):
    name: str = "recommender"

    def fit(self, X: sparse.csr_matrix, **kwargs) -> "Recommender":
        """Learn from a ``(n_users, n_items)`` interaction matrix."""
        self.X_ = sparse.csr_matrix(X)
        self.n_users_, self.n_items_ = X.shape
        self._fit(self.X_, **kwargs)
        return self

    def _fit(self, X: sparse.csr_matrix, **kwargs) -> None:  # pragma: no cover - optional hook
        pass

    @abstractmethod
    def score(self, users: np.ndarray) -> np.ndarray:
        """Return a dense ``(len(users), n_items)`` array of scores (higher = better)."""

    def recommend(
        self,
        users,
        k: int = 10,
        exclude_seen: bool = True,
        X: sparse.csr_matrix | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Top-``k`` item indices for each user, best first.

        ``X`` is the matrix whose non-zeros are treated as "already seen". It
        defaults to the matrix the model was fitted on.
        """
        users = np.atleast_1d(np.asarray(users, dtype=np.int64))
        X = self.X_ if X is None else X
        out = np.empty((len(users), k), dtype=np.int64)
        for start in range(0, len(users), batch_size):
            batch = users[start : start + batch_size]
            scores = np.array(self.score(batch), dtype=np.float32, copy=True)
            if exclude_seen:
                seen = X[batch]
                scores[seen.nonzero()] = -np.inf
            out[start : start + len(batch)] = top_k(scores, k)
        return out

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise indices of the ``k`` largest scores, sorted descending.

    ``argpartition`` finds the top k in O(n) and then only those k get sorted.
    That beats sorting every item when the catalogue has 100k+ items.
    """
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)
