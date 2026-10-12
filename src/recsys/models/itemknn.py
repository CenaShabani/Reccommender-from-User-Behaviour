"""Item-based collaborative filtering ("people who viewed this also viewed...").

Idea: two items are similar if the *same users* interact with both. Represent
each item as a column of the user x item matrix, measure the similarity
between columns, and score a candidate item for a user by summing its
similarity to everything in the user's history:

.. math::

    \\text{score}(u, i) = \\sum_{j \\in \\mathcal{H}(u)} x_{uj} \\cdot s(j, i)

Why item-based rather than user-based? Item neighbourhoods are more stable
than user ones (items do not change taste overnight), there are usually fewer
items than users, and the results explain themselves: "because you viewed X".
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .base import Recommender


def bm25_weight(X: sparse.csr_matrix, k1: float = 1.2, b: float = 0.75) -> sparse.csr_matrix:
    """BM25 re-weighting (borrowed from search). It down-weights very active
    users and saturates repeated interactions, which usually helps item-kNN.
    """
    X = sparse.csr_matrix(X, dtype=np.float32, copy=True)
    n_users = X.shape[0]
    idf = np.log(n_users / (1.0 + np.bincount(X.indices, minlength=X.shape[1]))).astype(np.float32)
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    avg_len = row_sums.mean() if n_users else 1.0
    length_norm = (1.0 - b) + b * row_sums / avg_len
    rows = np.repeat(np.arange(n_users), np.diff(X.indptr))
    X.data = X.data * (k1 + 1.0) / (k1 * length_norm[rows] + X.data) * idf[X.indices]
    return X


class ItemKNNRecommender(Recommender):
    """Item-item cosine similarity with top-``k_neighbors`` pruning.

    Parameters
    ----------
    k_neighbors
        Keep only each item's K most similar items. Pruning removes noise from
        weak co-occurrences and keeps the similarity matrix sparse.
    shrinkage
        ``s(i,j) = dot / (|i||j| + shrinkage)``. Pulls similarities that rest
        on very few co-occurring users towards zero.
    weighting
        ``"none"`` or ``"bm25"``.
    """

    name = "item_knn"

    def __init__(self, k_neighbors: int = 100, shrinkage: float = 10.0, weighting: str = "bm25"):
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.weighting = weighting

    def _fit(self, X, **kwargs):
        W = bm25_weight(X) if self.weighting == "bm25" else sparse.csr_matrix(X, dtype=np.float32)
        Wc = W.T.tocsr()  # items x users
        norms = np.sqrt(np.asarray(Wc.multiply(Wc).sum(axis=1)).ravel()).astype(np.float32)
        n_items = Wc.shape[0]
        k = min(self.k_neighbors, max(n_items - 1, 1))

        # Compute similarities one block of items at a time. A dense
        # n_items x n_items matrix would need 40 GB for 100k items.
        block = max(1, int(2e7 // max(n_items, 1)))
        rows, cols, vals = [], [], []
        WcT = Wc.T.tocsc()
        for start in range(0, n_items, block):
            stop = min(start + block, n_items)
            dots = (Wc[start:stop] @ WcT).toarray()
            sims = dots / (np.outer(norms[start:stop], norms) + self.shrinkage + 1e-12)
            sims[np.arange(stop - start), np.arange(start, stop)] = 0.0  # an item is not its own neighbour
            nbrs = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            nbr_sims = np.take_along_axis(sims, nbrs, axis=1)
            keep = nbr_sims > 0
            rows.append(np.repeat(np.arange(start, stop), k)[keep.ravel()])
            cols.append(nbrs[keep])
            vals.append(nbr_sims[keep])

        # similarity_[j, i] = s(j, i) for the top-K neighbours i of item j.
        self.similarity_ = sparse.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_items, n_items), dtype=np.float32
        )

    def score(self, users):
        return (self.X_[users] @ self.similarity_).toarray()

    def similar_items(self, item: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        row = self.similarity_[item]
        order = np.argsort(-row.data)[:n]
        return row.indices[order], row.data[order]

    def explain(self, user: int, item: int, n: int = 3) -> list[tuple[int, float]]:
        """Which items in the user's history contributed most to ``item``'s score?"""
        history = self.X_[user]
        contrib = history.multiply(self.similarity_[:, item].T).tocsr()
        order = np.argsort(-contrib.data)[:n]
        return [(int(contrib.indices[o]), float(contrib.data[o])) for o in order if contrib.data[o] > 0]
