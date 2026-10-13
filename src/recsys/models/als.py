"""Implicit-feedback matrix factorisation with Alternating Least Squares.

Reference: Hu, Koren & Volinsky (2008), *Collaborative Filtering for Implicit
Feedback Datasets*.

The problem with implicit data is that we only see positives. A missing entry
might mean "not interested" or "never saw it". HKV split each observation
into two parts:

* a **preference** ``p_ui = 1 if r_ui > 0 else 0``: did the user show interest?
* a **confidence** ``c_ui = 1 + alpha * r_ui``: how sure are we about p_ui?

and minimise, over user factors ``x_u`` and item factors ``y_i``:

.. math::

    \\sum_{u,i} c_{ui} (p_{ui} - x_u^\\top y_i)^2
    + \\lambda \\left( \\sum_u \\|x_u\\|^2 + \\sum_i \\|y_i\\|^2 \\right)

Note the sum runs over *all* user-item pairs, including the missing ones. That
is what makes it work for implicit data. Fixing the item factors turns it into
a ridge regression per user with a closed-form solution:

.. math::

    x_u = (Y^\\top C^u Y + \\lambda I)^{-1} Y^\\top C^u p(u)

The trick that makes this fast: ``Y^T C^u Y = Y^T Y + Y^T (C^u - I) Y``.
``Y^T Y`` is shared by every user and is computed once per sweep. ``C^u - I``
is non-zero only on the items user u interacted with. So each solve costs
O(f^2 * n_u + f^3) rather than O(f^2 * n_items).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .base import Recommender


def _als_half_step(C: sparse.csr_matrix, Y: np.ndarray, reg: float) -> np.ndarray:
    """Solve for every row factor of ``C`` given the column factors ``Y``.

    ``C`` holds ``alpha * r`` (that is, ``c - 1``) on its non-zeros.
    """
    n_rows, f = C.shape[0], Y.shape[1]
    YtY = Y.T @ Y
    base = YtY + reg * np.eye(f, dtype=Y.dtype)
    X = np.zeros((n_rows, f), dtype=Y.dtype)
    indptr, indices, data = C.indptr, C.indices, C.data
    for u in range(n_rows):
        lo, hi = indptr[u], indptr[u + 1]
        if lo == hi:
            continue  # no interactions: leave this factor at zero
        idx, conf_minus_1 = indices[lo:hi], data[lo:hi]
        Yu = Y[idx]
        A = base + (Yu.T * conf_minus_1) @ Yu  # Y^T Y + Y^T (C^u - I) Y + lambda I
        b = Yu.T @ (1.0 + conf_minus_1)  # Y^T C^u p(u), with p(u) = 1 on observed items
        X[u] = np.linalg.solve(A, b)
    return X


class ALSRecommender(Recommender):
    """Parameters
    ----------
    factors
        Dimension f of the latent space.
    regularization
        lambda: larger values shrink factors and fight overfitting.
    alpha
        Confidence scaling. Larger values trust observed interactions more,
        relative to the sea of unobserved zeros.
    iterations
        Number of (users, items) sweeps. ALS usually converges in 10-20.
    """

    name = "als"

    def __init__(
        self,
        factors: int = 32,
        regularization: float = 1.0,
        alpha: float = 10.0,
        iterations: int = 15,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose

    def _fit(self, X, **kwargs):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = X.shape
        Cui = sparse.csr_matrix(X, dtype=np.float64) * self.alpha
        Ciu = Cui.T.tocsr()
        self.user_factors_ = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors_ = rng.normal(0, 0.01, (n_items, self.factors))
        self.loss_history_ = []
        for it in range(self.iterations):
            self.user_factors_ = _als_half_step(Cui, self.item_factors_, self.regularization)
            self.item_factors_ = _als_half_step(Ciu, self.user_factors_, self.regularization)
            if self.verbose:
                loss = self.loss(Cui)
                self.loss_history_.append(loss)
                print(f"  ALS iteration {it + 1:>2}/{self.iterations}: loss = {loss:,.1f}")
        self.user_factors_ = self.user_factors_.astype(np.float32)
        self.item_factors_ = self.item_factors_.astype(np.float32)

    def loss(self, Cui: sparse.csr_matrix) -> float:
        """The full HKV objective, computed without materialising the dense matrix.

        Over all pairs, ``sum c (p - s)^2 = sum s^2 + sum_observed [c (1 - s)^2 - s^2]``
        where ``s = x_u . y_i``, and ``sum s^2 = trace((X^T X)(Y^T Y))``.
        """
        Xf, Yf = self.user_factors_, self.item_factors_
        rows = np.repeat(np.arange(Cui.shape[0]), np.diff(Cui.indptr))
        s = np.einsum("ij,ij->i", Xf[rows], Yf[Cui.indices])
        c = 1.0 + Cui.data
        all_pairs = float(np.sum((Xf.T @ Xf) * (Yf.T @ Yf)))
        observed = float(np.sum(c * (1.0 - s) ** 2 - s**2))
        reg = self.regularization * (float(np.sum(Xf**2)) + float(np.sum(Yf**2)))
        return all_pairs + observed + reg

    def score(self, users):
        return self.user_factors_[users] @ self.item_factors_.T

    def similar_items(self, item: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Nearest items by cosine similarity in the latent space."""
        Y = self.item_factors_
        norms = np.linalg.norm(Y, axis=1) + 1e-9
        sims = (Y @ Y[item]) / (norms * norms[item])
        sims[item] = -np.inf
        top = np.argsort(-sims)[:n]
        return top, sims[top]
