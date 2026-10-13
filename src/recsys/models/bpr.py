"""Bayesian Personalised Ranking (Rendle et al., 2009).

ALS tries to predict a *value* for every cell. BPR optimises what we actually
care about: the *order*. For a user u, an item i they interacted with, and an
item j they did not, it maximises the probability that u prefers i over j:

.. math::

    \\max \\sum_{(u,i,j)} \\ln \\sigma(\\hat{x}_{ui} - \\hat{x}_{uj}) - \\lambda \\|\\Theta\\|^2,
    \\qquad \\hat{x}_{ui} = w_u^\\top h_i + b_i

There are far too many (u, i, j) triples to enumerate, so we use stochastic
gradient ascent. Each step samples a positive pair (u, i) from the data and a
negative j uniformly from the catalogue. The gradient of ``ln σ(d)`` is
``σ(-d)``: triples the model already orders correctly (large d) barely move
the parameters, and badly ordered ones move them a lot.
"""

from __future__ import annotations

import numpy as np

from .base import Recommender


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class BPRRecommender(Recommender):
    name = "bpr"

    def __init__(
        self,
        factors: int = 32,
        learning_rate: float = 0.1,
        regularization: float = 0.01,
        epochs: int = 30,
        batch_size: int = 512,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.factors = factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose

    def _fit(self, X, **kwargs):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = X.shape
        pos_users = np.repeat(np.arange(n_users), np.diff(X.indptr))
        pos_items = X.indices.copy()
        n_pos = len(pos_items)

        W = rng.normal(0, 0.1, (n_users, self.factors)).astype(np.float32)
        H = rng.normal(0, 0.1, (n_items, self.factors)).astype(np.float32)
        b = np.zeros(n_items, dtype=np.float32)
        lr, reg = self.learning_rate, self.regularization
        self.auc_history_ = []

        for epoch in range(self.epochs):
            order = rng.permutation(n_pos)
            correct = 0
            for start in range(0, n_pos, self.batch_size):
                sel = order[start : start + self.batch_size]
                u, i = pos_users[sel], pos_items[sel]
                j = rng.integers(0, n_items, len(sel))
                # Occasionally j is actually a positive for u. With sparse data
                # that is rare, and the noise is cheaper than checking.

                wu, hi, hj = W[u], H[i], H[j]
                d = np.einsum("ij,ij->i", wu, hi - hj) + b[i] - b[j]
                g = _sigmoid(-d)[:, None]  # d/dd ln sigma(d) = sigma(-d)
                correct += int((d > 0).sum())

                # Gradient *ascent*. np.add.at accumulates correctly when the
                # same user/item appears twice in a batch.
                np.add.at(W, u, lr * (g * (hi - hj) - reg * wu))
                np.add.at(H, i, lr * (g * wu - reg * hi))
                np.add.at(H, j, lr * (-g * wu - reg * hj))
                np.add.at(b, i, lr * (g[:, 0] - reg * b[i]))
                np.add.at(b, j, lr * (-g[:, 0] - reg * b[j]))

            # Share of sampled triples ordered correctly: a cheap training AUC.
            self.auc_history_.append(correct / n_pos)
            if self.verbose:
                print(f"  BPR epoch {epoch + 1:>2}/{self.epochs}: train AUC ~ {self.auc_history_[-1]:.4f}")

        self.user_factors_, self.item_factors_, self.item_bias_ = W, H, b

    def score(self, users):
        return self.user_factors_[users] @ self.item_factors_.T + self.item_bias_
